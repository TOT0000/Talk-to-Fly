import json
import sys
from types import SimpleNamespace

import pytest

from proposer.consistency import CandidateConsistencyError, validate_candidate_contract_alignment
from proposer.propose_candidate import propose_next_candidate


PARENT_TRIGGER = '''from __future__ import annotations


def should_trigger_replan(state: dict, memory: dict, spec: dict) -> tuple[bool, str]:
    cfg = dict((spec or {}).get("trigger_policy") or {})
    trigger_type = str(cfg.get("type") or "periodic")
    risk = float(state.get("predicted_collision_probability") or 0.0)

    if trigger_type == "event_predicted_collision_probability":
        threshold = float(cfg.get("threshold", 0.5))
        strictly_greater = bool(cfg.get("strictly_greater", True))
        hit = (risk > threshold) if strictly_greater else (risk >= threshold)
        return (hit, f"risk_{risk:.3f}_threshold_{threshold:.3f}")

    return (False, "periodic_controller_driven")
'''

PARENT_STATE = '''from __future__ import annotations


def encode_state(snapshot: dict, spec: dict) -> dict:
    cfg = dict((spec or {}).get("state_encoder") or {})
    include = list(cfg.get("include_fields") or [])
    out = {k: snapshot.get(k) for k in include}
    if cfg.get("include_risk_related"):
        out["predicted_collision_probability"] = snapshot.get("predicted_collision_probability")
    out["summary_style"] = str(cfg.get("summary_style") or "structured")
    return out
'''

PARENT_PROMPT = '''from __future__ import annotations


def build_prompt(stage: str, task_description: str, encoded_state: dict, spec: dict) -> dict:
    cfg = dict((spec or {}).get("prompt_builder") or {})
    return {
        "template_family": cfg.get("template_family"),
        "include_example": bool(cfg.get("include_example", True)),
        "encoded_state": dict(encoded_state or {}),
    }
'''


def _write_parent(parent_dir):
    parent_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": "baseline3",
        "kind": "baseline",
        "trigger_policy": {
            "module": "trigger_policy.py",
            "type": "event_predicted_collision_probability",
            "heartbeat_seconds": None,
            "threshold": 0.5,
            "strictly_greater": True,
            "consecutive_high_risk": 1,
            "hysteresis": 0.05,
        },
        "state_encoder": {
            "module": "state_encoder.py",
            "include_fields": ["uav_pose_heading", "predicted_collision_probability"],
            "summary_style": "risk_aware",
            "include_risk_related": True,
            "include_targets": True,
            "include_geometry_flags": False,
        },
        "prompt_builder": {
            "module": "prompt_builder.py",
            "template_family": "baseline3_prompt",
            "include_example": True,
            "example_family": "baseline3_example",
        },
    }
    (parent_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (parent_dir / "trigger_policy.py").write_text(PARENT_TRIGGER, encoding="utf-8")
    (parent_dir / "state_encoder.py").write_text(PARENT_STATE, encoding="utf-8")
    (parent_dir / "prompt_builder.py").write_text(PARENT_PROMPT, encoding="utf-8")
    (parent_dir / "proposer_note.txt").write_text("parent note\n", encoding="utf-8")


def _write_candidate(candidate_dir, *, files_to_modify, trigger_type="event_predicted_collision_probability", threshold=0.45, note_extra=""):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": "candidate_9999",
        "kind": "candidate",
        "parent": "baseline3",
        "trigger_policy": {
            "module": "trigger_policy.py",
            "type": trigger_type,
            "heartbeat_seconds": None,
            "threshold": threshold,
            "strictly_greater": True,
            "consecutive_high_risk": 2,
            "hysteresis": 0.05,
        },
        "state_encoder": {
            "module": "state_encoder.py",
            "include_fields": ["uav_pose_heading", "predicted_collision_probability"],
            "summary_style": "risk_aware",
            "include_risk_related": True,
            "include_targets": True,
            "include_geometry_flags": False,
        },
        "prompt_builder": {
            "module": "prompt_builder.py",
            "template_family": "baseline3_prompt",
            "include_example": True,
            "example_family": "baseline3_example",
        },
        "proposal_contract": {
            "parent_harness": "baseline3",
            "one_sentence_hypothesis": "introduce tighter event threshold",
            "weakness_being_addressed": "late reaction",
            "expected_tradeoff": "slightly more replans",
            "files_to_create_or_modify": files_to_modify,
            "implementation_contract": {
                "trigger_policy": {"type": trigger_type, "threshold": threshold},
                "state_encoder": {"summary_style": "risk_aware", "include_risk_related": True, "include_fields_contains": ["predicted_collision_probability"]},
                "prompt_builder": {"template_family": "baseline3_prompt", "include_example": True, "example_family": "baseline3_example"},
            },
            "invariants": ["contract claims must match spec and code"],
        },
    }
    (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "trigger_policy.py").write_text(PARENT_TRIGGER.replace("0.5", "0.45"), encoding="utf-8")
    (candidate_dir / "state_encoder.py").write_text(PARENT_STATE, encoding="utf-8")
    (candidate_dir / "prompt_builder.py").write_text(PARENT_PROMPT, encoding="utf-8")
    (candidate_dir / "proposer_note.txt").write_text(
        "Hypothesis: introduce tighter event threshold\n"
        f"Implemented trigger type: {trigger_type}\n"
        f"{note_extra}\n",
        encoding="utf-8",
    )


def test_validator_passes_for_aligned_contract_spec_and_modules(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(
        cand,
        files_to_modify=["spec.json", "trigger_policy.py", "proposer_note.txt"],
    )

    result = validate_candidate_contract_alignment(cand, parent_dir=parent)
    assert result["trigger_type"] == "event_predicted_collision_probability"


def test_validator_fails_when_contract_trigger_claim_not_in_spec(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(
        cand,
        files_to_modify=["spec.json", "trigger_policy.py", "proposer_note.txt"],
        threshold=0.45,
    )

    spec = json.loads((cand / "spec.json").read_text(encoding="utf-8"))
    spec["proposal_contract"]["implementation_contract"]["trigger_policy"]["threshold"] = 0.40
    (cand / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(CandidateConsistencyError, match="trigger_policy mismatch"):
        validate_candidate_contract_alignment(cand, parent_dir=parent)


def test_validator_fails_when_spec_trigger_not_reflected_in_trigger_policy_module(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(
        cand,
        files_to_modify=["spec.json", "trigger_policy.py", "proposer_note.txt"],
    )
    (cand / "trigger_policy.py").write_text("def should_trigger_replan(state, memory, spec):\n    return (False, 'noop')\n", encoding="utf-8")

    with pytest.raises(CandidateConsistencyError, match="missing event trigger branch"):
        validate_candidate_contract_alignment(cand, parent_dir=parent)


def test_validator_fails_on_empty_files_to_modify(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(cand, files_to_modify=[])

    with pytest.raises(CandidateConsistencyError, match="must be non-empty"):
        validate_candidate_contract_alignment(cand, parent_dir=parent)


def test_validator_fails_when_declared_files_do_not_match_actual_changes(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(cand, files_to_modify=["spec.json", "proposer_note.txt"])

    with pytest.raises(CandidateConsistencyError, match="files_to_create_or_modify mismatch"):
        validate_candidate_contract_alignment(cand, parent_dir=parent)


def test_validator_accepts_natural_language_include_field_alias(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(
        cand,
        files_to_modify=["spec.json", "trigger_policy.py", "proposer_note.txt"],
    )

    spec = json.loads((cand / "spec.json").read_text(encoding="utf-8"))
    spec["proposal_contract"]["implementation_contract"]["state_encoder"]["include_fields_contains"] = [
        "predicted collision probability"
    ]
    (cand / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    # should pass due to canonicalized/tokenized include-field matching
    validate_candidate_contract_alignment(cand, parent_dir=parent)


def test_propose_flow_backfills_proposer_note_in_changed_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    baseline = repo / "harnesses" / "baseline3"
    (repo / "harnesses" / "candidates").mkdir(parents=True, exist_ok=True)
    _write_parent(baseline)

    class _FakeLLM:
        def request(self, prompt: str, model_name: str, stream: bool = False):
            if "Requested file: spec.json" in prompt:
                return json.dumps(
                    {
                        "id": "candidate_0001",
                        "kind": "candidate",
                        "name": "candidate",
                        "trigger_policy": {
                            "module": "trigger_policy.py",
                            "type": "event_predicted_collision_probability",
                            "heartbeat_seconds": None,
                            "threshold": 0.45,
                            "strictly_greater": True,
                            "consecutive_high_risk": 1,
                            "hysteresis": 0.05,
                        },
                        "state_encoder": {
                            "module": "state_encoder.py",
                            "include_fields": ["uav_pose_heading", "predicted_collision_probability"],
                            "summary_style": "risk_aware",
                            "include_risk_related": True,
                            "include_targets": True,
                            "include_geometry_flags": False,
                        },
                        "prompt_builder": {
                            "module": "prompt_builder.py",
                            "template_family": "baseline3_prompt",
                            "include_example": True,
                            "example_family": "baseline3_example",
                        },
                    }
                )
            return json.dumps(
                {
                    "parent_harness": "baseline3",
                    "candidate_id": "candidate_0001",
                    "one_sentence_hypothesis": "introduce tighter event threshold",
                    "weakness_being_addressed": "late event trigger",
                    "expected_tradeoff": "slightly more replans",
                    "files_to_create_or_modify": ["spec.json", "proposer_note.txt"],
                    "proposer_note_text": "placeholder",
                    "implementation_contract": {
                        "trigger_policy": {"type": "event_predicted_collision_probability", "threshold": 0.45},
                        "state_encoder": {"summary_style": "risk_aware", "include_risk_related": True, "include_fields_contains": ["predicted_collision_probability"]},
                        "prompt_builder": {"template_family": "baseline3_prompt", "include_example": True, "example_family": "baseline3_example"},
                    },
                    "invariants": ["contract claims must match spec and code"],
                }
            )

    monkeypatch.setitem(
        sys.modules,
        "controller.llm_wrapper",
        SimpleNamespace(LLMWrapper=lambda temperature=0.1: _FakeLLM(), MODEL_NAME="fake-model"),
    )

    created = propose_next_candidate(repo_root=repo, focus_text="test consistency")
    spec = json.loads((created / "spec.json").read_text(encoding="utf-8"))
    assert "proposer_note.txt" in spec["proposal_contract"]["files_to_create_or_modify"]
