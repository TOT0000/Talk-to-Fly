import json
import sys
from types import SimpleNamespace

import pytest

from proposer.consistency import CandidateConsistencyError, validate_candidate_contract_alignment
import proposer.propose_candidate as propose_candidate_module
from proposer.propose_candidate import propose_next_candidate


PARENT_TRIGGER = '''from __future__ import annotations

def should_trigger_replan(state: dict, memory: dict, spec: dict) -> tuple[bool, str]:
    risk = float(state.get("predicted_collision_probability") or 0.0)
    return (risk >= 0.5, "risk")
'''

PARENT_STATE = '''from __future__ import annotations

def encode_state(snapshot: dict, spec: dict) -> dict:
    return {"predicted_collision_probability": snapshot.get("predicted_collision_probability")}
'''

PARENT_PROMPT = '''from __future__ import annotations

def build_prompt(stage: str, task_description: str, encoded_state: dict, spec: dict) -> dict:
    return {"encoded_state": dict(encoded_state or {})}
'''


def _write_parent(parent_dir):
    parent_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": "baseline3",
        "kind": "baseline",
        "trigger_policy": {"module": "trigger_policy.py", "type": "event_predicted_collision_probability", "threshold": 0.5},
        "state_encoder": {"module": "state_encoder.py", "include_fields": ["predicted_collision_probability"]},
        "prompt_builder": {"module": "prompt_builder.py", "template_family": "baseline3_prompt", "include_example": True},
        "sandbox": {
            "state_features": {"module": "state_features.py", "enabled": True},
            "trigger_logic": {"module": "trigger_logic.py", "enabled": True},
            "prompt_composer": {"module": "prompt_composer.py", "enabled": True},
        },
    }
    (parent_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (parent_dir / "trigger_policy.py").write_text(PARENT_TRIGGER, encoding="utf-8")
    (parent_dir / "state_encoder.py").write_text(PARENT_STATE, encoding="utf-8")
    (parent_dir / "prompt_builder.py").write_text(PARENT_PROMPT, encoding="utf-8")
    (parent_dir / "trigger_logic.py").write_text(PARENT_TRIGGER, encoding="utf-8")
    (parent_dir / "state_features.py").write_text(PARENT_STATE, encoding="utf-8")
    (parent_dir / "prompt_composer.py").write_text("def compose_prompt_context(stage, task_description, encoded_state, snapshot, spec):\n    return 'ctx'\n", encoding="utf-8")
    (parent_dir / "validator_rules.py").write_text("def runtime_effect_modules():\n    return ['trigger_logic.py', 'state_features.py', 'prompt_composer.py']\n", encoding="utf-8")
    (parent_dir / "archive_selector.py").write_text("def select_entries(entries, max_entries):\n    return entries\n", encoding="utf-8")
    (parent_dir / "proposer_note.txt").write_text("parent note\n", encoding="utf-8")


def _write_parent_legacy(parent_dir):
    parent_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": "baseline3",
        "kind": "baseline",
        "trigger_policy": {"module": "trigger_policy.py", "type": "event_predicted_collision_probability", "threshold": 0.5},
        "state_encoder": {"module": "state_encoder.py", "include_fields": ["predicted_collision_probability"]},
        "prompt_builder": {"module": "prompt_builder.py", "template_family": "baseline3_prompt", "include_example": True},
    }
    (parent_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (parent_dir / "trigger_policy.py").write_text(PARENT_TRIGGER, encoding="utf-8")
    (parent_dir / "state_encoder.py").write_text(PARENT_STATE, encoding="utf-8")
    (parent_dir / "prompt_builder.py").write_text(PARENT_PROMPT, encoding="utf-8")
    (parent_dir / "proposer_note.txt").write_text("parent note\n", encoding="utf-8")


def _write_candidate(candidate_dir, *, files_to_modify, include_runtime_change=True):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": "candidate_9999",
        "kind": "candidate",
        "parent": "baseline3",
        "sandbox": {
            "state_features": {"module": "state_features.py", "enabled": True},
            "trigger_logic": {"module": "trigger_logic.py", "enabled": True},
            "prompt_composer": {"module": "prompt_composer.py", "enabled": True},
        },
        "manifest": {"active_sandbox_modules": ["state_features.py", "trigger_logic.py", "prompt_composer.py"]},
        "runtime_metadata": {
            "changed_files": list(files_to_modify),
            "diff_path": "parent_diff.patch",
            "parent_commit": "x",
        },
        "proposal_contract": {
            "parent_harness": "baseline3",
            "one_sentence_hypothesis": "tune trigger",
            "files_to_create_or_modify": list(files_to_modify),
        },
    }
    (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "proposer_note.txt").write_text("note\n", encoding="utf-8")
    (candidate_dir / "parent_diff.patch").write_text("diff\n", encoding="utf-8")
    if include_runtime_change:
        trigger_code = PARENT_TRIGGER.replace("0.5", "0.45")
    else:
        trigger_code = PARENT_TRIGGER
    (candidate_dir / "trigger_logic.py").write_text(trigger_code, encoding="utf-8")
    (candidate_dir / "state_features.py").write_text(PARENT_STATE, encoding="utf-8")
    (candidate_dir / "prompt_composer.py").write_text("def compose_prompt_context(stage, task_description, encoded_state, snapshot, spec):\n    return 'ctx'\n", encoding="utf-8")
    (candidate_dir / "validator_rules.py").write_text("def runtime_effect_modules():\n    return ['trigger_logic.py', 'state_features.py', 'prompt_composer.py']\n", encoding="utf-8")


def test_hard_guardrails_pass_for_minimal_valid_candidate(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(cand, files_to_modify=["spec.json", "trigger_logic.py", "proposer_note.txt"])
    out = validate_candidate_contract_alignment(cand, parent_dir=parent)
    assert "trigger_logic.py" in out["runtime_effect_modules"]


def test_hard_guardrails_fail_without_runtime_effect_change(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(cand, files_to_modify=["spec.json", "proposer_note.txt"], include_runtime_change=False)
    with pytest.raises(CandidateConsistencyError, match="runtime-effect"):
        validate_candidate_contract_alignment(cand, parent_dir=parent)


def test_hard_guardrails_fail_when_runtime_wiring_mismatch(tmp_path):
    parent = tmp_path / "baseline3"
    cand = tmp_path / "candidate_9999"
    _write_parent(parent)
    _write_candidate(cand, files_to_modify=["spec.json", "trigger_logic.py", "proposer_note.txt"])
    spec = json.loads((cand / "spec.json").read_text(encoding="utf-8"))
    spec["sandbox"]["trigger_logic"]["module"] = "wrong.py"
    (cand / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(CandidateConsistencyError, match="sandbox.trigger_logic.module mismatch"):
        validate_candidate_contract_alignment(cand, parent_dir=parent)


def test_propose_flow_runs_self_review_loop_and_emits_audit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    baseline = repo / "harnesses" / "baseline3"
    (repo / "harnesses" / "candidates").mkdir(parents=True, exist_ok=True)
    _write_parent(baseline)

    calls = {"self_review": 0}

    class _FakeLLM:
        def request(self, prompt: str, model_name: str, stream: bool = False):
            if "performing proposer self-review" in prompt:
                calls["self_review"] += 1
                if calls["self_review"] == 1:
                    return json.dumps({"status": "revise", "issues": ["tighten wiring"], "files_to_modify": ["trigger_logic.py"], "revision_plan": "fix trigger"})
                return json.dumps({"status": "pass", "issues": [], "files_to_modify": [], "revision_plan": "ok"})
            if "Requested file: spec.json" in prompt:
                return json.dumps(
                    {
                        "id": "candidate_0001",
                        "kind": "candidate",
                        "trigger_policy": {"module": "trigger_policy.py", "type": "event_predicted_collision_probability", "threshold": 0.45},
                        "state_encoder": {"module": "state_encoder.py", "include_fields": ["predicted_collision_probability"]},
                        "prompt_builder": {"module": "prompt_builder.py", "template_family": "baseline3_prompt", "include_example": True},
                    }
                )
            if "Requested file: trigger_logic.py" in prompt:
                return "def should_trigger_replan(state, memory, spec):\n    return (True, 'risk')\n"
            if "Requested file:" in prompt:
                return "def placeholder(*args, **kwargs):\n    return {}\n"
            return json.dumps(
                {
                    "parent_harness": "baseline3",
                    "candidate_id": "candidate_0001",
                    "one_sentence_hypothesis": "introduce tighter event threshold",
                    "weakness_being_addressed": "late event trigger",
                    "expected_tradeoff": "slightly more replans",
                    "expected_runtime_effect": "more responsive replan trigger",
                    "sandbox_modules_to_modify": ["trigger_logic.py", "state_features.py", "prompt_composer.py"],
                    "files_to_create_or_modify": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
                    "changed_files": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
                    "proposer_note_text": "placeholder",
                    "implementation_contract": {
                        "trigger_policy": {},
                        "state_encoder": {},
                        "prompt_builder": {},
                    },
                    "invariants": ["hard guardrails must pass"],
                }
            )

    monkeypatch.setitem(
        sys.modules,
        "controller.llm_wrapper",
        SimpleNamespace(LLMWrapper=lambda temperature=0.1: _FakeLLM(), MODEL_NAME="fake-model"),
    )
    monkeypatch.setattr(propose_candidate_module, "_run_candidate_smoke_checks", lambda _: None)
    monkeypatch.setattr(propose_candidate_module, "_run_import_checks", lambda _: None)
    monkeypatch.setattr(propose_candidate_module, "validate_candidate_contract_alignment", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TYPEFLY_PROPOSER_MODEL", "gpt-4.1")

    created = propose_next_candidate(repo_root=repo, focus_text="test consistency", max_revision_rounds=2)
    assert (created / "proposer_tool_audit.json").exists()
    spec = json.loads((created / "spec.json").read_text(encoding="utf-8"))
    assert spec["runtime_metadata"].get("proposer_tool_event_count", 0) > 0
    assert calls["self_review"] >= 2


def test_proposer_requires_openai_key_for_gpt_models(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    baseline = repo / "harnesses" / "baseline3"
    (repo / "harnesses" / "candidates").mkdir(parents=True, exist_ok=True)
    _write_parent(baseline)

    class _FakeLLM:
        def request(self, prompt: str, model_name: str, stream: bool = False):
            return "{}"

    monkeypatch.setitem(
        sys.modules,
        "controller.llm_wrapper",
        SimpleNamespace(LLMWrapper=lambda temperature=0.1: _FakeLLM(), MODEL_NAME="fake-model"),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TYPEFLY_PROPOSER_MODEL", "gpt-4.1")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        propose_next_candidate(repo_root=repo, focus_text="test consistency")


def test_propose_flow_backfills_missing_runtime_sandbox_modules_for_legacy_parent(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    baseline = repo / "harnesses" / "baseline3"
    (repo / "harnesses" / "candidates").mkdir(parents=True, exist_ok=True)
    _write_parent_legacy(baseline)

    class _FakeLLM:
        def request(self, prompt: str, model_name: str, stream: bool = False):
            if "performing proposer self-review" in prompt:
                return json.dumps({"status": "pass", "issues": [], "files_to_modify": [], "revision_plan": "ok"})
            if "Requested file: spec.json" in prompt:
                return json.dumps(
                    {
                        "id": "candidate_0001",
                        "kind": "candidate",
                        "trigger_policy": {"module": "trigger_policy.py", "type": "event_predicted_collision_probability", "threshold": 0.45},
                        "state_encoder": {"module": "state_encoder.py", "include_fields": ["predicted_collision_probability"]},
                        "prompt_builder": {"module": "prompt_builder.py", "template_family": "baseline3_prompt", "include_example": True},
                    }
                )
            if "Requested file: trigger_logic.py" in prompt:
                return "def should_trigger_replan(state, memory, spec):\n    return (True, 'risk')\n"
            if "Requested file:" in prompt:
                return "def placeholder(*args, **kwargs):\n    return {}\n"
            return json.dumps(
                {
                    "parent_harness": "baseline3",
                    "candidate_id": "candidate_0001",
                    "one_sentence_hypothesis": "test",
                    "weakness_being_addressed": "test",
                    "expected_tradeoff": "test",
                    "expected_runtime_effect": "test",
                    "sandbox_modules_to_modify": ["trigger_logic.py"],
                    "files_to_create_or_modify": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
                    "changed_files": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
                    "proposer_note_text": "test",
                    "implementation_contract": {"trigger_policy": {}, "state_encoder": {}, "prompt_builder": {}},
                    "invariants": ["test"],
                }
            )

    monkeypatch.setitem(
        sys.modules,
        "controller.llm_wrapper",
        SimpleNamespace(LLMWrapper=lambda temperature=0.1: _FakeLLM(), MODEL_NAME="fake-model"),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TYPEFLY_PROPOSER_MODEL", "gpt-4.1")
    created = propose_next_candidate(repo_root=repo, focus_text="legacy parent test", max_revision_rounds=0)
    assert (created / "state_features.py").exists()
    assert (created / "prompt_composer.py").exists()
