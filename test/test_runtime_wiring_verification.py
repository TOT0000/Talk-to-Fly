import inspect
import json
from pathlib import Path

from proposer.prompts import build_self_review_prompt
from proposer.propose_candidate import _runtime_wiring_smoke_verification


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _make_candidate_dir(tmp_path: Path) -> Path:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    _write(
        candidate_dir / "state_features.py",
        "def encode_state_features(snapshot, spec):\n    return {}\n",
    )
    _write(
        candidate_dir / "trigger_logic.py",
        "def should_trigger_replan(state, memory, spec):\n    return (False, 'ok')\n",
    )
    _write(
        candidate_dir / "prompt_composer.py",
        "def compose_prompt_context(stage, task_description, encoded_state, snapshot, spec):\n    return ''\n",
    )
    spec = {
        "id": "candidate_test",
        "kind": "candidate",
        "parent": "baseline2",
        "sandbox": {
            "state_features": {"module": "state_features.py", "enabled": True},
            "trigger_logic": {"module": "trigger_logic.py", "enabled": True},
            "prompt_composer": {"module": "prompt_composer.py", "enabled": True},
            "archive_selector": {"module": "archive_selector.py", "enabled": True},
            "validator_rules": {"module": "validator_rules.py", "enabled": True},
        },
        "manifest": {
            "active_sandbox_modules": [
                "state_features.py",
                "trigger_logic.py",
                "prompt_composer.py",
                "archive_selector.py",
                "validator_rules.py",
            ]
        },
    }
    (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return candidate_dir


def test_trigger_wiring_alignment_detected(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    proposal = {"sandbox_modules_to_modify": ["trigger_logic.py"]}
    out = _runtime_wiring_smoke_verification(candidate_dir=cdir, proposal_contract=proposal, changed_files={"trigger_logic.py", "spec.json"})
    assert out["trigger_alignment_ok"] is True
    assert out["loaded_trigger_module"] == "trigger_logic.py"
    assert out["loaded_trigger_function"] == "should_trigger_replan"


def test_state_wiring_alignment_detected(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    proposal = {"sandbox_modules_to_modify": ["state_features.py"]}
    out = _runtime_wiring_smoke_verification(candidate_dir=cdir, proposal_contract=proposal, changed_files={"state_features.py", "spec.json"})
    assert out["state_alignment_ok"] is True
    assert out["loaded_state_module"] == "state_features.py"
    assert out["loaded_state_function"] == "encode_state_features"


def test_prompt_wiring_alignment_detected(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    proposal = {"sandbox_modules_to_modify": ["prompt_composer.py"]}
    out = _runtime_wiring_smoke_verification(candidate_dir=cdir, proposal_contract=proposal, changed_files={"prompt_composer.py", "spec.json"})
    assert out["prompt_alignment_ok"] is True
    assert out["loaded_prompt_module"] == "prompt_composer.py"
    assert out["loaded_prompt_function"] == "compose_prompt_context"


def test_wiring_verification_fails_on_claim_runtime_mismatch(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    spec = json.loads((cdir / "spec.json").read_text(encoding="utf-8"))
    spec["sandbox"]["trigger_logic"]["module"] = "trigger_policy.py"
    (cdir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    _write(cdir / "trigger_policy.py", "def should_trigger_replan(state, memory, spec):\n    return (True, 'legacy')\n")
    proposal = {"sandbox_modules_to_modify": ["trigger_logic.py"]}
    out = _runtime_wiring_smoke_verification(candidate_dir=cdir, proposal_contract=proposal, changed_files={"trigger_logic.py", "spec.json"})
    assert out["passed"] is False
    assert out["trigger_alignment_ok"] is False


def test_self_review_prompt_includes_structured_wiring_verification():
    wiring = {"passed": False, "trigger_alignment_ok": False}
    prompt = build_self_review_prompt(
        proposal_contract_json="{}",
        candidate_spec_json="{}",
        changed_files_json="[]",
        runtime_wiring_verification_json=json.dumps(wiring),
        last_error="runtime wiring failed",
    )
    assert "Structured runtime wiring verification" in prompt
    assert '"trigger_alignment_ok": false' in prompt.lower()


def test_runtime_metadata_records_wiring_verification_artifact_path():
    src = inspect.getsource(__import__("proposer.propose_candidate", fromlist=["propose_next_candidate"]))
    assert "runtime_wiring_verification.json" in src
    assert "runtime_wiring_verification_path" in src
