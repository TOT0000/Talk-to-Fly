from __future__ import annotations

import json
from pathlib import Path

from proposer.candidate_manifest import validate_manifest_schema
from proposer.contract_validator import validate_candidate_contract
from proposer.error_classifier import classify_failure
from proposer.runtime_verifier import verify_runtime_artifact


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _candidate_spec() -> dict:
    return {
        "id": "candidate_0099",
        "kind": "candidate",
        "parent": "baseline1",
        "candidate_manifest": {
            "candidate_id": "candidate_0099",
            "parent_candidate_ids": ["baseline1"],
            "change_axes": ["trigger", "state"],
            "hypothesis": "reduce unnecessary replans",
            "expected_metric_direction": {"collision_count": "down"},
            "safety_constraints": {"safety_priority": True},
            "artifact_bindings": {
                "spec": "spec.json",
                "trigger_module": "trigger_logic.py",
                "state_module": "state_features.py",
                "prompt_module": "prompt_composer.py",
            },
            "declared_files_changed": ["spec.json", "trigger_logic.py", "state_features.py", "prompt_composer.py"],
            "evaluation_requirements": {"require_runtime_verifier": True},
            "rollback_conditions": {"collision_regression": True},
        },
    }


def test_candidate_manifest_schema_test():
    payload = _candidate_spec()["candidate_manifest"]
    out = validate_manifest_schema(payload)
    assert out.valid is True


def test_contract_validator_with_declared_files(tmp_path):
    parent = tmp_path / "parent"
    cand = tmp_path / "cand"
    _write(parent / "spec.json", json.dumps(_candidate_spec(), ensure_ascii=False, indent=2))
    _write(parent / "trigger_logic.py", "def should_trigger_replan(state, memory, spec):\n    return False, 'x'\n")
    _write(parent / "state_features.py", "def encode_state_features(snapshot, spec):\n    return {}\n")
    _write(parent / "prompt_composer.py", "def compose_prompt_context(stage, task_description, encoded_state, snapshot, spec):\n    return ''\n")

    spec = _candidate_spec()
    spec["new_field"] = "changed"
    _write(cand / "spec.json", json.dumps(spec, ensure_ascii=False, indent=2))
    _write(cand / "trigger_logic.py", "def should_trigger_replan(state, memory, spec):\n    return True, 'risk'\n")
    _write(cand / "state_features.py", "def encode_state_features(snapshot, spec):\n    return {'x':1}\n")
    _write(cand / "prompt_composer.py", "def compose_prompt_context(stage, task_description, encoded_state, snapshot, spec):\n    return 'ctx'\n")
    result = validate_candidate_contract(cand, parent)
    assert result["candidate_id"] == "candidate_0099"


def test_prompt_source_binding_test(tmp_path):
    rt = tmp_path / "runtime.jsonl"
    pt = tmp_path / "planning.jsonl"
    _write(rt, json.dumps({"event_type": "noop"}) + "\n")
    _write(pt, json.dumps({"planning_stage": "heartbeat", "replan_applied": True}) + "\n")
    metadata = {
        "evaluate_prompt_source": {
            "selected_prompt_asset_path": "/tmp/prompt.txt",
            "rendered_prompt_hash_sha256": "abc",
        },
        "run_summary": {"selected_trigger_policy_name": "tp1", "selected_harness_spec_path": "harnesses/c1/spec.json"},
        "provenance": {
            "code_commit_hash": "h",
            "config_hash": "h",
            "prompt_asset_hash": "h",
            "rendered_prompt_hash": "h",
            "state_schema_version": "v1",
            "trigger_policy_version": "v1",
            "benchmark_pack_id": "pack",
            "scene_id": "SCENE1",
            "zone_id": "zoneA",
            "seed": 0,
            "evaluator_version": "ev",
            "trace_schema_version": "tv",
        },
    }
    run = {
        "scene_id": "SCENE1",
        "task_zone": "zoneA",
        "planning_trace_path": pt.as_posix(),
        "runtime_trace_path": rt.as_posix(),
    }
    out = verify_runtime_artifact(candidate_manifest={}, run_artifact=run, metadata=metadata)
    assert out.checks["prompt_source_bound"] is True
    assert out.checks["module_binding_bound"] is True


def test_active_zone_correctness_and_mission_success_scope_test(tmp_path):
    _write(tmp_path / "rt.jsonl", "{}\n")
    _write(tmp_path / "pt.jsonl", json.dumps({"planning_stage": "heartbeat"}) + "\n")
    metadata = {
        "evaluate_prompt_source": {"selected_prompt_asset_path": "/tmp/p", "rendered_prompt_hash_sha256": "x"},
        "run_summary": {"selected_trigger_policy_name": "ok"},
        "provenance": {
            "code_commit_hash": "h", "config_hash": "h", "prompt_asset_hash": "h", "rendered_prompt_hash": "h",
            "state_schema_version": "v1", "trigger_policy_version": "v1", "benchmark_pack_id": "pack",
            "scene_id": "SCENE2", "zone_id": "zoneB", "seed": 0, "evaluator_version": "v", "trace_schema_version": "v",
        },
    }
    good = verify_runtime_artifact(candidate_manifest={}, run_artifact={"scene_id": "SCENE2", "task_zone": "zoneB", "planning_trace_path": (tmp_path / 'pt.jsonl').as_posix(), "runtime_trace_path": (tmp_path / 'rt.jsonl').as_posix()}, metadata=metadata)
    bad = verify_runtime_artifact(candidate_manifest={}, run_artifact={"scene_id": "SCENE2", "task_zone": "zoneA", "planning_trace_path": (tmp_path / 'pt.jsonl').as_posix(), "runtime_trace_path": (tmp_path / 'rt.jsonl').as_posix()}, metadata=metadata)
    assert good.checks["active_zone_correct"] is True
    assert bad.checks["active_zone_correct"] is False


def test_heartbeat_logging_and_replan_event_logging_and_trace_completeness(tmp_path):
    _write(tmp_path / "rt.jsonl", json.dumps({"event_type": "replan_interrupt"}) + "\n")
    _write(tmp_path / "pt.jsonl", json.dumps({"planning_stage": "heartbeat", "replan_applied": True}) + "\n")
    metadata = {
        "evaluate_prompt_source": {"selected_prompt_asset_path": "/tmp/p", "rendered_prompt_hash_sha256": "x"},
        "run_summary": {"selected_harness_spec_path": "x"},
        "provenance": {
            "code_commit_hash": "h", "config_hash": "h", "prompt_asset_hash": "h", "rendered_prompt_hash": "h",
            "state_schema_version": "v1", "trigger_policy_version": "v1", "benchmark_pack_id": "pack",
            "scene_id": "SCENE1", "zone_id": "zoneA", "seed": 0, "evaluator_version": "v", "trace_schema_version": "v",
        },
    }
    run = {"scene_id": "SCENE1", "task_zone": "zoneA", "planning_trace_path": (tmp_path / 'pt.jsonl').as_posix(), "runtime_trace_path": (tmp_path / 'rt.jsonl').as_posix()}
    out = verify_runtime_artifact(candidate_manifest={}, run_artifact=run, metadata=metadata)
    assert out.checks["heartbeat_observed"] is True
    assert out.checks["replan_trace_observed"] is True
    assert out.checks["provenance_complete"] is True


def test_screening_formal_evaluator_separation_test():
    from proposer.evaluation_pipeline import ScreeningEvaluator, FormalEvaluator

    s = ScreeningEvaluator()
    f = FormalEvaluator()
    assert s.name == "screening"
    assert f.name == "formal"


def test_error_classifier_test():
    out = classify_failure(
        verification={"prompt_source_bound": False, "active_zone_correct": True, "provenance_complete": True},
        run={"mission_success": False, "run_status": "failed"},
        metadata={"evaluate_error_report": {"error_type": "x"}},
    )
    assert out == "runtime_wiring_error"
