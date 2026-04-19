import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.pipeline_registry import derive_runtime_mode, get_pipeline_config


def test_baseline_runtime_mode_mapping_from_spec_trigger_policy():
    baseline1 = get_pipeline_config("baseline1")
    baseline2 = get_pipeline_config("baseline2")
    baseline3 = get_pipeline_config("baseline3")

    assert baseline1.runtime_mode == "agent-heartbeat-soft"
    assert baseline2.runtime_mode == "agent-heartbeat-soft"
    assert baseline3.runtime_mode == "typefly-threshold-replan"
    assert baseline1.runtime_mode_source == "harness_spec"
    assert baseline3.runtime_mode_source == "harness_spec"


def test_candidate_runtime_mode_mapping_from_spec_trigger_policy():
    candidate = get_pipeline_config("candidate_0001")
    assert candidate.trigger_type == "hybrid"
    assert candidate.runtime_mode == "agent-heartbeat-soft"
    assert candidate.runtime_mode_source == "harness_spec"


def test_candidate_spec_params_are_exposed_to_runtime_config():
    candidate = get_pipeline_config("candidate_0001")
    assert candidate.trigger_params.get("heartbeat_seconds") == 5.0
    assert candidate.trigger_params.get("threshold") == 0.45
    assert candidate.harness_spec_path.endswith("harnesses/candidates/candidate_0001/spec.json")


def test_threshold_trigger_mapping_helper():
    mode, source = derive_runtime_mode("threshold", "typefly-oneshot")
    assert mode == "typefly-threshold-replan"
    assert source == "harness_spec"
