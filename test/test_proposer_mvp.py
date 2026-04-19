from pathlib import Path

from controller.harness_protocol import EVALUATION_SCENE_TASK_MAPPING
from proposer.evaluate_candidate import mark_pareto
from proposer.registry import HarnessRegistry


def test_fixed_evaluation_protocol_mapping():
    assert EVALUATION_SCENE_TASK_MAPPING == {
        "SCENE1": "zoneA",
        "SCENE2": "zoneB",
        "SCENE3": "zoneC",
    }


def test_baselines_have_unified_harness_specs():
    reg = HarnessRegistry(Path(__file__).resolve().parents[1])
    ids = [x.harness_id for x in reg.list_baselines()]
    assert {"baseline1", "baseline2", "baseline3"}.issubset(set(ids))


def test_pareto_flagging():
    entries = [
        {"harness_id": "a", "metrics": {"collision_count_avg": 0, "near_miss_count_avg": 1, "completion_time_sec_avg": 10, "llm_call_count_avg": 5}},
        {"harness_id": "b", "metrics": {"collision_count_avg": 1, "near_miss_count_avg": 2, "completion_time_sec_avg": 11, "llm_call_count_avg": 8}},
    ]
    marked = {e["harness_id"]: e for e in mark_pareto(entries)}
    assert marked["a"]["pareto_frontier"] is True
    assert marked["b"]["pareto_frontier"] is False
