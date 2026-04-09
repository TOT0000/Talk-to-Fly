from pathlib import Path


PLANNER_SOURCE_PATH = Path('controller/llm_planner.py')
CONTROLLER_SOURCE_PATH = Path('controller/llm_controller.py')


def test_hardgate_rule_is_prompt_level_only():
    source = CONTROLLER_SOURCE_PATH.read_text(encoding='utf-8')
    assert 'current_collision_probability > 0.5, you MUST output response=full_replan_plan' not in source


def test_heartbeat_prompt_contains_hardgate_rule():
    source = PLANNER_SOURCE_PATH.read_text(encoding='utf-8')
    assert 'current_collision_probability > 0.5' in source
    assert 'MUST output response=full_replan_plan' in source


def test_no_two_stage_decision_call_for_heartbeat():
    source = PLANNER_SOURCE_PATH.read_text(encoding='utf-8')
    assert 'response must be one of: continue, full_replan_plan' in source
    assert 'decision=Replan' not in source
