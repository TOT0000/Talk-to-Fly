from pathlib import Path

PLANNER_SOURCE = Path('controller/llm_planner.py').read_text(encoding='utf-8')
SOFT_EXAMPLES = Path('controller/assets/tello/agent_heartbeat_soft_examples.txt').read_text(encoding='utf-8')
HARD_EXAMPLES = Path('controller/assets/tello/agent_heartbeat_hardgate_examples.txt').read_text(encoding='utf-8')


def test_soft_heartbeat_prompt_injects_soft_examples():
    assert 'agent_heartbeat_soft_examples_path' in PLANNER_SOURCE
    assert 'self.agent_heartbeat_soft_examples' in PLANNER_SOURCE
    assert 'if hard_gate' in PLANNER_SOURCE
    assert 'else self.agent_heartbeat_soft_examples' in PLANNER_SOURCE


def test_hardgate_heartbeat_prompt_injects_hardgate_examples():
    assert 'agent_heartbeat_hardgate_examples_path' in PLANNER_SOURCE
    assert 'self.agent_heartbeat_hardgate_examples' in PLANNER_SOURCE
    assert 'self.agent_heartbeat_hardgate_examples' in PLANNER_SOURCE


def test_hardgate_prompt_has_extra_hard_gate_rule():
    assert 'If current_collision_probability > 0.5, you MUST output response=full_replan_plan with a new complete MiniSpec plan.' in PLANNER_SOURCE
    assert 'You may choose continue or full_replan_plan based on your judgment.' in PLANNER_SOURCE


def test_heartbeat_prompt_output_format_unchanged():
    assert 'Return strict JSON with keys: response, reason, plan.' in PLANNER_SOURCE
    assert 'response must be one of: continue, full_replan_plan.' in PLANNER_SOURCE
    assert 'If response=continue, set plan to empty string.' in PLANNER_SOURCE


def test_examples_files_are_real_and_nonempty_and_referenced():
    assert 'Example S1 (Soft heartbeat: continue current plan under low risk)' in SOFT_EXAMPLES
    assert 'Example H1 (HardGate heartbeat: below threshold, continue)' in HARD_EXAMPLES
    assert 'Agent heartbeat examples:' in PLANNER_SOURCE
    assert '{heartbeat_examples}' in PLANNER_SOURCE
