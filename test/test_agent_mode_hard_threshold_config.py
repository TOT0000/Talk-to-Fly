from pathlib import Path


SOURCE_PATH = Path("controller/llm_controller.py")
AGENT_SOURCE_PATH = Path("controller/langgraph_agent.py")
PLANNER_SOURCE_PATH = Path("controller/llm_planner.py")


def test_agent_mode_no_longer_bypasses_interpreter_collision_abort():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert 'if str(getattr(self, "framework_mode", "")).strip().lower() == "langgraph_agent":' not in source


def test_gc_loop_collision_abort_not_guarded_by_framework_mode():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "if self.framework_mode != \"langgraph_agent\":" not in source
    assert "if self._should_trigger_auto_replan(current_p, source=\"go_checkpoint_loop\")" in source


def test_agent_mode_stall_tracking_fields_removed():
    agent_source = AGENT_SOURCE_PATH.read_text(encoding="utf-8")
    planner_source = PLANNER_SOURCE_PATH.read_text(encoding="utf-8")
    assert "no_progress_steps" not in agent_source
    assert "repeated_action_count" not in agent_source
    assert "stall_count" not in planner_source
    assert "repeated_action_count" not in planner_source
