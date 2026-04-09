from pathlib import Path


SOURCE_PATH = Path("controller/llm_controller.py")


def test_agent_mode_no_longer_bypasses_interpreter_collision_abort():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert 'if str(getattr(self, "framework_mode", "")).strip().lower() == "langgraph_agent":' not in source


def test_gc_loop_collision_abort_not_guarded_by_framework_mode():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "if self.framework_mode != \"langgraph_agent\":" not in source
    assert "if self._should_trigger_auto_replan(current_p, source=\"go_checkpoint_loop\")" in source
