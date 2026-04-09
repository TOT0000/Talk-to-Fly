from types import SimpleNamespace

import pytest

from controller.minispec_interpreter import MiniSpecInterpreter, MiniSpecReturnValue, Statement
from controller.abs.skill_item import SkillArg
from controller.skillset import LowLevelSkillItem, SkillSet


class _NoopLogger:
    def start_run(self, **kwargs):
        return None

    def update_plan_info(self, *args, **kwargs):
        return None

    def update_baseline_info(self, *args, **kwargs):
        return None

    def update_planner_info(self, *args, **kwargs):
        return None

    def update_execution_info(self, *args, **kwargs):
        return None

    def consume_runtime_snapshot(self, *args, **kwargs):
        return None

    def end_run(self, *args, **kwargs):
        return None


def _install_minimal_skillset(gc_replan: bool):
    low = SkillSet(level="low")
    counters = {"gc": 0, "d": 0}

    def _gc(_checkpoint):
        counters["gc"] += 1
        return "risk_abort", gc_replan

    def _d(_seconds):
        counters["d"] += 1
        return None, False

    low.add_skill(LowLevelSkillItem("go_checkpoint", _gc, args=[SkillArg("checkpoint_id", str)]))
    low.add_skill(LowLevelSkillItem("delay", _d, args=[SkillArg("seconds", float)]))

    Statement.low_level_skillset = low
    Statement.high_level_skillset = SkillSet(level="high", lower_level_skillset=low)
    return counters


def test_gc_replan_interrupt_clears_remaining_queue_and_skips_following_statements():
    counters = _install_minimal_skillset(gc_replan=True)
    interpreter = MiniSpecInterpreter(message_queue=None)

    interpreter.execute(["gc('A1');d(2.0);gc('A2');"])
    ret_val = interpreter.ret_queue.get(timeout=1.0)

    assert ret_val.replan is True
    assert counters["gc"] == 1
    assert counters["d"] == 0
    assert Statement.execution_queue.empty()


def test_typefly_replan_uses_fresh_llm_response_and_discards_old_queue(monkeypatch):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)

    displayed_messages = []
    queued_programs = []
    planner_calls = []

    class _Planner:
        def __init__(self):
            self._responses = ["gc('A1');d(2.0);", "ml(1.0);gc('A2');d(2.0);"]

        def plan(self, **kwargs):
            planner_calls.append(kwargs)
            return self._responses[len(planner_calls) - 1]

    def _execute_minispec_stub(program_text, silent=False, allow_auto_interrupt=True):
        queued_programs.append(program_text)
        if len(queued_programs) == 1:
            return MiniSpecReturnValue("MiniSpec execution interrupted for replan: current_collision_probability=0.700000", True)
        return MiniSpecReturnValue("ok", False)

    controller.controller_wait_takeoff = False
    controller.message_queue = None
    controller.execution_history = []
    controller.current_plan = None
    controller.framework_mode = "typefly-threshold-replan"
    controller.execution_mode = "Waiting"
    controller.active_objective_set = {"active_checkpoint_ids": []}
    controller.latest_benchmark_progress = {"completed": []}
    controller._benchmark_plan_checkpoint_sequence = []
    controller._task_id_counter = 0
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0
    controller.planner_mode = "llm"
    controller.planner = _Planner()
    controller.task_run_logger = _NoopLogger()
    controller.vision = SimpleNamespace(get_obj_list=lambda: "")
    controller.enable_video = False
    controller.state_provider = SimpleNamespace(debug_log_latest_localization_snapshot=lambda **kwargs: None)
    controller.safety_assessor = SimpleNamespace(build_from_provider=lambda provider: None)

    controller.append_message = displayed_messages.append
    controller._resolve_active_objective_set = lambda task_text: {"active_checkpoint_ids": []}
    controller._reset_benchmark_progress_tracking = lambda: None
    controller._format_planner_location_info = lambda: "loc"
    controller.get_live_ui_snapshot = lambda: {"safety_context": None, "benchmark_progress": {"completed": []}}
    controller._debug_log_safety_context = lambda safety: None
    controller._build_baseline_control_plan = lambda **kwargs: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.execute_minispec = _execute_minispec_stub
    controller.get_active_scenario_name = lambda: "test"

    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)

    controller.execute_task_description("run mission", framework_mode="typefly-threshold-replan")

    assert len(planner_calls) == 2
    assert queued_programs == ["gc('A1');d(2.0);", "ml(1.0);gc('A2');d(2.0);"]
    assert queued_programs[1] != queued_programs[0]
    plan_markers = [msg for msg in displayed_messages if isinstance(msg, str) and msg.startswith("[Plan]:")]
    assert len(plan_markers) == 2
