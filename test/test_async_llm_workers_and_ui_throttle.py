import json
import pytest
import queue
import threading
import time
from types import SimpleNamespace


from controller.task_run_logger import TaskRunLogger


class _TraceLogger:
    def __init__(self):
        self.traces = []

    def append_planning_trace(self, trace):
        self.traces.append(dict(trace or {}))


class _Planner:
    heartbeat_model_name = "planner-small"
    evaluator_model_name = "eval-small"

    def __init__(self, response=None, block_event=None):
        self.calls = []
        self.eval_calls = []
        self.response = response or {"response": "continue", "reason": "ok", "plan": "", "raw_response": '{"response":"continue"}', "parsed_ok": True}
        self.block_event = block_event

    def plan_agent_heartbeat(self, **kwargs):
        self.calls.append(kwargs)
        if self.block_event is not None:
            self.block_event.wait(timeout=1.0)
        return dict(self.response)

    def get_last_heartbeat_trace(self):
        return {"prompt": "p", "raw_response": self.response.get("raw_response", ""), "used_model_name": self.heartbeat_model_name}

    def evaluate_agent_replan_record(self, record):
        self.eval_calls.append(record)
        return {"prompt": "ep", "raw_response": '{"confidence":"high"}', "parsed": {"confidence": "high"}, "parsed_ok": True, "used_model_name": self.evaluator_model_name}


def _controller(planner):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController
    c = LLMController.__new__(LLMController)
    c.framework_mode = "agent-heartbeat-soft"
    c.selected_pipeline_id = "agent"
    c.baseline_scene_id = "SCENE"
    c.archive_enabled = True
    c._pending_heartbeat_replan_plan = None
    c._pending_heartbeat_reason = ""
    c._pending_heartbeat_replan_id = ""
    c.last_heartbeat_ts = 0.0
    c.heartbeat_interval_seconds = 5.0
    c.replan_limit = 8
    c._replan_attempts = 0
    c.current_task_description = "task"
    c.execution_history = []
    c.current_plan = "gc('A1');"
    c._mission_original_plan = c.current_plan
    c._current_active_plan = c.current_plan
    c._latest_full_replan_response = None
    c._accepted_replan_ids = set()
    c._accepted_replan_seq = 0
    c.latest_benchmark_progress = {"completed": [], "current_target": "A1"}
    c.active_objective_set = {"active_checkpoint_ids": ["A1", "A2"]}
    c._runtime_replan_event = threading.Event()
    c._runtime_replan_reason = ""
    c._replan_response_history = []
    c.mission_start_ts = time.time()
    c.planner = planner
    c.task_run_logger = _TraceLogger()
    c.append_message = lambda msg: None
    c._sanitize_minispec_plan = lambda raw: str(raw)
    c._record_replan_response = lambda **kwargs: c._replan_response_history.append(kwargs)
    c.replan_requested_count = 0
    c.full_replan_response_count = 0
    c.accepted_replan_count = 0
    c.replan_applied_count = 0
    c.replan_interrupt_count = 0
    c.replan_execution_resume_count = 0
    c.replan_discarded_count = 0
    c.replan_discard_reason_counts = {}
    c.latest_replan_request_reason = ""
    c.latest_replan_applied_plan = ""
    c.latest_replan_overwrote_previous_pending = False
    c.get_live_ui_snapshot = lambda: {"benchmark_progress": dict(c.latest_benchmark_progress), "active_objective_set": dict(c.active_objective_set), "workers": []}
    c._build_execution_history_for_llm = lambda: "history"
    return c


def test_planning_inflight_skips_second_heartbeat_and_logs_skip():
    block = threading.Event()
    planner = _Planner(block_event=block)
    c = _controller(planner)

    assert c._maybe_run_agent_heartbeat(force=True) == "request_started"
    assert c._planning_inflight is True
    assert c._maybe_run_agent_heartbeat(force=True) == "none"
    block.set()
    c._planning_worker_thread.join(timeout=1.0)
    c._consume_planning_response_queue()

    assert len(planner.calls) == 1
    assert not any(t.get("skipped_due_to_inflight") and t.get("llm_call_role") == "heartbeat" for t in c.task_run_logger.traces)


def test_planning_continue_and_full_replan_and_stale_response_handling():
    cont = _controller(_Planner())
    cont._maybe_run_agent_heartbeat(force=True)
    cont._planning_worker_thread.join(timeout=1.0)
    assert cont._consume_planning_response_queue() is False
    assert cont._pending_heartbeat_replan_plan is None
    cont_ts = cont.last_planning_response_received_ts
    assert cont.next_planning_allowed_ts >= cont_ts + 5.0

    replan_response = {"response": "full_replan_plan", "reason": "risk", "plan": "ml(0.3);gc('A2');", "raw_response": '{"response":"full_replan_plan"}', "parsed_ok": True}
    repl = _controller(_Planner(response=replan_response))
    repl._maybe_run_agent_heartbeat(force=True)
    repl._planning_worker_thread.join(timeout=1.0)
    assert repl._consume_planning_response_queue() is True
    assert repl._pending_heartbeat_replan_plan == "ml(0.3);gc('A2');"
    assert repl._runtime_replan_event.is_set()

    stale = _controller(_Planner(response=replan_response))
    stale._maybe_run_agent_heartbeat(force=True)
    stale.latest_benchmark_progress["current_target"] = "A2"
    stale._planning_worker_thread.join(timeout=1.0)
    assert stale._consume_planning_response_queue() is True
    assert stale._pending_heartbeat_replan_plan == "ml(0.3);gc('A2');"
    assert not any(t.get("response_discarded_reason") == "target_checkpoint_changed" for t in stale.task_run_logger.traces)
    assert stale.replan_requested_count == 1
    stale_ts = stale.last_planning_response_received_ts
    assert stale.next_planning_allowed_ts >= stale_ts + 5.0


def test_accepted_replan_count_is_idempotent_for_same_replan_id():
    c = _controller(_Planner())
    rid = c._new_replan_id()
    assert c._accept_replan_once(rid) is True
    assert c._accept_replan_once(rid) is False
    assert c.accepted_replan_count == 1
    assert c._replan_attempts == 1


def test_evaluator_worker_has_independent_inflight_and_latency_trace():
    planner = _Planner()
    c = _controller(planner)
    c._agent_ready_for_eval_records = [{"outcome_delta": {"replan_heartbeat_index": 1}}, {"outcome_delta": {"replan_heartbeat_index": 2}}]
    c._agent_eval_inflight = True
    c._start_agent_eval_worker_if_needed()
    assert len(planner.eval_calls) == 0
    assert any(t.get("skipped_due_to_inflight") and t.get("llm_call_role") == "evaluator" for t in c.task_run_logger.traces)

    c._agent_eval_inflight = False
    c._start_agent_eval_worker_if_needed()
    assert c._agent_eval_thread is not None
    assert c._planning_inflight is False
    c._agent_eval_thread.join(timeout=1.0)
    c._commit_agent_eval_results(current_heartbeat_index=3)
    assert len(planner.eval_calls) == 1
    assert any(t.get("llm_call_role") == "evaluator" and t.get("latency_sec") is not None for t in c.task_run_logger.traces)

    c._agent_eval_inflight = True
    assert c._maybe_run_agent_heartbeat(force=True) == "request_started"
    assert c._planning_inflight is True


def test_response_driven_window_starts_heartbeat_after_initial_response():
    planner = _Planner()
    c = _controller(planner)
    c.heartbeat_interval_seconds = 3.0
    t0 = time.time()
    c.last_planning_response_received_ts = t0
    c.next_planning_allowed_ts = t0 + 3.0
    assert c._maybe_run_agent_heartbeat(force=False) == "none"
    c.next_planning_allowed_ts = time.time() - 0.01
    status = c._maybe_run_agent_heartbeat(force=False)
    assert status == "request_started"
    c._planning_worker_thread.join(timeout=1.0)
    assert len(planner.calls) == 1
    assert c.llm_wait_event_count >= 1


def test_ui_render_throttle_uses_gr_update_without_skipping_history():
    pytest.importorskip("gradio")
    from serving.webui.typefly import TypeFly
    tf = TypeFly.__new__(TypeFly)
    tf.llm_controller = SimpleNamespace(
        get_live_ui_snapshot=lambda: {"drone_gt": (1, 2, 3), "drone_est": (1.1, 2.1, 3), "workers": [], "benchmark_progress": {}, "active_objective_set": {}},
        update_ui_collision_probability=lambda _p: None,
        task_run_logger=None,
    )
    tf.position_history = {"drone_gt": queue.deque(maxlen=100) if False else __import__('collections').deque(maxlen=100), "drone_est": __import__('collections').deque(maxlen=100)}
    tf.worker_collision_history = {"worker_1": __import__('collections').deque(maxlen=100), "worker_2": __import__('collections').deque(maxlen=100), "worker_3": __import__('collections').deque(maxlen=100)}
    tf.worker_collision_active = {"worker_1": False, "worker_2": False, "worker_3": False}
    tf.benchmark_progress = {"completed": set(), "active_enter_ts": None, "active_progress": 0.0, "order": []}
    tf.mission_clock = {"started_at": None, "completed_at": None, "is_running": False, "objective_completed": False}
    tf.mission_collision_count = 0
    tf._last_workspace_render_ts = time.time()
    tf._last_probability_render_ts = time.time()
    tf._last_status_render_ts = time.time()
    tf._last_postrun_render_ts = time.time()
    tf._workspace_render_interval_sec = 999
    tf._probability_render_interval_sec = 999
    tf._status_render_interval_sec = 999
    tf._postrun_render_interval_sec = 999
    tf._sync_objective_state = lambda snapshot: None
    tf._update_mission_collision_count = lambda snapshot: None
    tf._update_checkpoint_progress = lambda snapshot: None

    outputs = tf.update_and_step(0)
    assert outputs[1].get("__type__") == "update"
    assert outputs[3].get("__type__") == "update"
    assert len(tf.position_history["drone_gt"]) == 1
    assert len(tf.worker_collision_history["worker_1"]) == 1


def test_trajectory_xy_history_prefers_controller_sampler_points():
    pytest.importorskip("gradio")
    from serving.webui.typefly import TypeFly
    tf = TypeFly.__new__(TypeFly)
    tf.llm_controller = SimpleNamespace(get_uav_trajectory_points=lambda: [{"x": 9.0, "y": 8.0, "z": 1.0}])
    tf.uav_trajectory_points = __import__('collections').deque([{"x": 1.0, "y": 2.0, "z": 3.0}], maxlen=100)
    tf.position_history = {"drone_gt": __import__('collections').deque([(3.0, 4.0)], maxlen=100)}
    assert tf._trajectory_xy_history() == [(9.0, 8.0)]


def test_logger_latency_summary_excludes_skips_and_computes_parse_rate(tmp_path):
    logger = TaskRunLogger(excel_path=str(tmp_path / "runs.xlsx"))
    logger.start_run("task", "text", "scene", {"benchmark_progress": {}, "active_objective_set": {}})
    logger.append_planning_trace({"planning_stage": "heartbeat", "llm_call_id": "p1", "llm_call_role": "heartbeat", "latency_sec": 1.0, "json_parse_success": True, "skipped_due_to_inflight": False})
    logger.append_planning_trace({"planning_stage": "heartbeat", "llm_call_role": "heartbeat", "skipped_due_to_inflight": True})
    logger.append_planning_trace({"planning_stage": "evaluator", "llm_call_id": "e1", "llm_call_role": "evaluator", "latency_sec": 3.0, "json_parse_success": False, "skipped_due_to_inflight": False})
    logger.end_run("completed")
    summary = logger.get_pending_run_summary()
    assert summary["actual_planning_call_count"] == 1
    assert summary["actual_evaluator_call_count"] == 1
    assert summary["planning_skipped_due_to_inflight_count"] == 1
    assert summary["actual_llm_request_count"] == 2
    assert summary["planning_latency_mean_sec"] == 1.0
    assert summary["evaluator_latency_median_sec"] == 3.0
    assert summary["all_llm_latency_p95_sec"] is not None
    assert summary["json_parse_success_rate"] == 0.5
    assert logger.save_pending_run() is True
    saved = json.loads((tmp_path / "runs" / summary["run_id"] / f"{summary['run_id']}_summary.json").read_text())
    assert saved["all_llm_latency_mean_sec"] == 2.0


def test_reset_runtime_records_clears_ui_and_controller_trajectory_buffers():
    pytest.importorskip("gradio")
    from serving.webui.typefly import TypeFly
    from collections import deque

    tf = TypeFly.__new__(TypeFly)
    class DummyController:
        def __init__(self):
            self._latest_uav_trajectory_points = [{"x": 1.0, "y": 1.0, "z": 1.0}]
            self._latest_uav_trajectory_stats = {"trajectory_sample_count": 3}
            self.cleared = False
        def clear_uav_trajectory(self):
            self._latest_uav_trajectory_points = []
            self._latest_uav_trajectory_stats = {
                "trajectory_sample_count": 0,
                "trajectory_buffer_source": "reset_empty",
            }
            self.cleared = True
        def get_uav_trajectory_points(self):
            return list(self._latest_uav_trajectory_points)

    tf.llm_controller = DummyController()
    tf.position_history = {"drone_gt": deque([(1.0,2.0,3.0)], maxlen=100), "drone_est": deque([(1.1,2.1,3.1)], maxlen=100)}
    tf.uav_trajectory_points = deque([{"x": 1.0, "y": 2.0, "z": 3.0}], maxlen=100)
    tf.worker_collision_history = {"worker_1": deque([0.1], maxlen=100), "worker_2": deque([0.2], maxlen=100), "worker_3": deque([0.3], maxlen=100)}
    tf.worker_collision_active = {"worker_1": True, "worker_2": True, "worker_3": True}
    tf.objective_state = {"active_checkpoint_ids": set()}
    tf.benchmark_progress = {"order": [], "completed": set(), "active_enter_ts": None, "active_progress": 0.0, "current_target": None, "executed_gc_sequence": []}
    tf.mission_clock = {"started_at": 1.0, "completed_at": 2.0, "is_running": True, "objective_completed": True}
    tf.mission_collision_count = 10
    tf._last_position_map = {"drone": (1,2,3)}
    tf._last_print_position_map = {"drone": (1,2,3)}
    tf.uwb_queue = queue.Queue()
    tf.virtual_queue = queue.Queue()
    tf.message_queue = queue.Queue()

    tf._reset_runtime_records()

    assert len(tf.uav_trajectory_points) == 0
    assert len(tf.position_history["drone_gt"]) == 0
    assert len(tf.position_history["drone_est"]) == 0
    assert tf.llm_controller._latest_uav_trajectory_points == []
    assert tf.llm_controller._latest_uav_trajectory_stats["trajectory_sample_count"] == 0
    assert tf.llm_controller.cleared is True
    assert tf._trajectory_xy_history() == []

def test_should_skip_heartbeat_after_task_completion_returns_bool_only():
    c = _controller(_Planner())

    c.latest_benchmark_progress = {"completed": [], "current_target": "A1"}
    c.active_objective_set = {"active_checkpoint_ids": ["A1"]}
    assert c._should_skip_heartbeat_after_task_completion() is False
    assert isinstance(c._should_skip_heartbeat_after_task_completion(), bool)

    class _Q:
        def qsize(self):
            return 1

    from controller import llm_controller as llm_controller_module
    old_queue = llm_controller_module.Statement.execution_queue
    llm_controller_module.Statement.execution_queue = _Q()
    try:
        c.latest_benchmark_progress = {"completed": ["A1"], "current_target": "A1"}
        assert c._should_skip_heartbeat_after_task_completion() is False
    finally:
        llm_controller_module.Statement.execution_queue = old_queue

    c.latest_benchmark_progress = {"completed": ["A1"], "current_target": "A1"}
    c.active_objective_set = {"active_checkpoint_ids": ["A1"]}
    assert c._should_skip_heartbeat_after_task_completion() is True


def test_response_driven_heartbeat_not_misclassified_as_completed():
    c = _controller(_Planner())
    c.latest_benchmark_progress = {"completed": [], "current_target": "A1"}
    c.active_objective_set = {"active_checkpoint_ids": ["A1"]}
    c._planning_inflight = False
    c.awaiting_llm_response = False
    c.next_planning_allowed_ts = time.time() - 0.01

    status = c._maybe_run_agent_heartbeat(force=False)
    assert status == "request_started"
    assert c.last_heartbeat_skip_reason != "active_objective_completed"
    assert c.heartbeat_request_started_count >= 1
    c._planning_worker_thread.join(timeout=1.0)


def test_no_truthy_string_regression_for_bool_helpers():
    from pathlib import Path

    source = Path('controller/llm_controller.py').read_text(encoding='utf-8')
    for fn_name in ('def _should_skip_heartbeat_after_task_completion', 'def _should_trigger_auto_replan'):
        start = source.index(fn_name)
        end = source.find('\n    def ', start + 1)
        if end == -1:
            end = len(source)
        fn_source = source[start:end]
        assert 'return "none"' not in fn_source
        assert 'return "request_started"' not in fn_source
        assert 'return True,' not in fn_source
        assert 'return False,' not in fn_source
