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
    c.get_live_ui_snapshot = lambda: {"benchmark_progress": dict(c.latest_benchmark_progress), "active_objective_set": dict(c.active_objective_set), "workers": []}
    c._build_execution_history_for_llm = lambda: "history"
    return c


def test_planning_inflight_skips_second_heartbeat_and_logs_skip():
    block = threading.Event()
    planner = _Planner(block_event=block)
    c = _controller(planner)

    assert c._maybe_run_agent_heartbeat(force=True) is False
    assert c._planning_inflight is True
    assert c._maybe_run_agent_heartbeat(force=True) is False
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
    assert stale._consume_planning_response_queue() is False
    assert stale._pending_heartbeat_replan_plan is None
    assert any(t.get("response_discarded_reason") == "target_checkpoint_changed" for t in stale.task_run_logger.traces)
    stale_ts = stale.last_planning_response_received_ts
    assert stale.next_planning_allowed_ts >= stale_ts + 5.0


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
    assert c._maybe_run_agent_heartbeat(force=True) is False
    assert c._planning_inflight is True


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
