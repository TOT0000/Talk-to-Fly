import pytest
pytest.importorskip("PIL")
import queue
import time
from types import SimpleNamespace

import pytest

from controller.llm_controller import LLMController
from serving.webui.typefly import TypeFlyWebUI


class _FakeDrone:
    def __init__(self):
        self.hold_calls = 0
        self.move_calls = 0

    def hold_position(self):
        self.hold_calls += 1
        return (0.0, 0.0, 1.0)

    def get_ground_truth_drone_position(self):
        return (0.0, 0.0, 1.0)


def _minimal_controller():
    c = LLMController.__new__(LLMController)
    c.awaiting_llm_response = True
    c.llm_wait_hover_active = True
    c._planning_inflight = True
    c._pending_heartbeat_replan_plan = None
    c.gc_llm_wait_pause_count = 0
    c.gc_llm_wait_total_sec = 0.0
    c.gc_llm_wait_resume_count = 0
    c.gc_llm_wait_replan_preempt_count = 0
    c.drone = _FakeDrone()
    return c


def test_pause_gc_for_llm_wait_resume_on_continue():
    c = _minimal_controller()

    def _consume():
        c.awaiting_llm_response = False
        c.llm_wait_hover_active = False
        c._planning_inflight = False
        return False

    c._consume_planning_response_queue = _consume
    ret = c._pause_gc_for_llm_wait("C5")
    assert ret is False
    assert c.drone.hold_calls >= 1
    assert c.gc_llm_wait_pause_count == 1
    assert c.gc_llm_wait_resume_count == 1


def test_pause_gc_for_llm_wait_preempt_on_replan():
    c = _minimal_controller()

    def _consume():
        c.awaiting_llm_response = False
        c.llm_wait_hover_active = False
        c._planning_inflight = False
        c._pending_heartbeat_replan_plan = {"parsed_plan": "gc('C6');"}
        return True

    c._consume_planning_response_queue = _consume
    ret = c._pause_gc_for_llm_wait("C5")
    assert ret is True
    assert c.gc_llm_wait_replan_preempt_count == 1


def test_trajectory_sampler_uses_lightweight_reader_not_snapshot(monkeypatch):
    c = LLMController.__new__(LLMController)
    c.execution_mode = "Executing"
    c.latest_benchmark_progress = {"current_target": "C1"}
    c._uav_trajectory_lock = __import__("threading").Lock()
    c._uav_trajectory_epoch = 0
    c._latest_uav_trajectory_points = []
    c._latest_uav_trajectory_stats = {}
    c._uav_trajectory_sampler_stop_event = None
    c._uav_trajectory_sampler_thread = None
    c._uav_trajectory_sampler_interval_sec = 0.1
    c._uav_trajectory_sampler_active_during_run = False
    c.drone = SimpleNamespace(get_ground_truth_drone_position=lambda: (1.0, 2.0, 3.0))
    c.get_live_ui_snapshot = lambda: (_ for _ in ()).throw(AssertionError("should not be called"))

    c.start_uav_trajectory_sampler(0.1)
    time.sleep(1.05)
    c.stop_uav_trajectory_sampler()

    stats = getattr(c, "_latest_uav_trajectory_stats", {})
    assert len(c.get_uav_trajectory_points()) >= 10
    assert float(stats.get("trajectory_mean_sample_dt_sec") or 0.0) <= 0.25


def test_ui_trajectory_prefers_controller_buffer():
    ui = TypeFlyWebUI.__new__(TypeFlyWebUI)
    ui.llm_controller = SimpleNamespace(get_uav_trajectory_points=lambda: [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    ui.uav_trajectory_points = [{"x": 9, "y": 9}]
    ui.position_history = {"drone_gt": [(7, 7, 0)]}
    hist = ui._trajectory_xy_history()
    assert hist == [(1.0, 2.0), (3.0, 4.0)]
