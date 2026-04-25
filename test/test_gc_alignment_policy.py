import math

import pytest


pytest.importorskip("PIL")
from controller import llm_controller as llm_mod
from controller.llm_controller import LLMController
from controller.baseline_scenes import BENCHMARK_CHECKPOINTS_BY_ID


class _DroneKinematics:
    def __init__(self, x: float, y: float, yaw_rad: float):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw_rad)
        self.actions = []

    def move_forward(self, dist: float):
        d = float(dist)
        self.actions.append(("move_forward", d))
        self.x += math.cos(self.yaw) * d
        self.y += math.sin(self.yaw) * d

    def move_left(self, dist: float):
        d = float(dist)
        self.actions.append(("move_left", d))
        self.x += math.cos(self.yaw + math.pi / 2.0) * d
        self.y += math.sin(self.yaw + math.pi / 2.0) * d

    def move_right(self, dist: float):
        d = float(dist)
        self.actions.append(("move_right", d))
        self.x += math.cos(self.yaw - math.pi / 2.0) * d
        self.y += math.sin(self.yaw - math.pi / 2.0) * d

    def turn_ccw(self, deg: int):
        self.actions.append(("turn_ccw", int(deg)))
        self.yaw += math.radians(float(deg))

    def turn_cw(self, deg: int):
        self.actions.append(("turn_cw", int(deg)))
        self.yaw -= math.radians(float(deg))


class _FakePx4Base:
    pass


class _FakePx4Drone(_FakePx4Base, _DroneKinematics):
    def __init__(self, x: float, y: float, yaw_rad: float):
        super().__init__(x, y, yaw_rad)
        self._active_command_name = "scenario_reposition"
        self._active_setpoint = (1.0, 1.0, -1.5, 0.0)
        self._active_target_source = "scenario_reposition_hold"
        self.takeover_calls = []

    def get_active_setpoint_snapshot(self):
        return {
            "command": self._active_command_name,
            "target": self._active_setpoint,
            "target_source": self._active_target_source,
        }

    def get_drone_position(self):
        return (self.x, self.y, -1.5)

    def begin_go_checkpoint_context(self, *, checkpoint_id: str, checkpoint_xyz):
        self.takeover_calls.append((checkpoint_id, checkpoint_xyz))
        self._active_command_name = "go_checkpoint"
        self._active_setpoint = (float(checkpoint_xyz[0]), float(checkpoint_xyz[1]), -1.5, self.yaw)
        self._active_target_source = "go_checkpoint_takeover"


def _build_controller(drone: _DroneKinematics) -> LLMController:
    controller = LLMController.__new__(LLMController)
    controller.drone = drone
    controller._benchmark_executed_gc_sequence = []
    controller.set_benchmark_progress_focus_checkpoint = lambda _cp: None
    controller._maybe_run_agent_heartbeat = lambda: False
    controller._should_trigger_auto_replan = lambda _p, source=None: False

    def _snapshot():
        return {
            "drone_gt": (drone.x, drone.y, 0.0),
            "drone_est": (drone.x, drone.y, 0.0),
            "drone_est_bias_corrected": (drone.x, drone.y, 0.0),
            "drone_yaw_rad": drone.yaw,
            "benchmark_progress": {"completed": [], "current_target": None, "in_radius": False},
            "safety_context": None,
        }

    controller.get_live_ui_snapshot = _snapshot
    return controller


def test_gc_aligns_heading_before_forward_in_px4_sim(monkeypatch):
    monkeypatch.setattr(llm_mod, "Px4SimRobotWrapper", _FakePx4Base)

    cp = BENCHMARK_CHECKPOINTS_BY_ID["A1"]
    drone = _FakePx4Drone(float(cp.x) - 1.0, float(cp.y), math.pi / 2.0)
    controller = _build_controller(drone)

    summary, should_replan = controller.skill_go_checkpoint("A1")

    assert should_replan is False
    assert "reached" in summary

    first_forward_idx = next(i for i, (name, _) in enumerate(drone.actions) if name == "move_forward")
    assert first_forward_idx > 0
    assert all(name in {"turn_ccw", "turn_cw"} for name, _ in drone.actions[:first_forward_idx])
    assert not any(name in {"move_left", "move_right"} for name, _ in drone.actions)
    assert drone.takeover_calls and drone.takeover_calls[0][0] == "A1"
    assert drone.get_active_setpoint_snapshot()["target_source"] == "go_checkpoint_takeover"


def test_px4_takeover_overrides_scenario_reposition_hold(monkeypatch):
    from controller.px4_sim_robot_wrapper import Px4SimRobotWrapper

    wrapper = Px4SimRobotWrapper(enable_video=False)
    wrapper._active_command_name = "scenario_reposition"
    wrapper._active_setpoint = (1.0, 1.0, -1.5, 0.0)
    wrapper._setpoint_stream_active = True
    wrapper._active_target_source = "scenario_reposition_hold"
    wrapper._get_state = lambda: ((1.0, 1.0, -1.5), 0.1)

    wrapper.begin_go_checkpoint_context(checkpoint_id="C1", checkpoint_xyz=(1.6, 4.5, 0.0))
    snapshot = wrapper.get_active_setpoint_snapshot()

    assert snapshot["command"] == "go_checkpoint"
    assert snapshot["target_source"] == "go_checkpoint_takeover"
    assert snapshot["target"][0:3] == (1.6, 4.5, 0.0)
    assert snapshot["target"][0:2] != (1.0, 1.0)


def test_px4_publish_uses_takeover_source_after_gc_handoff(monkeypatch):
    from controller.px4_sim_robot_wrapper import Px4SimRobotWrapper

    logs = []

    def _capture_log(*args, **kwargs):
        logs.append(" ".join(str(v) for v in args))

    monkeypatch.setattr("controller.px4_sim_robot_wrapper.print_debug", _capture_log)

    class _DummyOffboardControlMode:
        def __init__(self):
            self.timestamp = 0
            self.position = False
            self.velocity = False
            self.acceleration = False
            self.attitude = False
            self.body_rate = False

    class _DummyTrajectorySetpoint:
        def __init__(self):
            self.timestamp = 0
            self.position = []
            self.yaw = 0.0

    class _DummyPub:
        def publish(self, _msg):
            return None

    wrapper = Px4SimRobotWrapper(enable_video=False)
    wrapper._msg_OffboardControlMode = _DummyOffboardControlMode
    wrapper._msg_TrajectorySetpoint = _DummyTrajectorySetpoint
    wrapper._pub_offboard_mode = _DummyPub()
    wrapper._pub_traj_sp = _DummyPub()
    wrapper._active_command_name = "scenario_reposition"
    wrapper._active_setpoint = (1.0, 1.0, -1.5, 0.0)
    wrapper._active_target_source = "scenario_reposition_hold"
    wrapper._setpoint_stream_active = True
    wrapper._get_state = lambda: ((1.0, 1.0, -1.5), 0.0)

    wrapper.begin_go_checkpoint_context(checkpoint_id="C1", checkpoint_xyz=(1.6, 4.5, 0.0))
    tx, ty, tz, tyaw = wrapper.get_active_setpoint_snapshot()["target"]
    wrapper._publish_offboard_setpoint(tx, ty, tz, yaw=tyaw)

    px4_sp_logs = [line for line in logs if "[PX4-SP]" in line]
    assert px4_sp_logs
    assert "command=go_checkpoint" in px4_sp_logs[-1]
    assert "source=go_checkpoint_takeover" in px4_sp_logs[-1]
    assert "scenario_reposition_hold" not in px4_sp_logs[-1]
    assert "target=(1.60, 4.50, 0.00)" in px4_sp_logs[-1]
