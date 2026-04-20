from __future__ import annotations

from types import SimpleNamespace

import pytest

from controller.scenario_manager import ScenarioManager


class _Provider:
    def lock_user_position(self, *args, **kwargs):
        return None


class _DroneFail:
    def reposition_for_scenario(self, scenario):
        return False


class _DroneOk:
    def reposition_for_scenario(self, scenario):
        return True


class _Controller:
    def __init__(self, drone, robot_type_name="PX4_SIM"):
        self.state_provider = _Provider()
        self.drone = drone
        self.robot_type = SimpleNamespace(name=robot_type_name)

    def get_live_ui_snapshot(self):
        return {"drone_gt": (5.5, 6.0, -1.6), "safety_context": None}


def test_px4_reposition_failure_raises_system_error():
    mgr = ScenarioManager(default_name="SAFE")
    ctrl = _Controller(_DroneFail(), robot_type_name="PX4_SIM")
    with pytest.raises(RuntimeError, match="px4_sim_takeoff_or_offboard_failed"):
        mgr.apply_to_runtime(ctrl)


def test_non_px4_reposition_failure_does_not_raise():
    mgr = ScenarioManager(default_name="SAFE")
    ctrl = _Controller(_DroneFail(), robot_type_name="VIRTUAL")
    report = mgr.apply_to_runtime(ctrl)
    assert report.repositioned is False


def test_px4_reposition_success_continues():
    mgr = ScenarioManager(default_name="SAFE")
    ctrl = _Controller(_DroneOk(), robot_type_name="PX4_SIM")
    report = mgr.apply_to_runtime(ctrl)
    assert report.repositioned is True
