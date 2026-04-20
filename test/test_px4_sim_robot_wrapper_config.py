from __future__ import annotations

import sys
import types

# Allow importing wrappers in environments without OpenCV installed.
sys.modules.setdefault("cv2", types.SimpleNamespace())

from controller.px4_sim_robot_wrapper import Px4SimRobotWrapper


def test_px4_offboard_defaults_are_more_robust(monkeypatch):
    monkeypatch.delenv("TYPEFLY_PX4_OFFBOARD_WARMUP_S", raising=False)
    monkeypatch.delenv("TYPEFLY_PX4_OFFBOARD_CONFIRM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TYPEFLY_PX4_OFFBOARD_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("TYPEFLY_PX4_POSITION_READY_TIMEOUT_S", raising=False)

    wrapper = Px4SimRobotWrapper(enable_video=False)

    assert wrapper._offboard_warmup_s == 0.35
    assert wrapper._offboard_confirm_timeout_s == 1.5
    assert wrapper._offboard_max_attempts == 6
    assert wrapper._position_ready_timeout_s == 8.0


def test_px4_offboard_retry_uses_force_arm_in_late_attempts(monkeypatch):
    wrapper = Px4SimRobotWrapper(enable_video=False)
    calls = []
    wrapper._msg_VehicleCommand = types.SimpleNamespace(VEHICLE_CMD_DO_SET_MODE=176, VEHICLE_CMD_COMPONENT_ARM_DISARM=400)

    monkeypatch.setattr("controller.px4_sim_robot_wrapper.time.sleep", lambda *_: None)
    monkeypatch.setattr(wrapper, "_set_active_target", lambda *a, **k: None)
    monkeypatch.setattr(wrapper, "_is_offboard_ready", lambda: False)
    monkeypatch.setattr(wrapper, "_wait_for_offboard_ready", lambda timeout_s: False)
    monkeypatch.setattr(wrapper, "get_navigation_state", lambda: wrapper._nav_state_offboard)
    monkeypatch.setattr(wrapper, "get_arming_state", lambda: 1)

    def _record(command, param1=0.0, param2=0.0, param7=0.0):
        calls.append((int(command), float(param1), float(param2)))

    monkeypatch.setattr(wrapper, "_publish_vehicle_command", _record)

    ok = wrapper._ensure_offboard_control(0.0, 0.0, 0.0, 0.0, warmup_s=0.0, confirm_timeout_s=0.0, max_attempts=6)
    assert ok is False
    arm_calls = [c for c in calls if c[0] == 400]
    assert len(arm_calls) == 6
    # late attempts should include force-arm param2=21196
    assert any(abs(c[2] - 21196.0) < 1e-6 for c in arm_calls[-3:])
