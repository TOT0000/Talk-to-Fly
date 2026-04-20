from __future__ import annotations

import pytest

pytest.importorskip("cv2")

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
