import math
import time
from typing import Optional, Tuple

from .state_provider import SimStateProvider
from .virtual_robot_wrapper import VirtualRobotWrapper


class Px4SimRobotWrapper(VirtualRobotWrapper):
    """Robot wrapper for PX4 simulator offboard movement control.

    This wrapper keeps the same low-level skill interface as other wrappers,
    while translating high-level movement commands into local ENU/NED position
    setpoints and yaw setpoints for a mission executor style adapter.
    """

    YAW_THRESHOLD_RAD = math.radians(3.0)
    POSITION_SETTLE_SEC = 0.15
    TURN_LOOP_SLEEP_SEC = 0.1
    TURN_TIMEOUT_SEC = 5.0

    def __init__(
        self,
        enable_video: bool = False,
        mission_executor_adapter: Optional[object] = None,
        local_frame: str = "ENU",
    ):
        super().__init__(enable_video=enable_video)
        self.adapter = mission_executor_adapter
        self.local_frame = local_frame.upper()

        self.movement_x_accumulator = 0.0
        self.movement_y_accumulator = 0.0
        self.movement_z_accumulator = 0.0
        self.rotation_accumulator = 0.0

        self._target_yaw = 0.0
        self._state_provider = SimStateProvider(self)

    def connect(self):
        if self.adapter and hasattr(self.adapter, "connect"):
            self.adapter.connect()

    def keep_active(self):
        self._publish_offboard_mode()

    def takeoff(self) -> bool:
        if self.adapter and hasattr(self.adapter, "arm_and_takeoff"):
            ok = bool(self.adapter.arm_and_takeoff())
            return ok
        return True

    def land(self):
        if self.adapter and hasattr(self.adapter, "land"):
            self.adapter.land()

    def _normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def get_drone_yaw(self) -> float:
        return self._normalize_angle(self.rotation_accumulator)

    def _get_current_pose(self) -> Tuple[float, float, float, float]:
        x, y, z = self._state_provider.get_user_position()
        yaw = self._state_provider.get_user_yaw()
        return float(x), float(y), float(z), float(yaw)

    def _publish_offboard_mode(self):
        if self.adapter and hasattr(self.adapter, "publish_offboard_control_mode"):
            self.adapter.publish_offboard_control_mode(position=True, velocity=False, attitude=False)

    def _publish_position_setpoint(self, x: float, y: float, z: float, yaw: float):
        if not self.adapter:
            return
        if hasattr(self.adapter, "publish_position_setpoint"):
            self.adapter.publish_position_setpoint(x=x, y=y, z=z, yaw=yaw, frame=self.local_frame)

    def _compute_relative_target(
        self,
        forward: float = 0.0,
        left: float = 0.0,
        up: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        x, y, z, yaw = self._get_current_pose()

        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        dx_enu = forward * cos_yaw - left * sin_yaw
        dy_enu = forward * sin_yaw + left * cos_yaw
        dz_enu = up

        if self.local_frame == "NED":
            target_x = x + dy_enu
            target_y = y + dx_enu
            target_z = z - dz_enu
        else:
            target_x = x + dx_enu
            target_y = y + dy_enu
            target_z = z + dz_enu

        return target_x, target_y, target_z, yaw

    def _move_relative(self, forward: float = 0.0, left: float = 0.0, up: float = 0.0) -> Tuple[bool, bool]:
        target_x, target_y, target_z, yaw = self._compute_relative_target(forward=forward, left=left, up=up)

        self._publish_offboard_mode()
        self._publish_position_setpoint(target_x, target_y, target_z, yaw)

        self.movement_x_accumulator = target_x
        self.movement_y_accumulator = target_y
        self.movement_z_accumulator = target_z

        time.sleep(self.POSITION_SETTLE_SEC)
        return True, False

    def move_forward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_relative(forward=distance)

    def move_backward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_relative(forward=-distance)

    def move_left(self, distance: float) -> Tuple[bool, bool]:
        return self._move_relative(left=distance)

    def move_right(self, distance: float) -> Tuple[bool, bool]:
        return self._move_relative(left=-distance)

    def move_up(self, distance: float) -> Tuple[bool, bool]:
        return self._move_relative(up=distance)

    def move_down(self, distance: float) -> Tuple[bool, bool]:
        return self._move_relative(up=-distance)

    def _turn_to_target_yaw(self, target_yaw: float) -> Tuple[bool, bool]:
        start_time = time.time()

        while time.time() - start_time < self.TURN_TIMEOUT_SEC:
            _, _, _, current_yaw = self._get_current_pose()
            yaw_error = self._normalize_angle(target_yaw - current_yaw)

            self._publish_offboard_mode()
            x, y, z = self._state_provider.get_user_position()
            self._publish_position_setpoint(float(x), float(y), float(z), target_yaw)

            if abs(yaw_error) < self.YAW_THRESHOLD_RAD:
                self.rotation_accumulator = target_yaw
                return True, False

            if not self.adapter:
                self.rotation_accumulator = self._normalize_angle(current_yaw + yaw_error * 0.5)

            time.sleep(self.TURN_LOOP_SLEEP_SEC)

        return False, True

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        _, _, _, current_yaw = self._get_current_pose()
        delta = math.radians(float(degree))
        self._target_yaw = self._normalize_angle(current_yaw + delta)
        return self._turn_to_target_yaw(self._target_yaw)

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        _, _, _, current_yaw = self._get_current_pose()
        delta = math.radians(float(degree))
        self._target_yaw = self._normalize_angle(current_yaw - delta)
        return self._turn_to_target_yaw(self._target_yaw)

    def get_drone_position(self) -> Tuple[float, float, float]:
        return (
            round(self.movement_x_accumulator, 3),
            round(self.movement_y_accumulator, 3),
            round(self.movement_z_accumulator, 3),
        )
