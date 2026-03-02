import math
import time
from typing import Optional, Tuple

from .virtual_robot_wrapper import VirtualRobotWrapper
from .sim_state_provider import _SharedRos2Context


class Px4SimRobotWrapper(VirtualRobotWrapper):
    """PX4 simulator robot wrapper backed by SimStateProvider cache.

    Minimal MVP:
    - get_drone_position() comes from SimStateProvider
    - takeoff() sends PX4 vehicle commands (arm + takeoff)
    - move_forward(distance) sends offboard position setpoints on ROS2 topics

    TODO:
    - Replace ad-hoc publishing with dedicated mission_executor adapter/service API.
    - Add robust frame conversion (NED/ENU) based on simulator config.
    """

    def __init__(self, enable_video: bool = False):
        super().__init__(enable_video=enable_video)
        self._state_provider = None
        self._rclpy = None
        self._node = None
        self._pub_offboard_mode = None
        self._pub_traj_sp = None
        self._pub_vehicle_cmd = None

        self._offboard_counter = 0
        self._ros_context_acquired = False

    def set_state_provider(self, state_provider):
        self._state_provider = state_provider

    def _ensure_ros_publishers(self) -> bool:
        if self._node is not None:
            return True
        try:
            from rclpy.node import Node
            from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        except ImportError as exc:
            print(f"[WARN] Px4SimRobotWrapper ROS2/PX4 unavailable: {exc}")
            return False

        self._rclpy = _SharedRos2Context.acquire()
        if self._rclpy is None:
            print("[WARN] Px4SimRobotWrapper ROS2 context unavailable")
            return False

        self._ros_context_acquired = True
        self._node = Node("px4_sim_robot_wrapper")
        self._msg_OffboardControlMode = OffboardControlMode
        self._msg_TrajectorySetpoint = TrajectorySetpoint
        self._msg_VehicleCommand = VehicleCommand

        self._pub_offboard_mode = self._node.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )
        self._pub_traj_sp = self._node.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )
        self._pub_vehicle_cmd = self._node.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )
        return True

    def _now_us(self) -> int:
        return int(time.time() * 1_000_000)

    def _spin_once(self):
        if self._rclpy is not None and self._node is not None and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0, param7: float = 0.0):
        msg = self._msg_VehicleCommand()
        msg.timestamp = self._now_us()
        msg.command = int(command)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param7 = float(param7)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._pub_vehicle_cmd.publish(msg)

    def _publish_offboard_setpoint(self, x: float, y: float, z: float, yaw: Optional[float] = None):
        mode = self._msg_OffboardControlMode()
        mode.timestamp = self._now_us()
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        self._pub_offboard_mode.publish(mode)

        sp = self._msg_TrajectorySetpoint()
        sp.timestamp = self._now_us()
        sp.position = [float(x), float(y), float(z)]
        if yaw is not None:
            sp.yaw = float(yaw)
        self._pub_traj_sp.publish(sp)

    def _get_state(self) -> Tuple[Tuple[float, float, float], float]:
        pos = self.get_drone_position()
        yaw = self.get_drone_yaw()
        return pos, yaw

    def _normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _wait_for_state(self, timeout_s: float = 2.0) -> bool:
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            return bool(self._state_provider.wait_for_position(timeout_s=timeout_s))
        return True

    def _move_to_target(self, target_x: float, target_y: float, target_z: float, yaw: float, timeout_s: float) -> Tuple[bool, bool]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._publish_offboard_setpoint(target_x, target_y, target_z, yaw=yaw)
            self._spin_once()
            cx, cy, cz = self.get_drone_position()
            err = math.sqrt((target_x - cx) ** 2 + (target_y - cy) ** 2 + (target_z - cz) ** 2)
            if err < 0.25:
                return True, False
            time.sleep(0.05)
        return True, False

    def get_drone_position(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_drone_position"):
            return self._state_provider.get_drone_position()
        return super().get_drone_position()

    def get_drone_velocity(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_drone_velocity"):
            return self._state_provider.get_drone_velocity()
        return (0.0, 0.0, 0.0)

    def get_drone_yaw(self) -> float:
        if self._state_provider is not None and hasattr(self._state_provider, "get_drone_yaw"):
            return self._state_provider.get_drone_yaw()
        return 0.0

    def get_navigation_state(self) -> int:
        if self._state_provider is not None and hasattr(self._state_provider, "get_navigation_state"):
            return self._state_provider.get_navigation_state()
        return 0

    def get_arming_state(self) -> int:
        if self._state_provider is not None and hasattr(self._state_provider, "get_arming_state"):
            return self._state_provider.get_arming_state()
        return 0

    def connect(self):
        self._ensure_ros_publishers()

    def takeoff(self) -> bool:
        if not self._ensure_ros_publishers():
            return False

        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            self._state_provider.wait_for_position(timeout_s=3.0)

        # Arm
        cmd_arm = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_COMPONENT_ARM_DISARM", 400)
        self._publish_vehicle_command(cmd_arm, param1=1.0)

        # Pre-stream some offboard setpoints
        (x, y, z), yaw = self._get_state()
        for _ in range(12):
            self._publish_offboard_setpoint(x, y, z, yaw=yaw)
            self._spin_once()
            time.sleep(0.05)

        # Switch to offboard mode (custom main mode=6 in PX4)
        cmd_mode = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_DO_SET_MODE", 176)
        self._publish_vehicle_command(cmd_mode, param1=1.0, param2=6.0)

        # Send takeoff command (altitude in param7, sign depends on frame; keep minimal)
        cmd_takeoff = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_NAV_TAKEOFF", 22)
        target_z = z - 1.0
        self._publish_vehicle_command(cmd_takeoff, param7=target_z)

        # Keep publishing position hold slightly above current position
        deadline = time.time() + 4.0
        while time.time() < deadline:
            self._publish_offboard_setpoint(x, y, target_z, yaw=yaw)
            self._spin_once()
            time.sleep(0.05)

        return True

    def move_forward(self, distance: float) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if not self._wait_for_state(timeout_s=2.0):
            return False, False

        (x, y, z), yaw = self._get_state()
        dx = float(distance) * math.cos(yaw)
        dy = float(distance) * math.sin(yaw)
        target_x = x + dx
        target_y = y + dy
        target_z = z
        return self._move_to_target(target_x, target_y, target_z, yaw=yaw, timeout_s=max(3.0, abs(distance) * 3.0))

    def move_backward(self, distance: float) -> Tuple[bool, bool]:
        return self.move_forward(-float(distance))

    def move_left(self, distance: float) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if not self._wait_for_state(timeout_s=2.0):
            return False, False

        (x, y, z), yaw = self._get_state()
        side_yaw = yaw + (math.pi / 2.0)
        dx = float(distance) * math.cos(side_yaw)
        dy = float(distance) * math.sin(side_yaw)
        return self._move_to_target(x + dx, y + dy, z, yaw=yaw, timeout_s=max(3.0, abs(distance) * 3.0))

    def move_right(self, distance: float) -> Tuple[bool, bool]:
        return self.move_left(-float(distance))

    def move_up(self, distance: float) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if not self._wait_for_state(timeout_s=2.0):
            return False, False

        (x, y, z), yaw = self._get_state()
        target_z = z - float(distance)
        return self._move_to_target(x, y, target_z, yaw=yaw, timeout_s=max(3.0, abs(distance) * 3.0))

    def move_down(self, distance: float) -> Tuple[bool, bool]:
        return self.move_up(-float(distance))

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if not self._wait_for_state(timeout_s=2.0):
            return False, False

        (x, y, z), yaw = self._get_state()
        target_yaw = self._normalize_angle(yaw - math.radians(float(degree)))
        deadline = time.time() + max(2.0, abs(float(degree)) / 45.0)
        while time.time() < deadline:
            self._publish_offboard_setpoint(x, y, z, yaw=target_yaw)
            self._spin_once()
            cyaw = self.get_drone_yaw()
            yaw_err = abs(self._normalize_angle(target_yaw - cyaw))
            if yaw_err < math.radians(5.0):
                return True, False
            time.sleep(0.05)
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        return self.turn_cw(-int(degree))

    def land(self):
        if not self._ensure_ros_publishers():
            return
        cmd_land = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_NAV_LAND", 21)
        self._publish_vehicle_command(cmd_land)


    def close(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._ros_context_acquired:
            _SharedRos2Context.release(self._rclpy)
            self._ros_context_acquired = False

        self._rclpy = None

    def stop_stream(self):
        super().stop_stream()

    def keep_active(self):
        # Maintain offboard stream if needed in future.
        pass
