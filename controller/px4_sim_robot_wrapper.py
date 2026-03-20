import math
import threading
import time
from typing import Optional, Tuple

from .virtual_robot_wrapper import VirtualRobotWrapper


class Px4SimRobotWrapper(VirtualRobotWrapper):
    """PX4 simulator robot wrapper backed by SimStateProvider cache.

    Minimal MVP:
    - get_drone_position() comes from SimStateProvider
    - takeoff() performs offboard arm + position-setpoint climb
    - move/turn skills update active setpoint targets in local frame
    - a background loop continuously publishes active offboard setpoints for stable hold

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

        self._target_lock = threading.Lock()
        self._active_setpoint: Optional[Tuple[float, float, float, Optional[float]]] = None
        self._setpoint_stream_active = False
        self._setpoint_thread: Optional[threading.Thread] = None
        self._setpoint_thread_running = False

    def set_state_provider(self, state_provider):
        self._state_provider = state_provider

    def _ensure_ros_publishers(self) -> bool:
        if self._node is not None:
            return True
        try:
            import rclpy
            from rclpy.node import Node
            from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        except ImportError as exc:
            print(f"[WARN] Px4SimRobotWrapper ROS2/PX4 unavailable: {exc}")
            return False

        self._rclpy = rclpy
        if not self._rclpy.ok():
            self._rclpy.init(args=None)

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

        self._start_setpoint_loop()
        return True

    def _start_setpoint_loop(self):
        if self._setpoint_thread is not None and self._setpoint_thread.is_alive():
            return

        self._setpoint_thread_running = True

        def _loop():
            while self._setpoint_thread_running:
                self._spin_once()
                with self._target_lock:
                    target = self._active_setpoint if self._setpoint_stream_active else None
                if target is not None:
                    tx, ty, tz, tyaw = target
                    self._publish_offboard_setpoint(tx, ty, tz, yaw=tyaw)
                time.sleep(0.05)  # 20 Hz offboard stream

        self._setpoint_thread = threading.Thread(target=_loop, daemon=True)
        self._setpoint_thread.start()

    def _stop_setpoint_loop(self):
        self._setpoint_thread_running = False
        if self._setpoint_thread and self._setpoint_thread.is_alive():
            self._setpoint_thread.join(timeout=1.0)
        self._setpoint_thread = None

    def _set_active_target(self, x: float, y: float, z: float, yaw: Optional[float]):
        with self._target_lock:
            self._active_setpoint = (float(x), float(y), float(z), None if yaw is None else float(yaw))
            self._setpoint_stream_active = True

    def _clear_active_target(self):
        with self._target_lock:
            self._setpoint_stream_active = False

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
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _move_to_local_target(self, target_x: float, target_y: float, target_z: float, yaw: float,
                              timeout_s: float = 5.0, pos_tol: float = 0.25) -> Tuple[bool, bool]:
        self._set_active_target(target_x, target_y, target_z, yaw)
        deadline = time.time() + timeout_s
        stable_since = None
        settle_s = 0.3
        while time.time() < deadline:
            cx, cy, cz = self.get_drone_position()
            err = math.sqrt((target_x - cx) ** 2 + (target_y - cy) ** 2 + (target_z - cz) ** 2)
            if err < pos_tol:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= settle_s:
                    return True, False
            else:
                stable_since = None
            time.sleep(0.05)
        return False, False

    def _move_by_body_offset(self, forward_m: float = 0.0, right_m: float = 0.0, up_m: float = 0.0,
                             timeout_scale: float = 3.0) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=2.0):
                return False, False

        (x, y, z), yaw = self._get_state()

        # Local NED frame assumption:
        # +X forward, +Y right, +Z down.
        dx = forward_m * math.cos(yaw) - right_m * math.sin(yaw)
        dy = forward_m * math.sin(yaw) + right_m * math.cos(yaw)
        dz = -up_m  # up in NED means z decreases

        target_x = x + dx
        target_y = y + dy
        target_z = z + dz
        timeout_s = max(3.0, (abs(forward_m) + abs(right_m) + abs(up_m)) * timeout_scale)
        return self._move_to_local_target(target_x, target_y, target_z, yaw=yaw, timeout_s=timeout_s)

    def _rotate_by(self, delta_yaw_rad: float, timeout_s: float = 4.0, yaw_tol: float = 0.12) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=2.0):
                return False, False

        (x, y, z), yaw = self._get_state()
        target_yaw = self._normalize_angle(yaw + delta_yaw_rad)
        self._set_active_target(x, y, z, target_yaw)

        stable_since = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            current_yaw = self.get_drone_yaw()
            err = abs(self._normalize_angle(target_yaw - current_yaw))
            if err <= yaw_tol:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= 0.3:
                    return True, False
            else:
                stable_since = None
            time.sleep(0.05)
        return False, False

    def get_drone_position(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_drone_position"):
            return self._state_provider.get_drone_position()
        return super().get_drone_position()

    def get_ground_truth_drone_position(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_ground_truth_drone_position"):
            return self._state_provider.get_ground_truth_drone_position()
        return self.get_drone_position()

    def get_estimated_drone_position(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_estimated_drone_position"):
            return self._state_provider.get_estimated_drone_position()
        return self.get_drone_position()

    def get_latest_localization_packet(self):
        if self._state_provider is not None and hasattr(self._state_provider, "get_latest_received_state_packet"):
            return self._state_provider.get_latest_received_state_packet()
        return None

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
        """Pure offboard takeoff using local position setpoints.

        Local frame assumption (PX4 local NED):
        - +X: forward, +Y: right, +Z: down
        - Ascend (go up) means Z becomes more negative.
        """
        if not self._ensure_ros_publishers():
            return False

        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=3.0):
                print("[PX4_SIM] takeoff aborted: no valid local position")
                return False

        (x, y, z), yaw = self._get_state()

        # 1) Warm-up offboard stream by holding current position.
        self._set_active_target(x, y, z, yaw)
        time.sleep(0.8)

        # 2) Switch to offboard mode (custom main mode = 6 in PX4).
        cmd_mode = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_DO_SET_MODE", 176)
        self._publish_vehicle_command(cmd_mode, param1=1.0, param2=6.0)

        # 3) Arm after mode switch.
        cmd_arm = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_COMPONENT_ARM_DISARM", 400)
        self._publish_vehicle_command(cmd_arm, param1=1.0)

        # 4) Climb by setting higher target in local NED (z more negative = up).
        takeoff_height_m = 1.0
        z_tolerance_m = 0.15
        settle_time_s = 1.0
        target_z = z - takeoff_height_m
        self._set_active_target(x, y, target_z, yaw)
        print(f"[PX4_SIM] takeoff start_z={z:.2f}, target_z={target_z:.2f} (NED)")

        stable_since = None
        deadline = time.time() + 10.0
        while time.time() < deadline:
            _, _, cz = self.get_drone_position()

            if abs(cz - target_z) <= z_tolerance_m:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= settle_time_s:
                    print(
                        f"[PX4_SIM] takeoff stabilized at target_z={target_z:.2f}, current_z={cz:.2f}; "
                        f"keep holding active setpoint"
                    )
                    return True
            else:
                stable_since = None

            time.sleep(0.05)

        _, _, final_z = self.get_drone_position()
        print(
            f"[PX4_SIM] takeoff timeout: target_z={target_z:.2f}, current_z={final_z:.2f}, "
            f"tolerance={z_tolerance_m:.2f}"
        )
        return False

    def move_forward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset(forward_m=float(distance))

    def move_backward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset(forward_m=-float(distance))

    def move_left(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset(right_m=-float(distance))

    def move_right(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset(right_m=float(distance))

    def move_up(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset(up_m=float(distance))

    def move_down(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset(up_m=-float(distance))

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        return self._rotate_by(math.radians(float(degree)))

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        return self._rotate_by(-math.radians(float(degree)))

    def land(self):
        if not self._ensure_ros_publishers():
            return
        # Stop offboard setpoint stream before LAND command to avoid fighting auto-land.
        self._clear_active_target()
        cmd_land = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_NAV_LAND", 21)
        self._publish_vehicle_command(cmd_land)

    def stop_stream(self):
        super().stop_stream()

    def keep_active(self):
        # Streaming is handled by the internal setpoint thread.
        pass
