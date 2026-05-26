import math
import os
import threading
import time
from typing import Optional, Tuple

from .virtual_robot_wrapper import VirtualRobotWrapper
from .utils import print_debug, print_t


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
        self._nav_state_offboard = 14
        self._arming_state_armed = 2
        self._state_provider = None
        self._rclpy = None
        self._context = None
        self._executor = None
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
        self._publish_log_interval_s = 1.0
        self._control_log_interval_s = 0.5
        self._last_setpoint_log_ts = 0.0
        self._last_control_log_ts = 0.0
        self._last_logged_setpoint: Optional[Tuple[float, float, float, Optional[float]]] = None
        self._last_logged_command: Optional[str] = None
        self._last_logged_source: Optional[str] = None
        self._last_offboard_publish_ts: Optional[float] = None
        self._last_traj_publish_ts: Optional[float] = None
        self._active_command_name: Optional[str] = None
        self._active_command_value: Optional[float] = None
        self._active_command_start_time: Optional[float] = None
        self._active_target_source: Optional[str] = None
        self._active_command_last_writer: Optional[str] = None
        self._active_target_source_last_writer: Optional[str] = None
        self._active_setpoint_last_writer: Optional[str] = None
        # Offboard warmup dominates per-command latency in px4_sim mode.
        # Keep it tunable and skip warmup entirely once already OFFBOARD+ARMED.
        self._offboard_warmup_s = max(0.0, float(os.getenv("TYPEFLY_PX4_OFFBOARD_WARMUP_S", "0.25")))
        self._offboard_confirm_timeout_s = max(0.1, float(os.getenv("TYPEFLY_PX4_OFFBOARD_CONFIRM_TIMEOUT_S", "1.0")))
        self._offboard_max_attempts = max(1, int(os.getenv("TYPEFLY_PX4_OFFBOARD_MAX_ATTEMPTS", "2")))
        self._offboard_stream_stable_duration_s = max(0.5, float(os.getenv("TYPEFLY_PX4_OFFBOARD_STABLE_S", "2.0")))
        self._offboard_stream_start_ts: Optional[float] = None
        self._startup_completed = False
        self._manual_recover_required = False

    def set_state_provider(self, state_provider):
        self._state_provider = state_provider

    def _ensure_ros_publishers(self) -> bool:
        if self._node is not None:
            if self._setpoint_thread is None or not self._setpoint_thread.is_alive():
                print("[PX4-THREAD] setpoint thread not alive; restarting setpoint loop")
                self._offboard_stream_start_ts = None
                self._last_offboard_publish_ts = None
                self._last_traj_publish_ts = None
                self._start_setpoint_loop()
            return True
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor
            from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        except ImportError as exc:
            print(f"[WARN] Px4SimRobotWrapper ROS2/PX4 unavailable: {exc}")
            return False

        self._rclpy = rclpy
        self._context = self._rclpy.context.Context()
        self._context.init(args=None)
        self._node = Node("px4_sim_robot_wrapper", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
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
                if self._node is None or self._context is None or (not self._context.ok()) or self._executor is None:
                    break
                self._spin_once()
                with self._target_lock:
                    target = self._active_setpoint if self._setpoint_stream_active else None
                if target is not None:
                    tx, ty, tz, tyaw = target
                    self._publish_offboard_setpoint(tx, ty, tz, yaw=tyaw)
                else:
                    # Keep offboard stream alive by holding current position when no explicit target yet.
                    x, y, z = self.get_drone_position()
                    self._publish_offboard_setpoint(x, y, z, yaw=self.get_drone_yaw())
                if self._offboard_stream_start_ts is None:
                    self._offboard_stream_start_ts = time.time()
                time.sleep(0.05)  # 20 Hz offboard stream
            self._setpoint_thread_running = False

        self._setpoint_thread = threading.Thread(target=_loop, daemon=True)
        self._setpoint_thread.start()

    def _stop_setpoint_loop(self):
        self._setpoint_thread_running = False
        if self._setpoint_thread and self._setpoint_thread.is_alive():
            self._setpoint_thread.join(timeout=1.0)
        self._setpoint_thread = None

    def _set_active_target(
        self,
        x: float,
        y: float,
        z: float,
        yaw: Optional[float],
        *,
        source: Optional[str] = None,
        writer: str = "unknown",
    ):
        with self._target_lock:
            old_target = self._active_setpoint
            old_command = self._active_command_name
            old_source = self._active_target_source
            self._active_setpoint = (float(x), float(y), float(z), None if yaw is None else float(yaw))
            self._active_setpoint_last_writer = str(writer)
            self._setpoint_stream_active = True
            if source is not None:
                self._active_target_source = str(source)
                self._active_target_source_last_writer = str(writer)
            print_debug(
                "[PX4-TARGET] "
                f"source={self._active_target_source or 'unspecified'} "
                f"old_command={old_command or 'None'} new_command={self._active_command_name or 'None'} "
                f"old_source={old_source or 'None'} "
                f"old_target={old_target} new_target={self._active_setpoint} "
                f"setpoint_writer={self._active_setpoint_last_writer or 'unknown'} "
                f"source_writer={self._active_target_source_last_writer or 'unknown'}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )

    def _clear_active_target(self):
        with self._target_lock:
            self._setpoint_stream_active = False
            self._active_target_source = None
            self._active_target_source_last_writer = "_clear_active_target"

    def _now_us(self) -> int:
        return int(time.time() * 1_000_000)

    def _spin_once(self):
        if (
            self._rclpy is not None
            and self._node is not None
            and self._context is not None
            and self._context.ok()
            and self._executor is not None
            and self._setpoint_thread_running
        ):
            try:
                self._executor.spin_once(timeout_sec=0.0)
            except Exception as exc:
                print(f"[PX4-THREAD][ERROR] spin_once exception: {exc}")
                print("[PX4-THREAD][ERROR] setpoint loop exited")
                self._setpoint_thread_running = False

    def shutdown(self):
        self._clear_active_target()
        self._stop_setpoint_loop()
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:
                pass
            self._executor = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._pub_offboard_mode = None
        self._pub_traj_sp = None
        self._pub_vehicle_cmd = None
        if self._context is not None:
            try:
                if self._context.ok():
                    self._context.shutdown()
            except Exception:
                pass
        self._context = None
        self._rclpy = None

    def has_active_runtime(self) -> bool:
        return bool(
            (self._setpoint_thread is not None and self._setpoint_thread.is_alive())
            or self._executor is not None
            or self._node is not None
            or self._context is not None
        )

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
        self._log_vehicle_command_debug(command=command, source="_publish_vehicle_command", timestamp_us=msg.timestamp)
        print_debug(
            f"[PX4-CMD] vehicle_command={int(command)} param1={param1:.2f} "
            f"param2={param2:.2f} param7={param7:.2f} timestamp_us={msg.timestamp}"
        )

    def _command_name(self, command: int) -> str:
        names = {176: "VEHICLE_CMD_DO_SET_MODE", 400: "VEHICLE_CMD_COMPONENT_ARM_DISARM", 21: "VEHICLE_CMD_NAV_LAND", 20: "VEHICLE_CMD_NAV_RETURN_TO_LAUNCH", 22: "VEHICLE_CMD_NAV_TAKEOFF"}
        return names.get(int(command), f"CMD_{int(command)}")

    def _is_failsafe_active(self) -> bool:
        if self._state_provider is not None and hasattr(self._state_provider, "is_failsafe"):
            return bool(self._state_provider.is_failsafe())
        return False

    def _preflight_checks_pass(self) -> bool:
        if self._state_provider is not None and hasattr(self._state_provider, "pre_flight_checks_pass"):
            return bool(self._state_provider.pre_flight_checks_pass())
        return True

    def _has_local_position(self) -> bool:
        if self._state_provider is not None and hasattr(self._state_provider, "has_valid_position"):
            return bool(self._state_provider.has_valid_position())
        return True

    def _is_nav_state_blocked_for_rearm(self) -> bool:
        return int(self.get_navigation_state()) in {4, 5, 12}

    def _offboard_stream_stable(self) -> bool:
        if self._offboard_stream_start_ts is None:
            return False
        now_ts = time.time()
        if (now_ts - self._offboard_stream_start_ts) < self._offboard_stream_stable_duration_s:
            return False
        if self._last_offboard_publish_ts is None or self._last_traj_publish_ts is None:
            return False
        return (now_ts - self._last_offboard_publish_ts) <= 0.2 and (now_ts - self._last_traj_publish_ts) <= 0.2

    def _log_vehicle_command_debug(self, command: int, source: str, timestamp_us: int):
        print_debug(
            "[PX4-CMD-DEBUG] "
            f"command={int(command)} command_name={self._command_name(command)} source={source} timestamp_us={int(timestamp_us)} "
            f"nav_state={self.get_navigation_state()} arming_state={self.get_arming_state()} "
            f"failsafe={self._is_failsafe_active()} pre_flight_checks_pass={self._preflight_checks_pass()}"
        )

    def _is_offboard_ready(self) -> bool:
        return (
            int(self.get_navigation_state()) == int(self._nav_state_offboard)
            and int(self.get_arming_state()) == int(self._arming_state_armed)
        )

    def _wait_for_offboard_ready(self, timeout_s: float = 2.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._is_offboard_ready():
                return True
            time.sleep(0.05)
        return self._is_offboard_ready()

    def _ensure_offboard_control(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        warmup_s: Optional[float] = None,
        confirm_timeout_s: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ) -> bool:
        cmd_mode = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_DO_SET_MODE", 176)
        cmd_arm = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_COMPONENT_ARM_DISARM", 400)
        if warmup_s is None:
            warmup_s = self._offboard_warmup_s
        if confirm_timeout_s is None:
            confirm_timeout_s = self._offboard_confirm_timeout_s
        if max_attempts is None:
            max_attempts = self._offboard_max_attempts

        self._set_active_target(x, y, z, yaw, writer="_ensure_offboard_control")
        if self._startup_completed and self._is_offboard_ready():
            return True
        if self._manual_recover_required:
            print_debug("[PX4-OFFBOARD] manual recovery required; auto OFFBOARD/ARM blocked")
            return False

        if self._is_failsafe_active() or self._is_nav_state_blocked_for_rearm():
            if self._state_provider is not None and hasattr(self._state_provider, "debug_log_px4_failure_snapshot"):
                self._state_provider.debug_log_px4_failure_snapshot(reason="offboard_arm_gate_blocked")
            self._manual_recover_required = True
            return False

        if not self._preflight_checks_pass() or not self._has_local_position() or not self._offboard_stream_stable():
            print_debug(
                "[PX4-OFFBOARD] arm gate blocked "
                f"preflight={self._preflight_checks_pass()} local_position={self._has_local_position()} stream_stable={self._offboard_stream_stable()}"
            )
            return False

        if int(self.get_arming_state()) == int(self._arming_state_armed):
            if int(self.get_navigation_state()) != int(self._nav_state_offboard):
                self._publish_vehicle_command(cmd_mode, param1=1.0, param2=6.0)
                if self._wait_for_offboard_ready(timeout_s=confirm_timeout_s):
                    self._startup_completed = True
                    return True
            else:
                self._startup_completed = True
                return True

        for attempt in range(1, max_attempts + 1):
            time.sleep(warmup_s)
            if self._is_failsafe_active():
                self._manual_recover_required = True
                print_debug("[PX4-OFFBOARD] failsafe during offboard attempt; block auto re-arm")
                return False
            if (
                not self._preflight_checks_pass()
                or not self._has_local_position()
                or not self._offboard_stream_stable()
                or self._is_nav_state_blocked_for_rearm()
            ):
                print_debug("[PX4-OFFBOARD] command 176 blocked by gate re-check")
                return False
            self._publish_vehicle_command(cmd_mode, param1=1.0, param2=6.0)
            time.sleep(0.1)
            if self._is_failsafe_active():
                self._manual_recover_required = True
                print_debug("[PX4-OFFBOARD] failsafe after mode command; block auto re-arm")
                return False
            if (
                self._preflight_checks_pass()
                and self._has_local_position()
                and self._offboard_stream_stable()
                and (not self._is_nav_state_blocked_for_rearm())
                and int(self.get_navigation_state()) == int(self._nav_state_offboard)
                and int(self.get_arming_state()) != int(self._arming_state_armed)
            ):
                self._publish_vehicle_command(cmd_arm, param1=1.0)
            if self._wait_for_offboard_ready(timeout_s=confirm_timeout_s):
                self._startup_completed = True
                return True
        return False

    def _watchdog_publish_timing(self, stream_name: str, now_ts: float, last_ts: Optional[float]):
        if last_ts is None:
            return
        dt = now_ts - float(last_ts)
        if dt > 0.5:
            print(f"[PX4-WATCHDOG][ERROR] {stream_name}_publish_gap={dt:.3f}s active_function={self._active_command_name or 'hold'}")
        elif dt > 0.2:
            print_debug(f"[PX4-WATCHDOG][WARN] {stream_name}_publish_gap={dt:.3f}s active_function={self._active_command_name or 'hold'}")

    def _publish_offboard_setpoint(self, x: float, y: float, z: float, yaw: Optional[float] = None):
        mode = self._msg_OffboardControlMode()
        mode.timestamp = self._now_us()
        mode_publish_ts = time.time()
        self._watchdog_publish_timing("offboard_control_mode", mode_publish_ts, self._last_offboard_publish_ts)
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        self._pub_offboard_mode.publish(mode)
        self._last_offboard_publish_ts = mode_publish_ts
        print_debug(f"[PX4-PUB] offboard_control_mode ts={mode_publish_ts:.6f}", env_var="TYPEFLY_VERBOSE_DEBUG")

        sp = self._msg_TrajectorySetpoint()
        sp.timestamp = self._now_us()
        sp.position = [float(x), float(y), float(z)]
        if yaw is not None:
            sp.yaw = float(yaw)
        traj_publish_ts = time.time()
        self._watchdog_publish_timing("trajectory_setpoint", traj_publish_ts, self._last_traj_publish_ts)
        self._pub_traj_sp.publish(sp)
        self._last_traj_publish_ts = traj_publish_ts
        print_debug(f"[PX4-PUB] trajectory_setpoint ts={traj_publish_ts:.6f}", env_var="TYPEFLY_VERBOSE_DEBUG")
        publish_ts = traj_publish_ts
        current_setpoint = (float(x), float(y), float(z), None if yaw is None else float(yaw))
        current_command = self._active_command_name or "hold"
        current_source = self._active_target_source or "unspecified"
        previous_setpoint = self._last_logged_setpoint
        target_delta = None
        if previous_setpoint is not None:
            target_delta = math.sqrt(
                (current_setpoint[0] - previous_setpoint[0]) ** 2
                + (current_setpoint[1] - previous_setpoint[1]) ** 2
                + (current_setpoint[2] - previous_setpoint[2]) ** 2
            )
        target_changed = (
            previous_setpoint is None
            or (target_delta is not None and target_delta >= 0.05)
            or previous_setpoint[3] != current_setpoint[3]
        )
        command_changed = self._last_logged_command != current_command
        source_changed = self._last_logged_source != current_source
        verbose_mode = bool(str(os.getenv("TYPEFLY_VERBOSE_DEBUG", "0")).strip() not in {"", "0", "false", "False", "FALSE"})
        should_log = bool(command_changed or source_changed or target_changed)
        if verbose_mode and (publish_ts - self._last_setpoint_log_ts) >= self._publish_log_interval_s:
            should_log = True
        if should_log:
            self._last_logged_setpoint = current_setpoint
            self._last_logged_command = current_command
            self._last_logged_source = current_source
            self._last_setpoint_log_ts = publish_ts
            message = (
                f"[PX4-SP] command={current_command} "
                f"target=({x:.2f}, {y:.2f}, {z:.2f}) yaw="
                f"{'None' if yaw is None else f'{yaw:.3f}'} "
                f"source={current_source} "
                f"command_writer={self._active_command_last_writer or 'unknown'} "
                f"source_writer={self._active_target_source_last_writer or 'unknown'} "
                f"setpoint_writer={self._active_setpoint_last_writer or 'unknown'} "
                f"log_reason={'change' if (command_changed or source_changed or target_changed) else 'verbose_interval'} "
                f"publish_ts={publish_ts:.3f}"
            )
            if command_changed or source_changed or target_changed:
                print_t(message)
            else:
                print_debug(message, env_var="TYPEFLY_VERBOSE_DEBUG")

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

    def _format_position(self, position: Tuple[float, float, float]) -> str:
        return f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"

    def _begin_motion_debug(self, skill_name: str, command_value: float):
        old_command = self._active_command_name
        self._active_command_name = skill_name
        self._active_command_last_writer = "_begin_motion_debug"
        self._active_command_value = float(command_value)
        self._active_command_start_time = time.time()
        print_debug(
            f"[PX4-MOVE] skill={skill_name} command_value={float(command_value):.2f}m "
            f"start_time={self._active_command_start_time:.3f} "
            f"old_command={old_command or 'None'} new_command={self._active_command_name} "
            f"command_writer={self._active_command_last_writer}"
        )

    def _begin_rotation_debug(self, skill_name: str, command_value_deg: float):
        old_command = self._active_command_name
        self._active_command_name = skill_name
        self._active_command_last_writer = "_begin_rotation_debug"
        self._active_command_value = float(command_value_deg)
        self._active_command_start_time = time.time()
        print_debug(
            f"[PX4-MOVE] skill={skill_name} command_value={float(command_value_deg):.2f}deg "
            f"start_time={self._active_command_start_time:.3f} "
            f"old_command={old_command or 'None'} new_command={self._active_command_name} "
            f"command_writer={self._active_command_last_writer}"
        )

    def get_active_setpoint_snapshot(self) -> dict:
        with self._target_lock:
            target = self._active_setpoint
            return {
                "command": self._active_command_name,
                "target": target,
                "target_source": self._active_target_source,
                "command_writer": self._active_command_last_writer,
                "target_source_writer": self._active_target_source_last_writer,
                "setpoint_writer": self._active_setpoint_last_writer,
            }

    def begin_go_checkpoint_context(
        self,
        *,
        checkpoint_id: str,
        checkpoint_xyz: Tuple[float, float, float],
    ):
        old_snapshot = self.get_active_setpoint_snapshot()
        (x, y, z), _ = self._get_state()
        cp_x = float(checkpoint_xyz[0])
        cp_y = float(checkpoint_xyz[1])
        cp_z = float(checkpoint_xyz[2]) if len(tuple(checkpoint_xyz)) >= 3 else float(z)
        takeover_yaw = math.atan2(cp_y - float(y), cp_x - float(x))
        print_debug(
            "[GC-HANDOFF] "
            f"begin checkpoint={checkpoint_id} checkpoint_xyz={checkpoint_xyz} "
            f"current_position=({x:.3f}, {y:.3f}, {z:.3f}) "
            f"old_command={old_snapshot.get('command')} old_target={old_snapshot.get('target')} "
            f"old_source={old_snapshot.get('target_source')} "
            f"old_command_writer={old_snapshot.get('command_writer')} "
            f"old_source_writer={old_snapshot.get('target_source_writer')} "
            f"old_setpoint_writer={old_snapshot.get('setpoint_writer')}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        self._active_command_name = "go_checkpoint"
        self._active_command_last_writer = "begin_go_checkpoint_context"
        self._active_command_value = None
        self._active_command_start_time = time.time()
        self._set_active_target(
            cp_x,
            cp_y,
            cp_z,
            takeover_yaw,
            source="go_checkpoint_takeover",
            writer="begin_go_checkpoint_context",
        )
        new_snapshot = self.get_active_setpoint_snapshot()
        handoff_ok = (
            str(new_snapshot.get("command") or "") == "go_checkpoint"
            and str(new_snapshot.get("target_source") or "") == "go_checkpoint_takeover"
        )
        print_debug(
            f"{'[GC-HANDOFF]' if handoff_ok else '[GC-HANDOFF-FAIL]'} "
            f"handoff checkpoint={checkpoint_id} new_command={new_snapshot.get('command')} "
            f"new_target={new_snapshot.get('target')} source={new_snapshot.get('target_source')} "
            f"command_writer={new_snapshot.get('command_writer')} "
            f"source_writer={new_snapshot.get('target_source_writer')} "
            f"setpoint_writer={new_snapshot.get('setpoint_writer')}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )

    def _log_tracking_state(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        scalar_error: float,
        target_yaw: Optional[float] = None,
        yaw_error: Optional[float] = None,
        force: bool = False,
    ):
        now = time.time()
        if not force and (now - self._last_control_log_ts) < self._control_log_interval_s:
            return
        self._last_control_log_ts = now
        gt_position = self.get_ground_truth_drone_position()
        nav_state = self.get_navigation_state()
        arming_state = self.get_arming_state()
        message = (
            f"[PX4-STATE] command={self._active_command_name or 'hold'} "
            f"gt_position={self._format_position(gt_position)} "
            f"target={self._format_position((target_x, target_y, target_z))} "
            f"position_error={scalar_error:.3f}m nav_state={nav_state} arming_state={arming_state}"
        )
        if target_yaw is not None:
            message += f" target_yaw={target_yaw:.3f}"
        if yaw_error is not None:
            message += f" yaw_error={yaw_error:.3f}rad"
        print_debug(message)

    def _move_to_local_target(self, target_x: float, target_y: float, target_z: float, yaw: float,
                              timeout_s: float = 5.0, pos_tol: float = 0.25) -> Tuple[bool, bool]:
        self._set_active_target(
            target_x,
            target_y,
            target_z,
            yaw,
            source=self._active_command_name or "move_to_local_target",
            writer="_move_to_local_target",
        )
        deadline = time.time() + timeout_s
        stable_since = None
        settle_s = 0.3
        best_error = float("inf")
        last_progress_ts = time.time()
        print_debug(
            f"[PX4-MOVE] target_setpoint={self._format_position((target_x, target_y, target_z))} "
            f"yaw={yaw:.3f} completion=position_error<{pos_tol:.2f}m for {settle_s:.2f}s timeout={timeout_s:.2f}s"
        )
        while time.time() < deadline:
            cx, cy, cz = self.get_drone_position()
            err = math.sqrt((target_x - cx) ** 2 + (target_y - cy) ** 2 + (target_z - cz) ** 2)
            self._log_tracking_state(target_x, target_y, target_z, err)
            if err + 0.05 < best_error:
                best_error = err
                last_progress_ts = time.time()
            elif err > pos_tol and (time.time() - last_progress_ts) < 1.0:
                deadline = max(deadline, time.time() + 0.75)
            if err < pos_tol:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= settle_s:
                    self._log_tracking_state(target_x, target_y, target_z, err, force=True)
                    print_debug(
                        f"[PX4-MOVE] completed command={self._active_command_name or 'move'} "
                        f"criterion=position_error<{pos_tol:.2f}m for {settle_s:.2f}s"
                    )
                    return True, False
            else:
                stable_since = None
            time.sleep(0.05)
        final_pos = self.get_ground_truth_drone_position()
        self._log_tracking_state(target_x, target_y, target_z, err, force=True)
        print_debug(
            f"[PX4-MOVE] timeout command={self._active_command_name or 'move'} "
            f"final_gt_position={self._format_position(final_pos)} target={self._format_position((target_x, target_y, target_z))} "
            f"criterion=position_error<{pos_tol:.2f}m for {settle_s:.2f}s"
        )
        return False, False


    def _ensure_airborne_for_translation(self) -> bool:
        x, y, z = self.get_ground_truth_drone_position()
        if z <= -0.35:
            return True
        print_debug(
            f"[PX4-MOVE] auto_takeoff_before_translation current_gt={self._format_position((x, y, z))}"
        )
        return bool(self.takeoff())

    def _move_by_body_offset(self, skill_name: str, command_distance: float, forward_m: float = 0.0, right_m: float = 0.0, up_m: float = 0.0,
                             timeout_scale: float = 6.0) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=2.0):
                print_debug(
                    f"[PX4-MOVE] command={skill_name} wait_for_position_failed; "
                    f"active_command={self._active_command_name or 'None'} active_target={self._active_setpoint}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
                return False, False
        self._begin_motion_debug(skill_name, command_distance)

        if (abs(forward_m) > 1e-6 or abs(right_m) > 1e-6) and not self._ensure_airborne_for_translation():
            print_debug(f"[PX4-MOVE] abort command={skill_name} reason=auto_takeoff_failed")
            return False, False

        (x, y, z), yaw = self._get_state()
        if not self._ensure_offboard_control(x, y, z, yaw):
            print_debug(f"[PX4-MOVE] abort command={skill_name} reason=offboard_not_ready")
            return False, False

        # Body-to-world mapping aligned to UI XY convention:
        # +forward moves toward heading; +right moves to drone starboard side.
        dx = forward_m * math.cos(yaw) + right_m * math.sin(yaw)
        dy = forward_m * math.sin(yaw) - right_m * math.cos(yaw)
        dz = -up_m  # up in NED means z decreases

        target_x = x + dx
        target_y = y + dy
        target_z = z + dz
        timeout_s = max(3.0, (abs(forward_m) + abs(right_m) + abs(up_m)) * timeout_scale)
        return self._move_to_local_target(target_x, target_y, target_z, yaw=yaw, timeout_s=timeout_s)

    def _rotate_by(self, skill_name: str, delta_yaw_rad: float, command_degrees: float, timeout_s: float = 4.0, yaw_tol: float = 0.12) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=2.0):
                print_debug(
                    f"[PX4-MOVE] command={skill_name} wait_for_position_failed; "
                    f"active_command={self._active_command_name or 'None'} active_target={self._active_setpoint}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
                return False, False
        self._begin_rotation_debug(skill_name, command_degrees)

        (x, y, z), yaw = self._get_state()
        if not self._ensure_offboard_control(x, y, z, yaw):
            print_debug(f"[PX4-MOVE] abort command={skill_name} reason=offboard_not_ready")
            return False, False
        target_yaw = self._normalize_angle(yaw + delta_yaw_rad)
        self._set_active_target(
            x,
            y,
            z,
            target_yaw,
            source=self._active_command_name or "rotate",
            writer="_rotate_by",
        )
        print_debug(
            f"[PX4-MOVE] target_setpoint={self._format_position((x, y, z))} "
            f"yaw={target_yaw:.3f} completion=yaw_error<{yaw_tol:.2f}rad for 0.30s timeout={timeout_s:.2f}s"
        )

        stable_since = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            current_yaw = self.get_drone_yaw()
            err = abs(self._normalize_angle(target_yaw - current_yaw))
            self._log_tracking_state(x, y, z, 0.0, target_yaw=target_yaw, yaw_error=err)
            if err <= yaw_tol:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= 0.3:
                    self._log_tracking_state(x, y, z, 0.0, target_yaw=target_yaw, yaw_error=err, force=True)
                    print_debug(
                        f"[PX4-MOVE] completed command={self._active_command_name or 'rotate'} "
                        f"criterion=yaw_error<{yaw_tol:.2f}rad for 0.30s"
                    )
                    return True, False
            else:
                stable_since = None
            time.sleep(0.05)
        self._log_tracking_state(x, y, z, 0.0, target_yaw=target_yaw, yaw_error=err, force=True)
        print_debug(
            f"[PX4-MOVE] timeout command={self._active_command_name or 'rotate'} "
            f"target_yaw={target_yaw:.3f} criterion=yaw_error<{yaw_tol:.2f}rad for 0.30s"
        )
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

    def get_latest_user_localization_packet(self):
        if self._state_provider is not None and hasattr(self._state_provider, "get_latest_received_user_packet"):
            return self._state_provider.get_latest_received_user_packet()
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
        self._begin_motion_debug("takeoff", 1.0)

        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=3.0):
                print("[PX4_SIM] takeoff aborted: no valid local position")
                return False

        (x, y, z), yaw = self._get_state()

        # 1) Warm-up offboard stream, switch to offboard mode, and arm.
        if not self._ensure_offboard_control(x, y, z, yaw):
            print("[PX4_SIM] takeoff aborted: failed to enter offboard+armed state")
            return False

        # 2) Climb by setting higher target in local NED (z more negative = up).
        takeoff_height_m = 1.0
        z_tolerance_m = 0.15
        settle_time_s = 1.0
        target_z = z - takeoff_height_m
        self._set_active_target(x, y, target_z, yaw, source="takeoff_hold", writer="takeoff")
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

    def hold_position(self):
        pos = self.get_ground_truth_drone_position()
        yaw = self.get_drone_yaw()
        self._active_command_name = "llm_wait_hover"
        self._active_command_last_writer = "hold_position"
        self._active_command_value = 0.0
        self._active_command_start_time = time.time()
        self._set_active_target(float(pos[0]), float(pos[1]), float(pos[2]), float(yaw), source="llm_wait_hover", writer="hold_position")
        print_debug(f"[GC-LLM-WAIT] holding position=({float(pos[0]):.3f},{float(pos[1]):.3f},{float(pos[2]):.3f})", env_var="TYPEFLY_VERBOSE_DEBUG")
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def move_forward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_forward", float(distance), forward_m=float(distance))

    def move_backward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_backward", float(distance), forward_m=-float(distance))

    def move_left(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_left", float(distance), right_m=-float(distance))

    def move_right(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_right", float(distance), right_m=float(distance))

    def move_up(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_up", float(distance), up_m=float(distance))

    def move_down(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_down", float(distance), up_m=-float(distance))

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        return self._rotate_by("turn_ccw", math.radians(float(degree)), float(degree))

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        return self._rotate_by("turn_cw", -math.radians(float(degree)), float(degree))

    def land(self):
        if not self._ensure_ros_publishers():
            return
        # Stop offboard setpoint stream before LAND command to avoid fighting auto-land.
        self._clear_active_target()
        cmd_land = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_NAV_LAND", 21)
        self._publish_vehicle_command(cmd_land)

    def reposition_for_scenario(self, scenario) -> bool:
        if not self._ensure_ros_publishers():
            return False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=3.0):
                return False

        # Support both ScenarioConfig fields (`drone_position_3d`, `drone_yaw_rad`)
        # and BaselineScene fields (`drone_initial_pose`, `drone_initial_yaw_rad`).
        target_pose = getattr(scenario, "drone_position_3d", None)
        if target_pose is None:
            target_pose = getattr(scenario, "drone_initial_pose", None)
        if target_pose is None:
            raise AttributeError(
                "scenario must define drone_position_3d or drone_initial_pose"
            )

        target_yaw = getattr(scenario, "drone_yaw_rad", None)
        if target_yaw is None:
            target_yaw = getattr(scenario, "drone_initial_yaw_rad", 0.0)

        tx, ty, tz = [float(v) for v in target_pose]
        target_yaw = float(target_yaw)
        (x, y, z), yaw = self._get_state()

        # Auto takeoff intentionally disabled during scenario init to prevent implicit ARM loops.

        if not self._ensure_offboard_control(x, y, z, yaw):
            return False

        self._begin_motion_debug("scenario_reposition", math.dist((x, y, z), (tx, ty, tz)))
        ok, _ = self._move_to_local_target(tx, ty, tz, yaw=target_yaw, timeout_s=15.0, pos_tol=0.35)
        if not ok:
            # Retry once with looser tolerance to handle simulator transient convergence issues.
            ok, _ = self._move_to_local_target(tx, ty, tz, yaw=target_yaw, timeout_s=8.0, pos_tol=0.50)
        if not ok:
            print_debug(
                f"[PX4-SCENARIO] reposition timeout target={self._format_position((tx, ty, tz))}"
            )
            return False
        self._set_active_target(
            tx,
            ty,
            tz,
            target_yaw,
            source="scenario_reposition_hold",
            writer="reposition_for_scenario",
        )
        print_t(
            f"[PX4-SCENARIO] repositioned to target={self._format_position((tx, ty, tz))} yaw={target_yaw:.2f}"
        )
        return True

    def stop_stream(self):
        super().stop_stream()

    def keep_active(self):
        # Streaming is handled by the internal setpoint thread.
        pass
