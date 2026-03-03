import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Type

from .state_provider import StateProvider


@dataclass
class _SimStateCache:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    nav_state: int = 0
    arming_state: int = 0


class _SharedRos2Context:
    """Process-wide ROS2 context guard with reference counting."""

    _lock = threading.Lock()
    _ref_count = 0
    _rclpy = None

    @classmethod
    def acquire(cls):
        try:
            import rclpy
        except ImportError:
            return None

        with cls._lock:
            if not rclpy.ok():
                rclpy.init(args=None)
            cls._ref_count += 1
            cls._rclpy = rclpy
            return cls._rclpy

    @classmethod
    def release(cls, rclpy_module):
        if rclpy_module is None:
            return

        with cls._lock:
            if cls._ref_count > 0:
                cls._ref_count -= 1

            if cls._ref_count == 0 and rclpy_module.ok():
                rclpy_module.shutdown()
                cls._rclpy = None


class SimStateProvider(StateProvider):
    """State provider for PX4 SITL via ROS2 topics.

    Subscribes to:
    - /fmu/out/vehicle_local_position_v1
    - /fmu/out/vehicle_status_v2
    """

    def __init__(self, fixed_user_position: Optional[Tuple[float, float, float]] = None):
        super().__init__()
        self._lock = threading.Lock()
        self._cache = _SimStateCache()
        self._active = False
        self._spin_thread: Optional[threading.Thread] = None
        self._node = None
        self._rclpy = None
        self._context = None
        self._executor = None

        # TODO(px4-sim): user position is currently fixed/env-driven, not dynamic from simulator topics.
        self._fixed_user_position = fixed_user_position or self._load_user_position_from_env()
        self._user_position: Tuple[float, float, float] = self._fixed_user_position
        self._last_position_ts: float = 0.0
        self._ros_ready: bool = False
        self._user_position_topic: str = os.getenv("SIM_USER_POSITION_TOPIC", "/sim/user_position")
        self._user_position_msg_type_name: str = os.getenv("SIM_USER_POSITION_MSG_TYPE", "PointStamped")

    def _load_user_position_from_env(self) -> Tuple[float, float, float]:
        raw = os.getenv("SIM_USER_POSITION", "0,0,0")
        try:
            x, y, z = [float(v.strip()) for v in raw.split(",")]
            return (x, y, z)
        except Exception:
            return (0.0, 0.0, 0.0)

    def _on_vehicle_local_position(self, msg):
        position = (float(msg.x), float(msg.y), float(msg.z))
        velocity = (float(msg.vx), float(msg.vy), float(msg.vz))
        yaw = float(getattr(msg, "heading", 0.0))

        with self._lock:
            self._cache.position = position
            self._cache.velocity = velocity
            self._cache.yaw = yaw
            self._last_position_ts = time.time()

        if self._callback:
            self._callback((time.time(), position[0], position[1], position[2]))

    def _on_vehicle_status(self, msg):
        with self._lock:
            self._cache.nav_state = int(getattr(msg, "nav_state", 0))
            self._cache.arming_state = int(getattr(msg, "arming_state", 0))

    def _resolve_user_position_msg_type(self, point_cls, point_stamped_cls) -> Optional[Type]:
        """Resolve a single ROS2 message type for /sim/user_position subscription."""
        choice = (self._user_position_msg_type_name or "PointStamped").strip().lower()
        if choice in {"pointstamped", "point_stamped", "stamped"}:
            return point_stamped_cls
        if choice in {"point", "geometry_msgs/point"}:
            return point_cls
        return point_stamped_cls

    def _on_user_position(self, msg):
        # Support the configured single subscription type (Point or PointStamped).
        point = getattr(msg, "point", msg)
        x = float(getattr(point, "x", 0.0))
        y = float(getattr(point, "y", 0.0))
        z = float(getattr(point, "z", 0.0))
        with self._lock:
            self._user_position = (x, y, z)

    def get_user_position(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._user_position

    def get_user_yaw(self) -> float:
        """Fallback user yaw for simulation.

        PX4_SIM currently has no subscribed user orientation source,
        so we intentionally return a fixed placeholder yaw (0.0 rad).
        """
        return 0.0

    def has_valid_position(self) -> bool:
        with self._lock:
            # PX4 local position (0,0,0) 可能是合法原點，故以時間戳判定是否曾收過資料
            return (time.time() - self._last_position_ts) < 1.0 and self._last_position_ts > 0.0

    def get_drone_position(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._cache.position

    def get_drone_velocity(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._cache.velocity

    def get_drone_yaw(self) -> float:
        with self._lock:
            return self._cache.yaw

    def get_navigation_state(self) -> int:
        with self._lock:
            return self._cache.nav_state

    def get_arming_state(self) -> int:
        with self._lock:
            return self._cache.arming_state

    def start(self):
        if self._active:
            return

        rclpy = None
        Node = None
        SingleThreadedExecutor = None
        VehicleLocalPosition = None
        VehicleStatus = None
        Point = None
        PointStamped = None
        qos_profile_sensor_data = None
        try:
            import rclpy as _rclpy
            from rclpy.node import Node as _Node
            from rclpy.executors import SingleThreadedExecutor as _SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data as _qos_profile_sensor_data
            from px4_msgs.msg import VehicleLocalPosition as _VehicleLocalPosition, VehicleStatus as _VehicleStatus
            try:
                from geometry_msgs.msg import Point as _Point, PointStamped as _PointStamped
            except ImportError:
                _Point = None
                _PointStamped = None
            rclpy = _rclpy
            Node = _Node
            SingleThreadedExecutor = _SingleThreadedExecutor
            qos_profile_sensor_data = _qos_profile_sensor_data
            VehicleLocalPosition = _VehicleLocalPosition
            VehicleStatus = _VehicleStatus
            Point = _Point
            PointStamped = _PointStamped
        except ImportError as exc:
            print(f"[WARN] SimStateProvider disabled (ROS2/PX4 messages unavailable): {exc}")
            self._active = True
            self._ros_ready = False
            return

        if rclpy is None or Node is None or SingleThreadedExecutor is None or VehicleLocalPosition is None or VehicleStatus is None:
            print("[WARN] SimStateProvider disabled: ROS2 symbols not initialized")
            self._active = True
            self._ros_ready = False
            return

        self._rclpy = rclpy
        # Use an isolated context so provider startup is robust even when
        # other components initialize/shutdown the default global context.
        self._context = self._rclpy.context.Context()
        self._context.init(args=None)
        self._node = Node("sim_state_provider", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._ros_ready = True

        sensor_qos = qos_profile_sensor_data if qos_profile_sensor_data is not None else 10

        self._node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self._on_vehicle_local_position,
            sensor_qos,
        )
        self._node.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v2",
            self._on_vehicle_status,
            sensor_qos,
        )

        # User position topic uses exactly one configured message type.
        user_position_msg_type = self._resolve_user_position_msg_type(Point, PointStamped)
        if user_position_msg_type is not None:
            self._node.create_subscription(
                user_position_msg_type,
                self._user_position_topic,
                self._on_user_position,
                sensor_qos,
            )
        else:
            print(
                f"[WARN] SimStateProvider user-position subscription disabled: "
                f"message type '{self._user_position_msg_type_name}' unavailable"
            )

        self._active = True

        def _spin():
            while self._active and self._context is not None and self._context.ok():
                self._executor.spin_once(timeout_sec=0.1)

        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()

    def is_ros_ready(self) -> bool:
        return self._ros_ready

    def wait_for_position(self, timeout_s: float = 3.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.has_valid_position():
                return True
            time.sleep(0.05)
        return self.has_valid_position()

    def stop(self):
        self._active = False

        if self._rclpy is None:
            return

        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)

        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
            self._executor = None

        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._context is not None and self._context.ok():
            self._context.shutdown()
        self._context = None
        self._executor = None
        self._ros_ready = False
