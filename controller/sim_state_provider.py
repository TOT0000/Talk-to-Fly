import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .state_provider import StateProvider


@dataclass
class _SimStateCache:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    nav_state: int = 0
    arming_state: int = 0


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

        self._fixed_user_position = fixed_user_position or self._load_user_position_from_env()

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

        if self._callback:
            self._callback((time.time(), position[0], position[1], position[2]))

    def _on_vehicle_status(self, msg):
        with self._lock:
            self._cache.nav_state = int(getattr(msg, "nav_state", 0))
            self._cache.arming_state = int(getattr(msg, "arming_state", 0))

    def get_user_position(self) -> Tuple[float, float, float]:
        return self._fixed_user_position

    def has_valid_position(self) -> bool:
        return True

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

        try:
            import rclpy
            from rclpy.node import Node
            from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
        except ImportError as exc:
            print(f"[WARN] SimStateProvider disabled (ROS2/PX4 messages unavailable): {exc}")
            self._active = True
            return

        self._rclpy = rclpy
        self._rclpy.init(args=None)
        self._node = Node("sim_state_provider")

        self._node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self._on_vehicle_local_position,
            10,
        )
        self._node.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v2",
            self._on_vehicle_status,
            10,
        )

        self._active = True

        def _spin():
            while self._active and self._rclpy.ok():
                self._rclpy.spin_once(self._node, timeout_sec=0.1)

        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()

    def stop(self):
        self._active = False

        if self._rclpy is None:
            return

        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)

        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._rclpy.ok():
            self._rclpy.shutdown()
