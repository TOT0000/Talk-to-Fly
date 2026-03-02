import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

from .abs.robot_wrapper import RobotWrapper
from .uwb_wrapper import UWBWrapper


class StateProvider(ABC):
    def __init__(self):
        self._callback: Optional[Callable[[Tuple[float, float, float, float]], None]] = None

    def register_callback(self, callback: Callable[[Tuple[float, float, float, float]], None]):
        self._callback = callback

    @abstractmethod
    def get_user_position(self) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def has_valid_position(self) -> bool:
        pass

    @abstractmethod
    def get_user_yaw(self) -> float:
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class UwbStateProvider(StateProvider):
    def __init__(self, uwb: Optional[UWBWrapper] = None):
        super().__init__()
        self.uwb = uwb or UWBWrapper()

    def register_callback(self, callback: Callable[[Tuple[float, float, float, float]], None]):
        super().register_callback(callback)
        self.uwb.register_callback(callback)

    def get_user_position(self) -> Tuple[float, float, float]:
        return self.uwb.get_user_position()

    def has_valid_position(self) -> bool:
        return self.uwb.latest_position != (0.00, 0.00, 0.00)

    def get_user_yaw(self) -> float:
        return 0.0

    def start(self):
        self.uwb.start_with_retry()

    def stop(self):
        self.uwb.stop()


class SimStateProvider(StateProvider):
    def __init__(self, robot: RobotWrapper):
        super().__init__()
        self.robot = robot
        self._active = False

    def get_user_position(self) -> Tuple[float, float, float]:
        return self.robot.get_drone_position()

    def has_valid_position(self) -> bool:
        return True

    def get_user_yaw(self) -> float:
        get_yaw = getattr(self.robot, "get_drone_yaw", None)
        if callable(get_yaw):
            return float(get_yaw())
        return 0.0

    def start(self):
        self._active = True

        def loop():
            while self._active:
                if self._callback:
                    x, y, z = self.get_user_position()
                    self._callback((time.time(), x, y, z))
                time.sleep(0.1)

        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._active = False
