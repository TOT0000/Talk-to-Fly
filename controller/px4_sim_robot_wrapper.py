from typing import Tuple

from .virtual_robot_wrapper import VirtualRobotWrapper


class Px4SimRobotWrapper(VirtualRobotWrapper):
    """PX4 simulator robot wrapper backed by SimStateProvider cache."""

    def __init__(self, enable_video: bool = False):
        super().__init__(enable_video=enable_video)
        self._state_provider = None

    def set_state_provider(self, state_provider):
        self._state_provider = state_provider

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
