from .virtual_robot_wrapper import VirtualRobotWrapper


class Px4SimRobotWrapper(VirtualRobotWrapper):
    """PX4 simulator robot wrapper.

    For now it reuses the virtual robot behavior, but keeps a dedicated type
    for PX4_SIM initialization and future simulator-specific extensions.
    """

    pass
