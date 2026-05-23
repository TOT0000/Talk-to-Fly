from types import SimpleNamespace

from controller.llm_controller import LLMController


def _make_controller_stub():
    c = LLMController.__new__(LLMController)
    c.manual_obstacle_selection_id = "obstacle_1"
    c.manual_obstacle_poses = {
        "obstacle_1": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_rad": 0.0},
        "obstacle_2": {"x": 1.0, "y": 1.0, "z": 0.0, "yaw_rad": 0.0},
        "obstacle_3": {"x": 2.0, "y": 2.0, "z": 0.0, "yaw_rad": 0.0},
    }
    c.get_baseline_scene = lambda: SimpleNamespace(id="SCENE_MANUAL_OBSTACLE_CONTROL")
    return c


def test_set_manual_obstacle_selection_no_recursion():
    c = _make_controller_stub()
    assert c.set_manual_obstacle_selection("obstacle_2") == "obstacle_2"


def test_move_selected_obstacle_relative_no_recursion():
    c = _make_controller_stub()
    out = c.move_selected_obstacle_relative(local_forward=1.0, local_right=0.0, step_m=0.5)
    assert out is not None
    assert out["obstacle_id"] == "obstacle_1"


def test_turn_selected_obstacle_no_recursion():
    c = _make_controller_stub()
    out = c.turn_selected_obstacle(15)
    assert out is not None
    assert out["obstacle_id"] == "obstacle_1"


def test_reset_system_state_does_not_have_self_recursive_wrappers():
    src = open("controller/llm_controller.py", "r", encoding="utf-8").read()
    assert "return self.set_manual_obstacle_selection(obstacle_id)" not in src
    assert "return self.move_selected_obstacle_relative(local_forward=local_forward, local_right=local_right, step_m=step_m)" not in src
    assert "return self.turn_selected_obstacle(delta_deg)" not in src
