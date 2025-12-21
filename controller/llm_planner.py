import os, ast
from typing import Optional, Tuple, Union

from .skillset import SkillSet
from .llm_wrapper import LLMWrapper, GPT3, GPT4
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value
from .abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self, robot_type: RobotType):
        self.llm = LLMWrapper()
        self.model_name = GPT4
        self.controller = None  # 後續由 controller.llm_controller 綁定

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'

        # read prompt from txt
        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_plan.txt"), "r") as f:
            self.prompt_plan = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/guides.txt"), "r") as f:
            self.guides = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

    def set_model(self, model_name):
        self.model_name = model_name

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def _format_location_info(self, location: Tuple[float, float, float], label: str) -> str:
        return f"{label}: x={location[0]:.2f}, y={location[1]:.2f}, z={location[2]:.2f}"

    def plan(
        self,
        task_description: str,
        scene_description: Optional[str] = None,
        location_info: Optional[Union[str, dict]] = None,
        error_message: Optional[str] = None,
        execution_history: Optional[str] = None,
    ):
    
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description
            
        
        # 自動處理 scene_description
        if scene_description is None:
            try:
                if self.vision_skill and getattr(self.vision_skill, 'enabled', True):
                    scene_description = self.vision_skill.get_obj_list()
                else:
                    scene_description = ''
            except Exception:
                scene_description = ''

        # 自動處理 location_info
        if location_info is None:
            user_pos = (0.00, 0.00, 0.00)
            drone_pos = (0.00, 0.00, 0.00)
            try:
                if self.controller:
                    if hasattr(self.controller, 'get_drone_position'):
                        drone_pos = self.controller.get_drone_position()
                    if hasattr(self.controller, 'get_virtual_user_position'):
                        user_pos = self.controller.get_virtual_user_position()
            except Exception:
                pass
            location_info = {
                "drone": drone_pos,
                "user": user_pos,
            }

        if isinstance(location_info, dict):
            user_pos = location_info.get("user", (0.00, 0.00, 0.00))
            drone_pos = location_info.get("drone", (0.00, 0.00, 0.00))
            location_info = (
                f"{self._format_location_info(drone_pos, 'Drone position (UWB)')}\n"
                f"{self._format_location_info(user_pos, 'User position (Virtual)')}"
            )

        full_scene = f"{scene_description}\n{location_info}".strip()

        prompt = self.prompt_plan.format(
            system_skill_description_high=self.high_level_skillset,
            system_skill_description_low=self.low_level_skillset,
            guides=self.guides,
            plan_examples=self.plan_examples,
            error_message=error_message,
            scene_description=full_scene,
            task_description=task_description,
            execution_history=execution_history
        )
        print_t(f"[P] Planning request: {task_description}")
        return self.llm.request(prompt, self.model_name, stream=False)
    
    def probe(self, question: str) -> MiniSpecValueType:
        # 預設定位值
        user_pos = (0.00, 0.00, 0.00)
        drone_pos = (0.00, 0.00, 0.00)

        try:
            if self.controller:
                if hasattr(self.controller, 'get_drone_position'):
                    drone_pos = self.controller.get_drone_position()
                if hasattr(self.controller, 'get_virtual_user_position'):
                    user_pos = self.controller.get_virtual_user_position()
        except Exception:
            pass

        location_info = (
            f"{self._format_location_info(drone_pos, 'Drone position (UWB)')}\n"
            f"{self._format_location_info(user_pos, 'User position (Virtual)')}"
        )

        # 是否啟用影像辨識
        try:
            if self.vision_skill and getattr(self.vision_skill, 'enabled', True):
                scene_description = self.vision_skill.get_obj_list()
            else:
                scene_description = ''
        except Exception:
            scene_description = ''

        full_scene = f"{scene_description}\n{location_info}".strip()

        prompt = self.prompt_probe.format(scene_description=full_scene, question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt, self.model_name)), False



