import os, ast
from typing import Optional

from .safety_context import SafetyContext
from .skillset import SkillSet
from .llm_wrapper import LLMWrapper, GPT3, GPT4, chat_log_path
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_debug, print_t
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
        self.prompt_plan_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_plan.txt")
        self.prompt_probe_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_probe.txt")
        self.guides_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/guides.txt")
        self.plan_examples_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/plan_examples.txt")
        with open(self.prompt_plan_path, "r") as f:
            self.prompt_plan = f.read()

        with open(self.prompt_probe_path, "r") as f:
            self.prompt_probe = f.read()

        with open(self.guides_path, "r") as f:
            self.guides = f.read()

        with open(self.plan_examples_path, "r") as f:
            self.plan_examples = f.read()

    def set_model(self, model_name):
        self.model_name = model_name

    def init(self, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def plan(self, task_description: str, scene_description: Optional[str] = None, location_info: Optional[str] = None, error_message: Optional[str] = None, execution_history: Optional[str] = None, safety_context: Optional[SafetyContext] = None):
    
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
            try:
                if self.controller and hasattr(self.controller, '_format_planner_location_info'):
                    location_info = self.controller._format_planner_location_info()
            except Exception:
                location_info = None
            if location_info is None:
                user_pos = (0.00, 0.00, 0.00)
                drone_pos = (0.00, 0.00, 0.00)
                try:
                    if self.controller:
                        if hasattr(self.controller, 'state_provider'):
                            get_est_drone = getattr(self.controller.state_provider, 'get_estimated_drone_position', None)
                            get_est_user = getattr(self.controller.state_provider, 'get_estimated_user_position', None)
                            if callable(get_est_drone):
                                value = get_est_drone()
                                if value is not None:
                                    drone_pos = value
                            if callable(get_est_user):
                                value = get_est_user()
                                if value is not None:
                                    user_pos = value
                except Exception:
                    pass
                location_info = (
                    f"Drone estimated position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}\n"
                    f"User estimated position: x={user_pos[0]:.2f}, y={user_pos[1]:.2f}, z={user_pos[2]:.2f}"
                )

        full_scene = f"{scene_description}\n{location_info}".strip()
        safety_context_block = (
            safety_context.to_prompt_block()
            if safety_context is not None
            else (
                "current_collision_probability: 0.000000\nhistorical_max_collision_probability: 0.000000\nsafety_score: 0.000\nreason_tags: ['safety_context_unavailable']\n"
                "dominant_threat_type: user\ndominant_threat_id: user\ndominant_gap_m: 0.00\ndominant_uncertainty_scale_m: 1.00\ndominant_freshness_s: unknown\n"
                "drone_to_user_distance_xy: 0.00\nenvelope_gap_m: 0.00\nuncertainty_scale_m: 1.00\nenvelopes_overlap: False\n"
                "latest_generation_timestamp: unknown\nlatest_receive_timestamp: unknown\ntiming_freshness_s: unknown\nmax_aoi_s: unknown\n"
                "TaskPoints:\n- (n/a)\nCandidateTargets:\n- (n/a)\nObstacles:\n- (n/a)\nPathSummaries:\n- (n/a)"
            )
        )

        prompt = self.prompt_plan.format(
            system_skill_description_low=self.low_level_skillset,
            guides=self.guides,
            plan_examples=self.plan_examples,
            error_message=error_message,
            scene_description=full_scene,
            task_description=task_description,
            execution_history=execution_history,
            safety_context=safety_context_block,
        )
        dump_prompt = str(os.getenv("TYPEFLY_DUMP_LLM_PROMPT", "1")).strip().lower() not in {"0", "false", "no"}
        if dump_prompt:
            dump_path = os.getenv("TYPEFLY_LAST_PROMPT_PATH", os.path.join(CURRENT_DIR, "..", "logs", "last_llm_prompt.txt"))
            try:
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                with open(dump_path, "w") as f:
                    f.write(prompt)
                print_debug(f"[P-PROMPT-DUMP] wrote final prompt to {dump_path}")
            except Exception as exc:
                print_debug(f"[P-PROMPT-DUMP] failed to write prompt: {exc}")
        print_t(f"[P] Planning request: {task_description}")
        print_debug(
            f"[P-PROMPT-PATHS] prompt_plan={self.prompt_plan_path} guides={self.guides_path} examples={self.plan_examples_path}"
        )
        print_debug(f"[P-SAFETY-CONTEXT]\n{safety_context_block}")
        print_debug(f"[P] Full prompt debug log: {chat_log_path}")
        return self.llm.request(prompt, self.model_name, stream=False)
    
    def probe(self, question: str) -> MiniSpecValueType:
        location_info = None
        try:
            if self.controller and hasattr(self.controller, '_format_planner_location_info'):
                location_info = self.controller._format_planner_location_info()
        except Exception:
            location_info = None

        if location_info is None:
            person_pos = (0.00, 0.00, 0.00)
            drone_pos = (0.00, 0.00, 0.00)

            try:
                if self.controller and hasattr(self.controller, 'state_provider'):
                    get_est_drone = getattr(self.controller.state_provider, 'get_estimated_drone_position', None)
                    get_est_user = getattr(self.controller.state_provider, 'get_estimated_user_position', None)
                    if callable(get_est_drone):
                        value = get_est_drone()
                        if value is not None:
                            drone_pos = value
                    if callable(get_est_user):
                        value = get_est_user()
                        if value is not None:
                            person_pos = value
            except Exception:
                pass

            location_info = (
                f"Drone estimated position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}\n"
                f"User estimated position: x={person_pos[0]:.2f}, y={person_pos[1]:.2f}, z={person_pos[2]:.2f}"
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
