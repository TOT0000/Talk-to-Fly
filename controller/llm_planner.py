import os, ast, re
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

    def _fmt_xyz(self, value) -> str:
        if value is None:
            return "(n/a)"
        try:
            x, y, z = value
            return f"({float(x):.2f}, {float(y):.2f}, {float(z):.2f})"
        except Exception:
            return "(n/a)"

    def _build_runtime_context_block(self, safety_context: Optional[SafetyContext]) -> str:
        snapshot = {}
        if self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot"):
            try:
                snapshot = self.controller.get_live_ui_snapshot() or {}
            except Exception:
                snapshot = {}

        drone_pos = snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est") or snapshot.get("drone_gt")
        workers = list(snapshot.get("workers") or [])
        workers_sorted = sorted(workers, key=lambda row: str(row.get("id", "")))
        worker_lines = []
        for idx in range(3):
            label = f"worker_{idx + 1}"
            if idx < len(workers_sorted):
                row = workers_sorted[idx]
                est_xy = row.get("est_xy_bias_corrected") or row.get("est_xy_raw")
                if est_xy is None:
                    worker_lines.append(f"- {label} bias-corrected estimated position: (n/a)")
                else:
                    worker_lines.append(f"- {label} bias-corrected estimated position: ({float(est_xy[0]):.2f}, {float(est_xy[1]):.2f}, 0.00)")
            else:
                worker_lines.append(f"- {label} bias-corrected estimated position: (n/a)")

        current_collision_probability = 0.0 if safety_context is None else float(safety_context.current_collision_probability)
        dominant_worker = "n/a"
        if safety_context is not None:
            dominant_worker = str(getattr(safety_context, "dominant_threat_id", "n/a") or "n/a")

        objective = dict(snapshot.get("active_objective_set") or {})
        active_zone_ids = [str(v) for v in objective.get("active_zone_ids", [])]
        active_checkpoint_ids = [str(v) for v in objective.get("active_checkpoint_ids", [])]
        checkpoint_map = {
            str(row.get("id")): row
            for row in (snapshot.get("benchmark_checkpoints") or [])
            if row.get("id") is not None
        }
        checkpoint_lines = []
        for cid in active_checkpoint_ids:
            row = checkpoint_map.get(cid)
            if row is None:
                checkpoint_lines.append(f"- {cid}: (x=n/a, y=n/a)")
            else:
                checkpoint_lines.append(f"- {cid}: (x={float(row.get('x')):.2f}, y={float(row.get('y')):.2f})")
        if not checkpoint_lines:
            checkpoint_lines.append("- (n/a)")

        return (
            "UAV state:\n"
            f"- UAV bias-corrected estimated position: {self._fmt_xyz(drone_pos)}\n"
            "Workers state:\n"
            + "\n".join(worker_lines)
            + "\n"
            f"- dominant risky worker: {dominant_worker}\n"
            f"- current collision probability: {current_collision_probability:.6f}\n"
            "Mission objective:\n"
            f"- active zones: {active_zone_ids if active_zone_ids else ['(n/a)']}\n"
            f"- active checkpoints: {active_checkpoint_ids if active_checkpoint_ids else ['(n/a)']}\n"
            "Active checkpoint coordinates (x, y):\n"
            + "\n".join(checkpoint_lines)
        )

    def _extract_completed_checkpoints_from_history(self, execution_history) -> list[str]:
        if execution_history is None:
            return []
        if isinstance(execution_history, list):
            history_text = ";".join(str(v) for v in execution_history)
        else:
            history_text = str(execution_history)
        found = re.findall(r"(?:gc|go_checkpoint)\(\s*['\"]?\s*([A-Za-z]\d+)\s*['\"]?\s*\)", history_text)
        ordered = []
        for cid in found:
            norm = str(cid).upper()
            if norm not in ordered:
                ordered.append(norm)
        return ordered

    def _build_replan_history_block(
        self,
        task_description: str,
        previous_plan: Optional[str],
        execution_history,
        safety_context: Optional[SafetyContext],
        active_checkpoint_ids: list[str],
        benchmark_progress: Optional[dict] = None,
    ) -> str:
        current_collision_probability = 0.0 if safety_context is None else float(safety_context.current_collision_probability)
        if current_collision_probability < 0.65:
            return ""
        if previous_plan is None and execution_history is None:
            return ""

        progress = dict(benchmark_progress or {})
        completed = [str(v).upper() for v in list(progress.get("completed") or [])]
        if not completed:
            completed = self._extract_completed_checkpoints_from_history(execution_history)
        remaining = [cid for cid in active_checkpoint_ids if cid not in completed]
        current_target = progress.get("current_target")
        if current_target is not None:
            current_target = str(current_target).upper()
        if not current_target:
            current_target = "(n/a)" if not remaining else remaining[0]
        dominant_worker = "n/a"
        if safety_context is not None:
            dominant_worker = str(getattr(safety_context, "dominant_threat_id", "n/a") or "n/a")

        previous_plan_text = str(previous_plan or "").strip()
        if not previous_plan_text:
            previous_plan_text = "(n/a)"

        return (
            "Replan runtime history (this call is replan, not a fresh task):\n"
            f"- original user task: {task_description}\n"
            f"- previous plan: {previous_plan_text}\n"
            f"- completed checkpoints: {completed if completed else ['(none_detected)']}\n"
            f"- remaining checkpoints: {remaining if remaining else ['(none)']}\n"
            f"- current target checkpoint: {current_target}\n"
            "- replan trigger reason:\n"
            f"  - current collision probability >= 0.65 (current={current_collision_probability:.6f})\n"
            f"  - dominant risky worker = {dominant_worker}"
        )

    def plan(self, task_description: str, scene_description: Optional[str] = None, location_info: Optional[str] = None, error_message: Optional[str] = None, execution_history: Optional[str] = None, safety_context: Optional[SafetyContext] = None, previous_plan: Optional[str] = None):
    
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
        runtime_context_block = self._build_runtime_context_block(safety_context)
        snapshot = {}
        if self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot"):
            try:
                snapshot = self.controller.get_live_ui_snapshot() or {}
            except Exception:
                snapshot = {}
        objective = dict(snapshot.get("active_objective_set") or {})
        active_checkpoint_ids = [str(v) for v in objective.get("active_checkpoint_ids", [])]
        benchmark_progress = dict(snapshot.get("benchmark_progress") or {})
        replan_history_block = self._build_replan_history_block(
            task_description=task_description,
            previous_plan=previous_plan,
            execution_history=execution_history,
            safety_context=safety_context,
            active_checkpoint_ids=active_checkpoint_ids,
            benchmark_progress=benchmark_progress,
        )

        prompt = self.prompt_plan.format(
            system_skill_description_low=self.low_level_skillset,
            guides=self.guides,
            plan_examples=self.plan_examples,
            error_message=error_message,
            scene_description=full_scene,
            task_description=task_description,
            execution_history=execution_history,
            runtime_context=runtime_context_block,
            replan_history_block=replan_history_block,
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
        print_debug(f"[P-RUNTIME-CONTEXT]\n{runtime_context_block}")
        if replan_history_block:
            print_debug(f"[P-REPLAN-HISTORY]\n{replan_history_block}")
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
