import os, ast, re, json
from typing import Optional

from .safety_context import SafetyContext
from .skillset import SkillSet
from .llm_wrapper import LLMWrapper, GPT3, GPT4, chat_log_path
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_debug, print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value
from .abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLISION_PROBABILITY_REPLAN_THRESHOLD = 0.50

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
        self.prompt_langgraph_decomposition = (
            "You are TypeFly LangGraph Task Decomposer.\n"
            "Return checkpoint IDs only, as a JSON array, in execution order.\n"
            "Rules:\n"
            "1) Only use IDs from allowed checkpoints.\n"
            "2) Prefer single-zone checkpoints that match the task.\n"
            "3) Do not output explanation.\n"
            "Task: {task_description}\n"
            "Allowed checkpoints: {allowed_checkpoints}\n"
            "Active zones: {active_zones}\n"
            "Output JSON array only."
        )
        self.prompt_langgraph_step = (
            "You are TypeFly LangGraph Step Planner.\n"
            "You are NOT planning a full mission. You are deciding only the NEXT step.\n"
            "Output one action or a very short action segment (max 2 statements), each ending with ';'.\n"
            "Allowed actions: move_forward(distance); move_backward(distance); move_left(distance); move_right(distance); "
            "turn_cw(degrees); turn_ccw(degrees); delay(seconds); go_checkpoint(\"ID\"); log(\"text\");\n"
            "Hard constraints:\n"
            "- Do not output multi-checkpoint full plans.\n"
            "- current_subgoal is the primary mission target for this step.\n"
            "- If the same action was already executed and no progress improved, avoid repeating it.\n"
            "- If risk is high, you may temporarily choose conservative avoidance/recovery action.\n"
            "- Avoidance is temporary, not a new mission objective.\n"
            "- Recovery mode is controlled by system state; follow it explicitly.\n"
            "- If recovery_mode is true, prioritize conservative recovery movement and re-observation over direct checkpoint approach.\n"
            "- If recovery_mode is false (or recovery just exited because risk <= 0.2), prioritize go_checkpoint(current_subgoal).\n"
            "- If current_subgoal risk is high, do not blindly rush go_checkpoint(current_subgoal).\n"
            "- Recovery should be step-by-step: after each recovery step, re-observe latest risk/state before deciding the next step.\n"
            "- During each re-observation, explicitly check current collision risk.\n"
            "- If current collision risk drops below 0.2, prioritize returning to go_checkpoint(current_subgoal).\n"
            "- When you judge immediate risk has improved to a safe level, return to go_checkpoint(current_subgoal).\n"
            "- If risk is still high after re-observation, choose another conservative recovery step and re-observe again.\n"
            "- Recovery goal is to leave danger and then resume current_subgoal, not endless wandering.\n"
            "- If stalled/no-progress, switch to a different conservative strategy.\n"
            "- Use strategy memory to avoid repeating known failed approach patterns for this subgoal.\n"
            "- Avoid long turn/move loops without objective progress.\n"
            "- reached checkpoint area is NOT completed checkpoint.\n"
            "- Completion is determined only by official completion_state/progress.\n"
            "- If reached but not completed, prioritize finishing current_subgoal (hold/re-approach/micro-adjust), not jumping to next checkpoint.\n"
            "Task: {task_description}\n"
            "Current subgoal: {current_subgoal}\n"
            "Remaining checkpoints: {remaining_checkpoints}\n"
            "Current collision risk: {current_collision_risk}\n"
            "Historical max collision risk: {historical_max_collision_risk}\n"
            "Dominant risky worker: {dominant_risky_worker}\n"
            "Per-worker collision risks: {per_worker_collision_risks}\n"
            "Recovery mode: {recovery_mode}\n"
            "Recovery reason: {recovery_reason}\n"
            "Strategy summary: {strategy_summary}\n"
            "Last failure reason: {last_failure_reason}\n"
            "Recent failed approach pattern: {failed_approach_pattern}\n"
            "Recent recovery hypothesis: {recovery_hypothesis}\n"
            "Blocked workers for current subgoal: {blocked_workers_for_subgoal}\n"
            "Attempt history for current subgoal: {subgoal_attempts}\n"
            "Worker states summary: {worker_states_summary}\n"
            "Last action: {last_action}\n"
            "Last result: {last_result}\n"
            "Stall count: {stall_count}\n"
            "Repeated action count: {repeated_action_count}\n"
            "Recent execution history: {recent_history}\n"
            "Return action statements only, no explanation."
        )
        self.prompt_langgraph_reflection = (
            "You are TypeFly LangGraph Strategy Reflector.\n"
            "Analyze last step outcome and decide strategy-level correction.\n"
            "Return JSON only with keys:\n"
            "- failure_reason: short string\n"
            "- failed_approach_pattern: short string for what should NOT be repeated now\n"
            "- recovery_hypothesis: short string describing safer next tactic\n"
            "- strategy_summary: short string for ongoing strategy memory\n"
            "- next_route_decision: one of [continue, retry_plan, reselect_subgoal]\n"
            "- next_subgoal: checkpoint id or null\n"
            "- reprioritized_subgoals: ordered checkpoint list or []\n"
            "Rules:\n"
            "- If last step failed or risk_abort happened, explain why and avoid repeating same approach.\n"
            "- You may reprioritize checkpoints when current subgoal is temporarily high-risk or blocked.\n"
            "- Reprioritization must be temporary and mission-oriented.\n"
            "- Keep every field concise.\n"
            "Task: {task_description}\n"
            "Current subgoal: {current_subgoal}\n"
            "Route decision from evaluator: {route_decision}\n"
            "Last action: {last_action}\n"
            "Last result: {last_result}\n"
            "Current collision risk: {current_collision_risk}\n"
            "Dominant risky worker: {dominant_risky_worker}\n"
            "Per-worker collision risks: {per_worker_collision_risks}\n"
            "Worker states summary: {worker_states_summary}\n"
            "Remaining checkpoints: {remaining_checkpoints}\n"
            "Current queue: {subgoal_queue}\n"
            "Current strategy summary: {current_strategy_summary}\n"
            "Last failure reason memory: {last_failure_reason}\n"
            "Recent failed approach memory: {recent_failed_approach_pattern}\n"
            "Recent recovery hypothesis memory: {recent_recovery_hypothesis}\n"
            "Blocked workers by subgoal: {blocked_workers_by_subgoal}\n"
            "Subgoal attempt history: {subgoal_attempt_history}\n"
            "Recent execution history: {recent_history}\n"
            "Output JSON only."
        )

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
        if current_collision_probability < float(COLLISION_PROBABILITY_REPLAN_THRESHOLD):
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
            f"  - current collision probability >= {float(COLLISION_PROBABILITY_REPLAN_THRESHOLD):.2f} "
            f"(current={current_collision_probability:.6f})\n"
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

    def decompose_task_for_langgraph(
        self,
        task_description: str,
        active_checkpoint_ids: list[str],
        active_zone_ids: list[str],
    ) -> list[str]:
        prompt = self.prompt_langgraph_decomposition.format(
            task_description=str(task_description or ""),
            allowed_checkpoints=[str(v).upper() for v in list(active_checkpoint_ids or [])],
            active_zones=[str(v) for v in list(active_zone_ids or [])],
        )
        raw = str(self.llm.request(prompt, self.model_name, stream=False) or "").strip()
        parsed: list[str] = []
        try:
            obj = ast.literal_eval(raw)
            if isinstance(obj, list):
                parsed = [str(v).upper() for v in obj]
        except Exception:
            tokens = re.findall(r"[A-Za-z]\d+", raw)
            parsed = [str(v).upper() for v in tokens]
        allowed = [str(v).upper() for v in list(active_checkpoint_ids or [])]
        filtered = [cid for cid in parsed if cid in allowed]
        if filtered:
            return filtered
        return allowed

    def plan_langgraph_step_action(
        self,
        task_description: str,
        current_subgoal: str | None,
        remaining_checkpoints: list[str],
        current_collision_risk: float,
        historical_max_collision_risk: float,
        per_worker_collision_risks: dict[str, float],
        dominant_risky_worker: str | None,
        worker_states_summary: list[dict],
        last_action: str,
        last_result: str,
        stall_count: int,
        repeated_action_count: int,
        recent_history: list[dict],
        recovery_mode: bool = False,
        recovery_reason: str | None = None,
        strategy_summary: str = "",
        last_failure_reason: str | None = None,
        failed_approach_pattern: str | None = None,
        recovery_hypothesis: str | None = None,
        blocked_workers_for_subgoal: list[str] | None = None,
        subgoal_attempts: list[str] | None = None,
    ) -> str:
        prompt = self.prompt_langgraph_step.format(
            task_description=str(task_description or ""),
            current_subgoal=str(current_subgoal or "None"),
            remaining_checkpoints=[str(v) for v in list(remaining_checkpoints or [])],
            current_collision_risk=f"{float(current_collision_risk):.6f}",
            historical_max_collision_risk=f"{float(historical_max_collision_risk):.6f}",
            per_worker_collision_risks=str(dict(per_worker_collision_risks or {})),
            dominant_risky_worker=str(dominant_risky_worker or "n/a"),
            recovery_mode=str(bool(recovery_mode)),
            recovery_reason=str(recovery_reason or "none"),
            strategy_summary=str(strategy_summary or ""),
            last_failure_reason=str(last_failure_reason or "none"),
            failed_approach_pattern=str(failed_approach_pattern or "none"),
            recovery_hypothesis=str(recovery_hypothesis or "none"),
            blocked_workers_for_subgoal=str(list(blocked_workers_for_subgoal or [])),
            subgoal_attempts=str(list(subgoal_attempts or [])),
            worker_states_summary=str(list(worker_states_summary or [])[:3]),
            last_action=str(last_action or ""),
            last_result=str(last_result or ""),
            stall_count=int(stall_count),
            repeated_action_count=int(repeated_action_count),
            recent_history=str(list(recent_history or [])[-4:]),
        )
        raw = str(self.llm.request(prompt, self.model_name, stream=False) or "").strip()
        action = self._sanitize_langgraph_action(raw)
        if action:
            return action
        if current_subgoal:
            return f'go_checkpoint("{str(current_subgoal).upper()}");'
        return "delay(1.0);"

    def reflect_langgraph_strategy(
        self,
        task_description: str,
        current_subgoal: str | None,
        route_decision: str,
        last_action: str,
        last_result: dict,
        current_collision_risk: float,
        per_worker_collision_risks: dict[str, float],
        dominant_risky_worker: str | None,
        worker_states_summary: list[dict],
        remaining_checkpoints: list[str],
        subgoal_queue: list[str],
        current_strategy_summary: str,
        last_failure_reason: str | None,
        recent_failed_approach_pattern: str | None,
        recent_recovery_hypothesis: str | None,
        blocked_workers_by_subgoal: dict[str, list[str]],
        subgoal_attempt_history: dict[str, list[str]],
        recent_history: list[dict],
    ) -> dict:
        prompt = self.prompt_langgraph_reflection.format(
            task_description=str(task_description or ""),
            current_subgoal=str(current_subgoal or "None"),
            route_decision=str(route_decision or "continue"),
            last_action=str(last_action or ""),
            last_result=str(last_result or {}),
            current_collision_risk=f"{float(current_collision_risk):.6f}",
            dominant_risky_worker=str(dominant_risky_worker or "n/a"),
            per_worker_collision_risks=str(dict(per_worker_collision_risks or {})),
            worker_states_summary=str(list(worker_states_summary or [])[:3]),
            remaining_checkpoints=str([str(v).upper() for v in list(remaining_checkpoints or [])]),
            subgoal_queue=str([str(v).upper() for v in list(subgoal_queue or [])]),
            current_strategy_summary=str(current_strategy_summary or ""),
            last_failure_reason=str(last_failure_reason or "none"),
            recent_failed_approach_pattern=str(recent_failed_approach_pattern or "none"),
            recent_recovery_hypothesis=str(recent_recovery_hypothesis or "none"),
            blocked_workers_by_subgoal=str(dict(blocked_workers_by_subgoal or {})),
            subgoal_attempt_history=str(dict(subgoal_attempt_history or {})),
            recent_history=str(list(recent_history or [])[-6:]),
        )
        raw = str(self.llm.request(prompt, self.model_name, stream=False) or "").strip()
        parsed = self._safe_json_object(raw)

        fallback_failure = ""
        if not bool((last_result or {}).get("ok", True)):
            fallback_failure = str((last_result or {}).get("message", "step_failed"))[:120]
        next_route = str(parsed.get("next_route_decision", route_decision or "continue"))
        if next_route not in {"continue", "retry_plan", "reselect_subgoal"}:
            next_route = str(route_decision or "continue")
        return {
            "failure_reason": str(parsed.get("failure_reason", fallback_failure)).strip()[:200],
            "failed_approach_pattern": str(parsed.get("failed_approach_pattern", "")).strip()[:120],
            "recovery_hypothesis": str(parsed.get("recovery_hypothesis", "")).strip()[:160],
            "strategy_summary": str(parsed.get("strategy_summary", current_strategy_summary or "")).strip()[:240],
            "next_route_decision": next_route,
            "next_subgoal": parsed.get("next_subgoal"),
            "reprioritized_subgoals": parsed.get("reprioritized_subgoals", []),
        }

    def _safe_json_object(self, raw_text: str) -> dict:
        text = str(raw_text or "").strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            try:
                obj = ast.literal_eval(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {}
        return {}

    def _sanitize_langgraph_action(self, raw_text: str) -> str:
        text = str(raw_text or "").strip().replace("\n", " ")
        if not text:
            return ""
        commands = []
        patterns = [
            r'go_checkpoint\(\s*[\'"]?([A-Za-z]\d+)[\'"]?\s*\)\s*;?',
            r'move_forward\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;?',
            r'move_backward\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;?',
            r'move_left\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;?',
            r'move_right\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;?',
            r'turn_cw\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;?',
            r'turn_ccw\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;?',
            r'delay\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;?',
            r'log\(\s*[\'"]([^\'"]{1,60})[\'"]\s*\)\s*;?',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                token = match.group(0)
                token = token.strip()
                if token.startswith("go_checkpoint"):
                    cid = match.group(1).upper()
                    commands.append(f'go_checkpoint("{cid}");')
                elif token.startswith("log("):
                    safe = re.sub(r"[^a-zA-Z0-9 _-]", "", match.group(1))[:60]
                    commands.append(f'log("{safe}");')
                else:
                    name = token.split("(", 1)[0]
                    value = float(match.group(1))
                    if name.startswith("turn_"):
                        commands.append(f"{name}({int(value)});")
                    else:
                        commands.append(f"{name}({value:.1f});")
                if len(commands) >= 2:
                    break
            if len(commands) >= 2:
                break
        return "".join(commands)
