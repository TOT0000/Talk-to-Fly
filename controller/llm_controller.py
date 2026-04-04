from PIL import Image
from dataclasses import replace
import math
import queue, time, os, sys, subprocess
import re
from typing import Optional, Tuple
import asyncio
import uuid
import threading
import numpy as np

from .shared_frame import SharedFrame, Frame
from .gcs_safety_assessment import GcsSafetyAssessmentService
from .yolo_client import YoloClient
from .yolo_grpc_client import YoloGRPCClient
from .tello_wrapper import TelloWrapper
from .virtual_robot_wrapper import VirtualRobotWrapper
from .px4_sim_robot_wrapper import Px4SimRobotWrapper
from .abs.robot_wrapper import RobotWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, SkillArg
from .utils import print_debug, print_t
from .minispec_interpreter import MiniSpecInterpreter, Statement
from .abs.robot_wrapper import RobotType
from .uwb_wrapper import UWBWrapper
from .state_provider import StateProvider, UwbStateProvider
from .sim_state_provider import SimStateProvider
from .scenario_manager import ScenarioManager
from .safety_context import SafetyContext
from .task_run_logger import TaskRunLogger
from .baseline_scenes import (
    BASELINE_SCENES,
    BaselineScene,
    build_all_scene_expectations,
    build_scene_expectations,
    compute_obstacle_envelope_states,
    evaluate_path_clear,
    get_task_point,
    normalize_baseline_scene_id,
)
from .benchmark_layout import (
    BENCHMARK_CHECKPOINT_ORDER,
    BENCHMARK_CHECKPOINTS_BY_ID,
    BENCHMARK_CHECKPOINTS,
    BENCHMARK_ZONES,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD = 0.30
COLLISION_PROBABILITY_REPLAN_THRESHOLD = 0.65

class LLMController():
    def __init__(self, robot_type, virtual_queue, use_http=False, message_queue: Optional[queue.Queue]=None, enable_video=False, state_provider: Optional[StateProvider]=None):
        self.virtual_queue = virtual_queue
        self.robot_type = robot_type
        self.enable_video = enable_video
        self.shared_frame = SharedFrame()
        if use_http:
            self.yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
        self.vision = VisionSkillWrapper(self.shared_frame, enabled=enable_video)
        self.latest_frame = None
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            try:
                self.cache_folder = message_queue.get(timeout=1.0)
            except queue.Empty:
                self.cache_folder = "cache/default"

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        
        match robot_type:
            case RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                self.drone: RobotWrapper = TelloWrapper()
            case RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from .gear_wrapper import GearWrapper
                self.drone: RobotWrapper = GearWrapper()
            case RobotType.PX4_SIM:
                print_t("[C] Start PX4 sim drone...")
                self.drone: RobotWrapper = Px4SimRobotWrapper(enable_video=self.enable_video)
            case _:
                print_t("[C] Start virtual drone...")
                self.drone: RobotWrapper = VirtualRobotWrapper(enable_video=self.enable_video)
        
        self.planner = LLMPlanner(robot_type)
        self.planner.controller = self
        self.safety_assessor = GcsSafetyAssessmentService()

        # state provider
        self.uwb = UWBWrapper()
        if robot_type == RobotType.PX4_SIM:
            self.state_provider = SimStateProvider()
        elif state_provider is not None:
            self.state_provider = state_provider
        else:
            self.state_provider = UwbStateProvider(self.uwb, self.drone)


        # inject provider into PX4 sim wrapper
        if robot_type == RobotType.PX4_SIM and hasattr(self.drone, "set_state_provider"):
            self.drone.set_state_provider(self.state_provider)

        self.position_update_callback = None
        self.state_provider.register_callback(self.notify_user_position_updated)

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("go_checkpoint", self.skill_go_checkpoint, "Navigate toward a benchmark checkpoint by ID", args=[SkillArg("checkpoint_id", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("takeoff", self.skill_takeoff, "Take off and climb to a safe hover height"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("land", self.skill_land, "Land safely at current location"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.skill_delay, "Wait for specified seconds", args=[SkillArg("seconds", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("probe", self.planner.probe, "Probe the LLM for reasoning", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log", self.skill_log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("take_picture", self.skill_take_picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("re_plan", self.skill_re_plan, "Replanning"))

        self.low_level_skillset.add_skill(LowLevelSkillItem("time", self.skill_time, "Get current execution time", args=[]))
      

        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        self.planner.init(low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

        self.current_plan = None
        self.execution_history = None
        self.execution_time = time.time()
        self.latest_safety_context = None
        self.scenario_manager = ScenarioManager(default_name=os.getenv("TYPEFLY_SCENARIO", "SAFE"))
        self.task_run_logger = TaskRunLogger(excel_path=os.getenv("TYPEFLY_TASK_LOG_XLSX", "logs/task_runs.xlsx"))
        self._task_id_counter = 0
        self.latest_scenario_report = None
        self.initial_scenario_state = None
        self.baseline_scene_id = normalize_baseline_scene_id(os.getenv("TYPEFLY_BASELINE_SCENE", "SCENE_1_CLEAR_PATH"))
        self.baseline_scene_state = None
        self.latest_baseline_decision = None
        self.execution_mode = "Waiting"
        self.active_objective_set = self._default_active_objective_set()
        self.latest_benchmark_progress = {
            "completed": [],
            "current_target": None,
        }
        self.latest_ui_collision_probability = None
        self.latest_ui_collision_timestamp = 0.0
        self.planner_mode = str(os.getenv("TYPEFLY_PLANNER_MODE", "llm_baseline")).strip().lower()
        if self.planner_mode not in {"llm_baseline", "rule_baseline"}:
            self.planner_mode = "llm_baseline"
        self.user_heading_yaw_rad = 0.0

        # PX4_SIM optional managed user-position publisher lifecycle
        self._sim_user_publisher_proc: Optional[subprocess.Popen] = None
        self._owns_sim_user_publisher = False

    def _default_active_objective_set(self) -> dict:
        return {
            "active_zone_ids": [zone.id for zone in BENCHMARK_ZONES],
            "active_checkpoint_ids": list(BENCHMARK_CHECKPOINT_ORDER),
            "source": "default_all",
        }

    def _resolve_active_objective_set(self, task_text: str) -> dict:
        text = str(task_text or "")
        normalized = text.upper()
        all_zone_ids = [zone.id for zone in BENCHMARK_ZONES]
        zone_to_checkpoints = {
            "zone_A": [cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid.startswith("A")],
            "zone_B": [cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid.startswith("B")],
            "zone_C": [cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid.startswith("C")],
        }

        all_keywords = (
            "ALL ZONES",
            "ALL CHECKPOINT",
            "COMPLETE ALL",
            "全部區域",
            "全部检查点",
            "全部檢查點",
            "全部巡檢點",
        )
        if any(key in normalized for key in all_keywords) or any(key in text for key in all_keywords):
            return {
                "active_zone_ids": all_zone_ids,
                "active_checkpoint_ids": list(BENCHMARK_CHECKPOINT_ORDER),
                "source": "task_parse_all",
            }

        zone_tokens = set()
        context_hits = any(word in normalized for word in ("ZONE", "ZONES", "AREA", "CHECKPOINT", "INSPECT", "SEARCH"))
        context_hits = context_hits or any(word in text for word in ("區域", "巡檢", "搜尋", "搜索", "檢查點", "检查点"))
        for token, zone_id in (("A", "zone_A"), ("B", "zone_B"), ("C", "zone_C")):
            if re.search(rf"\b(?:ZONE|AREA)\s*{token}\b", normalized):
                zone_tokens.add(zone_id)
            if re.search(rf"\b{token}\b\s*(?:ZONE|AREA)\b", normalized):
                zone_tokens.add(zone_id)
            if f"{token}區域" in text or f"{token} 區域" in text or f"區域{token}" in text:
                zone_tokens.add(zone_id)
            if context_hits and re.search(rf"\b{token}\b", normalized):
                zone_tokens.add(zone_id)

        if not zone_tokens:
            return self._default_active_objective_set()

        active_zone_ids = sorted(zone_tokens)
        active_checkpoint_ids = []
        for zid in active_zone_ids:
            active_checkpoint_ids.extend(zone_to_checkpoints.get(zid, []))
        return {
            "active_zone_ids": active_zone_ids,
            "active_checkpoint_ids": active_checkpoint_ids,
            "source": "task_parse_zone",
        }
        
    def register_position_callback(self, callback):
        self.position_update_callback = callback
        
    def notify_user_position_updated(self, position: Tuple[float, float, float, float]):
        timestamp, x, y, z = position
        position_str = f"User position updated: x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        if hasattr(self, 'position_update_callback') and self.position_update_callback:
            source = "user"
            self.position_update_callback(x, y, z, source)
    
    def _get_planner_position_context(self):
        drone_pos = (0.00, 0.00, 0.00)
        user_pos = (0.00, 0.00, 0.00)
        drone_source = "fallback_zero"
        user_source = "fallback_zero"

        try:
            provider = getattr(self, "state_provider", None)
            drone = getattr(self, "drone", None)

            if provider is not None:
                get_est_drone = getattr(provider, "get_estimated_drone_position", None)
                if callable(get_est_drone):
                    value = get_est_drone()
                    if value is not None:
                        drone_pos = tuple(float(v) for v in value)
                        drone_source = "provider_estimated"

                get_est_user = getattr(provider, "get_estimated_user_position", None)
                if callable(get_est_user):
                    value = get_est_user()
                    if value is not None:
                        user_pos = tuple(float(v) for v in value)
                        user_source = "provider_estimated"

                if drone_source == "fallback_zero":
                    packet = None
                    get_packet = getattr(provider, "get_latest_received_drone_packet", None)
                    if callable(get_packet):
                        packet = get_packet()
                    if packet is not None:
                        drone_pos = tuple(float(v) for v in packet.estimated_position_3d)
                        drone_source = "received_packet_estimated"

                if user_source == "fallback_zero":
                    packet = None
                    get_packet = getattr(provider, "get_latest_received_user_packet", None)
                    if callable(get_packet):
                        packet = get_packet()
                    if packet is not None:
                        user_pos = tuple(float(v) for v in packet.estimated_position_3d)
                        user_source = "received_packet_estimated"

                if drone_source == "fallback_zero":
                    get_gt_drone = getattr(provider, "get_drone_position", None)
                    if callable(get_gt_drone):
                        value = get_gt_drone()
                        if value is not None:
                            drone_pos = tuple(float(v) for v in value)
                            drone_source = "provider_position_fallback"

                if user_source == "fallback_zero":
                    get_user = getattr(provider, "get_user_position", None)
                    if callable(get_user):
                        value = get_user()
                        if value is not None:
                            user_pos = tuple(float(v) for v in value)
                            user_source = "provider_position_fallback"

            if drone_source == "fallback_zero" and drone is not None:
                get_drone_position = getattr(drone, "get_drone_position", None)
                if callable(get_drone_position):
                    value = get_drone_position()
                    if value is not None:
                        drone_pos = tuple(float(v) for v in value)
                        drone_source = "drone_position_fallback"
        except Exception:
            pass

        return {
            "drone_pos": drone_pos,
            "user_pos": user_pos,
            "drone_source": drone_source,
            "user_source": user_source,
        }

    def _format_planner_location_info(self) -> str:
        context = self._get_planner_position_context()
        drone_pos = context["drone_pos"]
        user_pos = context["user_pos"]
        print_debug(
            "[P-LOCATION-CONTEXT] "
            f"drone_source={context['drone_source']} user_source={context['user_source']} "
            f"drone_est={drone_pos} user_est={user_pos}"
        )
        return (
            f"Drone estimated position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}\n"
            f"User estimated position: x={user_pos[0]:.2f}, y={user_pos[1]:.2f}, z={user_pos[2]:.2f}"
        )

    def skill_get_drone_position(self) -> Tuple[str, bool]:
        context = self._get_planner_position_context()
        x, y, z = context["drone_pos"]
        position_str = f"Drone estimated position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        return position_str, False
        
    def skill_get_user_position(self) -> Tuple[str, bool]:
        context = self._get_planner_position_context()
        x, y, z = context["user_pos"]
        position_str = f"User estimated position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        return position_str, False
        
    def start_uwb(self):
        print_t("[C] Starting UWB tracking...")
        self.state_provider.start()
        self.uwb_active = True
        
    def stop_uwb(self):
        print_t("[C] Stopping UWB tracking...")
        self.state_provider.stop()
        self.uwb_active = False
        
    def start_virtual_position_loop(self):
        self.virtual_position_active = True
        def loop():
            while self.controller_active and self.virtual_position_active:
                x, y, z = self.drone.get_drone_position()
                try:
                    self.virtual_queue.put_nowait((time.time(), x, y, z))
                except queue.Full:
                    self.virtual_queue.get()
                    self.virtual_queue.put_nowait((time.time(), x, y, z))
                if self.position_update_callback:
                    self.position_update_callback(x, y, z, "drone")
                time.sleep(0.1)
        threading.Thread(target=loop, daemon=True).start()
        
    def stop_virtual_position_loop(self):
        self.virtual_position_active = False


    def skill_time(self) -> Tuple[float, bool]:
        return time.time() - self.execution_time, False

    def skill_take_picture(self) -> Tuple[None, bool]:
        img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
        Image.fromarray(self.latest_frame).save(img_path)
        print_t(f"[C] Picture saved to {img_path}")
        self.append_message((img_path,))
        return None, False

    def skill_log(self, text: str) -> Tuple[None, bool]:
        self.append_message(f"[LOG] {text}")
        print_t(f"[LOG] {text}")
        return None, False
    
    def skill_re_plan(self) -> Tuple[None, bool]:
        print_t("[REPLAN_DEBUG] source=rp_skill")
        return None, True

    def skill_takeoff(self) -> Tuple[None, bool]:
        ok = self.drone.takeoff()
        if isinstance(ok, tuple):
            return ok
        return (None, not bool(ok))

    def skill_land(self) -> Tuple[None, bool]:
        self.drone.land()
        return None, False

    def skill_delay(self, s: float) -> Tuple[None, bool]:
        time.sleep(s)
        return None, False

    def skill_go_checkpoint(self, checkpoint_id: str) -> Tuple[str, bool]:
        checkpoint_key = str(checkpoint_id or "").strip().strip("'\"").upper()
        checkpoint = BENCHMARK_CHECKPOINTS_BY_ID.get(checkpoint_key)
        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint_id `{checkpoint_id}`")

        max_step_m = 1.0
        min_axis_step_m = 0.15
        max_iterations = 8

        reached = False
        final_dist = None
        for _ in range(max_iterations):
            context = self._get_planner_position_context()
            drone_x, drone_y, _ = context["drone_pos"]
            dx = float(checkpoint.x) - float(drone_x)
            dy = float(checkpoint.y) - float(drone_y)
            dist = math.hypot(dx, dy)
            final_dist = dist

            if dist <= float(checkpoint.radius_m):
                reached = True
                break

            move_x = max(-max_step_m, min(max_step_m, dx))
            move_y = max(-max_step_m, min(max_step_m, dy))

            if abs(move_x) >= min_axis_step_m:
                if move_x > 0:
                    self.drone.move_right(abs(move_x))
                else:
                    self.drone.move_left(abs(move_x))

            if abs(move_y) >= min_axis_step_m:
                if move_y > 0:
                    self.drone.move_forward(abs(move_y))
                else:
                    self.drone.move_backward(abs(move_y))

        status = "reached" if reached else "approached"
        summary = (
            f"go_checkpoint({checkpoint.id}) {status}: "
            f"zone={checkpoint.zone_id}, target=({checkpoint.x:.2f},{checkpoint.y:.2f}), "
            f"remaining_dist={0.0 if final_dist is None else float(final_dist):.2f}m"
        )
        print_t(f"[C] {summary}")
        return summary, False

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def update_benchmark_progress(self, completed_checkpoint_ids, current_target_checkpoint):
        completed = [str(v).upper() for v in list(completed_checkpoint_ids or [])]
        self.latest_benchmark_progress = {
            "completed": sorted(set(completed)),
            "current_target": (None if current_target_checkpoint is None else str(current_target_checkpoint).upper()),
        }

    def update_ui_collision_probability(self, current_collision_probability: Optional[float]):
        if current_collision_probability is None:
            return
        self.latest_ui_collision_probability = float(current_collision_probability)
        self.latest_ui_collision_timestamp = time.time()

    def stop_controller(self):
        self.controller_active = False

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            self.vision.update()
            YoloClient.plot_results_oi(image, self.vision.object_list)
        return image
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter(self.message_queue, should_abort=self._should_abort_current_execution_for_replan)
        interpreter.execute(minispec)
        self.execution_history = interpreter.execution_history
        ret_val = interpreter.ret_queue.get()
        if hasattr(ret_val, "value") and isinstance(ret_val.value, str) and ret_val.value.startswith("MiniSpec execution error"):
            raise RuntimeError(ret_val.value)
        return ret_val

    def _should_abort_current_execution_for_replan(self) -> Tuple[bool, str]:
        snapshot = self.get_live_ui_snapshot()
        if not isinstance(snapshot, dict):
            return False, ""
        safety_context = snapshot.get("safety_context")
        if safety_context is None:
            return False, ""
        current_p = float(getattr(safety_context, "current_collision_probability", 0.0))
        should_abort = bool(current_p >= COLLISION_PROBABILITY_REPLAN_THRESHOLD)
        if should_abort:
            dominant = str(getattr(safety_context, "dominant_threat_id", "unknown"))
            ui_p = self.latest_ui_collision_probability
            ui_p_text = "n/a" if ui_p is None else f"{float(ui_p):.6f}"
            print_t(
                "[REPLAN_DEBUG] "
                f"source=collision_threshold "
                f"ui_pc={ui_p_text} "
                f"callback_pc={current_p:.6f} "
                f"threshold={COLLISION_PROBABILITY_REPLAN_THRESHOLD:.6f} "
                f"should_abort={should_abort} "
                f"dominant={dominant}"
            )
            return True, f"current_collision_probability={current_p:.6f}>=0.65, dominant={dominant}"
        return False, ""

    def _sanitize_minispec_plan(self, raw_plan: str) -> str:
        if raw_plan is None:
            return ""
        text = str(raw_plan).strip()
        if not text:
            return ""

        response_match = re.search(r"(?is)\bresponse\s*:\s*(.+)$", text)
        if response_match:
            text = response_match.group(1).strip()

        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("```"):
                continue
            if re.match(r"(?is)^(plan|analysis|thought|reasoning)\s*:", stripped):
                continue
            lines.append(stripped)
        text = " ".join(lines).strip()

        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,;:?!&|<>=(){}[]'\"+-*/% \t")
        filtered = "".join(ch for ch in text if ch in allowed_chars).strip()
        if filtered:
            text = filtered

        command_pattern = re.compile(
            r"([A-Za-z_][A-Za-z0-9_-]*\s*\([^;{}]*\)|->[A-Za-z0-9_.'\"+\-*/%()]+|[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^;{}]+|\d+\s*\{|[{}])\s*;?",
        )
        matches = [m.group(1).strip() for m in command_pattern.finditer(text)]
        if matches:
            rebuilt = []
            for token in matches:
                if token in {"{", "}"}:
                    rebuilt.append(token)
                elif token.endswith("{"):
                    rebuilt.append(token)
                else:
                    rebuilt.append(f"{token};")
            candidate = "".join(rebuilt).strip()
            if candidate:
                return candidate
        return text

    def _has_live_sim_user_position(self) -> bool:
        if not hasattr(self, "state_provider"):
            return False
        last_ts = getattr(self.state_provider, "_last_user_position_ts", 0.0)
        return bool(last_ts and (time.time() - float(last_ts) < 1.5))

    def _start_sim_user_position_publisher_if_needed(self):
        if self.robot_type != RobotType.PX4_SIM:
            return

        autostart = os.getenv("SIM_USER_POSITION_AUTOSTART", "1").strip().lower()
        if autostart in {"0", "false", "no", "off"}:
            return

        # If external source already publishes user position, do not start another publisher.
        deadline = time.time() + 0.6
        while time.time() < deadline:
            if self._has_live_sim_user_position():
                return
            time.sleep(0.1)

        if self._sim_user_publisher_proc is not None and self._sim_user_publisher_proc.poll() is None:
            return

        script_path = os.path.join(CURRENT_DIR, "sim_user_position_publisher.py")
        if not os.path.exists(script_path):
            print_t(f"[WARN] sim user publisher script not found: {script_path}")
            return

        topic = os.getenv("SIM_USER_POSITION_TOPIC", "/sim/user_position")
        x = os.getenv("SIM_USER_POSITION_PUB_X", "8.0")
        y = os.getenv("SIM_USER_POSITION_PUB_Y", "8.0")
        z = os.getenv("SIM_USER_POSITION_PUB_Z", "0.0")
        rate = os.getenv("SIM_USER_POSITION_PUB_RATE", "10.0")

        cmd = [sys.executable, script_path, "--topic", topic, "--x", x, "--y", y, "--z", z, "--rate", rate]
        try:
            self._sim_user_publisher_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._owns_sim_user_publisher = True
            print_t(f"[C] Started sim user position publisher on {topic}")
        except Exception as exc:
            print_t(f"[WARN] Failed to start sim user position publisher: {exc}")

    def _stop_sim_user_position_publisher(self):
        if not self._owns_sim_user_publisher:
            return
        proc = self._sim_user_publisher_proc
        if proc is None:
            return

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)

        self._sim_user_publisher_proc = None
        self._owns_sim_user_publisher = False

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        self.execution_mode = "Planning"
        self.active_objective_set = self._resolve_active_objective_set(task_description)
        self._task_id_counter += 1
        task_id = f"task_{self._task_id_counter:05d}"
        initial_snapshot = self.get_live_ui_snapshot()
        self.task_run_logger.start_run(
            task_id=task_id,
            task_text=task_description,
            scenario_name=self.get_active_scenario_name(),
            initial_snapshot=initial_snapshot,
        )
        self.append_message('[TASK]: ' + task_description)
        ret_val = None
        monitor_stop = threading.Event()
        monitor_thread = None
        replan_attempts = 0
        max_replan_attempts = 3
        def _run_monitor():
            while not monitor_stop.is_set():
                try:
                    self.task_run_logger.consume_runtime_snapshot(self.get_live_ui_snapshot())
                except Exception:
                    pass
                monitor_stop.wait(0.2)
        monitor_thread = threading.Thread(target=_run_monitor, daemon=True)
        monitor_thread.start()
        while True:
            location_info = self._format_planner_location_info()
            runtime_snapshot = self.get_live_ui_snapshot()

            scene_description = self.vision.get_obj_list() if self.enable_video else ''
            if hasattr(self.state_provider, "debug_log_latest_localization_snapshot"):
                self.state_provider.debug_log_latest_localization_snapshot(reason="pre-plan")
            safety_context = runtime_snapshot.get("safety_context") if isinstance(runtime_snapshot, dict) else None
            if safety_context is None:
                safety_context = self.state_provider.get_latest_safety_context() if hasattr(self.state_provider, "get_latest_safety_context") else None
            if safety_context is None:
                safety_context = self.safety_assessor.build_from_provider(self.state_provider)
            self._debug_log_safety_context(safety_context)
            
            try:
                baseline = self._build_baseline_control_plan(task_description=task_description, snapshot=initial_snapshot)
                if baseline is not None:
                    self.latest_baseline_decision = baseline
                    self.task_run_logger.update_baseline_info(baseline)

                llm_called = False
                final_plan_source = "llm"
                baseline_shortcut_triggered = False
                if self.planner_mode == "rule_baseline" and baseline is not None:
                    self.current_plan = baseline["plan"]
                    llm_called = False
                    final_plan_source = "baseline_rule"
                    baseline_shortcut_triggered = True
                else:
                    previous_plan = self.current_plan
                    self.execution_mode = "Planning"
                    self.current_plan = self.planner.plan(
                        task_description=task_description,
                        scene_description=scene_description,
                        location_info=location_info,
                        execution_history=self.execution_history,
                        safety_context=safety_context,
                        previous_plan=previous_plan,
                    )
                    llm_called = True
                    final_plan_source = "llm"
                self.current_plan = self._sanitize_minispec_plan(self.current_plan)
                self.latest_safety_context = safety_context
                self.task_run_logger.update_plan_info(self.current_plan, generation_success=True)
                debug_info = {
                    "task_text": task_description,
                    "planner_mode": self.planner_mode,
                    "llm_called": llm_called,
                    "llm_function": "LLMPlanner.plan" if llm_called else "None",
                    "baseline_shortcut_triggered": baseline_shortcut_triggered,
                    "selected_target": None if baseline is None else baseline.get("target_task_point"),
                    "path_clear": None if baseline is None else baseline.get("path_clear"),
                    "blocking_entity": None if baseline is None else baseline.get("blocking_entity"),
                    "final_plan_source": final_plan_source,
                    "final_plan_text": self.current_plan,
                }
                if hasattr(self.task_run_logger, "update_planner_info"):
                    self.task_run_logger.update_planner_info(debug_info)
                print_debug(
                    "[TASK-PLANNER-FLOW] "
                    + ", ".join(f"{k}={v}" for k, v in debug_info.items())
                )

                self.append_message(f'[Plan]: \\\\')
                self.execution_time = time.time()
                self.execution_mode = "Executing"
                ret_val = self.execute_minispec(self.current_plan)
                execution_success = True
                task_completed = True
                if isinstance(ret_val, tuple) and len(ret_val) >= 2:
                    execution_success = bool(ret_val[0] is not False)
                if hasattr(ret_val, "replan") and bool(ret_val.replan):
                    replan_source = "interpreter_return_flag"
                    replan_value = str(getattr(ret_val, "value", "") or "")
                    if "interrupted for replan" in replan_value:
                        replan_source = "collision_threshold_callback"
                    elif "High-level skill" in replan_value:
                        replan_source = "high_level_skill_failure"
                    print_t(f"[REPLAN_DEBUG] source={replan_source} ret_val={replan_value}")
                    replan_attempts += 1
                    self.task_run_logger.update_execution_info(
                        execution_success=False,
                        failure_reason="replan_requested",
                        task_completed=False,
                    )
                    if replan_attempts > max_replan_attempts:
                        raise RuntimeError(f"Exceeded max replan attempts ({max_replan_attempts})")
                    self.append_message(f"[LOG] Replan requested, attempt={replan_attempts}")
                    self.execution_mode = "Planning"
                    continue
                self.task_run_logger.update_execution_info(
                    execution_success=execution_success,
                    task_completed=task_completed,
                )
            except Exception as e:
                self.execution_mode = "Yielding"
                error_message = f"[C] Error: {e}"
                print_t(error_message)
                self.append_message(error_message)
                self.task_run_logger.update_plan_info(self.current_plan or "", generation_success=bool(self.current_plan))
                self.task_run_logger.update_execution_info(
                    execution_success=False,
                    failure_reason=str(e),
                    task_completed=False,
                )
                self.task_run_logger.end_run(run_status="exception", failure_reason=str(e))
                monitor_stop.set()
                if monitor_thread is not None:
                    monitor_thread.join(timeout=1.0)
                self.append_message(f'\n[Task ended]')
                self.append_message('end')
                self.current_plan = None
                self.execution_history = None
                self.execution_mode = "Waiting"
                return

            break
        self.execution_mode = "Completed"
        self.task_run_logger.consume_runtime_snapshot(self.get_live_ui_snapshot())
        self.task_run_logger.end_run(run_status="completed")
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        self.append_message(f'\n[Task ended]')
        self.append_message('end')
        self.current_plan = None
        self.execution_history = None
        self.execution_mode = "Waiting"

    def get_active_scenario_name(self) -> str:
        return self.scenario_manager.selected_name()

    def get_baseline_scene(self) -> BaselineScene:
        return BASELINE_SCENES[self.baseline_scene_id]

    def set_baseline_scene(self, scene_id: str) -> str:
        self.baseline_scene_id = normalize_baseline_scene_id(scene_id)
        return self.baseline_scene_id

    def apply_baseline_scene(self):
        scene = self.get_baseline_scene()
        provider = getattr(self, "state_provider", None)
        drone = getattr(self, "drone", None)
        repositioned = False
        if drone is not None and hasattr(drone, "reposition_for_scenario"):
            repositioned = bool(drone.reposition_for_scenario(scene))
        if provider is not None and hasattr(provider, "lock_user_position"):
            provider.lock_user_position(True, reason=f"baseline_scene:{scene.id}")
        if provider is not None and hasattr(provider, "set_user_position"):
            ux, uy, uz = scene.user_position
            provider.set_user_position(float(ux), float(uy), float(uz), source=f"baseline_scene:{scene.id}")
        self.baseline_scene_state = {
            "scene_id": scene.id,
            "drone_initial_pose": tuple(float(v) for v in scene.drone_initial_pose),
            "drone_initial_yaw_rad": float(scene.drone_initial_yaw_rad),
            "user_position": tuple(float(v) for v in scene.user_position),
            "repositioned": repositioned,
            "notes": scene.notes,
            "captured_at": time.time(),
        }
        self.user_heading_yaw_rad = float(scene.user_initial_yaw_rad)
        return self.baseline_scene_state

    def get_baseline_scene_state(self):
        return self.baseline_scene_state

    def get_user_heading_yaw(self) -> float:
        return float(self.user_heading_yaw_rad)

    def set_user_heading_yaw(self, yaw_rad: float):
        self.user_heading_yaw_rad = float(yaw_rad)
        return self.user_heading_yaw_rad

    def turn_user_heading(self, delta_deg: float):
        self.user_heading_yaw_rad = float(self.user_heading_yaw_rad + math.radians(float(delta_deg)))
        return self.user_heading_yaw_rad

    def set_active_scenario(self, scenario_name: str):
        scenario = self.scenario_manager.select(scenario_name)
        return scenario.name

    def apply_selected_scenario(self):
        report = self.scenario_manager.apply_to_runtime(self)
        self.latest_scenario_report = report
        self.initial_scenario_state = {
            "selected_mode": report.selected_mode,
            "target_drone_gt": report.target_drone_position_3d,
            "target_user_gt": report.target_user_position_3d,
            "actual_drone_gt": report.actual_drone_gt_position_3d,
            "actual_user_gt": report.actual_user_gt_position_3d,
            "safety_score": report.measured_initial_safety_score,
            "current_collision_probability": report.measured_initial_collision_probability,
            "envelope_gap_m": report.measured_initial_envelope_gap_m,
            "uncertainty_scale_m": report.measured_initial_uncertainty_scale_m,
            "repositioned": report.repositioned,
            "calibration_iterations": report.calibration_iterations,
            "captured_at": time.time(),
        }
        # Force one immediate safety refresh after scenario apply.
        try:
            now = time.time()
            if hasattr(self.state_provider, "flush_due_packets"):
                self.state_provider.flush_due_packets(now=now)
        except Exception:
            pass
        return report

    def get_initial_scenario_state(self):
        return self.initial_scenario_state

    def get_scenario_projection(self):
        baseline_uncertainty = 0.85
        snapshot = self.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if snapshot else None
        if safety_context is not None:
            baseline_uncertainty = float(safety_context.uncertainty_scale_m)
        return self.scenario_manager.projected_assessment(baseline_uncertainty_scale_m=baseline_uncertainty)

    def get_scenario_runtime_status(self):
        report = self.latest_scenario_report
        snapshot = self.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if snapshot else None
        return {
            "selected_mode": None if report is None else report.selected_mode,
            "target_drone_gt": None if report is None else report.target_drone_position_3d,
            "target_user_gt": None if report is None else report.target_user_position_3d,
            "actual_drone_gt": snapshot.get("drone_gt") if snapshot else None,
            "actual_user_gt": snapshot.get("user_gt") if snapshot else None,
            "current_collision_probability": None if safety_context is None else float(safety_context.current_collision_probability),
            "historical_max_collision_probability": None if safety_context is None else float(safety_context.historical_max_collision_probability),
            "safety_score": None if safety_context is None else float(safety_context.safety_score),
            "envelope_gap_m": None if safety_context is None else float(safety_context.envelope_gap_m),
            "uncertainty_scale_m": None if safety_context is None else float(safety_context.uncertainty_scale_m),
        }

    def move_user_world(self, dx: float, dy: float, dz: float = 0.0):
        provider = getattr(self, "state_provider", None)
        if provider is None or not hasattr(provider, "get_user_position") or not hasattr(provider, "set_user_position"):
            return None
        if hasattr(provider, "lock_user_position"):
            provider.lock_user_position(True, reason="ui_manual_user_move")
        ux, uy, uz = provider.get_user_position()
        nx = float(ux + dx)
        ny = float(uy + dy)
        nz = float(uz + dz)
        provider.set_user_position(nx, ny, nz, source="ui_manual_user_move")
        if hasattr(provider, "flush_due_packets"):
            provider.flush_due_packets(now=time.time())
        return (nx, ny, nz)

    def _debug_log_safety_context(self, safety_context):
        if safety_context is None:
            print_debug("[SAFETY] unavailable")
            return
        print_debug(
            "[SAFETY]\n"
            f"  distance_xy={safety_context.drone_to_user_distance_xy:.3f}\n"
            f"  gap={safety_context.envelope_gap_m:.3f}\n"
            f"  uncertainty={safety_context.uncertainty_scale_m:.3f}\n"
            f"  overlap={safety_context.envelopes_overlap}\n"
            f"  score={safety_context.safety_score:.3f}\n"
            f"  current_collision_probability={safety_context.current_collision_probability:.6f}\n"
            f"  historical_max_collision_probability={safety_context.historical_max_collision_probability:.6f}\n"
            f"  standoff={safety_context.preferred_standoff_m:.3f}\n"
            f"  dominant_threat={safety_context.dominant_threat_type}:{safety_context.dominant_threat_id}\n"
            f"  dominant_gap={safety_context.dominant_gap_m:.3f}\n"
            f"  dominant_uncertainty={safety_context.dominant_uncertainty_scale_m:.3f}\n"
            f"  reasons={safety_context.reason_tags}"
        )

    def _simulate_obstacle_returns(self, obstacle_states, now: float):
        _ = now
        return list(obstacle_states or [])

    def _build_dominant_threat_context(
        self,
        safety_state,
        obstacle_states,
        scene,
        now: float,
        candidate_targets_summary=None,
        candidate_path_summaries=None,
    ):
        if safety_state is None:
            return None

        worker_packets = [("user", safety_state.user_packet)]
        for obs in obstacle_states or []:
            worker_packets.append((str(obs.id), obs.localization_packet))

        assessed_context = self.safety_assessor.build_from_packets(
            drone_packet=safety_state.drone_packet,
            worker_packets=worker_packets,
            now=now,
            safety_state=safety_state,
        )

        task_points_summary = []
        for point in scene.task_points if scene is not None else []:
            task_points_summary.append({"id": point.id, "x": float(point.x), "y": float(point.y), "z": float(point.z)})
        obstacles_summary = []
        for obs in obstacle_states or []:
            obs_packet = obs.localization_packet
            obstacles_summary.append(
                {
                    "id": obs.id,
                    "est_x": float(obs_packet.estimated_position_3d[0]),
                    "est_y": float(obs_packet.estimated_position_3d[1]),
                    "major_axis_m": float(obs.envelope.major_axis_radius),
                    "minor_axis_m": float(obs.envelope.minor_axis_radius),
                    "orientation_deg": float(obs.envelope.orientation_deg),
                }
            )
        assessed_context.task_points_summary = task_points_summary
        assessed_context.obstacles_summary = obstacles_summary
        assessed_context.path_summary = None
        assessed_context.candidate_targets_summary = candidate_targets_summary or []
        assessed_context.candidate_path_summaries = candidate_path_summaries or []
        return assessed_context

    def get_live_ui_snapshot(self):
        provider = getattr(self, "state_provider", None)
        if provider is None:
            return {}

        def _as_position_tuple(value):
            if value is None:
                return None
            return tuple(float(v) for v in value)

        def _call_position(source, attr):
            fn = getattr(source, attr, None)
            if callable(fn):
                value = fn()
                if value is not None:
                    return _as_position_tuple(value)
            return None

        now = time.time()
        if hasattr(provider, "flush_due_packets"):
            provider.flush_due_packets(now=now)
        safety_state = provider.get_latest_gcs_safety_state(now=now) if hasattr(provider, "get_latest_gcs_safety_state") else None
        safety_context = None
        if safety_state is not None:
            safety_context = self.safety_assessor.build_from_safety_state(safety_state, now=now)
        elif hasattr(provider, "get_latest_safety_context"):
            safety_context = provider.get_latest_safety_context(now=now)
        else:
            safety_context = self.latest_safety_context

        drone_gt = (
            _as_position_tuple(getattr(getattr(provider, "_cache", None), "position", None))
            or _call_position(self.drone, "get_ground_truth_drone_position")
            or _call_position(provider, "get_ground_truth_drone_position")
            or _call_position(self.drone, "get_drone_position")
            or _call_position(provider, "get_drone_position")
            or (0.0, 0.0, 0.0)
        )
        user_gt = (
            _call_position(provider, "get_user_position")
            or _as_position_tuple(getattr(provider, "_user_position", None))
            or _call_position(provider, "get_ground_truth_user_position")
            or (0.0, 0.0, 0.0)
        )

        drone_est = _call_position(self.drone, "get_estimated_drone_position")
        if drone_est is None:
            drone_packet = provider.get_latest_received_drone_packet() if hasattr(provider, "get_latest_received_drone_packet") else None
            drone_est = None if drone_packet is None else _as_position_tuple(drone_packet.estimated_position_3d)
        else:
            drone_packet = provider.get_latest_received_drone_packet() if hasattr(provider, "get_latest_received_drone_packet") else None

        user_est = _call_position(provider, "get_estimated_user_position")
        if user_est is None:
            user_packet = provider.get_latest_received_user_packet() if hasattr(provider, "get_latest_received_user_packet") else None
            user_est = None if user_packet is None else _as_position_tuple(user_packet.estimated_position_3d)
        else:
            user_packet = provider.get_latest_received_user_packet() if hasattr(provider, "get_latest_received_user_packet") else None

        baseline_state = self.baseline_scene_state or {}
        scene_start_ts = float(baseline_state.get("captured_at", now))
        elapsed_scene_s = max(0.0, now - scene_start_ts)
        obstacle_states_generated = compute_obstacle_envelope_states(self.get_baseline_scene(), now_s=elapsed_scene_s)
        obstacle_states = self._simulate_obstacle_returns(obstacle_states_generated, now=now)
        user_heading = float(self.get_user_heading_yaw())
        user_ref = user_est or user_gt
        right_offset_m = 1.0
        right_dx = math.sin(user_heading) * right_offset_m
        right_dy = -math.cos(user_heading) * right_offset_m
        left_dx = -right_dx
        left_dy = -right_dy
        candidate_targets = []
        for point in self.get_baseline_scene().task_points:
            candidate_targets.append({"id": point.id, "x": float(point.x), "y": float(point.y), "z": float(point.z)})
        candidate_targets.append(
            {
                "id": "user",
                "x": float(user_ref[0]),
                "y": float(user_ref[1]),
                "z": float(user_ref[2]),
            }
        )
        candidate_targets.append(
            {
                "id": "user_right_side",
                "x": float(user_ref[0] + right_dx),
                "y": float(user_ref[1] + right_dy),
                "z": float(user_ref[2]),
            }
        )
        candidate_targets.append(
            {
                "id": "user_left_side",
                "x": float(user_ref[0] + left_dx),
                "y": float(user_ref[1] + left_dy),
                "z": float(user_ref[2]),
            }
        )
        candidate_path_summaries = []
        for target in candidate_targets:
            path = self._compute_path_eval_for_target(
                self.get_baseline_scene(),
                drone_est or drone_gt,
                user_est or user_gt,
                target_xy=(target["x"], target["y"]),
                obstacle_envelopes=obstacle_states,
                user_envelope=(None if safety_state is None else safety_state.user_envelope),
            )
            candidate_path_summaries.append(
                {
                    "target_id": target["id"],
                    "path_clear": bool(path.path_clear),
                    "blocking_entity": str(path.blocking_entity),
                    "corridor_min_gap": float(path.corridor_min_gap),
                }
            )
        dominant_safety_context = self._build_dominant_threat_context(
            safety_state=safety_state,
            obstacle_states=obstacle_states,
            scene=self.get_baseline_scene(),
            now=now,
            candidate_targets_summary=candidate_targets,
            candidate_path_summaries=candidate_path_summaries,
        )
        if dominant_safety_context is not None:
            dominant_safety_context.path_summary = None
            safety_context = dominant_safety_context
        elif drone_packet is not None:
            # Fallback: when provider-level safety_state is unavailable, still
            # compute collision probability with UAV + user/worker localization packets.
            worker_packets = []
            if user_packet is not None:
                worker_packets.append(("user", user_packet))
            worker_packets.extend((str(obs.id), obs.localization_packet) for obs in obstacle_states)
            if worker_packets:
                safety_context = self.safety_assessor.build_from_packets(
                    drone_packet=drone_packet,
                    worker_packets=worker_packets,
                    now=now,
                    safety_state=None,
                )
        drone_bias_xy = None if drone_packet is None else tuple(float(v) for v in drone_packet.b_xy[:2])
        user_bias_xy = None if user_packet is None else tuple(float(v) for v in user_packet.b_xy[:2])
        drone_corrected = None if drone_est is None else (
            float(drone_est[0] - (0.0 if drone_bias_xy is None else drone_bias_xy[0])),
            float(drone_est[1] - (0.0 if drone_bias_xy is None else drone_bias_xy[1])),
            float(drone_est[2]),
        )
        user_corrected = None if user_est is None else (
            float(user_est[0] - (0.0 if user_bias_xy is None else user_bias_xy[0])),
            float(user_est[1] - (0.0 if user_bias_xy is None else user_bias_xy[1])),
            float(user_est[2]),
        )

        snapshot = {
            "drone_gt": drone_gt,
            "drone_est": drone_est,
            "drone_est_bias_corrected": drone_corrected,
            "drone_est_raw": drone_est,
            "drone_bias_xy": drone_bias_xy,
            "drone_P_xy": (None if drone_packet is None else np.asarray(drone_packet.P_xy, dtype=float).copy()),
            "user_gt": user_gt,
            "user_est": user_est,
            "user_est_bias_corrected": user_corrected,
            "user_est_raw": user_est,
            "user_bias_xy": user_bias_xy,
            "user_P_xy": (None if user_packet is None else np.asarray(user_packet.P_xy, dtype=float).copy()),
            "safety_state": safety_state,
            "safety_context": safety_context,
            "drone_yaw_rad": self._get_drone_yaw_rad(),
            "user_heading_yaw_rad": self.get_user_heading_yaw(),
            "baseline_scene_id": self.baseline_scene_id,
            "baseline_scene": self.get_baseline_scene(),
            "baseline_scene_state": self.baseline_scene_state,
            "obstacle_envelope_states": obstacle_states,
            "workers": [
                {
                    "id": str(obs.id),
                    "gt_xy": tuple(float(v) for v in obs.gt_xy),
                    "est_xy_raw": (
                        float(obs.localization_packet.estimated_position_3d[0]),
                        float(obs.localization_packet.estimated_position_3d[1]),
                    ),
                    "bias_xy": tuple(float(v) for v in obs.localization_packet.b_xy[:2]),
                    "est_xy_bias_corrected": (
                        float(obs.localization_packet.estimated_position_3d[0] - obs.localization_packet.b_xy[0]),
                        float(obs.localization_packet.estimated_position_3d[1] - obs.localization_packet.b_xy[1]),
                    ),
                    "P_xy": np.asarray(obs.localization_packet.P_xy, dtype=float).copy(),
                }
                for obs in obstacle_states
            ],
            "candidate_targets": candidate_targets,
            "candidate_path_summaries": candidate_path_summaries,
            "baseline_decision": self.latest_baseline_decision,
            "framework_name": "TypeFly baseline",
            "mode_name": self.get_active_scenario_name(),
            "execution_mode": self.execution_mode,
            "active_objective_set": dict(self.active_objective_set),
            "benchmark_progress": dict(self.latest_benchmark_progress),
            "checkpoint_order": list(BENCHMARK_CHECKPOINT_ORDER),
            "benchmark_checkpoints": [
                {"id": cp.id, "zone_id": cp.zone_id, "x": float(cp.x), "y": float(cp.y), "radius_m": float(cp.radius_m)}
                for cp in BENCHMARK_CHECKPOINTS
            ],
            "benchmark_zones": [
                {"id": zone.id, "x_range": zone.x_range, "y_range": zone.y_range, "label_xy": zone.label_xy}
                for zone in BENCHMARK_ZONES
            ],
            "original_planned_path": None,
            "updated_path": None,
        }
        if safety_state is not None and safety_context is not None:
            consistency_from_gap = bool(float(safety_state.envelope_gap_m) < 0.0)
            print_debug(
                "[UI-SAFETY-SNAPSHOT] "
                f"gap={safety_context.envelope_gap_m:.6f} "
                f"overlap={safety_context.envelopes_overlap} "
                f"overlap_from_gap={consistency_from_gap} "
                f"drone_center={tuple(float(v) for v in safety_state.drone_center_xy)} "
                f"user_center={tuple(float(v) for v in safety_state.user_center_xy)} "
                f"drone_radius={safety_state.drone_radius_along_user_direction:.6f} "
                f"user_radius={safety_state.user_radius_along_drone_direction:.6f} "
                f"score={safety_context.safety_score:.6f} "
                f"current_p={safety_context.current_collision_probability:.6f} "
                f"reason_tags={safety_context.reason_tags}"
            )
        self._debug_log_obstacle_envelopes(snapshot.get("obstacle_envelope_states"))
        self._debug_log_localization_pipeline_comparison(snapshot)
        print_debug(
            "[UI-SNAPSHOT] "
            f"drone_gt={snapshot['drone_gt']} drone_est={snapshot['drone_est']} "
            f"user_gt={snapshot['user_gt']} user_est={snapshot['user_est']}"
        )
        return snapshot

    def _get_drone_yaw_rad(self) -> float:
        get_yaw = getattr(self.drone, "get_drone_yaw", None)
        if callable(get_yaw):
            try:
                return float(get_yaw())
            except Exception:
                pass
        if hasattr(self.drone, "rotation_accumulator"):
            try:
                return math.radians(float(self.drone.rotation_accumulator))
            except Exception:
                pass
        return 0.0

    def _extract_target_task_point(self, task_text: str) -> str:
        text = (task_text or "").upper()
        for token in ("A", "B", "C"):
            if f" {token}" in f" {text}" or f"{token} " in text or f"TO {token}" in text:
                return token
        return "A"

    def _compute_path_eval(self, scene: BaselineScene, drone_pos, user_pos, now_s: float = 0.0, obstacle_envelopes=None, user_envelope=None):
        target_id = "A"
        if isinstance(self.latest_baseline_decision, dict):
            target_id = str(self.latest_baseline_decision.get("target_task_point") or "A")
        point = get_task_point(scene, target_id) or scene.task_points[0]
        safety_context = None
        snapshot = getattr(self, "latest_safety_context", None)
        if snapshot is not None:
            safety_context = snapshot
        user_radius = 0.75
        if safety_context is not None:
            user_radius = max(0.45, float(safety_context.uncertainty_scale_m) * 0.5)
        obstacle_envelopes = obstacle_envelopes if obstacle_envelopes is not None else compute_obstacle_envelope_states(scene, now_s=now_s)
        return evaluate_path_clear(
            drone_xy=(float(drone_pos[0]), float(drone_pos[1])),
            target_xy=(float(point.x), float(point.y)),
            user_xy=(None if user_pos is None else (float(user_pos[0]), float(user_pos[1]))),
            user_radius_m=user_radius,
            user_envelope=user_envelope,
            obstacle_envelopes=obstacle_envelopes,
            corridor_half_width_m=0.35,
        )

    def _compute_path_eval_for_target(self, scene: BaselineScene, drone_pos, user_pos, target_xy, obstacle_envelopes=None, user_envelope=None):
        safety_context = getattr(self, "latest_safety_context", None)
        user_radius = 0.75
        if safety_context is not None:
            user_radius = max(0.45, float(getattr(safety_context, "uncertainty_scale_m", 1.0)) * 0.5)
        return evaluate_path_clear(
            drone_xy=(float(drone_pos[0]), float(drone_pos[1])),
            target_xy=(float(target_xy[0]), float(target_xy[1])),
            user_xy=(None if user_pos is None else (float(user_pos[0]), float(user_pos[1]))),
            user_radius_m=user_radius,
            user_envelope=user_envelope,
            obstacle_envelopes=obstacle_envelopes or [],
            corridor_half_width_m=0.35,
        )

    def _build_baseline_control_plan(self, task_description: str, snapshot):
        scene = self.get_baseline_scene()
        target_id = self._extract_target_task_point(task_description)
        target = get_task_point(scene, target_id)
        if target is None:
            return None
        drone_pos = snapshot.get("drone_est") or snapshot.get("drone_gt")
        user_pos = snapshot.get("user_est") or snapshot.get("user_gt")
        if drone_pos is None:
            return None
        safety_context = snapshot.get("safety_context")
        obstacle_envelopes = snapshot.get("obstacle_envelope_states")
        if obstacle_envelopes is None:
            obstacle_envelopes = compute_obstacle_envelope_states(scene, now_s=time.time())
        path_eval = self._compute_path_eval(
            scene,
            drone_pos,
            user_pos,
            now_s=time.time(),
            obstacle_envelopes=obstacle_envelopes,
            user_envelope=(None if snapshot.get("safety_state") is None else snapshot.get("safety_state").user_envelope),
        )
        risk_high = bool(
            safety_context is not None
            and float(safety_context.current_collision_probability) >= COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD
        )
        direct_go_to = bool(path_eval.path_clear and not risk_high)
        if direct_go_to:
            plan = "mf(1.0);d(0.2);mf(0.8);d(0.2);"
            mode = "direct_staged"
        else:
            side = "left"
            if path_eval.blocking_entity.startswith("O"):
                obstacle = next((o for o in obstacle_envelopes if o.id == path_eval.blocking_entity), None)
                if obstacle is not None:
                    side = "left" if float(obstacle.est_xy[1]) <= float(drone_pos[1]) else "right"
            elif path_eval.blocking_entity == "user" and user_pos is not None:
                side = "left" if float(user_pos[1]) <= float(drone_pos[1]) else "right"
            if side == "left":
                plan = "tu(35);mf(0.9);tc(35);mf(0.9);d(0.2);"
            else:
                plan = "tc(35);mf(0.9);tu(35);mf(0.9);d(0.2);"
            mode = "staged_detour"
        note = (
            f"target={target.id}; path_clear={path_eval.path_clear}; "
            f"blocked_by={path_eval.blocking_entity}; direct_go_to={direct_go_to}; chosen={mode}"
        )
        obstacle_summary = [
            {
                "id": obs.id,
                "gt_xy": [float(obs.gt_xy[0]), float(obs.gt_xy[1])],
                "est_xy": [float(obs.est_xy[0]), float(obs.est_xy[1])],
                "matrix_xy": obs.matrix_xy,
                "envelope_major_axis_m": float(obs.envelope.major_axis_radius),
                "envelope_minor_axis_m": float(obs.envelope.minor_axis_radius),
                "orientation_deg": float(obs.envelope.orientation_deg),
            }
            for obs in obstacle_envelopes
        ]
        task_points_summary = [
            {"id": pt.id, "x": float(pt.x), "y": float(pt.y), "z": float(pt.z)}
            for pt in scene.task_points
        ]
        return {
            "scene_id": scene.id,
            "target_task_point": target.id,
            "task_text": task_description,
            "obstacle_positions_summary": obstacle_summary,
            "task_point_positions_summary": task_points_summary,
            "path_clear": bool(path_eval.path_clear),
            "blocking_entity": path_eval.blocking_entity,
            "corridor_min_gap": float(path_eval.corridor_min_gap),
            "chosen_motion_mode": mode,
            "direct_go_to": direct_go_to,
            "staged_detour": not direct_go_to,
            "recovery_first": False,
            "unknown": False,
            "generated_control_plan": plan,
            "decision_note": note,
            "plan": plan,
        }

    def get_baseline_expectation_summary(self, safety_context=None):
        scene = self.get_baseline_scene()
        risk_high = bool(
            safety_context is not None
            and float(safety_context.current_collision_probability) >= COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD
        )
        user_radius = max(0.45, float(getattr(safety_context, "uncertainty_scale_m", 1.0)) * 0.5)
        expectations = build_scene_expectations(
            scene=scene,
            user_radius_m=user_radius,
            corridor_half_width_m=0.35,
            high_risk=risk_high,
            now_s=time.time(),
        )
        return [
            {
                "scene_id": item.scene_id,
                "target_task_point": item.target_task_point,
                "expected_path_clear": item.expected_path_clear,
                "expected_blocking_entity": item.expected_blocking_entity,
                "expected_motion_mode": item.expected_motion_mode,
            }
            for item in expectations
        ]

    def get_all_scene_expectation_summary(self, safety_context=None):
        risk_high = bool(
            safety_context is not None
            and float(safety_context.current_collision_probability) >= COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD
        )
        user_radius = max(0.45, float(getattr(safety_context, "uncertainty_scale_m", 1.0)) * 0.5)
        expectations = build_all_scene_expectations(
            user_radius_m=user_radius,
            corridor_half_width_m=0.35,
            high_risk=risk_high,
            now_s=time.time(),
        )
        return [
            {
                "scene_id": item.scene_id,
                "target_task_point": item.target_task_point,
                "expected_path_clear": item.expected_path_clear,
                "expected_blocking_entity": item.expected_blocking_entity,
                "expected_motion_mode": item.expected_motion_mode,
            }
            for item in expectations
        ]

    def _debug_log_obstacle_envelopes(self, obstacle_states):
        if not obstacle_states:
            return
        parts = []
        for obs in obstacle_states:
            packet = obs.localization_packet
            parts.append(
                f"{obs.id}:gt=({obs.gt_xy[0]:.2f},{obs.gt_xy[1]:.2f}) "
                f"est=({obs.est_xy[0]:.2f},{obs.est_xy[1]:.2f}) "
                f"matrix={obs.matrix_xy} "
                f"axes=({obs.envelope.major_axis_radius:.3f},{obs.envelope.minor_axis_radius:.3f}) "
                f"ori={obs.envelope.orientation_deg:.1f}"
            )
        print_debug("[BASELINE-OBS] " + " | ".join(parts))

    def _build_envelope_audit_summary(self, snapshot):
        safety_state = snapshot.get("safety_state")
        scene = snapshot.get("baseline_scene")
        obstacle_states = snapshot.get("obstacle_envelope_states") or []
        result = {"entities": []}
        if safety_state is None:
            return result

        drone_matrix = getattr(safety_state.drone_packet, "M_xy", None)
        user_matrix = getattr(safety_state.user_packet, "M_xy", None)
        result["entities"].append(
            {
                "id": "drone",
                "gt": snapshot.get("drone_gt"),
                "est": snapshot.get("drone_est"),
                "matrix_xy": drone_matrix,
                "matrix_source": "provider_packet.M_xy",
                "called_function": "build_safety_envelope",
                "localization_pipeline_function": "IterativeLeastSquaresEstimator3D.estimate",
                "base_sigma": "packet covariance",
                "nominal_size_used": False,
                "bias_used": False,
                "extra_inflation": "none",
                "chi2": getattr(safety_state.drone_envelope, "chi2_val", None),
                "major_axis": float(safety_state.drone_envelope.major_axis_radius),
                "minor_axis": float(safety_state.drone_envelope.minor_axis_radius),
                "orientation_deg": float(safety_state.drone_envelope.orientation_deg),
            }
        )
        result["entities"].append(
            {
                "id": "user",
                "gt": snapshot.get("user_gt"),
                "est": snapshot.get("user_est"),
                "matrix_xy": user_matrix,
                "matrix_source": "provider_packet.M_xy",
                "called_function": "build_safety_envelope",
                "localization_pipeline_function": "IterativeLeastSquaresEstimator3D.estimate",
                "base_sigma": "packet covariance",
                "nominal_size_used": False,
                "bias_used": False,
                "extra_inflation": "none",
                "chi2": getattr(safety_state.user_envelope, "chi2_val", None),
                "major_axis": float(safety_state.user_envelope.major_axis_radius),
                "minor_axis": float(safety_state.user_envelope.minor_axis_radius),
                "orientation_deg": float(safety_state.user_envelope.orientation_deg),
            }
        )
        obstacles_by_id = {}
        if scene is not None:
            obstacles_by_id = {obs.id: obs for obs in scene.obstacles}
        for obs_state in obstacle_states:
            base = obstacles_by_id.get(obs_state.id)
            result["entities"].append(
                {
                    "id": obs_state.id,
                    "gt": [float(obs_state.gt_xy[0]), float(obs_state.gt_xy[1])],
                    "est": [float(obs_state.est_xy[0]), float(obs_state.est_xy[1])],
                    "matrix_xy": obs_state.matrix_xy,
                    "matrix_source": "obstacle_state_packet.M_xy (consumed by build_safety_envelope)",
                    "called_function": obs_state.envelope_builder_function,
                    "localization_pipeline_function": obs_state.localization_pipeline_function,
                    "base_sigma": "packet covariance (anchor-based localization output)",
                    "nominal_size_used": False,
                    "bias_used": None if base is None else "range-bias model via LocalizationErrorModel",
                    "extra_inflation": "none (single chi2 expansion)",
                    "chi2": float(obs_state.envelope.chi2_val),
                    "major_axis": float(obs_state.envelope.major_axis_radius),
                    "minor_axis": float(obs_state.envelope.minor_axis_radius),
                    "orientation_deg": float(obs_state.envelope.orientation_deg),
                }
            )

        drone_major = float(safety_state.drone_envelope.major_axis_radius)
        user_major = float(safety_state.user_envelope.major_axis_radius)
        ratios = {}
        for obs_state in obstacle_states:
            ratios[f"{obs_state.id}_to_drone_major"] = float(obs_state.envelope.major_axis_radius / max(1e-6, drone_major))
            ratios[f"{obs_state.id}_to_user_major"] = float(obs_state.envelope.major_axis_radius / max(1e-6, user_major))
        result["ratios"] = ratios
        return result

    def _debug_log_envelope_audit(self, audit):
        if not audit or not audit.get("entities"):
            return
        lines = []
        for entity in audit["entities"]:
            lines.append(
                f"{entity['id']}: gt={entity['gt']} est={entity['est']} "
                f"matrix={entity['matrix_xy']} source={entity['matrix_source']} "
                f"loc_fn={entity.get('localization_pipeline_function', 'unknown')} "
                f"env_fn={entity.get('called_function', 'build_safety_envelope(via GCS service)')} "
                f"base_sigma={entity['base_sigma']} nominal_size_used={entity['nominal_size_used']} "
                f"bias={entity['bias_used']} inflate={entity['extra_inflation']} "
                f"chi2={entity['chi2']} major={entity['major_axis']:.3f} "
                f"minor={entity['minor_axis']:.3f} ori={entity['orientation_deg']:.2f}"
            )
        ratio_text = ", ".join(f"{k}={v:.3f}" for k, v in sorted((audit.get("ratios") or {}).items()))
        print_debug("[ENVELOPE-AUDIT] " + " || ".join(lines))
        if ratio_text:
            print_debug("[ENVELOPE-RATIOS] " + ratio_text)

    def _fmt_array_debug(self, arr):
        values = np.asarray(arr, dtype=float).reshape(-1)
        return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"

    def _debug_log_localization_pipeline_comparison(self, snapshot):
        safety_state = snapshot.get("safety_state")
        if safety_state is None:
            return
        user_packet = getattr(safety_state, "user_packet", None)
        obstacle_states = snapshot.get("obstacle_envelope_states") or []
        if user_packet is None:
            return

        lines = [
            "[PIPELINE-CHECK] USER",
            f"  GT position: ({float(user_packet.gt_position_3d[0]):.3f}, {float(user_packet.gt_position_3d[1]):.3f}, {float(user_packet.gt_position_3d[2]):.3f})",
            f"  simulated anchor measurements: {self._fmt_array_debug(user_packet.measured_ranges)}",
            f"  estimated position: ({float(user_packet.estimated_position_3d[0]):.3f}, {float(user_packet.estimated_position_3d[1]):.3f}, {float(user_packet.estimated_position_3d[2]):.3f})",
            f"  localization covariance-like M_xy: {np.asarray(user_packet.M_xy, dtype=float).tolist()}",
            "  localization pipeline function: IterativeLeastSquaresEstimator3D.estimate",
            "  envelope builder function: build_safety_envelope",
            (
                "  final major/minor/orientation: "
                f"{float(safety_state.user_envelope.major_axis_radius):.3f}/"
                f"{float(safety_state.user_envelope.minor_axis_radius):.3f}/"
                f"{float(safety_state.user_envelope.orientation_deg):.3f}"
            ),
        ]
        print_debug("\n".join(lines))

        for obs in obstacle_states:
            packet = obs.localization_packet
            obs_lines = [
                f"[PIPELINE-CHECK] {obs.id}",
                f"  GT position: ({float(packet.gt_position_3d[0]):.3f}, {float(packet.gt_position_3d[1]):.3f}, {float(packet.gt_position_3d[2]):.3f})",
                f"  simulated anchor measurements: {self._fmt_array_debug(packet.measured_ranges)}",
                f"  estimated position: ({float(packet.estimated_position_3d[0]):.3f}, {float(packet.estimated_position_3d[1]):.3f}, {float(packet.estimated_position_3d[2]):.3f})",
                f"  localization covariance-like M_xy: {np.asarray(packet.M_xy, dtype=float).tolist()}",
                f"  localization pipeline function: {obs.localization_pipeline_function}",
                f"  envelope builder function: {obs.envelope_builder_function}",
                (
                    "  final major/minor/orientation: "
                    f"{float(obs.envelope.major_axis_radius):.3f}/"
                    f"{float(obs.envelope.minor_axis_radius):.3f}/"
                    f"{float(obs.envelope.orientation_deg):.3f}"
                ),
            ]
            print_debug("\n".join(obs_lines))

    def start_robot(self):
        print_t("[C] Connecting to robot...")

        # Start state provider first so PX4_SIM wrapper can immediately consume live state.
        self.start_uwb()
        self.drone.connect()
        print_t("[C] Starting robot...")

        # Start state provider before PX4_SIM takeoff so wrapper has live sim state.
        self.start_uwb()

        if self.robot_type != RobotType.PX4_SIM:
            self.drone.takeoff()
            self.drone.move_up(0.25)
        self.apply_selected_scenario()

        if self.enable_video:
            print_t("[C] Starting stream...")
            self.drone.start_stream()
        if self.robot_type != RobotType.PX4_SIM:
            print_t("[C] Starting virtual position loop...")
            self.start_virtual_position_loop()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        if self.enable_video:
            self.drone.stop_stream()
        print_t("[C] Stopping UWB tracking...")
        self.stop_uwb()
        if self.robot_type != RobotType.PX4_SIM:
            print_t("[C] Stopping virtual position loop...")
            self.stop_virtual_position_loop()
        self.controller_wait_takeoff = True

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")
        frame_reader = self.drone.get_frame_reader()
        
        if frame_reader is None:
            print_t("[WARN] frame_reader is None, skipping capture loop")
            return
        
        while self.controller_active:
            self.drone.keep_active()
            if frame_reader is None or not hasattr(frame_reader, 'frame'):
                time.sleep(0.1)
                continue
            
            self.latest_frame = frame_reader.frame
            frame = Frame(frame_reader.frame,
                          frame_reader.depth if hasattr(frame_reader, 'depth') else None)
            
            if self.yolo_client.is_local_service():
                self.yolo_client.detect_local(frame)
            else:
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(frame))
            
            time.sleep(0.10)
            
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        self.drone.stop_stream()
        self.drone.land()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")
