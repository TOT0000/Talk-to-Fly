from PIL import Image
import queue, time, os, json, sys, subprocess
from typing import Optional, Tuple
import asyncio
import uuid
import threading

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
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_debug, print_t
from .minispec_interpreter import MiniSpecInterpreter, Statement
from .abs.robot_wrapper import RobotType
from .uwb_wrapper import UWBWrapper
from .state_provider import StateProvider, UwbStateProvider
from .sim_state_provider import SimStateProvider

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

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

      # self.low_level_skillset.add_skill(LowLevelSkillItem("goto", self.skill_goto, "goto the object", args=[SkillArg("object_name[*x-value]", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("time", self.skill_time, "Get current execution time", args=[]))
      

        # load high-level skills
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'
        with open(os.path.join(CURRENT_DIR, f"assets/{type_folder_name}/high_level_skills copy.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

        self.current_plan = None
        self.execution_history = None
        self.execution_time = time.time()
        self.latest_safety_context = None

        # PX4_SIM optional managed user-position publisher lifecycle
        self._sim_user_publisher_proc: Optional[subprocess.Popen] = None
        self._owns_sim_user_publisher = False
        
    def register_position_callback(self, callback):
        self.position_update_callback = callback
        
    def notify_user_position_updated(self, position: Tuple[float, float, float, float]):
        timestamp, x, y, z = position
        position_str = f"User position updated: x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        if hasattr(self, 'position_update_callback') and self.position_update_callback:
            source = "user"
            self.position_update_callback(x, y, z, source)
    
    def skill_get_drone_position(self) -> Tuple[str, bool]:
        x, y, z = self.drone.get_drone_position()
        position_str = f"Drone position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        return position_str, False
        
    def skill_get_user_position(self) -> Tuple[str, bool]:
        x, y, z = self.state_provider.get_user_position()
        position_str = f"User position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
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

    def skill_goto(self, object_name: str) -> Tuple[None, bool]:
        print(f'Goto {object_name}')
        if '[' in object_name:
            x = float(object_name.split('[')[1].split(']')[0])
        else:
            x = self.vision.object_x(object_name)[0]

        print(f'>> GOTO x {x} {type(x)}')

        if x > 0.55:
            self.drone.turn_cw(int((x - 0.5) * 70))
        elif x < 0.45:
            self.drone.turn_ccw(int((0.5 - x) * 70))

        self.drone.move_forward(110)
        return None, False

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

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        self.controller_active = False

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            self.vision.update()
            YoloClient.plot_results_oi(image, self.vision.object_list)
        return image
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter(self.message_queue)
        interpreter.execute(minispec)
        self.execution_history = interpreter.execution_history
        ret_val = interpreter.ret_queue.get()
        return ret_val

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
        self.append_message('[TASK]: ' + task_description)
        ret_val = None
        while True:
            user_pos = self.state_provider.get_user_position() if hasattr(self, "state_provider") else (0.00, 0.00, 0.00)
            drone_pos = self.drone.get_drone_position() if hasattr(self, "drone") else (0.00, 0.00, 0.00)
            
            location_info = (
                f"Drone position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}\n"
                f"User position: x={user_pos[0]:.2f}, y={user_pos[1]:.2f}, z={user_pos[2]:.2f}"
            )

            scene_description = self.vision.get_obj_list() if self.enable_video else ''
            if hasattr(self.state_provider, "debug_log_latest_localization_snapshot"):
                self.state_provider.debug_log_latest_localization_snapshot(reason="pre-plan")
            safety_context = self.state_provider.get_latest_safety_context() if hasattr(self.state_provider, "get_latest_safety_context") else None
            if safety_context is None:
                safety_context = self.safety_assessor.build_from_provider(self.state_provider)
            self._debug_log_safety_context(safety_context)
            
            self.current_plan = self.planner.plan(
                task_description=task_description,
                scene_description=scene_description,
                location_info=location_info,
                execution_history=self.execution_history,
                safety_context=safety_context,
            )
            self.latest_safety_context = safety_context
            
            self.append_message(f'[Plan]: \\\\')
            try:
                self.execution_time = time.time()

                ret_val = self.execute_minispec(self.current_plan)
            except Exception as e:
                print_t(f"[C] Error: {e}")
            
            break
        self.append_message(f'\n[Task ended]')
        self.append_message('end')
        self.current_plan = None
        self.execution_history = None

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
            f"  level={safety_context.safety_level}\n"
            f"  bias={safety_context.planning_bias}\n"
            f"  standoff={safety_context.preferred_standoff_m:.3f}\n"
            f"  reasons={safety_context.reason_tags}"
        )

    def get_live_ui_snapshot(self):
        provider = getattr(self, "state_provider", None)
        if provider is None:
            return {}

        now = time.time()
        if hasattr(provider, "flush_due_packets"):
            provider.flush_due_packets(now=now)
        safety_state = provider.get_latest_gcs_safety_state(now=now) if hasattr(provider, "get_latest_gcs_safety_state") else None
        safety_context = provider.get_latest_safety_context(now=now) if hasattr(provider, "get_latest_safety_context") else self.latest_safety_context

        drone_gt = provider.get_ground_truth_drone_position() if hasattr(provider, "get_ground_truth_drone_position") else self.drone.get_drone_position()
        user_gt = provider.get_ground_truth_user_position() if hasattr(provider, "get_ground_truth_user_position") else provider.get_user_position()

        drone_packet = provider.get_latest_received_drone_packet() if hasattr(provider, "get_latest_received_drone_packet") else None
        user_packet = provider.get_latest_received_user_packet() if hasattr(provider, "get_latest_received_user_packet") else None
        drone_est = None if drone_packet is None else tuple(float(v) for v in drone_packet.estimated_position_3d)
        user_est = None if user_packet is None else tuple(float(v) for v in user_packet.estimated_position_3d)

        def _timing(packet):
            if packet is None:
                return None, None
            aoi_s = now - float(packet.state_generation_timestamp)
            delay_s = None
            if packet.received_packet_timestamp is not None:
                delay_s = float(packet.received_packet_timestamp - packet.state_generation_timestamp)
            return aoi_s, delay_s

        drone_aoi_s, drone_delay_s = _timing(drone_packet)
        user_aoi_s, user_delay_s = _timing(user_packet)

        snapshot = {
            "drone_gt": tuple(float(v) for v in drone_gt),
            "drone_est": None if drone_est is None else tuple(float(v) for v in drone_est),
            "user_gt": tuple(float(v) for v in user_gt),
            "user_est": None if user_est is None else tuple(float(v) for v in user_est),
            "drone_aoi_s": drone_aoi_s,
            "drone_delay_s": drone_delay_s,
            "user_aoi_s": user_aoi_s,
            "user_delay_s": user_delay_s,
            "safety_state": safety_state,
            "safety_context": safety_context,
        }
        print_debug(
            "[TRACE-UI-SNAPSHOT] "
            f"drone_gt={snapshot['drone_gt']} drone_est={snapshot['drone_est']} "
            f"user_gt={snapshot['user_gt']} user_est={snapshot['user_est']}"
        )
        return snapshot

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
