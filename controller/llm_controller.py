from dataclasses import dataclass
from PIL import Image
import queue, time, os, json, math, random
from typing import Optional, Tuple
import asyncio
import uuid
import threading

# 保留原本的引用
from .shared_frame import SharedFrame, Frame
from .yolo_client import YoloClient
from .yolo_grpc_client import YoloGRPCClient
from .tello_wrapper import TelloWrapper
from .virtual_robot_wrapper import VirtualRobotWrapper
from .abs.robot_wrapper import RobotWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_t
from .minispec_interpreter import MiniSpecInterpreter, Statement
from .abs.robot_wrapper import RobotType
from .uwb_wrapper import UWBWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 新增：設定檔管理類別 ---
@dataclass
class ControllerConfig:
    cache_dir: Optional[str] = None
    assets_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ControllerConfig":
        return cls(
            cache_dir=os.getenv("LLM_CONTROLLER_CACHE_DIR"),
            assets_dir=os.getenv("LLM_CONTROLLER_ASSETS_DIR"),
        )

    def resolve(self, base_dir: str) -> "ControllerConfig":
        return ControllerConfig(
            cache_dir=self.cache_dir or os.path.join(base_dir, "cache"),
            assets_dir=self.assets_dir or os.path.join(base_dir, "assets"),
        )

class LLMController():
    # --- 重構後的 __init__ ---
    def __init__(self, robot_type, virtual_queue, use_http=False, message_queue: Optional[queue.Queue]=None, enable_video=False, config: Optional[ControllerConfig] = None):
        self.virtual_queue = virtual_queue
        self.enable_video = enable_video
        self.shared_frame = SharedFrame()
        self.message_queue = message_queue
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.latest_virtual_user_position = (0.0, 0.0, 0.0)
        self.latest_drone_position = (0.0, 0.0, 0.0)
        
        # 1. 初始化設定 (路徑)
        self.config = self._init_config(config)
        self.cache_folder = self._resolve_cache_folder()
        
        # 2. 初始化視覺 (YOLO)
        self.yolo_client, self.vision = self._init_vision(use_http)
        self.latest_frame = None
        
        # 3. 初始化機器人與規劃器
        self.drone: RobotWrapper = self._init_robot(robot_type)
        self.planner = self._init_planner(robot_type)
        
        # 4. 初始化 UWB
        self._init_uwb()
        
        # 5. 載入技能 (Skills)
        self.low_level_skillset, self.high_level_skillset = self._init_skills(robot_type)
        
        # 綁定全域變數 (為了讓其他模組存取)
        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = self.high_level_skillset
        
        self.current_plan = None
        self.execution_history = None
        self.execution_time = time.time()

    # --- 新增的 Helper Methods (讓 __init__ 變乾淨) ---

    def _init_config(self, config: Optional[ControllerConfig]) -> ControllerConfig:
        env_config = ControllerConfig.from_env()
        if config is None:
            resolved_config = env_config.resolve(CURRENT_DIR)
        else:
            merged = ControllerConfig(
                cache_dir=config.cache_dir or env_config.cache_dir,
                assets_dir=config.assets_dir or env_config.assets_dir,
            )
            resolved_config = merged.resolve(CURRENT_DIR)
        return resolved_config

    def _resolve_cache_folder(self) -> str:
        cache_dir = self.config.cache_dir
        if self.message_queue is not None:
            try:
                # 嘗試從 Queue 獲取路徑 (相容舊邏輯)
                cache_dir = self.message_queue.get(timeout=1.0)
            except queue.Empty:
                pass
        
        cache_dir = cache_dir or os.path.join(CURRENT_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _init_vision(self, use_http: bool):
        if use_http:
            yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
        
        vision = VisionSkillWrapper(self.shared_frame, enabled=self.enable_video)
        return yolo_client, vision

    def _init_robot(self, robot_type) -> RobotWrapper:
        match robot_type:
            case RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                return TelloWrapper()
            case RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from .gear_wrapper import GearWrapper
                return GearWrapper()
            case _:
                print_t("[C] Start virtual drone...")
                return VirtualRobotWrapper(enable_video=self.enable_video)

    def _init_planner(self, robot_type):
        planner = LLMPlanner(robot_type)
        planner.controller = self
        return planner

    def _init_uwb(self):
        self.uwb = UWBWrapper()
        self.position_update_callback = None
        self.uwb.register_callback(self.notify_drone_position_updated)

    def _init_skills(self, robot_type):
        # 載入低階技能 (Low Level)
        low_level_skillset = SkillSet(level="low")
        # --- 移動類 ---
        low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance", args=[SkillArg("distance", float)]))
        low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance", args=[SkillArg("distance", float)]))
        low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance", args=[SkillArg("distance", float)]))
        low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance", args=[SkillArg("distance", float)]))
        low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", float)]))
        low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", float)]))
        low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        low_level_skillset.add_skill(LowLevelSkillItem("delay", self.skill_delay, "Wait for specified seconds", args=[SkillArg("seconds", float)]))
        
        # --- 視覺類 ---
        low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm", args=[SkillArg("object_name", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("take_picture", self.skill_take_picture, "Take a picture"))
        
        # --- 邏輯類 ---
        low_level_skillset.add_skill(LowLevelSkillItem("probe", self.planner.probe, "Probe the LLM for reasoning", args=[SkillArg("question", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("log", self.skill_log, "Output text to console", args=[SkillArg("text", str)]))
        low_level_skillset.add_skill(LowLevelSkillItem("re_plan", self.skill_re_plan, "Replanning"))
        low_level_skillset.add_skill(LowLevelSkillItem("time", self.skill_time, "Get current execution time", args=[]))

        # 載入高階技能 (High Level) - 從 JSON 讀取
        high_level_skillset = SkillSet(level="high", lower_level_skillset=low_level_skillset)

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'
        
        # 使用設定檔中的路徑
        assets_path = os.path.join(self.config.assets_dir, f"{type_folder_name}/high_level_skills copy.json")
        
        # 容錯：如果 config 路徑找不到，回退到原本的 assets
        if not os.path.exists(assets_path):
             assets_path = os.path.join(CURRENT_DIR, f"assets/{type_folder_name}/high_level_skills copy.json")

        with open(assets_path, "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))
        
        self.planner.init(high_level_skillset=high_level_skillset, low_level_skillset=low_level_skillset, vision_skill=self.vision)
        
        return low_level_skillset, high_level_skillset

    # --- 以下保持原本的邏輯 ---

    def register_position_callback(self, callback):
        self.position_update_callback = callback

    def notify_drone_position_updated(self, position: Tuple[float, float, float, float]):
        timestamp, x, y, z = position
        position_str = f"Drone position updated (UWB): x={x:.2f}, y={y:.2f}, z={z:.2f}"
        # self.append_message(f"[LOG] {position_str}")
        self.latest_drone_position = (x, y, z)
        if hasattr(self, 'position_update_callback') and self.position_update_callback:
            self.position_update_callback(x, y, z, "uwb")

    def notify_virtual_drone_position_updated(self, x: float, y: float, z: float):
        self.latest_drone_position = (x, y, z)
        if hasattr(self, 'position_update_callback') and self.position_update_callback:
            self.position_update_callback(x, y, z, "virtual_drone")

    def get_drone_position(self) -> Tuple[float, float, float]:
        # Always return the latest known drone position, preferring UWB updates when available.
        if hasattr(self, "uwb") and self.uwb.latest_position != (0.0, 0.0, 0.0):
            return self.uwb.get_drone_position()
        return self.latest_drone_position

    def skill_get_drone_position(self) -> Tuple[str, bool]:
        x, y, z = self.get_drone_position()
        position_str = f"Drone position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
        # self.append_message(f"[LOG] {position_str}")
        return position_str, False

    def skill_get_user_position(self) -> Tuple[str, bool]:
        x, y, z = self.get_virtual_user_position()
        position_str = f"User position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
        # self.append_message(f"[LOG] {position_str}")
        return position_str, False
        
    def start_uwb(self):
        print_t("[C] Starting UWB tracking...")
        self.uwb.start_with_retry()
        self.uwb_active = True
        
    def stop_uwb(self):
        print_t("[C] Stopping UWB tracking...")
        self.uwb.stop()
        self.uwb_active = False
        
    def _publish_virtual_user_position(self, x: float, y: float, z: float):
        self.latest_virtual_user_position = (x, y, z)
        try:
            self.virtual_queue.put_nowait((time.time(), x, y, z))
        except queue.Full:
            self.virtual_queue.get()
            self.virtual_queue.put_nowait((time.time(), x, y, z))
        if self.position_update_callback:
            self.position_update_callback(x, y, z, "virtual")

    def set_random_virtual_user_position(self, margin: float = 0.5, xlim=(0.0, 5.0), ylim=(0.0, 5.0), z: float = 0.25):
        """Assign a stable virtual user position per task away from plot corners."""
        x_min, x_max = xlim[0] + margin, xlim[1] - margin
        y_min, y_max = ylim[0] + margin, ylim[1] - margin
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        self._publish_virtual_user_position(x, y, z)

    def start_virtual_user_position_loop(self):
        """Initialize a virtual user position without continuous drifting."""
        self.virtual_position_active = True
        self.set_random_virtual_user_position()

    def stop_virtual_user_position_loop(self):
        self.virtual_position_active = False

    def start_virtual_drone_position_loop(self):
        """Continuously mirror the simulated drone pose for fallback positioning."""
        def loop():
            while self.controller_active:
                x, y, z = self.drone.get_drone_position()
                self.notify_virtual_drone_position_updated(x, y, z)
                time.sleep(0.1)

        threading.Thread(target=loop, daemon=True).start()

    def get_virtual_user_position(self) -> Tuple[float, float, float]:
        return self.latest_virtual_user_position


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

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        self.append_message('[TASK]: ' + task_description)
        # 每個任務開始時，為虛擬使用者隨機生成一個固定位置，避免落在座標邊角。
        self.set_random_virtual_user_position()
        ret_val = None
        while True:
            user_pos = self.get_virtual_user_position()
            drone_pos = self.get_drone_position()
            
            location_info = {
                "user": {"x": round(user_pos[0], 2), "y": round(user_pos[1], 2), "z": round(user_pos[2], 2)},
                "drone": {"x": round(drone_pos[0], 2), "y": round(drone_pos[1], 2), "z": round(drone_pos[2], 2)},
            }

            scene_description = self.vision.get_obj_list() if self.enable_video else ''
            
            self.current_plan = self.planner.plan(
                task_description=task_description,
                scene_description=scene_description,
                location_info=location_info,
                execution_history=self.execution_history
            )
            
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

    def start_robot(self):
        print_t("[C] Connecting to robot...")
        self.drone.connect()
        print_t("[C] Starting robot...")
        self.drone.takeoff()
        if self.enable_video:
            print_t("[C] Starting stream...")
            self.drone.start_stream()
        self.start_uwb()
        self.start_virtual_drone_position_loop()
        print_t("[C] Starting virtual user position loop...")
        self.start_virtual_user_position_loop()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        if self.enable_video:
            self.drone.stop_stream()
        print_t("[C] Stopping UWB tracking...")
        self.stop_uwb()
        print_t("[C] Stopping virtual user position loop...")
        self.stop_virtual_user_position_loop()
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
