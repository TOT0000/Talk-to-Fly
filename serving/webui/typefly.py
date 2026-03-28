import queue
import time
import sys, os
import asyncio
import io, time
from collections import deque
import gradio as gr
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非互動後端避免開啟GUI視窗
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from PIL import Image
from threading import Thread
from flask import Flask, Response, request

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PARENT_DIR)

from controller.llm_controller import LLMController
from controller.utils import print_debug, print_t
from controller.llm_wrapper import GPT4, LLAMA3
from controller.abs.robot_wrapper import RobotType
from controller.experiment_scenarios import SCENARIOS, normalize_scenario_name
from controller.baseline_scenes import BASELINE_SCENES, normalize_baseline_scene_id
from gradio import Timer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TypeFly:
    def __init__(self, robot_type, use_http=False, enable_video=False, backend="uwb", initial_scenario="SAFE"):
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self.message_queue = queue.Queue()
        self.uwb_queue = queue.Queue(maxsize=500)
        self.virtual_queue = queue.Queue(maxsize=500)

        controller_robot_type = RobotType.PX4_SIM if backend == "sim" else robot_type
        self.llm_controller = LLMController(controller_robot_type, self.virtual_queue, use_http, self.message_queue, enable_video=enable_video)
        self.llm_controller.register_position_callback(self.receive_position)
        self.active_scenario = self.llm_controller.set_active_scenario(initial_scenario)
        self.active_baseline_scene = self.llm_controller.set_baseline_scene(
            normalize_baseline_scene_id(os.getenv("TYPEFLY_BASELINE_SCENE", "SCENE_1_CLEAR_PATH"))
        )
        
        self.system_stop = False
        self.ui_css = """
            .gradio-container, .gr-markdown, .gr-image, .gr-image img, .prose {
                opacity: 1 !important;
                filter: none !important;
                transition: none !important;
                animation: none !important;
            }
            .user-move-panel {
                max-width: 340px;
                margin: 0 auto;
            }
            .user-move-step {
                margin-bottom: 6px !important;
            }
            .scenario-panel {
                max-width: 320px;
            }
            .user-move-row {
                justify-content: center;
                gap: 8px;
                margin: 1px 0 !important;
            }
            .user-move-btn button {
                width: 96px !important;
                min-width: 96px !important;
                padding: 6px 8px !important;
            }
            """
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        self.use_llama3 = False
        self.robot_type = controller_robot_type

        # 狀態資料
        self.anchor_count = 0
        self.anchor_input_history = ""
        self.position_history = {
            "drone_gt": deque(maxlen=100),
            "drone_est": deque(maxlen=100),
            "user_gt": deque(maxlen=100),
            "user_est": deque(maxlen=100),
        }
        self.timing_history = {
            "drone_aoi_s": deque(maxlen=100),
            "user_aoi_s": deque(maxlen=100),
            "drone_delay_s": deque(maxlen=100),
            "user_delay_s": deque(maxlen=100),
        }
        self.plot_style = {
            "drone": {"main": "#0B57D0", "light": "#8AB4F8"},
            "user": {"main": "#C5221F", "light": "#F28B82"},
        }

        # 浮動提示 internal state
        self._temp_message = ""
        self._temp_message_expire = 0.0
        '''
        default_sentences = [
            "Find something I can eat.",
            "What you can see?",
            "Follow that ball for 20 seconds",
            "Find a chair for me.",
            "Go to the chair without book."
        ]
        '''

        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())

            # 浮動提示（頂端）
            self.message_markdown = gr.Markdown(value="", visible=False)

            with gr.Row():
                with gr.Column(scale=1, min_width=260, elem_classes="scenario-panel"):
                    self.scenario_selector = gr.Dropdown(
                        choices=list(SCENARIOS.keys()),
                        value=self.active_scenario,
                        label="Scenario Mode",
                    )
                    self.scenario_apply_btn = gr.Button("Apply Scenario")
                    self.baseline_scene_selector = gr.Dropdown(
                        choices=list(BASELINE_SCENES.keys()),
                        value=normalize_baseline_scene_id(os.getenv("TYPEFLY_BASELINE_SCENE", "SCENE_1_CLEAR_PATH")),
                        label="Baseline Scene",
                    )
                    self.baseline_scene_apply_btn = gr.Button("Apply Baseline Scene")
                with gr.Column(scale=1, min_width=320, elem_classes="user-move-panel"):
                    self.user_move_step = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="User Move Step (m)",
                        elem_classes="user-move-step",
                    )
                    with gr.Row(elem_classes="user-move-row"):
                        gr.Markdown("")
                        self.user_move_forward_btn = gr.Button("Forward", elem_classes="user-move-btn")
                        gr.Markdown("")
                    with gr.Row(elem_classes="user-move-row"):
                        self.user_move_left_btn = gr.Button("Left", elem_classes="user-move-btn")
                        gr.Markdown("")
                        self.user_move_right_btn = gr.Button("Right", elem_classes="user-move-btn")
                    with gr.Row(elem_classes="user-move-row"):
                        gr.Markdown("")
                        self.user_move_backward_btn = gr.Button("Backward", elem_classes="user-move-btn")
                        gr.Markdown("")
            self.scenario_status = gr.Markdown(value="")

            self.scenario_apply_btn.click(
                fn=self.apply_scenario,
                inputs=[self.scenario_selector],
                outputs=[self.scenario_status],
            )
            self.baseline_scene_apply_btn.click(
                fn=self.apply_baseline_scene,
                inputs=[self.baseline_scene_selector],
                outputs=[self.scenario_status],
            )

            self.user_move_forward_btn.click(
                fn=self.move_user_forward,
                inputs=[self.user_move_step],
                outputs=[self.scenario_status],
            )
            self.user_move_backward_btn.click(
                fn=self.move_user_backward,
                inputs=[self.user_move_step],
                outputs=[self.scenario_status],
            )
            self.user_move_left_btn.click(
                fn=self.move_user_left,
                inputs=[self.user_move_step],
                outputs=[self.scenario_status],
            )
            self.user_move_right_btn.click(
                fn=self.move_user_right,
                inputs=[self.user_move_step],
                outputs=[self.scenario_status],
            )

            # floating message refresher
            self.message_timer = Timer(value=0.5)
            self.message_timer.tick(
                fn=self._refresh_temp_message,
                inputs=[],
                outputs=[self.message_markdown]
            )

            # plots
            with gr.Row():
                self.global_xy_plot = gr.Image(
                    value=self.create_blank_plot(
                        "Global XY Map (Fixed 0-12m Workspace)",
                        "X (m)",
                        "Y (m)",
                        xlim=(0, 12),
                        ylim=(0, 12),
                        figsize=(10, 8),
                    ),
                    label="Global XY Map",
                    height=640,
                )
            with gr.Row():
                self.xy_plot = gr.Image(
                    value=self.create_blank_plot(
                        "Drone / User Localization & Safety Envelope (XY view)",
                        "X (m)",
                        "Y (m)",
                        xlim=(0, 12),
                        ylim=(0, 12),
                        figsize=(5, 4),
                    ),
                    label="XY Plot",
                    height=320,
                )
                self.z_plot = gr.Image(
                    value=self.create_sequence_plot("Z in Sequence", "Index", "Z (m)", xlim=(0, 1), ylim=(0, 6)),
                    label="Z in Sequence",
                    height=320,
                )
            with gr.Row():
                self.x_plot = gr.Image(
                    value=self.create_sequence_plot("X in Sequence", "Index", "X (m)", xlim=(0, 1), ylim=(0, 12)),
                    label="X in Sequence",
                    height=320,
                )
                self.y_plot = gr.Image(
                    value=self.create_sequence_plot("Y in Sequence", "Index", "Y (m)", xlim=(0, 1), ylim=(0, 12)),
                    label="Y in Sequence",
                    height=320,
                )
            with gr.Row():
                self.aoi_plot = gr.Image(
                    value=self.create_sequence_plot("AoI Trend", "Sample", "AoI (s)", xlim=(0, 1), ylim=(0, 1)),
                    label="AoI Trend",
                    height=320,
                )
                self.delay_plot = gr.Image(
                    value=self.create_sequence_plot("Delay Trend", "Sample", "Delay (s)", xlim=(0, 1), ylim=(0, 1)),
                    label="Delay Trend",
                    height=320,
                )
            with gr.Row():
                self.coordinate_markdown = gr.Markdown(value="### Coordinates\nWaiting for live data...")
                self.safety_markdown = gr.Markdown(value="### Safety / Risk\nWaiting for safety state...")
                self.delay_markdown = gr.Markdown(value="### AoI / Delay\nWaiting for packets...")
                self.baseline_markdown = gr.Markdown(value="### Baseline Status\nWaiting for baseline scene...")

            self.counter = gr.State(0)
            self.timer = Timer(value=0.08)
            self.timer.tick(
                fn=self.update_and_step,
                inputs=[self.counter],
                outputs=[
                    self.global_xy_plot,
                    self.xy_plot,
                    self.x_plot,
                    self.y_plot,
                    self.z_plot,
                    self.aoi_plot,
                    self.delay_plot,
                    self.counter,
                    self.coordinate_markdown,
                    self.safety_markdown,
                    self.delay_markdown,
                    self.baseline_markdown,
                ]
            )

            self.chat = gr.ChatInterface(self.process_message, fill_height=False)

    def show_temporary_message(self, text, duration=3):
        self._temp_message = text
        self._temp_message_expire = time.time() + duration

    def _refresh_temp_message(self):
        if hasattr(self, "_temp_message") and time.time() < self._temp_message_expire:
            return gr.update(value=f"**{self._temp_message}**", visible=True)
        else:
            return gr.update(value="", visible=False)
            
    def reset_anchors(self):
        # 同步到 UWBWrapper 透過 controller
        if hasattr(self, "llm_controller") and hasattr(self.llm_controller, "uwb"):
            self.llm_controller.uwb.reset_anchors()

        self.anchor_input_history = ""
        self.show_temporary_message("Reset anchor count and positions.")
        # 清空 input，恢復 submit 按鈕可用，anchor line 按鈕不可用
        return "", "", "", gr.update(interactive=True), gr.update(interactive=True)


    def receive_position(self, *args):
        if len(args) == 4:
            x, y, z, source = args
        elif len(args) == 3:
            x, y, z = args
            source = "unknown"
        else:
            return

        timestamp = time.time()
        tag = "[DronePos]" if source == "drone" else "[UserPos]" if source == "user" else "[Pos]"

        # 初始化紀錄字典
        if not hasattr(self, '_last_position_map'):
            self._last_position_map = {}

        current_pos = (x, y, z)
        last_pos = self._last_position_map.get(source)

        if current_pos != last_pos:
            # 有改變才放入 queue
            try:
                if source == "user":
                    self.uwb_queue.put_nowait((timestamp, x, y, z))
                elif source == "drone":
                    self.virtual_queue.put_nowait((timestamp, x, y, z))
            except queue.Full:
                if source == "user":
                    self.uwb_queue.get()
                    self.uwb_queue.put_nowait((timestamp, x, y, z))
                elif source == "drone":
                    self.virtual_queue.get()
                    self.virtual_queue.put_nowait((timestamp, x, y, z))

            # 每種來源各自印出，每 5 秒一次
            if not hasattr(self, '_last_print_position_map'):
                self._last_print_position_map = {}

            last_print_time = self._last_print_position_map.get(source, 0)
            if timestamp - last_print_time > 5:
                print_debug(f"{tag} x={x:.2f}, y={y:.2f}, z={z:.2f}")
                self._last_print_position_map[source] = timestamp

            # 更新最後位置
            self._last_position_map[source] = current_pos



    def set_anchor_count(self, anchor_count_input):
        try:
            n = int(anchor_count_input)
            if n <= 0:
                self.show_temporary_message("Anchor count must be a positive integer.")
                return anchor_count_input, gr.update(), gr.update(value=self.anchor_input_history), gr.update(interactive=True), gr.update(interactive=True)
            self.llm_controller = getattr(self, "llm_controller", None)
            # call underlying wrapper
            self.llm_controller.uwb.set_anchor_count(n)
            self.anchor_count = n
            self.anchor_input_history = ""
            self.show_temporary_message(f"Anchor count set to {n}. Please enter anchor 1's position.")
            return anchor_count_input, gr.update(value=""), gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True)
        except Exception as e:
            self.show_temporary_message(f"Failed to set anchor count: {e}")
            return anchor_count_input, gr.update(), gr.update(value=self.anchor_input_history), gr.update(interactive=True), gr.update(interactive=True)

    def input_anchor_line(self, line_str):
        if self.llm_controller.uwb.anchor_count <= 0:
            self.show_temporary_message("Please submit a valid anchor count first.")
            return "", self.anchor_input_history, gr.update(interactive=True)

        line_str = line_str.strip()
        if not line_str:
            self.show_temporary_message("Anchor position input cannot be empty.")
            return "", self.anchor_input_history, gr.update(interactive=True)

        parts = line_str.split(',')
        if len(parts) != 4:
            self.show_temporary_message("Invalid format. Please use: i,x,y,z")
            return "", self.anchor_input_history, gr.update(interactive=True)

        try:
            user_i = int(parts[0])
            if user_i < 1 or user_i > self.llm_controller.uwb.anchor_count:
                self.show_temporary_message(f"Anchor index {user_i} out of range (1 to {self.llm_controller.uwb.anchor_count})")
                return "", self.anchor_input_history, gr.update(interactive=True)
            self.llm_controller.uwb.input_anchor_line(line_str)
        except Exception:
            self.show_temporary_message("Invalid format. i must be int, x,y,z must be float")
            return "", self.anchor_input_history, gr.update(interactive=True)

        idx = user_i - 1
        anchor = self.llm_controller.uwb.anchors[idx]
        if anchor is not None:
            x, y, z = anchor
            self.anchor_input_history += f"Anchor {user_i} set to ({x:.2f}, {y:.2f}, {z:.2f})\n"

        if all(a is not None for a in self.llm_controller.uwb.anchors):
            self.show_temporary_message("All anchors set. Starting positioning.")
            # 禁用按鈕，防止再新增或更新
            return "", self.anchor_input_history, gr.update(interactive=False)
        else:
            next_idx = self.llm_controller.uwb.anchors.index(None) + 1
            self.show_temporary_message(f"Please enter anchor {next_idx}'s position.")
            return "", self.anchor_input_history, gr.update(interactive=True)


    def checkbox_llama3(self):
        self.use_llama3 = not self.use_llama3
        if self.use_llama3:
            print_t("Switch to llama3")
            self.llm_controller.planner.set_model(LLAMA3)
        else:
            print_t("Switch to gpt4")
            self.llm_controller.planner.set_model(GPT4)

    def apply_scenario(self, scenario_name):
        normalized, report, runtime = self._apply_mode_and_collect(scenario_name)
        return (
            f"Scenario `{normalized}` applied. "
            f"Live safety: {runtime.get('safety_level')} "
            f"(score={self._fmt_float(runtime.get('safety_score'))})"
        )

    def apply_baseline_scene(self, scene_id):
        normalized = self.llm_controller.set_baseline_scene(scene_id)
        state = self.llm_controller.apply_baseline_scene()
        return f"Baseline scene `{normalized}` applied. drone_init={self._fmt_vec(state.get('drone_initial_pose'))} user={self._fmt_vec(state.get('user_position'))}"

    def _apply_mode_and_collect(self, scenario_name):
        normalized = normalize_scenario_name(scenario_name)
        self.llm_controller.set_active_scenario(normalized)
        report = self.llm_controller.apply_selected_scenario()
        runtime = self.llm_controller.get_scenario_runtime_status()
        self.active_scenario = normalized
        return normalized, report, runtime

    def _move_user(self, dx: float, dy: float, step_m: float):
        step = float(step_m)
        updated = self.llm_controller.move_user_world(dx=dx * step, dy=dy * step, dz=0.0)
        if updated is None:
            return "User move failed: no simulation user-position provider."
        runtime = self.llm_controller.get_scenario_runtime_status()
        return (
            f"User moved to {self._fmt_vec(updated)} | "
            f"live safety={runtime.get('safety_level')} "
            f"(score={self._fmt_float(runtime.get('safety_score'))})"
        )

    def move_user_forward(self, step_m: float):
        return self._move_user(dx=0.0, dy=1.0, step_m=step_m)

    def move_user_backward(self, step_m: float):
        return self._move_user(dx=0.0, dy=-1.0, step_m=step_m)

    def move_user_left(self, step_m: float):
        return self._move_user(dx=-1.0, dy=0.0, step_m=step_m)

    def move_user_right(self, step_m: float):
        return self._move_user(dx=1.0, dy=0.0, step_m=step_m)

    def process_message(self, message, history):
        print_t(f"[S] Receiving task description: {message}")
        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
        elif len(message) == 0:
            return "[WARNING] Empty command!"
        else:
            task_thread = Thread(target=self.llm_controller.execute_task_description, args=(message,))
            task_thread.start()
            complete_response = ''
            while True:
                msg = self.message_queue.get()
                if isinstance(msg, tuple):
                    history.append((None, msg))
                elif isinstance(msg, str):
                    if msg == 'end':
                        return "Command Complete!"
                    if msg.startswith('[LOG]'):
                        complete_response += '\n'
                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                yield complete_response

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame(True)
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.start()

        self.llm_controller.start_robot()
        try:
            self.llm_controller.apply_baseline_scene()
        except Exception:
            pass

        if self.llm_controller.enable_video:
            llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
            llmc_thread.start()
        else:
            llmc_thread = None

        if self.llm_controller.enable_video:
            app = Flask(__name__)

            @app.route('/drone-pov/')
            def video_feed():
                return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

            @app.route('/shutdown', methods=['POST'])
            def shutdown():
                func = request.environ.get('werkzeug.server.shutdown')
                if func:
                    func()
                    return 'Server shutting down...'
                else:
                    return 'Unable to shut down server', 500

            PORT = int(os.environ.get("FLASK_PORT", 50000))
            flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': PORT, 'debug': False, 'use_reloader': False})
            flask_thread.start()
        else:
            flask_thread = None

        self.chat.queue()
        self.ui.launch(server_port=50001, prevent_thread_lock=True, css=self.ui_css)

        while not self.system_stop:
            time.sleep(1)

        print_t("[C] Shutting down system...")
        self.llm_controller.stop_robot()

        if self.llm_controller.enable_video and flask_thread:
            try:
                import requests
                requests.post("http://localhost:50000/shutdown")
            except Exception as e:
                print_t(f"[WARN] Failed to shutdown Flask server: {e}")

        if llmc_thread:
            llmc_thread.join()
        asyncio_thread.join()

        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file))

    def create_blank_plot(self, title="Empty Plot", xlabel="X", ylabel="Y", xlim=(0, 1), ylim=(0, 1), figsize=(5, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([round(xlim[0] + i * 0.5, 2) for i in range(int((xlim[1]-xlim[0])/0.5)+1)])
        ax.set_yticks([round(ylim[0] + i * 0.5, 2) for i in range(int((ylim[1]-ylim[0])/0.5)+1)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def create_sequence_plot(self, title, xlabel, ylabel, xlim=(0, 1), ylim=(0, 5)):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([i * 0.2 for i in range(6)])
        ax.set_yticks([i * 0.5 for i in range(11)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def update_and_step(self, counter):
        snapshot = self.llm_controller.get_live_ui_snapshot()
        self._append_history(snapshot)
        global_xy, xy, x, y, z = self.update_position_plot(snapshot)
        aoi_img, delay_img = self.update_timing_plots()
        coordinate_md = self.render_coordinate_markdown(snapshot)
        safety_md = self.render_safety_markdown(snapshot)
        delay_md = self.render_delay_markdown(snapshot)
        baseline_md = self.render_baseline_markdown(snapshot)
        counter += 1
        print_debug(
            "[UI-CALLBACK] "
            "outputs=[global_xy_plot,xy_plot,x_plot,y_plot,z_plot,aoi_plot,delay_plot,counter,coordinate_markdown,safety_markdown,delay_markdown] "
            f"drone_gt={None if not snapshot else snapshot.get('drone_gt')} "
            f"drone_est={None if not snapshot else snapshot.get('drone_est')} "
            f"user_gt={None if not snapshot else snapshot.get('user_gt')} "
            f"user_est={None if not snapshot else snapshot.get('user_est')} "
            f"counter={counter}"
        )
        return global_xy, xy, x, y, z, aoi_img, delay_img, counter, coordinate_md, safety_md, delay_md, baseline_md

    def _fmt_vec(self, value):
        if value is None:
            return "(n/a)"
        return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"

    def _fmt_float(self, value, suffix=""):
        if value is None:
            return "n/a"
        return f"{value:.3f}{suffix}"

    def _extract_ui_positions(self, snapshot):
        if not snapshot:
            return {
                "drone_gt": None,
                "drone_est": None,
                "user_gt": None,
                "user_est": None,
            }
        positions = {key: snapshot.get(key) for key in ("drone_gt", "drone_est", "user_gt", "user_est")}
        return positions

    def _append_history(self, snapshot):
        positions = self._extract_ui_positions(snapshot)
        if not snapshot:
            return
        for key, value in positions.items():
            if value is not None:
                self.position_history[key].append(tuple(float(v) for v in value))
                print_debug(f"[UI-HISTORY] key={key} appended={self.position_history[key][-1]}")
        for key in ("drone_aoi_s", "user_aoi_s", "drone_delay_s", "user_delay_s"):
            value = snapshot.get(key)
            if value is not None:
                self.timing_history[key].append(float(value))

    def render_coordinate_markdown(self, snapshot):
        if not snapshot:
            return "### Coordinates\nWaiting for live data..."
        positions = self._extract_ui_positions(snapshot)
        initial_state = self.llm_controller.get_initial_scenario_state()
        print_debug(
            "[UI-MARKDOWN] "
            f"drone_gt={positions['drone_gt']} drone_est={positions['drone_est']} "
            f"user_gt={positions['user_gt']} user_est={positions['user_est']}"
        )
        initial_block = ""
        if initial_state:
            initial_block = (
                f"**Initial scenario (locked before task)**\n"
                f"- selected_mode: {initial_state.get('selected_mode')}\n"
                f"- actual initial drone GT: {self._fmt_vec(initial_state.get('actual_drone_gt'))}\n"
                f"- actual initial user GT: {self._fmt_vec(initial_state.get('actual_user_gt'))}\n\n"
            )
        return (
            "### Coordinates\n"
            f"- Drone GT: {self._fmt_vec(positions['drone_gt'])}\n"
            f"- Drone EST: {self._fmt_vec(positions['drone_est'])}\n"
            f"- User GT: {self._fmt_vec(positions['user_gt'])}\n"
            f"- User EST: {self._fmt_vec(positions['user_est'])}"
        )

    def render_safety_markdown(self, snapshot):
        safety_context = snapshot.get("safety_context") if snapshot else None
        if safety_context is None:
            return "### Safety / Risk\nWaiting for safety state..."
        return "\n".join(
            [
                "### Safety / Risk",
                f"- safety_score: {safety_context.safety_score:.3f}",
                f"- safety_level: {safety_context.safety_level}",
                f"- envelope_gap_m (centerline ray-gap): {safety_context.envelope_gap_m:.3f} m",
                f"- uncertainty_scale_m: {safety_context.uncertainty_scale_m:.3f} m",
                f"- envelopes_overlap (centerline): {safety_context.envelopes_overlap}",
            ]
        )

    def render_delay_markdown(self, snapshot):
        if not snapshot:
            return "### AoI / Delay\nWaiting for packets..."
        return (
            "### AoI / Delay\n"
            f"- drone AoI (blue): {self._fmt_float(snapshot.get('drone_aoi_s'), ' s')}\n"
            f"- drone observed uplink delay (blue): {self._fmt_float(snapshot.get('drone_delay_s'), ' s')}\n"
            f"- user AoI (red): {self._fmt_float(snapshot.get('user_aoi_s'), ' s')}\n"
            f"- user observed uplink delay (red): {self._fmt_float(snapshot.get('user_delay_s'), ' s')}"
        )

    def render_baseline_markdown(self, snapshot):
        if not snapshot:
            return "### Baseline Status\nWaiting for baseline state..."
        scene = snapshot.get("baseline_scene")
        path_eval = snapshot.get("path_eval")
        target_task_point = snapshot.get("target_task_point") or "A"
        blocking = "none"
        path_clear = "n/a"
        min_gap = "n/a"
        if path_eval is not None:
            blocking = path_eval.blocking_entity
            path_clear = str(bool(path_eval.path_clear))
            min_gap = f"{float(path_eval.corridor_min_gap):.3f}"
        expectation_lines = []
        for row in snapshot.get("baseline_expectation_summary") or []:
            expectation_lines.append(
                f"  - {row.get('target_task_point')}: clear={row.get('expected_path_clear')} "
                f"blocker={row.get('expected_blocking_entity')} mode={row.get('expected_motion_mode')}"
            )
        expectation_block = "\n".join(expectation_lines) if expectation_lines else "  - (n/a)"
        all_rows = snapshot.get("baseline_all_scene_expectations") or []
        all_scene_block = {}
        for row in all_rows:
            scene_id = row.get("scene_id")
            item = (
                f"{row.get('target_task_point')}="
                f"{row.get('expected_motion_mode')}/"
                f"{row.get('expected_blocking_entity')}"
            )
            all_scene_block.setdefault(scene_id, []).append(item)
        all_scene_lines = []
        for scene_id, items in all_scene_block.items():
            all_scene_lines.append(f"  - {scene_id}: " + ", ".join(items))
        all_scene_summary = "\n".join(all_scene_lines) if all_scene_lines else "  - (n/a)"
        return (
            "### Baseline Status\n"
            f"- current scene id: {None if scene is None else scene.id}\n"
            f"- current target task point: {target_task_point}\n"
            f"- path_clear: {path_clear}\n"
            f"- blocking entity: {blocking}\n"
            f"- corridor_min_gap_m: {min_gap}\n"
            f"- current scene expected behavior:\n{expectation_block}\n"
            f"- all scenes quick matrix (mode/blocker):\n{all_scene_summary}"
        )

    def _estimate_heading_from_history(self, primary_key: str, fallback_key: str = None):
        history = list(self.position_history.get(primary_key, []))
        if len(history) < 2 and fallback_key:
            history = list(self.position_history.get(fallback_key, []))
        if len(history) >= 2:
            p0 = history[-2]
            p1 = history[-1]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                return float(np.arctan2(dy, dx)), "trajectory_history"
        return 0.0, "fallback_zero"

    def _axis_limits_from_snapshot(self, snapshot):
        positions = self._extract_ui_positions(snapshot)
        xs, ys = [], []
        for key in ("drone_gt", "drone_est", "user_gt", "user_est"):
            value = positions.get(key)
            if value is not None:
                xs.append(float(value[0]))
                ys.append(float(value[1]))
            history = self.position_history.get(key)
            if history:
                xs.extend(float(point[0]) for point in history)
                ys.extend(float(point[1]) for point in history)
        safety_state = snapshot.get("safety_state") if snapshot else None
        if safety_state is not None:
            for envelope in (safety_state.drone_envelope, safety_state.user_envelope):
                xs.extend([
                    float(envelope.center_xy[0] - envelope.major_axis_radius),
                    float(envelope.center_xy[0] + envelope.major_axis_radius),
                ])
                ys.extend([
                    float(envelope.center_xy[1] - envelope.major_axis_radius),
                    float(envelope.center_xy[1] + envelope.major_axis_radius),
                ])
        if not xs or not ys:
            return (0.0, 5.0), (0.0, 5.0)
        pad = 0.5
        return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)

    def _render_timing_plot(self, history_keys, title, ylabel):
        fig, ax = plt.subplots(figsize=(5, 4))
        values = []
        series_specs = [
            (history_keys[0], "Drone", self.plot_style["drone"]["main"]),
            (history_keys[1], "User", self.plot_style["user"]["main"]),
        ]
        for key, label, color in series_specs:
            history = list(self.timing_history[key])
            if not history:
                ax.plot([], [], color=color, label=label)
                continue
            x_vals = list(range(len(history)))
            values.extend(history)
            ax.plot(x_vals, history, color=color, linestyle="-", marker="o", markersize=3, label=label)

        max_len = max((len(self.timing_history[key]) for key, _, _ in series_specs), default=1)
        ax.set_xlim(0, max(max_len - 1, 1))
        if values:
            ymin = max(0.0, min(values) - 0.05)
            ymax = max(values) + 0.05
        else:
            ymin, ymax = 0.0, 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def update_timing_plots(self):
        aoi_img = self._render_timing_plot(("drone_aoi_s", "user_aoi_s"), "AoI Trend", "AoI (s)")
        delay_img = self._render_timing_plot(("drone_delay_s", "user_delay_s"), "Delay Trend", "Delay (s)")
        return aoi_img, delay_img

    def _render_xy_view(self, snapshot, xlim, ylim, title, figsize=(5, 4), show_legend=True):
        positions = self._extract_ui_positions(snapshot)
        fig_xy, ax_xy = plt.subplots(figsize=figsize)

        point_specs = [
            ("drone_gt", "Drone GT", self.plot_style["drone"]["main"], "o", True),
            ("drone_est", "Drone EST", self.plot_style["drone"]["light"], "X", True),
            ("user_gt", "User GT", self.plot_style["user"]["main"], "o", True),
            ("user_est", "User EST", self.plot_style["user"]["light"], "X", True),
        ]
        for spec in point_specs:
            key = spec[0]
            label = spec[1]
            color = spec[2]
            marker = spec[3] if len(spec) > 3 else "o"
            filled = spec[4] if len(spec) > 4 else True
            value = positions.get(key)
            if value is None:
                ax_xy.scatter([], [], c=color, marker=marker, label=label)
                continue
            facecolors = color if filled else "none"
            ax_xy.scatter([value[0]], [value[1]], edgecolors=color, facecolors=facecolors, marker=marker, s=70, linewidths=1.8, label=label)
            ax_xy.text(value[0] + 0.03, value[1] + 0.03, f"{label}: ({value[0]:.3f}, {value[1]:.3f})", fontsize=8, color=color)

        if positions.get("drone_gt") is not None and positions.get("drone_est") is not None:
            ax_xy.plot(
                [positions["drone_gt"][0], positions["drone_est"][0]],
                [positions["drone_gt"][1], positions["drone_est"][1]],
                color=self.plot_style["drone"]["main"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="Drone GT→EST",
            )
        if positions.get("user_gt") is not None and positions.get("user_est") is not None:
            ax_xy.plot(
                [positions["user_gt"][0], positions["user_est"][0]],
                [positions["user_gt"][1], positions["user_est"][1]],
                color=self.plot_style["user"]["main"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="User GT→EST",
            )

        safety_state = snapshot.get("safety_state") if snapshot else None
        if safety_state is not None:
            for label, envelope, color in (
                ("Drone envelope", safety_state.drone_envelope, self.plot_style["drone"]["light"]),
                ("User envelope", safety_state.user_envelope, self.plot_style["user"]["light"]),
            ):
                ellipse = Ellipse(
                    xy=(float(envelope.center_xy[0]), float(envelope.center_xy[1])),
                    width=2.0 * float(envelope.major_axis_radius),
                    height=2.0 * float(envelope.minor_axis_radius),
                    angle=float(envelope.orientation_deg),
                    edgecolor=color,
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="--",
                    label=label,
                )
                ax_xy.add_patch(ellipse)

        baseline_scene = snapshot.get("baseline_scene") if snapshot else None
        obstacle_states = snapshot.get("obstacle_envelope_states") if snapshot else None
        if baseline_scene is not None:
            for point in baseline_scene.task_points:
                ax_xy.scatter([point.x], [point.y], c="#2E7D32", marker="D", s=65, label="Task point")
                ax_xy.text(point.x + 0.06, point.y + 0.06, f"{point.id}", fontsize=8, color="#1B5E20")
            for obstacle in obstacle_states or []:
                ax_xy.scatter([obstacle.gt_xy[0]], [obstacle.gt_xy[1]], c="#5D4037", marker="s", s=62, label="Obstacle GT")
                ax_xy.scatter([obstacle.est_xy[0]], [obstacle.est_xy[1]], c="#8D6E63", marker="X", s=58, label="Obstacle EST")
                obstacle_ellipse = Ellipse(
                    xy=(float(obstacle.est_xy[0]), float(obstacle.est_xy[1])),
                    width=2.0 * float(obstacle.envelope_major_axis_m),
                    height=2.0 * float(obstacle.envelope_minor_axis_m),
                    angle=float(obstacle.orientation_deg),
                    edgecolor="#8D6E63",
                    facecolor="none",
                    linestyle=":",
                    linewidth=1.4,
                    label="Obstacle envelope",
                )
                ax_xy.add_patch(obstacle_ellipse)
                ax_xy.text(obstacle.est_xy[0] + 0.08, obstacle.est_xy[1] + 0.08, obstacle.id, fontsize=8, color="#4E342E")

        drone_for_heading = positions.get("drone_gt") or positions.get("drone_est")
        yaw_rad = float(snapshot.get("drone_yaw_rad") or 0.0) if snapshot else 0.0
        if drone_for_heading is not None:
            hx = float(drone_for_heading[0])
            hy = float(drone_for_heading[1])
            arrow_len = 0.55
            dx = arrow_len * float(np.cos(yaw_rad))
            dy = arrow_len * float(np.sin(yaw_rad))
            ax_xy.arrow(hx, hy, dx, dy, head_width=0.16, head_length=0.18, color="#0B57D0", linewidth=1.6, length_includes_head=True, zorder=5)
            ax_xy.text(hx + dx + 0.05, hy + dy + 0.05, "Heading", fontsize=8, color="#0B57D0")

        user_for_heading = positions.get("user_gt") or positions.get("user_est")
        if user_for_heading is not None:
            user_yaw_rad, user_heading_source = self._estimate_heading_from_history("user_gt", fallback_key="user_est")
            ux = float(user_for_heading[0])
            uy = float(user_for_heading[1])
            arrow_len = 0.45
            udx = arrow_len * float(np.cos(user_yaw_rad))
            udy = arrow_len * float(np.sin(user_yaw_rad))
            ax_xy.arrow(ux, uy, udx, udy, head_width=0.14, head_length=0.16, color="#C5221F", linewidth=1.4, length_includes_head=True, zorder=5)
            ax_xy.text(ux + udx + 0.04, uy + udy + 0.04, f"User Heading ({user_heading_source})", fontsize=7, color="#C5221F")

        ax_xy.set_xlim(*xlim)
        ax_xy.set_ylim(*ylim)
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_title(title)
        ax_xy.grid(True, linestyle='--', linewidth=0.5)
        if show_legend:
            handles, labels = ax_xy.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_xy.legend(dedup.values(), dedup.keys(), fontsize=8)

        buf_xy = io.BytesIO()
        fig_xy.savefig(buf_xy, format='png')
        buf_xy.seek(0)
        plt.close(fig_xy)
        return Image.open(buf_xy)

    def update_position_plot(self, snapshot):
        positions = self._extract_ui_positions(snapshot)
        dynamic_xlim, dynamic_ylim = self._axis_limits_from_snapshot(snapshot)
        print_debug(
            "[UI-PLOT] "
            f"drone_gt={positions['drone_gt']} "
            f"drone_est={positions['drone_est']} "
            f"user_gt={positions['user_gt']} "
            f"user_est={positions['user_est']}"
        )

        global_xy = self._render_xy_view(
            snapshot=snapshot,
            xlim=(0.0, 12.0),
            ylim=(0.0, 12.0),
            title="Global XY Map (Fixed 0-12m Workspace)",
            figsize=(10, 8),
            show_legend=True,
        )
        local_xy = self._render_xy_view(
            snapshot=snapshot,
            xlim=dynamic_xlim,
            ylim=dynamic_ylim,
            title="Drone / User Localization & Safety Envelope (XY view)",
            figsize=(5.8, 4.4),
            show_legend=False,
        )

        series_specs = [
            ("drone_gt", "Drone GT", self.plot_style["drone"]["main"], "-"),
            ("drone_est", "Drone EST", self.plot_style["drone"]["main"], "--"),
            ("user_gt", "User GT", self.plot_style["user"]["main"], "-"),
            ("user_est", "User EST", self.plot_style["user"]["main"], "--"),
        ]
        imgs = []
        axis_map = {"x": 0, "y": 1, "z": 2}
        for axis in ["x", "y", "z"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            values = []
            axis_idx = axis_map[axis]
            for spec in series_specs:
                key = spec[0]
                label = spec[1]
                color = spec[2]
                linestyle = spec[3] if len(spec) > 3 else "-"
                history = list(self.position_history[key])
                if not history:
                    ax.plot([], [], color=color, label=label)
                    continue
                x_vals = list(range(len(history)))
                y_vals = [point[axis_idx] for point in history]
                values.extend(y_vals)
                markerfacecolor = color if linestyle == "-" else "none"
                ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    linestyle=linestyle,
                    marker='o',
                    markersize=3,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=color,
                    label=label,
                )
            max_len = max((len(self.position_history[spec[0]]) for spec in series_specs), default=1)
            ax.set_xlim(0, max(max_len - 1, 1))
            if values:
                ymin = min(values) - 0.5
                ymax = max(values) + 0.5
            else:
                ymin, ymax = 0.0, 5.0
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{axis.upper()} History")
            ax.set_xlabel("Sample")
            ax.set_ylabel(f"{axis.upper()} (m)")
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.legend(fontsize=8)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            imgs.append(Image.open(buf))

        return global_xy, local_xy, imgs[0], imgs[1], imgs[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_virtual_robot', action='store_true')
    parser.add_argument('--use_http', action='store_true')
    parser.add_argument('--gear', action='store_true')
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--px4_sim', action='store_true')
    parser.add_argument('--scenario', type=str, default=os.getenv("TYPEFLY_SCENARIO", "SAFE"))

    args = parser.parse_args()
    robot_type = RobotType.TELLO
    backend = "uwb"
    if args.px4_sim:
        robot_type = RobotType.PX4_SIM
        backend = "sim"
    elif args.use_virtual_robot:
        robot_type = RobotType.VIRTUAL
    elif args.gear:
        robot_type = RobotType.GEAR

    typefly = TypeFly(
        robot_type,
        use_http=args.use_http,
        enable_video=args.image,
        backend=backend,
        initial_scenario=normalize_scenario_name(args.scenario),
    )
    typefly.run()
