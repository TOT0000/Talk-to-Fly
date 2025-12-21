import queue
import time
import sys, os
import asyncio
import io
import gradio as gr
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非互動後端避免開啟GUI視窗
import matplotlib.pyplot as plt
from PIL import Image
from threading import Thread
from flask import Flask, Response, request

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PARENT_DIR)

from controller.llm_controller import LLMController
from controller.utils import print_t
from controller.llm_wrapper import GPT4, LLAMA3
from controller.abs.robot_wrapper import RobotType
from controller.uwb_wrapper import UWBWrapper
from gradio import Timer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TypeFly:
    def __init__(self, robot_type, use_http=False, enable_video=False):
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self.message_queue = queue.Queue()
        self.uwb_queue = queue.Queue(maxsize=500)
        self.virtual_queue = queue.Queue(maxsize=500)

        self.llm_controller = LLMController(robot_type, self.virtual_queue, use_http, self.message_queue, enable_video=enable_video)
        self.llm_controller.register_position_callback(self.receive_position)
        
        self.system_stop = False
        self.ui = gr.Blocks(title="Talk-to-Fly")
        self.asyncio_loop = asyncio.get_event_loop()
        self.use_llama3 = False
        self.robot_type = robot_type

        # 狀態資料
        self.anchor_count = 0
        self.anchor_input_history = ""

        # 浮動提示 internal state
        self._temp_message = ""
        self._temp_message_expire = 0.0
        
        """
        default_sentences = [
            "Find something I can eat.",
            "What you can see?",
            "Follow that ball for 20 seconds",
            "Find a chair for me.",
            "Go to the chair without book."
        ]
        """

        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())

            # 浮動提示（頂端）
            self.message_markdown = gr.Markdown(value="", visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    self.anchor_count_input = gr.Textbox(
                        label="Anchor Count (Enter integer)",
                        placeholder="e.g. 4",
                        value=""
                    )
                    self.anchor_count_submit_btn = gr.Button("Submit Anchor Count")
                    self.anchor_reset_btn = gr.Button("Reset Anchors")
                with gr.Column(scale=3):
                    self.anchor_line_input = gr.Textbox(
                        label="Enter Anchor Position (format: i,x,y,z)",
                        placeholder="e.g. 1,1.23,4.56,7.89",
                        interactive=True,
                        value=""
                    )
                    self.anchor_line_btn = gr.Button("Add/Update Anchor Position")
                    self.anchor_history_display = gr.Textbox(
                        label="Anchor Input History",
                        lines=6,
                        interactive=False,
                        value=""
                    )

            # 綁定事件
            self.anchor_count_submit_btn.click(
                fn=self.set_anchor_count,
                inputs=[self.anchor_count_input],
                outputs=[
                    self.anchor_count_input,
                    self.anchor_line_input,
                    self.anchor_history_display,
                    self.anchor_count_submit_btn,
                    self.anchor_line_btn,
                ],
            )
            self.anchor_count_input.submit(
                fn=self.set_anchor_count,
                inputs=[self.anchor_count_input],
                outputs=[
                    self.anchor_count_input,
                    self.anchor_line_input,
                    self.anchor_history_display,
                    self.anchor_count_submit_btn,
                    self.anchor_line_btn,
                ],
            )

            # anchor line submit
            self.anchor_line_btn.click(
                fn=self.input_anchor_line,
                inputs=[self.anchor_line_input],
                outputs=[self.anchor_line_input, self.anchor_history_display, self.anchor_line_btn]
            )
            self.anchor_line_input.submit(
                fn=self.input_anchor_line,
                inputs=[self.anchor_line_input],
                outputs=[self.anchor_line_input, self.anchor_history_display, self.anchor_line_btn]
            )

            # reset 按鈕
            self.anchor_reset_btn.click(
                fn=self.reset_anchors,
                inputs=[],
                outputs=[
                    self.anchor_count_input,
                    self.anchor_line_input,
                    self.anchor_history_display,
                    self.anchor_count_submit_btn,
                    self.anchor_line_btn,
                ],
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
                with gr.Column(scale=2):
                    self.xy_plot = gr.Image(
                        value=self.create_blank_plot("UWB & Virtual Drone Position (XY view)", "X (m)", "Y (m)", xlim=(0, 5), ylim=(0, 5)),
                        label="XY Plot"
                    )
                    self.x_plot = gr.Image(
                        value=self.create_sequence_plot("X in Sequence", "Index", "X (m)", xlim=(0, 1), ylim=(0, 5)),
                        label="X in Sequence"
                    )
                with gr.Column(scale=2):
                    self.z_plot = gr.Image(
                        value=self.create_sequence_plot("Z in Sequence", "Index", "Z (m)", xlim=(0, 1), ylim=(0, 5)),
                        label="Z in Sequence"
                    )
                    self.y_plot = gr.Image(
                        value=self.create_sequence_plot("Y in Sequence", "Index", "Y (m)", xlim=(0, 1), ylim=(0, 5)),
                        label="Y in Sequence"
                    )

            self.counter = gr.State(0)
            self.timer = Timer(value=0.2)
            self.timer.tick(
                fn=self.update_and_step,
                inputs=[self.counter],
                outputs=[self.xy_plot, self.x_plot, self.y_plot, self.z_plot, self.counter]
            )

            #self.chat = gr.ChatInterface(self.process_message, fill_height=False, examples=default_sentences)
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
        tag = "[User/Virtual]" if source == "virtual" else "[Drone/UWB]" if source == "uwb" else "[Drone/Virtual]" if source == "virtual_drone" else "[Pos]"

        # 初始化紀錄字典
        if not hasattr(self, '_last_position_map'):
            self._last_position_map = {}

        current_pos = (x, y, z)
        last_pos = self._last_position_map.get(source)

        if current_pos != last_pos:
            # 有改變才放入 queue
            try:
                if source in ("uwb", "virtual_drone"):
                    print(f"[receive_position] UWB pos: {x}, {y}, {z}") # 確認 callback 有進資料
                    self.uwb_queue.put_nowait((timestamp, x, y, z))
                else:
                    self.virtual_queue.put_nowait((timestamp, x, y, z))
            except queue.Full:
                if source in ("uwb", "virtual_drone"):
                    self.uwb_queue.get()
                    self.uwb_queue.put_nowait((timestamp, x, y, z))
                else:
                    self.virtual_queue.get()
                    self.virtual_queue.put_nowait((timestamp, x, y, z))

            # 每種來源各自印出，每 5 秒一次
            if not hasattr(self, '_last_print_position_map'):
                self._last_print_position_map = {}

            last_print_time = self._last_print_position_map.get(source, 0)
            if timestamp - last_print_time > 5:
                print_t(f"{tag} x={x:.2f}, y={y:.2f}, z={z:.2f}")
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
        # `show_api` parameter was removed in newer Gradio versions; omit it for compatibility.
        self.ui.launch(server_port=50001, prevent_thread_lock=True)

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

    def create_blank_plot(self, title="Empty Plot", xlabel="X", ylabel="Y", xlim=(0, 1), ylim=(0, 1)):
        fig, ax = plt.subplots(figsize=(5, 4))
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
        xy, x, y, z = self.update_position_plot()
        counter += 1
        return xy, x, y, z, counter

    def update_position_plot(self):  
        uwb_data = list(self.uwb_queue.queue)[-1:]
        virtual_data = list(self.virtual_queue.queue)[-1:]

        df_uwb = pd.DataFrame(uwb_data, columns=["t", "x", "y", "z"]) if uwb_data else pd.DataFrame()
        df_virt = pd.DataFrame(virtual_data, columns=["t", "x", "y", "z"]) if virtual_data else pd.DataFrame()

        plot_df = pd.DataFrame()
        if not df_uwb.empty:
            df_uwb["type"] = "Drone (UWB)"
            plot_df = pd.concat([plot_df, df_uwb])
        if not df_virt.empty:
            df_virt["type"] = "User (Virtual)"
            plot_df = pd.concat([plot_df, df_virt])

        if plot_df.empty:
            return (
                self.create_blank_plot("XY Plot", "X", "Y", (0, 5), (0, 5)),
                self.create_sequence_plot("X in Sequence", "Index", "X (m)", (0, 1), (0, 5)),
                self.create_sequence_plot("Y in Sequence", "Index", "Y (m)", (0, 1), (0, 5)),
                self.create_sequence_plot("Z in Sequence", "Index", "Z (m)", (0, 1), (0, 5)),
            )

        colors_map = {"Drone (UWB)": "red", "User (Virtual)": "blue"}

        fig_xy, ax_xy = plt.subplots(figsize=(5, 4))
        for dtype, color in colors_map.items():
            subset = plot_df[plot_df["type"] == dtype]
            if not subset.empty:
                ax_xy.scatter(subset["x"], subset["y"], c=color, label=dtype, marker='o')
                latest = subset.iloc[-1]
                ax_xy.text(
                    latest["x"] + 0.05,
                    latest["y"] + 0.05,
                    f"({latest['x']:.2f}, {latest['y']:.2f})",
                    fontsize=8,
                    color=color,
                    ha='left',
                    va='bottom'
                )
            else:
                ax_xy.scatter([], [], c=color, label=dtype, marker='o')
        ax_xy.set_xlim(0, 5)
        ax_xy.set_ylim(0, 5)
        ax_xy.set_xticks([i * 0.5 for i in range(11)])
        ax_xy.set_yticks([i * 0.5 for i in range(11)])
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_title("Drone (UWB) & User (Virtual) Position (XY view)")
        ax_xy.grid(True, linestyle='--', linewidth=0.5)
        ax_xy.legend()

        buf_xy = io.BytesIO()
        fig_xy.savefig(buf_xy, format='png')
        buf_xy.seek(0)
        plt.close(fig_xy)
        img_xy = Image.open(buf_xy)

        imgs = []
        for axis in ["x", "y", "z"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            for label, color in colors_map.items():
                ax.scatter([], [], color=color, label=label, marker='o')
            for label, group in plot_df.groupby("type"):
                if len(group) >= 1:
                    x_vals = group.index / max(len(group.index) - 1, 1)
                    y_vals = group[axis]
                    ax.scatter(x_vals, y_vals, color=colors_map[label], marker='o')
                    latest_x = x_vals.iloc[-1] if isinstance(x_vals, pd.Series) else x_vals[-1]
                    latest_y = y_vals.iloc[-1] if isinstance(y_vals, pd.Series) else y_vals[-1]
                    ax.text(latest_x + 0.01, latest_y + 0.05, f"{axis}={latest_y:.2f}",
                            fontsize=8, color=colors_map[label], ha='left', va='bottom')
            ax.set_xlim(0, 1)
            ax.set_xticks([i * 0.2 for i in range(6)])
            ax.set_ylim(0, 5)
            ax.set_yticks([i * 0.5 for i in range(11)])
            ax.set_title(f"{axis.upper()} in Sequence")
            ax.set_xlabel("Index")
            ax.set_ylabel(f"{axis.upper()} (m)")
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.legend()

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            imgs.append(Image.open(buf))

        return img_xy, imgs[0], imgs[1], imgs[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_virtual_robot', action='store_true')
    parser.add_argument('--use_http', action='store_true')
    parser.add_argument('--gear', action='store_true')
    parser.add_argument('--image', action='store_true')

    args = parser.parse_args()
    robot_type = RobotType.TELLO
    if args.use_virtual_robot:
        robot_type = RobotType.VIRTUAL
    elif args.gear:
        robot_type = RobotType.GEAR

    typefly = TypeFly(robot_type, use_http=args.use_http, enable_video=args.image)
    typefly.run()

