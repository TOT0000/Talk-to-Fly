from __future__ import annotations

import json
import os
import queue
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class RunArtifact:
    run_id: str
    scene_id: str
    task_zone: str
    harness_id: str
    run_status: str
    mission_success: bool
    collision_count: int
    near_miss_count: int
    completion_time_mission_sec: float | None
    llm_call_count: int
    replan_count: int
    runtime_trace_path: str
    planning_trace_path: str
    metadata_path: str


class LiveBenchmarkRunner:
    """Runs live benchmark protocol for the requested evaluation mode."""

    def __init__(self, repo_root: Path, output_root: Path, harness_id: str, evaluation_protocol: Dict):
        self.repo_root = Path(repo_root)
        self.output_root = Path(output_root)
        self.harness_id = str(harness_id)
        self.evaluation_protocol = dict(evaluation_protocol or {})
        self._pairs = list(self.evaluation_protocol.get("pairs") or [])
        self.run_dir = self.output_root / "runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_root = self.output_root / "_live_logs"
        self._log_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _task_text_for_zone(zone: str) -> str:
        # Keep text explicit so objective parser deterministically chooses exactly one zone.
        return f"Search zone {zone[-1]} only and complete all checkpoints in this zone."

    def _new_controller(self):
        xlsx = self._log_root / "task_runs.xlsx"
        os.environ["TYPEFLY_TASK_LOG_XLSX"] = xlsx.as_posix()
        os.environ["TYPEFLY_BASELINE_ID"] = self.harness_id
        from controller.abs.robot_wrapper import RobotType
        from controller.llm_controller import LLMController

        controller = LLMController(
            robot_type=RobotType.PX4_SIM,
            virtual_queue=queue.Queue(maxsize=500),
            use_http=False,
            message_queue=queue.Queue(),
            enable_video=False,
        )
        controller.set_archive_enabled(True)
        controller.set_selected_pipeline(self.harness_id)
        controller.start_robot()
        return controller

    def _shutdown_controller(self, controller):
        try:
            controller.stop_robot()
        except Exception:
            pass

    def _capture_latest_saved_run(self, logger, scene_id: str, zone: str) -> RunArtifact:
        summary = logger.get_pending_run_summary() or {}
        if not summary:
            raise RuntimeError("No pending run summary available after execution.")

        run_id = str(summary.get("run_id"))
        logger.save_pending_run()

        src_runtime = self._log_root / f"{run_id}_runtime_trace.jsonl"
        src_planning = self._log_root / f"{run_id}_planning_trace.jsonl"
        src_summary = self._log_root / f"{run_id}_summary.json"
        src_debug = self._log_root / f"{run_id}_debug.json"

        target = self.run_dir / run_id
        target.mkdir(parents=True, exist_ok=True)

        runtime_out = target / "runtime_trace.jsonl"
        planning_out = target / "planning_trace.jsonl"
        metadata_out = target / "metadata.json"

        shutil.copy2(src_runtime, runtime_out)
        shutil.copy2(src_planning, planning_out)

        run_summary = json.loads(src_summary.read_text(encoding="utf-8")) if src_summary.exists() else {}
        debug_payload = json.loads(src_debug.read_text(encoding="utf-8")) if src_debug.exists() else {}

        metadata = {
            "run_id": run_id,
            "scene_id": scene_id,
            "task_zone": zone,
            "baseline_or_candidate_id": self.harness_id,
            "evaluation_stage": self.evaluation_protocol.get("mode"),
            "evaluation_protocol_name": self.evaluation_protocol.get("name"),
            "evaluation_protocol_version": self.evaluation_protocol.get("version"),
            "evaluation_timestamp": time.time(),
            "run_summary": run_summary,
            "debug_summary": debug_payload,
        }
        metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return RunArtifact(
            run_id=run_id,
            scene_id=scene_id,
            task_zone=zone,
            harness_id=self.harness_id,
            run_status=str(summary.get("run_status") or run_summary.get("run_status") or "unknown"),
            mission_success=bool(summary.get("mission_success")),
            collision_count=int(summary.get("collision_count") or 0),
            near_miss_count=int(summary.get("near_miss_count") or 0),
            completion_time_mission_sec=summary.get("completion_time_mission_sec"),
            llm_call_count=int(run_summary.get("llm_call_count") or 0),
            replan_count=int(summary.get("replan_count") or 0),
            runtime_trace_path=runtime_out.as_posix(),
            planning_trace_path=planning_out.as_posix(),
            metadata_path=metadata_out.as_posix(),
        )

    def run(self) -> List[RunArtifact]:
        controller = self._new_controller()
        artifacts: List[RunArtifact] = []
        logger = controller.task_run_logger

        try:
            for pair in self._pairs:
                scene_id = str(pair["scene_id"])
                zone = str(pair["task_zone"])
                runs = int(pair["runs"])
                task_text = self._task_text_for_zone(zone)
                for _ in range(runs):
                    controller.set_baseline_scene(scene_id)
                    controller.apply_baseline_scene()
                    controller.execute_task_description(task_text)
                    art = self._capture_latest_saved_run(logger=logger, scene_id=scene_id, zone=zone)
                    artifacts.append(art)
        finally:
            self._shutdown_controller(controller)

        return artifacts
