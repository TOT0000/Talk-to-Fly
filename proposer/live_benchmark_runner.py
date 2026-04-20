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
    seed: int
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


    @staticmethod
    def _extract_prompt_source_evidence(planning_trace_path: Path) -> dict:
        if not planning_trace_path.exists():
            return {}
        try:
            for raw in planning_trace_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                if isinstance(row, dict) and row.get("evaluate_prompt_source"):
                    return dict(row.get("evaluate_prompt_source") or {})
        except Exception:
            return {}
        return {}

    @staticmethod
    def _latest_planning_row(planning_trace_path: Path) -> dict:
        if not planning_trace_path.exists():
            return {}
        last_row: dict = {}
        try:
            for raw in planning_trace_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                if isinstance(row, dict):
                    last_row = row
        except Exception:
            return {}
        return last_row

    @staticmethod
    def _classify_error_type(*, run_status: str, mission_success: bool, termination_reason: str, failure_reason: str) -> str:
        if mission_success and str(run_status).strip().lower() in {"completed", "success", "ok"}:
            return "none"
        reason_blob = f"{termination_reason} {failure_reason}".strip().lower()
        if "collision" in reason_blob:
            return "collision_failure"
        if "queue_exhausted" in reason_blob:
            return "queue_exhausted_with_unfinished"
        if "replan_interrupt" in reason_blob:
            return "replan_interrupt"
        if "timeout" in reason_blob:
            return "timeout"
        return "mission_failure_or_unknown"

    @staticmethod
    def _build_config_key_alignment_block(summary: dict, planning_row: dict) -> dict:
        observed = {
            "selected_trigger_policy_name": summary.get("selected_trigger_policy_name") or planning_row.get("selected_trigger_policy_name"),
            "selected_threshold_value": summary.get("selected_threshold_value") or planning_row.get("selected_threshold_value"),
            "selected_heartbeat_seconds": summary.get("selected_heartbeat_seconds") or planning_row.get("selected_heartbeat_seconds"),
            "selected_cooldown_seconds": summary.get("selected_cooldown_seconds") or planning_row.get("selected_cooldown_seconds"),
        }
        has_any = any(v is not None and str(v) != "" for v in observed.values())
        return {
            "status": ("observed_runtime_trigger_config" if has_any else "insufficient_evidence"),
            "observed_runtime_trigger_config": observed,
            "note": (
                "Current runtime traces expose selected trigger config values, "
                "but do not emit a full consumed-key list from trigger module internals."
            ),
        }

    def _build_evaluate_error_report(
        self,
        *,
        summary: dict,
        run_summary: dict,
        prompt_source_evidence: dict,
        planning_row: dict,
    ) -> dict:
        run_status = str(summary.get("run_status") or run_summary.get("run_status") or "unknown")
        mission_success = bool(summary.get("mission_success"))
        termination_reason = str(summary.get("termination_reason") or run_summary.get("termination_reason") or "")
        failure_reason = str(summary.get("failure_reason") or run_summary.get("failure_reason") or "")
        return {
            "run_id": str(summary.get("run_id") or run_summary.get("run_id") or ""),
            "harness_id": self.harness_id,
            "error_type": self._classify_error_type(
                run_status=run_status,
                mission_success=mission_success,
                termination_reason=termination_reason,
                failure_reason=failure_reason,
            ),
            "failure_stage": str(planning_row.get("planning_stage") or planning_row.get("llm_call_purpose") or "unknown"),
            "run_status": run_status,
            "mission_success": mission_success,
            "termination_reason": termination_reason,
            "failure_reason": failure_reason,
            "module_context": {
                "selected_harness_id": summary.get("selected_harness_id"),
                "selected_harness_spec_path": summary.get("selected_harness_spec_path"),
                "selected_trigger_policy_name": summary.get("selected_trigger_policy_name") or planning_row.get("selected_trigger_policy_name"),
                "selected_trigger_mode": summary.get("selected_trigger_mode") or planning_row.get("selected_trigger_mode"),
                "selected_prompt_module": prompt_source_evidence.get("selected_prompt_module") or planning_row.get("selected_prompt_module"),
            },
            "prompt_source": dict(prompt_source_evidence or {}),
            "config_key_alignment": self._build_config_key_alignment_block(summary, planning_row),
            "evidence_paths": {
                "planning_trace": "planning_trace.jsonl",
                "runtime_trace": "runtime_trace.jsonl",
                "metadata": "metadata.json",
            },
        }

    def _capture_latest_saved_run(self, logger, scene_id: str, zone: str, seed: int) -> RunArtifact:
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
        error_report_out = target / "evaluate_error_report.json"

        if src_runtime.exists():
            shutil.copy2(src_runtime, runtime_out)
        else:
            runtime_out.write_text("", encoding="utf-8")
        if src_planning.exists():
            shutil.copy2(src_planning, planning_out)
        else:
            planning_out.write_text("", encoding="utf-8")

        run_summary = json.loads(src_summary.read_text(encoding="utf-8")) if src_summary.exists() else {}
        debug_payload = json.loads(src_debug.read_text(encoding="utf-8")) if src_debug.exists() else {}

        prompt_source_evidence = self._extract_prompt_source_evidence(planning_out)
        latest_planning_row = self._latest_planning_row(planning_out)
        evaluate_error_report = self._build_evaluate_error_report(
            summary=summary,
            run_summary=run_summary,
            prompt_source_evidence=prompt_source_evidence,
            planning_row=latest_planning_row,
        )
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
            "evaluate_prompt_source": prompt_source_evidence,
            "evaluate_error_report": evaluate_error_report,
            "evaluate_error_report_path": error_report_out.as_posix(),
            "seed": int(seed),
        }
        metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        error_report_out.write_text(json.dumps(evaluate_error_report, ensure_ascii=False, indent=2), encoding="utf-8")

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
            seed=int(seed),
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
                for idx in range(runs):
                    controller.set_baseline_scene(scene_id)
                    controller.apply_baseline_scene()
                    controller.execute_task_description(task_text)
                    art = self._capture_latest_saved_run(logger=logger, scene_id=scene_id, zone=zone, seed=idx)
                    artifacts.append(art)
        finally:
            self._shutdown_controller(controller)

        return artifacts
