from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

try:
    from openpyxl import Workbook, load_workbook
    _OPENPYXL_AVAILABLE = True
except Exception:
    Workbook = None
    load_workbook = None
    _OPENPYXL_AVAILABLE = False


RUNS_SHEET = "runs"
EVENTS_SHEET = "events"

RUN_COLUMNS = [
    "timestamp", "scene_id", "target_task_point", "drone_initial_position", "user_position",
    "obstacle_positions_summary", "task_point_positions_summary", "path_clear", "blocking_entity", "corridor_min_gap",
    "chosen_motion_mode", "direct_go_to", "staged_detour", "recovery_first", "unknown",
    "generated_control_plan", "task_start_time", "task_end_time", "task_success",
    "min_clearance_during_run", "any_overlap", "any_collision", "notes", "debug_reason",
    "decision_note",
    "run_id", "task_id", "task_text", "scenario_name", "start_timestamp", "end_timestamp", "task_completion_time_sec",
    "task_completed_bool", "timeout_bool", "collision_or_abort_bool", "run_status", "failure_reason",
    "plan_generation_success", "plan_execution_success", "actual_plan_text",
    "initial_drone_gt_position_3d", "initial_drone_est_position_3d", "initial_user_gt_position_3d", "initial_user_est_position_3d",
    "initial_safety_score", "initial_safety_level", "initial_planning_bias", "initial_envelope_gap_m", "initial_uncertainty_scale_m",
    "initial_envelopes_overlap", "initial_reason_tags", "initial_max_aoi_s",
    "final_drone_gt_position_3d", "final_drone_est_position_3d", "final_user_gt_position_3d", "final_user_est_position_3d",
    "final_safety_score", "final_safety_level", "final_planning_bias", "final_envelope_gap_m", "final_uncertainty_scale_m",
    "final_envelopes_overlap", "final_reason_tags", "final_max_aoi_s",
    "any_envelope_overlap_during_run", "overlap_event_count", "min_envelope_gap_m_during_run",
    "any_collision_during_run", "collision_event_count",
    "max_uncertainty_scale_m_during_run", "max_aoi_s_during_run", "worst_safety_score_during_run",
    "worst_safety_level_during_run", "any_level_drop_during_run", "safety_level_transition_trace",
    "initial_distance_xy_m", "final_distance_xy_m", "min_distance_xy_m_during_run", "max_distance_xy_m_during_run",
    "initial_timing_freshness_s", "final_timing_freshness_s", "max_timing_freshness_s_during_run",
]

EVENT_COLUMNS = [
    "run_id", "event_timestamp", "event_type", "envelope_gap_m", "distance_xy_m", "safety_level", "safety_score", "details",
]

LEVEL_RANK = {"SAFE": 3, "CAUTION": 2, "WARNING": 1, "DANGER": 0}


@dataclass
class _RunRecord:
    run_id: str
    task_id: str
    task_text: str
    scenario_name: str
    start_time: float
    start_iso: str
    run_status: str = "running"
    plan_generation_success: bool = False
    plan_execution_success: bool = False
    actual_plan_text: str = ""
    timeout_bool: bool = False
    failure_reason: str = ""
    task_completed_bool: bool = False

    initial_snapshot: Dict = field(default_factory=dict)
    final_snapshot: Dict = field(default_factory=dict)

    any_envelope_overlap_during_run: bool = False
    overlap_event_count: int = 0
    min_envelope_gap_m_during_run: Optional[float] = None
    any_collision_during_run: bool = False
    collision_event_count: int = 0
    max_uncertainty_scale_m_during_run: float = 0.0
    max_aoi_s_during_run: float = 0.0
    worst_safety_score_during_run: float = 1.0
    worst_safety_level_during_run: str = "SAFE"
    any_level_drop_during_run: bool = False
    safety_level_transition_trace: str = ""
    min_distance_xy_m_during_run: Optional[float] = None
    max_distance_xy_m_during_run: Optional[float] = None
    max_timing_freshness_s_during_run: float = 0.0

    _level_trace: list = field(default_factory=list)
    _last_overlap_state: bool = False
    _last_collision_state: bool = False
    baseline_info: Dict = field(default_factory=dict)


class TaskRunLogger:
    def __init__(self, excel_path: str = "logs/task_runs.xlsx"):
        self.excel_path = excel_path
        self._lock = threading.Lock()
        self._active: Optional[_RunRecord] = None
        self._enabled = _OPENPYXL_AVAILABLE
        self._warned_disabled = False
        self._ensure_workbook()

    @staticmethod
    def _to_iso(ts: float) -> str:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()

    @staticmethod
    def _to_pos(value):
        if value is None:
            return ""
        return json.dumps([round(float(v), 4) for v in value])

    @staticmethod
    def _json_text(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _ensure_workbook(self):
        if not self._enabled:
            self._warn_once_disabled()
            return
        os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)
        if os.path.exists(self.excel_path):
            wb = load_workbook(self.excel_path)
            if RUNS_SHEET not in wb.sheetnames:
                ws = wb.create_sheet(RUNS_SHEET)
                ws.append(RUN_COLUMNS)
            if EVENTS_SHEET not in wb.sheetnames:
                ws = wb.create_sheet(EVENTS_SHEET)
                ws.append(EVENT_COLUMNS)
            wb.save(self.excel_path)
            return
        wb = Workbook()
        ws_runs = wb.active
        ws_runs.title = RUNS_SHEET
        ws_runs.append(RUN_COLUMNS)
        ws_events = wb.create_sheet(EVENTS_SHEET)
        ws_events.append(EVENT_COLUMNS)
        wb.save(self.excel_path)

    def start_run(self, task_id: str, task_text: str, scenario_name: str, initial_snapshot: Dict):
        if not self._enabled:
            return
        with self._lock:
            if self._active is not None:
                return
            now = time.time()
            self._active = _RunRecord(
                run_id=f"run_{uuid4().hex[:12]}",
                task_id=task_id,
                task_text=task_text,
                scenario_name=scenario_name,
                start_time=now,
                start_iso=self._to_iso(now),
                initial_snapshot=initial_snapshot or {},
            )
            self._consume_snapshot(initial_snapshot, now=now)

    def update_plan_info(self, plan_text: str, generation_success: bool):
        if not self._enabled:
            return
        with self._lock:
            if self._active is None:
                return
            self._active.actual_plan_text = plan_text or ""
            self._active.plan_generation_success = bool(generation_success)

    def update_execution_info(self, execution_success: bool, failure_reason: str = "", timeout_bool: bool = False, task_completed: bool = False):
        if not self._enabled:
            return
        with self._lock:
            if self._active is None:
                return
            self._active.plan_execution_success = bool(execution_success)
            self._active.timeout_bool = bool(timeout_bool)
            self._active.task_completed_bool = bool(task_completed)
            if failure_reason:
                self._active.failure_reason = str(failure_reason)

    def update_baseline_info(self, baseline_info: Dict):
        if not self._enabled:
            return
        with self._lock:
            if self._active is None:
                return
            self._active.baseline_info = dict(baseline_info or {})

    def consume_runtime_snapshot(self, snapshot: Dict):
        if not self._enabled:
            return
        with self._lock:
            if self._active is None:
                return
            self._consume_snapshot(snapshot, now=time.time())

    def _consume_snapshot(self, snapshot: Dict, now: float):
        if snapshot is None or self._active is None:
            return
        self._active.final_snapshot = snapshot
        safety_context = snapshot.get("safety_context")
        if safety_context is None:
            return

        gap = float(safety_context.envelope_gap_m)
        overlap_now = bool(safety_context.envelopes_overlap or gap < 0.0)
        if self._active.min_envelope_gap_m_during_run is None or gap < self._active.min_envelope_gap_m_during_run:
            self._active.min_envelope_gap_m_during_run = gap
        if overlap_now and not self._active._last_overlap_state:
            self._active.overlap_event_count += 1
            self._append_event(now, "envelope_overlap", snapshot)
        self._active._last_overlap_state = overlap_now
        self._active.any_envelope_overlap_during_run = self._active.any_envelope_overlap_during_run or overlap_now

        distance_xy = float(safety_context.drone_to_user_distance_xy)
        if self._active.min_distance_xy_m_during_run is None or distance_xy < self._active.min_distance_xy_m_during_run:
            self._active.min_distance_xy_m_during_run = distance_xy
        if self._active.max_distance_xy_m_during_run is None or distance_xy > self._active.max_distance_xy_m_during_run:
            self._active.max_distance_xy_m_during_run = distance_xy

        uncertainty = float(safety_context.uncertainty_scale_m)
        self._active.max_uncertainty_scale_m_during_run = max(self._active.max_uncertainty_scale_m_during_run, uncertainty)

        max_aoi_s = float(safety_context.max_aoi_s or 0.0)
        self._active.max_aoi_s_during_run = max(self._active.max_aoi_s_during_run, max_aoi_s)
        timing_freshness = float(safety_context.timing_freshness_s or 0.0)
        self._active.max_timing_freshness_s_during_run = max(self._active.max_timing_freshness_s_during_run, timing_freshness)

        score = float(safety_context.safety_score)
        if score < self._active.worst_safety_score_during_run:
            self._active.worst_safety_score_during_run = score

        level = str(safety_context.safety_level)
        if not self._active._level_trace or self._active._level_trace[-1] != level:
            self._active._level_trace.append(level)
        prev_rank = LEVEL_RANK.get(self._active.worst_safety_level_during_run, 3)
        cur_rank = LEVEL_RANK.get(level, 3)
        if cur_rank < prev_rank:
            self._active.worst_safety_level_during_run = level
            self._active.any_level_drop_during_run = True

        collision_now = self._detect_collision(snapshot)
        if collision_now and not self._active._last_collision_state:
            self._active.collision_event_count += 1
            self._append_event(now, "collision", snapshot)
        self._active._last_collision_state = collision_now
        self._active.any_collision_during_run = self._active.any_collision_during_run or collision_now

    def _detect_collision(self, snapshot: Dict) -> bool:
        drone_gt = snapshot.get("drone_gt")
        user_gt = snapshot.get("user_gt")
        if drone_gt is None or user_gt is None:
            return False
        dx = float(drone_gt[0] - user_gt[0])
        dy = float(drone_gt[1] - user_gt[1])
        dz = float(drone_gt[2] - user_gt[2])
        distance_3d = (dx * dx + dy * dy + dz * dz) ** 0.5
        threshold = float(os.getenv("TYPEFLY_COLLISION_DISTANCE_M", "0.30"))
        return distance_3d <= threshold

    def _append_event(self, now: float, event_type: str, snapshot: Dict):
        if not self._enabled:
            return
        safety_context = snapshot.get("safety_context")
        details = {
            "scenario": self._active.scenario_name if self._active else "",
            "drone_gt": snapshot.get("drone_gt"),
            "user_gt": snapshot.get("user_gt"),
        }
        row = {
            "run_id": self._active.run_id if self._active else "",
            "event_timestamp": self._to_iso(now),
            "event_type": event_type,
            "envelope_gap_m": "" if safety_context is None else float(safety_context.envelope_gap_m),
            "distance_xy_m": "" if safety_context is None else float(safety_context.drone_to_user_distance_xy),
            "safety_level": "" if safety_context is None else str(safety_context.safety_level),
            "safety_score": "" if safety_context is None else float(safety_context.safety_score),
            "details": self._json_text(details),
        }
        wb = load_workbook(self.excel_path)
        ws = wb[EVENTS_SHEET]
        ws.append([row[col] for col in EVENT_COLUMNS])
        wb.save(self.excel_path)

    def end_run(self, run_status: str, failure_reason: str = ""):
        if not self._enabled:
            return
        with self._lock:
            if self._active is None:
                return
            active = self._active
            self._active = None

        end_ts = time.time()
        active.run_status = run_status
        if failure_reason and not active.failure_reason:
            active.failure_reason = failure_reason
        active.safety_level_transition_trace = " -> ".join(active._level_trace)

        initial = active.initial_snapshot
        final = active.final_snapshot or initial
        initial_ctx = initial.get("safety_context") if initial else None
        final_ctx = final.get("safety_context") if final else None

        collision_or_abort_bool = bool(active.any_collision_during_run or run_status in {"abort", "exception", "failed"})
        row = {
            "timestamp": active.start_iso,
            "scene_id": active.baseline_info.get("scene_id", active.scenario_name),
            "target_task_point": active.baseline_info.get("target_task_point", ""),
            "drone_initial_position": self._to_pos(initial.get("drone_gt") if initial else None),
            "user_position": self._to_pos(initial.get("user_gt") if initial else None),
            "obstacle_positions_summary": self._json_text(active.baseline_info.get("obstacle_positions_summary", "")),
            "task_point_positions_summary": self._json_text(active.baseline_info.get("task_point_positions_summary", "")),
            "path_clear": active.baseline_info.get("path_clear", ""),
            "blocking_entity": active.baseline_info.get("blocking_entity", ""),
            "corridor_min_gap": active.baseline_info.get("corridor_min_gap", ""),
            "chosen_motion_mode": active.baseline_info.get("chosen_motion_mode", ""),
            "direct_go_to": active.baseline_info.get("direct_go_to", ""),
            "staged_detour": active.baseline_info.get("staged_detour", ""),
            "recovery_first": active.baseline_info.get("recovery_first", ""),
            "unknown": active.baseline_info.get("unknown", ""),
            "generated_control_plan": active.baseline_info.get("generated_control_plan", active.actual_plan_text),
            "task_start_time": active.start_iso,
            "task_end_time": self._to_iso(end_ts),
            "task_success": bool(active.task_completed_bool and active.plan_execution_success),
            "min_clearance_during_run": "" if active.min_distance_xy_m_during_run is None else float(active.min_distance_xy_m_during_run),
            "any_overlap": bool(active.any_envelope_overlap_during_run),
            "any_collision": bool(active.any_collision_during_run),
            "notes": active.baseline_info.get("decision_note", ""),
            "debug_reason": active.failure_reason or "",
            "decision_note": active.baseline_info.get("decision_note", ""),
            "run_id": active.run_id,
            "task_id": active.task_id,
            "task_text": active.task_text,
            "scenario_name": active.scenario_name,
            "start_timestamp": active.start_iso,
            "end_timestamp": self._to_iso(end_ts),
            "task_completion_time_sec": round(end_ts - active.start_time, 3),
            "task_completed_bool": bool(active.task_completed_bool),
            "timeout_bool": bool(active.timeout_bool),
            "collision_or_abort_bool": collision_or_abort_bool,
            "run_status": active.run_status,
            "failure_reason": active.failure_reason,
            "plan_generation_success": bool(active.plan_generation_success),
            "plan_execution_success": bool(active.plan_execution_success),
            "actual_plan_text": active.actual_plan_text,
            "initial_drone_gt_position_3d": self._to_pos(initial.get("drone_gt") if initial else None),
            "initial_drone_est_position_3d": self._to_pos(initial.get("drone_est") if initial else None),
            "initial_user_gt_position_3d": self._to_pos(initial.get("user_gt") if initial else None),
            "initial_user_est_position_3d": self._to_pos(initial.get("user_est") if initial else None),
            "initial_safety_score": "" if initial_ctx is None else float(initial_ctx.safety_score),
            "initial_safety_level": "" if initial_ctx is None else str(initial_ctx.safety_level),
            "initial_planning_bias": "" if initial_ctx is None else str(initial_ctx.planning_bias),
            "initial_envelope_gap_m": "" if initial_ctx is None else float(initial_ctx.envelope_gap_m),
            "initial_uncertainty_scale_m": "" if initial_ctx is None else float(initial_ctx.uncertainty_scale_m),
            "initial_envelopes_overlap": "" if initial_ctx is None else bool(initial_ctx.envelopes_overlap),
            "initial_reason_tags": "" if initial_ctx is None else self._json_text(initial_ctx.reason_tags),
            "initial_max_aoi_s": "" if initial_ctx is None else float(initial_ctx.max_aoi_s or 0.0),
            "final_drone_gt_position_3d": self._to_pos(final.get("drone_gt") if final else None),
            "final_drone_est_position_3d": self._to_pos(final.get("drone_est") if final else None),
            "final_user_gt_position_3d": self._to_pos(final.get("user_gt") if final else None),
            "final_user_est_position_3d": self._to_pos(final.get("user_est") if final else None),
            "final_safety_score": "" if final_ctx is None else float(final_ctx.safety_score),
            "final_safety_level": "" if final_ctx is None else str(final_ctx.safety_level),
            "final_planning_bias": "" if final_ctx is None else str(final_ctx.planning_bias),
            "final_envelope_gap_m": "" if final_ctx is None else float(final_ctx.envelope_gap_m),
            "final_uncertainty_scale_m": "" if final_ctx is None else float(final_ctx.uncertainty_scale_m),
            "final_envelopes_overlap": "" if final_ctx is None else bool(final_ctx.envelopes_overlap),
            "final_reason_tags": "" if final_ctx is None else self._json_text(final_ctx.reason_tags),
            "final_max_aoi_s": "" if final_ctx is None else float(final_ctx.max_aoi_s or 0.0),
            "any_envelope_overlap_during_run": bool(active.any_envelope_overlap_during_run),
            "overlap_event_count": int(active.overlap_event_count),
            "min_envelope_gap_m_during_run": "" if active.min_envelope_gap_m_during_run is None else float(active.min_envelope_gap_m_during_run),
            "any_collision_during_run": bool(active.any_collision_during_run),
            "collision_event_count": int(active.collision_event_count),
            "max_uncertainty_scale_m_during_run": float(active.max_uncertainty_scale_m_during_run),
            "max_aoi_s_during_run": float(active.max_aoi_s_during_run),
            "worst_safety_score_during_run": float(active.worst_safety_score_during_run),
            "worst_safety_level_during_run": active.worst_safety_level_during_run,
            "any_level_drop_during_run": bool(active.any_level_drop_during_run),
            "safety_level_transition_trace": active.safety_level_transition_trace,
            "initial_distance_xy_m": "" if initial_ctx is None else float(initial_ctx.drone_to_user_distance_xy),
            "final_distance_xy_m": "" if final_ctx is None else float(final_ctx.drone_to_user_distance_xy),
            "min_distance_xy_m_during_run": "" if active.min_distance_xy_m_during_run is None else float(active.min_distance_xy_m_during_run),
            "max_distance_xy_m_during_run": "" if active.max_distance_xy_m_during_run is None else float(active.max_distance_xy_m_during_run),
            "initial_timing_freshness_s": "" if initial_ctx is None else float(initial_ctx.timing_freshness_s or 0.0),
            "final_timing_freshness_s": "" if final_ctx is None else float(final_ctx.timing_freshness_s or 0.0),
            "max_timing_freshness_s_during_run": float(active.max_timing_freshness_s_during_run),
        }

        wb = load_workbook(self.excel_path)
        ws = wb[RUNS_SHEET]
        ws.append([row[col] for col in RUN_COLUMNS])
        wb.save(self.excel_path)

    def _warn_once_disabled(self):
        if self._warned_disabled:
            return
        self._warned_disabled = True
        print(
            "[WARN] TaskRunLogger disabled because openpyxl is not installed. "
            "Install dependency with: pip install openpyxl"
        )
