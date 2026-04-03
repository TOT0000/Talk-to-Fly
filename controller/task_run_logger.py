from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
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
DEBUG_SHEET = "debug"

RUN_COLUMNS = [
    "timestamp",
    "scene_id",
    "task_text",
    "planner_mode",
    "llm_called",
    "final_plan_source",
    "generated_plan",
    "selected_target",
    "path_clear",
    "blocking_entity",
    "corridor_min_gap",
    "dominant_threat_type",
    "dominant_threat_id",
    "current_collision_probability",
    "historical_max_collision_probability",
    "task_success",
    "task_completion_time_sec",
    "min_envelope_gap_m_during_run",
    "any_envelope_overlap_during_run",
    "any_collision_during_run",
    "decision_note",
    "run_id",
    "task_id",
    "run_status",
]

EVENT_COLUMNS = [
    "run_id", "event_timestamp", "event_type", "envelope_gap_m", "distance_xy_m",
    "current_collision_probability", "historical_max_collision_probability", "safety_score", "details",
]


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
    peak_current_collision_probability_during_run: float = 0.0
    peak_historical_max_collision_probability_during_run: float = 0.0
    min_distance_xy_m_during_run: Optional[float] = None
    max_distance_xy_m_during_run: Optional[float] = None
    max_timing_freshness_s_during_run: float = 0.0

    _last_overlap_state: bool = False
    _last_collision_state: bool = False
    baseline_info: Dict = field(default_factory=dict)
    planner_info: Dict = field(default_factory=dict)


class TaskRunLogger:
    def __init__(self, excel_path: str = "logs/task_runs.xlsx"):
        self.excel_path = excel_path
        self.debug_jsonl_path = os.path.join(
            os.path.dirname(self.excel_path) or ".",
            "task_runs_debug.jsonl",
        )
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
        def _default(o):
            if is_dataclass(o):
                return asdict(o)
            if hasattr(o, "to_dict") and callable(getattr(o, "to_dict")):
                try:
                    return o.to_dict()
                except Exception:
                    pass
            if hasattr(o, "__dict__"):
                try:
                    return {
                        k: v for k, v in vars(o).items()
                        if not str(k).startswith("_")
                    }
                except Exception:
                    pass
            return str(o)
        return json.dumps(value, ensure_ascii=False, default=_default)

    def _ensure_workbook(self):
        if not self._enabled:
            self._warn_once_disabled()
            return
        os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)
        if os.path.exists(self.excel_path):
            wb = load_workbook(self.excel_path)
            self._ensure_sheet_schema(wb, RUNS_SHEET, RUN_COLUMNS)
            self._ensure_sheet_schema(wb, EVENTS_SHEET, EVENT_COLUMNS)
            self._ensure_sheet_schema(wb, DEBUG_SHEET, ["run_id", "timestamp", "debug_json"])
            wb.save(self.excel_path)
            return
        wb = Workbook()
        ws_runs = wb.active
        ws_runs.title = RUNS_SHEET
        ws_runs.append(RUN_COLUMNS)
        ws_events = wb.create_sheet(EVENTS_SHEET)
        ws_events.append(EVENT_COLUMNS)
        ws_debug = wb.create_sheet(DEBUG_SHEET)
        ws_debug.append(["run_id", "timestamp", "debug_json"])
        wb.save(self.excel_path)

    def _ensure_sheet_schema(self, wb, sheet_name: str, expected_columns):
        if sheet_name not in wb.sheetnames:
            ws = wb.create_sheet(sheet_name)
            ws.append(expected_columns)
            return
        ws = wb[sheet_name]
        headers = [cell.value for cell in ws[1]] if ws.max_row >= 1 else []
        if headers != list(expected_columns):
            del wb[sheet_name]
            ws = wb.create_sheet(sheet_name)
            ws.append(expected_columns)

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

    def update_planner_info(self, planner_info: Dict):
        if not self._enabled:
            return
        with self._lock:
            if self._active is None:
                return
            self._active.planner_info = dict(planner_info or {})

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

        current_collision_probability = float(safety_context.current_collision_probability)
        self._active.peak_current_collision_probability_during_run = max(
            self._active.peak_current_collision_probability_during_run,
            current_collision_probability,
        )
        historical_max_collision_probability = float(safety_context.historical_max_collision_probability)
        self._active.peak_historical_max_collision_probability_during_run = max(
            self._active.peak_historical_max_collision_probability_during_run,
            historical_max_collision_probability,
        )

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
            "current_collision_probability": "" if safety_context is None else float(safety_context.current_collision_probability),
            "historical_max_collision_probability": "" if safety_context is None else float(safety_context.historical_max_collision_probability),
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

        initial = active.initial_snapshot
        final = active.final_snapshot or initial
        initial_ctx = initial.get("safety_context") if initial else None
        final_ctx = final.get("safety_context") if final else None

        collision_or_abort_bool = bool(active.any_collision_during_run or run_status in {"abort", "exception", "failed"})
        planner_info = active.planner_info or {}
        selected_target = active.baseline_info.get("target_task_point", "")
        if not selected_target:
            selected_target = planner_info.get("selected_target", "")
        row = {
            "timestamp": active.start_iso,
            "scene_id": active.baseline_info.get("scene_id", active.scenario_name),
            "task_text": active.task_text,
            "planner_mode": planner_info.get("planner_mode", ""),
            "llm_called": planner_info.get("llm_called", ""),
            "final_plan_source": planner_info.get("final_plan_source", ""),
            "generated_plan": active.actual_plan_text,
            "selected_target": selected_target,
            "path_clear": active.baseline_info.get("path_clear", ""),
            "blocking_entity": active.baseline_info.get("blocking_entity", ""),
            "corridor_min_gap": active.baseline_info.get("corridor_min_gap", ""),
            "dominant_threat_type": "" if final_ctx is None else str(final_ctx.dominant_threat_type),
            "dominant_threat_id": "" if final_ctx is None else str(final_ctx.dominant_threat_id),
            "current_collision_probability": "" if final_ctx is None else float(final_ctx.current_collision_probability),
            "historical_max_collision_probability": "" if final_ctx is None else float(final_ctx.historical_max_collision_probability),
            "task_success": bool(active.task_completed_bool and active.plan_execution_success),
            "task_completion_time_sec": round(end_ts - active.start_time, 3),
            "any_envelope_overlap_during_run": bool(active.any_envelope_overlap_during_run),
            "min_envelope_gap_m_during_run": "" if active.min_envelope_gap_m_during_run is None else float(active.min_envelope_gap_m_during_run),
            "any_collision_during_run": bool(active.any_collision_during_run),
            "decision_note": active.baseline_info.get("decision_note", "") or active.failure_reason,
            "run_id": active.run_id,
            "task_id": active.task_id,
            "run_status": active.run_status,
        }

        wb = load_workbook(self.excel_path)
        ws = wb[RUNS_SHEET]
        ws.append([row[col] for col in RUN_COLUMNS])
        debug_payload = {
            "run_id": active.run_id,
            "task_id": active.task_id,
            "task_text": active.task_text,
            "scenario_name": active.scenario_name,
            "run_status": active.run_status,
            "failure_reason": active.failure_reason,
            "planner_info": planner_info,
            "baseline_info": active.baseline_info,
            "initial_snapshot": initial,
            "final_snapshot": final,
            "metrics": {
                "collision_or_abort_bool": collision_or_abort_bool,
                "overlap_event_count": int(active.overlap_event_count),
                "collision_event_count": int(active.collision_event_count),
                "max_uncertainty_scale_m_during_run": float(active.max_uncertainty_scale_m_during_run),
                "max_aoi_s_during_run": float(active.max_aoi_s_during_run),
                "worst_safety_score_during_run": float(active.worst_safety_score_during_run),
                "peak_current_collision_probability_during_run": float(active.peak_current_collision_probability_during_run),
                "peak_historical_max_collision_probability_during_run": float(active.peak_historical_max_collision_probability_during_run),
                "min_distance_xy_m_during_run": active.min_distance_xy_m_during_run,
                "max_distance_xy_m_during_run": active.max_distance_xy_m_during_run,
                "max_timing_freshness_s_during_run": active.max_timing_freshness_s_during_run,
            },
        }
        ws_debug = wb[DEBUG_SHEET]
        ws_debug.append([active.run_id, self._to_iso(end_ts), self._json_text(debug_payload)])
        wb.save(self.excel_path)
        os.makedirs(os.path.dirname(self.debug_jsonl_path) or ".", exist_ok=True)
        with open(self.debug_jsonl_path, "a", encoding="utf-8") as f:
            f.write(self._json_text(debug_payload) + "\n")

    def _warn_once_disabled(self):
        if self._warned_disabled:
            return
        self._warned_disabled = True
        print(
            "[WARN] TaskRunLogger disabled because openpyxl is not installed. "
            "Install dependency with: pip install openpyxl"
        )
