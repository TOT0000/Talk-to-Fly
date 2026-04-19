from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from controller.harness_protocol import EVALUATION_PROTOCOL_VERSION, EVALUATION_SCENE_TASK_MAPPING


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def read_manual_runs(debug_jsonl: Path) -> List[Dict]:
    records: List[Dict] = []
    with Path(debug_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            run = dict(payload.get("run_summary") or {})
            if not run:
                continue
            baseline_id = str(run.get("selected_baseline_id") or "").strip()
            if not baseline_id:
                continue
            run_id = str(run.get("run_id") or "")
            scene = str(run.get("scene_id") or run.get("baseline_scene_id") or "")
            records.append(
                {
                    "run_id": run_id,
                    "harness_id": baseline_id,
                    "scene": scene,
                    "mission_success": bool(run.get("mission_success")),
                    "run_status": str(run.get("run_status") or ""),
                    "collision_count": _safe_int(run.get("collision_count")),
                    "near_miss_count": _safe_int(run.get("near_miss_count")),
                    "completion_time_mission_sec": _safe_float(run.get("completion_time_mission_sec"), default=1e9),
                    "llm_call_count": _safe_int(run.get("llm_call_count")),
                    "replan_count": _safe_int(run.get("replan_count")),
                }
            )
    return records


def aggregate_by_harness(records: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in records:
        grouped.setdefault(row["harness_id"], []).append(row)

    out: Dict[str, Dict] = {}
    for harness_id, rows in grouped.items():
        per_scene = {}
        for scene, zone in EVALUATION_SCENE_TASK_MAPPING.items():
            srows = [r for r in rows if str(r["scene"]).upper() == scene]
            if not srows:
                continue
            per_scene[scene] = {
                "task_zone": zone,
                "runs": len(srows),
                "mission_success_rate": mean(1.0 if r["mission_success"] else 0.0 for r in srows),
                "collision_count_avg": mean(r["collision_count"] for r in srows),
                "near_miss_count_avg": mean(r["near_miss_count"] for r in srows),
                "completion_time_sec_avg": mean(r["completion_time_mission_sec"] for r in srows),
                "llm_call_count_avg": mean(r["llm_call_count"] for r in srows),
                "replan_count_avg": mean(r["replan_count"] for r in srows),
            }

        out[harness_id] = {
            "harness_id": harness_id,
            "total_runs": len(rows),
            "evaluation_protocol": {
                "version": EVALUATION_PROTOCOL_VERSION,
                "scene_to_task_zone": dict(EVALUATION_SCENE_TASK_MAPPING),
            },
            "metrics": {
                "mission_success_rate": mean(1.0 if r["mission_success"] else 0.0 for r in rows),
                "collision_count_avg": mean(r["collision_count"] for r in rows),
                "near_miss_count_avg": mean(r["near_miss_count"] for r in rows),
                "completion_time_sec_avg": mean(r["completion_time_mission_sec"] for r in rows),
                "llm_call_count_avg": mean(r["llm_call_count"] for r in rows),
                "replan_count_avg": mean(r["replan_count"] for r in rows),
            },
            "per_scene_metrics": per_scene,
        }
    return out



def select_representative_trace_paths(repo_root: Path, limit: int = 12) -> List[str]:
    runs_root = Path(repo_root) / "proposer_archive/manual_runs/runs"
    out: List[str] = []
    for p in sorted(runs_root.glob("run_*/run_*_runtime_trace.jsonl"))[:limit]:
        out.append(str(p.as_posix()))
    for p in sorted(runs_root.glob("run_*/run_*_planning_trace.jsonl"))[:limit]:
        out.append(str(p.as_posix()))
    return out
