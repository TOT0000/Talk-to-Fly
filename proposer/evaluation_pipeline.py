from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from proposer.error_classifier import classify_failure


class ScreeningEvaluator:
    name = "screening"
    evaluator_version = "screening_v1"

    def should_promote(self, candidate_summary: Dict, baseline_summary: Dict) -> bool:
        cand = dict(candidate_summary.get("metrics") or {})
        base = dict(baseline_summary.get("metrics") or {})
        # safety-first gate: collision and near-miss cannot be worse than baseline.
        return float(cand.get("collision_count_avg", 9999)) <= float(base.get("collision_count_avg", 9999)) and float(
            cand.get("near_miss_count_avg", 9999)
        ) <= float(base.get("near_miss_count_avg", 9999))


class FormalEvaluator:
    name = "formal"
    evaluator_version = "formal_v1"


def build_failure_dossier(*, candidate_id: str, baseline_id: str, run: Dict, metadata: Dict, verification: Dict) -> Dict:
    classification = classify_failure(verification=verification, run=run, metadata=metadata)
    report = dict(metadata.get("evaluate_error_report") or {})
    return {
        "candidate_id": candidate_id,
        "baseline_id": baseline_id,
        "scene": run.get("scene_id"),
        "zone": run.get("task_zone"),
        "seed": metadata.get("provenance", {}).get("seed", 0),
        "termination_reason": report.get("termination_reason") or report.get("failure_reason") or run.get("run_status"),
        "metrics": {
            "collision_count": run.get("collision_count"),
            "near_miss_count": run.get("near_miss_count"),
            "mission_success": run.get("mission_success"),
            "completion_time_mission_sec": run.get("completion_time_mission_sec"),
            "replan_count": run.get("replan_count"),
            "llm_calls": run.get("llm_call_count"),
        },
        "key_trace_paths": {
            "runtime": run.get("runtime_trace_path"),
            "planning": run.get("planning_trace_path"),
            "metadata": run.get("metadata_path"),
        },
        "error_classification": classification,
        "harness_vs_system_reason": (
            "runtime/semantics/trace failure indicates harness-system integration issue"
            if classification in {"runtime_wiring_error", "metric_or_benchmark_semantics_error", "trace_incomplete_error"}
            else "task-level mission failure with complete trace; likely genuine harness regression"
        ),
    }


def write_dossier(path: Path, dossier: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
