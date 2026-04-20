from __future__ import annotations

from typing import Dict

ERROR_CLASSES = {
    "proposal_error",
    "runtime_wiring_error",
    "benchmark_semantics_error",
    "trace_incomplete_error",
    "infra_crash_error",
    "genuine_harness_regression",
    "unsafe_candidate_regression",
}


def classify_failure(*, verification: Dict, run: Dict, metadata: Dict) -> str:
    if not bool(verification.get("prompt_source_bound", True)):
        return "runtime_wiring_error"
    if not bool(verification.get("active_zone_correct", True)):
        return "benchmark_semantics_error"
    if not bool(verification.get("provenance_complete", True)):
        return "trace_incomplete_error"

    run_status = str(run.get("run_status") or "").lower()
    if "timeout" in run_status or "crash" in run_status:
        return "infra_crash_error"

    report = dict(metadata.get("evaluate_error_report") or {})
    err_type = str(report.get("error_type") or "").lower()

    if any(k in err_type for k in ("contract", "proposal")):
        return "proposal_error"

    collision = float(run.get("collision_count", 0) or 0)
    near_miss = float(run.get("near_miss_count", 0) or 0)
    if collision > 0 or near_miss > 0:
        return "unsafe_candidate_regression"

    if not bool(run.get("mission_success")):
        return "genuine_harness_regression"

    return "genuine_harness_regression"
