from __future__ import annotations

from typing import Dict

ERROR_CLASSES = {
    "proposal_error",
    "runtime_wiring_error",
    "metric_or_benchmark_semantics_error",
    "trace_incomplete_error",
    "infra_crash_timeout_error",
    "genuine_harness_regression",
}


def classify_failure(*, verification: Dict, run: Dict, metadata: Dict) -> str:
    if not bool(verification.get("prompt_source_bound", True)):
        return "runtime_wiring_error"
    if not bool(verification.get("active_zone_correct", True)):
        return "metric_or_benchmark_semantics_error"
    if not bool(verification.get("provenance_complete", True)):
        return "trace_incomplete_error"

    run_status = str(run.get("run_status") or "").lower()
    if "timeout" in run_status or "crash" in run_status:
        return "infra_crash_timeout_error"

    if not bool(run.get("mission_success")):
        report = dict(metadata.get("evaluate_error_report") or {})
        err_type = str(report.get("error_type") or "")
        if "contract" in err_type or "proposal" in err_type:
            return "proposal_error"
        return "genuine_harness_regression"

    return "genuine_harness_regression"
