from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from controller.harness_protocol import EVALUATION_SCENE_TASK_MAPPING

TRACE_SCHEMA_VERSION = "runtime_trace_v1"


@dataclass(frozen=True)
class RuntimeVerificationResult:
    passed: bool
    checks: Dict[str, bool]
    errors: List[str]


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def verify_runtime_artifact(*, candidate_manifest: Dict, run_artifact: Dict, metadata: Dict) -> RuntimeVerificationResult:
    checks = {
        "prompt_source_bound": False,
        "module_binding_bound": False,
        "active_zone_correct": False,
        "heartbeat_observed": False,
        "replan_trace_observed": False,
        "provenance_complete": False,
    }
    errors: List[str] = []

    planning_rows = _load_jsonl(Path(run_artifact.get("planning_trace_path") or ""))
    runtime_rows = _load_jsonl(Path(run_artifact.get("runtime_trace_path") or ""))

    eval_prompt = dict(metadata.get("evaluate_prompt_source") or {})
    if eval_prompt.get("selected_prompt_asset_path") and eval_prompt.get("rendered_prompt_hash_sha256"):
        checks["prompt_source_bound"] = True
    else:
        errors.append("missing prompt source evidence in metadata.evaluate_prompt_source")

    run_summary = dict(metadata.get("run_summary") or {})
    if run_summary.get("selected_trigger_policy_name") or run_summary.get("selected_harness_spec_path"):
        checks["module_binding_bound"] = True
    else:
        errors.append("missing module binding evidence in run_summary")

    scene_id = str(run_artifact.get("scene_id") or "")
    zone = str(run_artifact.get("task_zone") or "")
    expected_zone = EVALUATION_SCENE_TASK_MAPPING.get(scene_id)
    if expected_zone == zone:
        checks["active_zone_correct"] = True
    else:
        errors.append(f"scene-zone mismatch: scene={scene_id} expected={expected_zone} observed={zone}")

    if any(str(r.get("planning_stage") or "") == "heartbeat" for r in planning_rows):
        checks["heartbeat_observed"] = True
    else:
        errors.append("heartbeat planning stage not observed")

    if any(bool(r.get("replan_applied")) for r in planning_rows) or any("replan" in str(r.get("event_type") or "") for r in runtime_rows):
        checks["replan_trace_observed"] = True
    else:
        errors.append("replan evidence not observed in planning/runtime trace")

    provenance = dict(metadata.get("provenance") or {})
    required_prov = [
        "code_commit_hash",
        "config_hash",
        "prompt_asset_hash",
        "rendered_prompt_hash",
        "state_schema_version",
        "trigger_policy_version",
        "benchmark_pack_id",
        "scene_id",
        "zone_id",
        "seed",
        "evaluator_version",
        "trace_schema_version",
    ]
    missing = [k for k in required_prov if k not in provenance]
    if not missing:
        checks["provenance_complete"] = True
    else:
        errors.append("missing provenance fields: " + ",".join(missing))

    passed = all(checks.values())
    return RuntimeVerificationResult(passed=passed, checks=checks, errors=errors)
