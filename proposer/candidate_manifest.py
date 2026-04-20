from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

ChangeAxis = Literal["trigger", "state", "prompt", "runtime_adapter"]
ALLOWED_CHANGE_AXES = {"trigger", "state", "prompt", "runtime_adapter"}


@dataclass(frozen=True)
class ManifestValidationResult:
    valid: bool
    errors: List[str]


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _required_non_empty_str(payload: Dict[str, Any], key: str, errors: List[str]) -> None:
    if not str(payload.get(key) or "").strip():
        errors.append(f"missing required non-empty field: {key}")


def validate_manifest_schema(manifest: Dict[str, Any]) -> ManifestValidationResult:
    errors: List[str] = []
    _required_non_empty_str(manifest, "candidate_id", errors)

    parent_ids = manifest.get("parent_candidate_ids")
    if not isinstance(parent_ids, list):
        errors.append("parent_candidate_ids must be a list")

    axes = manifest.get("change_axes")
    if not isinstance(axes, list) or not axes:
        errors.append("change_axes must be a non-empty list")
    else:
        invalid = [x for x in axes if str(x) not in ALLOWED_CHANGE_AXES]
        if invalid:
            errors.append(f"change_axes has invalid values: {sorted(set(invalid))}")

    _required_non_empty_str(manifest, "hypothesis", errors)

    for key in [
        "expected_metric_direction",
        "safety_constraints",
        "artifact_bindings",
        "declared_files_changed",
        "evaluation_requirements",
        "rollback_conditions",
    ]:
        if key not in manifest:
            errors.append(f"missing required field: {key}")

    for dict_key in ["expected_metric_direction", "safety_constraints", "artifact_bindings", "evaluation_requirements", "rollback_conditions"]:
        if dict_key in manifest and not isinstance(manifest.get(dict_key), dict):
            errors.append(f"{dict_key} must be a dict")

    if "declared_files_changed" in manifest and not isinstance(manifest.get("declared_files_changed"), list):
        errors.append("declared_files_changed must be a list")

    return ManifestValidationResult(valid=(len(errors) == 0), errors=errors)


def load_manifest(candidate_dir: Path) -> Dict[str, Any]:
    spec_path = Path(candidate_dir) / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = dict(spec.get("candidate_manifest") or spec.get("proposal_contract") or {})
    return manifest


def build_provenance_bundle(
    *,
    candidate_id: str,
    commit_hash: str,
    config_payload: Dict[str, Any],
    prompt_asset_text: str,
    rendered_prompt_text: str,
    state_schema_version: str,
    trigger_policy_version: str,
    benchmark_pack_id: str,
    scene_id: str,
    zone_id: str,
    seed: int,
    evaluator_version: str,
    trace_schema_version: str,
) -> Dict[str, Any]:
    config_json = json.dumps(config_payload, ensure_ascii=False, sort_keys=True)
    return {
        "candidate_id": candidate_id,
        "code_commit_hash": str(commit_hash),
        "config_hash": sha256_text(config_json),
        "prompt_asset_hash": sha256_text(prompt_asset_text),
        "rendered_prompt_hash": sha256_text(rendered_prompt_text),
        "state_schema_version": str(state_schema_version),
        "trigger_policy_version": str(trigger_policy_version),
        "benchmark_pack_id": str(benchmark_pack_id),
        "scene_id": str(scene_id),
        "zone_id": str(zone_id),
        "seed": int(seed),
        "evaluator_version": str(evaluator_version),
        "trace_schema_version": str(trace_schema_version),
    }
