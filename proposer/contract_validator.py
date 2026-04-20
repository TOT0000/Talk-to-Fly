from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from proposer.candidate_manifest import validate_manifest_schema


class ContractValidationError(ValueError):
    pass


def _read_spec(candidate_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(candidate_dir) / "spec.json").read_text(encoding="utf-8"))


def _detect_changed_files(candidate_dir: Path, parent_dir: Path, declared_files: List[str]) -> Set[str]:
    changed: Set[str] = set()
    for name in set([Path(x).name for x in declared_files]):
        cp = Path(candidate_dir) / name
        pp = Path(parent_dir) / name
        if cp.exists() and (not pp.exists()):
            changed.add(name)
            continue
        if cp.exists() and pp.exists() and cp.read_text(encoding="utf-8") != pp.read_text(encoding="utf-8"):
            changed.add(name)
    return changed


def validate_candidate_contract(candidate_dir: Path, parent_dir: Path | None = None) -> Dict[str, Any]:
    spec = _read_spec(candidate_dir)
    manifest = dict(spec.get("candidate_manifest") or spec.get("proposal_contract") or {})

    schema = validate_manifest_schema(manifest)
    if not schema.valid:
        raise ContractValidationError("manifest schema validation failed: " + "; ".join(schema.errors))

    declared = [Path(x).name for x in list(manifest.get("declared_files_changed") or [])]
    if not declared:
        raise ContractValidationError("declared_files_changed is empty")

    for axis in manifest.get("change_axes") or []:
        if axis not in {"trigger", "state", "prompt", "runtime_adapter"}:
            raise ContractValidationError(f"invalid change axis: {axis}")

    artifact_bindings = dict(manifest.get("artifact_bindings") or {})
    for key, rel in artifact_bindings.items():
        if not str(rel).strip():
            raise ContractValidationError(f"artifact binding path empty: {key}")
        if not (Path(candidate_dir) / str(rel)).exists():
            raise ContractValidationError(f"artifact binding not found: {key} -> {rel}")

    undeclared: List[str] = []
    if parent_dir is not None:
        observed = _detect_changed_files(candidate_dir=Path(candidate_dir), parent_dir=Path(parent_dir), declared_files=declared)
        if observed != set(declared):
            undeclared = sorted(observed.symmetric_difference(set(declared)))
            raise ContractValidationError(
                f"declared_files_changed mismatch observed; declared={sorted(set(declared))} observed={sorted(observed)}"
            )

    return {
        "candidate_id": manifest.get("candidate_id"),
        "declared_files_changed": sorted(set(declared)),
        "observed_undeclared": undeclared,
        "change_axes": list(manifest.get("change_axes") or []),
    }
