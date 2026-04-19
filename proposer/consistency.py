from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, Optional, Set

from proposer.registry import ALLOWED_MUTATION_FILES, TRACKED_CONTRACT_FILES


class CandidateConsistencyError(ValueError):
    """Raised when generated candidate artifacts violate hard guardrails."""


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateConsistencyError(message)


def _detect_changed_files(candidate_dir: Path, parent_dir: Path) -> Set[str]:
    changed: Set[str] = set()
    for name in TRACKED_CONTRACT_FILES:
        cp = candidate_dir / name
        pp = parent_dir / name
        if cp.exists() and not pp.exists():
            changed.add(name)
            continue
        if cp.exists() and pp.exists() and _read_text(cp) != _read_text(pp):
            changed.add(name)
    return changed


def _runtime_effect_modules_from_rules(candidate_dir: Path) -> Set[str]:
    default = {"state_features.py", "trigger_logic.py", "prompt_composer.py", "state_encoder.py", "trigger_policy.py", "prompt_builder.py"}
    rules_path = candidate_dir / "validator_rules.py"
    if not rules_path.exists():
        return default
    try:
        spec = importlib.util.spec_from_file_location("candidate_validator_rules", str(rules_path))
        if spec is None or spec.loader is None:
            return default
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "runtime_effect_modules", None)
        if callable(fn):
            out = [Path(str(x)).name for x in list(fn() or [])]
            if out:
                return set(out)
    except Exception:
        return default
    return default


def _assert_importable(path: Path) -> None:
    try:
        spec = importlib.util.spec_from_file_location(f"cand_{path.stem}", str(path))
        _assert(spec is not None and spec.loader is not None, f"module not loadable: {path.name}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise CandidateConsistencyError(f"module import failed: {path.name}: {exc}") from exc


def _validate_required_artifacts(candidate_dir: Path, spec: Dict) -> None:
    for fname in ["spec.json", "proposer_note.txt"]:
        _assert((candidate_dir / fname).exists(), f"missing required file: {fname}")
    _assert(str((candidate_dir / "proposer_note.txt").read_text(encoding="utf-8")).strip(), "proposer_note.txt must not be empty")

    _assert(isinstance(spec.get("manifest"), dict), "spec.manifest is required")
    runtime_meta = dict(spec.get("runtime_metadata") or {})
    _assert(isinstance(runtime_meta, dict) and runtime_meta, "spec.runtime_metadata is required")
    _assert(isinstance(runtime_meta.get("changed_files"), list), "runtime_metadata.changed_files is required")
    _assert(str(runtime_meta.get("diff_path") or "").strip(), "runtime_metadata.diff_path is required")
    _assert((candidate_dir / str(runtime_meta.get("diff_path"))).exists(), "diff artifact missing")


def _validate_runtime_wiring(candidate_dir: Path, spec: Dict) -> None:
    sandbox = dict(spec.get("sandbox") or {})
    wiring_targets = {
        "state_features": "state_features.py",
        "trigger_logic": "trigger_logic.py",
        "prompt_composer": "prompt_composer.py",
    }
    for key, expected_file in wiring_targets.items():
        cfg = dict(sandbox.get(key) or {})
        if not cfg:
            continue
        enabled = bool(cfg.get("enabled"))
        module_name = Path(str(cfg.get("module") or "")).name
        _assert(enabled, f"sandbox.{key}.enabled must be true")
        _assert(module_name == expected_file, f"sandbox.{key}.module mismatch: {module_name} != {expected_file}")
        path = candidate_dir / module_name
        _assert(path.exists(), f"wired sandbox module missing: {module_name}")
        _assert_importable(path)


def validate_candidate_contract_alignment(
    candidate_dir: Path,
    *,
    parent_dir: Optional[Path],
    proposal_contract: Optional[Dict] = None,
) -> Dict:
    """Minimal hard guardrails for candidate acceptance.

    This intentionally avoids high-level narrative/label strictness and focuses on
    boundary-safe, executable, and runtime-wiring-safe checks.
    """
    candidate_dir = Path(candidate_dir)
    spec = _load_json(candidate_dir / "spec.json")
    contract = dict(proposal_contract or spec.get("proposal_contract") or {})

    _assert(str(spec.get("id") or "").strip(), "spec.id is required")
    _assert(str(spec.get("kind") or "").strip(), "spec.kind is required")
    _assert(str(spec.get("parent") or contract.get("parent_harness") or "").strip(), "parent metadata is required")

    _validate_required_artifacts(candidate_dir, spec)
    _validate_runtime_wiring(candidate_dir, spec)

    declared_files = [Path(str(x)).name for x in list(contract.get("files_to_create_or_modify") or []) if Path(str(x)).name in ALLOWED_MUTATION_FILES]
    if parent_dir is not None:
        changed_files = _detect_changed_files(candidate_dir, Path(parent_dir))
        _assert(changed_files, "candidate must differ from parent in tracked files")
        if declared_files:
            _assert(set(declared_files) == changed_files, f"files_to_create_or_modify mismatch: declared={sorted(declared_files)} actual={sorted(changed_files)}")

        runtime_meta_changed = {Path(str(x)).name for x in list((spec.get("runtime_metadata") or {}).get("changed_files") or [])}
        if runtime_meta_changed:
            _assert(runtime_meta_changed == changed_files, "runtime_metadata.changed_files mismatch")

        runtime_effect_modules = _runtime_effect_modules_from_rules(candidate_dir)
        _assert(bool(runtime_effect_modules.intersection(changed_files)), "candidate must modify at least one runtime-effect sandbox module")

    return {
        "declared_files": declared_files,
        "runtime_effect_modules": sorted(_runtime_effect_modules_from_rules(candidate_dir)),
    }
