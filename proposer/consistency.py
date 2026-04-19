from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from proposer.registry import ALLOWED_MUTATION_FILES


class CandidateConsistencyError(ValueError):
    """Raised when generated candidate artifacts are not contract-aligned."""


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateConsistencyError(message)


def _canonicalize_field_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def _field_expected_in_spec(expected: str, spec_fields: List[str]) -> bool:
    expected_norm = _canonicalize_field_name(expected)
    spec_norm = {_canonicalize_field_name(v) for v in spec_fields}
    if expected_norm in spec_norm:
        return True

    expected_tokens = [t for t in expected_norm.split("_") if t]
    if not expected_tokens:
        return False
    for candidate in spec_norm:
        candidate_tokens = set([t for t in candidate.split("_") if t])
        if set(expected_tokens).issubset(candidate_tokens):
            return True
    return False


def _normalize_style_value(value: str) -> str:
    raw = _canonicalize_field_name(value)
    aliases = {
        "full": "risk_aware",
        "detailed": "risk_aware",
        "rich": "risk_aware",
        "compact": "structured",
    }
    return aliases.get(raw, raw)


def _normalized_contract_files(contract: Dict) -> List[str]:
    raw = list(contract.get("files_to_create_or_modify") or [])
    out: List[str] = []
    for item in raw:
        name = Path(str(item)).name
        if name in ALLOWED_MUTATION_FILES and name not in out:
            out.append(name)
    return out


def _detect_changed_files(candidate_dir: Path, parent_dir: Path) -> Set[str]:
    changed: Set[str] = set()
    for name in ALLOWED_MUTATION_FILES:
        cp = candidate_dir / name
        pp = parent_dir / name
        if cp.exists() and not pp.exists():
            changed.add(name)
            continue
        if cp.exists() and pp.exists() and _read_text(cp) != _read_text(pp):
            changed.add(name)
    return changed


def _validate_contract_vs_spec(contract: Dict, spec: Dict) -> None:
    impl = dict(contract.get("implementation_contract") or {})

    trigger_claim = dict(impl.get("trigger_policy") or {})
    trigger_spec = dict(spec.get("trigger_policy") or {})
    for key in ["type", "heartbeat_seconds", "threshold", "strictly_greater", "consecutive_high_risk", "hysteresis"]:
        if key in trigger_claim:
            _assert(
                trigger_spec.get(key) == trigger_claim.get(key),
                f"trigger_policy mismatch for '{key}': contract={trigger_claim.get(key)!r} spec={trigger_spec.get(key)!r}",
            )

    state_claim = dict(impl.get("state_encoder") or {})
    state_spec = dict(spec.get("state_encoder") or {})
    for key in ["summary_style", "include_risk_related", "include_targets", "include_geometry_flags"]:
        if key in state_claim:
            claim_value = state_claim.get(key)
            spec_value = state_spec.get(key)
            if key == "summary_style":
                claim_value = _normalize_style_value(str(claim_value))
                spec_value = _normalize_style_value(str(spec_value))
                if claim_value not in {"risk_aware", "structured"}:
                    # Some proposer outputs may put baseline labels here (e.g. "baseline2");
                    # treat those as non-actionable style descriptors instead of hard mismatches.
                    continue
            _assert(
                spec_value == claim_value,
                f"state_encoder mismatch for '{key}': contract={state_claim.get(key)!r} spec={state_spec.get(key)!r}",
            )
    include_subset = list(state_claim.get("include_fields_contains") or [])
    state_include_fields = list(state_spec.get("include_fields") or [])
    for item in include_subset:
        _assert(
            _field_expected_in_spec(str(item), state_include_fields),
            f"state_encoder.include_fields missing required field: {item}",
        )

    prompt_claim = dict(impl.get("prompt_builder") or {})
    prompt_spec = dict(spec.get("prompt_builder") or {})
    for key in ["template_family", "include_example", "example_family"]:
        if key in prompt_claim:
            _assert(
                prompt_spec.get(key) == prompt_claim.get(key),
                f"prompt_builder mismatch for '{key}': contract={prompt_claim.get(key)!r} spec={prompt_spec.get(key)!r}",
            )


def _validate_spec_vs_module_behavior(candidate_dir: Path, spec: Dict) -> None:
    trigger_cfg = dict(spec.get("trigger_policy") or {})
    trigger_code = _read_text(candidate_dir / "trigger_logic.py") or _read_text(candidate_dir / "trigger_policy.py")
    trigger_type = str(trigger_cfg.get("type") or "")

    if trigger_type == "event_predicted_collision_probability":
        _assert("event_predicted_collision_probability" in trigger_code, "trigger_policy.py missing event trigger branch")
        _assert("threshold" in trigger_code, "trigger_policy.py missing threshold usage for event trigger")
    if trigger_cfg.get("threshold") is not None:
        _assert("threshold" in trigger_code, "trigger_policy.py must reference threshold when spec defines threshold")
    if trigger_type in {"periodic", "hybrid", "heartbeat"}:
        _assert(
            ("periodic_controller_driven" in trigger_code) or ("heartbeat" in trigger_code),
            "trigger_policy.py missing periodic/heartbeat handling for periodic or hybrid policy",
        )

    state_cfg = dict(spec.get("state_encoder") or {})
    state_code = _read_text(candidate_dir / "state_features.py") or _read_text(candidate_dir / "state_encoder.py")
    _assert("include_fields" in state_code, "state_encoder.py missing include_fields handling")
    if state_cfg.get("include_risk_related"):
        _assert(
            "predicted_collision_probability" in state_code,
            "state_encoder.py missing risk-related handling required by spec",
        )

    prompt_cfg = dict(spec.get("prompt_builder") or {})
    prompt_code = _read_text(candidate_dir / "prompt_composer.py") or _read_text(candidate_dir / "prompt_builder.py")
    _assert("template_family" in prompt_code, "prompt_builder.py missing template_family handling")
    if prompt_cfg.get("include_example"):
        _assert("include_example" in prompt_code, "prompt_builder.py missing include_example handling")


def _validate_note_grounding(note_text: str, contract: Dict, spec: Dict) -> None:
    hypothesis = str(contract.get("one_sentence_hypothesis") or "").strip()
    trigger_type = str(((spec.get("trigger_policy") or {}).get("type") or "")).strip()
    _assert(note_text.strip(), "proposer_note.txt must not be empty")
    if hypothesis:
        _assert(hypothesis in note_text, "proposer_note.txt must include the final hypothesis from proposal_contract")
    if trigger_type:
        _assert(trigger_type in note_text, "proposer_note.txt must mention the implemented trigger_policy.type")


def validate_candidate_contract_alignment(
    candidate_dir: Path,
    *,
    parent_dir: Optional[Path],
    proposal_contract: Optional[Dict] = None,
) -> Dict:
    candidate_dir = Path(candidate_dir)
    spec = _load_json(candidate_dir / "spec.json")
    contract = dict(proposal_contract or spec.get("proposal_contract") or {})

    _assert(str(contract.get("parent_harness") or "").strip(), "proposal_contract.parent_harness is required")
    _assert(str(contract.get("one_sentence_hypothesis") or "").strip(), "proposal_contract.one_sentence_hypothesis is required")

    declared_files = _normalized_contract_files(contract)
    _assert(declared_files, "proposal_contract.files_to_create_or_modify must be non-empty")

    for required in ["spec.json", "proposer_note.txt"]:
        _assert(required in declared_files, f"proposal_contract.files_to_create_or_modify must include {required}")

    if parent_dir is not None:
        changed_files = _detect_changed_files(candidate_dir, Path(parent_dir))
        _assert(changed_files, "candidate must differ from parent in at least one allowed file")
        _assert(set(declared_files) == changed_files, f"files_to_create_or_modify mismatch: declared={sorted(declared_files)} actual={sorted(changed_files)}")
        runtime_effect_modules = {
            "state_features.py",
            "trigger_logic.py",
            "prompt_composer.py",
            # legacy runtime-effect module names (kept for compatibility)
            "state_encoder.py",
            "trigger_policy.py",
            "prompt_builder.py",
        }
        _assert(
            bool(runtime_effect_modules.intersection(changed_files)),
            "candidate must modify at least one runtime-effect sandbox module",
        )

    _validate_contract_vs_spec(contract, spec)
    _validate_spec_vs_module_behavior(candidate_dir, spec)
    _validate_note_grounding(_read_text(candidate_dir / "proposer_note.txt"), contract, spec)

    return {
        "declared_files": declared_files,
        "trigger_type": (spec.get("trigger_policy") or {}).get("type"),
    }
