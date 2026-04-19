from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Optional


def _load_module_from_path(module_name: str, path: Path) -> Optional[ModuleType]:
    if not path.exists() or not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_callable(module: Optional[ModuleType], candidates: list[str]) -> Optional[Callable]:
    if module is None:
        return None
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _wrap_prompt_callable(fn: Optional[Callable]) -> Optional[Callable]:
    if not callable(fn):
        return None

    def _wrapped(*args, **kwargs):
        if kwargs:
            stage = kwargs.get("stage")
            task_description = kwargs.get("task_description")
            encoded_state = kwargs.get("encoded_state")
            snapshot = kwargs.get("snapshot")
            spec_payload = kwargs.get("spec")
            try:
                return fn(**kwargs)
            except TypeError:
                pass
            variants = [
                (stage, task_description, encoded_state, snapshot, spec_payload),
                (stage, task_description, encoded_state, snapshot),
                (stage, task_description, encoded_state, spec_payload),
                (stage, task_description, encoded_state),
            ]
            last_exc: Exception | None = None
            for variant in variants:
                try:
                    return fn(*variant)
                except TypeError as exc:
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
        return fn(*args, **kwargs)

    return _wrapped


def load_harness_sandbox_profile(harness_spec_path: str | None) -> Dict:
    if not harness_spec_path:
        return {"enabled": False}
    spec_path = Path(str(harness_spec_path))
    if not spec_path.exists():
        return {"enabled": False}

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    harness_dir = spec_path.parent
    sandbox_cfg = dict(payload.get("sandbox") or {})

    state_cfg = dict(sandbox_cfg.get("state_features") or {})
    trigger_cfg = dict(sandbox_cfg.get("trigger_logic") or {})
    prompt_cfg = dict(sandbox_cfg.get("prompt_composer") or {})

    state_module_name = str(state_cfg.get("module") or "")
    trigger_module_name = str(trigger_cfg.get("module") or "")
    prompt_module_name = str(prompt_cfg.get("module") or "")

    state_module = _load_module_from_path("harness_state_features", harness_dir / state_module_name) if state_module_name else None
    trigger_module = _load_module_from_path("harness_trigger_logic", harness_dir / trigger_module_name) if trigger_module_name else None
    prompt_module = _load_module_from_path("harness_prompt_composer", harness_dir / prompt_module_name) if prompt_module_name else None

    return {
        "enabled": bool(sandbox_cfg),
        "spec": payload,
        "harness_dir": str(harness_dir),
        "state_features": {
            "enabled": bool(state_cfg.get("enabled", False)),
            "module": state_module_name,
            "fn": _pick_callable(state_module, ["encode_state_features", "encode_state"]),
        },
        "trigger_logic": {
            "enabled": bool(trigger_cfg.get("enabled", False)),
            "module": trigger_module_name,
            "fn": _pick_callable(trigger_module, ["should_trigger_replan"]),
        },
        "prompt_composer": {
            "enabled": bool(prompt_cfg.get("enabled", False)),
            "module": prompt_module_name,
            "fn": _wrap_prompt_callable(_pick_callable(prompt_module, ["compose_prompt_context", "build_prompt"])),
        },
    }
