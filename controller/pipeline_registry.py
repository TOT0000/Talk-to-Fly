from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PipelineConfig:
    id: str
    name: str
    description: str
    base_mode: str
    trigger_type: str
    trigger_params: Dict[str, float]
    prompt_variant: str
    example_variant: str
    state_fields: List[str]
    use_output_example: bool
    replan_cap: int
    archive_enabled_default: Optional[bool] = None
    harness_spec_path: Optional[str] = None
    kind: str = "baseline"
    parent_id: Optional[str] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_harness_specs(root: Path):
    for spec_path in sorted((root / "harnesses").glob("baseline*/spec.json")):
        yield spec_path
    for spec_path in sorted((root / "harnesses/candidates").glob("candidate_*/spec.json")):
        yield spec_path


def _load_pipeline_registry_from_harness_specs() -> Dict[str, PipelineConfig]:
    registry: Dict[str, PipelineConfig] = {}
    for spec_path in _iter_harness_specs(_repo_root()):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        runtime = dict(spec.get("runtime") or {})
        trigger = dict(spec.get("trigger_policy") or {})
        state_encoder = dict(spec.get("state_encoder") or {})
        pipeline = PipelineConfig(
            id=str(spec["id"]),
            name=str(spec.get("name") or spec["id"]),
            description=str(spec.get("description") or ""),
            base_mode=str(spec.get("base_mode") or "agent-heartbeat-soft"),
            trigger_type=str(trigger.get("type") or "periodic"),
            trigger_params={k: v for k, v in trigger.items() if isinstance(v, (int, float, bool))},
            prompt_variant=str(runtime.get("prompt_variant") or "default"),
            example_variant=str(runtime.get("example_variant") or "default"),
            state_fields=list(state_encoder.get("include_fields") or []),
            use_output_example=bool(runtime.get("use_output_example", True)),
            replan_cap=int(runtime.get("replan_cap", 8)),
            archive_enabled_default=True,
            harness_spec_path=str(spec_path),
            kind=str(spec.get("kind") or ("candidate" if "candidate_" in str(spec.get("id")) else "baseline")),
            parent_id=spec.get("parent"),
        )
        registry[pipeline.id] = pipeline
    return registry


PIPELINE_REGISTRY: Dict[str, PipelineConfig] = _load_pipeline_registry_from_harness_specs()
if not PIPELINE_REGISTRY:
    PIPELINE_REGISTRY = {
        "baseline1": PipelineConfig(
            id="baseline1",
            name="Periodic-Minimal",
            description="Heartbeat periodic trigger with minimal state payload.",
            base_mode="agent-heartbeat-soft",
            trigger_type="periodic",
            trigger_params={"heartbeat_seconds": 5.0},
            prompt_variant="baseline1_prompt",
            example_variant="baseline1_example",
            state_fields=["uav_pose_heading", "worker_positions", "remaining_checkpoints", "task_progress"],
            use_output_example=True,
            replan_cap=8,
            archive_enabled_default=True,
        ),
    }


def normalize_pipeline_id(pipeline_id: Optional[str]) -> str:
    candidate = str(pipeline_id or "baseline1").strip().lower()
    if candidate not in PIPELINE_REGISTRY:
        return "baseline1"
    return candidate


def get_pipeline_config(pipeline_id: Optional[str]) -> PipelineConfig:
    return PIPELINE_REGISTRY[normalize_pipeline_id(pipeline_id)]


def list_pipeline_options() -> List[str]:
    return list(PIPELINE_REGISTRY.keys())
