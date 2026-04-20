from __future__ import annotations

from typing import Dict, List

FORMAL_EVALUATION_PROTOCOL_VERSION = "uav_search_v2_live24_formal"
SCREENING_EVALUATION_PROTOCOL_VERSION = "uav_search_v2_live6_screening"
RUNS_PER_SCENE_TASK_FORMAL = 8
RUNS_PER_SCENE_TASK_SCREENING = 2

# Fixed benchmark protocol requested by project owner.
# Keep this mapping stable for both baselines and proposer-generated candidates.
EVALUATION_SCENE_TASK_MAPPING = {
    "SCENE1": "zoneA",
    "SCENE2": "zoneB",
    "SCENE3": "zoneC",
}

def _build_sequence(runs_per_scene: int) -> List[Dict]:
    return [
        {
            "scene_id": scene_id,
            "task_zone": task_zone,
            "runs": int(runs_per_scene),
        }
        for scene_id, task_zone in EVALUATION_SCENE_TASK_MAPPING.items()
    ]


EVALUATION_PROTOCOLS: Dict[str, Dict] = {
    "screening": {
        "mode": "screening",
        "name": "candidate_screening_v1",
        "version": SCREENING_EVALUATION_PROTOCOL_VERSION,
        "runs_per_scene": RUNS_PER_SCENE_TASK_SCREENING,
        "pairs": _build_sequence(RUNS_PER_SCENE_TASK_SCREENING),
        "total_runs": RUNS_PER_SCENE_TASK_SCREENING * len(EVALUATION_SCENE_TASK_MAPPING),
    },
    "formal": {
        "mode": "formal",
        "name": "formal_v1",
        "version": FORMAL_EVALUATION_PROTOCOL_VERSION,
        "runs_per_scene": RUNS_PER_SCENE_TASK_FORMAL,
        "pairs": _build_sequence(RUNS_PER_SCENE_TASK_FORMAL),
        "total_runs": RUNS_PER_SCENE_TASK_FORMAL * len(EVALUATION_SCENE_TASK_MAPPING),
    },
}


def resolve_evaluation_mode(*, kind: str, requested_mode: str | None = None) -> str:
    normalized_kind = str(kind or "").strip().lower()
    normalized_mode = str(requested_mode or "").strip().lower() or None
    if normalized_mode and normalized_mode not in EVALUATION_PROTOCOLS:
        raise ValueError(f"Unsupported evaluation mode: {requested_mode}")
    if normalized_kind == "baseline":
        if normalized_mode in {None, "formal"}:
            return "formal"
        raise ValueError("Baselines only support formal evaluation mode.")
    # Candidates default to screening for low-cost triage.
    return normalized_mode or "screening"


def get_evaluation_protocol(*, kind: str, requested_mode: str | None = None) -> Dict:
    mode = resolve_evaluation_mode(kind=kind, requested_mode=requested_mode)
    return dict(EVALUATION_PROTOCOLS[mode])


# Backward-compatible aliases: formal protocol remains the fixed baseline standard.
EVALUATION_PROTOCOL_VERSION = FORMAL_EVALUATION_PROTOCOL_VERSION
RUNS_PER_SCENE_TASK = RUNS_PER_SCENE_TASK_FORMAL
EVALUATION_PROTOCOL_SEQUENCE = list(EVALUATION_PROTOCOLS["formal"]["pairs"])
TOTAL_EVAL_RUNS = int(EVALUATION_PROTOCOLS["formal"]["total_runs"])

# Lower-case aliases for tools/CLIs that normalize scene names.
EVALUATION_SCENE_TASK_MAPPING_LOWER = {
    k.lower(): v for k, v in EVALUATION_SCENE_TASK_MAPPING.items()
}
