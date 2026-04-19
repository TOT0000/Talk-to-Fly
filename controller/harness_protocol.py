from __future__ import annotations

EVALUATION_PROTOCOL_VERSION = "uav_search_v2_live24"
RUNS_PER_SCENE_TASK = 8

# Fixed benchmark protocol requested by project owner.
# Keep this mapping stable for both baselines and proposer-generated candidates.
EVALUATION_SCENE_TASK_MAPPING = {
    "SCENE1": "zoneA",
    "SCENE2": "zoneB",
    "SCENE3": "zoneC",
}

EVALUATION_PROTOCOL_SEQUENCE = [
    {
        "scene_id": scene_id,
        "task_zone": task_zone,
        "runs": RUNS_PER_SCENE_TASK,
    }
    for scene_id, task_zone in EVALUATION_SCENE_TASK_MAPPING.items()
]

TOTAL_EVAL_RUNS = RUNS_PER_SCENE_TASK * len(EVALUATION_SCENE_TASK_MAPPING)

# Lower-case aliases for tools/CLIs that normalize scene names.
EVALUATION_SCENE_TASK_MAPPING_LOWER = {
    k.lower(): v for k, v in EVALUATION_SCENE_TASK_MAPPING.items()
}
