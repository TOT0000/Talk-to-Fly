from __future__ import annotations

EVALUATION_PROTOCOL_VERSION = "uav_search_v1"

# Fixed benchmark protocol requested by project owner.
# Keep this mapping stable for both baselines and proposer-generated candidates.
EVALUATION_SCENE_TASK_MAPPING = {
    "SCENE1": "zoneA",
    "SCENE2": "zoneB",
    "SCENE3": "zoneC",
}

# Lower-case aliases for tools/CLIs that normalize scene names.
EVALUATION_SCENE_TASK_MAPPING_LOWER = {
    k.lower(): v for k, v in EVALUATION_SCENE_TASK_MAPPING.items()
}
