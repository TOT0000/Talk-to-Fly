from __future__ import annotations

# IMPORTANT:
# Keep these model IDs aligned with LM Studio /v1/models ids.
# The runner performs a visibility check and can resolve simple
# case-insensitive ID differences against currently visible models.
DEFAULT_PLANNER_MODEL_IDS: list[str] = [
    "meta-llama-3.1-8b-instruct",
    "google/gemma-2-9b",
]

DEFAULT_EVALUATOR_MODEL_IDS: list[str] = [
    "meta-llama-3.1-8b-instruct",
    "google/gemma-2-9b",
    "deepseek/deepseek-r1-0528-qwen3-8b",
]

DEFAULT_FIXED_PLANNER_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_FIXED_EVALUATOR_MODEL = "meta-llama-3.1-8b-instruct"

DEFAULT_PIPELINE_ID = "agent"
DEFAULT_SCENE_ID = "SCENE3"
DEFAULT_ZONE_ID = "zone_C"
DEFAULT_REPEAT_COUNT = 10
DEFAULT_EXPERIMENT_TAG = "model_pair_agent_scene3_zonec"

ZONE_TO_CHECKPOINTS: dict[str, list[str]] = {
    "zone_A": ["A1", "A2", "A3", "A4"],
    "zone_B": ["B1", "B2", "B3", "B4"],
    "zone_C": ["C1", "C2", "C3", "C4", "C5", "C6"],
}
