from __future__ import annotations

# IMPORTANT:
# Keep these model IDs exactly aligned with the identifiers exposed by your
# local LM Studio OpenAI-compatible endpoint. If LM Studio shows different
# names in /v1/models, update this single list only.
DEFAULT_MODEL_GRID_IDS: list[str] = [
    "microsoft_phi-4-mini-instruct",
    "meta-llama-3.1-8b-instruct",
    "google/gemma-2-9b",
    "openai/gpt-oss-20b",
]

DEFAULT_PIPELINE_ID = "agent"
DEFAULT_SCENE_ID = "SCENE3"
DEFAULT_ZONE_ID = "zone_C"
DEFAULT_REPEAT_COUNT = 10
DEFAULT_EXPERIMENT_TAG = "model_grid_agent_scene3_zonec"

ZONE_TO_CHECKPOINTS: dict[str, list[str]] = {
    "zone_A": ["A1", "A2", "A3", "A4"],
    "zone_B": ["B1", "B2", "B3", "B4"],
    "zone_C": ["C1", "C2", "C3", "C4", "C5", "C6"],
}
