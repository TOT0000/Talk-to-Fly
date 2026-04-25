from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ManualModelPair:
    pair_id: str
    label: str
    planner_model_id: str
    evaluator_model_id: str


MANUAL_MODEL_PAIRS: Dict[str, ManualModelPair] = {
    "pair_gemma_gemma": ManualModelPair(
        pair_id="pair_gemma_gemma",
        label="gemma → gemma",
        planner_model_id="google/gemma-2-9b",
        evaluator_model_id="google/gemma-2-9b",
    ),
    "pair_gemma_deepseek": ManualModelPair(
        pair_id="pair_gemma_deepseek",
        label="gemma → deepseek",
        planner_model_id="google/gemma-2-9b",
        evaluator_model_id="deepseek/deepseek-r1-0528-qwen3-8b",
    ),
    "pair_deepseek_gemma": ManualModelPair(
        pair_id="pair_deepseek_gemma",
        label="deepseek → gemma",
        planner_model_id="deepseek/deepseek-r1-0528-qwen3-8b",
        evaluator_model_id="google/gemma-2-9b",
    ),
    "pair_deepseek_deepseek": ManualModelPair(
        pair_id="pair_deepseek_deepseek",
        label="deepseek → deepseek",
        planner_model_id="deepseek/deepseek-r1-0528-qwen3-8b",
        evaluator_model_id="deepseek/deepseek-r1-0528-qwen3-8b",
    ),
}

DEFAULT_MANUAL_MODEL_PAIR_ID = "pair_gemma_gemma"


def normalize_manual_model_pair_id(pair_id: str | None) -> str:
    candidate = str(pair_id or DEFAULT_MANUAL_MODEL_PAIR_ID).strip()
    if candidate not in MANUAL_MODEL_PAIRS:
        return DEFAULT_MANUAL_MODEL_PAIR_ID
    return candidate


def get_manual_model_pair(pair_id: str | None) -> ManualModelPair:
    return MANUAL_MODEL_PAIRS[normalize_manual_model_pair_id(pair_id)]


def list_manual_model_pair_options() -> List[Tuple[str, str]]:
    return [(pair.label, pair.pair_id) for pair in MANUAL_MODEL_PAIRS.values()]
