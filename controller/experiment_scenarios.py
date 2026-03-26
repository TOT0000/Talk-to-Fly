from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ExperimentScenario:
    name: str
    drone_position_3d: Tuple[float, float, float]
    user_position_3d: Tuple[float, float, float]
    drone_yaw_rad: float = 0.0
    notes: str = ""


# PX4 local frame assumes NED (+Z is down). Negative z means higher altitude.
# The positions are fixed and deterministic for repeatable tests.
SCENARIOS: Dict[str, ExperimentScenario] = {
    "SAFE": ExperimentScenario(
        name="SAFE",
        drone_position_3d=(1.5, -1.5, -1.6),
        user_position_3d=(12.0, 7.8, 0.0),
        drone_yaw_rad=0.0,
        notes="Large standoff, strong geometric separation.",
    ),
    "CAUTION": ExperimentScenario(
        name="CAUTION",
        drone_position_3d=(3.8, -1.2, -1.5),
        user_position_3d=(9.2, 2.6, 0.0),
        drone_yaw_rad=0.7,
        notes="Mid-range separation near transition boundary.",
    ),
    "WARNING": ExperimentScenario(
        name="WARNING",
        drone_position_3d=(5.2, -1.1, -1.4),
        user_position_3d=(7.4, 0.2, 0.0),
        drone_yaw_rad=1.1,
        notes="Tight separation tuned to produce warning-level planning bias.",
    ),
    "DANGER": ExperimentScenario(
        name="DANGER",
        drone_position_3d=(6.4, -1.0, -1.3),
        user_position_3d=(7.3, -0.4, 0.0),
        drone_yaw_rad=0.9,
        notes="Close but non-overlapping initial geometry with high risk characteristics.",
    ),
}


def normalize_scenario_name(name: str | None) -> str:
    candidate = (name or "SAFE").strip().upper()
    if candidate not in SCENARIOS:
        return "SAFE"
    return candidate
