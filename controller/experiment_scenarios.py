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
        user_position_3d=(8.5, 8.2, 0.0),
        drone_yaw_rad=0.0,
        notes="Large standoff, strong geometric separation.",
    ),
    "CAUTION": ExperimentScenario(
        name="CAUTION",
        drone_position_3d=(5.6, 5.1, -1.3),
        user_position_3d=(7.6, 7.0, 0.0),
        drone_yaw_rad=0.7,
        notes="Mid-range separation near transition boundary.",
    ),
    "WARNING": ExperimentScenario(
        name="WARNING",
        drone_position_3d=(6.8, 6.0, -1.1),
        user_position_3d=(7.4, 6.9, 0.0),
        drone_yaw_rad=1.2,
        notes="Tight separation tuned to produce warning-level planning bias.",
    ),
    "DANGER": ExperimentScenario(
        name="DANGER",
        drone_position_3d=(7.2, 6.6, -1.0),
        user_position_3d=(7.4, 6.8, 0.0),
        drone_yaw_rad=1.57,
        notes="Very close placement intended to trigger overlap/negative gap risk.",
    ),
}


def normalize_scenario_name(name: str | None) -> str:
    candidate = (name or "SAFE").strip().upper()
    if candidate not in SCENARIOS:
        return "SAFE"
    return candidate
