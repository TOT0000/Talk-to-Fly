from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Optional

from .experiment_scenarios import SCENARIOS, ExperimentScenario, normalize_scenario_name
from .fuzzy_safety_assessor import FuzzySafetyAssessor
from .utils import print_t


class ScenarioManager:
    """Deterministic scenario selection + projected safety-level preview."""

    def __init__(self, default_name: Optional[str] = None):
        env_name = os.getenv("TYPEFLY_SCENARIO")
        self._selected_name = normalize_scenario_name(default_name or env_name)
        self._assessor = FuzzySafetyAssessor()

    def selected_name(self) -> str:
        return self._selected_name

    def select(self, name: str) -> ExperimentScenario:
        self._selected_name = normalize_scenario_name(name)
        return self.current()

    def current(self) -> ExperimentScenario:
        return SCENARIOS[self._selected_name]

    def names(self):
        return list(SCENARIOS.keys())

    def as_dict(self) -> Dict:
        scenario = self.current()
        return asdict(scenario)

    def projected_assessment(self, baseline_uncertainty_scale_m: float = 0.85, baseline_aoi_s: float = 0.05) -> Dict:
        """Project expected FIS level using current scenario geometry + baseline uncertainty."""
        scenario = self.current()
        dx = float(scenario.drone_position_3d[0] - scenario.user_position_3d[0])
        dy = float(scenario.drone_position_3d[1] - scenario.user_position_3d[1])
        distance_xy = (dx * dx + dy * dy) ** 0.5
        envelope_gap_m = distance_xy - float(baseline_uncertainty_scale_m)
        result = self._assessor.assess(
            envelope_gap_m=float(envelope_gap_m),
            uncertainty_scale_m=float(baseline_uncertainty_scale_m),
            envelopes_overlap=bool(envelope_gap_m < 0.0),
            freshness_aoi_s=float(baseline_aoi_s),
        )
        return {
            "scenario_name": scenario.name,
            "distance_xy": distance_xy,
            "projected_envelope_gap_m": envelope_gap_m,
            "projected_level": result.safety_level,
            "projected_score": result.safety_score,
            "projected_reason_tags": list(result.reason_tags),
        }

    def apply_to_runtime(self, controller) -> bool:
        """Apply the selected scenario to sim/user providers with best-effort reposition flow."""
        scenario = self.current()
        provider = getattr(controller, "state_provider", None)
        drone = getattr(controller, "drone", None)

        if provider is not None and hasattr(provider, "lock_user_position"):
            provider.lock_user_position(True, reason=f"scenario:{scenario.name}")

        if provider is not None and hasattr(provider, "set_user_position"):
            provider.set_user_position(*scenario.user_position_3d, source=f"scenario:{scenario.name}")

        repositioned = False
        if drone is not None and hasattr(drone, "reposition_for_scenario"):
            repositioned = bool(drone.reposition_for_scenario(scenario))

        print_t(
            f"[SCENARIO] active={scenario.name} user={scenario.user_position_3d} "
            f"drone_target={scenario.drone_position_3d} repositioned={repositioned}"
        )
        return repositioned
