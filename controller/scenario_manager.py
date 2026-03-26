from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Optional
import math

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

        repositioned = False
        if drone is not None and hasattr(drone, "reposition_for_scenario"):
            repositioned = bool(drone.reposition_for_scenario(scenario))

        # Calibrate user placement from the *actual* runtime drone position so scenario
        # level and measured safety context stay aligned even if PX4 reposition is imperfect.
        drone_pos = None
        if drone is not None and hasattr(drone, "get_ground_truth_drone_position"):
            drone_pos = drone.get_ground_truth_drone_position()
        if drone_pos is None and provider is not None and hasattr(provider, "get_drone_position"):
            drone_pos = provider.get_drone_position()

        applied_user = scenario.user_position_3d
        if provider is not None and hasattr(provider, "set_user_position"):
            uncertainty = 0.85
            try:
                snapshot = controller.get_live_ui_snapshot()
                safety_context = snapshot.get("safety_context") if snapshot else None
                if safety_context is not None:
                    uncertainty = float(safety_context.uncertainty_scale_m)
            except Exception:
                pass
            desired_distance = self._distance_for_level(scenario.name, uncertainty)
            yaw = float(getattr(scenario, "drone_yaw_rad", 0.0))
            anchor_x, anchor_y, _ = scenario.drone_position_3d
            if drone_pos is not None:
                anchor_x, anchor_y = float(drone_pos[0]), float(drone_pos[1])
            ux = float(anchor_x + desired_distance * math.cos(yaw))
            uy = float(anchor_y + desired_distance * math.sin(yaw))
            uz = float(scenario.user_position_3d[2])
            provider.set_user_position(ux, uy, uz, source=f"scenario:{scenario.name}:calibrated")
            applied_user = (ux, uy, uz)

        print_t(
            f"[SCENARIO] active={scenario.name} user={applied_user} "
            f"drone_target={scenario.drone_position_3d} repositioned={repositioned}"
        )
        return repositioned

    @staticmethod
    def _distance_for_level(level: str, uncertainty_scale_m: float) -> float:
        # Target envelope gaps tuned to FIS boundaries:
        # SAFE > CAUTION > WARNING > DANGER
        target_gap_by_level = {
            "SAFE": 3.2,
            "CAUTION": 1.3,
            "WARNING": 0.45,
            "DANGER": -0.20,
        }
        target_gap = target_gap_by_level.get(level, 1.3)
        return max(0.15, float(uncertainty_scale_m) + float(target_gap))
