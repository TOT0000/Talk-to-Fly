from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from .experiment_scenarios import SCENARIOS, ExperimentScenario, normalize_scenario_name
from .fuzzy_safety_assessor import FuzzySafetyAssessor
from .utils import print_t


LEVEL_RANK = {"SAFE": 3, "CAUTION": 2, "WARNING": 1, "DANGER": 0}


@dataclass
class ScenarioApplyReport:
    selected_mode: str
    target_drone_position_3d: tuple[float, float, float]
    target_user_position_3d: tuple[float, float, float]
    actual_drone_gt_position_3d: tuple[float, float, float]
    actual_user_gt_position_3d: tuple[float, float, float]
    measured_initial_safety_score: Optional[float]
    measured_initial_safety_level: Optional[str]
    measured_initial_envelope_gap_m: Optional[float]
    measured_initial_uncertainty_scale_m: Optional[float]
    measured_initial_max_aoi_s: Optional[float]
    repositioned: bool
    calibration_iterations: int


class ScenarioManager:
    """Deterministic scenario selection + runtime-validated scenario application."""

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

    def apply_to_runtime(self, controller) -> ScenarioApplyReport:
        scenario = self.current()
        provider = getattr(controller, "state_provider", None)
        drone = getattr(controller, "drone", None)

        if provider is not None and hasattr(provider, "lock_user_position"):
            provider.lock_user_position(True, reason=f"scenario:{scenario.name}")

        repositioned = False
        if drone is not None and hasattr(drone, "reposition_for_scenario"):
            repositioned = bool(drone.reposition_for_scenario(scenario))

        # Runtime calibration: adjust user offset using measured uncertainty/gap until
        # measured level is close to selected mode.
        calibration_iterations = 0
        if provider is not None and hasattr(provider, "set_user_position"):
            calibration_iterations = self._calibrate_user_offset(controller, scenario)

        snapshot = controller.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if snapshot else None
        actual_drone = tuple(float(v) for v in (snapshot.get("drone_gt") or scenario.drone_position_3d))
        actual_user = tuple(float(v) for v in (snapshot.get("user_gt") or scenario.user_position_3d))

        report = ScenarioApplyReport(
            selected_mode=scenario.name,
            target_drone_position_3d=tuple(float(v) for v in scenario.drone_position_3d),
            target_user_position_3d=tuple(float(v) for v in scenario.user_position_3d),
            actual_drone_gt_position_3d=actual_drone,
            actual_user_gt_position_3d=actual_user,
            measured_initial_safety_score=None if safety_context is None else float(safety_context.safety_score),
            measured_initial_safety_level=None if safety_context is None else str(safety_context.safety_level),
            measured_initial_envelope_gap_m=None if safety_context is None else float(safety_context.envelope_gap_m),
            measured_initial_uncertainty_scale_m=None if safety_context is None else float(safety_context.uncertainty_scale_m),
            measured_initial_max_aoi_s=None if safety_context is None else float(safety_context.max_aoi_s or 0.0),
            repositioned=repositioned,
            calibration_iterations=calibration_iterations,
        )

        print_t(
            "[SCENARIO-VALIDATION] "
            f"selected={report.selected_mode} "
            f"target_drone={report.target_drone_position_3d} target_user={report.target_user_position_3d} "
            f"actual_drone={report.actual_drone_gt_position_3d} actual_user={report.actual_user_gt_position_3d} "
            f"score={report.measured_initial_safety_score} level={report.measured_initial_safety_level} "
            f"gap={report.measured_initial_envelope_gap_m} uncertainty={report.measured_initial_uncertainty_scale_m} "
            f"aoi={report.measured_initial_max_aoi_s} repositioned={report.repositioned} iterations={report.calibration_iterations}"
        )
        return report

    def _calibrate_user_offset(self, controller, scenario: ExperimentScenario) -> int:
        provider = controller.state_provider
        yaw = float(getattr(scenario, "drone_yaw_rad", 0.0))
        profile = {
            # (target_gap_m, freshness_wait_s)
            "SAFE": (5.2, 0.03),
            "CAUTION": (2.9, 0.10),
            "WARNING": (0.60, 0.34),
            "DANGER": (-0.10, 0.72),
        }.get(scenario.name, (2.9, 0.10))

        snapshot = controller.get_live_ui_snapshot()
        drone_gt = snapshot.get("drone_gt") if snapshot else None
        safety_context = snapshot.get("safety_context") if snapshot else None
        uncertainty = 0.85 if safety_context is None else float(safety_context.uncertainty_scale_m)
        if drone_gt is None:
            drone_gt = scenario.drone_position_3d

        target_gap_m, freshness_wait_s = profile
        desired_distance = max(0.15, uncertainty + float(target_gap_m))
        ux = float(drone_gt[0] + desired_distance * math.cos(yaw))
        uy = float(drone_gt[1] + desired_distance * math.sin(yaw))
        uz = float(scenario.user_position_3d[2])
        provider.set_user_position(ux, uy, uz, source=f"scenario:{scenario.name}:profile")
        self._wait_stable_cycles(controller, cycles=3)
        if freshness_wait_s > 0:
            time.sleep(float(freshness_wait_s))
        # Runtime calibration for riskier modes: if measured level is too safe,
        # iteratively reduce target gap but avoid immediate envelope-overlap starts.
        if scenario.name in {"WARNING", "DANGER"}:
            target_rank = LEVEL_RANK[scenario.name]
            current_gap = float(target_gap_m)
            for _ in range(3):
                measured = controller.get_live_ui_snapshot()
                measured_ctx = measured.get("safety_context") if measured else None
                measured_rank = LEVEL_RANK.get(str(measured_ctx.safety_level), 3) if measured_ctx else 3
                if measured_rank <= target_rank:
                    break
                current_gap = current_gap - (0.20 if scenario.name == "WARNING" else 0.15)
                current_gap = max(current_gap, -0.20)
                desired_distance = max(0.50, uncertainty + current_gap)
                ux = float(drone_gt[0] + desired_distance * math.cos(yaw))
                uy = float(drone_gt[1] + desired_distance * math.sin(yaw))
                provider.set_user_position(ux, uy, uz, source=f"scenario:{scenario.name}:calib")
                self._wait_stable_cycles(controller, cycles=2, sleep_s=0.08)
        # Refresh UI/provider cache after intentional AoI shaping.
        self._wait_stable_cycles(controller, cycles=1, sleep_s=0.05)
        return 1

    @staticmethod
    def _wait_stable_cycles(controller, cycles: int = 3, sleep_s: float = 0.12):
        for _ in range(max(1, cycles)):
            try:
                snapshot = controller.get_live_ui_snapshot()
                safety_context = snapshot.get("safety_context") if snapshot else None
                if safety_context is not None and safety_context.max_aoi_s is not None and float(safety_context.max_aoi_s) <= 0.35:
                    pass
            except Exception:
                pass
            time.sleep(sleep_s)
