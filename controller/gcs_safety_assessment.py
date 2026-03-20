from __future__ import annotations

import math
import time
from typing import Optional

from .fuzzy_safety_assessor import FuzzySafetyAssessor
from .gcs_safety_state import GcsSafetyStateService
from .safety_context import SafetyContext


class GcsSafetyAssessmentService:

    @staticmethod
    def _observed_xy_localization_error_m(safety_state) -> float:
        drone_xy_error = math.hypot(
            float(safety_state.drone_packet.localization_error_vector_3d[0]),
            float(safety_state.drone_packet.localization_error_vector_3d[1]),
        )
        user_xy_error = math.hypot(
            float(safety_state.user_packet.localization_error_vector_3d[0]),
            float(safety_state.user_packet.localization_error_vector_3d[1]),
        )
        return drone_xy_error + user_xy_error

    def __init__(self):
        self.assessor = FuzzySafetyAssessor()

    def build_from_provider(self, state_provider, now: Optional[float] = None) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        safety_state = GcsSafetyStateService.build_from_provider(state_provider, now=now)
        return self.build_from_safety_state(safety_state, now=now)

    def build_from_safety_state(self, safety_state, now: Optional[float] = None) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        if safety_state is None:
            return SafetyContext(
                safety_score=0.50,
                safety_level="CAUTION",
                planning_bias="balanced",
                preferred_standoff_m=1.5,
                reason_tags=["safety_state_unavailable"],
                envelope_gap_m=0.0,
                uncertainty_scale_m=1.0,
                drone_to_user_distance_xy=0.0,
                envelopes_overlap=False,
                latest_generation_timestamp=None,
                latest_receive_timestamp=None,
                timing_freshness_s=None,
                max_aoi_s=None,
            )

        max_aoi_s = max(
            now - float(safety_state.drone_packet.state_generation_timestamp),
            now - float(safety_state.user_packet.state_generation_timestamp),
        )
        geometric_uncertainty_m = (
            safety_state.drone_radius_along_user_direction
            + safety_state.user_radius_along_drone_direction
        )
        observed_error_margin_m = self._observed_xy_localization_error_m(safety_state)
        uncertainty_scale_m = geometric_uncertainty_m + (0.75 * observed_error_margin_m)
        result = self.assessor.assess(
            envelope_gap_m=safety_state.envelope_gap_m,
            uncertainty_scale_m=uncertainty_scale_m,
            envelopes_overlap=safety_state.envelopes_overlap,
            freshness_aoi_s=max_aoi_s,
        )

        freshness = None
        if safety_state.latest_receive_timestamp is not None:
            freshness = now - float(safety_state.latest_receive_timestamp)

        return SafetyContext(
            safety_score=result.safety_score,
            safety_level=result.safety_level,
            planning_bias=result.planning_bias,
            preferred_standoff_m=result.preferred_standoff_m,
            reason_tags=result.reason_tags,
            envelope_gap_m=safety_state.envelope_gap_m,
            uncertainty_scale_m=uncertainty_scale_m,
            drone_to_user_distance_xy=safety_state.drone_to_user_distance_xy,
            envelopes_overlap=safety_state.envelopes_overlap,
            latest_generation_timestamp=safety_state.latest_generation_timestamp,
            latest_receive_timestamp=safety_state.latest_receive_timestamp,
            timing_freshness_s=freshness,
            max_aoi_s=max_aoi_s,
        )
