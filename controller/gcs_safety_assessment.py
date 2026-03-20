from __future__ import annotations

import time
from typing import Optional

from .fuzzy_safety_assessor import FuzzySafetyAssessor
from .gcs_safety_state import GcsSafetyStateService
from .safety_context import SafetyContext


class GcsSafetyAssessmentService:
    def __init__(self):
        self.assessor = FuzzySafetyAssessor()

    def build_from_provider(self, state_provider, now: Optional[float] = None) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        safety_state = GcsSafetyStateService.build_from_provider(state_provider, now=now)
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
            )

        result = self.assessor.assess(
            envelope_gap_m=safety_state.envelope_gap_m,
            uncertainty_scale_m=(
                safety_state.drone_radius_along_user_direction
                + safety_state.user_radius_along_drone_direction
            ),
            envelopes_overlap=safety_state.envelopes_overlap,
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
            uncertainty_scale_m=(
                safety_state.drone_radius_along_user_direction
                + safety_state.user_radius_along_drone_direction
            ),
            drone_to_user_distance_xy=safety_state.drone_to_user_distance_xy,
            envelopes_overlap=safety_state.envelopes_overlap,
            latest_generation_timestamp=safety_state.latest_generation_timestamp,
            latest_receive_timestamp=safety_state.latest_receive_timestamp,
            timing_freshness_s=freshness,
        )
