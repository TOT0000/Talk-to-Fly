from __future__ import annotations

import time
from typing import Optional
import os
import numpy as np

from .collision_probability_core import CollisionEntity2D, CollisionProbabilityCore
from .gcs_safety_state import GcsSafetyStateService
from .safety_context import SafetyContext


class GcsSafetyAssessmentService:

    def __init__(self):
        # Collision-probability is the only primary risk core.
        self._core = CollisionProbabilityCore()
        self._uav_radius_m = float(os.getenv("TYPEFLY_UAV_RADIUS_M", "0.35"))
        self._worker_radius_m = float(os.getenv("TYPEFLY_WORKER_RADIUS_M", "0.35"))

    @staticmethod
    def _probability_to_level(probability: float) -> str:
        """Map collision probability to compatibility labels used by existing UI/loggers."""
        p = float(probability)
        if p < 0.05:
            return "SAFE"
        if p < 0.15:
            return "CAUTION"
        if p < 0.30:
            return "WARNING"
        return "DANGER"

    @staticmethod
    def _planning_bias(level: str) -> str:
        return {
            "SAFE": "efficiency",
            "CAUTION": "balanced",
            "WARNING": "safety",
            "DANGER": "safety",
        }.get(level, "balanced")

    def _build_context_from_scene_summary(
        self,
        *,
        current_probability: float,
        historical_max_probability: float,
        per_worker_probs: list[dict],
        dominant_worker_id: str,
        safety_state,
        now: float,
    ) -> SafetyContext:
        level = self._probability_to_level(current_probability)
        overlap_flag = bool(current_probability >= 0.5)
        if safety_state is not None:
            envelope_gap_m = float(safety_state.envelope_gap_m)
            uncertainty_scale_m = float(
                safety_state.drone_radius_along_user_direction
                + safety_state.user_radius_along_drone_direction
            )
            distance_xy = float(safety_state.drone_to_user_distance_xy)
            latest_gen = safety_state.latest_generation_timestamp
            latest_recv = safety_state.latest_receive_timestamp
            max_aoi_s = max(
                now - float(safety_state.drone_packet.state_generation_timestamp),
                now - float(safety_state.user_packet.state_generation_timestamp),
            )
        else:
            envelope_gap_m = 0.0
            uncertainty_scale_m = 0.0
            distance_xy = 0.0
            latest_gen = None
            latest_recv = None
            max_aoi_s = None
        freshness = None if latest_recv is None else now - float(latest_recv)

        return SafetyContext(
            # Keep compatibility fields while changing semantics:
            # safety_score now represents current scene collision probability.
            safety_score=float(current_probability),
            safety_level=str(level),
            planning_bias=str(self._planning_bias(level)),
            preferred_standoff_m=float(self._uav_radius_m + self._worker_radius_m),
            reason_tags=[
                "collision_probability_core",
                f"dominant_worker_{dominant_worker_id}",
            ],
            envelope_gap_m=float(envelope_gap_m),
            uncertainty_scale_m=float(uncertainty_scale_m),
            drone_to_user_distance_xy=float(distance_xy),
            envelopes_overlap=bool(overlap_flag),
            latest_generation_timestamp=latest_gen,
            latest_receive_timestamp=latest_recv,
            timing_freshness_s=freshness,
            max_aoi_s=max_aoi_s,
            dominant_threat_type="worker",
            dominant_threat_id=str(dominant_worker_id),
            dominant_gap_m=float(envelope_gap_m),
            dominant_uncertainty_scale_m=float(uncertainty_scale_m),
            dominant_freshness_s=max_aoi_s,
            current_collision_probability=float(current_probability),
            historical_max_collision_probability=float(historical_max_probability),
            per_worker_collision_probabilities=per_worker_probs,
        )

    def build_from_packets(
        self,
        *,
        drone_packet,
        worker_packets: list[tuple[str, object]],
        now: Optional[float] = None,
        safety_state=None,
    ) -> SafetyContext:
        now = time.time() if now is None else float(now)
        uav_entity = CollisionEntity2D(
            entity_id="uav",
            mean_xy=np.asarray(drone_packet.estimated_position_3d[:2], dtype=float),
            cov_xy=np.asarray(drone_packet.P_xy, dtype=float),
            bias_xy=np.asarray(drone_packet.b_xy, dtype=float),
            radius_m=float(self._uav_radius_m),
        )
        worker_entities = []
        for worker_id, packet in worker_packets:
            worker_entities.append(
                CollisionEntity2D(
                    entity_id=str(worker_id),
                    mean_xy=np.asarray(packet.estimated_position_3d[:2], dtype=float),
                    cov_xy=np.asarray(packet.P_xy, dtype=float),
                    bias_xy=np.asarray(packet.b_xy, dtype=float),
                    radius_m=float(self._worker_radius_m),
                )
            )

        summary = self._core.evaluate_scene(
            uav=uav_entity,
            workers=worker_entities,
        )
        per_worker_probs = [
            {"id": item.entity_id, "collision_probability": float(item.probability)}
            for item in summary.per_entity
        ]
        return self._build_context_from_scene_summary(
            current_probability=float(summary.current_probability),
            historical_max_probability=float(summary.historical_max_probability),
            per_worker_probs=per_worker_probs,
            dominant_worker_id=str(summary.dominant_entity_id),
            safety_state=safety_state,
            now=now,
        )

    def build_from_provider(self, state_provider, now: Optional[float] = None) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        safety_state = GcsSafetyStateService.build_from_provider(state_provider, now=now)
        return self.build_from_safety_state(safety_state, now=now)

    def build_from_safety_state(self, safety_state, now: Optional[float] = None) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        if safety_state is None:
            return SafetyContext(
                safety_score=0.0,
                safety_level="CAUTION",
                planning_bias="balanced",
                preferred_standoff_m=float(self._uav_radius_m + self._worker_radius_m),
                reason_tags=["collision_probability_core", "safety_state_unavailable"],
                envelope_gap_m=0.0,
                uncertainty_scale_m=0.0,
                drone_to_user_distance_xy=0.0,
                envelopes_overlap=False,
                latest_generation_timestamp=None,
                latest_receive_timestamp=None,
                timing_freshness_s=None,
                max_aoi_s=None,
                dominant_threat_type="worker",
                dominant_threat_id="none",
                dominant_gap_m=0.0,
                dominant_uncertainty_scale_m=0.0,
                dominant_freshness_s=None,
                current_collision_probability=0.0,
                historical_max_collision_probability=float(self._core.get_historical_max_probability()),
                per_worker_collision_probabilities=[],
            )
        return self.build_from_packets(
            drone_packet=safety_state.drone_packet,
            worker_packets=[("user", safety_state.user_packet)],
            now=now,
            safety_state=safety_state,
        )
