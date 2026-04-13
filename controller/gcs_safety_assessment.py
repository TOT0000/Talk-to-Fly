from __future__ import annotations

import time
from typing import Optional
import os
import numpy as np

from .collision_probability_core import CollisionEntity2D, CollisionProbabilityCore, hard_collision_probability_gauss_hermite
from .gcs_safety_state import GcsSafetyStateService
from .safety_context import SafetyContext
from .benchmark_layout import PREDICTION_HORIZON_SECONDS, PREDICTION_DT_SECONDS


class GcsSafetyAssessmentService:

    def __init__(self):
        # Collision-probability is the only primary risk core.
        self._core = CollisionProbabilityCore()
        self._uav_radius_m = float(os.getenv("TYPEFLY_UAV_RADIUS_M", "0.22"))
        self._worker_radius_m = float(os.getenv("TYPEFLY_WORKER_RADIUS_M", "0.30"))
        self._risk_worker_ids = ("worker_1", "worker_2", "worker_3")
        self._last_uav_samples: list[tuple[float, np.ndarray]] = []
        self._last_worker_samples: dict[str, list[tuple[float, np.ndarray]]] = {}

    def _build_context_from_scene_summary(
        self,
        *,
        current_probability: float,
        predicted_probability: float,
        per_worker_probs: list[dict],
        collision_debug_info: Optional[dict],
        dominant_worker_id: str,
        safety_state,
        now: float,
    ) -> SafetyContext:
        overlap_flag = bool(predicted_probability >= 0.3)
        if safety_state is not None:
            envelope_gap_m = float(safety_state.envelope_gap_m)
            geometric_uncertainty_m = float(
                safety_state.drone_radius_along_user_direction
                + safety_state.user_radius_along_drone_direction
            )
            uncertainty_scale_m = geometric_uncertainty_m
            distance_xy = float(safety_state.drone_to_user_distance_xy)
        else:
            envelope_gap_m = 0.0
            uncertainty_scale_m = 0.0
            distance_xy = 0.0

        return SafetyContext(
            # Keep compatibility fields while changing semantics:
            # safety_score now represents current scene collision probability.
            safety_score=float(current_probability),
            preferred_standoff_m=float(self._uav_radius_m + self._worker_radius_m),
            reason_tags=[
                "collision_probability_core",
                f"dominant_worker_{dominant_worker_id}",
            ],
            envelope_gap_m=float(envelope_gap_m),
            uncertainty_scale_m=float(uncertainty_scale_m),
            drone_to_user_distance_xy=float(distance_xy),
            envelopes_overlap=bool(overlap_flag),
            dominant_threat_type="worker",
            dominant_threat_id=str(dominant_worker_id),
            dominant_gap_m=float(envelope_gap_m),
            dominant_uncertainty_scale_m=float(uncertainty_scale_m),
            current_collision_probability=float(current_probability),
            predicted_collision_probability=float(predicted_probability),
            per_worker_collision_probabilities=per_worker_probs,
            collision_debug_info=collision_debug_info,
        )

    @staticmethod
    def _estimate_velocity(samples: list[tuple[float, np.ndarray]]) -> np.ndarray:
        if len(samples) < 2:
            return np.zeros(2, dtype=float)
        t0, p0 = samples[-2]
        t1, p1 = samples[-1]
        dt = max(1e-3, float(t1 - t0))
        return (np.asarray(p1, dtype=float).reshape(2) - np.asarray(p0, dtype=float).reshape(2)) / dt

    def _push_samples(
        self,
        *,
        now: float,
        uav_xy: np.ndarray,
        worker_packet_map: dict[str, object],
    ):
        self._last_uav_samples.append((float(now), np.asarray(uav_xy, dtype=float).reshape(2)))
        self._last_uav_samples = self._last_uav_samples[-3:]
        for worker_id, packet in worker_packet_map.items():
            samples = self._last_worker_samples.setdefault(str(worker_id), [])
            samples.append((float(now), np.asarray(packet.estimated_position_3d[:2], dtype=float).reshape(2)))
            self._last_worker_samples[str(worker_id)] = samples[-3:]

    def _compute_predicted_collision_probability(
        self,
        *,
        now: float,
        uav_entity: CollisionEntity2D,
        worker_entities: list[CollisionEntity2D],
        uav_velocity_hint_xy: Optional[np.ndarray] = None,
    ) -> tuple[float, float, str, dict[str, float]]:
        if not worker_entities:
            return 0.0, 0.0, "none", {}
        uav_vel = (
            np.asarray(uav_velocity_hint_xy, dtype=float).reshape(2)
            if uav_velocity_hint_xy is not None
            else self._estimate_velocity(self._last_uav_samples)
        )
        max_prob = 0.0
        max_tau = 0.0
        dominant_worker = "none"
        per_worker_max: dict[str, float] = {str(w.entity_id): 0.0 for w in worker_entities}
        tau = float(PREDICTION_DT_SECONDS)
        while tau <= float(PREDICTION_HORIZON_SECONDS) + 1e-9:
            uav_xy = np.asarray(uav_entity.mean_xy, dtype=float).reshape(2) + (uav_vel * tau)
            for worker in worker_entities:
                worker_samples = self._last_worker_samples.get(str(worker.entity_id), [])
                worker_vel = self._estimate_velocity(worker_samples)
                worker_xy = np.asarray(worker.mean_xy, dtype=float).reshape(2) + (worker_vel * tau)
                mu_k = worker_xy - uav_xy
                sigma_rel = np.asarray(worker.cov_xy, dtype=float).reshape(2, 2) + np.asarray(uav_entity.cov_xy, dtype=float).reshape(2, 2)
                p = hard_collision_probability_gauss_hermite(
                    mu_xy=mu_k,
                    sigma_xy=sigma_rel,
                    r_c=float(self._uav_radius_m + self._worker_radius_m),
                )
                if p > max_prob:
                    max_prob = float(p)
                    max_tau = float(tau)
                    dominant_worker = str(worker.entity_id)
                per_worker_max[str(worker.entity_id)] = max(per_worker_max.get(str(worker.entity_id), 0.0), float(p))
            tau += float(PREDICTION_DT_SECONDS)
        return float(max_prob), float(max_tau), str(dominant_worker), per_worker_max

    def build_from_packets(
        self,
        *,
        drone_packet,
        worker_packets: list[tuple[str, object]],
        now: Optional[float] = None,
        safety_state=None,
        uav_velocity_hint_xy: Optional[np.ndarray] = None,
    ) -> SafetyContext:
        now = time.time() if now is None else float(now)
        uav_entity = CollisionEntity2D(
            entity_id="uav",
            mean_xy=np.asarray(drone_packet.estimated_position_3d[:2], dtype=float),
            cov_xy=np.asarray(drone_packet.P_xy, dtype=float),
            bias_xy=np.asarray(drone_packet.b_xy, dtype=float),
            radius_m=float(self._uav_radius_m),
        )
        worker_packet_map: dict[str, object] = {}
        for worker_id, packet in worker_packets:
            worker_key = str(worker_id)
            if worker_key not in self._risk_worker_ids:
                continue
            worker_packet_map[worker_key] = packet
        worker_entities = []
        for worker_key in self._risk_worker_ids:
            packet = worker_packet_map.get(worker_key)
            if packet is None:
                continue
            worker_entities.append(
                CollisionEntity2D(
                    entity_id=str(worker_key),
                    mean_xy=np.asarray(packet.estimated_position_3d[:2], dtype=float),
                    cov_xy=np.asarray(packet.P_xy, dtype=float),
                    bias_xy=np.asarray(packet.b_xy, dtype=float),
                    radius_m=float(self._worker_radius_m),
                )
            )
        risk_entity_ids = [str(entity.entity_id) for entity in worker_entities]
        self._push_samples(now=now, uav_xy=np.asarray(drone_packet.estimated_position_3d[:2], dtype=float), worker_packet_map=worker_packet_map)

        summary = self._core.evaluate_scene(
            uav=uav_entity,
            workers=worker_entities,
        )
        predicted_probability, max_tau, dominant_predicted_worker, per_worker_predicted_max = self._compute_predicted_collision_probability(
            now=now,
            uav_entity=uav_entity,
            worker_entities=worker_entities,
            uav_velocity_hint_xy=uav_velocity_hint_xy,
        )
        per_worker_probs = [
            {
                "id": item.entity_id,
                "collision_probability": float(item.probability),
                "soft_probability": float(item.soft_probability),
                "approximate_probability": float(item.approximate_probability),
                "hard_approx_probability": float(item.hard_approx_probability),
                "exact_series_probability": float(item.exact_series_probability),
                "monte_carlo_probability": (None if item.monte_carlo_probability is None else float(item.monte_carlo_probability)),
                "mu_xy": [float(item.mu_xy[0]), float(item.mu_xy[1])],
                "sigma_rel": [[float(item.sigma_rel[0][0]), float(item.sigma_rel[0][1])], [float(item.sigma_rel[1][0]), float(item.sigma_rel[1][1])]],
                "r_u": float(self._uav_radius_m),
                "r_h": float(self._worker_radius_m),
                "r_c": float(self._uav_radius_m + self._worker_radius_m),
                "predicted_collision_probability": float(per_worker_predicted_max.get(str(item.entity_id), 0.0)),
            }
            for item in summary.per_entity
        ]
        collision_debug_info = {
            "sanity_case_probabilities": dict(summary.sanity_case_probabilities or {}),
            "uav_radius_m": float(self._uav_radius_m),
            "worker_radius_m": float(self._worker_radius_m),
            "collision_radius_m": float(self._uav_radius_m + self._worker_radius_m),
            "risk_entities": list(risk_entity_ids),
            "risk_entities_expected": list(self._risk_worker_ids),
            "predicted_collision_probability": float(predicted_probability),
            "prediction_horizon_seconds": float(PREDICTION_HORIZON_SECONDS),
            "prediction_dt_seconds": float(PREDICTION_DT_SECONDS),
            "max_risk_tau_seconds": float(max_tau),
            "dominant_predicted_worker": str(dominant_predicted_worker),
        }
        return self._build_context_from_scene_summary(
            current_probability=float(summary.current_probability),
            predicted_probability=float(predicted_probability),
            per_worker_probs=per_worker_probs,
            collision_debug_info=collision_debug_info,
            dominant_worker_id=str(dominant_predicted_worker or summary.dominant_entity_id),
            safety_state=safety_state,
            now=now,
        )

    def build_from_provider(self, state_provider, now: Optional[float] = None) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        safety_state = GcsSafetyStateService.build_from_provider(state_provider, now=now)
        return self.build_from_safety_state(safety_state, now=now, worker_packets=None)

    def build_from_safety_state(
        self,
        safety_state,
        now: Optional[float] = None,
        worker_packets: Optional[list[tuple[str, object]]] = None,
    ) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        if safety_state is None:
            return SafetyContext(
                safety_score=0.0,
                preferred_standoff_m=float(self._uav_radius_m + self._worker_radius_m),
                reason_tags=["collision_probability_core", "safety_state_unavailable"],
                envelope_gap_m=0.0,
                uncertainty_scale_m=0.0,
                drone_to_user_distance_xy=0.0,
                envelopes_overlap=False,
                dominant_threat_type="worker",
                dominant_threat_id="none",
                dominant_gap_m=0.0,
                dominant_uncertainty_scale_m=0.0,
                current_collision_probability=0.0,
                predicted_collision_probability=0.0,
                per_worker_collision_probabilities=[],
                collision_debug_info=None,
            )
        if worker_packets is None:
            return SafetyContext(
                safety_score=0.0,
                preferred_standoff_m=float(self._uav_radius_m + self._worker_radius_m),
                reason_tags=["collision_probability_core", "risk_workers_unavailable"],
                envelope_gap_m=0.0,
                uncertainty_scale_m=0.0,
                drone_to_user_distance_xy=0.0,
                envelopes_overlap=False,
                dominant_threat_type="worker",
                dominant_threat_id="none",
                dominant_gap_m=0.0,
                dominant_uncertainty_scale_m=0.0,
                current_collision_probability=0.0,
                predicted_collision_probability=0.0,
                per_worker_collision_probabilities=[],
                collision_debug_info={
                    "risk_entities": [],
                    "risk_entities_expected": list(self._risk_worker_ids),
                },
            )
        return self.build_from_packets(
            drone_packet=safety_state.drone_packet,
            worker_packets=worker_packets,
            now=now,
            safety_state=safety_state,
        )
