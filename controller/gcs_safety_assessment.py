from __future__ import annotations

import time
from typing import Optional
import os
import numpy as np

from .collision_probability_core import CollisionEntity2D, CollisionProbabilityCore, hard_collision_probability_gauss_hermite
from .gcs_safety_state import GcsSafetyStateService
from .safety_context import SafetyContext
from .benchmark_layout import PREDICTION_HORIZON_SECONDS, PREDICTION_DT_SECONDS
from .utils import print_debug


class GcsSafetyAssessmentService:

    def __init__(self):
        # Collision-probability is the only primary risk core.
        self._core = CollisionProbabilityCore()
        self._uav_radius_m = float(os.getenv("TYPEFLY_UAV_RADIUS_M", "0.22"))
        self._obstacle_radius_m = float(os.getenv("TYPEFLY_OBSTACLE_RADIUS_M", "0.30"))
        self._risk_obstacle_ids = ("obstacle_1", "obstacle_2", "obstacle_3")
        self._last_uav_samples: list[tuple[float, np.ndarray]] = []
        self._last_obstacle_samples: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._uav_process_noise_var_m2ps = float(os.getenv("TYPEFLY_UAV_PROCESS_NOISE_VAR_M2PS", "0.01"))
        self._obstacle_process_noise_var_m2ps = float(os.getenv("TYPEFLY_WORKER_PROCESS_NOISE_VAR_M2PS", "0.015"))

    def _build_context_from_scene_summary(
        self,
        *,
        current_probability: float,
        predicted_probability: float,
        per_obstacle_probs: list[dict],
        collision_debug_info: Optional[dict],
        dominant_obstacle_id: str,
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
            preferred_standoff_m=float(self._uav_radius_m + self._obstacle_radius_m),
            reason_tags=[
                "collision_probability_core",
                f"dominant_obstacle_{dominant_obstacle_id}",
            ],
            envelope_gap_m=float(envelope_gap_m),
            uncertainty_scale_m=float(uncertainty_scale_m),
            drone_to_user_distance_xy=float(distance_xy),
            envelopes_overlap=bool(overlap_flag),
            dominant_threat_type="obstacle",
            dominant_threat_id=str(dominant_obstacle_id),
            dominant_gap_m=float(envelope_gap_m),
            dominant_uncertainty_scale_m=float(uncertainty_scale_m),
            current_collision_probability=float(current_probability),
            predicted_collision_probability=float(predicted_probability),
            per_obstacle_collision_probabilities=per_obstacle_probs,
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

    @staticmethod
    def _bias_corrected_xy(mean_xy: np.ndarray, bias_xy: np.ndarray) -> np.ndarray:
        return np.asarray(mean_xy, dtype=float).reshape(2) - np.asarray(bias_xy, dtype=float).reshape(2)

    @staticmethod
    def _predict_uav_xy_at_tau(
        *,
        tau: float,
        uav_start_xy: np.ndarray,
        fallback_vel_xy: np.ndarray,
        uav_prediction_intent: Optional[dict],
    ) -> tuple[np.ndarray, str]:
        start = np.asarray(uav_start_xy, dtype=float).reshape(2)
        fallback = np.asarray(fallback_vel_xy, dtype=float).reshape(2)
        intent = dict(uav_prediction_intent or {})
        mode = str(intent.get("mode", "velocity_fallback"))
        if mode == "gc_target":
            target_xy = intent.get("target_xy")
            speed_mps = float(intent.get("speed_mps", 0.0))
            if target_xy is not None:
                target = np.asarray(target_xy, dtype=float).reshape(2)
                delta = target - start
                dist = float(np.linalg.norm(delta))
                if dist > 1e-6 and speed_mps > 0.0:
                    dir_xy = delta / dist
                    travel = min(dist, max(0.0, speed_mps * float(tau)))
                    return start + dir_xy * travel, "gc_target"
            return start + fallback * float(tau), "gc_target_fallback_velocity"
        if mode == "body_action":
            vel_xy = intent.get("body_velocity_xy")
            if vel_xy is not None:
                return start + np.asarray(vel_xy, dtype=float).reshape(2) * float(tau), "body_action"
            return start + fallback * float(tau), "body_action_fallback_velocity"
        return start + fallback * float(tau), "velocity_fallback"

    def _push_samples(
        self,
        *,
        now: float,
        uav_corrected_xy: np.ndarray,
        obstacle_packet_map: dict[str, object],
    ):
        self._last_uav_samples.append((float(now), np.asarray(uav_corrected_xy, dtype=float).reshape(2)))
        self._last_uav_samples = self._last_uav_samples[-3:]
        for obstacle_id, packet in obstacle_packet_map.items():
            samples = self._last_obstacle_samples.setdefault(str(obstacle_id), [])
            corrected_xy = self._bias_corrected_xy(
                np.asarray(packet.estimated_position_3d[:2], dtype=float),
                np.asarray(packet.b_xy[:2], dtype=float),
            )
            samples.append((float(now), corrected_xy))
            self._last_obstacle_samples[str(obstacle_id)] = samples[-3:]

    def _compute_predicted_collision_probability(
        self,
        *,
        now: float,
        uav_entity: CollisionEntity2D,
        obstacle_entities: list[CollisionEntity2D],
        uav_start_xy: np.ndarray,
        obstacle_start_xy_map: dict[str, np.ndarray],
        uav_velocity_hint_xy: Optional[np.ndarray] = None,
        uav_prediction_intent: Optional[dict] = None,
    ) -> tuple[float, float, str, dict[str, float], dict]:
        if not obstacle_entities:
            return 0.0, 0.0, "none", {}, {}
        uav_vel = (
            np.asarray(uav_velocity_hint_xy, dtype=float).reshape(2)
            if uav_velocity_hint_xy is not None
            else self._estimate_velocity(self._last_uav_samples)
        )
        max_prob = 0.0
        max_tau = 0.0
        dominant_obstacle = "none"
        per_obstacle_max: dict[str, float] = {str(w.entity_id): 0.0 for w in obstacle_entities}
        tau_debug: list[dict] = []
        tau = float(PREDICTION_DT_SECONDS)
        while tau <= float(PREDICTION_HORIZON_SECONDS) + 1e-9:
            uav_xy, uav_branch = self._predict_uav_xy_at_tau(
                tau=tau,
                uav_start_xy=uav_start_xy,
                fallback_vel_xy=uav_vel,
                uav_prediction_intent=uav_prediction_intent,
            )
            for obstacle in obstacle_entities:
                obstacle_samples = self._last_obstacle_samples.get(str(obstacle.entity_id), [])
                obstacle_vel = self._estimate_velocity(obstacle_samples)
                obstacle_start = np.asarray(
                    obstacle_start_xy_map.get(str(obstacle.entity_id), np.asarray(obstacle.mean_xy, dtype=float).reshape(2)),
                    dtype=float,
                ).reshape(2)
                obstacle_xy = obstacle_start + (obstacle_vel * tau)
                mu_k = obstacle_xy - uav_xy
                sigma_rel_now = np.asarray(obstacle.cov_xy, dtype=float).reshape(2, 2) + np.asarray(uav_entity.cov_xy, dtype=float).reshape(2, 2)
                process_var = max(
                    0.0,
                    (float(self._uav_process_noise_var_m2ps) + float(self._obstacle_process_noise_var_m2ps)) * float(tau),
                )
                sigma_rel = sigma_rel_now + (process_var * np.eye(2, dtype=float))
                p = hard_collision_probability_gauss_hermite(
                    mu_xy=mu_k,
                    sigma_xy=sigma_rel,
                    r_c=float(self._uav_radius_m + self._obstacle_radius_m),
                )
                if p > max_prob:
                    max_prob = float(p)
                    max_tau = float(tau)
                    dominant_obstacle = str(obstacle.entity_id)
                per_obstacle_max[str(obstacle.entity_id)] = max(per_obstacle_max.get(str(obstacle.entity_id), 0.0), float(p))
                tau_debug.append(
                    {
                        "tau": float(tau),
                        "obstacle_id": str(obstacle.entity_id),
                        "uav_xy": [float(uav_xy[0]), float(uav_xy[1])],
                        "obstacle_xy": [float(obstacle_xy[0]), float(obstacle_xy[1])],
                        "obstacle_velocity_xy": [float(obstacle_vel[0]), float(obstacle_vel[1])],
                        "uav_branch": str(uav_branch),
                        "probability": float(p),
                    }
                )
            tau += float(PREDICTION_DT_SECONDS)
        print_debug(
            "[PREDICTED-RISK-DEBUG] "
            f"raw_uav={np.asarray(uav_entity.mean_xy, dtype=float).reshape(2).tolist()} "
            f"corrected_uav={np.asarray(uav_start_xy, dtype=float).reshape(2).tolist()} "
            f"uav_velocity={uav_vel.tolist()} "
            f"dominant_obstacle={dominant_obstacle} "
            f"max_tau={max_tau:.3f} "
            f"predicted_max={max_prob:.6f} "
            f"per_obstacle_max={per_obstacle_max} "
            f"tau_samples={tau_debug}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        return float(max_prob), float(max_tau), str(dominant_obstacle), per_obstacle_max, {
            "raw_uav_xy": np.asarray(uav_entity.mean_xy, dtype=float).reshape(2).tolist(),
            "corrected_uav_xy": np.asarray(uav_start_xy, dtype=float).reshape(2).tolist(),
            "uav_velocity_xy": uav_vel.tolist(),
            "obstacle_start_xy_map": {k: np.asarray(v, dtype=float).reshape(2).tolist() for k, v in obstacle_start_xy_map.items()},
            "tau_samples": tau_debug,
        }

    def build_from_packets(
        self,
        *,
        drone_packet,
        obstacle_packets: Optional[list[tuple[str, object]]] = None,
        obstacle_packets: Optional[list[tuple[str, object]]] = None,
        now: Optional[float] = None,
        safety_state=None,
        uav_velocity_hint_xy: Optional[np.ndarray] = None,
        uav_prediction_intent: Optional[dict] = None,
    ) -> SafetyContext:
        if obstacle_packets is None:
            obstacle_packets = obstacle_packets or []
        now = time.time() if now is None else float(now)
        uav_entity = CollisionEntity2D(
            entity_id="uav",
            mean_xy=np.asarray(drone_packet.estimated_position_3d[:2], dtype=float),
            cov_xy=np.asarray(drone_packet.P_xy, dtype=float),
            bias_xy=np.asarray(drone_packet.b_xy, dtype=float),
            radius_m=float(self._uav_radius_m),
        )
        uav_corrected_xy = self._bias_corrected_xy(
            np.asarray(drone_packet.estimated_position_3d[:2], dtype=float),
            np.asarray(drone_packet.b_xy[:2], dtype=float),
        )
        obstacle_packet_map: dict[str, object] = {}
        for obstacle_id, packet in obstacle_packets:
            obstacle_key = str(obstacle_id)
            if obstacle_key not in self._risk_obstacle_ids:
                continue
            obstacle_packet_map[obstacle_key] = packet
        obstacle_entities = []
        for obstacle_key in self._risk_obstacle_ids:
            packet = obstacle_packet_map.get(obstacle_key)
            if packet is None:
                continue
            obstacle_entities.append(
                CollisionEntity2D(
                    entity_id=str(obstacle_key),
                    mean_xy=np.asarray(packet.estimated_position_3d[:2], dtype=float),
                    cov_xy=np.asarray(packet.P_xy, dtype=float),
                    bias_xy=np.asarray(packet.b_xy, dtype=float),
                    radius_m=float(self._obstacle_radius_m),
                )
            )
        risk_entity_ids = [str(entity.entity_id) for entity in obstacle_entities]
        obstacle_start_xy_map: dict[str, np.ndarray] = {}
        for obstacle_key, packet in obstacle_packet_map.items():
            obstacle_start_xy_map[str(obstacle_key)] = self._bias_corrected_xy(
                np.asarray(packet.estimated_position_3d[:2], dtype=float),
                np.asarray(packet.b_xy[:2], dtype=float),
            )
        self._push_samples(
            now=now,
            uav_corrected_xy=uav_corrected_xy,
            obstacle_packet_map=obstacle_packet_map,
        )

        summary = self._core.evaluate_scene(
            uav=uav_entity,
            obstacles=obstacle_entities,
        )
        predicted_probability, max_tau, dominant_predicted_obstacle, per_obstacle_predicted_max, predicted_debug = self._compute_predicted_collision_probability(
            now=now,
            uav_entity=uav_entity,
            obstacle_entities=obstacle_entities,
            uav_start_xy=uav_corrected_xy,
            obstacle_start_xy_map=obstacle_start_xy_map,
            uav_velocity_hint_xy=uav_velocity_hint_xy,
            uav_prediction_intent=uav_prediction_intent,
        )
        per_obstacle_probs = [
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
                "r_h": float(self._obstacle_radius_m),
                "r_c": float(self._uav_radius_m + self._obstacle_radius_m),
                "predicted_collision_probability": float(per_obstacle_predicted_max.get(str(item.entity_id), 0.0)),
            }
            for item in summary.per_entity
        ]
        collision_debug_info = {
            "sanity_case_probabilities": dict(summary.sanity_case_probabilities or {}),
            "uav_radius_m": float(self._uav_radius_m),
            "obstacle_radius_m": float(self._obstacle_radius_m),
            "collision_radius_m": float(self._uav_radius_m + self._obstacle_radius_m),
            "risk_entities": list(risk_entity_ids),
            "risk_entities_expected": list(self._risk_obstacle_ids),
            "predicted_collision_probability": float(predicted_probability),
            "prediction_horizon_seconds": float(PREDICTION_HORIZON_SECONDS),
            "prediction_dt_seconds": float(PREDICTION_DT_SECONDS),
            "max_risk_tau_seconds": float(max_tau),
            "dominant_predicted_obstacle": str(dominant_predicted_obstacle),
            "prediction_debug": predicted_debug,
        }
        return self._build_context_from_scene_summary(
            current_probability=float(summary.current_probability),
            predicted_probability=float(predicted_probability),
            per_obstacle_probs=per_obstacle_probs,
            collision_debug_info=collision_debug_info,
            dominant_obstacle_id=str(dominant_predicted_obstacle or summary.dominant_entity_id),
            safety_state=safety_state,
            now=now,
        )

    def build_from_provider(self, state_provider, now: Optional[float] = None) -> Optional[SafetyContext]:
        now = time.time() if now is None else float(now)
        safety_state = GcsSafetyStateService.build_from_provider(state_provider, now=now)
        return self.build_from_safety_state(safety_state, now=now, obstacle_packets=None)

    def build_from_safety_state(
        self,
        safety_state,
        now: Optional[float] = None,
        obstacle_packets: Optional[list[tuple[str, object]]] = None,
        obstacle_packets: Optional[list[tuple[str, object]]] = None,
    ) -> Optional[SafetyContext]:
        if obstacle_packets is None:
            obstacle_packets = obstacle_packets
        now = time.time() if now is None else float(now)
        if safety_state is None:
            return SafetyContext(
                safety_score=0.0,
                preferred_standoff_m=float(self._uav_radius_m + self._obstacle_radius_m),
                reason_tags=["collision_probability_core", "safety_state_unavailable"],
                envelope_gap_m=0.0,
                uncertainty_scale_m=0.0,
                drone_to_user_distance_xy=0.0,
                envelopes_overlap=False,
                dominant_threat_type="obstacle",
                dominant_threat_id="none",
                dominant_gap_m=0.0,
                dominant_uncertainty_scale_m=0.0,
                current_collision_probability=0.0,
                predicted_collision_probability=0.0,
                per_obstacle_collision_probabilities=[],
                collision_debug_info=None,
            )
        if obstacle_packets is None:
            return SafetyContext(
                safety_score=0.0,
                preferred_standoff_m=float(self._uav_radius_m + self._obstacle_radius_m),
                reason_tags=["collision_probability_core", "risk_obstacles_unavailable"],
                envelope_gap_m=0.0,
                uncertainty_scale_m=0.0,
                drone_to_user_distance_xy=0.0,
                envelopes_overlap=False,
                dominant_threat_type="obstacle",
                dominant_threat_id="none",
                dominant_gap_m=0.0,
                dominant_uncertainty_scale_m=0.0,
                current_collision_probability=0.0,
                predicted_collision_probability=0.0,
                per_obstacle_collision_probabilities=[],
                collision_debug_info={
                    "risk_entities": [],
                    "risk_entities_expected": list(self._risk_obstacle_ids),
                },
            )
        return self.build_from_packets(
            drone_packet=safety_state.drone_packet,
            obstacle_packets=obstacle_packets,
            now=now,
            safety_state=safety_state,
        )
