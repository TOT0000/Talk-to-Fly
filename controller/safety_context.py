from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SafetyContext:
    safety_score: float
    safety_level: str
    planning_bias: str
    preferred_standoff_m: float
    reason_tags: List[str]
    envelope_gap_m: float
    uncertainty_scale_m: float
    drone_to_user_distance_xy: float
    envelopes_overlap: bool
    latest_generation_timestamp: Optional[float] = None
    latest_receive_timestamp: Optional[float] = None
    timing_freshness_s: Optional[float] = None
    max_aoi_s: Optional[float] = None
    dominant_threat_type: str = "user"
    dominant_threat_id: str = "user"
    dominant_gap_m: float = 0.0
    dominant_uncertainty_scale_m: float = 1.0
    dominant_freshness_s: Optional[float] = None
    task_points_summary: Optional[List[Dict[str, Any]]] = None
    obstacles_summary: Optional[List[Dict[str, Any]]] = None
    path_summary: Optional[Dict[str, Any]] = None
    candidate_targets_summary: Optional[List[Dict[str, Any]]] = None
    candidate_path_summaries: Optional[List[Dict[str, Any]]] = None

    def to_prompt_block(self) -> str:
        freshness = "unknown" if self.timing_freshness_s is None else f"{self.timing_freshness_s:.2f} s"
        max_aoi = "unknown" if self.max_aoi_s is None else f"{self.max_aoi_s:.2f} s"
        latest_gen = "unknown" if self.latest_generation_timestamp is None else f"{self.latest_generation_timestamp:.3f}"
        latest_recv = "unknown" if self.latest_receive_timestamp is None else f"{self.latest_receive_timestamp:.3f}"
        dominant_freshness = "unknown" if self.dominant_freshness_s is None else f"{self.dominant_freshness_s:.2f} s"
        task_points = self.task_points_summary or []
        obstacles = self.obstacles_summary or []
        candidate_targets = self.candidate_targets_summary or []
        candidate_paths = self.candidate_path_summaries or []
        task_points_block = (
            "\n".join(
                f"- {row.get('id')}: x={float(row.get('x')):.2f}, y={float(row.get('y')):.2f}, z={float(row.get('z')):.2f}"
                for row in task_points
            )
            if task_points
            else "- (n/a)"
        )
        obstacle_block = (
            "\n".join(
                (
                    f"- {row.get('id')}: est=({float(row.get('est_x')):.2f},{float(row.get('est_y')):.2f}), "
                    f"env=major:{float(row.get('major_axis_m')):.2f}/minor:{float(row.get('minor_axis_m')):.2f}/ori:{float(row.get('orientation_deg')):.1f}, "
                    f"freshness_s={float(row.get('freshness_s', 0.0)):.2f}"
                )
                for row in obstacles
            )
            if obstacles
            else "- (n/a)"
        )
        candidate_targets_block = (
            "\n".join(
                f"- {row.get('id')}: est=({float(row.get('x')):.2f},{float(row.get('y')):.2f},{float(row.get('z')):.2f})"
                for row in candidate_targets
            )
            if candidate_targets
            else "- (n/a)"
        )
        candidate_paths_block = (
            "\n".join(
                (
                    f"- to {row.get('target_id')}: path_clear={row.get('path_clear')}, "
                    f"blocking_entity={row.get('blocking_entity')}, "
                    f"corridor_min_gap_m={row.get('corridor_min_gap')}"
                )
                for row in candidate_paths
            )
            if candidate_paths
            else "- (n/a)"
        )
        return (
            f"safety_score: {self.safety_score:.3f}\n"
            f"safety_level: {self.safety_level}\n"
            f"planning_bias: {self.planning_bias}\n"
            f"reason_tags: {self.reason_tags}\n"
            f"dominant_threat_type: {self.dominant_threat_type}\n"
            f"dominant_threat_id: {self.dominant_threat_id}\n"
            f"dominant_gap_m: {self.dominant_gap_m:.2f}\n"
            f"dominant_uncertainty_scale_m: {self.dominant_uncertainty_scale_m:.2f}\n"
            f"dominant_freshness_s: {dominant_freshness}\n"
            f"drone_to_user_distance_xy: {self.drone_to_user_distance_xy:.2f}\n"
            f"envelope_gap_m(centerline_ray_gap): {self.envelope_gap_m:.2f}\n"
            f"uncertainty_scale_m: {self.uncertainty_scale_m:.2f}\n"
            f"envelopes_overlap(centerline): {self.envelopes_overlap}\n"
            f"latest_generation_timestamp: {latest_gen}\n"
            f"latest_receive_timestamp: {latest_recv}\n"
            f"timing_freshness_s: {freshness}\n"
            f"max_aoi_s: {max_aoi}\n"
            f"TaskPoints:\n{task_points_block}\n"
            f"CandidateTargets:\n{candidate_targets_block}\n"
            f"Obstacles:\n{obstacle_block}\n"
            f"PathSummaries:\n{candidate_paths_block}"
        )
