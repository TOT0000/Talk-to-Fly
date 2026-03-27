from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


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

    def to_prompt_block(self) -> str:
        freshness = "unknown" if self.timing_freshness_s is None else f"{self.timing_freshness_s:.2f} s"
        max_aoi = "unknown" if self.max_aoi_s is None else f"{self.max_aoi_s:.2f} s"
        latest_gen = "unknown" if self.latest_generation_timestamp is None else f"{self.latest_generation_timestamp:.3f}"
        latest_recv = "unknown" if self.latest_receive_timestamp is None else f"{self.latest_receive_timestamp:.3f}"
        return (
            f"safety_score: {self.safety_score:.3f}\n"
            f"safety_level: {self.safety_level}\n"
            f"planning_bias: {self.planning_bias}\n"
            f"reason_tags: {self.reason_tags}\n"
            f"drone_to_user_distance_xy: {self.drone_to_user_distance_xy:.2f}\n"
            f"envelope_gap_m(centerline_ray_gap): {self.envelope_gap_m:.2f}\n"
            f"uncertainty_scale_m: {self.uncertainty_scale_m:.2f}\n"
            f"envelopes_overlap(centerline): {self.envelopes_overlap}\n"
            f"latest_generation_timestamp: {latest_gen}\n"
            f"latest_receive_timestamp: {latest_recv}\n"
            f"timing_freshness_s: {freshness}\n"
            f"max_aoi_s: {max_aoi}"
        )
