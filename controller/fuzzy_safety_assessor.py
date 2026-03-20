from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / max(b - a, 1e-9)
    return (d - x) / max(d - c, 1e-9)


def triangle(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / max(b - a, 1e-9)
    return (c - x) / max(c - b, 1e-9)


@dataclass(frozen=True)
class FuzzyAssessmentResult:
    safety_score: float
    safety_level: str
    planning_bias: str
    preferred_standoff_m: float
    reason_tags: List[str]
    dominant_gap_label: str
    dominant_uncertainty_label: str
    dominant_freshness_label: str


class FuzzySafetyAssessor:
    GAP_RULES: Dict[Tuple[str, str], float] = {
        ("OVERLAP_OR_NEGATIVE", "LARGE"): 0.03,
        ("OVERLAP_OR_NEGATIVE", "MEDIUM"): 0.08,
        ("OVERLAP_OR_NEGATIVE", "SMALL"): 0.18,
        ("TIGHT", "LARGE"): 0.16,
        ("TIGHT", "MEDIUM"): 0.32,
        ("TIGHT", "SMALL"): 0.56,
        ("CLEAR", "LARGE"): 0.42,
        ("CLEAR", "MEDIUM"): 0.76,
        ("CLEAR", "SMALL"): 0.97,
    }


    FRESHNESS_FACTORS: Dict[str, float] = {
        "FRESH": 1.00,
        "MODERATE": 0.82,
        "STALE": 0.58,
    }

    @staticmethod
    def gap_memberships(envelope_gap_m: float) -> Dict[str, float]:
        return {
            "OVERLAP_OR_NEGATIVE": trapezoid(envelope_gap_m, -5.0, -0.15, 0.15, 0.85),
            "TIGHT": triangle(envelope_gap_m, 0.2, 1.0, 1.9),
            "CLEAR": trapezoid(envelope_gap_m, 1.3, 2.1, 8.0, 12.0),
        }

    @staticmethod
    def uncertainty_memberships(uncertainty_scale_m: float) -> Dict[str, float]:
        return {
            "SMALL": trapezoid(uncertainty_scale_m, -1.0, 0.0, 0.75, 1.0),
            "MEDIUM": triangle(uncertainty_scale_m, 0.85, 1.2, 1.75),
            "LARGE": trapezoid(uncertainty_scale_m, 1.45, 1.95, 4.0, 5.0),
        }

    @staticmethod
    def freshness_memberships(freshness_aoi_s: float) -> Dict[str, float]:
        return {
            "FRESH": trapezoid(freshness_aoi_s, -1.0, 0.0, 0.20, 0.45),
            "MODERATE": triangle(freshness_aoi_s, 0.25, 0.70, 1.40),
            "STALE": trapezoid(freshness_aoi_s, 0.90, 1.80, 10.0, 12.0),
        }

    @staticmethod
    def _dominant_label(memberships: Dict[str, float]) -> str:
        return max(memberships.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 0.75:
            return "SAFE"
        if score >= 0.50:
            return "CAUTION"
        if score >= 0.25:
            return "WARNING"
        return "DANGER"

    @staticmethod
    def _planning_bias(level: str) -> str:
        return {
            "SAFE": "efficiency",
            "CAUTION": "balanced",
            "WARNING": "safety",
            "DANGER": "safety",
        }[level]

    @staticmethod
    def _safety_buffer(level: str) -> float:
        return {
            "SAFE": 0.5,
            "CAUTION": 1.0,
            "WARNING": 1.5,
            "DANGER": 2.0,
        }[level]

    def assess(
        self,
        envelope_gap_m: float,
        uncertainty_scale_m: float,
        envelopes_overlap: bool,
        freshness_aoi_s: float = 0.0,
    ) -> FuzzyAssessmentResult:
        gap_memberships = self.gap_memberships(envelope_gap_m)
        uncertainty_memberships = self.uncertainty_memberships(uncertainty_scale_m)
        freshness_memberships = self.freshness_memberships(freshness_aoi_s)

        weighted_sum = 0.0
        total_weight = 0.0
        for gap_label, gap_mu in gap_memberships.items():
            for uncertainty_label, uncertainty_mu in uncertainty_memberships.items():
                weight = gap_mu * uncertainty_mu
                if weight <= 0.0:
                    continue
                total_weight += weight
                weighted_sum += weight * self.GAP_RULES[(gap_label, uncertainty_label)]

        base_score = weighted_sum / total_weight if total_weight > 0.0 else 0.5
        freshness_weight_sum = 0.0
        freshness_total_weight = 0.0
        for freshness_label, freshness_mu in freshness_memberships.items():
            if freshness_mu <= 0.0:
                continue
            freshness_total_weight += freshness_mu
            freshness_weight_sum += freshness_mu * self.FRESHNESS_FACTORS[freshness_label]
        freshness_factor = (
            freshness_weight_sum / freshness_total_weight
            if freshness_total_weight > 0.0
            else self.FRESHNESS_FACTORS["FRESH"]
        )
        safety_score = base_score * freshness_factor
        safety_level = self._score_to_level(safety_score)
        planning_bias = self._planning_bias(safety_level)
        preferred_standoff_m = max(0.5, uncertainty_scale_m + self._safety_buffer(safety_level))

        dominant_gap_label = self._dominant_label(gap_memberships)
        dominant_uncertainty_label = self._dominant_label(uncertainty_memberships)
        dominant_freshness_label = self._dominant_label(freshness_memberships)

        reason_tags = []
        if envelopes_overlap or envelope_gap_m < 0.0:
            reason_tags.append("envelope_overlap")
        if dominant_gap_label == "TIGHT":
            reason_tags.append("gap_tight")
        elif dominant_gap_label == "CLEAR":
            reason_tags.append("gap_clear")
        else:
            reason_tags.append("gap_overlap_or_negative")

        if dominant_uncertainty_label == "LARGE":
            reason_tags.append("uncertainty_large")
        elif dominant_uncertainty_label == "MEDIUM":
            reason_tags.append("uncertainty_medium")
        else:
            reason_tags.append("uncertainty_small")

        if dominant_freshness_label == "STALE":
            reason_tags.append("aoi_stale")
        elif dominant_freshness_label == "MODERATE":
            reason_tags.append("aoi_moderate")
        else:
            reason_tags.append("aoi_fresh")

        return FuzzyAssessmentResult(
            safety_score=float(safety_score),
            safety_level=safety_level,
            planning_bias=planning_bias,
            preferred_standoff_m=float(preferred_standoff_m),
            reason_tags=reason_tags,
            dominant_gap_label=dominant_gap_label,
            dominant_uncertainty_label=dominant_uncertainty_label,
            dominant_freshness_label=dominant_freshness_label,
        )
