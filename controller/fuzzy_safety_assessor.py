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


class FuzzySafetyAssessor:
    GAP_RULES: Dict[Tuple[str, str], float] = {
        ("OVERLAP_OR_NEGATIVE", "LARGE"): 0.05,
        ("OVERLAP_OR_NEGATIVE", "MEDIUM"): 0.10,
        ("OVERLAP_OR_NEGATIVE", "SMALL"): 0.18,
        ("TIGHT", "LARGE"): 0.20,
        ("TIGHT", "MEDIUM"): 0.38,
        ("TIGHT", "SMALL"): 0.52,
        ("CLEAR", "LARGE"): 0.45,
        ("CLEAR", "MEDIUM"): 0.68,
        ("CLEAR", "SMALL"): 0.90,
    }

    @staticmethod
    def gap_memberships(envelope_gap_m: float) -> Dict[str, float]:
        return {
            "OVERLAP_OR_NEGATIVE": trapezoid(envelope_gap_m, -5.0, -0.1, 0.0, 1.0),
            "TIGHT": triangle(envelope_gap_m, 0.0, 1.0, 2.5),
            "CLEAR": trapezoid(envelope_gap_m, 1.5, 3.0, 10.0, 12.0),
        }

    @staticmethod
    def uncertainty_memberships(uncertainty_scale_m: float) -> Dict[str, float]:
        return {
            "SMALL": trapezoid(uncertainty_scale_m, -1.0, 0.0, 1.0, 2.0),
            "MEDIUM": triangle(uncertainty_scale_m, 1.0, 2.0, 3.5),
            "LARGE": trapezoid(uncertainty_scale_m, 2.5, 4.0, 10.0, 12.0),
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

    def assess(self, envelope_gap_m: float, uncertainty_scale_m: float, envelopes_overlap: bool) -> FuzzyAssessmentResult:
        gap_memberships = self.gap_memberships(envelope_gap_m)
        uncertainty_memberships = self.uncertainty_memberships(uncertainty_scale_m)

        weighted_sum = 0.0
        total_weight = 0.0
        for gap_label, gap_mu in gap_memberships.items():
            for uncertainty_label, uncertainty_mu in uncertainty_memberships.items():
                weight = gap_mu * uncertainty_mu
                if weight <= 0.0:
                    continue
                total_weight += weight
                weighted_sum += weight * self.GAP_RULES[(gap_label, uncertainty_label)]

        safety_score = weighted_sum / total_weight if total_weight > 0.0 else 0.5
        safety_level = self._score_to_level(safety_score)
        planning_bias = self._planning_bias(safety_level)
        preferred_standoff_m = max(0.5, uncertainty_scale_m + self._safety_buffer(safety_level))

        dominant_gap_label = self._dominant_label(gap_memberships)
        dominant_uncertainty_label = self._dominant_label(uncertainty_memberships)

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

        return FuzzyAssessmentResult(
            safety_score=float(safety_score),
            safety_level=safety_level,
            planning_bias=planning_bias,
            preferred_standoff_m=float(preferred_standoff_m),
            reason_tags=reason_tags,
            dominant_gap_label=dominant_gap_label,
            dominant_uncertainty_label=dominant_uncertainty_label,
        )
