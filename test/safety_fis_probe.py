from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
import pathlib
import statistics
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FUZZY_MODULE_PATH = REPO_ROOT / "controller" / "fuzzy_safety_assessor.py"

spec = importlib.util.spec_from_file_location("fuzzy_safety_assessor_probe", FUZZY_MODULE_PATH)
fuzzy_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = fuzzy_module
spec.loader.exec_module(fuzzy_module)
FuzzySafetyAssessor = fuzzy_module.FuzzySafetyAssessor
trapezoid = fuzzy_module.trapezoid
triangle = fuzzy_module.triangle


@dataclass(frozen=True)
class ProbeResult:
    score: float
    level: str


class LegacyFuzzySafetyAssessor:
    GAP_RULES = {
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
    FRESHNESS_FACTORS = {"FRESH": 1.00, "MODERATE": 0.90, "STALE": 0.72}

    @staticmethod
    def gap_memberships(envelope_gap_m: float):
        return {
            "OVERLAP_OR_NEGATIVE": trapezoid(envelope_gap_m, -5.0, -0.15, 0.15, 0.85),
            "TIGHT": triangle(envelope_gap_m, 0.2, 1.0, 1.9),
            "CLEAR": trapezoid(envelope_gap_m, 1.3, 2.1, 8.0, 12.0),
        }

    @staticmethod
    def uncertainty_memberships(uncertainty_scale_m: float):
        return {
            "SMALL": trapezoid(uncertainty_scale_m, -1.0, 0.0, 0.75, 1.0),
            "MEDIUM": triangle(uncertainty_scale_m, 0.85, 1.2, 1.75),
            "LARGE": trapezoid(uncertainty_scale_m, 1.45, 1.95, 4.0, 5.0),
        }

    @staticmethod
    def freshness_memberships(freshness_aoi_s: float):
        return {
            "FRESH": trapezoid(freshness_aoi_s, -1.0, 0.0, 0.20, 0.45),
            "MODERATE": triangle(freshness_aoi_s, 0.25, 0.70, 1.40),
            "STALE": trapezoid(freshness_aoi_s, 0.90, 1.80, 10.0, 12.0),
        }

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 0.75:
            return "SAFE"
        if score >= 0.50:
            return "CAUTION"
        if score >= 0.25:
            return "WARNING"
        return "DANGER"

    def assess(self, envelope_gap_m: float, uncertainty_scale_m: float, freshness_aoi_s: float) -> ProbeResult:
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
        score = base_score * freshness_factor
        return ProbeResult(score=score, level=self._score_to_level(score))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(name: str, scores: list[float], levels: list[str]) -> None:
    print(
        f"{name}: score_min={min(scores):.3f} score_p10={percentile(scores, 0.10):.3f} "
        f"score_mean={statistics.fmean(scores):.3f} score_p90={percentile(scores, 0.90):.3f} score_max={max(scores):.3f}"
    )
    print(f"{name}: levels={dict(Counter(levels))}")


def run_grid() -> None:
    legacy = LegacyFuzzySafetyAssessor()
    current = FuzzySafetyAssessor()

    gaps = (-0.4, -0.1, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0, 2.6, 3.2)
    geometric_uncertainties = (0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8)
    quality_margins = (0.0, 0.2, 0.5, 0.9)
    aois = (0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 2.0)

    legacy_scores, legacy_levels = [], []
    current_scores, current_levels = [], []
    for gap in gaps:
        for geom_uncertainty in geometric_uncertainties:
            for quality_margin in quality_margins:
                for aoi in aois:
                    legacy_result = legacy.assess(gap, geom_uncertainty + quality_margin, aoi)
                    legacy_scores.append(legacy_result.score)
                    legacy_levels.append(legacy_result.level)
            for aoi in aois:
                current_result = current.assess(gap, geom_uncertainty, gap < 0.0, aoi)
                current_scores.append(current_result.safety_score)
                current_levels.append(current_result.safety_level)

    summarize("legacy", legacy_scores, legacy_levels)
    summarize("current", current_scores, current_levels)

    print("representative_gap_probe")
    for gap in (0.0, 0.5, 1.0, 1.8, 2.8):
        result = current.assess(gap, uncertainty_scale_m=0.7, envelopes_overlap=gap < 0.0, freshness_aoi_s=0.2)
        print(f"  gap={gap:.2f} -> score={result.safety_score:.3f} level={result.safety_level}")

    print("representative_uncertainty_probe")
    for uncertainty in (0.25, 0.55, 0.9, 1.3, 1.7):
        result = current.assess(1.2, uncertainty_scale_m=uncertainty, envelopes_overlap=False, freshness_aoi_s=0.2)
        print(f"  geometric_uncertainty={uncertainty:.2f} -> score={result.safety_score:.3f} level={result.safety_level}")

    print("representative_aoi_probe")
    for aoi in (0.05, 0.3, 0.7, 1.2, 2.0):
        result = current.assess(1.2, uncertainty_scale_m=0.7, envelopes_overlap=False, freshness_aoi_s=aoi)
        print(f"  max_aoi_s={aoi:.2f} -> score={result.safety_score:.3f} level={result.safety_level}")

    print("assumed_runtime_ranges")
    print("  envelope_gap_m ~= [-0.4, 3.2]")
    print("  geometric_uncertainty_m ~= [0.2, 1.8]")
    print("  max_aoi_s ~= [0.0, 2.0]")


if __name__ == "__main__":
    run_grid()
