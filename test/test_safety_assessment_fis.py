from __future__ import annotations

import ast
import importlib.util
import pathlib
import statistics
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FUZZY_MODULE_PATH = REPO_ROOT / "controller" / "fuzzy_safety_assessor.py"
ASSESSMENT_MODULE_PATH = REPO_ROOT / "controller" / "gcs_safety_assessment.py"
STATE_PACKET_PATH = REPO_ROOT / "controller" / "state_packet.py"
LOCALIZATION_ESTIMATOR_PATH = REPO_ROOT / "controller" / "localization_estimator.py"


spec = importlib.util.spec_from_file_location("fuzzy_safety_assessor_under_test", FUZZY_MODULE_PATH)
fuzzy_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = fuzzy_module
spec.loader.exec_module(fuzzy_module)
FuzzySafetyAssessor = fuzzy_module.FuzzySafetyAssessor


class SafetyAssessmentFisTests(unittest.TestCase):
    def test_uncertainty_scale_is_geometry_only_in_assessment(self):
        source = ASSESSMENT_MODULE_PATH.read_text()
        self.assertNotIn("quality_margin_m", source)
        self.assertNotIn("range_residual_rms_m", source)
        self.assertIn("uncertainty_scale_m = geometric_uncertainty_m", source)

    def test_residual_diagnostics_are_still_present_in_packet_pipeline(self):
        packet_source = STATE_PACKET_PATH.read_text()
        estimator_source = LOCALIZATION_ESTIMATOR_PATH.read_text()
        self.assertIn("range_residuals", packet_source)
        self.assertIn("range_residual_rms_m", packet_source)
        self.assertIn("normalized_range_residual_rms", packet_source)
        self.assertIn("range_residual_rms_m=residual_rms_m", estimator_source)
        self.assertIn("normalized_range_residual_rms=normalized_residual_rms", estimator_source)

    def test_score_reacts_to_gap_uncertainty_and_aoi(self):
        assessor = FuzzySafetyAssessor()

        gap_low = assessor.assess(envelope_gap_m=0.1, uncertainty_scale_m=0.7, envelopes_overlap=False, freshness_aoi_s=0.2)
        gap_high = assessor.assess(envelope_gap_m=2.4, uncertainty_scale_m=0.7, envelopes_overlap=False, freshness_aoi_s=0.2)
        self.assertGreater(gap_high.safety_score, gap_low.safety_score)

        uncertainty_low = assessor.assess(envelope_gap_m=1.4, uncertainty_scale_m=0.3, envelopes_overlap=False, freshness_aoi_s=0.2)
        uncertainty_high = assessor.assess(envelope_gap_m=1.4, uncertainty_scale_m=1.5, envelopes_overlap=False, freshness_aoi_s=0.2)
        self.assertGreater(uncertainty_low.safety_score, uncertainty_high.safety_score)

        freshness_fresh = assessor.assess(envelope_gap_m=1.4, uncertainty_scale_m=0.7, envelopes_overlap=False, freshness_aoi_s=0.1)
        freshness_stale = assessor.assess(envelope_gap_m=1.4, uncertainty_scale_m=0.7, envelopes_overlap=False, freshness_aoi_s=1.8)
        self.assertGreater(freshness_fresh.safety_score, freshness_stale.safety_score)


    def test_large_positive_gap_does_not_fall_back_to_overlap_tag(self):
        assessor = FuzzySafetyAssessor()

        result = assessor.assess(
            envelope_gap_m=9.876,
            uncertainty_scale_m=2.564,
            envelopes_overlap=False,
            freshness_aoi_s=0.07,
        )

        self.assertNotAlmostEqual(result.safety_score, 0.5, places=6)
        self.assertIn("gap_clear", result.reason_tags)
        self.assertNotIn("gap_overlap_or_negative", result.reason_tags)
        self.assertNotIn("envelope_overlap", result.reason_tags)

    def test_grid_spans_multiple_levels(self):
        assessor = FuzzySafetyAssessor()
        levels = set()
        scores = []
        for gap in (-0.3, 0.0, 0.5, 1.0, 1.8, 3.0):
            for uncertainty in (0.2, 0.4, 0.7, 1.0, 1.4, 1.8):
                for aoi in (0.0, 0.2, 0.5, 0.9, 1.4, 2.0):
                    result = assessor.assess(
                        envelope_gap_m=float(gap),
                        uncertainty_scale_m=float(uncertainty),
                        envelopes_overlap=bool(gap < 0.0),
                        freshness_aoi_s=float(aoi),
                    )
                    levels.add(result.safety_level)
                    scores.append(result.safety_score)
        self.assertTrue({"DANGER", "WARNING", "CAUTION", "SAFE"}.issubset(levels))
        self.assertGreater(max(scores) - min(scores), 0.7)


if __name__ == "__main__":
    unittest.main()
