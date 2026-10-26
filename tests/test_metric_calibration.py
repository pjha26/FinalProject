"""Fit-only calibration recovery, robustness and paired promotion checks."""
import unittest
import numpy as np
from src.metric_calibration import apply_power_depth, fit_power_depth, paired_promotion


class MetricCalibrationTests(unittest.TestCase):
    def test_recovers_known_monotonic_mapping(self):
        x = np.linspace(.2, 5, 200)[None, :]
        fitted = fit_power_depth([x], [3 * x ** .8])
        self.assertAlmostEqual(fitted["scale"], 3, places=8)
        self.assertAlmostEqual(fitted["exponent"], .8, places=8)
        np.testing.assert_allclose(apply_power_depth(x, fitted["scale"], fitted["exponent"]), 3*x**.8)

    def test_robust_to_small_fraction_of_large_outliers(self):
        x = np.linspace(.2, 5, 1000)[None, :]
        reference = 3 * x ** .8
        reference[0, ::100] *= 10
        fitted = fit_power_depth([x], [reference])
        self.assertLess(abs(fitted["scale"] - 3), .1)
        self.assertLess(abs(fitted["exponent"] - .8), .05)

    def test_equal_samples_per_frame(self):
        a = np.linspace(.2, 5, 30)[None, :]
        b = np.linspace(.2, 5, 80)[None, :]
        fitted = fit_power_depth([a,b], [a*2,b*2])
        self.assertEqual(fitted["sampled_fit_pixels"], 60)

    def test_invalid_fit_or_mapping_rejected(self):
        for predictions, references in [([],[]), ([np.ones((1,30))],[np.ones((1,30))]), ([np.ones((1,30))],[np.ones((1,31))]), ([np.ones((1,2))],[np.ones((1,2))])]:
            with self.assertRaises(ValueError):
                fit_power_depth(predictions,references)
        for scale, exponent in [(0,1),(1,-1),(1,5),(float("nan"),1)]:
            with self.assertRaises(ValueError):
                apply_power_depth(np.ones((2,2)),scale,exponent)
        with self.assertRaises(ValueError):
            apply_power_depth(np.full((1,2),1e308),2,4)

    def rows(self, error, scene="a"):
        return [dict(scene=scene,frame=i,abs_rel=error) for i in range(10)]

    def test_promotion_requires_gates_gain_and_scene_guard(self):
        self.assertTrue(paired_promotion(self.rows(.3),self.rows(.2))["passed"])
        self.assertFalse(paired_promotion(self.rows(.3),self.rows(.26))["passed"])
        self.assertFalse(paired_promotion(self.rows(.2),self.rows(.195))["passed"])
        baseline=self.rows(.4,"a")+self.rows(.1,"b")
        candidate=self.rows(.15,"a")+self.rows(.12,"b")
        self.assertFalse(paired_promotion(baseline,candidate)["passed"])

    def test_pair_identity_mismatch_and_duplicates_rejected(self):
        a=self.rows(.3);b=self.rows(.2)
        b[0]["frame"]=999
        with self.assertRaises(ValueError):
            paired_promotion(a,b)
        with self.assertRaises(ValueError):
            paired_promotion(a+a,a+a)


if __name__ == "__main__":
    unittest.main()

