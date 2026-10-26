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





class AbsRelScaleTests(unittest.TestCase):
    def test_relative_error_weighting_favors_near_reference(self):
        from src.metric_calibration import fit_absrel_scale
        p=np.ones((1,2));g=np.array([[1.,10.]])
        scale=fit_absrel_scale([p],[g])
        self.assertEqual(scale,1.)
        cost=lambda s:np.mean(np.abs(s*p-g)/g)
        self.assertTrue(all(cost(scale)<=cost(s)+1e-12 for s in np.linspace(.5,11,101)))

    def test_exact_mapping_and_frame_balance(self):
        from src.metric_calibration import fit_absrel_scale
        p=np.arange(1,31,dtype=float)[None,:]
        self.assertEqual(fit_absrel_scale([p],[p*2]),2.)
        a=fit_absrel_scale([p,p],[p*2,p*3])
        b=fit_absrel_scale([np.tile(p,(1,10)),p],[np.tile(p*2,(1,10)),p*3])
        self.assertEqual(a,b)

    def test_bad_pairs_or_unstable_ratios_rejected(self):
        from src.metric_calibration import fit_absrel_scale
        for p,g in [([],[]),([np.ones((1,2))],[np.ones((1,3))]),([np.zeros((1,2))],[np.ones((1,2))]),([np.full((1,2),1e300)],[np.full((1,2),1e-300)])]:
            with self.assertRaises(ValueError):
                fit_absrel_scale(p,g)



class HoldoutGuardTests(unittest.TestCase):
    def test_consumed_holdout_cannot_be_reused(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tools.evaluate_metric_holdout import run
        base=Path(__file__).resolve().parents[1]/"tmp";base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(base)) as folder:
            p=Path(folder);freeze=p/"freeze.json"
            freeze.write_text(json.dumps(dict(status="frozen_before_holdout_acquisition",candidate_path="unused.json",candidate_sha256="abc",checkpoint_sha256={})),encoding="utf-8")
            (p/"holdout_consumed.json").write_text("{}",encoding="utf-8")
            with patch("tools.evaluate_metric_holdout.sha256",return_value="abc"),patch("tools.evaluate_metric_holdout.checkpoint_hashes",return_value={}),patch("tools.evaluate_metric_holdout.load_model") as load:
                with self.assertRaisesRegex(ValueError,"already consumed"):
                    run(freeze,p/"new_output")
                load.assert_not_called()
                self.assertFalse((p/"new_output").exists())

    def test_changed_candidate_is_refused_before_loading(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tools.evaluate_metric_holdout import run
        base=Path(__file__).resolve().parents[1]/"tmp";base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(base)) as folder:
            p=Path(folder);freeze=p/"freeze.json"
            freeze.write_text(json.dumps(dict(status="frozen_before_holdout_acquisition",candidate_path="unused.json",candidate_sha256="abc",checkpoint_sha256={})),encoding="utf-8")
            with patch("tools.evaluate_metric_holdout.sha256",return_value="changed"),patch("tools.evaluate_metric_holdout.load_model") as load:
                with self.assertRaisesRegex(ValueError,"identity mismatch"):
                    run(freeze,p/"new_output")
                load.assert_not_called()

if __name__ == "__main__":
    unittest.main()
