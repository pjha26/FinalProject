"""Numerical and failure-path checks for the separate metric/control experiment."""
import unittest
import numpy as np
from src.metric_control import fixed_scale, scaled_depth, calibration_gate, corridor_clearance, steering_to_waypoint, bounded_control
from src.preprocess import camera_intrinsics


class MetricControlTests(unittest.TestCase):
    def test_fixed_scale_uses_calibration_only(self):
        self.assertEqual(fixed_scale([np.ones((2, 3)), np.ones((2, 3))], [np.full((2, 3), 2), np.full((2, 3), 4)]), 3.0)

    def test_invalid_scale_and_geometry(self):
        for scale in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                scaled_depth(np.ones((2, 2)), scale)
        with self.assertRaises(ValueError):
            fixed_scale([np.ones((2, 2))], [np.ones((1, 2))])
        with self.assertRaises(ValueError):
            fixed_scale([], [])

    def test_frozen_acceptance_thresholds(self):
        self.assertTrue(calibration_gate([dict(abs_rel=.20)])["passed"])
        self.assertFalse(calibration_gate([dict(abs_rel=.26)])["passed"])
        self.assertFalse(calibration_gate([dict(abs_rel=.1)] * 9 + [dict(abs_rel=1.0)])["passed"])
        with self.assertRaises(ValueError):
            calibration_gate([dict(abs_rel=float("nan"))])

    def test_obstacle_corridor_and_off_axis_rejection(self):
        k = camera_intrinsics(640, 480)
        depth = np.full((480, 640), 1000.0)
        depth[210:250, 300:340] = 12
        self.assertAlmostEqual(corridor_clearance(depth, k), 12)
        depth[210:250, 300:340] = 4
        self.assertAlmostEqual(corridor_clearance(depth, k), 4)

    def test_nonfinite_or_zero_depth_is_rejected(self):
        for invalid in [0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                corridor_clearance(np.full((480, 640), invalid), camera_intrinsics(640, 480))

    def test_clear_depth_gives_no_obstacle_within_range(self):
        self.assertEqual(corridor_clearance(np.full((480, 640), 1000.0), camera_intrinsics(640, 480)), 50.0)

    def test_steering_sign_and_bound(self):
        self.assertEqual(steering_to_waypoint((0, 0), 0, (10, 0)), 0)
        self.assertGreater(steering_to_waypoint((0, 0), 0, (10, 1)), 0)
        self.assertLess(steering_to_waypoint((0, 0), 0, (10, -1)), 0)
        self.assertEqual(steering_to_waypoint((0, 0), 0, (0, 10)), .5)

    def test_stop_on_obstacle_invalid_stale_and_latched(self):
        for distance, age in [(3, .1), (None, .1), (float("nan"), .1), (20, .36)]:
            control = bounded_control(2, distance, age)
            self.assertEqual(control["throttle"], 0)
            self.assertEqual(control["brake"], 1)
        self.assertEqual(bounded_control(0, 30, .1, stop=True)["reason"], "latched_stop")

    def test_low_speed_and_overspeed(self):
        self.assertGreater(bounded_control(0, 30, .1)["throttle"], 0)
        self.assertEqual(bounded_control(3, 30, .1)["throttle"], 0)
        self.assertGreater(bounded_control(3, 30, .1)["brake"], 0)
        self.assertEqual(bounded_control(-1, 30, .1)["brake"], 1)


if __name__ == "__main__":
    unittest.main()

