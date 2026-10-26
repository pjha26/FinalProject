"""Distance display labels/geometry and source-statistic checks."""
import unittest
from unittest.mock import patch

import numpy as np

from src.control_visualize import render_control_frame
from src.metric_control import clearance_sample_mask, corridor_clearance
from src.preprocess import camera_intrinsics


class ControlDisplayTests(unittest.TestCase):
    def fixture(self):
        depth = np.full((480, 640), 1000.0, dtype=np.float32)
        depth[210:250, 300:340] = 12
        return np.zeros((480, 640, 3), dtype=np.uint8), np.ones((192, 640), dtype=np.float32), depth, camera_intrinsics(640, 480)

    def test_reference_mask_matches_clearance(self):
        rgb, inverse, depth, k = self.fixture()
        mask = clearance_sample_mask(depth, k)
        self.assertTrue(mask.any())
        self.assertTrue((depth[mask] <= corridor_clearance(depth, k)).all())
        self.assertFalse(mask[:, :290].any())
        self.assertFalse(mask[:, 350:].any())

    def test_clear_scene_has_no_fake_obstacle_outline(self):
        depth = np.full((480, 640), 1000.0)
        self.assertFalse(clearance_sample_mask(depth, camera_intrinsics(640, 480)).any())

    def test_render_separates_reference_model_and_gap(self):
        from PIL import ImageDraw
        rgb, inverse, depth, k = self.fixture()
        texts = []
        original = ImageDraw.ImageDraw.text
        def record_text(draw, xy, text, *args, **kwargs):
            texts.append(text)
            return original(draw, xy, text, *args, **kwargs)
        with patch.object(ImageDraw.ImageDraw, "text", record_text):
            result = render_control_frame(rgb, inverse, depth, k, 12, 18, 1, .5, "Following route", 10.5)
        self.assertEqual(result.shape, (742, 1920, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertIn("Camera clearance - SIMULATOR REFERENCE", texts)
        self.assertIn("Model estimate - UNVALIDATED", texts)
        self.assertIn("18.0 m  (not used for control)", texts)
        self.assertTrue(any("Vehicle gap (simulator): 10.5 m" in text for text in texts))

    def test_invalid_measurement_rejected(self):
        args = self.fixture()
        with self.assertRaises(ValueError):
            render_control_frame(*args, 12, float("nan"), 1, 0, "invalid")


if __name__ == "__main__":
    unittest.main()

