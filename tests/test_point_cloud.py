"""Relative cloud geometry, bounded exports and optional CLI integration."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import project
from src.point_cloud import project_points, export_point_cloud


class PointCloudTests(unittest.TestCase):
    def setUp(self):
        (project.ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(project.ROOT / "tmp"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_plane_backprojection_and_rgb_are_aligned_without_mutating_depth(self):
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        depth = np.full((2, 3), 2, dtype=np.float32)
        original = depth.copy()
        points, colours, metadata = project_points(rgb, depth, np.eye(3))
        np.testing.assert_array_equal(points, [[0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1]])
        np.testing.assert_array_equal(colours, rgb.reshape(-1, 3))
        np.testing.assert_array_equal(depth, original)
        self.assertEqual(metadata["raw_depth_median"], 2)
        self.assertIn("not metres", metadata["units"])

    def test_projection_is_scale_ambiguous_and_uses_supplied_intrinsics(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.full((2, 3), 3, dtype=np.float32)
        matrix = np.asarray([[2, 0, 1], [0, 2, .5], [0, 0, 1]])
        a = project_points(rgb, depth, matrix)[0]
        b = project_points(rgb, depth * 7, matrix)[0]
        np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(a[0], [-.5, -.25, 1])

    def test_invalid_sky_floor_and_far_outliers_are_counted(self):
        rgb = np.zeros((2, 4, 3), dtype=np.uint8)
        depth = np.asarray([[1, 1, 1, 1], [np.nan, -1, 1e6, 100]], dtype=np.float32)
        points, _, metadata = project_points(rgb, depth, np.eye(3))
        self.assertEqual(len(points), 4)
        self.assertEqual(metadata["invalid_or_floor_pixels"], 3)
        self.assertEqual(metadata["far_pixels_filtered"], 1)
        self.assertEqual(metadata["retained_before_sampling"], 4)

    def test_point_budget_is_bounded_deterministic_and_preserves_endpoints(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        depth = np.ones((100, 200), dtype=np.float32)
        a = project_points(rgb, depth, np.eye(3), max_points=31)[0]
        b = project_points(rgb, depth, np.eye(3), max_points=31)[0]
        self.assertEqual(len(a), 31)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[[0, -1]], [[0, 0, 1], [199, 99, 1]])

    def test_invalid_geometry_and_empty_inputs_cannot_export_a_fake_cloud(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.ones((2, 3), dtype=np.float32)
        for matrix in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.asarray([[1, 1, 0], [1, 1, 0], [0, 0, 1]])):
            with self.assertRaises(ValueError):
                project_points(rgb, depth, matrix)
        for invalid in (np.zeros((2, 3)), np.full((2, 3), np.nan), np.full((2, 3), 1e6), np.asarray([])):
            with self.assertRaises(ValueError):
                project_points(rgb, invalid, np.eye(3))
        for budget in (0, 10001):
            with self.assertRaises(ValueError):
                project_points(rgb, depth, np.eye(3), budget)

    def test_real_export_has_finite_ply_vertices_png_and_offline_safe_viewer(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        details = dict(model_name="Model </script><unsafe>", backend="fixture")
        result = export_point_cloud(self.root, rgb, np.ones((3, 4), dtype=np.float32), np.eye(3), details)
        payload = (self.root / result["ply"]).read_bytes()
        header, body = payload.split(b"end_header\n", 1)
        self.assertIn(b"element vertex 12", header)
        self.assertEqual(len(body), 12 * 15)
        records = np.frombuffer(body, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        self.assertTrue(np.isfinite(records["xyz"]).all())
        np.testing.assert_array_equal(records["rgb"], np.tile([255, 0, 0], (12, 1)))
        viewer = (self.root / result["viewer"]).read_text(encoding="utf-8")
        self.assertIn("&lt;/script&gt;", viewer)
        self.assertNotIn("__DATA__", viewer)
        self.assertNotIn("https://", viewer)
        self.assertIn("pointermove", viewer)
        self.assertIn("not a live 3D stream", viewer)
        with Image.open(self.root / result["preview"]) as picture:
            self.assertGreater(picture.width, 400)
        metadata = json.loads((self.root / result["metadata"]).read_text())
        self.assertEqual(metadata["point_count"], result["point_count"])
        with self.assertRaises(FileExistsError):
            export_point_cloud(self.root, rgb, np.ones((3, 4), dtype=np.float32), np.eye(3), details)

    def test_optional_infer_passes_the_actual_map_intrinsics_and_records_export(self):
        image = self.root / "rgb.png"
        Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(image)
        output = self.root / "inference"
        model = SimpleNamespace(checkpoint=project.BASELINE, device="cpu")
        maps = dict(inverse_depth=np.ones((2, 3), dtype=np.float32), relative_depth=np.ones((2, 3), dtype=np.float32))
        matrix = np.asarray([[2, 0, 1], [0, 2, .5], [0, 0, 1]])
        args = SimpleNamespace(image=image, output=output, point_cloud=True)
        with patch("project.selected_model", return_value=model), patch("project.timed_prediction", return_value=(maps, matrix, 1, 2)), \
                patch("src.model.sha256", return_value="fixture"), patch("src.point_cloud.export_point_cloud", return_value={"point_count": 6}) as export:
            project.infer_image(args)
        self.assertIs(export.call_args.args[2], maps["relative_depth"])
        self.assertIs(export.call_args.args[3], matrix)
        self.assertEqual(json.loads((output / "summary.json").read_text())["point_cloud"]["point_count"], 6)
        self.assertTrue(project.parser().parse_args(["demo", "--point-cloud"]).point_cloud)
        self.assertFalse(project.parser().parse_args(["infer", "--image", "fixture.png"]).point_cloud)


if __name__ == "__main__":
    unittest.main()
