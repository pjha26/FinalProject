"""Numerical, synchronization, safety and real-checkpoint Phase 1 regression checks."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
from PIL import Image

import project
from src.carla_capture import CarlaCapture, CaptureSettings, FrameSynchronizer, collect_scene, replay_scene, scene_path
from src.metrics import compute_metrics
from src.model import load_model, predict
from src.preprocess import camera_intrinsics, decode_depth, decode_rgb, preprocess_rgb, resize_intrinsics, PREPROCESSING_ID
from src.visualize import render_frame

TEST_TEMP = (project.ROOT / "tmp").resolve()
assert TEST_TEMP.parent == project.ROOT.resolve()
TEST_TEMP.mkdir(exist_ok=True)


class PreprocessingTests(unittest.TestCase):
    def test_bgra_channel_order(self):
        self.assertEqual(decode_rgb(bytes([10, 20, 30, 255]), 1, 1).tolist(), [[[30, 20, 10]]])

    def test_depth_byte_significance_and_overflow(self):
        raw = bytes([0, 0, 0, 255, 1, 0, 0, 255, 255, 255, 255, 255])
        depth = decode_depth(raw, 3, 1)
        np.testing.assert_allclose(depth, [[0, 1000 * 65536 / 16777215, 1000]], rtol=1e-6)
        self.assertEqual(depth.dtype, np.float32)

    def test_invalid_sensor_bytes(self):
        with self.assertRaises(ValueError):
            decode_depth(b"short", 640, 480)

    def test_rgb_shape_range_and_no_double_normalization(self):
        import torch
        tensor, resized_k = preprocess_rgb(np.full((480, 640, 3), 128, dtype=np.uint8))
        self.assertEqual(tuple(tensor.shape), (1, 3, 192, 640))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(tensor.mean().item(), 128 / 255, places=6)
        self.assertAlmostEqual(resized_k[1, 1], 128)
        with self.assertRaises(ValueError):
            preprocess_rgb(np.zeros((480, 640, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            preprocess_rgb(np.zeros((480, 640, 3), dtype=np.float32))

    def test_camera_metadata_mismatch(self):
        with self.assertRaises(ValueError):
            preprocess_rgb(np.zeros((480, 640, 3), dtype=np.uint8), {"width": 100})

    def test_projection_resize_matches_sampling_geometry(self):
        matrix = camera_intrinsics(640, 480, 90)
        points = np.array([[0.2, 0.5, 2.0], [-0.3, -0.2, 3.0]]).T
        original = matrix @ points
        original /= original[2]
        resized = resize_intrinsics(matrix, (640, 480), (640, 192)) @ points
        resized /= resized[2]
        expected = original.copy()
        expected[1] = (original[1] + 0.5) * 0.4 - 0.5
        np.testing.assert_allclose(resized, expected)
        np.testing.assert_allclose(np.linalg.inv(matrix) @ matrix, np.eye(3))

    def test_invalid_fov(self):
        for fov in (0, 180, float("nan")):
            with self.assertRaises(ValueError):
                camera_intrinsics(640, 480, fov)


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        gt = np.array([[1, 2], [3, 4]])
        result = compute_metrics(gt, gt)
        for key in ("abs_rel", "sq_rel", "rmse", "rmse_log"):
            self.assertEqual(result[key], 0)
        for key in ("a1", "a2", "a3"):
            self.assertEqual(result[key], 1)

    def test_known_error_and_log_formula(self):
        result = compute_metrics([[2]], [[4]], median_scale=False)
        self.assertEqual(result["abs_rel"], 1)
        self.assertEqual(result["sq_rel"], 2)
        self.assertEqual(result["rmse"], 2)
        self.assertAlmostEqual(result["rmse_log"], np.log(2))

    def test_median_alignment_is_explicit(self):
        result = compute_metrics([[2, 4]], [[1, 2]])
        self.assertEqual(result["scale_ratio"], 2)
        self.assertEqual(result["abs_rel"], 0)

    def test_invalid_gt_excluded_but_bad_valid_predictions_rejected(self):
        result = compute_metrics([[0, np.nan, 80, 10]], [[np.nan, np.nan, np.nan, 10]])
        self.assertEqual(result["valid_pixels"], 1)
        for prediction in (0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                compute_metrics([[10]], [[prediction]])
        with self.assertRaises(ValueError):
            compute_metrics([[0]], [[10]])

    def test_delta_threshold_order(self):
        result = compute_metrics([[10, 10, 10]], [[12, 14, 18]], median_scale=False)
        self.assertLessEqual(result["a1"], result["a2"])
        self.assertLessEqual(result["a2"], result["a3"])
        self.assertAlmostEqual(result["a1"], 1 / 3)
        self.assertAlmostEqual(result["a2"], 2 / 3)


class SynchronizationTests(unittest.TestCase):
    def test_out_of_order_delivery_preserves_future_pair(self):
        sync = FrameSynchronizer()
        for name in ("rgb", "depth"):
            for frame in (3, 2):
                sync.put(name, SimpleNamespace(frame=frame))
        paired = sync.get_pair(2, 0)
        self.assertEqual(paired["rgb"].frame, 2)
        self.assertGreater(paired["arrival_timestamp"], 0)
        self.assertEqual(sync.get_pair(3, 0)["depth"].frame, 3)

    def test_missing_frame_times_out(self):
        sync = FrameSynchronizer()
        sync.put("rgb", SimpleNamespace(frame=1))
        with self.assertRaises(TimeoutError):
            sync.get_pair(1, 0)

    def test_overflow_never_silently_records_incomplete_data(self):
        sync = FrameSynchronizer(1)
        sync.put("rgb", SimpleNamespace(frame=1))
        sync.put("rgb", SimpleNamespace(frame=2))
        with self.assertRaises(RuntimeError):
            sync.get_pair(1, 0)

    def test_cleanup_attempts_all_actors_even_when_stop_fails(self):
        capture = CarlaCapture(CaptureSettings())
        sensor, vehicle = MagicMock(), MagicMock()
        sensor.type_id, sensor.is_listening = "sensor.camera.rgb", True
        vehicle.type_id = "vehicle.test"
        sensor.stop.side_effect = RuntimeError("stop failed")
        capture.actors = [vehicle, sensor]
        capture.world = MagicMock()
        capture.previous_settings = object()
        with self.assertRaises(RuntimeError):
            capture.__exit__(None, None, None)
        sensor.destroy.assert_called_once()
        vehicle.destroy.assert_called_once()
        capture.world.apply_settings.assert_called_once_with(capture.previous_settings)
        self.assertEqual(capture.actors, [])

    def test_failed_capture_is_marked_partial_and_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            output = Path(temporary) / "pilot"
            with patch("src.carla_capture.CarlaCapture", side_effect=RuntimeError("server absent")):
                with self.assertRaises(RuntimeError):
                    collect_scene(output, CaptureSettings(), 1)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "partial")
            self.assertEqual(metadata["captured_frames"], 0)
            with self.assertRaises(FileExistsError):
                collect_scene(output, CaptureSettings(), 1)

    def test_world_restored_even_if_traffic_manager_cleanup_fails(self):
        capture = CarlaCapture(CaptureSettings())
        capture.world = MagicMock()
        capture.previous_settings, capture.previous_weather = object(), object()
        capture.traffic_manager = MagicMock()
        capture.traffic_manager.set_synchronous_mode.side_effect = RuntimeError("TM lost")
        with self.assertRaises(RuntimeError):
            capture.__exit__(None, None, None)
        capture.world.apply_settings.assert_called_once_with(capture.previous_settings)
        capture.world.set_weather.assert_called_once_with(capture.previous_weather)


class ReplayAndSafetyTests(unittest.TestCase):
    def test_replay_round_trip_and_frame_count_validation(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            scene = Path(temporary)
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(scene / "rgb.png")
            np.save(scene / "depth.npy", np.full((3, 4), 10, dtype=np.float32), allow_pickle=False)
            metadata = {"schema_version": 1, "status": "complete", "preprocessing_id": PREPROCESSING_ID,
                        "scene_id": scene.name, "settings": {"width": 4, "height": 3},
                        "intrinsics": camera_intrinsics(4, 3).tolist(), "captured_frames": 1}
            project.write_json(scene / "metadata.json", metadata)
            with (scene / "frames.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["scene_id", "frame_id", "rgb_path", "depth_path"])
                writer.writeheader()
                writer.writerow(dict(scene_id=scene.name, frame_id=1, rgb_path="rgb.png", depth_path="depth.npy"))
            records = list(replay_scene(scene))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["rgb"].shape, (3, 4, 3))
            args = SimpleNamespace(source="replay", scene=str(scene), headless=True, frames=1,
                                   output=str(scene / "run"), checkpoint="mocked", device="cpu")
            maps = {"inverse_depth": np.full((3, 4), 0.1, dtype=np.float32),
                    "relative_depth": np.full((3, 4), 10, dtype=np.float32)}
            with patch("src.model.load_model", return_value=SimpleNamespace(checkpoint=Path("mocked"))), \
                    patch("project.timed_prediction", return_value=(maps, np.eye(3), 1.0, 2.0)):
                project.demo(args)
            summary = json.loads((scene / "run" / "summary.json").read_text())
            self.assertEqual(summary["frames"], 1)
            self.assertTrue(summary["headless"])
            self.assertTrue((scene / "run" / "runtime.csv").is_file())
            np.save(scene / "depth.npy", np.zeros((3, 4), dtype=np.float32), allow_pickle=False)
            args.output = str(scene / "unavailable_gt")
            with patch("src.model.load_model", return_value=SimpleNamespace(checkpoint=Path("mocked"))), \
                    patch("project.timed_prediction", return_value=(maps, np.eye(3), 1.0, 2.0)):
                project.demo(args)
            unavailable = json.loads((scene / "unavailable_gt" / "metrics.json").read_text())
            self.assertEqual(unavailable[0]["status"], "invalid")
            metadata["captured_frames"] = 2
            project.write_json(scene / "metadata.json", metadata)
            with self.assertRaises(ValueError):
                list(replay_scene(scene))
            with self.assertRaises(ValueError):
                scene_path(scene, "../outside.png")

    def test_archive_traversal_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            archive = Path(temporary) / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../escaped.txt", "no")
            with self.assertRaises(ValueError):
                project.safe_extract(archive, Path(temporary) / "output")
            self.assertFalse((Path(temporary) / "escaped.txt").exists())

    def test_archive_size_budget(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            archive = Path(temporary) / "large.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("file", "12345")
            with self.assertRaises(ValueError):
                project.safe_extract(archive, Path(temporary) / "output", max_bytes=4)

    def test_colorization_is_fixed_and_numerical_input_unchanged(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        inverse = np.full((192, 640), 0.1, dtype=np.float32)
        original = inverse.copy()
        image = render_frame(rgb, inverse)
        self.assertEqual(image.shape, (524, 1920, 3))
        np.testing.assert_array_equal(inverse, original)
        np.testing.assert_array_equal(image, render_frame(rgb, inverse))


class ModelTests(unittest.TestCase):
    def test_checkpoint_hash_mismatch_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            checkpoint = Path(temporary)
            for name in ("encoder.pth", "depth.pth"):
                (checkpoint / name).write_bytes(b"not a checkpoint")
            project.write_json(checkpoint / "manifest.json", {"preprocessing_id": PREPROCESSING_ID,
                               "sha256": {"encoder.pth": "bad", "depth.pth": "bad"}})
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_model(checkpoint, "cpu")

    def test_invalid_model_input_rejected(self):
        import torch
        for tensor in (torch.zeros(1, 3, 480, 640), torch.full((1, 3, 192, 640), 2.0), torch.zeros(1, 3, 192, 640, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                predict(None, tensor)

    @unittest.skipUnless((project.BASELINE / "manifest.json").is_file(), "Run project.py assets for real-checkpoint integration")
    def test_real_checkpoint_finite_and_repeatable(self):
        import torch
        model = load_model(project.BASELINE)
        rgb = np.asarray(Image.open(project.ROOT / "data" / "fixtures" / "test_image.jpg").convert("RGB"))
        tensor, _ = preprocess_rgb(rgb)
        first, second = predict(model, tensor), predict(model, tensor)
        self.assertEqual(tuple(first["inverse_depth"].shape), (1, 1, 192, 640))
        self.assertTrue(torch.isfinite(first["relative_depth"]).all().item())
        self.assertTrue(torch.allclose(first["inverse_depth"], second["inverse_depth"], rtol=1e-4, atol=1e-6))
        self.assertTrue(torch.allclose(first["relative_depth"], first["inverse_depth"].reciprocal()))


if __name__ == "__main__":
    unittest.main()
