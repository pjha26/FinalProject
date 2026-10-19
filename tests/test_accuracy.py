"""Accuracy backend boundaries and cross-phase regression checks."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import project
from src.depth_anything import SmallModel, numerical_maps, timed_prediction, verify_assets, load_model as load_small, WEIGHTS, WEIGHTS_SHA256
from src.evaluation import score_inverse_depth
from src.profiling import prediction_fingerprint
from tools.compare_accuracy import compare, check_reports
from tools.validate_delivery import run_delivery


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        (project.ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(project.ROOT / "tmp"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_zero_floor_keeps_reciprocals_finite_without_hiding_invalid_predictions(self):
        raw = np.asarray([[0, 2]], dtype=np.float32)
        result = numerical_maps(raw)
        self.assertEqual(raw[0, 0], 0)
        self.assertAlmostEqual(float(result["relative_depth"][0, 1]), .5)
        np.testing.assert_allclose(result["inverse_depth"] * result["relative_depth"], 1)
        self.assertTrue(np.isfinite(result["relative_depth"]).all())
        for invalid in (np.asarray([1]), np.empty((0, 1)), np.asarray([[np.nan]]), np.asarray([[-1]])):
            with self.assertRaises(ValueError):
                numerical_maps(invalid)

    def test_pinned_source_is_checked_before_loading_weights(self):
        source = self.root / "source"
        source.mkdir()
        (source / "test.py").write_bytes(b"changed")
        with patch("src.depth_anything.SOURCE", source), patch("src.depth_anything.SOURCE_FILES", {"test.py": "not-the-hash"}), patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError, "changed pinned source"):
                load_small("cpu")
        load.assert_not_called()

    def test_bad_weights_are_rejected_independently_of_a_manifest(self):
        source, checkpoint = self.root / "source", self.root / "checkpoint"
        source.mkdir()
        checkpoint.mkdir()
        (source / "test.py").write_bytes(b"source")
        (checkpoint / WEIGHTS).write_bytes(b"bad weights")
        with patch("src.depth_anything.SOURCE", source), patch("src.depth_anything.CHECKPOINT", checkpoint), \
                patch("src.depth_anything.SOURCE_FILES", {"test.py": hashlib.sha256(b"source").hexdigest()}):
            with self.assertRaisesRegex(ValueError, "changed pinned Depth Anything checkpoint"):
                verify_assets()

    def test_native_rgb_preprocessing_and_explicit_cpu_device(self):
        import torch
        observed = {}
        def transform(sample):
            observed["rgb"] = sample["image"].copy()
            return {"image": sample["image"].astype(np.float32).transpose(2, 0, 1).copy()}
        def network(tensor):
            observed["device"] = tensor.device.type
            return torch.ones((1, tensor.shape[2], tensor.shape[3]), device=tensor.device)
        model = SmallModel(network, transform, torch.device("cpu"), 294)
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.synchronize") as synchronize:
            maps, intrinsics, model_ms, processing_ms = timed_prediction(model, rgb)
        self.assertEqual(observed["rgb"][0, 0].tolist(), [1, 0, 0])
        self.assertEqual(observed["device"], "cpu")
        synchronize.assert_not_called()
        self.assertEqual(maps["inverse_depth"].shape, rgb.shape[:2])
        self.assertEqual(model.input_shape, [8, 12])
        self.assertEqual(intrinsics.shape, (3, 3))
        self.assertGreaterEqual(processing_ms, model_ms)
        for bad in (np.zeros((8, 12, 3), dtype=np.float32), np.zeros((8, 12), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                timed_prediction(model, bad)
        with self.assertRaisesRegex(ValueError, "metadata"):
            timed_prediction(model, rgb, {"width": 1})
        with self.assertRaisesRegex(ValueError, "long-edge budget"):
            timed_prediction(model, np.zeros((1, 640, 3), dtype=np.uint8))

    def test_variable_shape_fingerprints_preserve_baseline_compatibility(self):
        baseline = np.ones((192, 640), dtype=np.float32)
        self.assertEqual(prediction_fingerprint(dict(inverse_depth=baseline, relative_depth=baseline)),
                         hashlib.sha256(baseline.tobytes() + baseline.tobytes()).hexdigest())
        a, b = np.ones((2, 6), dtype=np.float32), np.ones((3, 4), dtype=np.float32)
        self.assertNotEqual(prediction_fingerprint(dict(inverse_depth=a, relative_depth=a)),
                            prediction_fingerprint(dict(inverse_depth=b, relative_depth=b)))
        with self.assertRaises(ValueError):
            prediction_fingerprint(dict(inverse_depth=a, relative_depth=b))

    def test_selected_model_cannot_silently_ignore_a_custom_checkpoint(self):
        args = SimpleNamespace(model="depth-anything", checkpoint=self.root, device="cpu", input_size=294)
        with patch("src.depth_anything.load_model") as load:
            with self.assertRaisesRegex(ValueError, "separately pinned"):
                project.selected_model(args)
        load.assert_not_called()
        args.model, args.input_size = "monodepth2", 392
        with self.assertRaisesRegex(ValueError, "only to"):
            project.selected_model(args)

    def test_candidate_metadata_records_actual_weights_instead_of_reference_weights(self):
        model = SmallModel(None, None, "cpu", 294)
        details = project.model_details(model)
        self.assertEqual(details["checkpoint_sha256"], {WEIGHTS: WEIGHTS_SHA256})
        self.assertIn("dav2_rgb_imagenet", details["preprocessing_id"])

    def test_demo_gt_scoring_matches_the_paired_evaluation_protocol(self):
        inverse = np.asarray([[.2, .5], [1, 2]], dtype=np.float32)
        gt = np.arange(1, 97, dtype=np.float32).reshape(8, 12) / 10
        maps = numerical_maps(inverse)
        expected = score_inverse_depth(gt, inverse)
        record = dict(frame_id=1, rgb=np.zeros((8, 12, 3), dtype=np.uint8), camera=None, input_fps=30, depth=gt)
        output = self.root / "demo"
        args = SimpleNamespace(source="webcam", frames=1, headless=True, output=output, checkpoint=project.BASELINE, device="cpu", fps=60)
        with patch("project.video_frames", return_value=(row for row in [record])), \
                patch("src.model.load_model", return_value=SimpleNamespace(checkpoint=project.BASELINE)), \
                patch("project.timed_prediction", return_value=(maps, None, 1, 2)):
            project.demo(args)
        result = json.loads((output / "metrics.json").read_text())[0]
        for key, value in expected.items():
            self.assertEqual(result[key], value)

    def test_delivery_propagates_model_and_size_to_every_inference_step(self):
        with patch("tools.validate_delivery.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            result = run_delivery(self.root / "delivery", "cpu", model="depth-anything", input_size=392)
        self.assertEqual(result["model_requested"], "depth-anything")
        inference_steps = 0
        for call in run.call_args_list:
            command = call.args[0]
            if len(command) > 2 and command[2] in ("infer", "profile", "evaluate", "demo"):
                self.assertEqual(command[command.index("--model") + 1], "depth-anything")
                self.assertEqual(command[command.index("--input-size") + 1], "392")
                inference_steps += 1
        self.assertEqual(inference_steps, 4)

    def test_failed_model_comparison_retains_partial_evidence_and_stops(self):
        record = dict(frame_id=200, rgb=np.zeros((8, 12, 3), dtype=np.uint8))
        with patch("tools.compare_accuracy.verify_manifest", return_value={"video": "driving.mp4"}), \
                patch("tools.compare_accuracy.file_hash", return_value="fixture"), \
                patch("tools.compare_accuracy.video_frames", return_value=(row for row in [record])), \
                patch("tools.compare_accuracy.subprocess.run", return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, "Comparison failed"):
                compare(self.root / "comparison", "cpu")
        evidence = json.loads((self.root / "comparison/summary.json").read_text())
        self.assertEqual(evidence["status"], "partial")
        self.assertEqual(len(evidence["configurations"]), 1)
        self.assertEqual(run.call_count, 1)

    def test_comparison_rejects_changed_input_and_reference_identities(self):
        identity = dict(evaluation_manifest_sha256="reference", input_manifest_sha256="driving")
        evaluation = dict(status="complete", manifest_sha256="reference")
        profile = dict(status="pass", input_manifest_sha256="driving")
        check_reports(evaluation, profile, identity)
        for changed in (dict(identity, evaluation_manifest_sha256="other"), dict(identity, input_manifest_sha256="other")):
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                check_reports(evaluation, profile, changed)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            check_reports(dict(evaluation, status="partial"), profile, identity)


if __name__ == "__main__":
    unittest.main()
