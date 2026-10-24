"""Phase 4 protocol, integrity, aggregation and acceptance-gate regression checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import project
from src.evaluation import METRICS, aggregate_scores, evaluation_manifest, paired_samples, score_inverse_depth, runtime_evidence, save_review_snapshot
from src.preprocess import PREPROCESSING_ID

TEST_TEMP = (project.ROOT / "tmp").resolve()
assert TEST_TEMP.parent == project.ROOT.resolve()
TEST_TEMP.mkdir(exist_ok=True)


class EvaluationTests(unittest.TestCase):
    def fixture(self, root):
        rgb, depth = root / "rgb.png", root / "depth.npy"
        Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(rgb)
        np.save(depth, np.ones((8, 12), dtype=np.float32), allow_pickle=False)
        manifest = dict(schema_version=1, dataset="unit fixture", source="generated test data", limitations="synthetic correctness test only",
                        data_kind="synthetic_test_fixture", split="diagnostic", preprocessing_id=PREPROCESSING_ID,
                        checkpoint_sha256={"encoder.pth": "fixture"},
                        protocol=dict(depth_units="metres", aligned_to_rgb=True, min_depth=0.001, max_depth=80,
                                      crop="none", scaling="per_image_median", aggregation="mean_of_images"),
                        samples=[dict(id="fixture1", scene="fixture_scene", rgb="rgb.png", depth="depth.npy",
                                      rgb_sha256=project.file_hash(rgb), depth_sha256=project.file_hash(depth))])
        path = root / "evaluation_manifest.json"
        project.write_json(path, manifest)
        return path, manifest

    def test_inverse_is_resized_before_inversion_and_scaling(self):
        inverse = np.asarray([[1, 3], [2, 4]], dtype=np.float32)
        gt = 5 / cv2.resize(inverse, (8, 6), interpolation=cv2.INTER_LINEAR)
        score = score_inverse_depth(gt, inverse)
        self.assertAlmostEqual(score["abs_rel"], 0, places=6)
        self.assertAlmostEqual(score["scale_ratio"], 5, places=5)
        self.assertEqual(score["valid_pixels"], 48)
        self.assertEqual(score["a1"], 1)

    def test_crop_and_invalid_gt_mask_are_applied_before_median_scaling(self):
        gt = np.ones((10, 10), dtype=np.float32)
        gt[:4] = 70
        gt[9:] = 0
        score = score_inverse_depth(gt, np.ones((2, 2), dtype=np.float32), "garg")
        self.assertEqual(score["valid_pixels"], 45)
        self.assertEqual(score["abs_rel"], 0)
        self.assertEqual(score["scale_ratio"], 1)
        with self.assertRaises(ValueError):
            score_inverse_depth(gt, np.full((2, 2), np.nan))

    def test_aggregation_weights_images_equally_and_never_invents_empty_accuracy(self):
        rows = [dict.fromkeys(METRICS, value) for value in (1, 3)]
        rows[0]["valid_pixels"], rows[1]["valid_pixels"] = 1000, 1
        self.assertEqual(aggregate_scores(rows)["rmse"], 2)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            aggregate_scores([])

    def test_manifest_rejects_changed_paths_duplicates_and_wrong_units(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            path, manifest = self.fixture(root)
            self.assertEqual(len(list(paired_samples(path, evaluation_manifest(path)))), 1)
            manifest["samples"] *= 2
            project.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                evaluation_manifest(path)
            manifest["samples"] = manifest["samples"][:1]
            manifest["samples"][0]["depth"] = "../outside.npy"
            project.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "outside dataset"):
                evaluation_manifest(path)
            manifest["protocol"]["depth_units"] = "millimetres"
            project.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "aligned-metre"):
                evaluation_manifest(path)

    def test_invalid_paired_depth_cannot_be_accepted_as_aligned_gt(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            path, manifest = self.fixture(root)
            np.save(root / "depth.npy", np.zeros((4, 6), dtype=np.float32), allow_pickle=False)
            manifest["samples"][0]["depth_sha256"] = project.file_hash(root / "depth.npy")
            project.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "aligned to RGB"):
                list(paired_samples(path, evaluation_manifest(path)))

    def test_evaluator_uses_only_rgb_for_prediction_and_records_partial_failures(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            path, manifest = self.fixture(root)
            args = SimpleNamespace(manifest=path, checkpoint=project.BASELINE, device="cpu", output=root / "evaluation")
            model = SimpleNamespace(device="cpu")
            maps = dict(inverse_depth=np.ones((192, 640), dtype=np.float32), relative_depth=np.ones((192, 640), dtype=np.float32))
            def prediction(model_argument, rgb):
                self.assertIs(model_argument, model)
                self.assertEqual(rgb.shape, (8, 12, 3))
                return maps, None, 1, 2
            with patch("src.demo_data.checkpoint_hashes", return_value=manifest["checkpoint_sha256"]), patch("src.model.load_model", return_value=model), patch("project.timed_prediction", side_effect=prediction):
                project.evaluate_model(args)
            result = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["data_kind"], "synthetic_test_fixture")
            self.assertEqual(result["mean_metrics"]["abs_rel"], 0)
            args.output = root / "failure"
            with patch("src.demo_data.checkpoint_hashes", return_value=manifest["checkpoint_sha256"]), patch("src.model.load_model", return_value=model), patch("project.timed_prediction", side_effect=RuntimeError("inference failed")):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    project.evaluate_model(args)
            failed = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(failed["status"], "partial")
            self.assertIsNone(failed["mean_metrics"])


class RuntimeEvidenceTests(unittest.TestCase):
    def test_display_gate_requires_duration_and_a_real_display(self):
        rows = [dict(model_ms=1, processing_ms=2, capture_to_display_ms=3, full_cycle_ms=4)]
        self.assertEqual(runtime_evidence(rows, 61, False, 60)["runtime_acceptance"]["status"], "pass")
        self.assertEqual(runtime_evidence(rows, 59, False, 60)["runtime_acceptance"]["status"], "needs_review")
        self.assertEqual(runtime_evidence(rows, 61, True, 60)["runtime_acceptance"]["status"], "needs_review")
        self.assertEqual(runtime_evidence(rows, 1, True)["runtime_acceptance"]["status"], "not_requested")
        with self.assertRaises(ValueError):
            runtime_evidence(rows, float("nan"), False, 60)

    def test_review_preserves_numerical_maps_and_avoids_duplicate_exports(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            output = Path(temporary)
            rgb = np.zeros((48, 64, 3), dtype=np.uint8)
            maps = dict(inverse_depth=np.ones((192, 640), dtype=np.float32), relative_depth=np.ones((192, 640), dtype=np.float32))
            saved = save_review_snapshot(output, 20, rgb, maps, rgb)
            np.testing.assert_array_equal(np.load(output / saved["inverse_depth"], allow_pickle=False), maps["inverse_depth"])
            self.assertTrue((output / saved["preview"]).is_file())
            self.assertIsNone(save_review_snapshot(output, 20, rgb, maps, rgb))
            with self.assertRaises(ValueError):
                save_review_snapshot(output, "../escape", rgb, maps, rgb)

    def test_headless_demo_cannot_claim_display_acceptance(self):
        args = SimpleNamespace(source="video", headless=True, min_seconds=60, fps=10)
        with patch("src.model.load_model") as model:
            with self.assertRaisesRegex(ValueError, "cannot run with --headless"):
                project.demo(args)
            model.assert_not_called()

    def test_duration_stop_is_bounded_and_invalid_duration_fails_before_loading(self):
        import itertools
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            args=SimpleNamespace(source='video',headless=True,min_seconds=0,fps=30,frames=100,output=Path(temporary)/'duration',checkpoint=project.BASELINE,device='cpu',stop_after_seconds=.01)
            record=dict(frame_id=1,rgb=np.zeros((48,64,3),dtype=np.uint8),camera=None,input_fps=25)
            maps=dict(inverse_depth=np.ones((192,640),dtype=np.float32),relative_depth=np.ones((192,640),dtype=np.float32))
            with patch('src.model.load_model',return_value=SimpleNamespace(checkpoint=project.BASELINE)),patch('project.video_frames',return_value=(dict(record,frame_id=i) for i in range(100))),patch('project.timed_prediction',return_value=(maps,None,1,2)),patch('project.time.perf_counter',side_effect=itertools.count(1,.01)),patch('project.time.sleep'):
                project.demo(args)
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertGreater(summary['frames'],0);self.assertLess(summary['frames'],100)
            self.assertEqual(summary['stop_after_seconds'],.01)
            args.stop_after_seconds=float('nan')
            with patch('src.model.load_model') as model:
                with self.assertRaises(ValueError):project.demo(args)
                model.assert_not_called()

    def test_short_display_run_saves_failed_gate_and_releases_window(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            output = Path(temporary) / "short_run"
            args = SimpleNamespace(source="video", headless=False, min_seconds=60, fps=10,
                                   frames=1, output=output, checkpoint=project.BASELINE, device="cpu")
            record = dict(frame_id=1, rgb=np.zeros((48, 64, 3), dtype=np.uint8), camera=None, input_fps=25)
            maps = dict(inverse_depth=np.ones((192, 640), dtype=np.float32), relative_depth=np.ones((192, 640), dtype=np.float32))
            with patch("src.model.load_model", return_value=SimpleNamespace(checkpoint=project.BASELINE)), patch("project.video_frames", return_value=(value for value in [record])), patch("project.timed_prediction", return_value=(maps, None, 1, 2)), patch("pygame.init"), patch("pygame.display.set_mode"), patch("pygame.display.set_caption"), patch("pygame.event.get", return_value=[]), patch("pygame.surfarray.make_surface"), patch("pygame.transform.smoothscale"), patch("pygame.display.flip"), patch("pygame.quit") as quit_window:
                with self.assertRaisesRegex(RuntimeError, "did not meet"):
                    project.demo(args)
            quit_window.assert_called_once()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["runtime_acceptance"]["status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
