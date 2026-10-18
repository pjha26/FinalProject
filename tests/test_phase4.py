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
from src.evaluation import METRICS, aggregate_scores, evaluation_manifest, paired_samples, score_inverse_depth
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


if __name__ == "__main__":
    unittest.main()
