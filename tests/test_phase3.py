"""Phase 3 timing, warmup exclusion, and repeatability regression checks."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import project
from src.profiling import TIMING_FIELDS, prediction_fingerprint, summarize_timings

TEST_TEMP = (project.ROOT / "tmp").resolve()
assert TEST_TEMP.parent == project.ROOT.resolve()
TEST_TEMP.mkdir(exist_ok=True)


def maps(inverse=1):
    return dict(inverse_depth=np.full((192, 640), inverse, dtype=np.float32),
                relative_depth=np.full((192, 640), 1 / inverse, dtype=np.float32))


class ProfilingTests(unittest.TestCase):
    def test_display_palette_is_preserved_without_changing_numerical_depth(self):
        from matplotlib import colormaps
        from src.visualize import render_frame
        rgb = np.full((1, 640, 3), 100, dtype=np.uint8)
        inverse = np.linspace(0.01, 10, 640, dtype=np.float32)[None, :]
        original = inverse.copy()
        result = render_frame(rgb, inverse)
        normalized = np.clip((inverse - 0.01) / (10 - 0.01), 0, 1)
        expected_colors = (colormaps["plasma"](normalized)[:, :, :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(result[44:, 640:1280], expected_colors)
        expected_overlay = (rgb * 0.6 + expected_colors * 0.4).astype(np.uint8)
        np.testing.assert_allclose(result[44:, 1280:], expected_overlay, atol=1)
        np.testing.assert_array_equal(inverse, original)

    def test_timing_percentiles_and_invalid_samples(self):
        rows = [dict.fromkeys(TIMING_FIELDS, value) for value in (1, 3, 5)]
        summary = summarize_timings(rows)["model_ms"]
        self.assertEqual(summary["mean"], 3)
        self.assertEqual(summary["median"], 3)
        self.assertAlmostEqual(summary["p95"], 4.8)
        for invalid in ([], [dict.fromkeys(TIMING_FIELDS, -1)], [dict.fromkeys(TIMING_FIELDS, float("nan"))]):
            with self.assertRaises(ValueError):
                summarize_timings(invalid)

    def test_fingerprints_check_numerical_contract_and_detect_changes(self):
        self.assertEqual(prediction_fingerprint(maps()), prediction_fingerprint(maps()))
        self.assertNotEqual(prediction_fingerprint(maps()), prediction_fingerprint(maps(2)))
        for value in (np.nan, -1, 0):
            invalid = maps()
            invalid["inverse_depth"][0, 0] = value
            with self.assertRaises(ValueError):
                prediction_fingerprint(invalid)
        invalid = maps()
        invalid["relative_depth"] *= 2
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            prediction_fingerprint(invalid)

    def run_profile(self, root, change=False, short=False):
        import torch
        manifest = root / "input_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        args = SimpleNamespace(frames=2, warmup=1, repeats=2, manifest=manifest,
                               checkpoint=project.BASELINE, device="cpu", output=root / "profile")
        model = SimpleNamespace(device=torch.device("cpu"), encoder=SimpleNamespace(training=False), decoder=SimpleNamespace(training=False))
        metadata = dict(decoded_frames=3, video="driving.mp4", source_revision="fixture", checkpoint_sha256={"encoder.pth": "fixture"})
        def records(_):
            for frame in range(1, 2 if short else 4):
                yield dict(frame_id=frame, rgb=np.zeros((48, 64, 3), dtype=np.uint8))
        calls = [0]
        def prediction(*_):
            calls[0] += 1
            return maps(2 if change and calls[0] > 3 else 1), None, 1, 2
        with patch("src.demo_data.verify_manifest", return_value=metadata), patch("src.model.load_model", return_value=model), patch("project.video_frames", side_effect=records), patch("project.timed_prediction", side_effect=prediction), patch("src.profiling.process_memory_mib", return_value=None), patch("src.visualize.render_frame", return_value=np.zeros((48, 192, 3), dtype=np.uint8)) as render:
            project.profile_model(args)
        return args.output, calls[0], render.call_count

    def test_warmup_is_excluded_and_matching_repeats_are_recorded(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            output, predictions, renders = self.run_profile(Path(temporary))
            self.assertEqual(predictions, 6)
            self.assertEqual(renders, 6)
            with (output / "runtime.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual([int(row["frame_id"]) for row in rows], [2, 3, 2, 3])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "pass")
            self.assertTrue(summary["exact_prediction_repeatability"])
            self.assertEqual(summary["ground_truth_accuracy"], "unavailable")
            self.assertTrue((output / "preview.png").is_file())

    def test_unstable_predictions_cannot_pass_the_phase_gate(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Repeated predictions differ"):
                self.run_profile(root, change=True)
            summary = json.loads((root / "profile" / "summary.json").read_text())
            self.assertEqual(summary["status"], "needs_review")
            self.assertFalse(summary["exact_prediction_repeatability"])

    def test_short_input_and_invalid_profile_settings_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            with self.assertRaisesRegex(ValueError, "Video ended"):
                self.run_profile(Path(temporary), short=True)
        for frames, warmup, repeats in ((0, 1, 2), (301, 1, 2), (2, -1, 2), (2, 1, 1)):
            with self.assertRaisesRegex(ValueError, "Profile requires"):
                project.profile_model(SimpleNamespace(frames=frames, warmup=warmup, repeats=repeats))


if __name__ == "__main__":
    unittest.main()
