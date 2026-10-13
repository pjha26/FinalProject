"""Phase 2 portable input integrity and decoding checks; no network needed."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import project
from src import demo_data
from src.preprocess import PREPROCESSING_ID

TEST_TEMP = (project.ROOT / "tmp").resolve()
assert TEST_TEMP.parent == project.ROOT.resolve()
TEST_TEMP.mkdir(exist_ok=True)


class DemoPreparationTests(unittest.TestCase):
    def fixture(self, root):
        video = root / "fixture.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(3):
                writer.write(np.full((48, 64, 3), [0, 0, 240], dtype=np.uint8))
        finally:
            writer.release()
        license_file = root / "fixture_license.txt"
        license_file.write_text("Synthetic test fixture only", encoding="utf-8")
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        for name in ("encoder.pth", "depth.pth"):
            (checkpoint / name).write_bytes(b"hash fixture; never deserialized")
        hashes = {name: project.file_hash(checkpoint / name) for name in ("encoder.pth", "depth.pth")}
        project.write_json(checkpoint / "manifest.json", dict(preprocessing_id=PREPROCESSING_ID,
                           upstream_revision=demo_data.UPSTREAM_REVISION, sha256=hashes))
        files = {"driving.mp4": ("project_video.mp4", project.file_hash(video)),
                 "UDACITY_LICENSE.txt": ("LICENSE", project.file_hash(license_file))}
        def download(url, destination):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video if url.endswith("project_video.mp4") else license_file, destination)
            return destination
        return checkpoint, files, download

    def test_preparation_is_portable_and_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            checkpoint, files, download = self.fixture(root)
            args = SimpleNamespace(output=root / "demo", checkpoint=checkpoint)
            with patch("src.demo_data.SAMPLE_FILES", files), patch("project.download", side_effect=download) as fetch:
                project.prepare_demo(args)
                manifest_path = args.output / "input_manifest.json"
                original = manifest_path.read_bytes()
                manifest = demo_data.verify_manifest(manifest_path, checkpoint)
                self.assertEqual(manifest["decoded_frames"], 3)
                self.assertEqual(manifest["ground_truth"], "unavailable")
                self.assertEqual(manifest["video"], "driving.mp4")
                self.assertGreater(cv2.imread(str(args.output / "driving_first_frame.png"))[0, 0, 2], 230)
                project.prepare_demo(args)
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(manifest_path.read_bytes(), original)
                moved = root / "other_laptop" / "demo"
                shutil.copytree(args.output, moved)
                self.assertEqual(demo_data.verify_manifest(moved / "input_manifest.json", checkpoint), manifest)

    def test_changed_inputs_metadata_and_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            checkpoint, files, download = self.fixture(root)
            args = SimpleNamespace(output=root / "demo", checkpoint=checkpoint)
            with patch("src.demo_data.SAMPLE_FILES", files), patch("project.download", side_effect=download):
                project.prepare_demo(args)
                manifest_path = args.output / "input_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["decoded_frames"] += 1
                project.write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    demo_data.verify_manifest(manifest_path, checkpoint)
                manifest["decoded_frames"] -= 1
                manifest["files"]["../outside.mp4"] = manifest["files"].pop("driving.mp4")
                project.write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "asset paths"):
                    demo_data.verify_manifest(manifest_path, checkpoint)
                manifest["files"]["driving.mp4"] = manifest["files"].pop("../outside.mp4")
                project.write_json(manifest_path, manifest)
                (args.output / "UDACITY_LICENSE.txt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    demo_data.verify_manifest(manifest_path, checkpoint)
                download("LICENSE", args.output / "UDACITY_LICENSE.txt")
                (checkpoint / "encoder.pth").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Checkpoint checksum"):
                    demo_data.verify_manifest(manifest_path, checkpoint)

    def test_cached_bad_download_cannot_create_an_accepted_manifest(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            checkpoint, files, download = self.fixture(root)
            files["driving.mp4"] = ("project_video.mp4", "bad")
            args = SimpleNamespace(output=root / "demo", checkpoint=checkpoint)
            with patch("src.demo_data.SAMPLE_FILES", files), patch("project.download", side_effect=download):
                with self.assertRaisesRegex(ValueError, "Pinned demo checksum"):
                    project.prepare_demo(args)
            self.assertFalse((args.output / "input_manifest.json").exists())

    def test_malformed_checkpoint_manifest_is_a_clear_validation_error(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            for manifest in ([], dict(preprocessing_id=PREPROCESSING_ID, upstream_revision=demo_data.UPSTREAM_REVISION, sha256=None)):
                project.write_json(root / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    demo_data.checkpoint_hashes(root)

    def test_incomplete_decode_and_variable_resolution_release_capture(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = [10, 3]
        capture.read.side_effect = [(True, np.zeros((48, 64, 3), dtype=np.uint8)), (False, None)]
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "declared frame count"):
                demo_data.inspect_video("fixture")
        capture.release.assert_called_once()
        capture.reset_mock()
        capture.get.side_effect = [10, 2]
        capture.read.side_effect = [(True, np.zeros((48, 64, 3), dtype=np.uint8)), (True, np.zeros((24, 32, 3), dtype=np.uint8))]
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "dimensions changed"):
                demo_data.inspect_video("fixture")
        capture.release.assert_called_once()

    def test_manifest_cannot_verify_a_different_demo_video(self):
        with patch("src.demo_data.verify_manifest", return_value={"video": "driving.mp4"}), patch("src.model.load_model") as model:
            args = SimpleNamespace(source="video", fps=10, manifest="data/demo/input_manifest.json", video="other.mp4", checkpoint=project.BASELINE)
            with self.assertRaisesRegex(ValueError, "does not match"):
                project.demo(args)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
