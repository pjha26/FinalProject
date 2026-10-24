"""Presentation failures must be visible, inspectable and release input/display resources."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import project


class DemoReliabilityTests(unittest.TestCase):
    def setUp(self):
        (project.ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=project.ROOT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(source="video", video=self.root / "clip.mp4", frames=4, headless=True,
                                    output=self.root / "run", checkpoint=project.BASELINE, device="cpu", fps=30)
        self.args.video.touch()
        self.rgb = np.zeros((48, 64, 3), dtype=np.uint8)

    def capture(self, count, frames):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 25 if prop == cv2.CAP_PROP_FPS else count
        capture.read.side_effect = frames
        return capture

    def test_truncated_video_does_not_look_like_normal_eof_and_releases_decoder(self):
        capture = self.capture(3, [(True, self.rgb), (False, None)])
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "before its declared end"):
                list(project.video_frames(self.args))
        capture.release.assert_called_once()

    def test_normal_eof_unknown_count_and_requested_prefix_still_work(self):
        for count in (1, 0, float("nan")):
            capture = self.capture(count, [(True, self.rgb), (False, None)])
            with patch("cv2.VideoCapture", return_value=capture):
                self.assertEqual(len(list(project.video_frames(self.args))), 1)
            capture.release.assert_called_once()
        self.args.frames = 1
        capture = self.capture(100, [(True, self.rgb)])
        with patch("cv2.VideoCapture", return_value=capture):
            self.assertEqual(len(list(project.video_frames(self.args))), 1)
        capture.release.assert_called_once()

    def test_unreadable_first_frame_and_non_uint8_input_fail(self):
        for frames, error in (([(False, None)], RuntimeError), ([(True, self.rgb.astype(np.float32))], ValueError)):
            capture = self.capture(0, frames)
            with patch("cv2.VideoCapture", return_value=capture):
                with self.assertRaises(error):
                    list(project.video_frames(self.args))
            capture.release.assert_called_once()

    def test_model_load_failure_saves_diagnostic_without_completed_summary(self):
        with patch("project.selected_model", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                project.demo(self.args)
        failure = json.loads((self.args.output / "failure.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["completed_frames"], 0)
        self.assertEqual(failure["device_requested"], "cpu")
        self.assertFalse((self.args.output / "summary.json").exists())

    def test_midrun_camera_failure_keeps_completed_count_and_closes_generator(self):
        released = []
        def records():
            try:
                yield dict(frame_id=1, rgb=self.rgb, camera=None, input_fps=25)
                raise RuntimeError("Camera stopped supplying frames")
            finally:
                released.append(True)
        maps = dict(inverse_depth=np.ones((2, 3), dtype=np.float32), relative_depth=np.ones((2, 3), dtype=np.float32))
        with patch("project.selected_model", return_value=SimpleNamespace(checkpoint=project.BASELINE, device="cpu")), \
                patch("project.video_frames", return_value=records()), \
                patch("project.timed_prediction", return_value=(maps, np.eye(3), 1, 2)):
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                project.demo(self.args)
        failure = json.loads((self.args.output / "failure.json").read_text())
        self.assertEqual(failure["completed_frames"], 1)
        self.assertEqual(failure["last_completed_frame"], 1)
        self.assertTrue((self.args.output / "preview.png").exists())
        self.assertFalse((self.args.output / "summary.json").exists())
        self.assertEqual(released, [True])

    def test_failed_display_creation_still_quits_pygame(self):
        self.args.headless = False
        with patch("project.selected_model", return_value=SimpleNamespace(device="cpu")), patch("pygame.init"), \
                patch("pygame.display.set_mode", side_effect=RuntimeError("display unavailable")), patch("pygame.quit") as quit_window:
            with self.assertRaisesRegex(RuntimeError, "display unavailable"):
                project.demo(self.args)
        quit_window.assert_called_once()
        self.assertEqual(json.loads((self.args.output / "failure.json").read_text())["completed_frames"], 0)


if __name__ == "__main__":
    unittest.main()
