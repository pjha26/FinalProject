"""Presentation launcher checks without requiring a GPU or simulator."""
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools import present


class PresentationLauncherTests(unittest.TestCase):
    def test_native_args_keep_accepted_configuration(self):
        args = present.demo_args("carla", "runs/example")
        for flag, value in [("--map", "Town04"), ("--seed", "42"), ("--weather", "ClearNoon"), ("--stop-after-seconds", "65"), ("--min-seconds", "60")]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn("--checkpoint", args)
        self.assertNotIn("--headless", args)

    def test_video_args_identify_recorded_input(self):
        args = present.demo_args("video", "runs/example")
        self.assertEqual(args[args.index("--source") + 1], "video")
        self.assertIn("data/demo/input_manifest.json", args)
        self.assertNotIn("--map", args)

    def test_busy_port_refuses_without_starting_server(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["checkpoints/baseline_mono_640x192/encoder.pth", "checkpoints/baseline_mono_640x192/depth.pth", ".runtime/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"]:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            with patch.object(present, "ROOT", root), patch("sys.argv", ["present.py"]), patch.object(present.subprocess, "run"), patch.object(present.socket, "create_connection", return_value=MagicMock()), patch.object(present.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "Port 2000 occupied"):
                    present.main()
                spawn.assert_not_called()

    def test_check_only_does_not_open_network_or_spawn(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / "checkpoints/baseline_mono_640x192"
            p.mkdir(parents=True)
            (p / "encoder.pth").touch()
            (p / "depth.pth").touch()
            with patch.object(present, "ROOT", root), patch("sys.argv", ["present.py", "--mode", "video", "--check-only"]), patch.object(present.subprocess, "run") as run, patch.object(present.socket, "create_connection") as connect, patch.object(present.subprocess, "Popen") as spawn:
                present.main()
                run.assert_called_once()
                connect.assert_not_called()
                spawn.assert_not_called()

    def test_fresh_native_run_creates_log_parent_and_cleans_up_owned_server(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["checkpoints/baseline_mono_640x192/encoder.pth", "checkpoints/baseline_mono_640x192/depth.pth", ".runtime/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.object(present, "ROOT", root), patch("sys.argv", ["present.py"]), patch.object(present.subprocess, "run") as run, patch.object(present.socket, "create_connection", side_effect=ConnectionRefusedError), patch("src.carla_capture.create_client"), patch.object(present.subprocess, "Popen") as spawn:
                spawn.return_value.poll.return_value = None
                present.main()
                self.assertEqual(len(list((root / "runs").glob("*_server.log"))), 1)
                self.assertEqual(run.call_count, 2)
                spawn.return_value.terminate.assert_called_once()
                spawn.return_value.wait.assert_called_once_with(timeout=30)

    def test_port_timeout_cannot_be_treated_as_a_free_port(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["checkpoints/baseline_mono_640x192/encoder.pth", "checkpoints/baseline_mono_640x192/depth.pth", ".runtime/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.object(present, "ROOT", root), patch("sys.argv", ["present.py"]), patch.object(present.subprocess, "run"), patch.object(present.socket, "create_connection", side_effect=socket.timeout), patch.object(present.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "Cannot confirm.*2000"):
                    present.main()
                spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()

