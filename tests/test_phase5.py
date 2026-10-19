"""Delivery orchestration checks; actual model runs are separate acceptance evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.validate_delivery import ROOT, run_delivery


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(ROOT / "tmp"))
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "delivery"

    def test_all_phases_use_same_interpreter_and_checkpoint_defaults(self):
        with patch("tools.validate_delivery.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            result = run_delivery(self.output, "cuda")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["displayed_rehearsal"], "pending")
        self.assertEqual([row["name"] for row in result["steps"]],
                         ["tests", "inputs", "environment", "image", "profile", "evaluation", "video"])
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], result["executable"])
            self.assertEqual(call.kwargs["cwd"], str(ROOT))
        self.assertIn("--headless", run.call_args_list[-1].args[0])
        self.assertIn("project.py", result["source_sha256"])
        with self.assertRaises(FileExistsError):
            run_delivery(self.output)

    def test_failure_stops_later_phases_and_retains_logs_and_summary(self):
        with patch("tools.validate_delivery.subprocess.run", return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, "tests failed"):
                run_delivery(self.output)
        evidence = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["steps"][-1]["status"], "failed")
        self.assertTrue((self.output / "tests.log").is_file())
        self.assertEqual(run.call_count, 1)

    def test_process_launch_error_is_recorded(self):
        with patch("tools.validate_delivery.subprocess.run", side_effect=OSError("launch failed")):
            with self.assertRaisesRegex(OSError, "launch failed"):
                run_delivery(self.output)
        evidence = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["steps"][0]["status"], "failed")

    def test_display_requires_saved_gate_and_never_silently_passes(self):
        def completed(command, **kwargs):
            if "--min-seconds" in command:
                target = Path(command[command.index("--output") + 1])
                target.mkdir()
                (target / "summary.json").write_text(json.dumps({"runtime_acceptance": {"status": "needs_review"}}))
            return SimpleNamespace(returncode=0)
        with patch("tools.validate_delivery.subprocess.run", side_effect=completed):
            with self.assertRaisesRegex(RuntimeError, "duration gate"):
                run_delivery(self.output, display=True)
        evidence = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["displayed_rehearsal"], "pending")

    def test_changed_source_cannot_pass_delivery(self):
        with patch("tools.validate_delivery.subprocess.run", return_value=SimpleNamespace(returncode=0)), \
                patch("tools.validate_delivery.source_hashes", side_effect=[{"project.py": "before"}, {"project.py": "after"}]):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                run_delivery(self.output)
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["status"], "failed")

    def test_display_success_records_acceptance(self):
        def completed(command, **kwargs):
            if "--min-seconds" in command:
                target = Path(command[command.index("--output") + 1])
                target.mkdir()
                (target / "summary.json").write_text(json.dumps({"runtime_acceptance": {"status": "pass"}}))
            return SimpleNamespace(returncode=0)
        with patch("tools.validate_delivery.subprocess.run", side_effect=completed):
            result = run_delivery(self.output, display=True)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["displayed_rehearsal"], "pass")
        self.assertEqual(result["steps"][-1]["name"], "display")


if __name__ == "__main__":
    unittest.main()
