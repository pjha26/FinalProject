"""CARLA map caching must remain in the writable project workspace."""
import unittest
from unittest.mock import MagicMock, call, patch

import project
from src.carla_capture import CaptureSettings, create_client


class CarlaCacheTests(unittest.TestCase):
    def test_cache_is_configured_before_rpc_and_timeout_preserves_caller_choice(self):
        module = MagicMock()
        client = module.Client.return_value
        client.set_files_base_folder.return_value = True
        with patch.dict('sys.modules', {'carla': module}):
            result = create_client('127.0.0.1', 2010, 7)
        self.assertIs(result, client)
        module.Client.assert_called_once_with('127.0.0.1', 2010)
        expected = project.ROOT / '.cache/carla'
        self.assertEqual(client.mock_calls, [call.set_files_base_folder(str(expected)), call.set_timeout(7)])
        client.get_world.assert_not_called()

    def test_refused_project_cache_fails_without_falling_back_to_home(self):
        module = MagicMock()
        client = module.Client.return_value
        client.set_files_base_folder.return_value = False
        with patch.dict('sys.modules', {'carla': module}):
            with self.assertRaisesRegex(RuntimeError, 'project-local'):
                create_client('127.0.0.1', 2000, 3)
        client.set_timeout.assert_not_called()
        client.get_world.assert_not_called()


    def test_capture_commands_preserve_default_and_honor_explicit_timeout(self):
        for command, extra in (("collect", ["--output", "unused"]), ("demo", []), ("calibrate-depth", ["--output", "unused"])):
            args = project.parser().parse_args([command] + extra)
            self.assertEqual(project.capture_settings(args).timeout, 15)
            args = project.parser().parse_args([command] + extra + ["--timeout", "60"])
            self.assertEqual(project.capture_settings(args).timeout, 60)

    def test_nonfinite_and_nonpositive_timeouts_are_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                CaptureSettings(timeout=value)


if __name__ == '__main__':
    unittest.main()
