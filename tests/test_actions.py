import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.actions import ActionError, ActionExecutor


class ActionExecutorTests(unittest.TestCase):
    def test_application_requires_enabled_absolute_allowlist_path(self):
        launched = []
        executor = ActionExecutor(
            [{"name": "Python", "path": sys.executable, "enabled": True}],
            app_launcher=launched.append,
        )
        message = executor.execute(
            {"type": "application", "target": sys.executable}
        )
        self.assertEqual(launched, [sys.executable])
        self.assertIn("Python", message)

    def test_application_outside_allowlist_is_denied(self):
        executor = ActionExecutor([])
        with self.assertRaises(ActionError):
            executor.validate({"type": "application", "target": sys.executable})

    def test_system_action_uses_injected_hotkey_sender(self):
        sent = []
        executor = ActionExecutor(hotkey_sender=sent.append)
        executor.execute({"type": "system", "target": "show_desktop"})
        self.assertEqual(sent, [["win", "d"]])

    def test_shell_commands_are_not_supported(self):
        executor = ActionExecutor()
        with self.assertRaises(ActionError):
            executor.validate({"type": "command", "target": "echo unsafe"})


if __name__ == "__main__":
    unittest.main()
