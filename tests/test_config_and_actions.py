import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.assistant.action_executor import ActionExecutor
from src.utils import config_store
from src.utils.config_store import DEFAULT_CONFIG, _merge


class ConfigStoreTests(unittest.TestCase):
    def test_nested_saved_values_preserve_new_defaults(self):
        config = _merge(DEFAULT_CONFIG, {"camera": {"device_index": 3}})
        self.assertEqual(config["camera"]["device_index"], 3)
        self.assertIn("bindings", config["gesture"])
        self.assertIn("error_phrases", config["voice"])

    def test_default_config_contains_no_api_secret(self):
        self.assertEqual(DEFAULT_CONFIG["voice"]["openrouter_key"], "")
        self.assertEqual(DEFAULT_CONFIG["voice"]["custom_api_key"], "")

    def test_save_uses_local_file_and_is_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "config.json"
            local = Path(directory) / "config.local.json"
            template.write_text('{"camera": {"device_index": 1}}', encoding="utf-8")
            with (
                patch.object(config_store, "CONFIG_PATH", template),
                patch.object(config_store, "LOCAL_CONFIG_PATH", local),
            ):
                saved = config_store.load_config()
                saved["camera"]["device_index"] = 4
                config_store.save_config(saved)
                self.assertEqual(
                    config_store.load_config()["camera"]["device_index"], 4
                )
                self.assertIn('"device_index": 1', template.read_text(encoding="utf-8"))


class ActionExecutorTests(unittest.TestCase):
    def test_unknown_application_is_denied(self):
        ok, message = ActionExecutor([]).execute("open_app", "calculator")
        self.assertFalse(ok)
        self.assertIn("не разрешено", message)

    def test_missing_allowed_application_is_not_started(self):
        missing = str(Path(tempfile.gettempdir()) / "axi-missing-program.exe")
        ok, message = ActionExecutor([{"name": "test", "path": missing}]).execute(
            "open_app", "test"
        )
        self.assertFalse(ok)
        self.assertIn("не найден", message)


if __name__ == "__main__":
    unittest.main()
