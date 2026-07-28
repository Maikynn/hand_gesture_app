import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.config_store import ConfigStore


class ConfigStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.public = root / "config.json"
        self.local = root / "config.local.json"
        self.public.write_text(
            json.dumps({"assistant": {"llm_provider": "none"}, "ui": {"theme": "dark"}}),
            encoding="utf-8",
        )
        self.store = ConfigStore(self.public, self.local)

    def tearDown(self):
        self.temp.cleanup()

    def test_api_key_is_written_only_to_local_config(self):
        self.store.set("assistant.api_keys.openrouter", "secret")
        public = json.loads(self.public.read_text(encoding="utf-8"))
        local = json.loads(self.local.read_text(encoding="utf-8"))
        self.assertNotIn("api_keys", public["assistant"])
        self.assertEqual(local["assistant"]["api_keys"]["openrouter"], "secret")
        self.assertEqual(self.store.api_key("openrouter"), "secret")

    def test_sections_merge_and_replace(self):
        self.store.replace_section("permissions", [{"name": "Test", "path": "X", "enabled": True}])
        self.assertEqual(self.store.get("permissions.0", None), None)
        self.assertEqual(self.store.get("permissions")[0]["name"], "Test")
        self.assertEqual(self.store.get("ui.theme"), "dark")


if __name__ == "__main__":
    unittest.main()
