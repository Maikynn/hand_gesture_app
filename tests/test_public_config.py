import json
import unittest
from pathlib import Path


class PublicConfigTests(unittest.TestCase):
    def test_tracked_config_has_no_api_keys(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "config.json").read_text(encoding="utf-8"))
        assistant = data.get("assistant", {})
        self.assertNotIn("api_keys", assistant)
        serialized = json.dumps(data).lower()
        self.assertNotIn("sk-or-", serialized)


if __name__ == "__main__":
    unittest.main()
