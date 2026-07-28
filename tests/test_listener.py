import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.listener import WakeWordListener


class WakeWordListenerTests(unittest.TestCase):
    def setUp(self):
        self.heard = []
        self.listener = WakeWordListener(wake_word="джарвис, аксиос")
        self.listener.callback = self.heard.append

    def test_command_after_wake_phrase_is_forwarded(self):
        result = self.listener.process_recognized_text("джарвис открой блокнот")
        self.assertEqual(result, "открой блокнот")
        self.assertEqual(self.heard, ["открой блокнот"])

    def test_wake_phrase_arms_next_utterance(self):
        self.assertEqual(self.listener.process_recognized_text("аксиос"), "")
        self.assertEqual(self.heard, [])
        self.listener.process_recognized_text("сделай громче")
        self.assertEqual(self.heard, ["сделай громче"])

    def test_unrelated_phrase_is_ignored(self):
        self.assertIsNone(self.listener.process_recognized_text("обычный разговор"))
        self.assertEqual(self.heard, [])

    def test_continuous_mode_forwards_everything(self):
        self.listener.continuous_mode = True
        self.listener.process_recognized_text("обычный разговор")
        self.assertEqual(self.heard, ["обычный разговор"])


if __name__ == "__main__":
    unittest.main()
