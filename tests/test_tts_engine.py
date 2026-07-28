import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.tts_engine import DEFAULT_NEURAL_VOICE, TTSEngine


class TTSEngineTests(unittest.TestCase):
    @patch("threading.Thread.start")
    def test_settings_are_clamped_and_invalid_engine_is_safe(self, _start):
        tts = TTSEngine(engine="unknown", rate=999, volume=-1, pitch=-99)
        self.assertEqual(tts.engine_name, "edge")
        self.assertEqual(tts.voice_id, DEFAULT_NEURAL_VOICE)
        self.assertEqual(tts.rate, 260)
        self.assertEqual(tts.volume, 0.0)
        self.assertEqual(tts.pitch, -50)

        tts.configure(
            engine="system",
            voice_id="voice",
            rate=1,
            volume=2,
            pitch=99,
        )
        self.assertEqual(tts.engine_name, "system")
        self.assertEqual(tts.rate, 90)
        self.assertEqual(tts.volume, 1.0)
        self.assertEqual(tts.pitch, 50)


if __name__ == "__main__":
    unittest.main()
