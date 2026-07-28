import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.document_index import DocumentIndex
from assistant.voice_auth import VoiceAuthenticator, voice_signature
from utils.active_window import match_profile, parse_profile_rules


class AssistantIntelligenceTests(unittest.TestCase):
    def test_local_document_index_returns_relevant_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manual.md").write_text(
                "Калибровка камеры выполняется при открытой ладони.",
                encoding="utf-8",
            )
            index = DocumentIndex(str(root), root / "index.json")
            self.assertGreater(index.rebuild(), 0)
            context = index.prompt_context("Как выполняется калибровка камеры?")
            self.assertIn("открытой ладони", context)
            self.assertIn("недоверенные", context)

    def test_voice_signature_is_stable_for_same_speaker_sample(self):
        rate = 16000
        time_axis = np.arange(rate * 2) / rate
        sample = (
            np.sin(2 * np.pi * 130 * time_axis)
            + 0.4 * np.sin(2 * np.pi * 260 * time_axis)
        )
        sample = (sample * 12000).astype(np.int16)
        first = voice_signature(sample)
        second = voice_signature(sample.copy())
        self.assertGreater(float(np.dot(first, second)), 0.99)
        with tempfile.TemporaryDirectory() as directory:
            auth = VoiceAuthenticator(Path(directory) / "voice.json")
            auth.enroll(sample)
            self.assertGreater(auth.score(sample), 0.99)

    def test_active_application_profile_rules(self):
        rules = parse_profile_rules(
            "powerpnt=presentation;spotify=music;steam=games"
        )
        self.assertEqual(match_profile("POWERPNT.EXE", rules), "presentation")
        self.assertEqual(match_profile("unknown.exe", rules), "")


if __name__ == "__main__":
    unittest.main()
