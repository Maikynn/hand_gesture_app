import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.tts_engine import (
    DEFAULT_NEURAL_VOICE,
    SpeechRequest,
    TTSEngine,
)


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

    @patch("threading.Thread.start")
    def test_repeated_neural_failure_uses_immediate_offline_fallback(self, _start):
        events = []
        tts = TTSEngine(on_event=lambda state, message: events.append((state, message)))
        request = SpeechRequest(
            text="Проверка",
            generation=0,
            engine="edge",
            voice_id=DEFAULT_NEURAL_VOICE,
            rate=175,
            volume=0.9,
            pitch=-12,
        )
        with (
            patch.object(tts, "_speak_edge", side_effect=TimeoutError("offline")) as edge,
            patch.object(tts, "_speak_offline") as offline,
        ):
            tts._speak(request)
            tts._speak(request)

        self.assertEqual(edge.call_count, 1)
        self.assertEqual(offline.call_count, 2)
        self.assertEqual(sum(state == "done" for state, _ in events), 2)
        self.assertTrue(any(state == "fallback" for state, _ in events))

    @patch("threading.Thread.start")
    def test_stop_cancels_old_generation_without_reusing_com_engine(self, _start):
        tts = TTSEngine(engine="system")
        tts.speak("Старая фраза")
        old_request = tts.speech_queue.get_nowait()
        tts.speech_queue.task_done()
        self.assertIsNotNone(old_request)

        tts.stop()
        tts.speak("Новая фраза")
        new_request = tts.speech_queue.get_nowait()
        tts.speech_queue.task_done()
        self.assertIsNotNone(new_request)
        self.assertNotEqual(old_request.generation, new_request.generation)
        self.assertTrue(tts._is_cancelled(old_request.generation))
        self.assertFalse(tts._is_cancelled(new_request.generation))

    @patch("threading.Thread.start")
    def test_piper_never_silently_becomes_windows_voice(self, _start):
        events = []
        tts = TTSEngine(
            engine="piper",
            fallback_engine="none",
            on_event=lambda state, message: events.append((state, message)),
        )
        request = SpeechRequest(
            text="Проверка",
            generation=0,
            engine="piper",
            voice_id="ru_RU-denis-medium",
            rate=175,
            volume=0.9,
            pitch=0,
            fallback_engine="none",
        )
        with (
            patch.object(tts, "_speak_piper", side_effect=RuntimeError("broken")),
            patch.object(tts, "_speak_offline") as offline,
        ):
            tts._speak(request)
        offline.assert_not_called()
        self.assertTrue(any(state == "error" for state, _ in events))


if __name__ == "__main__":
    unittest.main()
