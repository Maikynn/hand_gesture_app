import unittest
from unittest.mock import patch

from src.assistant import listener


class WakeWordListenerTests(unittest.TestCase):
    def test_missing_backends_fail_cleanly_when_starting(self):
        with (patch.object(listener, "VOSK_AVAILABLE", False),
              patch.object(listener, "SR_AVAILABLE", False)):
            wake = listener.WakeWordListener(stt_engine="vosk")
            with self.assertRaisesRegex(RuntimeError, "распознавания речи"):
                wake.start_listening(lambda _text: None)
            self.assertFalse(wake.is_listening)

    def test_duplicate_start_does_not_create_second_thread(self):
        wake = object.__new__(listener.WakeWordListener)
        wake.is_listening = True
        wake.thread = object()
        wake.start_listening(lambda _text: None)
        self.assertIsNotNone(wake.thread)


if __name__ == "__main__":
    unittest.main()
