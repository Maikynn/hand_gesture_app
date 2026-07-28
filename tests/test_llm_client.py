import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.llm_client import LLMClient, repair_mojibake


class LLMEncodingTests(unittest.TestCase):
    def test_repairs_utf8_read_as_latin1(self):
        broken = "Привет! Как я могу помочь?".encode("utf-8").decode("latin-1")
        self.assertEqual(repair_mojibake(broken), "Привет! Как я могу помочь?")

    def test_clean_russian_and_english_are_unchanged(self):
        for value in ("Системы работают штатно.", "JARVIS is online."):
            self.assertEqual(repair_mojibake(value), value)

    def test_stream_buffer_is_repaired_before_sentence_emission(self):
        emitted = []
        broken = "Привет! Всё работает.".encode("utf-8").decode("latin-1")
        buffer = repair_mojibake(broken)
        remainder = LLMClient._emit_sentences(buffer, emitted.append, flush=True)
        self.assertEqual(emitted, ["Привет!", "Всё работает."])
        self.assertEqual(remainder, "")

    def test_empty_gateway_stream_retries_non_streaming(self):
        response = Mock(status_code=200)
        response.iter_lines.return_value = []
        emitted = []
        client = LLMClient()
        with (
            patch("assistant.llm_client.requests.post", return_value=response),
            patch.object(
                client, "_post_openai", return_value="Нормальный ответ"
            ) as fallback,
        ):
            result = client._stream_openai(
                "https://example.test/chat/completions",
                "router/auto",
                [{"role": "user", "content": "Привет"}],
                "key",
                emitted.append,
            )
        fallback.assert_called_once()
        self.assertEqual(result, "Нормальный ответ")
        self.assertEqual(emitted, ["Нормальный ответ"])


if __name__ == "__main__":
    unittest.main()
