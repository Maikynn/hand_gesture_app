import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.actions import ActionExecutor
from assistant.embedded_jarvis import EmbeddedJarvis
from assistant.llm_client import LLMError
from utils.config_store import ConfigStore


class FakeVoice:
    def __init__(self):
        self.groups = []

    def play(self, group):
        self.groups.append(group)
        return True


class FakeLLM:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.settings = None

    def configure(self, settings):
        self.settings = settings

    def send_message(self, _message):
        if self.error:
            raise LLMError(self.error)
        return self.answer


class EmbeddedJarvisTests(unittest.TestCase):
    def make_store(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        config = {
            "assistant": {
                "wake_phrases": "джарвис, аксиос",
                "llm_provider": "none",
                "command_match_threshold": 60,
                "error_phrases": ["Смешная ошибка"],
            },
            "permissions": [
                {"name": "Python", "path": sys.executable, "enabled": True}
            ],
            "commands": [
                {
                    "id": "open_python",
                    "phrases": ["открой питон", "запусти питон"],
                    "type": "application",
                    "target": sys.executable,
                    "reply": "Открываю Python",
                    "enabled": True,
                }
            ],
        }
        public = root / "config.json"
        public.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return temp, ConfigStore(public, root / "config.local.json")

    def test_fuzzy_command_executes_allowlisted_application(self):
        temp, store = self.make_store()
        self.addCleanup(temp.cleanup)
        launched = []
        voice = FakeVoice()
        core = EmbeddedJarvis(
            store,
            executor=ActionExecutor(
                store.get("permissions"), app_launcher=launched.append
            ),
            llm=FakeLLM(answer="unused"),
            voice_pack=voice,
        )
        result = core.handle_text("Джарвис, открой питно")
        self.assertEqual(result.kind, "command")
        self.assertEqual(result.command_id, "open_python")
        self.assertEqual(launched, [sys.executable])
        self.assertIn("ok", voice.groups)

    def test_question_uses_configured_llm(self):
        temp, store = self.make_store()
        self.addCleanup(temp.cleanup)
        llm = FakeLLM(answer="Ответ модели")
        core = EmbeddedJarvis(store, llm=llm, voice_pack=FakeVoice())
        result = core.handle_text("Как дела?")
        self.assertEqual(result.kind, "answer")
        self.assertEqual(result.text, "Ответ модели")

    def test_llm_error_returns_configured_funny_phrase(self):
        temp, store = self.make_store()
        self.addCleanup(temp.cleanup)
        core = EmbeddedJarvis(
            store, llm=FakeLLM(error="offline"), voice_pack=FakeVoice()
        )
        result = core.handle_text("Расскажи про космос")
        self.assertEqual(result.kind, "fallback")
        self.assertEqual(result.text, "Смешная ошибка")
        self.assertEqual(result.error, "offline")

    def test_wake_phrase_without_command_acknowledges(self):
        temp, store = self.make_store()
        self.addCleanup(temp.cleanup)
        result = EmbeddedJarvis(
            store, llm=FakeLLM(answer="unused"), voice_pack=FakeVoice()
        ).handle_text("джарвис")
        self.assertEqual(result.kind, "wake")
        self.assertEqual(result.text, "Слушаю.")


if __name__ == "__main__":
    unittest.main()
