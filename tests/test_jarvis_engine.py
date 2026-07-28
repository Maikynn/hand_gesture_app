import unittest

from src.assistant.jarvis_engine import JarvisEngine


class StubExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, action, value=""):
        self.calls.append((action, value))
        return True, f"done:{action}:{value}"


class StubLLM:
    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    def send_message(self, text):
        self.calls.append(text)
        return self.answer


class JarvisEngineTests(unittest.TestCase):
    def make_engine(self, llm_answer=None):
        config = {
            "command_match_threshold": 0.7,
            "commands": [
                {
                    "phrase": "открой браузер|запусти браузер",
                    "action": "open_url",
                    "value": "https://example.com",
                }
            ],
            "error_phrases": ["fallback"],
        }
        executor = StubExecutor()
        llm = StubLLM(llm_answer)
        return JarvisEngine(config, executor, llm), executor, llm

    def test_exact_alias_dispatches_action_without_llm(self):
        engine, executor, llm = self.make_engine("unused")
        reply = engine.process("Запусти браузер")
        self.assertEqual(executor.calls, [("open_url", "https://example.com")])
        self.assertEqual(llm.calls, [])
        self.assertTrue(reply.handled)

    def test_minor_speech_typo_still_matches_command(self):
        engine, executor, _ = self.make_engine()
        engine.process("открой браузир")
        self.assertEqual(len(executor.calls), 1)

    def test_unmatched_question_uses_llm(self):
        engine, _, llm = self.make_engine("ответ модели")
        reply = engine.process("почему небо синее")
        self.assertEqual(reply.text, "ответ модели")
        self.assertEqual(llm.calls, ["почему небо синее"])

    def test_provider_error_uses_friendly_fallback(self):
        engine, _, _ = self.make_engine("Ошибка API: 500")
        self.assertEqual(engine.process("вопрос").text, "fallback")


if __name__ == "__main__":
    unittest.main()
