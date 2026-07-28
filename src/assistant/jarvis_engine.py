"""Embedded command core for the Priler/Jarvis-style assistant.

The original project is available at https://github.com/Priler/jarvis.  This
module keeps the assistant inside our Qt tab and follows Jarvis' central idea:
commands are configured as phrases/aliases and dispatched to small handlers;
unmatched speech is delegated to the selected conversational provider.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

if TYPE_CHECKING:
    from .action_executor import ActionExecutor
    from .llm_client import LLMClient


@dataclass(frozen=True)
class JarvisReply:
    text: str
    handled: bool


class JarvisEngine:
    """Resolve exact/fuzzy voice aliases, then fall back to a configured LLM."""

    CONTROL_PREFIXES = ("открой ", "запусти ", "open ", "run ")

    def __init__(
        self,
        voice_config: Dict[str, Any],
        executor: ActionExecutor,
        llm: LLMClient,
    ) -> None:
        self.voice_config = voice_config
        self.executor = executor
        self.llm = llm

    def update(
        self,
        voice_config: Dict[str, Any],
        executor: ActionExecutor,
        llm: LLMClient,
    ) -> None:
        self.voice_config = voice_config
        self.executor = executor
        self.llm = llm

    def process(self, text: str) -> JarvisReply:
        normalized = self._normalize(text)
        if not normalized:
            return JarvisReply("Я вас не расслышал.", True)

        builtin = self._builtin(normalized)
        if builtin:
            return JarvisReply(builtin, True)

        command = self._find_command(normalized)
        if command:
            ok, reply = self.executor.execute(
                command.get("action", "none"), command.get("value", "")
            )
            return JarvisReply(reply, ok)

        app_reply = self._launch_named_app(normalized)
        if app_reply:
            return app_reply

        answer = self.llm.send_message(text)
        if answer and not self._is_provider_error(answer):
            return JarvisReply(answer, False)
        return JarvisReply(self._error_phrase(), False)

    def _find_command(self, text: str) -> Optional[Dict[str, str]]:
        best: Optional[Dict[str, str]] = None
        best_score = 0.0
        threshold = float(self.voice_config.get("command_match_threshold", 0.78))
        for command in self.voice_config.get("commands", []):
            for alias in self._aliases(command):
                if text == alias:
                    return command
                score = SequenceMatcher(None, text, alias).ratio()
                if score > best_score:
                    best, best_score = command, score
        return best if best_score >= threshold else None

    def _launch_named_app(self, text: str) -> Optional[JarvisReply]:
        prefix = next(
            (item for item in self.CONTROL_PREFIXES if text.startswith(item)), None
        )
        if not prefix:
            return None
        requested = text[len(prefix) :].strip()
        if not requested:
            return JarvisReply("Скажите, какое приложение открыть.", True)
        ok, reply = self.executor.execute("open_app", requested)
        return JarvisReply(reply, ok)

    @staticmethod
    def _aliases(command: Dict[str, Any]) -> Iterable[str]:
        phrases = command.get("aliases") or command.get("phrase") or ""
        if isinstance(phrases, str):
            phrases = phrases.split("|")
        return (
            JarvisEngine._normalize(str(item)) for item in phrases if str(item).strip()
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().replace("ё", "е").strip().split())

    @staticmethod
    def _builtin(text: str) -> Optional[str]:
        now = dt.datetime.now()
        if text in {"привет", "здравствуй", "доброе утро", "добрый вечер"}:
            return random.choice(("Привет! Чем помочь?", "Я на связи.", "Слушаю вас."))
        if text in {"который час", "сколько времени", "время"}:
            return f"Сейчас {now:%H:%M}."
        if text in {"какая дата", "какое сегодня число", "дата"}:
            return f"Сегодня {now:%d.%m.%Y}."
        if text in {"спасибо", "благодарю"}:
            return random.choice(("Всегда пожалуйста!", "Рад помочь.", "Обращайтесь!"))
        return None

    def _error_phrase(self) -> str:
        phrases = self.voice_config.get("error_phrases") or [
            "Не удалось получить ответ. Проверьте настройки модели."
        ]
        return random.choice(phrases)

    @staticmethod
    def _is_provider_error(answer: str) -> bool:
        lowered = answer.casefold()
        return lowered.startswith(
            ("ошибка", "error", "network error", "не удалось подключиться")
        )
