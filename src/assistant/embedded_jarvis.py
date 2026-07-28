from __future__ import annotations

import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from assistant.actions import ActionError, ActionExecutor
from assistant.document_index import DocumentIndex
from assistant.llm_client import LLMClient, LLMError, LLMSettings
from assistant.memory_store import MemoryStore
from utils.config_store import ConfigStore, PROJECT_ROOT


@dataclass(frozen=True)
class AssistantResult:
    text: str
    kind: str
    command_id: str = ""
    score: float = 0.0
    error: str = ""


class PrilerVoicePack:
    """Plays the original Priler/Jarvis WAV reactions when available."""

    def __init__(self, root: Path | None = None):
        self.root = root or (
            PROJECT_ROOT
            / "third_party"
            / "priler_jarvis"
            / "resources"
            / "sound"
            / "voices"
            / "jarvis-og"
            / "ru"
        )
        self.groups = {
            "reply": ["reply1.wav", "reply2.wav", "reply3.wav"],
            "ok": ["ok1.wav", "ok2.wav", "ok3.wav", "ok4.wav"],
            "thanks": ["thanks.wav"],
            "not_found": ["not_found.wav"],
            "start": ["run.wav"],
            "stop": ["off.wav"],
        }

    def play(self, group: str) -> bool:
        files = [self.root / name for name in self.groups.get(group, [])]
        files = [path for path in files if path.is_file()]
        if not files:
            return False
        try:
            import winsound

            winsound.PlaySound(
                str(random.choice(files)),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
            return True
        except Exception:
            return False


class EmbeddedJarvis:
    """Native assistant engine inspired by and compatible with Priler/Jarvis.

    It runs in the host Python process. No Tauri window, local web server, or
    secondary assistant executable is started.
    """

    def __init__(
        self,
        store: ConfigStore,
        executor: ActionExecutor | None = None,
        llm: LLMClient | None = None,
        voice_pack: PrilerVoicePack | None = None,
    ):
        self.store = store
        self.executor = executor or ActionExecutor(store.get("permissions", []))
        self.llm = llm or LLMClient()
        self.voice_pack = voice_pack or PrilerVoicePack()
        self.memory = MemoryStore()
        self.documents = DocumentIndex()
        self.commands: List[Dict[str, Any]] = []
        self.error_phrases: List[str] = []
        self.threshold = 66.0
        self.wake_phrases: List[str] = []
        self.confirmation_level = "none"
        self._pending_command: Optional[Tuple[Dict[str, Any], float]] = None
        self.reload()

    def reload(self) -> None:
        self.commands = [
            command
            for command in self.store.get("commands", [])
            if isinstance(command, dict)
        ]
        self.executor.set_permissions(self.store.get("permissions", []))
        self.executor.set_scenarios(self.store.get("scenarios", []))
        assistant = self.store.get("assistant", {}) or {}
        self.threshold = float(assistant.get("command_match_threshold", 66))
        self.error_phrases = [
            str(item)
            for item in assistant.get("error_phrases", [])
            if str(item).strip()
        ]
        self.wake_phrases = self._split_phrases(
            str(assistant.get("wake_phrases", "джарвис, аксиос"))
        )
        self.confirmation_level = str(
            assistant.get("confirmation_level", "none")
        ).lower()
        self.documents.set_root(str(assistant.get("documents_folder", "")))
        provider = str(assistant.get("llm_provider", "none"))
        endpoint = ""
        model = ""
        if provider == "openrouter":
            model = str(assistant.get("openrouter_model", "openrouter/auto"))
        elif provider == "ollama":
            endpoint = str(assistant.get("ollama_url", "http://127.0.0.1:11434"))
            model = str(assistant.get("ollama_model", "llama3.2"))
        elif provider == "custom":
            endpoint = str(assistant.get("custom_api_url", ""))
            model = str(assistant.get("custom_model", ""))
        self.llm.configure(
            LLMSettings(
                provider=provider,
                endpoint=endpoint,
                model=model,
                api_key=self.store.api_key(provider),
                system_prompt=str(assistant.get("system_prompt", "")),
                max_tokens=int(assistant.get("max_tokens", 350)),
                fallback_models=[
                    str(item)
                    for item in assistant.get("openrouter_fallback_models", [])
                    if str(item).strip()
                ],
            )
        )

    @staticmethod
    def _split_phrases(value: str | Iterable[str]) -> List[str]:
        if isinstance(value, str):
            parts = re.split(r"[,|;\n]+", value)
        else:
            parts = list(value)
        return [str(part).strip().lower() for part in parts if str(part).strip()]

    @staticmethod
    def _word_score(actual: str, expected: str) -> float:
        actual_words = actual.split()
        expected_words = expected.split()
        if not actual_words or not expected_words:
            return 0.0
        matched = 0.0
        for actual_word in actual_words:
            best = max(
                SequenceMatcher(None, actual_word, expected_word).ratio()
                for expected_word in expected_words
            )
            if best >= 0.7:
                matched += best
        return matched / max(len(actual_words), len(expected_words)) * 100.0

    @classmethod
    def similarity(cls, actual: str, expected: str) -> float:
        actual = " ".join(actual.lower().strip().split())
        expected = " ".join(expected.lower().strip().split())
        if actual == expected:
            return 100.0
        char_score = SequenceMatcher(None, actual, expected).ratio() * 100.0
        return char_score * 0.6 + cls._word_score(actual, expected) * 0.4

    def find_command(self, phrase: str) -> Optional[Tuple[Dict[str, Any], float]]:
        best: Optional[Tuple[Dict[str, Any], float]] = None
        for command in self.commands:
            if not command.get("enabled", True):
                continue
            phrases = command.get("phrases", [])
            if isinstance(phrases, str):
                phrases = self._split_phrases(phrases)
            for candidate in phrases:
                score = self.similarity(phrase, str(candidate))
                if score >= self.threshold and (best is None or score > best[1]):
                    best = (command, score)
                if score == 100.0:
                    return best
        return best

    def strip_wake_phrase(self, text: str) -> str:
        normalized = text.lower().strip()
        for phrase in self.wake_phrases:
            position = normalized.find(phrase)
            if position >= 0:
                return normalized[position + len(phrase) :].strip(" ,.!?-")
        return text.strip()

    def handle_text(self, text: str) -> AssistantResult:
        clean = self.strip_wake_phrase(text)
        if not clean:
            self.voice_pack.play("reply")
            return AssistantResult("Слушаю.", "wake")
        normalized = clean.lower().strip(" ,.!?")
        if self._pending_command is not None:
            if normalized in {"да", "подтверждаю", "выполняй", "согласен", "окей"}:
                command, score = self._pending_command
                self._pending_command = None
                return self._execute_command(command, score)
            if normalized in {"нет", "отмена", "не надо", "отставить"}:
                self._pending_command = None
                return AssistantResult("Отменено.", "cancelled")
            return AssistantResult(
                "Жду «да» для выполнения или «нет» для отмены.", "confirmation"
            )
        if normalized in {"отмени", "отмени последнее", "верни как было"}:
            try:
                return AssistantResult(
                    self.executor.undo_last(), "command", "undo", 100.0
                )
            except ActionError as exc:
                return AssistantResult(str(exc), "error", "undo", 100.0, str(exc))
        if normalized.startswith(("запомни ", "запомни что ")):
            value = re.sub(
                r"^запомни(?:\s+что)?\s+", "", clean, flags=re.IGNORECASE
            )
            try:
                self.memory.remember(value)
                return AssistantResult(
                    "Запомнил. Это можно удалить в настройках памяти.", "command"
                )
            except ValueError as exc:
                return AssistantResult(str(exc), "error", error=str(exc))
        if normalized in {"что ты помнишь", "покажи память", "моя память"}:
            values = [str(item.get("text", "")) for item in self.memory.items[-10:]]
            memory_text = "\n".join(f"• {value}" for value in values if value)
            return AssistantResult(memory_text or "Память пока пуста.", "answer")
        match = self.find_command(clean)
        if match:
            command, score = match
            if self._needs_confirmation(command):
                self._pending_command = (command, score)
                return AssistantResult(
                    f"Подтвердить действие: "
                    f"{command.get('reply') or command.get('target')}? "
                    "Скажи «да» или «нет».",
                    "confirmation",
                    command_id=str(command.get("id", "")),
                    score=score,
                )
            return self._execute_command(command, score)
        try:
            prompt = self._prompt_for(clean)
            try:
                answer = self.llm.send_message(clean, system_prompt=prompt)
            except TypeError:
                # Small injected test/third-party clients may expose the older
                # one-argument surface.
                answer = self.llm.send_message(clean)
            return AssistantResult(answer, "answer")
        except LLMError as exc:
            self.voice_pack.play("not_found")
            return AssistantResult(self._funny_error(), "fallback", error=str(exc))

    def _needs_confirmation(self, command: Dict[str, Any]) -> bool:
        if bool(command.get("requires_confirmation", False)):
            return True
        kind = str(command.get("type", "")).lower()
        if self.confirmation_level == "all":
            return kind != "response"
        if self.confirmation_level != "dangerous":
            return False
        target = str(command.get("target", "")).lower()
        return kind in {"application", "url", "hotkey", "scenario"} or (
            kind == "system" and target in {"lock", "screenshot"}
        )

    def _execute_command(
        self, command: Dict[str, Any], score: float
    ) -> AssistantResult:
        if (
            bool(self.store.get("assistant.dry_run_commands", False))
            and str(command.get("type", "")) != "response"
        ):
            description = command.get("reply") or (
                f"{command.get('type', '')}: {command.get('target', '')}"
            )
            return AssistantResult(
                f"SIMULATION // Я бы выполнил: {description}. Действие не запускалось.",
                "preview",
                command_id=str(command.get("id", "")),
                score=score,
            )
        try:
            response = self.executor.execute(command)
            response = str(command.get("reply") or response)
            self.voice_pack.play("ok")
            return AssistantResult(
                response,
                "command",
                command_id=str(command.get("id", "")),
                score=score,
            )
        except ActionError as exc:
            self.voice_pack.play("not_found")
            return AssistantResult(
                self._funny_error(),
                "error",
                str(command.get("id", "")),
                score,
                str(exc),
            )
        except Exception as exc:
            self.voice_pack.play("not_found")
            return AssistantResult(
                self._funny_error(),
                "error",
                str(command.get("id", "")),
                score,
                str(exc),
            )

    def handle_text_stream(
        self, text: str, on_sentence
    ) -> AssistantResult:
        """Stream ordinary LLM answers; commands keep the deterministic path."""
        clean = self.strip_wake_phrase(text)
        normalized = clean.lower().strip(" ,.!?")
        special = (
            self._pending_command is not None
            or normalized.startswith(("запомни ", "запомни что "))
            or normalized
            in {
                "отмени",
                "отмени последнее",
                "верни как было",
                "что ты помнишь",
                "покажи память",
                "моя память",
            }
        )
        if special or not clean or self.find_command(clean) or not self.llm.is_configured():
            return self.handle_text(text)
        prompt = self._prompt_for(clean)
        try:
            answer = self.llm.send_message_stream(clean, on_sentence, prompt)
            return AssistantResult(answer, "streamed")
        except LLMError as exc:
            self.voice_pack.play("not_found")
            return AssistantResult(self._funny_error(), "fallback", error=str(exc))

    def _prompt_for(self, query: str) -> str:
        assistant = self.store.get("assistant", {}) or {}
        base_prompt = str(self.llm.settings.system_prompt or "")
        memory = self.memory.prompt_context()
        documents = ""
        if bool(assistant.get("documents_enabled", False)):
            documents = self.documents.prompt_context(query)
        language_mode = str(assistant.get("language_mode", "ru"))
        if language_mode == "auto":
            cyrillic = bool(re.search(r"[А-Яа-яЁё]", query))
            latin = bool(re.search(r"[A-Za-z]", query))
            language = "Russian" if cyrillic or not latin else "English"
            language_rule = f"Reply in {language}, matching the user's language."
        else:
            language_rule = "Всегда отвечай на русском языке."
        return "\n\n".join(
            value for value in (base_prompt, language_rule, memory, documents) if value
        )

    def _funny_error(self) -> str:
        if self.error_phrases:
            return random.choice(self.error_phrases)
        return "Мои нейроны ушли пить чай. Настрой модель — и я вернусь умнее."
