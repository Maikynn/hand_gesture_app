from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class LLMSettings:
    provider: str = "none"
    model: str = ""
    endpoint: str = ""
    api_key: str = ""
    system_prompt: str = ""
    max_tokens: int = 350
    fallback_models: Optional[List[str]] = None


class LLMClient:
    """OpenRouter, Ollama and generic OpenAI-compatible chat client."""

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()

    def configure(self, settings: LLMSettings) -> None:
        self.settings = settings

    def is_configured(self) -> bool:
        provider = self.settings.provider.lower()
        if provider == "ollama":
            return bool(self.settings.endpoint and self.settings.model)
        if provider in {"openrouter", "custom"}:
            return bool(self.settings.api_key and self.settings.model)
        return False

    def send_message(self, message: str, system_prompt: str | None = None) -> str:
        provider = self.settings.provider.lower()
        prompt = self.settings.system_prompt if system_prompt is None else system_prompt
        if provider == "ollama":
            return self._ollama(message, prompt)
        if provider == "openrouter":
            return self._openrouter(message, prompt)
        if provider == "custom":
            return self._openai_compatible(message, prompt)
        raise LLMError("Провайдер ответов не настроен")

    def _messages(self, message: str, prompt: str) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        if prompt.strip():
            result.append({"role": "system", "content": prompt.strip()})
        result.append({"role": "user", "content": message.strip()})
        return result

    def _openrouter(self, message: str, prompt: str) -> str:
        if not self.settings.api_key:
            raise LLMError("Не указан ключ OpenRouter")
        models = [self.settings.model or "openrouter/auto"]
        for model in self.settings.fallback_models or []:
            if model and model not in models:
                models.append(model)
        last_error = ""
        for model in models:
            try:
                return self._post_openai(
                    "https://openrouter.ai/api/v1/chat/completions",
                    model,
                    self._messages(message, prompt),
                    self.settings.api_key,
                )
            except LLMError as exc:
                last_error = str(exc)
        raise LLMError(last_error or "Все модели OpenRouter недоступны")

    def _openai_compatible(self, message: str, prompt: str) -> str:
        endpoint = self.settings.endpoint.strip().rstrip("/")
        if not endpoint:
            raise LLMError("Не указан адрес совместимого API")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        return self._post_openai(
            endpoint,
            self.settings.model,
            self._messages(message, prompt),
            self.settings.api_key,
        )

    def _post_openai(
        self, endpoint: str, model: str, messages: List[Dict[str, str]], api_key: str
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.65,
            "max_tokens": int(self.settings.max_tokens),
        }
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=45)
        except requests.RequestException as exc:
            raise LLMError(f"Нет связи с API: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(f"API вернуло HTTP {response.status_code}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("API вернуло неожиданный формат ответа") from exc
        if not str(content).strip():
            raise LLMError("API вернуло пустой ответ")
        return str(content).strip()

    def _ollama(self, message: str, prompt: str) -> str:
        endpoint = (self.settings.endpoint or "http://127.0.0.1:11434").rstrip("/")
        payload = {
            "model": self.settings.model or "llama3.2",
            "messages": self._messages(message, prompt),
            "stream": False,
            "think": False,
            "options": {"num_predict": max(200, int(self.settings.max_tokens))},
        }
        try:
            response = requests.post(endpoint + "/api/chat", json=payload, timeout=120)
        except requests.RequestException as exc:
            raise LLMError(f"Ollama не отвечает: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(f"Ollama вернула HTTP {response.status_code}")
        try:
            content = str(response.json().get("message", {}).get("content", ""))
        except ValueError as exc:
            raise LLMError("Ollama вернула некорректный JSON") from exc
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        if not content:
            raise LLMError("Локальная модель не вернула ответ")
        return content
