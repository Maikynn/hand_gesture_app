#!/usr/bin/env python3
"""
Unified LLM client.

Supports two backends selected by `engine`:
  * "openrouter" - cloud API at https://openrouter.ai/api/v1
  * "ollama"     - local model served by Ollama (native /api/chat endpoint)

Both expose a single send_message(user_text) -> str method.

OpenRouter resilience
---------------------
* 429 (rate limit) and 5xx responses are retried with exponential backoff,
  honouring the server's `Retry-After` header when present.
* A fallback chain of models is tried in order, so a single rate-limited or
  unavailable model automatically rolls over to the next one (e.g. the free
  tier of one model is exhausted -> try another free model with the same key).
  This is the main mitigation for the 429 "limit reached" errors.
"""
import time
import requests
import re
from typing import List, Optional


class LLMClient:
    def __init__(self, engine: str = "openrouter", api_key: str = "",
                 model: str = "", ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3", system_prompt: str = "",
                 max_tokens: int = 200, fallback_models: Optional[List[str]] = None):
        self.engine = (engine or "openrouter").lower()
        self.api_key = api_key or ""
        self.model = model or ""
        self.ollama_url = ollama_url.rstrip("/") or "http://localhost:11434"
        self.ollama_model = ollama_model or "llama3"
        self.system_prompt = system_prompt or ""
        self.max_tokens = max_tokens
        # Ordered list of alternative OpenRouter models to try when the
        # primary is rate-limited / unavailable.
        self.fallback_models = [m for m in (fallback_models or []) if m]

    def send_message(self, message: str, system_prompt: Optional[str] = None) -> Optional[str]:
        sp = self.system_prompt if system_prompt is None else system_prompt
        if self.engine == "ollama":
            return self._ollama(message, sp)
        return self._openrouter(message, sp)

    # -- OpenRouter (cloud) -------------------------------------------------
    def _openrouter(self, message: str, sp: str) -> Optional[str]:
        if not self.api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": message})

        # Primary model first, then the configured fallbacks (deduped).
        models = [self.model or "openai/gpt-3.5-turbo"]
        for m in self.fallback_models:
            if m and m not in models:
                models.append(m)

        last_err = None
        for model in models:
            ok, content, err = self._try_openrouter(headers, messages, model)
            if ok:
                return content
            # A hard (non-retryable) error on one model still lets us try the
            # next model, because a different model may succeed.
            last_err = err
        return last_err or "Ошибка OpenRouter: все модели недоступны."

    def _try_openrouter(self, headers, messages, model, max_retries: int = 4):
        """Attempt one model with retry/backoff on 429 and 5xx.

        Returns (success: bool, content: str|None, error: str|None).
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
        }
        url = "https://openrouter.ai/api/v1/chat/completions"
        delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=30)
                if r.status_code == 200:
                    return True, r.json()["choices"][0]["message"]["content"], None
                # Retryable: rate limit or server error.
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after is not None else delay
                    except (TypeError, ValueError):
                        wait = delay
                    if attempt < max_retries:
                        time.sleep(wait)
                        delay = min(delay * 2, 30.0)
                        continue
                    return False, None, (
                        f"Ошибка OpenRouter ({model}): {r.status_code} "
                        f"{r.text[:200]}")
                # Non-retryable client error (401, 403, 404, 422, ...).
                return False, None, (
                    f"Ошибка OpenRouter ({model}): {r.status_code} {r.text[:200]}")
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                return False, None, f"Ошибка сети OpenRouter: {e}"
        return False, None, f"Ошибка OpenRouter ({model}): превышено число попыток."

    # -- Ollama (local) -----------------------------------------------------
    def _ollama(self, message: str, sp: str) -> Optional[str]:
        url = self.ollama_url + "/api/chat"
        messages = []
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": message})
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            # Disable chain-of-thought for reasoning models (e.g. deepseek-r1)
            # so the assistant gets a direct, speakable answer instead of a
            # <think>...</think> block that would otherwise eat the whole
            # token budget and leave `content` empty.
            "think": False,
            # Reasoning models need headroom; floor the budget so a final
            # answer is always produced even if `think` is ignored.
            "options": {"num_predict": max(self.max_tokens, 400)},
        }
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                content = r.json().get("message", {}).get("content", "") or ""
                # Defensive: drop any residual reasoning blocks.
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
                return content
            return f"Ошибка Ollama: {r.status_code} {r.text[:200]}"
        except Exception as e:
            return (f"Не удалось подключиться к Ollama по адресу "
                    f"{self.ollama_url}: {e}")
