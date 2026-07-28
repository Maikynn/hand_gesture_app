"""Persistent application configuration with safe, portable defaults."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.json"
LOCAL_CONFIG_PATH = ROOT / "config.local.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "appearance": {"theme": "dark"},
    "camera": {
        "device_index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "brightness": 0,
        "flip": True,
        "show_hands": True,
    },
    "gesture": {
        "static_model_id": "mediapipe",
        "confidence_threshold": 0.65,
        "hand_crop_padding": 30,
        "action_cooldown": 2.0,
        "bindings": [
            {"hand": "Любая", "gesture": "palm", "action": "volume_up", "value": ""},
            {"hand": "Любая", "gesture": "fist", "action": "volume_down", "value": ""},
            {
                "hand": "Правая",
                "gesture": "one",
                "action": "open_url",
                "value": "https://www.google.com",
            },
        ],
    },
    "voice": {
        "enabled": False,
        "wake_word": "аксиос",
        "stt_engine": "vosk",
        "whisper_model": "bond005/whisper-podlodka-turbo",
        "mic_index": None,
        "mic_gain": 2.0,
        "tts_engine": "pyttsx3",
        "tts_speaker": "xenia",
        "tts_rate": 190,
        "volume": 0.9,
        "llm_engine": "none",
        "command_match_threshold": 0.78,
        "openrouter_key": "",
        "openrouter_model": "qwen/qwen3-coder:free",
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3",
        "custom_api_url": "",
        "custom_api_key": "",
        "custom_api_model": "",
        "system_prompt": "Ты — голосовой помощник Акси. Отвечай кратко и по делу на русском.",
        "error_phrases": [
            "Мои нейроны ушли пить чай. Настройте модель — и я вернусь!",
            "Кажется, интернет спрятался. Но я всё ещё умею запускать ваши команды!",
            "Без модели я сегодня философ-молчун. Выберите провайдера в настройках.",
        ],
        "allowed_apps": [],
        "commands": [
            {
                "phrase": "открой браузер|запусти браузер",
                "action": "open_url",
                "value": "https://www.google.com",
            },
            {
                "phrase": "прибавь громкость|сделай громче",
                "action": "volume_up",
                "value": "",
            },
            {
                "phrase": "убавь громкость|сделай тише",
                "action": "volume_down",
                "value": "",
            },
        ],
    },
}


def _merge(defaults: Dict[str, Any], saved: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(defaults)
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    for path in (CONFIG_PATH, LOCAL_CONFIG_PATH):
        config = _merge(config, _read_json(path))
    return config


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def save_config(config: Dict[str, Any]) -> None:
    """Atomically save user settings outside the tracked config template."""
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".config-", suffix=".tmp", dir=LOCAL_CONFIG_PATH.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, LOCAL_CONFIG_PATH)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
