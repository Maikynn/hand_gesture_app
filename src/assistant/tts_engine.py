#!/usr/bin/env python3
"""Queued Jarvis-style neural TTS with a reliable offline fallback."""

from __future__ import annotations

import ctypes
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pyttsx3


NEURAL_VOICES = [
    ("ru-RU-DmitryNeural", "Дмитрий · мужской, глубокий"),
    ("ru-RU-SvetlanaNeural", "Светлана · женский, мягкий"),
]
DEFAULT_NEURAL_VOICE = "ru-RU-DmitryNeural"


def list_voices() -> List[Tuple[str, str]]:
    """Return installed Windows SAPI voices without failing application startup."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        return [(voice.id, voice.name) for voice in voices]
    except Exception:
        return []


class TTSEngine:
    """Serializes all speech work on one thread.

    ``edge`` uses a high-quality Microsoft neural voice over the network.
    When the service is unavailable, the same utterance automatically falls
    back to the local Windows SAPI voice.
    """

    def __init__(
        self,
        *,
        engine: str = "edge",
        voice_id: Optional[str] = DEFAULT_NEURAL_VOICE,
        rate: int = 175,
        volume: float = 0.9,
        pitch: int = -12,
    ):
        self.engine_name = engine if engine in {"edge", "system"} else "edge"
        self.voice_id = voice_id or DEFAULT_NEURAL_VOICE
        self.rate = max(90, min(260, int(rate)))
        self.volume = max(0.0, min(1.0, float(volume)))
        self.pitch = max(-50, min(50, int(pitch)))
        self.is_speaking = False

        self.speech_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._offline_engine: Any = None
        self._engine_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._mci_alias = ""
        self._edge_retry_after = 0.0
        self._stopping = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._process_queue,
            name="jarvis-tts",
            daemon=True,
        )
        self.worker_thread.start()

    def _process_queue(self) -> None:
        while not self._stopping.is_set():
            item = self.speech_queue.get()
            try:
                if item is None:
                    return
                self._speak(item)
            finally:
                self.speech_queue.task_done()

    def _speak(self, text: str) -> None:
        if not text.strip():
            return
        self.is_speaking = True
        try:
            if self.engine_name == "edge":
                if time.monotonic() >= self._edge_retry_after:
                    try:
                        self._speak_edge(text)
                        self._edge_retry_after = 0.0
                        return
                    except Exception as exc:
                        # Do not make every reply wait on the same unavailable
                        # online service. Retry after five minutes or settings save.
                        self._edge_retry_after = time.monotonic() + 300.0
                        print(f"Neural TTS unavailable, using Windows voice: {exc}")
            self._speak_offline(text)
        except Exception as exc:
            print(f"Speech error: {exc}")
            self._offline_engine = None
        finally:
            self.is_speaking = False

    def _speak_edge(self, text: str) -> None:
        import edge_tts

        rate_percent = round((self.rate - 175) / 1.75)
        volume_percent = round((self.volume - 1.0) * 100)
        communicator = edge_tts.Communicate(
            text,
            self.voice_id or DEFAULT_NEURAL_VOICE,
            rate=f"{rate_percent:+d}%",
            volume=f"{volume_percent:+d}%",
            pitch=f"{self.pitch:+d}Hz",
            connect_timeout=5,
            receive_timeout=30,
        )
        handle, raw_path = tempfile.mkstemp(prefix="axi-jarvis-", suffix=".mp3")
        os.close(handle)
        path = Path(raw_path)
        try:
            communicator.save_sync(str(path))
            if not path.is_file() or path.stat().st_size < 256:
                raise RuntimeError("Сервис не вернул аудио")
            self._play_mp3(path)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _play_mp3(self, path: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Neural playback is currently configured for Windows")
        alias = f"axi_jarvis_{threading.get_ident()}"
        with self._state_lock:
            self._mci_alias = alias
        try:
            self._mci(f'open "{path}" type mpegvideo alias {alias}')
            self._mci(f"play {alias} wait")
        finally:
            try:
                self._mci(f"close {alias}", check=False)
            finally:
                with self._state_lock:
                    self._mci_alias = ""

    @staticmethod
    def _mci(command: str, *, check: bool = True) -> None:
        winmm = ctypes.windll.winmm
        code = int(winmm.mciSendStringW(command, None, 0, None))
        if code and check:
            buffer = ctypes.create_unicode_buffer(256)
            winmm.mciGetErrorStringW(code, buffer, len(buffer))
            raise RuntimeError(buffer.value or f"MCI error {code}")

    def _ensure_offline_engine(self):
        if self._offline_engine is None:
            self._offline_engine = pyttsx3.init()
        self._apply_offline_settings()
        return self._offline_engine

    def _apply_offline_settings(self) -> None:
        if self._offline_engine is None:
            return
        engine = self._offline_engine
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        voices = engine.getProperty("voices") or []
        selected = None
        if self.engine_name == "system" and self.voice_id:
            selected = next((voice.id for voice in voices if voice.id == self.voice_id), None)
        if not selected:
            selected = next(
                (
                    voice.id
                    for voice in voices
                    if "russian" in voice.name.lower()
                    or "ru-ru" in voice.id.lower()
                    or "irina" in voice.name.lower()
                ),
                voices[0].id if voices else None,
            )
        if selected:
            engine.setProperty("voice", selected)

    def _speak_offline(self, text: str) -> None:
        with self._engine_lock:
            engine = self._ensure_offline_engine()
            engine.say(text)
            engine.runAndWait()

    def speak(self, text: str) -> None:
        value = str(text).strip()
        if value:
            self.speech_queue.put(value)

    def configure(
        self,
        *,
        engine: str,
        voice_id: str,
        rate: int,
        volume: float,
        pitch: int,
    ) -> None:
        self.engine_name = engine if engine in {"edge", "system"} else "edge"
        self.voice_id = voice_id or DEFAULT_NEURAL_VOICE
        self.rate = max(90, min(260, int(rate)))
        self.volume = max(0.0, min(1.0, float(volume)))
        self.pitch = max(-50, min(50, int(pitch)))
        self._edge_retry_after = 0.0

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))

    def set_rate(self, rate: int) -> None:
        self.rate = max(90, min(260, int(rate)))

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id

    def set_engine(self, engine: str) -> None:
        self.engine_name = engine if engine in {"edge", "system"} else "edge"

    def set_pitch(self, pitch: int) -> None:
        self.pitch = max(-50, min(50, int(pitch)))

    def get_voices(self) -> List[Tuple[str, str]]:
        return list_voices()

    def stop(self) -> None:
        while True:
            try:
                item = self.speech_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.speech_queue.task_done()
                if item is None:
                    self.speech_queue.put(None)
                    break
        with self._state_lock:
            alias = self._mci_alias
        if alias and os.name == "nt":
            self._mci(f"stop {alias}", check=False)
        with self._engine_lock:
            if self._offline_engine is not None:
                try:
                    self._offline_engine.stop()
                except Exception:
                    pass
        self.is_speaking = False

    def shutdown(self) -> None:
        self._stopping.set()
        self.stop()
        self.speech_queue.put(None)
        self.worker_thread.join(timeout=2.5)
