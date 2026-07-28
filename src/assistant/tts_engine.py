#!/usr/bin/env python3
"""Queued Jarvis-style neural TTS with a reliable offline fallback."""

from __future__ import annotations

import asyncio
import ctypes
import os
import queue
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pyttsx3


NEURAL_VOICES = [
    ("ru-RU-DmitryNeural", "Дмитрий · мужской, глубокий"),
    ("ru-RU-SvetlanaNeural", "Светлана · женский, мягкий"),
]
DEFAULT_NEURAL_VOICE = "ru-RU-DmitryNeural"


class SpeechCancelled(Exception):
    """Internal control-flow exception for an interrupted utterance."""


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    generation: int
    engine: str
    voice_id: str
    rate: int
    volume: float
    pitch: int


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
        on_event: Optional[Callable[[str, str], None]] = None,
    ):
        self.engine_name = engine if engine in {"edge", "system"} else "edge"
        self.voice_id = voice_id or DEFAULT_NEURAL_VOICE
        self.rate = max(90, min(260, int(rate)))
        self.volume = max(0.0, min(1.0, float(volume)))
        self.pitch = max(-50, min(50, int(pitch)))
        self.is_speaking = False

        self.speech_queue: "queue.Queue[Optional[SpeechRequest]]" = queue.Queue()
        self._state_lock = threading.RLock()
        self._generation = 0
        self._mci_alias = ""
        self._edge_retry_after = 0.0
        self._on_event = on_event
        self._stopping = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._process_queue,
            name="jarvis-tts",
            daemon=True,
        )
        self.worker_thread.start()

    def _process_queue(self) -> None:
        while True:
            item = self.speech_queue.get()
            try:
                if item is None:
                    return
                self._speak(item)
            finally:
                self.speech_queue.task_done()

    def _speak(self, request: SpeechRequest) -> None:
        if not request.text.strip() or self._is_cancelled(request.generation):
            return
        self.is_speaking = True
        self._notify("synthesizing", "Готовлю голос…")
        try:
            if request.engine == "edge":
                if time.monotonic() >= self._edge_retry_after:
                    try:
                        self._speak_edge(request)
                        self._edge_retry_after = 0.0
                        self._notify("done", "Голос воспроизведён")
                        return
                    except SpeechCancelled:
                        self._notify("cancelled", "Озвучка остановлена")
                        return
                    except Exception as exc:
                        # Do not make every reply wait on the same unavailable
                        # online service. Retry after five minutes or settings save.
                        self._edge_retry_after = time.monotonic() + 300.0
                        print(f"Neural TTS unavailable, using Windows voice: {exc}")
                        self._notify(
                            "fallback",
                            "Neural недоступен — использую голос Windows",
                        )
            self._speak_offline(request)
            self._notify("done", "Голос воспроизведён")
        except SpeechCancelled:
            self._notify("cancelled", "Озвучка остановлена")
        except Exception as exc:
            print(f"Speech error: {exc}")
            self._notify("error", f"Ошибка озвучки: {exc}")
        finally:
            self.is_speaking = False

    def _speak_edge(self, request: SpeechRequest) -> None:
        import edge_tts

        rate_percent = round((request.rate - 175) / 1.75)
        volume_percent = round((request.volume - 1.0) * 100)
        communicator = edge_tts.Communicate(
            request.text,
            request.voice_id or DEFAULT_NEURAL_VOICE,
            rate=f"{rate_percent:+d}%",
            volume=f"{volume_percent:+d}%",
            pitch=f"{request.pitch:+d}Hz",
            connect_timeout=5,
            receive_timeout=12,
        )
        handle, raw_path = tempfile.mkstemp(prefix="axi-jarvis-", suffix=".mp3")
        os.close(handle)
        path = Path(raw_path)
        try:
            asyncio.run(
                self._save_edge_with_cancel(
                    communicator, path, request.generation, timeout=15.0
                )
            )
            if not path.is_file() or path.stat().st_size < 256:
                raise RuntimeError("Сервис не вернул аудио")
            self._notify("playing", "Воспроизвожу Neural-голос")
            self._play_mp3(path, request.generation)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    async def _save_edge_with_cancel(
        self, communicator, path: Path, generation: int, timeout: float
    ) -> None:
        task = asyncio.create_task(communicator.save(str(path)))
        deadline = time.monotonic() + timeout
        try:
            while not task.done():
                if self._is_cancelled(generation):
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    raise SpeechCancelled
                if time.monotonic() >= deadline:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    raise TimeoutError("Neural TTS не ответил за 15 секунд")
                await asyncio.sleep(0.08)
            await task
        finally:
            if not task.done():
                task.cancel()

    def _play_mp3(self, path: Path, generation: int) -> None:
        if os.name != "nt":
            raise RuntimeError("Neural playback is currently configured for Windows")
        alias = f"axi_jarvis_{threading.get_ident()}"
        with self._state_lock:
            self._mci_alias = alias
        try:
            self._mci(f'open "{path}" type mpegvideo alias {alias}')
            self._mci(f"play {alias}")
            while True:
                if self._is_cancelled(generation):
                    self._mci(f"stop {alias}", check=False)
                    raise SpeechCancelled
                mode = self._mci(f"status {alias} mode", result_chars=64).lower()
                if mode not in {"playing", "seeking", "paused"}:
                    break
                time.sleep(0.06)
        finally:
            try:
                self._mci(f"close {alias}", check=False)
            finally:
                with self._state_lock:
                    self._mci_alias = ""

    @staticmethod
    def _mci(
        command: str, *, check: bool = True, result_chars: int = 0
    ) -> str:
        winmm = ctypes.windll.winmm
        buffer = ctypes.create_unicode_buffer(result_chars) if result_chars else None
        code = int(
            winmm.mciSendStringW(
                command,
                buffer,
                result_chars,
                None,
            )
        )
        if code and check:
            error_buffer = ctypes.create_unicode_buffer(256)
            winmm.mciGetErrorStringW(code, error_buffer, len(error_buffer))
            raise RuntimeError(error_buffer.value or f"MCI error {code}")
        return buffer.value if buffer is not None else ""

    def _speak_offline(self, request: SpeechRequest) -> None:
        """Use a fresh SAPI COM object on the worker thread for every phrase."""
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        voice = None
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            tokens = voice.GetVoices()
            selected = None
            for index in range(tokens.Count):
                token = tokens.Item(index)
                token_id = str(token.Id)
                description = str(token.GetDescription())
                if request.voice_id and token_id.casefold() == request.voice_id.casefold():
                    selected = token
                    break
                if selected is None and (
                    "russian" in description.casefold()
                    or "ирина" in description.casefold()
                    or "ru-ru" in token_id.casefold()
                ):
                    selected = token
            if selected is not None:
                voice.Voice = selected
            voice.Rate = max(-10, min(10, round((request.rate - 175) / 17)))
            voice.Volume = max(0, min(100, round(request.volume * 100)))
            self._notify("playing", "Воспроизвожу голос Windows")
            voice.Speak(request.text, 1)  # SVSFlagsAsync
            while not voice.WaitUntilDone(80):
                if self._is_cancelled(request.generation):
                    voice.Speak("", 3)  # async + purge before speak
                    raise SpeechCancelled
        finally:
            voice = None
            pythoncom.CoUninitialize()

    def speak(self, text: str) -> None:
        value = str(text).strip()
        if not value or self._stopping.is_set():
            return
        with self._state_lock:
            request = SpeechRequest(
                text=value,
                generation=self._generation,
                engine=self.engine_name,
                voice_id=self.voice_id,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch,
            )
        self.speech_queue.put(request)
        self._notify("queued", "Фраза добавлена в очередь")

    def configure(
        self,
        *,
        engine: str,
        voice_id: str,
        rate: int,
        volume: float,
        pitch: int,
    ) -> None:
        new_engine = engine if engine in {"edge", "system"} else "edge"
        new_voice = voice_id or DEFAULT_NEURAL_VOICE
        with self._state_lock:
            provider_changed = (
                new_engine != self.engine_name or new_voice != self.voice_id
            )
            self.engine_name = new_engine
            self.voice_id = new_voice
            self.rate = max(90, min(260, int(rate)))
            self.volume = max(0.0, min(1.0, float(volume)))
            self.pitch = max(-50, min(50, int(pitch)))
            if provider_changed:
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
        with self._state_lock:
            self._generation += 1
            alias = self._mci_alias
        sentinel_seen = False
        while True:
            try:
                item = self.speech_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.speech_queue.task_done()
                if item is None:
                    sentinel_seen = True
        if sentinel_seen:
            self.speech_queue.put(None)
        if alias and os.name == "nt":
            self._mci(f"stop {alias}", check=False)
        self.is_speaking = False

    def shutdown(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        self.stop()
        self.speech_queue.put(None)
        self.worker_thread.join(timeout=4.0)

    def set_event_callback(
        self, callback: Optional[Callable[[str, str], None]]
    ) -> None:
        self._on_event = callback

    def _is_cancelled(self, generation: int) -> bool:
        with self._state_lock:
            return self._stopping.is_set() or generation != self._generation

    def _notify(self, state: str, message: str) -> None:
        callback = self._on_event
        if callback is None:
            return
        try:
            callback(state, message)
        except Exception:
            pass
