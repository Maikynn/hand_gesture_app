#!/usr/bin/env python3
"""Queued Jarvis-style neural TTS with a reliable offline fallback."""

from __future__ import annotations

import asyncio
import ctypes
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import wave
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyttsx3

# ONNX Runtime must be loaded before Qt on Windows. Loading it lazily after
# PyQt5 can fail in DLL initialization even though Piper itself is installed.
try:
    from piper import PiperVoice, SynthesisConfig

    PIPER_AVAILABLE = True
    PIPER_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on optional native runtime
    PiperVoice = SynthesisConfig = None  # type: ignore[assignment]
    PIPER_AVAILABLE = False
    PIPER_IMPORT_ERROR = exc

from utils.config_store import PROJECT_ROOT

NEURAL_VOICES = [
    ("ru-RU-DmitryNeural", "Дмитрий · мужской, глубокий"),
    ("ru-RU-SvetlanaNeural", "Светлана · женский, мягкий"),
]
DEFAULT_NEURAL_VOICE = "ru-RU-DmitryNeural"
PIPER_VOICES = [
    ("ru_RU-denis-medium", "Денис · мужской нейронный, офлайн"),
]
DEFAULT_PIPER_VOICE = "ru_RU-denis-medium"
SILERO_VOICES = [
    ("eugene", "Eugene · глубокий мужской, JARVIS"),
    ("aidar", "Aidar · спокойный мужской"),
    ("xenia", "Xenia · выразительный женский"),
    ("baya", "Baya · мягкий женский"),
    ("kseniya", "Kseniya · нейтральный женский"),
]
DEFAULT_SILERO_VOICE = "eugene"


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
    fallback_engine: str = "system"
    profile: str = "calm"


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

    ``silero`` is the default high-quality Russian local neural voice. Piper is
    a lightweight local option, ``edge`` is online, and ``system`` is explicit
    SAPI. A selected neural voice is never silently replaced with SAPI.
    """

    def __init__(
        self,
        *,
        engine: str = "silero",
        voice_id: Optional[str] = None,
        rate: int = 175,
        volume: float = 0.9,
        pitch: int = -12,
        fallback_engine: str = "none",
        profile: str = "calm",
        on_event: Optional[Callable[[str, str], None]] = None,
    ):
        self.engine_name = (
            engine
            if engine in {"silero", "piper", "xtts", "edge", "system"}
            else "edge"
        )
        self.voice_id = voice_id or (
            DEFAULT_SILERO_VOICE
            if self.engine_name == "silero"
            else (
                DEFAULT_PIPER_VOICE
                if self.engine_name == "piper"
                else ("personal" if self.engine_name == "xtts" else DEFAULT_NEURAL_VOICE)
            )
        )
        self.rate = max(90, min(260, int(rate)))
        self.volume = max(0.0, min(1.0, float(volume)))
        self.pitch = max(-50, min(50, int(pitch)))
        self.fallback_engine = fallback_engine if fallback_engine in {"none", "system"} else "none"
        self.profile = profile if profile in {"calm", "strict", "emotional"} else "calm"
        self.is_speaking = False

        self.speech_queue: "queue.Queue[Optional[SpeechRequest]]" = queue.Queue()
        self._state_lock = threading.RLock()
        self._generation = 0
        self._mci_alias = ""
        self._edge_retry_after = 0.0
        self._piper_voice = None
        self._piper_voice_id = ""
        self._silero_model = None
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
            if request.engine == "silero":
                try:
                    self._speak_silero(request)
                    self._notify("done", "Silero Neural воспроизведён локально")
                    return
                except SpeechCancelled:
                    self._notify("cancelled", "Озвучка остановлена")
                    return
                except Exception as exc:
                    if request.fallback_engine != "system":
                        raise RuntimeError(
                            f"Silero Neural недоступен: {exc}. "
                            "Системный голос не подменял выбранный."
                        ) from exc
                    self._notify(
                        "fallback",
                        f"Silero недоступен — явно включён резерв Windows: {exc}",
                    )
            if request.engine == "xtts":
                try:
                    self._speak_xtts(request)
                    self._notify("done", "Персональный XTTS воспроизведён локально")
                    return
                except SpeechCancelled:
                    self._notify("cancelled", "Озвучка остановлена")
                    return
                except Exception as exc:
                    if request.fallback_engine != "system":
                        raise RuntimeError(
                            f"XTTS недоступен: {exc}. "
                            "Системный голос не подменял выбранный."
                        ) from exc
                    self._notify(
                        "fallback",
                        f"XTTS недоступен — явно включён резерв Windows: {exc}",
                    )
            if request.engine == "piper":
                try:
                    self._speak_piper(request)
                    self._notify("done", "Локальный Neural-голос воспроизведён")
                    return
                except SpeechCancelled:
                    self._notify("cancelled", "Озвучка остановлена")
                    return
                except Exception as exc:
                    if request.fallback_engine != "system":
                        raise RuntimeError(
                            f"Локальный Neural недоступен: {exc}. "
                            "Системный голос не подменял выбранный."
                        ) from exc
                    self._notify(
                        "fallback",
                        f"Piper недоступен — явно включён резерв Windows: {exc}",
                    )
            if request.engine == "edge":
                if (
                    time.monotonic() < self._edge_retry_after
                    and request.fallback_engine != "system"
                ):
                    raise RuntimeError(
                        "Онлайн Neural временно недоступен; резерв Windows выключен"
                    )
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
                        if request.fallback_engine != "system":
                            raise RuntimeError(
                                f"Онлайн Neural недоступен: {exc}. "
                                "Системный голос не подменял выбранный."
                            ) from exc
                        self._notify("fallback", "Neural недоступен — включён резерв Windows")
            self._speak_offline(request)
            self._notify("done", "Голос воспроизведён")
        except SpeechCancelled:
            self._notify("cancelled", "Озвучка остановлена")
        except Exception as exc:
            print(f"Speech error: {exc}")
            self._notify("error", f"Ошибка озвучки: {exc}")
        finally:
            self.is_speaking = False

    def _speak_silero(self, request: SpeechRequest) -> None:
        import torch

        model_path = PROJECT_ROOT / "models" / "silero" / "v5_5_ru.pt"
        if not model_path.is_file():
            raise RuntimeError("модель v5_5_ru не установлена; запустите setup_venv.bat")
        if self._silero_model is None:
            self._notify("loading", "Загружаю Silero v5.5 · русский Neural…")
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
            self._silero_model = torch.package.PackageImporter(model_path).load_pickle(
                "tts_models", "model"
            )
            self._silero_model.to(torch.device("cpu"))
        if self._is_cancelled(request.generation):
            raise SpeechCancelled
        speaker = (
            request.voice_id
            if request.voice_id in {voice_id for voice_id, _ in SILERO_VOICES}
            else DEFAULT_SILERO_VOICE
        )
        audio = self._silero_model.apply_tts(
            text=request.text,
            speaker=speaker,
            sample_rate=48000,
        )
        samples = audio.detach().cpu().numpy().astype(np.float32)
        speed = max(0.86, min(1.16, request.rate / 175.0))
        if abs(speed - 1.0) > 0.01 and samples.size > 16:
            source = np.arange(samples.size, dtype=np.float32)
            target = np.linspace(
                0,
                samples.size - 1,
                max(16, round(samples.size / speed)),
            )
            samples = np.interp(target, source, samples).astype(np.float32)
        samples = np.clip(samples * request.volume, -1.0, 1.0)
        if self._is_cancelled(request.generation):
            raise SpeechCancelled
        self._notify("playing", f"Silero v5.5 · {speaker} · полностью офлайн")
        self._play_audio(samples, 48000, request.generation)

    def _speak_xtts(self, request: SpeechRequest) -> None:
        personal_dir = PROJECT_ROOT / "user_data" / "personal_voice"
        reference = personal_dir / "reference.wav"
        consent = personal_dir / "xtts_consent"
        if not reference.is_file():
            raise RuntimeError(
                "референс не записан; используй студию персонального голоса"
            )
        if not consent.is_file():
            raise RuntimeError(
                "нет подтверждённого согласия с условиями XTTS и локальной обработкой"
            )
        environment_dir = PROJECT_ROOT / "models" / "xtts_env"
        python_path = (
            environment_dir / "Scripts" / "python.exe"
            if os.name == "nt"
            else environment_dir / "bin" / "python"
        )
        if not python_path.is_file():
            raise RuntimeError(
                "изолированный XTTS-компонент не установлен; нажми «Установить XTTS»"
            )
        worker = Path(__file__).with_name("xtts_worker.py")
        text_handle, text_name = tempfile.mkstemp(prefix="axi-xtts-", suffix=".txt")
        os.close(text_handle)
        output_handle, output_name = tempfile.mkstemp(prefix="axi-xtts-", suffix=".wav")
        os.close(output_handle)
        text_path = Path(text_name)
        output_path = Path(output_name)
        output_path.unlink(missing_ok=True)
        text_path.write_text(request.text, encoding="utf-8")
        env = os.environ.copy()
        env["TTS_HOME"] = str(PROJECT_ROOT / "models" / "xtts_cache")
        # This variable is only set after the user explicitly accepts the
        # displayed XTTS model terms and a local consent marker exists.
        env["COQUI_TOS_AGREED"] = "1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [
            str(python_path),
            str(worker),
            "--text-file",
            str(text_path),
            "--reference",
            str(reference),
            "--output",
            str(output_path),
            "--language",
            "ru",
        ]
        self._notify("loading", "XTTS v2 · синтез в изолированном процессе…")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=flags,
        )
        try:
            started = time.monotonic()
            while process.poll() is None:
                if self._is_cancelled(request.generation):
                    process.terminate()
                    raise SpeechCancelled
                if time.monotonic() - started > 900:
                    process.terminate()
                    raise RuntimeError("синтез XTTS превысил лимит 15 минут")
                time.sleep(0.1)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                details = (stderr or stdout or "неизвестная ошибка").strip()
                raise RuntimeError(details[-500:])
            if not output_path.is_file() or output_path.stat().st_size < 128:
                raise RuntimeError("XTTS не создал звуковой файл")
            if self._is_cancelled(request.generation):
                raise SpeechCancelled
            self._notify("playing", "XTTS v2 · персональный локальный голос")
            self._play_wav(output_path, request.generation)
        finally:
            if process.poll() is None:
                process.kill()
            text_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def _play_audio(
        self, audio: np.ndarray, sample_rate: int, generation: int
    ) -> None:
        import sounddevice as sd

        sd.play(audio, sample_rate, blocking=False)
        try:
            while bool(sd.get_stream().active):
                if self._is_cancelled(generation):
                    sd.stop()
                    raise SpeechCancelled
                time.sleep(0.04)
        finally:
            if self._is_cancelled(generation):
                sd.stop()

    def _speak_piper(self, request: SpeechRequest) -> None:
        if not PIPER_AVAILABLE or PiperVoice is None or SynthesisConfig is None:
            raise RuntimeError(f"Piper не загрузился: {PIPER_IMPORT_ERROR}")

        voice_id = request.voice_id or DEFAULT_PIPER_VOICE
        model_path = PROJECT_ROOT / "models" / "piper" / f"{voice_id}.onnx"
        config_path = model_path.with_suffix(".onnx.json")
        if not model_path.is_file() or not config_path.is_file():
            raise RuntimeError(
                f"модель {voice_id} не установлена; запустите setup_venv.bat"
            )
        if self._piper_voice is None or self._piper_voice_id != voice_id:
            self._notify("loading", f"Загружаю локальный голос {voice_id}…")
            self._piper_voice = PiperVoice.load(model_path, config_path)
            self._piper_voice_id = voice_id
        if self._is_cancelled(request.generation):
            raise SpeechCancelled
        handle, raw_path = tempfile.mkstemp(prefix="axi-piper-", suffix=".wav")
        os.close(handle)
        path = Path(raw_path)
        try:
            length_scale = max(0.62, min(1.65, 175.0 / max(1, request.rate)))
            noise_scale, noise_w_scale = {
                "strict": (0.38, 0.48),
                "calm": (0.55, 0.72),
                "emotional": (0.82, 0.92),
            }.get(request.profile, (0.55, 0.72))
            config = SynthesisConfig(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w_scale,
                volume=request.volume,
                normalize_audio=True,
            )
            with wave.open(str(path), "wb") as output:
                self._piper_voice.synthesize_wav(request.text, output, config)
            if self._is_cancelled(request.generation):
                raise SpeechCancelled
            self._notify("playing", "Воспроизвожу Piper · полностью офлайн")
            self._play_wav(path, request.generation)
        finally:
            path.unlink(missing_ok=True)

    def _play_wav(self, path: Path, generation: int) -> None:
        import sounddevice as sd

        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        if sample_width != 2:
            raise RuntimeError(f"Неподдерживаемая разрядность WAV: {sample_width * 8}")
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, sample_rate, blocking=False)
        try:
            while bool(sd.get_stream().active):
                if self._is_cancelled(generation):
                    sd.stop()
                    raise SpeechCancelled
                time.sleep(0.04)
        finally:
            if self._is_cancelled(generation):
                sd.stop()

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
                fallback_engine=self.fallback_engine,
                profile=self.profile,
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
        fallback_engine: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> None:
        new_engine = (
            engine
            if engine in {"silero", "piper", "xtts", "edge", "system"}
            else "edge"
        )
        default_voice = {
            "silero": DEFAULT_SILERO_VOICE,
            "piper": DEFAULT_PIPER_VOICE,
            "xtts": "personal",
        }.get(new_engine, DEFAULT_NEURAL_VOICE)
        new_voice = voice_id or default_voice
        with self._state_lock:
            provider_changed = (
                new_engine != self.engine_name or new_voice != self.voice_id
            )
            self.engine_name = new_engine
            self.voice_id = new_voice
            self.rate = max(90, min(260, int(rate)))
            self.volume = max(0.0, min(1.0, float(volume)))
            self.pitch = max(-50, min(50, int(pitch)))
            if fallback_engine is not None:
                self.fallback_engine = (
                    fallback_engine if fallback_engine in {"none", "system"} else "none"
                )
            if profile is not None:
                self.profile = (
                    profile if profile in {"calm", "strict", "emotional"} else "calm"
                )
            if provider_changed:
                self._edge_retry_after = 0.0

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))

    def set_rate(self, rate: int) -> None:
        self.rate = max(90, min(260, int(rate)))

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id

    def set_engine(self, engine: str) -> None:
        self.engine_name = (
            engine
            if engine in {"silero", "piper", "xtts", "edge", "system"}
            else "edge"
        )

    def set_pitch(self, pitch: int) -> None:
        self.pitch = max(-50, min(50, int(pitch)))

    def get_voices(self) -> List[Tuple[str, str]]:
        return list_voices()

    def speak_stream(self, text: str) -> None:
        """Queue sentence-sized chunks for earlier, interruptible playback."""
        chunks = [
            part.strip()
            for part in re.split(r"(?<=[.!?…])\s+|\n+", str(text).strip())
            if part.strip()
        ]
        for chunk in chunks or [str(text).strip()]:
            self.speak(chunk)

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
        with suppress(Exception):
            import sounddevice as sd

            sd.stop()
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
