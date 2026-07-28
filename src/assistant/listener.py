from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# CTranslate2 is another native runtime that must be initialized before Qt on
# Windows. The actual model remains lazy; this only stabilizes DLL loading.
try:
    from faster_whisper import WhisperModel as _WhisperRuntime  # noqa: F401
except Exception:
    _WhisperRuntime = None  # type: ignore[assignment]

from utils.config_store import PROJECT_ROOT

try:
    from vosk import KaldiRecognizer, Model

    VOSK_AVAILABLE = True
except ImportError:
    KaldiRecognizer = Model = None  # type: ignore[assignment]
    VOSK_AVAILABLE = False

try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    sr = None  # type: ignore[assignment]
    SR_AVAILABLE = False


class WakeWordListener:
    """Single-threaded microphone listener with an armed wake-word state."""

    def __init__(
        self,
        wake_word: str = "джарвис, аксиос",
        model_path: str = "models/vosk-ru",
        stt_engine: str = "vosk",
        whisper_model: str = "",
        mic_index: Optional[int] = None,
        gain: float = 1.0,
    ):
        self.wake_word = wake_word
        self.wake_phrases = self._split_wake_phrases(wake_word)
        path = Path(model_path)
        self.model_path = path if path.is_absolute() else PROJECT_ROOT / path
        self.stt_engine = (stt_engine or "vosk").lower()
        self.whisper_model = whisper_model
        self.mic_index = mic_index
        self.gain = max(0.5, min(10.0, float(gain)))
        self.is_listening = False
        self.continuous_mode = False
        self.callback: Optional[Callable[[str], None]] = None
        self.status_callback: Optional[Callable[[str, str], None]] = None
        self.thread: Optional[threading.Thread] = None
        self._model = None
        self._whisper_stream = None
        self._armed_until = 0.0
        self._last_speech_status = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _split_wake_phrases(value: str) -> List[str]:
        value = value.replace("|", ",").replace(";", ",")
        return [part.strip().lower() for part in value.split(",") if part.strip()]

    @staticmethod
    def list_input_devices() -> List[Dict[str, object]]:
        devices: List[Dict[str, object]] = []
        try:
            import pyaudio

            audio = pyaudio.PyAudio()
            try:
                for index in range(audio.get_device_count()):
                    info = audio.get_device_info_by_index(index)
                    if int(info.get("maxInputChannels", 0)) > 0:
                        devices.append(
                            {"index": index, "name": str(info.get("name", f"Микрофон {index}"))}
                        )
            finally:
                audio.terminate()
        except Exception:
            pass
        return devices

    def configure(
        self,
        *,
        wake_word: str,
        model_path: str,
        stt_engine: str,
        mic_index: Optional[int],
        gain: float,
        whisper_model: Optional[str] = None,
    ) -> None:
        was_listening = self.is_listening
        callback = self.callback
        status_callback = self.status_callback
        continuous = self.continuous_mode
        if was_listening:
            self.stop_listening()
        self.wake_word = wake_word
        self.wake_phrases = self._split_wake_phrases(wake_word)
        path = Path(model_path)
        self.model_path = path if path.is_absolute() else PROJECT_ROOT / path
        self.stt_engine = (stt_engine or "vosk").lower()
        self.mic_index = mic_index
        self.gain = max(0.5, min(10.0, float(gain)))
        if whisper_model is not None:
            self.whisper_model = whisper_model
        self._model = None
        if was_listening and callback:
            self.start_listening(callback, continuous, status_callback)

    def start_listening(
        self,
        callback: Callable[[str], None],
        continuous: bool = False,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> bool:
        with self._lock:
            if self.thread and self.thread.is_alive():
                return False
            self.is_listening = True
            self.callback = callback
            self.continuous_mode = continuous
            self.status_callback = status_callback
            targets = {
                "vosk": self._listen_vosk,
                "whisper": self._listen_whisper,
                "google": self._listen_sr,
            }
            target = targets.get(self.stt_engine, self._listen_vosk)
            self.thread = threading.Thread(target=target, name="jarvis-listener", daemon=True)
            self.thread.start()
        self._status("listening", "Микрофон включён")
        return True

    def stop_listening(self) -> None:
        self.is_listening = False
        stream = self._whisper_stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            self._whisper_stream = None
        thread = self.thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self.thread = None
        self._status("idle", "Микрофон выключен")

    def _status(self, state: str, message: str) -> None:
        if self.status_callback:
            self.status_callback(state, message)

    def _load_vosk(self):
        if not VOSK_AVAILABLE or Model is None:
            raise RuntimeError("Пакет vosk не установлен")
        required = (
            self.model_path / "am" / "final.mdl",
            self.model_path / "conf" / "model.conf",
            self.model_path / "graph" / "HCLr.fst",
            self.model_path / "graph" / "Gr.fst",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"Модель Vosk неполная ({', '.join(missing)}): {self.model_path}"
            )
        if self._model is None:
            self._status("loading", "Загружаю офлайн-модель Vosk…")
            self._model = Model(str(self.model_path))
        return self._model

    def _listen_vosk(self) -> None:
        audio = None
        stream = None
        try:
            import pyaudio

            model = self._load_vosk()
            audio = pyaudio.PyAudio()
            kwargs = {
                "format": pyaudio.paInt16,
                "channels": 1,
                "rate": 16000,
                "input": True,
                "frames_per_buffer": 4000,
            }
            if self.mic_index is not None and int(self.mic_index) >= 0:
                kwargs["input_device_index"] = int(self.mic_index)
            stream = audio.open(**kwargs)
            recognizer = KaldiRecognizer(model, 16000)
            self._status("ready", "Vosk слушает локально")
            while self.is_listening:
                data = stream.read(4000, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if self.gain != 1.0:
                    samples = np.clip(samples * self.gain, -32768, 32767).astype(np.int16)
                    data = samples.tobytes()
                level_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                level = (
                    float(np.sqrt(np.mean(level_samples * level_samples)))
                    if level_samples.size
                    else 0.0
                )
                if level > 1200 and time.monotonic() - self._last_speech_status > 1.0:
                    self._last_speech_status = time.monotonic()
                    self._status("speech", "Слышу речь — озвучка прервана")
                if recognizer.AcceptWaveform(data):
                    text = str(json.loads(recognizer.Result()).get("text", "")).strip()
                    if text:
                        self.process_recognized_text(text)
        except Exception as exc:
            self._status("error", str(exc))
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if audio is not None:
                audio.terminate()
            self.is_listening = False

    def _listen_sr(self) -> None:
        if not SR_AVAILABLE or sr is None:
            self._status("error", "SpeechRecognition не установлен")
            self.is_listening = False
            return
        try:
            recognizer = sr.Recognizer()
            microphone = sr.Microphone(device_index=self.mic_index)
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
            self._status("ready", "Онлайн-распознавание слушает")
            while self.is_listening:
                try:
                    with microphone as source:
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=8)
                    text = recognizer.recognize_google(audio, language="ru-RU")
                    if text:
                        self.process_recognized_text(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as exc:
                    self._status("error", str(exc))
                    time.sleep(0.5)
        except Exception as exc:
            self._status("error", str(exc))
        finally:
            self.is_listening = False

    def _listen_whisper(self) -> None:
        try:
            from assistant.whisper_stt import DEFAULT_MODEL, WhisperStream

            self._status("loading", "Загружаю локальную модель Whisper…")
            model_name = self.whisper_model or DEFAULT_MODEL
            bundled_model = PROJECT_ROOT / "models" / "whisper" / model_name
            stream = WhisperStream(
                model_name=model_name,
                model_path=str(bundled_model) if bundled_model.is_dir() else None,
                mic_index=self.mic_index,
                gain=self.gain,
                device="cpu",
                compute_type="int8",
                on_partial=lambda _text: self._status(
                    "speech", "Слышу речь — озвучка прервана"
                ),
                on_final=self.process_recognized_text,
            )
            self._whisper_stream = stream
            stream.start()
            self._status("ready", "Whisper слушает локально")
            while self.is_listening and stream.running:
                time.sleep(0.1)
        except Exception as exc:
            self._status("error", f"Whisper: {exc}")
        finally:
            stream = self._whisper_stream
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
            self._whisper_stream = None
            self.is_listening = False

    def process_recognized_text(self, text: str) -> Optional[str]:
        """Route one finalized transcript. Kept public for deterministic tests."""
        normalized = " ".join(text.lower().strip().split())
        if not normalized:
            return None
        now = time.monotonic()
        if self.continuous_mode:
            if self.callback:
                self.callback(normalized)
            return normalized
        for phrase in self.wake_phrases:
            position = normalized.find(phrase)
            if position >= 0:
                command = normalized[position + len(phrase) :].strip(" ,.!?-")
                self._armed_until = now + 8.0
                self._status("wake", f"Услышал «{phrase}»")
                if command and self.callback:
                    self._armed_until = 0.0
                    self.callback(command)
                return command
        if now <= self._armed_until:
            self._armed_until = 0.0
            if self.callback:
                self.callback(normalized)
            return normalized
        return None

    def test_microphone(self) -> bool:
        return bool(self.list_input_devices())
