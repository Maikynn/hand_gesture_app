#!/usr/bin/env python3
"""
Silero TTS v5 engine for the Axi assistant.

Drop-in replacement for the pyttsx3-based :class:`TTSEngine`. It exposes the
same public surface (``speak``, ``set_volume``, ``set_rate``, ``set_voice``,
``stop``, ``shutdown``, ``is_speaking``, ``get_voices``/``list_voices``) so the
UI can swap engines without changes.

The model is loaded lazily on the worker thread (first ``speak`` call) so that
importing this module never fails if torch / the model is unavailable. If the
model cannot be loaded, :meth:`available` returns ``False`` and the caller can
fall back to the offline pyttsx3 engine.

Silero TTS v5 (Russian) is loaded via ``torch.hub`` (or the ``silero`` package)
and produces a float waveform that we play with ``sounddevice`` (falling back to
PyAudio / winsound if needed). Speech rate is mapped to a playback-speed factor
and volume to a waveform gain.
"""
import os
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np


# Valid Silero TTS v5 Russian speaker voices.
V5_RU_SPEAKERS = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
DEFAULT_SPEAKER = 'xenia'  # pleasant female Russian voice (matches prior TTS)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def available() -> bool:
    """Whether Silero TTS can be used in this environment (torch present)."""
    return _torch_available()


def list_voices() -> List[Tuple[str, str]]:
    """Return [(speaker_id, label), ...] for every Silero v5 Russian voice."""
    return [(s, s) for s in V5_RU_SPEAKERS]


class SileroTTSEngine:
    """Neural TTS engine backed by Silero TTS v5 (Russian)."""

    def __init__(self, speaker: str = DEFAULT_SPEAKER,
                 sample_rate: int = 48000,
                 device: Optional[str] = None,
                 rate: int = 196, volume: float = 0.9,
                 language: str = 'ru'):
        self.speaker = speaker if speaker in V5_RU_SPEAKERS else DEFAULT_SPEAKER
        self.sample_rate = int(sample_rate)
        self.language = language
        # Speech rate -> speed factor (196 is the app default, treated as 1.0).
        self.rate = max(50, min(400, int(rate)))
        self.volume = max(0.0, min(1.0, float(volume)))
        self.device = device or ("cuda" if self._cuda_ok() else "cpu")

        self.is_speaking = False
        self._model = None
        self._model_lock = threading.Lock()
        self._sd_stream = None  # current sounddevice output stream

        self.speech_queue: "queue.Queue" = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    @staticmethod
    def _cuda_ok() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    # ----------------------------- model loading -----------------------------
    def _ensure_model(self):
        """Load the Silero v5 model on this (worker) thread if needed."""
        if self._model is not None:
            return self._model
        import torch

        def _unpack(res):
            """Normalise torch.hub / silero-package return values to a model."""
            if isinstance(res, tuple):
                # Typical torch.hub shape: (example_text, model) or
                # (example_text, model, available_speakers, sample_rate).
                model = res[1] if len(res) > 1 else res[0]
                if len(res) > 3 and isinstance(res[3], int) and res[3] > 0:
                    self.sample_rate = res[3]
                return model
            return res

        model = None
        # Preferred: torch.hub (no extra package required).
        try:
            res = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=self.language,
                speaker='v5_ru',
                trust_repo=True,
            )
            model = _unpack(res)
        except Exception as e:  # pragma: no cover - network / offline
            print(f"[SileroTTS] torch.hub load failed: {e}")
            # Fallback: the `silero` pip package (newer API).
            try:
                from silero import silero_tts  # type: ignore
                res = silero_tts(language=self.language, speaker='v5_ru')
                model = _unpack(res)
            except Exception as e2:
                raise RuntimeError(f"Silero TTS unavailable: {e2}") from e2

        if model is None:
            raise RuntimeError("Silero TTS model could not be loaded.")

        model = model.to(self.device)
        model.eval()
        self._model = model
        print(f"[SileroTTS] model loaded (speaker={self.speaker}, sr={self.sample_rate})")
        return self._model

    # ----------------------------- worker loop -----------------------------
    def _process_queue(self):
        while True:
            try:
                item = self.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            self._speak(item)
            self.speech_queue.task_done()

    def _speak(self, text: str):
        if not text or not text.strip():
            return
        self.is_speaking = True
        try:
            model = self._ensure_model()
            with self._model_lock:
                try:
                    out = model.apply_tts(
                        texts=[text], speaker=self.speaker,
                        sample_rate=self.sample_rate,
                    )
                except TypeError:
                    out = model.apply_tts(
                        text=text, speaker=self.speaker,
                        sample_rate=self.sample_rate,
                    )
            audio = out[0] if isinstance(out, (list, tuple)) else out
            audio = np.asarray(audio, dtype=np.float32)
            audio = self._apply_speed(audio)
            audio = np.clip(audio * self.volume, -1.0, 1.0)
            self._play(audio)
        except Exception as e:
            print("Silero TTS speech error:", e)
        finally:
            self.is_speaking = False

    # ----------------------------- audio helpers -----------------------------
    def _apply_speed(self, audio: np.ndarray) -> np.ndarray:
        """Resample the waveform to realize the configured speech rate.

        rate 196 -> speed 1.0. Higher rate => faster (shorter) audio.
        Simple linear interpolation (changes pitch, acceptable for TTS).
        """
        speed = self.rate / 196.0
        if abs(speed - 1.0) < 0.02:
            return audio
        n = audio.shape[0]
        new_n = max(1, int(round(n / speed)))
        idx = np.linspace(0, n - 1, new_n)
        return np.interp(idx, np.arange(n), audio).astype(np.float32)

    def _play(self, audio: np.ndarray):
        """Play a float32 [-1,1] waveform using the best available backend."""
        # 1) sounddevice (installed, simplest, non-blocking capable)
        try:
            import sounddevice as sd
            self._sd_stream = sd.play(audio, self.sample_rate)
            sd.wait()
            self._sd_stream = None
            return
        except Exception as e:
            print(f"[SileroTTS] sounddevice playback failed, trying PyAudio: {e}")
        # 2) PyAudio (always available via vosk dependency)
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1,
                            rate=self.sample_rate, output=True)
            data = audio.astype(np.float32).tobytes()
            chunk = 4096
            for i in range(0, len(data), chunk):
                stream.write(data[i:i + chunk])
            stream.stop_stream()
            stream.close()
            p.terminate()
            return
        except Exception as e:
            print(f"[SileroTTS] PyAudio playback failed, trying winsound: {e}")
        # 3) winsound (Windows built-in, needs a WAV file)
        try:
            import wave
            import tempfile
            import winsound  # type: ignore
            pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                path = tf.name
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm.tobytes())
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            try:
                os.remove(path)
            except Exception:
                pass
        except Exception as e:
            print(f"[SileroTTS] all playback backends failed: {e}")

    # ----------------------------- public API -----------------------------
    def speak(self, text: str):
        if text and text.strip():
            self.speech_queue.put(text.strip())

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, float(volume)))

    def set_rate(self, rate: int):
        self.rate = max(50, min(400, int(rate)))

    def set_voice(self, speaker: str):
        if speaker in V5_RU_SPEAKERS:
            self.speaker = speaker

    def get_voices(self) -> List[Tuple[str, str]]:
        return list_voices()

    def stop(self):
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self.is_speaking = False

    def shutdown(self):
        self.speech_queue.put(None)
        self.worker_thread.join(timeout=2.0)


if __name__ == "__main__":
    if not available():
        print("torch not available - cannot run Silero TTS")
    else:
        tts = SileroTTSEngine()
        tts.speak("Привет! Я Акси, ваш голосовой помощник. Тест нейросетевого синтеза речи.")
        time.sleep(4)
        tts.shutdown()
