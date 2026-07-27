#!/usr/bin/env python3
"""
Offline speech recognition via Vosk.

Provides:
  * vosk_available()            - whether the vosk package is importable
  * get_default_model_path()    - path to the best bundled Russian model
  * list_microphones()          - [(index, name), ...] of input devices
  * VoskStream                  - continuous streaming recognizer over a mic

Vosk runs fully offline and streams partial + final results, which makes the
wake-word detection far more reliable than a cloud "listen for N seconds" loop.

Input gain (AGC)
----------------
The optional digital `gain` (>= 1.0) is a CEILING for automatic gain control.
Quiet speech is boosted up to `gain` so a weak microphone is usable, but loud
speech is left untouched. This avoids the harsh clipping that a fixed multiplier
causes: clipping distorts the waveform and wrecks STT accuracy (e.g. the wake
word "аксиос" gets mangled into "такой"). Peaks are soft-limited with tanh so
transients never hard-clip.
"""
import os
import json
import struct
import numpy as np
from typing import Callable, List, Optional, Tuple

try:
    from vosk import Model, KaldiRecognizer
    _VOSK_OK = True
except Exception as _e:  # pragma: no cover
    Model = None
    KaldiRecognizer = None
    _VOSK_OK = False
    _VOSK_ERR = _e

try:
    import pyaudio
    _PYAUDIO_OK = True
except Exception:  # pragma: no cover
    pyaudio = None
    _PYAUDIO_OK = False


_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "models")
_DEFAULT_MODEL = os.path.join(_MODELS_DIR, "vosk-ru")

# Preference order for the bundled Russian model. The LARGE model
# (vosk-model-ru-0.22) is dramatically more accurate than the small one and is
# auto-selected here IF it has been downloaded into models/. Download:
#   https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip
_MODEL_CANDIDATES = [
    "vosk-model-ru-0.22",        # large, ~2.5 GB, best accuracy
    "vosk-ru",                   # small (currently bundled)
    "vosk-model-small-ru-0.22",  # small, alternative name
]


def vosk_available() -> bool:
    return _VOSK_OK


def get_default_model_path() -> str:
    """Return the path to the most accurate bundled Russian model that exists.

    Falls back to the small bundled model so the app always works out of the
    box; if the user drops the large model into models/ it is picked up
    automatically with no config change.
    """
    for name in _MODEL_CANDIDATES:
        p = os.path.join(_MODELS_DIR, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "am")):
            return p
    return _DEFAULT_MODEL


def list_microphones() -> List[Tuple[int, str]]:
    """Return [(index, name), ...] for every input device PyAudio can see."""
    if not _PYAUDIO_OK:
        return []
    p = pyaudio.PyAudio()
    devices: List[Tuple[int, str]] = []
    try:
        default_idx = p.get_default_input_device_info().get("index")
    except Exception:
        default_idx = None
    try:
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
            except Exception:
                continue
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            name = info.get("name", f"Устройство {i}")
            if i == default_idx:
                name += "  (по умолчанию)"
            devices.append((i, name))
    finally:
        p.terminate()
    return devices


class VoskStream:
    """Continuous Vosk recognition over a chosen microphone.

    on_partial(text) / on_final(text) are called from the reading thread.
    read_chunk() returns an RMS level in 0..1 for a level meter.
    """

    def __init__(self, model_path: Optional[str] = None,
                 mic_index: Optional[int] = None,
                 sample_rate: int = 16000,
                 gain: float = 1.0,
                 on_partial: Optional[Callable[[str], None]] = None,
                 on_final: Optional[Callable[[str], None]] = None):
        if not _VOSK_OK:
            raise RuntimeError(f"Vosk недоступен: {_VOSK_ERR}")
        if not _PYAUDIO_OK:
            raise RuntimeError("PyAudio недоступен")
        self.model_path = model_path or _DEFAULT_MODEL
        self.mic_index = mic_index
        self.sample_rate = sample_rate
        self.gain = max(1.0, float(gain))
        self.on_partial = on_partial
        self.on_final = on_final
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(False)
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.running = False

    def start(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=8000,
        )
        self.stream.start_stream()
        self.running = True

    def _apply_gain(self, data: bytes) -> bytes:
        """Automatic gain control.

        Boost quiet speech up to `self.gain` (so a quiet mic is usable) but
        never scale loud speech (which would clip and distort). Near-silence is
        left alone so background noise is not amplified. Peaks are soft-limited
        with tanh to avoid harsh clipping on transients.
        """
        if not data or len(data) < 2:
            return data
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(arr * arr)))
        if rms < 50.0:
            # Essentially silence: do not amplify noise.
            return data
        target = 0.22 * 32768.0  # comfortable speech level, well below clipping
        # Scale up to `gain`, but cap so we never push the average into clipping.
        scale = min(self.gain, target / rms)
        if scale <= 1.0:
            # Already loud enough: leave untouched (no distortion introduced).
            return data
        arr = arr * scale
        # Soft saturation instead of hard clipping.
        arr = 32767.0 * np.tanh(arr / 32767.0)
        return arr.astype(np.int16).tobytes()

    def read_chunk(self, size: int = 4000, feed: bool = True) -> float:
        """Read one audio chunk. Feed it to the recognizer unless feed=False
        (used to drain the buffer while the assistant is speaking)."""
        data = self.stream.read(size, exception_on_overflow=False)
        data = self._apply_gain(data)
        level = self._level(data)
        if not feed:
            return level
        if self.recognizer.AcceptWaveform(data):
            res = json.loads(self.recognizer.Result())
            text = res.get("text", "").strip()
            if text and self.on_final:
                self.on_final(text)
        else:
            if self.on_partial:
                part = json.loads(self.recognizer.PartialResult()).get("partial", "")
                self.on_partial(part)
        return level

    @staticmethod
    def _level(data: bytes) -> float:
        if not data or len(data) < 2:
            return 0.0
        count = len(data) // 2
        shorts = struct.unpack(f"{count}h", data[:count * 2])
        rms = (sum(s * s for s in shorts) / max(1, count)) ** 0.5
        # Scale for a usable meter (normal speech peaks well below 32768).
        return min(1.0, (rms / 32768.0) * 4.0)

    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass
        try:
            self.pa.terminate()
        except Exception:
            pass


def list_vosk_models() -> List[Tuple[str, str]]:
    """Return [(model_path, label), ...] for every usable Russian model found
    under models/. Includes the bundled small model and any large model the
    user has downloaded. The first entry is the auto-selected default.

    Used by the Settings UI so the user can pick which Vosk model to use.
    The LARGE model (vosk-model-ru-0.22) is far more accurate than the small
    one and is the recommended choice for better word recognition.
    """
    found = []
    if os.path.isdir(_MODELS_DIR):
        for name in _MODEL_CANDIDATES:
            p = os.path.join(_MODELS_DIR, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "am")):
                big = ("0.22" in name) and ("small" not in name)
                label = name + ("  (большая, точнее)" if big else "")
                found.append((p, label))
    if not found:
        found.append((_DEFAULT_MODEL, "vosk-ru (по умолчанию)"))
    return found
