#!/usr/bin/env python3
"""
Whisper STT engine for the Axi assistant.

Provides :class:`WhisperStream` - a streaming-style recognizer built on
``faster-whisper`` using the Russian model ``bond005/whisper-podlodka-turbo``.
It mirrors the public surface of :class:`VoskStream` (``on_partial`` /
``on_final`` callbacks, ``read_chunk``-style level reporting, ``start`` /
``stop``, ``list_microphones``) so it can be dropped into the existing
wake-word listener.

Whisper is not a true streaming model, so we accumulate microphone audio,
detect speech with a lightweight energy/VAD gate, and transcribe the buffered
utterance once a silence gap is detected. The recognized text is delivered via
``on_final``; ``on_partial`` receives a short "listening" indicator while audio
is being captured.
"""
import struct
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np


try:
    from faster_whisper import WhisperModel
    _WHISPER_OK = True
    _WHISPER_ERR = None
except Exception as _e:  # pragma: no cover
    WhisperModel = None
    _WHISPER_OK = False
    _WHISPER_ERR = _e

try:
    import pyaudio
    _PYAUDIO_OK = True
except Exception:  # pragma: no cover
    pyaudio = None
    _PYAUDIO_OK = False


# Russian podcast/turbo model from HuggingFace (user-requested).
DEFAULT_MODEL = "bond005/whisper-podlodka-turbo"


def whisper_available() -> bool:
    return _WHISPER_OK


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


class WhisperStream:
    """Continuous Whisper recognition over a chosen microphone.

    Audio is buffered until a silence gap ends the utterance, then transcribed
    with ``faster-whisper``. ``on_final(text)`` is called with the result.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 model_path: Optional[str] = None,
                 mic_index: Optional[int] = None,
                 sample_rate: int = 16000,
                 gain: float = 1.0,
                 on_partial: Optional[Callable[[str], None]] = None,
                 on_final: Optional[Callable[[str], None]] = None,
                 device: str = "cpu",
                 compute_type: str = "int8",
                 vad_threshold: float = 0.012,
                 silence_gap_sec: float = 0.6,
                 max_utterance_sec: float = 20.0):
        if not _WHISPER_OK:
            raise RuntimeError(f"faster-whisper недоступен: {_WHISPER_ERR}")
        if not _PYAUDIO_OK:
            raise RuntimeError("PyAudio недоступен")
        self.model_name = model_name
        self.model_path = model_path
        self.mic_index = mic_index
        self.sample_rate = sample_rate
        self.gain = max(1.0, float(gain))
        self.on_partial = on_partial
        self.on_final = on_final
        self.device = device
        self.compute_type = compute_type
        self.vad_threshold = vad_threshold
        self.silence_gap_sec = silence_gap_sec
        self.max_utterance_sec = max_utterance_sec

        print(f"[WhisperSTT] loading model {model_path or model_name} ({device}/{compute_type}) ...")
        self.model = WhisperModel(
            model_path or model_name, device=device, compute_type=compute_type
        )
        print("[WhisperSTT] model loaded")

        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer: List[bytes] = []
        self._speaking = False
        self._silence_reads = 0
        self._reads_per_gap = max(1, int(silence_gap_sec * sample_rate / 4000))
        self._max_reads = max(1, int(max_utterance_sec * sample_rate / 4000))

    def start(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=4000,
        )
        self.stream.start_stream()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        reads = 0
        while self.running:
            try:
                data = self.stream.read(4000, exception_on_overflow=False)
            except Exception:
                break
            data = self._apply_gain(data)
            level = self._level(data)
            if level > self.vad_threshold:
                if not self._speaking and self.on_partial:
                    self.on_partial("слушаю…")
                self._speaking = True
                self._silence_reads = 0
                self._buffer.append(data)
                reads += 1
                if reads >= self._max_reads:
                    self._flush()
                    reads = 0
            else:
                if self._speaking:
                    self._silence_reads += 1
                    if self._silence_reads >= self._reads_per_gap:
                        self._flush()
                        reads = 0
                else:
                    # drain a little so we don't build up latency while idle
                    pass

    def _flush(self):
        if not self._buffer:
            self._speaking = False
            return
        audio = np.frombuffer(b"".join(self._buffer), dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = []
        self._speaking = False
        try:
            segments, _ = self.model.transcribe(
                audio, language="ru", beam_size=5, vad_filter=True
            )
            text = " ".join(getattr(s, "text", "") for s in segments).strip()
        except Exception as e:
            print(f"[WhisperSTT] transcribe error: {e}")
            text = ""
        if text and self.on_final:
            self.on_final(text)

    def _apply_gain(self, data: bytes) -> bytes:
        """Automatic gain control (same approach as VoskStream)."""
        if not data or len(data) < 2:
            return data
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(arr * arr)))
        if rms < 50.0:
            return data
        target = 0.22 * 32768.0
        scale = min(self.gain, target / rms)
        if scale <= 1.0:
            return data
        arr = arr * scale
        arr = 32767.0 * np.tanh(arr / 32767.0)
        return arr.astype(np.int16).tobytes()

    @staticmethod
    def _level(data: bytes) -> float:
        if not data or len(data) < 2:
            return 0.0
        count = len(data) // 2
        shorts = struct.unpack(f"{count}h", data[:count * 2])
        rms = (sum(s * s for s in shorts) / max(1, count)) ** 0.5
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
        if self._thread is not None:
            self._thread.join(timeout=1.0)


if __name__ == "__main__":
    if not whisper_available():
        print("faster-whisper not available")
    else:
        def on_final(t):
            print("FINAL:", t)
        st = WhisperStream(on_final=on_final)
        st.start()
        try:
            while True:
                time_sleep = __import__("time").sleep
                time_sleep(1)
        except KeyboardInterrupt:
            st.stop()
