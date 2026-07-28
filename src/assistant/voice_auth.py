from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "user_data" / "speaker_profile.json"


def voice_signature(samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size < sample_rate // 3:
        raise ValueError("Слишком короткая запись голоса")
    audio -= float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak < 80:
        raise ValueError("Голос почти не слышен")
    audio /= peak
    window_size = 1024
    hop = 512
    spectra = []
    for start in range(0, audio.size - window_size, hop):
        window = audio[start : start + window_size] * np.hanning(window_size)
        power = np.abs(np.fft.rfft(window)) ** 2
        spectra.append(power)
    if not spectra:
        raise ValueError("Недостаточно данных для голосового профиля")
    mean_power = np.mean(spectra, axis=0)
    frequencies = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
    bands = np.geomspace(80, 7600, 33)
    features = []
    for low, high in zip(bands[:-1], bands[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        band = mean_power[mask]
        features.append(
            math.log1p(float(np.mean(band))) if band.size else 0.0
        )
    vector = np.asarray(features, dtype=np.float32)
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError("Не удалось выделить голосовые признаки")
    return vector / norm


class VoiceAuthenticator:
    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        self.profile: Optional[np.ndarray] = None
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            vector = np.asarray(data.get("signature", []), dtype=np.float32)
            self.profile = vector if vector.size == 32 else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.profile = None

    @property
    def enrolled(self) -> bool:
        return self.profile is not None

    def enroll(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        self.profile = voice_signature(samples, sample_rate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "version": 1,
                    "sample_rate": sample_rate,
                    "signature": self.profile.tolist(),
                }
            ),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def score(self, samples: np.ndarray, sample_rate: int = 16000) -> float:
        if self.profile is None:
            return 0.0
        candidate = voice_signature(samples, sample_rate)
        cosine = float(np.dot(self.profile, candidate))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    @staticmethod
    def capture(
        mic_index: Optional[int] = None, duration: float = 3.5
    ) -> np.ndarray:
        import pyaudio

        audio = pyaudio.PyAudio()
        stream = None
        chunks = []
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=4000,
            )
            for _ in range(max(2, round(duration * 16000 / 4000))):
                chunks.append(stream.read(4000, exception_on_overflow=False))
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()
        return np.frombuffer(b"".join(chunks), dtype=np.int16)
