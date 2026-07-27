#!/usr/bin/env python3
"""
TTS Engine for the Axi assistant.

Offline text-to-speech via pyttsx3 (SAPI5 on Windows).

A SINGLE pyttsx3 engine is created (lazily, on the worker thread) and reused
for every utterance. This avoids two well-known SAPI5/pyttsx3 pitfalls:

  * Creating a fresh engine per utterance eventually raises
    "run loop already started". Each pyttsx3.init() advises its own SAPI5
    event sink on the thread; the COM message loop is left in a "running"
    state, so the next engine's runAndWait() collides with it. Reusing one
    engine per thread eliminates the collision.

  * The legacy "speaks only once" bug is not present in current pyttsx3, so a
    persistent engine driven with say()+runAndWait() per utterance is reliable.

All engine access happens on the worker thread (the engine is created and
driven there), so there are no cross-thread COM calls.
"""
import pyttsx3
import threading
import queue
import time
from typing import List, Optional, Tuple


def list_voices() -> List[Tuple[str, str]]:
    """Return [(voice_id, name), ...] for every installed TTS voice."""
    try:
        eng = pyttsx3.init()
        voices = eng.getProperty("voices") or []
        return [(v.id, v.name) for v in voices]
    except Exception:
        return []


class TTSEngine:
    def __init__(self, use_offline: bool = True, voice_id: Optional[str] = None,
                 rate: int = 160, volume: float = 0.9):
        self.use_offline = use_offline
        self.speech_queue: "queue.Queue" = queue.Queue()
        self.is_speaking = False
        self.volume = max(0.0, min(1.0, volume))
        self.rate = max(50, min(400, rate))
        self.voice_id = voice_id
        # One persistent engine, created lazily on the worker thread.
        self._engine = None
        self._engine_lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _ensure_engine(self):
        """Create the (single, persistent) pyttsx3 engine on this thread if
        needed, and apply the current voice/rate/volume settings."""
        if self._engine is None:
            eng = pyttsx3.init()
            self._apply_settings(eng)
            self._engine = eng
        return self._engine

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

    def _apply_settings(self, eng):
        try:
            eng.setProperty("rate", self.rate)
            eng.setProperty("volume", self.volume)
            vid = self.voice_id
            voices = eng.getProperty("voices") or []
            if not vid:
                for v in voices:
                    if "russian" in v.name.lower() or "ru" in v.id.lower():
                        vid = v.id
                        break
                if not vid and voices:
                    vid = voices[0].id
            if vid:
                eng.setProperty("voice", vid)
                self.voice_id = vid
        except Exception as e:
            print("TTS settings error:", e)

    def _speak(self, text: str):
        if not text or not text.strip():
            return
        self.is_speaking = True
        try:
            with self._engine_lock:
                eng = self._ensure_engine()
                # Apply live settings right before speaking (worker thread
                # only -> no cross-thread COM calls).
                self._apply_settings(eng)
                eng.say(text)
                eng.runAndWait()
        except Exception as e:
            print("Speech error:", e)
            # The engine's COM loop may be in a bad state; drop it so the next
            # utterance rebuilds a fresh one (single recovery, not a loop).
            try:
                with self._engine_lock:
                    self._engine = None
            except Exception:
                pass
        finally:
            self.is_speaking = False

    def speak(self, text: str):
        if text and text.strip():
            self.speech_queue.put(text.strip())

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))

    def set_rate(self, rate: int):
        self.rate = max(50, min(400, rate))

    def set_voice(self, voice_id: str):
        self.voice_id = voice_id

    def get_voices(self):
        try:
            return pyttsx3.init().getProperty("voices")
        except Exception:
            return []

    def stop(self):
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        self.is_speaking = False

    def shutdown(self):
        self.speech_queue.put(None)
        self.worker_thread.join(timeout=2.0)


if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Привет! Я Акси, ваш голосовой помощник.")
    time.sleep(2)
    tts.shutdown()
