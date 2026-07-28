#!/usr/bin/env python3
"""
Wake-word detection and speech recognition for the Hand Gesture Application.
Uses Vosk (offline) or Whisper (bond005/whisper-podlodka-turbo) for speech
recognition and wake-word detection.
"""

import os
import json
import threading
import time
from typing import Optional, Callable

# Optional speech backends are detected independently.  A missing Vosk model
# must still be able to fall back to SpeechRecognition.
try:
    from vosk import Model, KaldiRecognizer

    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    sr = None
    SR_AVAILABLE = False


class WakeWordListener:
    """
    Listens for wake word and transcribes speech.
    """

    def __init__(
        self,
        wake_word: str = "аксиос",
        model_path: str = "models/vosk-model-ru",
        stt_engine: str = "vosk",
        whisper_model: str = "bond005/whisper-podlodka-turbo",
        mic_index: Optional[int] = None,
        gain: float = 1.0,
    ):
        self.wake_word = wake_word.lower()
        self.model_path = model_path
        self.stt_engine = stt_engine.lower()
        self.whisper_model = whisper_model
        self.mic_index = mic_index
        self.gain = max(1.0, float(gain))
        self.is_listening = False
        self.callback = None
        self.thread = None
        self.continuous_mode = (
            False  # If True, process all speech; if False, only after wake word
        )
        self._whisper_stream = None

        if self.stt_engine == "whisper":
            # Whisper stream is created lazily in _listen_whisper.
            pass
        elif VOSK_AVAILABLE:
            self.setup_vosk()
        else:
            self.setup_speech_recognition()

    def setup_vosk(self):
        """Setup Vosk model for offline recognition."""
        if not os.path.exists(self.model_path):
            print(f"Vosk model not found at {self.model_path}")
            print("Please download from https://alphacephei.com/vosk/models")
            self.vosk_model = None
            self.setup_speech_recognition()
        else:
            self.vosk_model = Model(self.model_path)

    def setup_speech_recognition(self):
        """Setup speech_recognition as fallback."""
        if not SR_AVAILABLE:
            self.recognizer = None
            self.microphone = None
            return
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone(device_index=self.mic_index)
        except Exception:
            self.microphone = None

    def start_listening(
        self, callback: Callable[[str], None], continuous: bool = False
    ):
        """
        Start listening for wake word and speech.

        Args:
            callback: Function to call with transcribed text (command after wake word)
            continuous: If True, process all speech; if False, only after wake word
        """
        if self.is_listening:
            return
        self.is_listening = True
        self.callback = callback
        self.continuous_mode = continuous

        if self.stt_engine == "whisper":
            self.thread = threading.Thread(target=self._listen_whisper)
        elif VOSK_AVAILABLE and self.vosk_model:
            self.thread = threading.Thread(target=self._listen_vosk)
        elif (
            getattr(self, "recognizer", None) is not None
            and getattr(self, "microphone", None) is not None
        ):
            self.thread = threading.Thread(target=self._listen_sr)
        else:
            self.is_listening = False
            raise RuntimeError("Не найден доступный модуль распознавания речи")

        self.thread.daemon = True
        self.thread.start()

    def stop_listening(self):
        """Stop listening."""
        self.is_listening = False
        if self._whisper_stream is not None:
            try:
                self._whisper_stream.stop()
            except Exception:
                pass
            self._whisper_stream = None
        if self.thread:
            self.thread.join(timeout=1.0)

    def _listen_vosk(self):
        """Listen using Vosk."""
        import pyaudio

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000,
        )
        stream.start_stream()

        rec = KaldiRecognizer(self.vosk_model, 16000)

        while self.is_listening:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()
                if text:
                    self._process_text(text)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _listen_whisper(self):
        """Listen using the Whisper STT engine (bond005/whisper-podlodka-turbo)."""
        from assistant.whisper_stt import WhisperStream, whisper_available

        if not whisper_available():
            print("[WakeWordListener] Whisper unavailable, falling back to Vosk")
            if VOSK_AVAILABLE and getattr(self, "vosk_model", None):
                self._listen_vosk()
            elif (getattr(self, "recognizer", None) is not None
                  and getattr(self, "microphone", None) is not None):
                self._listen_sr()
            else:
                self.is_listening = False
            return

        def on_final(text: str):
            lowered = text.lower().strip()
            if lowered:
                self._process_text(lowered)

        try:
            self._whisper_stream = WhisperStream(
                model_name=self.whisper_model,
                mic_index=self.mic_index,
                gain=self.gain,
                on_partial=None,
                on_final=on_final,
            )
            self._whisper_stream.start()
            while self.is_listening:
                time.sleep(0.2)
        except Exception as e:
            print(f"[WakeWordListener] Whisper error: {e}")
        finally:
            if self._whisper_stream is not None:
                try:
                    self._whisper_stream.stop()
                except Exception:
                    pass
                self._whisper_stream = None

    def _listen_sr(self):
        """Listen using speech_recognition."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1.0)
                text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                if text:
                    self._process_text(text)
            except Exception:
                continue

    def _process_text(self, text: str):
        """
        Process transcribed text for wake word.

        Args:
            text: Lowercased transcribed text
        """
        if self.wake_word in text:
            # Extract command after wake word
            idx = text.find(self.wake_word)
            command = text[idx + len(self.wake_word) :].strip()
            # Only call callback if there's a command or in continuous mode
            if command:
                if self.callback:
                    self.callback(command)
            else:
                # Wake word detected but no command - in continuous mode, ignore
                # Otherwise, we could prompt for command, but for now just ignore
                pass
        elif self.continuous_mode and self.callback:
            # Continuous mode: process all speech
            self.callback(text)

    def test_microphone(self) -> bool:
        """Test if microphone is available."""
        try:
            if VOSK_AVAILABLE:
                import pyaudio

                p = pyaudio.PyAudio()
                p.terminate()
                return True
            else:
                with self.microphone:
                    return True
        except Exception:
            return False


if __name__ == "__main__":
    # Example usage
    listener = WakeWordListener()

    def on_speech(text):
        print(f"Heard: {text}")

    listener.start_listening(on_speech)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop_listening()
