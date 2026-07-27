#!/usr/bin/env python3
"""
Assistant Window for the Hand Gesture Application.
Voice-controlled AI assistant named Axi with OpenRouter integration.

TTS / STT engines are selected from config.json (voice.tts_engine /
voice.stt_engine). Silero TTS v5 and Whisper (bond005/whisper-podlodka-turbo)
are used when configured and available, with automatic fallback to the
offline pyttsx3 / Vosk engines.
"""

import threading
import time
import subprocess
import webbrowser
import os
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QSlider, QLineEdit, QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QFont

from assistant.openrouter_client import OpenRouterClient
from assistant.tts_engine import TTSEngine
from assistant.silero_tts import SileroTTSEngine, available as silero_available
from assistant.whisper_stt import whisper_available
from assistant.listener import WakeWordListener
from utils.logger import setup_logger


class AssistantSignals(QObject):
    """Signals for thread-safe UI updates."""
    transcription_received = pyqtSignal(str)
    response_received = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    volume_changed = pyqtSignal(int)


class AssistantWindow(QWidget):
    """
    Voice assistant interface with wake-word detection, OpenRouter, and TTS.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.logger = setup_logger("AssistantWindow")

        self.config = self._load_config()
        voice = self.config.get("voice", {})

        # Initialize components
        self.openrouter = OpenRouterClient()
        self.tts = self._build_tts(voice)
        self.listener = WakeWordListener(
            wake_word=voice.get("wake_word", "аксиос"),
            stt_engine=voice.get("stt_engine", "vosk"),
            whisper_model=voice.get("whisper_model", "bond005/whisper-podlodka-turbo"),
            mic_index=voice.get("mic_index"),
            gain=voice.get("mic_gain", 10.0),
        )

        # State
        self.is_listening = False
        self.api_key = voice.get("openrouter_key", "")
        self.system_prompt = voice.get(
            "system_prompt",
            "Ты — голосовой помощник Акси. Отвечай кратко и по делу на русском."
        )

        # Signals for thread-safe UI updates
        self.signals = AssistantSignals()
        self.signals.transcription_received.connect(self.add_transcription)
        self.signals.response_received.connect(self.add_response)
        self.signals.status_changed.connect(self.update_status)
        self.signals.volume_changed.connect(self.update_volume_slider)

        self.init_ui()

    # ----------------------------- config / engines -----------------------------
    def _load_config(self) -> dict:
        """Load config.json from the project root (hand_gesture_app)."""
        try:
            root = Path(__file__).resolve().parent.parent.parent
            cfg_path = root / "config.json"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[AssistantWindow] config load error: {e}")
        return {}

    def _build_tts(self, voice: dict):
        """Build the TTS engine from config, falling back to pyttsx3."""
        engine = (voice.get("tts_engine") or "pyttsx3").lower()
        speaker = voice.get("tts_speaker") or "xenia"
        rate = int(voice.get("tts_rate", 196))
        volume = float(voice.get("volume", 0.9))
        if engine == "silero" and silero_available():
            try:
                print("[AssistantWindow] using Silero TTS v5")
                return SileroTTSEngine(speaker=speaker, rate=rate, volume=volume)
            except Exception as e:
                print(f"[AssistantWindow] Silero TTS failed, falling back to pyttsx3: {e}")
        print("[AssistantWindow] using pyttsx3 TTS")
        return TTSEngine(rate=rate, volume=volume)

    def init_ui(self):
        """Initialize the assistant UI."""
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("Axi - Voice Assistant")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label)

        # Engine status
        tts_name = "Silero TTS v5" if isinstance(self.tts, SileroTTSEngine) else "pyttsx3"
        stt_name = "Whisper" if self.listener.stt_engine == "whisper" else "Vosk"
        self.engine_label = QLabel(f"TTS: {tts_name}   |   STT: {stt_name}")
        self.engine_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.engine_label)

        # Status indicator
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        # Wake-word indicator
        self.wake_word_label = QLabel("Wake word: 'аксиос'")
        self.wake_word_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.wake_word_label)

        # API Key input
        api_group = QGroupBox("OpenRouter Settings")
        api_layout = QVBoxLayout()
        api_key_layout = QHBoxLayout()
        api_key_label = QLabel("API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter OpenRouter API key")
        self.api_key_input.textChanged.connect(self.on_api_key_changed)
        if self.api_key:
            self.api_key_input.setText(self.api_key)
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(self.api_key_input)
        api_layout.addLayout(api_key_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["free", "gpt-3.5-turbo", "claude-3-haiku", "gemini-pro"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        api_layout.addLayout(model_layout)

        api_group.setLayout(api_layout)
        self.layout.addWidget(api_group)

        # Transcription display
        transcription_group = QGroupBox("Transcription Log")
        transcription_layout = QVBoxLayout()
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setMaximumHeight(120)
        transcription_layout.addWidget(self.transcription_text)
        transcription_group.setLayout(transcription_layout)
        self.layout.addWidget(transcription_group)

        # Response display
        response_group = QGroupBox("Assistant Response")
        response_layout = QVBoxLayout()
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setMaximumHeight(120)
        response_layout.addWidget(self.response_text)
        response_group.setLayout(response_layout)
        self.layout.addWidget(response_group)

        # Volume control
        volume_group = QGroupBox("Volume")
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self.config.get("voice", {}).get("volume", 0.9) * 100))
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_group.setLayout(volume_layout)
        self.layout.addWidget(volume_group)

        # Command input
        command_group = QGroupBox("Quick Command / Question")
        command_layout = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Type command or ask question...")
        self.command_input.returnPressed.connect(self.on_send_command)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_command)
        command_layout.addWidget(self.command_input)
        command_layout.addWidget(self.send_button)
        command_group.setLayout(command_layout)
        self.layout.addWidget(command_group)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Listening")
        self.start_button.clicked.connect(self.toggle_listening)
        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(self.clear_log)
        self.test_tts_button = QPushButton("Test TTS")
        self.test_tts_button.clicked.connect(self.test_tts)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.test_tts_button)
        self.layout.addLayout(control_layout)

        self.setLayout(self.layout)

    def apply_settings(self, settings: dict):
        """Apply settings from settings window."""
        # Could apply volume, etc. from settings
        if 'volume' in settings:
            self.volume_slider.setValue(int(settings['volume'] * 100))
        elif self.config.get("voice", {}).get("volume") is not None:
            self.volume_slider.setValue(int(self.config["voice"]["volume"] * 100))

    def on_api_key_changed(self, text: str):
        """Handle API key change."""
        self.api_key = text
        self.openrouter.set_api_key(text)

    def on_model_changed(self, model: str):
        """Handle model selection change."""
        self.openrouter.set_model(model)

    def on_volume_changed(self, value: int):
        """Handle volume slider change."""
        self.tts.set_volume(value / 100.0)

    def update_volume_slider(self, value: int):
        """Update volume slider from signal."""
        self.volume_slider.setValue(value)

    def toggle_listening(self):
        """Toggle voice listening state."""
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.start_button.setText("Stop Listening")
            self.signals.status_changed.emit("Status: Listening for 'аксиос'...")
            self.listener.start_listening(self.on_wake_word_detected)
        else:
            self.start_button.setText("Start Listening")
            self.signals.status_changed.emit("Status: Idle")
            self.listener.stop_listening()

    def on_wake_word_detected(self, command: str):
        """Handle wake word detection with command."""
        if not command:
            # Wake word detected but no command - just acknowledge
            self.signals.transcription_received.emit("[Wake word detected]")
            return

        self.signals.transcription_received.emit(command)
        self.process_command(command)

    def on_send_command(self):
        """Send typed command to assistant."""
        command = self.command_input.text()
        if command:
            self.signals.transcription_received.emit(command)
            self.process_command(command)
            self.command_input.clear()

    def process_command(self, command: str):
        """Process user command."""
        self.signals.status_changed.emit("Status: Processing...")

        # Run in background thread
        thread = threading.Thread(target=self._process_command_thread, args=(command,))
        thread.daemon = True
        thread.start()

    def _process_command_thread(self, command: str):
        """Process command in background thread."""
        try:
            command_lower = command.lower()

            # Check for system commands
            if any(cmd in command_lower for cmd in ["открой", "open", "запусти", "run"]):
                self.execute_system_command(command)
            elif "открой ссылку" in command_lower or "open link" in command_lower:
                self.open_url(command)
            else:
                # Send to OpenRouter for general questions
                if not self.api_key:
                    self.signals.response_received.emit(
                        "Error: OpenRouter API key not set. Please enter it in the settings above."
                    )
                    self.signals.status_changed.emit("Status: Idle")
                    return

                self.signals.response_received.emit("Thinking...")
                response = self.openrouter.send_message(command, self.system_prompt)
                self.signals.response_received.emit(response)
                # Speak response
                self.tts.speak(response)

        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            self.signals.response_received.emit(f"Error: {str(e)}")
        finally:
            self.signals.status_changed.emit("Status: Listening for 'аксиос'..." if self.is_listening else "Status: Idle")

    def execute_system_command(self, command: str):
        """Execute system command (open file, run app)."""
        # Extract target from command
        parts = command.split()
        target = None
        for i, part in enumerate(parts):
            if part.lower() in ["открой", "open", "запусти", "run"]:
                if i + 1 < len(parts):
                    target = " ".join(parts[i + 1:])  # Rest of command as target
                    break

        if target:
            self.signals.response_received.emit(f"Opening {target}...")
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(target)
                else:  # Linux/Mac
                    subprocess.Popen(['xdg-open', target])
                self.signals.response_received.emit(f"Opened {target}")
            except Exception as e:
                self.signals.response_received.emit(f"Failed to open {target}: {e}")
        else:
            self.signals.response_received.emit("Please specify what to open")

    def open_url(self, command: str):
        """Open URL from command."""
        parts = command.split()
        url = None
        for i, part in enumerate(parts):
            if part.lower() in ["ссылку", "link", "url"]:
                if i + 1 < len(parts):
                    url = parts[i + 1]
                    break

        if not url:
            # Try to find URL in command
            for part in parts:
                if part.startswith(('http://', 'https://', 'www.')):
                    url = part
                    break

        if url:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            self.signals.response_received.emit(f"Opening {url}...")
            try:
                webbrowser.open(url)
                self.signals.response_received.emit(f"Opened {url}")
            except Exception as e:
                self.signals.response_received.emit(f"Failed to open URL: {e}")
        else:
            self.signals.response_received.emit("Please specify a URL to open")

    def add_transcription(self, text: str):
        """Add transcription to display."""
        timestamp = time.strftime("%H:%M:%S")
        self.transcription_text.append(f"[{timestamp}] {text}")

    def add_response(self, text: str):
        """Add response to display."""
        timestamp = time.strftime("%H:%M:%S")
        self.response_text.append(f"[{timestamp}] {text}")

    def update_status(self, text: str):
        """Update status label."""
        self.status_label.setText(text)

    def clear_log(self):
        """Clear transcription and response logs."""
        self.transcription_text.clear()
        self.response_text.clear()

    def test_tts(self):
        """Test TTS engine."""
        self.tts.speak("Привет! Я Акси, ваш голосовой помощник. Тест синтеза речи работает.")

    def cleanup(self):
        """Cleanup resources."""
        self.listener.stop_listening()
        self.tts.shutdown()
