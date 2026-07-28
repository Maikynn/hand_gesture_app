from __future__ import annotations

import html
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from assistant.actions import ACTION_LABELS, ActionExecutor
from assistant.document_index import DocumentIndex
from assistant.embedded_jarvis import AssistantResult, EmbeddedJarvis
from assistant.listener import WakeWordListener
from assistant.tts_engine import (
    NEURAL_VOICES,
    PIPER_VOICES,
    SILERO_VOICES,
    TTSEngine,
    list_voices,
)
from assistant.voice_auth import VoiceAuthenticator
from utils.config_store import ConfigStore


PROVIDERS = {
    "Не использовать": "none",
    "OpenRouter": "openrouter",
    "Ollama · локально": "ollama",
    "Другое OpenAI API": "custom",
}
PROVIDER_LABELS = {value: key for key, value in PROVIDERS.items()}


class AssistantSignals(QObject):
    result_ready = pyqtSignal(object)
    recognized = pyqtSignal(str)
    listener_status = pyqtSignal(str, str)
    tts_status = pyqtSignal(str, str)
    diagnostics_ready = pyqtSignal(str)
    stream_sentence = pyqtSignal(str)
    background_result = pyqtSignal(str, str)


class AssistantMiniHud(QWidget):
    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(430, 92)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 13, 18, 13)
        self.state = QLabel("● JARVIS // ONLINE")
        self.state.setObjectName("CoreStatus")
        self.message = QLabel("Ожидаю команду")
        self.message.setWordWrap(True)
        layout.addWidget(self.state)
        layout.addWidget(self.message)
        self.setStyleSheet(
            """
            QWidget { background: rgba(4, 17, 24, 228); border: 1px solid #39dff5;
                      color: #d9fbff; font-family: 'Cascadia Mono'; }
            QLabel { background: transparent; border: 0; }
            """
        )

    def show_message(self, state: str, message: str) -> None:
        self.state.setText(f"● JARVIS // {state.upper()}")
        self.message.setText(str(message)[:180])
        screen = self.screen().availableGeometry()
        self.move(screen.right() - self.width() - 24, screen.bottom() - self.height() - 24)
        if not self.isVisible():
            self.show()


class AssistantWindow(QWidget):
    """Minimal chat view with a separate advanced JARVIS Core page."""

    status_changed = pyqtSignal(str, str)

    def __init__(self, store: ConfigStore | None = None, parent=None):
        super().__init__(parent)
        self.store = store or ConfigStore()
        self.executor = ActionExecutor(
            self.store.get("permissions", []),
            scenarios=self.store.get("scenarios", []),
        )
        self.core = EmbeddedJarvis(self.store, executor=self.executor)
        assistant = self.store.get("assistant", {}) or {}
        self.signals = AssistantSignals()
        self.tts = TTSEngine(
            engine=str(assistant.get("tts_engine", "silero")),
            voice_id=str(assistant.get("tts_voice", "eugene")),
            rate=int(assistant.get("tts_rate", 175)),
            volume=float(assistant.get("tts_volume", 0.9)),
            pitch=int(assistant.get("tts_pitch", -12)),
            fallback_engine=str(assistant.get("tts_fallback_engine", "none")),
            profile=str(assistant.get("tts_profile", "calm")),
            on_event=self.signals.tts_status.emit,
        )
        self.listener = WakeWordListener(
            wake_word=str(assistant.get("wake_phrases", "джарвис, аксиос")),
            model_path=str(assistant.get("vosk_model_path", "models/vosk-ru")),
            stt_engine=str(assistant.get("stt_engine", "vosk")),
            mic_index=self._mic_index(assistant.get("microphone_index", -1)),
            gain=float(assistant.get("microphone_gain", 1.0)),
            whisper_model=str(assistant.get("whisper_model", "tiny")),
            noise_suppression=bool(assistant.get("noise_suppression", True)),
            echo_suppression=bool(assistant.get("echo_suppression", True)),
            language_mode=str(assistant.get("language_mode", "ru")),
        )
        self.voice_authenticator = VoiceAuthenticator()
        self.document_index = DocumentIndex(
            str(assistant.get("documents_folder", ""))
        )
        self.mini_hud = AssistantMiniHud()
        self.signals.result_ready.connect(self._finish_result)
        self.signals.recognized.connect(self._handle_recognized)
        self.signals.listener_status.connect(self._listener_status)
        self.signals.tts_status.connect(self._tts_status)
        self.signals.diagnostics_ready.connect(self._show_diagnostics)
        self.signals.stream_sentence.connect(self._stream_sentence)
        self.signals.background_result.connect(self._background_result)
        self._busy = False
        self._active_provider = str(assistant.get("llm_provider", "none"))
        self._draft_keys = {
            "openrouter": self.store.api_key("openrouter"),
            "custom": self.store.api_key("custom"),
        }
        self._build_ui()
        self.set_theme(str(self.store.get("ui.theme", "dark")))
        self._load_settings()
        self._append_message(
            "assistant",
            "Готов к работе. Напиши команду или включи микрофон. "
            "Все действия с приложениями выполняются только из списка разрешений.",
        )
        self.device_watch_timer = QTimer(self)
        self.device_watch_timer.timeout.connect(self._auto_switch_microphone)
        self.device_watch_timer.start(3500)
        if bool(assistant.get("overlay_enabled", False)):
            self.mini_hud.show_message("online", "Ожидаю команду")

    @staticmethod
    def _mic_index(value: Any) -> Optional[int]:
        try:
            index = int(value)
            return index if index >= 0 else None
        except (TypeError, ValueError):
            return None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("JARVIS // ASSISTANT")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Диалог, голосовые команды и Push-to-Talk"
        )
        subtitle.setObjectName("PageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.status = QLabel("Готов")
        self.status.setObjectName("StatusGood")
        header.addWidget(self.status)
        root.addLayout(header)

        root.addWidget(self._build_chat(), 1)
        self.advanced_settings_page = self._build_settings()

    def _card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        return card

    def _build_chat(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        top = QHBoxLayout()
        heading = QLabel("Чат")
        heading.setStyleSheet("font-size: 14pt; font-weight: 700;")
        top.addWidget(heading)
        top.addStretch(1)
        self.mic_button = QPushButton("Включить микрофон")
        self.mic_button.setCheckable(True)
        self.mic_button.clicked.connect(self._toggle_listening)
        clear = QPushButton("Очистить")
        clear.setProperty("secondary", True)
        clear.clicked.connect(lambda: self.chat.clear())
        top.addWidget(self.mic_button)
        top.addWidget(clear)
        layout.addLayout(top)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        self.chat.document().setDefaultStyleSheet(
            """
            .row{margin:8px 0}.who{font-size:10px;color:#8390a8;margin-bottom:3px}
            .user{background:#0b7182;color:white;padding:10px;border-radius:4px}
            .assistant{background:#0b202b;color:#dffcff;padding:10px;border-radius:4px;
                       border-left:2px solid #39dff5}
            .meta{color:#8fa0bb;font-size:10px;margin-top:3px}
            """
        )
        layout.addWidget(self.chat, 1)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Команда или вопрос…")
        self.input.returnPressed.connect(self._send)
        self.send_button = QPushButton("Отправить")
        self.send_button.clicked.connect(self._send)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send_button)
        layout.addLayout(input_row)
        hint = QLabel(
            "Пример: «Джарвис, открой блокнот» · обычные вопросы уйдут выбранной модели"
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return card

    def _build_settings(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(12)
        title = QLabel("JARVIS CORE // CONFIG")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Голос, нейросети, память, безопасность, сценарии и разрешения"
        )
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._voice_group())
        layout.addWidget(self._intelligence_group())
        layout.addWidget(self._llm_group())
        layout.addWidget(self._safety_group())
        layout.addWidget(self._memory_group())
        layout.addWidget(self._scenarios_group())
        layout.addWidget(self._diagnostics_group())
        layout.addWidget(self._permissions_group())
        layout.addWidget(self._commands_group())
        save = QPushButton("Сохранить и применить всё")
        save.clicked.connect(self._save_all)
        layout.addWidget(save)
        layout.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def set_theme(self, theme: str) -> None:
        if (theme or "dark").lower() == "dark":
            css = """
            .row{margin:8px 0}.who{font-size:10px;color:#8390a8;margin-bottom:3px}
            .user{background:#0b7182;color:white;padding:10px;border-radius:4px}
            .assistant{background:#0b202b;color:#dffcff;padding:10px;border-radius:4px;
                       border-left:2px solid #39dff5}
            .meta{color:#8fa0bb;font-size:10px;margin-top:3px}
            """
        else:
            css = """
            .row{margin:8px 0}.who{font-size:10px;color:#64748b;margin-bottom:3px}
            .user{background:#087d8e;color:white;padding:10px;border-radius:4px}
            .assistant{background:#e4f4f5;color:#12343b;padding:10px;border-radius:4px;
                       border-left:2px solid #1593a4}
            .meta{color:#64748b;font-size:10px;margin-top:3px}
            """
        self.chat.document().setDefaultStyleSheet(css)

    def _voice_group(self) -> QGroupBox:
        group = QGroupBox("Голос и пробуждение")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.wake_phrases = QLineEdit()
        self.wake_phrases.setPlaceholderText("джарвис, аксиос")
        self.wake_phrases.setToolTip("Несколько фраз разделяются запятой")
        form.addRow("Wake-фразы", self.wake_phrases)

        self.stt_engine = QComboBox()
        self.stt_engine.addItem("Vosk · офлайн", "vosk")
        self.stt_engine.addItem("Whisper · нейросеть офлайн", "whisper")
        self.stt_engine.addItem("Google · онлайн", "google")
        form.addRow("Распознавание", self.stt_engine)

        self.whisper_model = QComboBox()
        self.whisper_model.addItem("Tiny · быстрее", "tiny")
        self.whisper_model.addItem("Base · точнее", "base")
        self.whisper_model.addItem("Small · максимум качества", "small")
        form.addRow("Модель Whisper", self.whisper_model)

        mic_row = QHBoxLayout()
        self.microphone = QComboBox()
        refresh = QPushButton("↻")
        refresh.setFixedWidth(42)
        refresh.setProperty("secondary", True)
        refresh.setToolTip("Обновить список микрофонов")
        refresh.clicked.connect(self._refresh_microphones)
        mic_row.addWidget(self.microphone, 1)
        mic_row.addWidget(refresh)
        form.addRow("Микрофон", mic_row)

        self.mic_gain = QDoubleSpinBox()
        self.mic_gain.setRange(0.5, 10.0)
        self.mic_gain.setSingleStep(0.25)
        self.mic_gain.setSuffix("×")
        form.addRow("Усиление", self.mic_gain)

        self.continuous = QCheckBox("Обрабатывать речь без wake-фразы")
        form.addRow("", self.continuous)
        self.tts_enabled = QCheckBox("Озвучивать ответы Jarvis")
        form.addRow("", self.tts_enabled)

        self.tts_engine = QComboBox()
        self.tts_engine.addItem("Silero v5.5 · лучший русский офлайн", "silero")
        self.tts_engine.addItem("Piper Neural · красивый офлайн-голос", "piper")
        self.tts_engine.addItem("XTTS v2 · персональный голос", "xtts")
        self.tts_engine.addItem("Edge Neural · красивый онлайн-голос", "edge")
        self.tts_engine.addItem("Windows · системный офлайн-голос", "system")
        self.tts_engine.currentIndexChanged.connect(self._tts_engine_changed)
        form.addRow("Голосовой движок", self.tts_engine)

        voice_row = QHBoxLayout()
        self.tts_voice = QComboBox()
        for voice_id, label in SILERO_VOICES:
            self.tts_voice.addItem(label, voice_id)
        self.test_voice = QPushButton("▶ Проверить")
        self.test_voice.setProperty("secondary", True)
        self.test_voice.clicked.connect(self._test_voice)
        voice_row.addWidget(self.tts_voice, 1)
        voice_row.addWidget(self.test_voice)
        form.addRow("Голос", voice_row)

        self.tts_fallback = QComboBox()
        self.tts_fallback.addItem("Не подменять выбранный голос", "none")
        self.tts_fallback.addItem("Windows, только если Neural сломался", "system")
        form.addRow("Резервный голос", self.tts_fallback)

        self.tts_profile = QComboBox()
        self.tts_profile.addItem("Спокойный", "calm")
        self.tts_profile.addItem("Строгий", "strict")
        self.tts_profile.addItem("Эмоциональный", "emotional")
        self.tts_profile.currentIndexChanged.connect(self._apply_voice_profile)
        form.addRow("Профиль речи", self.tts_profile)

        self.tts_rate = QSlider(Qt.Horizontal)
        self.tts_rate.setRange(90, 260)
        form.addRow("Скорость речи", self.tts_rate)
        self.tts_pitch = QSlider(Qt.Horizontal)
        self.tts_pitch.setRange(-50, 50)
        self.tts_pitch.setToolTip("Отрицательное значение делает голос глубже")
        form.addRow("Глубина голоса", self.tts_pitch)
        self.tts_volume = QSlider(Qt.Horizontal)
        self.tts_volume.setRange(0, 100)
        form.addRow("Громкость TTS", self.tts_volume)
        return group

    def _intelligence_group(self) -> QGroupBox:
        group = QGroupBox("JARVIS Intelligence // звук, безопасность и локальные данные")
        layout = QVBoxLayout(group)

        audio = QHBoxLayout()
        self.noise_suppression = QCheckBox("Шумоподавление")
        self.echo_suppression = QCheckBox("Подавление эха")
        self.auto_mic_switch = QCheckBox("Автопереключение микрофона")
        self.language_mode = QComboBox()
        self.language_mode.addItem("Только русский", "ru")
        self.language_mode.addItem("Русский + English автоматически", "auto")
        audio.addWidget(self.noise_suppression)
        audio.addWidget(self.echo_suppression)
        audio.addWidget(self.auto_mic_switch)
        audio.addStretch(1)
        audio.addWidget(self.language_mode)
        layout.addLayout(audio)

        security = QHBoxLayout()
        self.speaker_auth = QCheckBox("Проверять владельца для голосовых команд")
        enroll = QPushButton("Записать голос владельца")
        enroll.setProperty("secondary", True)
        enroll.clicked.connect(self._enroll_voice)
        verify = QPushButton("Проверить голос")
        verify.setProperty("secondary", True)
        verify.clicked.connect(self._verify_voice)
        self.speaker_status = QLabel(
            "Профиль голоса записан"
            if self.voice_authenticator.enrolled
            else "Профиль голоса не записан"
        )
        self.speaker_status.setProperty("muted", True)
        security.addWidget(self.speaker_auth)
        security.addWidget(enroll)
        security.addWidget(verify)
        security.addWidget(self.speaker_status, 1)
        layout.addLayout(security)

        documents = QHBoxLayout()
        self.documents_enabled = QCheckBox("Локальный поиск по документам")
        self.documents_folder = QLineEdit()
        self.documents_folder.setPlaceholderText("Папка с TXT, Markdown, DOCX и PDF")
        choose_documents = QPushButton("Папка…")
        choose_documents.setProperty("secondary", True)
        choose_documents.clicked.connect(self._choose_documents_folder)
        index_documents = QPushButton("Переиндексировать")
        index_documents.setProperty("secondary", True)
        index_documents.clicked.connect(self._reindex_documents)
        self.documents_status = QLabel("Индекс не запускался")
        self.documents_status.setProperty("muted", True)
        documents.addWidget(self.documents_enabled)
        documents.addWidget(self.documents_folder, 1)
        documents.addWidget(choose_documents)
        documents.addWidget(index_documents)
        documents.addWidget(self.documents_status)
        layout.addLayout(documents)

        behavior = QHBoxLayout()
        self.overlay_enabled = QCheckBox("Мини-HUD поверх программ")
        self.overlay_enabled.toggled.connect(self._toggle_overlay)
        self.dry_run_commands = QCheckBox("SIMULATION: не выполнять команды")
        self.adaptive_prosody = QCheckBox("Автоинтонация ответа")
        behavior.addWidget(self.overlay_enabled)
        behavior.addWidget(self.dry_run_commands)
        behavior.addWidget(self.adaptive_prosody)
        behavior.addStretch(1)
        layout.addLayout(behavior)

        personal = QHBoxLayout()
        self.personal_voice_consent = QCheckBox(
            "Я согласен на локальную обработку записи и условия модели XTTS (CPML)"
        )
        self.personal_voice_consent.setToolTip(
            "Референс остаётся на этом ПК. Модель XTTS v2 используется по Coqui Public "
            "Model License: https://coqui.ai/cpml"
        )
        record_reference = QPushButton("Записать референс для XTTS")
        record_reference.setProperty("secondary", True)
        record_reference.clicked.connect(self._record_personal_voice)
        install_xtts = QPushButton("Установить XTTS-компонент")
        install_xtts.setProperty("secondary", True)
        install_xtts.clicked.connect(self._install_xtts)
        self.personal_voice_status = QLabel("XTTS загружается только по запросу")
        self.personal_voice_status.setProperty("muted", True)
        personal.addWidget(self.personal_voice_consent)
        personal.addWidget(record_reference)
        personal.addWidget(install_xtts)
        personal.addWidget(self.personal_voice_status, 1)
        layout.addLayout(personal)

        vault = QLabel(
            "API-ключи сохраняются в Windows Credential Manager, а не в config.local.json."
        )
        vault.setProperty("muted", True)
        layout.addWidget(vault)
        return group

    def _llm_group(self) -> QGroupBox:
        group = QGroupBox("Ответы на обычные вопросы")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.provider = QComboBox()
        for label, value in PROVIDERS.items():
            self.provider.addItem(label, value)
        self.provider.currentIndexChanged.connect(self._provider_changed)
        form.addRow("Провайдер", self.provider)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("Хранится только в config.local.json")
        form.addRow("API-ключ", self.api_key)
        self.model = QLineEdit()
        form.addRow("Модель", self.model)
        self.endpoint = QLineEdit()
        form.addRow("Адрес API", self.endpoint)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(64, 4096)
        form.addRow("Макс. токенов", self.max_tokens)
        self.system_prompt = QTextEdit()
        self.system_prompt.setMaximumHeight(86)
        form.addRow("Инструкция", self.system_prompt)
        self.provider_hint = QLabel()
        self.provider_hint.setProperty("muted", True)
        self.provider_hint.setWordWrap(True)
        form.addRow("", self.provider_hint)
        return group

    def _safety_group(self) -> QGroupBox:
        group = QGroupBox("Безопасность и фоновый режим")
        form = QFormLayout(group)
        self.confirmation_level = QComboBox()
        self.confirmation_level.addItem("Только важные действия", "dangerous")
        self.confirmation_level.addItem("Каждое действие", "all")
        self.confirmation_level.addItem("Не спрашивать", "none")
        form.addRow("Подтверждение", self.confirmation_level)
        self.push_to_talk_hotkey = QLineEdit()
        self.push_to_talk_hotkey.setPlaceholderText("ctrl+alt+j")
        form.addRow("Global Push-to-Talk", self.push_to_talk_hotkey)
        self.close_to_tray = QCheckBox("Сворачивать в трей при закрытии окна")
        form.addRow("", self.close_to_tray)
        hint = QLabel("Фраза «отмени последнее» возвращает последнее обратимое действие.")
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        form.addRow("", hint)
        return group

    def _memory_group(self) -> QGroupBox:
        group = QGroupBox("Долговременная память")
        layout = QVBoxLayout(group)
        self.memory_table = QTableWidget(0, 1)
        self.memory_table.setHorizontalHeaderLabels(["Что помнит Jarvis"])
        self.memory_table.horizontalHeader().setStretchLastSection(True)
        self.memory_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.memory_table.setMaximumHeight(150)
        layout.addWidget(self.memory_table)
        row = QHBoxLayout()
        self.memory_input = QLineEdit()
        self.memory_input.setPlaceholderText("Например: меня зовут Николай")
        add = QPushButton("+ Запомнить")
        add.setProperty("secondary", True)
        add.clicked.connect(self._add_memory)
        remove = QPushButton("Удалить")
        remove.setProperty("danger", True)
        remove.clicked.connect(self._remove_memory)
        row.addWidget(self.memory_input, 1)
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)
        return group

    def _scenarios_group(self) -> QGroupBox:
        group = QGroupBox("Многошаговые сценарии")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "Шаги: system=volume_mute | application=C:\\Windows\\System32\\notepad.exe"
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.scenarios_table = QTableWidget(0, 4)
        self.scenarios_table.setHorizontalHeaderLabels(
            ["ID", "Название", "Шаги type=target", "Ответ"]
        )
        self.scenarios_table.horizontalHeader().setStretchLastSection(True)
        self.scenarios_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scenarios_table.setMaximumHeight(175)
        layout.addWidget(self.scenarios_table)
        row = QHBoxLayout()
        add = QPushButton("+ Сценарий")
        add.setProperty("secondary", True)
        add.clicked.connect(self._add_scenario)
        remove = QPushButton("Удалить")
        remove.setProperty("danger", True)
        remove.clicked.connect(lambda: self._remove_selected(self.scenarios_table))
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        return group

    def _diagnostics_group(self) -> QGroupBox:
        group = QGroupBox("Диагностика")
        layout = QVBoxLayout(group)
        self.diagnostics = QTextBrowser()
        self.diagnostics.setMaximumHeight(145)
        self.diagnostics.setText("Нажми проверку — статусы появятся здесь.")
        run = QPushButton("Проверить микрофон · STT · TTS · LLM")
        run.setProperty("secondary", True)
        run.clicked.connect(self._run_diagnostics)
        layout.addWidget(self.diagnostics)
        layout.addWidget(run)
        return group

    def _permissions_group(self) -> QGroupBox:
        group = QGroupBox("Доступ к приложениям")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "Помощник может запускать только включённые .exe с полным путём из этого списка."
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.permissions_table = QTableWidget(0, 3)
        self.permissions_table.setHorizontalHeaderLabels(["Вкл.", "Название", "Полный путь"])
        self.permissions_table.horizontalHeader().setStretchLastSection(True)
        self.permissions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.permissions_table.setMaximumHeight(185)
        layout.addWidget(self.permissions_table)
        row = QHBoxLayout()
        add = QPushButton("+ Приложение")
        add.setProperty("secondary", True)
        add.clicked.connect(self._add_permission)
        choose = QPushButton("Выбрать .exe")
        choose.setProperty("secondary", True)
        choose.clicked.connect(self._choose_permission_path)
        remove = QPushButton("Удалить")
        remove.setProperty("danger", True)
        remove.clicked.connect(lambda: self._remove_selected(self.permissions_table))
        row.addWidget(add)
        row.addWidget(choose)
        row.addWidget(remove)
        layout.addLayout(row)
        return group

    def _commands_group(self) -> QGroupBox:
        group = QGroupBox("Команды Jarvis")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "Фразы разделяются символом |. Для типа «Приложение» путь обязан быть в разрешениях."
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.commands_table = QTableWidget(0, 5)
        self.commands_table.setHorizontalHeaderLabels(
            ["Вкл.", "Фразы", "Действие", "Значение", "Ответ"]
        )
        self.commands_table.horizontalHeader().setStretchLastSection(True)
        self.commands_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.commands_table.setMinimumHeight(260)
        layout.addWidget(self.commands_table)
        row = QHBoxLayout()
        add = QPushButton("+ Команда")
        add.setProperty("secondary", True)
        add.clicked.connect(self._add_command)
        choose = QPushButton("Выбрать приложение")
        choose.setProperty("secondary", True)
        choose.clicked.connect(self._choose_command_target)
        remove = QPushButton("Удалить")
        remove.setProperty("danger", True)
        remove.clicked.connect(lambda: self._remove_selected(self.commands_table))
        row.addWidget(add)
        row.addWidget(choose)
        row.addWidget(remove)
        layout.addLayout(row)
        return group

    def _load_settings(self) -> None:
        assistant = self.store.get("assistant", {}) or {}
        self.wake_phrases.setText(str(assistant.get("wake_phrases", "джарвис, аксиос")))
        self._set_combo_data(self.stt_engine, assistant.get("stt_engine", "vosk"))
        self._set_combo_data(self.whisper_model, assistant.get("whisper_model", "tiny"))
        self.mic_gain.setValue(float(assistant.get("microphone_gain", 1.0)))
        self.continuous.setChecked(bool(assistant.get("continuous_listening", False)))
        self.tts_enabled.setChecked(bool(assistant.get("tts_enabled", True)))
        self._set_combo_data(self.tts_engine, assistant.get("tts_engine", "silero"))
        self._tts_engine_changed()
        self._set_combo_data(
            self.tts_voice, assistant.get("tts_voice", "eugene")
        )
        self._set_combo_data(
            self.tts_fallback, assistant.get("tts_fallback_engine", "none")
        )
        self._set_combo_data(self.tts_profile, assistant.get("tts_profile", "calm"))
        self.tts_rate.setValue(int(assistant.get("tts_rate", 175)))
        self.tts_pitch.setValue(int(assistant.get("tts_pitch", -12)))
        self.tts_volume.setValue(int(float(assistant.get("tts_volume", 0.9)) * 100))
        self.noise_suppression.setChecked(
            bool(assistant.get("noise_suppression", True))
        )
        self.echo_suppression.setChecked(
            bool(assistant.get("echo_suppression", True))
        )
        self.auto_mic_switch.setChecked(
            bool(assistant.get("auto_microphone_switch", True))
        )
        self._set_combo_data(
            self.language_mode, assistant.get("language_mode", "ru")
        )
        self.speaker_auth.setChecked(
            bool(assistant.get("speaker_auth_enabled", False))
        )
        self.documents_enabled.setChecked(
            bool(assistant.get("documents_enabled", False))
        )
        self.documents_folder.setText(
            str(assistant.get("documents_folder", ""))
        )
        self.overlay_enabled.setChecked(
            bool(assistant.get("overlay_enabled", False))
        )
        self.dry_run_commands.setChecked(
            bool(assistant.get("dry_run_commands", False))
        )
        self.adaptive_prosody.setChecked(
            bool(assistant.get("adaptive_prosody", True))
        )
        self.personal_voice_consent.setChecked(
            bool(assistant.get("personal_voice_consent", False))
        )
        self._set_combo_data(
            self.confirmation_level,
            assistant.get("confirmation_level", "dangerous"),
        )
        self.push_to_talk_hotkey.setText(
            str(assistant.get("push_to_talk_hotkey", "ctrl+alt+j"))
        )
        self.close_to_tray.setChecked(bool(assistant.get("close_to_tray", True)))
        self._set_combo_data(self.provider, assistant.get("llm_provider", "none"))
        self.max_tokens.setValue(int(assistant.get("max_tokens", 350)))
        self.system_prompt.setPlainText(str(assistant.get("system_prompt", "")))
        self._active_provider = str(self.provider.currentData())
        self._refresh_microphones(int(assistant.get("microphone_index", -1)))
        self._load_permissions(self.store.get("permissions", []))
        self._load_commands(self.store.get("commands", []))
        self._load_memory()
        self._load_scenarios(self.store.get("scenarios", []))
        self._provider_changed()

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(max(0, index))

    def _refresh_microphones(self, selected: Optional[int] = None) -> None:
        if isinstance(selected, bool):
            selected = None
        if selected is None and self.microphone.count():
            selected = int(self.microphone.currentData())
        self.microphone.clear()
        self.microphone.addItem("Системный микрофон", -1)
        for item in WakeWordListener.list_input_devices():
            self.microphone.addItem(str(item["name"]), int(item["index"]))
        index = self.microphone.findData(selected if selected is not None else -1)
        self.microphone.setCurrentIndex(max(0, index))

    def _auto_switch_microphone(self) -> None:
        if not hasattr(self, "auto_mic_switch") or not self.auto_mic_switch.isChecked():
            return
        selected = int(self.microphone.currentData() or -1)
        if selected < 0:
            return
        available = {
            int(item["index"]) for item in WakeWordListener.list_input_devices()
        }
        if selected in available:
            return
        was_listening = self.listener.is_listening
        if was_listening:
            self.listener.stop_listening()
        self._refresh_microphones(-1)
        self.listener.mic_index = None
        self._set_status(
            "warn", "Выбранный микрофон отключён — включён системный"
        )
        if was_listening:
            self._toggle_listening(True)

    def _choose_documents_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Папка локальных документов", self.documents_folder.text()
        )
        if folder:
            self.documents_folder.setText(folder)

    def _reindex_documents(self) -> None:
        folder = self.documents_folder.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(
                self, "Локальные документы", "Выбери существующую папку."
            )
            return
        self.documents_status.setText("Индексирую…")

        def work() -> None:
            self.document_index.set_root(folder)
            count = self.document_index.rebuild()
            self.signals.background_result.emit(
                "documents", f"Готово: {count} фрагментов"
            )

        threading.Thread(target=work, name="document-index", daemon=True).start()

    def _enroll_voice(self) -> None:
        if self.listener.is_listening:
            QMessageBox.information(
                self, "Голос владельца", "Сначала останови микрофон помощника."
            )
            return
        self.speaker_status.setText("Говори обычным голосом 3–4 секунды…")
        mic = self._mic_index(self.microphone.currentData())

        def work() -> None:
            try:
                samples = VoiceAuthenticator.capture(mic, 3.5)
                self.voice_authenticator.enroll(samples)
                self.listener.voice_auth.load()
                message = "Профиль владельца записан"
            except Exception as exc:
                message = f"Ошибка записи: {exc}"
            self.signals.background_result.emit("speaker_enroll", message)

        threading.Thread(target=work, name="speaker-enroll", daemon=True).start()

    def _verify_voice(self) -> None:
        if not self.voice_authenticator.enrolled:
            QMessageBox.information(
                self, "Проверка голоса", "Сначала запиши голос владельца."
            )
            return
        self.speaker_status.setText("Говори для проверки…")
        mic = self._mic_index(self.microphone.currentData())

        def work() -> None:
            try:
                samples = VoiceAuthenticator.capture(mic, 2.8)
                score = self.voice_authenticator.score(samples)
                message = f"Совпадение: {score:.0%}"
            except Exception as exc:
                message = f"Ошибка проверки: {exc}"
            self.signals.background_result.emit("speaker_verify", message)

        threading.Thread(target=work, name="speaker-verify", daemon=True).start()

    def _record_personal_voice(self) -> None:
        if not self.personal_voice_consent.isChecked():
            QMessageBox.warning(
                self,
                "Персональный голос",
                "Нужно явно подтвердить согласие на локальную обработку записи.",
            )
            return
        if self.listener.is_listening:
            QMessageBox.information(
                self, "Персональный голос", "Сначала останови микрофон помощника."
            )
            return
        self.personal_voice_status.setText(
            "Читай любую фразу спокойно, запись длится 8 секунд…"
        )
        mic = self._mic_index(self.microphone.currentData())

        def work() -> None:
            try:
                import wave

                samples = VoiceAuthenticator.capture(mic, 8.0)
                path = (
                    Path(__file__).resolve().parents[2]
                    / "user_data"
                    / "personal_voice"
                    / "reference.wav"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(samples.astype("<i2").tobytes())
                (path.parent / "xtts_consent").write_text(
                    "accepted: local-processing-and-coqui-cpml\n",
                    encoding="utf-8",
                )
                message = "Референс сохранён локально · можно выбрать XTTS"
            except Exception as exc:
                message = f"Ошибка записи: {exc}"
            self.signals.background_result.emit("personal_voice", message)

        threading.Thread(target=work, name="personal-voice", daemon=True).start()

    def _install_xtts(self) -> None:
        if not self.personal_voice_consent.isChecked():
            QMessageBox.warning(
                self,
                "XTTS",
                "Подтверди согласие на локальное создание персонального голоса.",
            )
            return
        self.personal_voice_status.setText("Устанавливаю XTTS-компонент…")

        def work() -> None:
            import subprocess
            import sys

            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            project_root = Path(__file__).resolve().parents[2]
            environment_dir = project_root / "models" / "xtts_env"
            python_path = environment_dir / "Scripts" / "python.exe"
            commands = []
            if not python_path.is_file():
                commands.append(
                    [sys.executable, "-m", "venv", str(environment_dir)]
                )
            commands.extend(
                [
                    [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                    [
                        str(python_path),
                        "-m",
                        "pip",
                        "install",
                        "torch==2.8.0",
                        "torchaudio==2.8.0",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                    ],
                    [
                        str(python_path),
                        "-m",
                        "pip",
                        "install",
                        "coqui-tts==0.27.5",
                        "transformers>=4.57,<5",
                    ],
                ]
            )
            result = None
            try:
                for command in commands:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=1800,
                        creationflags=flags,
                        check=False,
                    )
                    if result.returncode != 0:
                        break
            except Exception as exc:
                message = f"Ошибка установки XTTS: {exc}"
            else:
                message = (
                    "XTTS установлен отдельно · MediaPipe не затронут"
                    if result is not None and result.returncode == 0
                    else "Не удалось установить XTTS; проверь соединение и повтори"
                )
            self.signals.background_result.emit("personal_voice", message)

        threading.Thread(target=work, name="xtts-install", daemon=True).start()

    def _toggle_overlay(self, enabled: bool) -> None:
        if enabled:
            self.mini_hud.show_message("online", self.status.text())
        else:
            self.mini_hud.hide()

    def _background_result(self, target: str, message: str) -> None:
        if target == "documents":
            self.documents_status.setText(message)
            self.core.documents.set_root(self.documents_folder.text().strip())
            self.core.documents._load()
        elif target.startswith("speaker"):
            self.speaker_status.setText(message)
        elif target == "personal_voice":
            self.personal_voice_status.setText(message)
        self._set_status("good" if "Ошибка" not in message else "bad", message)

    def _tts_engine_changed(self, *_args) -> None:
        engine = str(self.tts_engine.currentData())
        selected = self.tts_voice.currentData()
        if engine == "silero":
            choices = SILERO_VOICES
        elif engine == "piper":
            choices = PIPER_VOICES
        elif engine == "xtts":
            choices = [("personal", "Мой локальный голос · XTTS v2")]
        elif engine == "edge":
            choices = NEURAL_VOICES
        else:
            choices = list_voices()
        self.tts_voice.blockSignals(True)
        self.tts_voice.clear()
        if choices:
            for voice_id, label in choices:
                self.tts_voice.addItem(label, voice_id)
        else:
            self.tts_voice.addItem("Автовыбор системного голоса", "")
        index = self.tts_voice.findData(selected)
        self.tts_voice.setCurrentIndex(max(0, index))
        self.tts_voice.blockSignals(False)
        self.tts_voice.setEnabled(bool(choices))
        self.tts_pitch.setEnabled(engine == "edge")
        tooltips = {
            "silero": "Silero v5.5: русский, ударения, омографы и вопросы. Полностью локально.",
            "piper": "Настоящий локальный нейронный голос. Интернет после установки не нужен.",
            "xtts": "Клонирование по твоему референсу. Требует явного согласия и мощного ПК.",
            "edge": "Онлайн Neural. При сбое Windows включится только если это разрешено ниже.",
            "system": "Обычный установленный голос Windows.",
        }
        self.tts_engine.setToolTip(tooltips.get(engine, ""))

    def _apply_voice_profile(self, *_args) -> None:
        if not hasattr(self, "tts_rate"):
            return
        profile = str(self.tts_profile.currentData())
        rate, pitch = {
            "strict": (165, -18),
            "calm": (175, -10),
            "emotional": (198, 4),
        }.get(profile, (175, -10))
        self.tts_rate.setValue(rate)
        self.tts_pitch.setValue(pitch)

    def _test_voice(self) -> None:
        self.tts.stop()
        self.tts.configure(
            engine=str(self.tts_engine.currentData()),
            voice_id=str(self.tts_voice.currentData()),
            rate=self.tts_rate.value(),
            volume=self.tts_volume.value() / 100.0,
            pitch=self.tts_pitch.value(),
            fallback_engine=str(self.tts_fallback.currentData()),
            profile=str(self.tts_profile.currentData()),
        )
        self.tts.speak(
            "Добрый вечер. Системы работают штатно. Я готов к вашим командам."
        )

    def _tts_status(self, state: str, message: str) -> None:
        active = state in {"queued", "synthesizing", "playing", "fallback"}
        self.test_voice.setEnabled(not active)
        if state == "error":
            self._set_status("bad", message)
        elif state in {"fallback", "cancelled"}:
            self._set_status("warn", message)
        elif active:
            self._set_status("warn" if state == "synthesizing" else "good", message)
        elif state == "done":
            self._set_status("good", message)

    def _provider_changed(self, *_args) -> None:
        previous = self._active_provider
        if previous in {"openrouter", "custom"}:
            self._draft_keys[previous] = self.api_key.text().strip()
        provider = str(self.provider.currentData())
        self._active_provider = provider
        assistant = self.store.get("assistant", {}) or {}
        if provider == "openrouter":
            self.model.setText(str(assistant.get("openrouter_model", "openrouter/auto")))
            self.endpoint.setText("https://openrouter.ai/api/v1")
            self.provider_hint.setText("Облачный API. Ключ сохраняется локально и не попадает в Git.")
        elif provider == "ollama":
            self.model.setText(str(assistant.get("ollama_model", "llama3.2")))
            self.endpoint.setText(str(assistant.get("ollama_url", "http://127.0.0.1:11434")))
            self.provider_hint.setText("Полностью локально. Запусти Ollama и скачай выбранную модель.")
        elif provider == "custom":
            self.model.setText(str(assistant.get("custom_model", "")))
            self.endpoint.setText(str(assistant.get("custom_api_url", "")))
            self.provider_hint.setText("Любой API с форматом OpenAI /chat/completions.")
        else:
            self.model.clear()
            self.endpoint.clear()
            self.provider_hint.setText("Без модели Jarvis отвечает весёлыми фразами из списка ошибок.")
        self.api_key.setEnabled(provider in {"openrouter", "custom"})
        self.api_key.setText(self._draft_keys.get(provider, ""))
        self.model.setEnabled(provider != "none")
        self.endpoint.setEnabled(provider in {"ollama", "custom"})

    def _load_permissions(self, permissions: List[Dict[str, Any]]) -> None:
        self.permissions_table.setRowCount(0)
        for permission in permissions:
            self._add_permission(permission)

    def _load_memory(self) -> None:
        self.memory_table.setRowCount(0)
        for item in self.core.memory.items:
            row = self.memory_table.rowCount()
            self.memory_table.insertRow(row)
            self.memory_table.setItem(row, 0, QTableWidgetItem(str(item.get("text", ""))))

    def _add_memory(self) -> None:
        value = self.memory_input.text().strip()
        if not value:
            return
        self.core.memory.remember(value)
        self.memory_input.clear()
        self._load_memory()
        self._set_status("good", "Добавлено в локальную память")

    def _remove_memory(self) -> None:
        rows = sorted(
            {index.row() for index in self.memory_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.core.memory.delete(row)
        self._load_memory()

    def _load_scenarios(self, scenarios: List[Dict[str, Any]]) -> None:
        self.scenarios_table.setRowCount(0)
        for scenario in scenarios:
            self._add_scenario(scenario)

    def _add_scenario(self, scenario: Optional[Dict[str, Any]] = None) -> None:
        scenario = scenario or {
            "id": f"scenario_{self.scenarios_table.rowCount() + 1}",
            "name": "Новый сценарий",
            "steps": [{"type": "system", "target": "volume_mute"}],
            "reply": "Сценарий выполнен",
        }
        row = self.scenarios_table.rowCount()
        self.scenarios_table.insertRow(row)
        steps = " | ".join(
            f"{step.get('type', '')}={step.get('target', '')}"
            for step in scenario.get("steps", [])
            if isinstance(step, dict)
        )
        values = (
            scenario.get("id", ""),
            scenario.get("name", ""),
            steps,
            scenario.get("reply", ""),
        )
        for column, value in enumerate(values):
            self.scenarios_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _collect_scenarios(self) -> List[Dict[str, Any]]:
        scenarios: List[Dict[str, Any]] = []
        for row in range(self.scenarios_table.rowCount()):
            cells = [
                self.scenarios_table.item(row, column)
                for column in range(self.scenarios_table.columnCount())
            ]
            values = [cell.text().strip() if cell else "" for cell in cells]
            if not values[0]:
                continue
            steps = []
            for value in values[2].split("|"):
                kind, separator, target = value.strip().partition("=")
                if separator and kind.strip() and target.strip():
                    steps.append({"type": kind.strip(), "target": target.strip()})
            scenarios.append(
                {
                    "id": values[0],
                    "name": values[1] or values[0],
                    "steps": steps,
                    "reply": values[3],
                }
            )
        return scenarios

    def _run_diagnostics(self) -> None:
        self.diagnostics.setText("Проверяю компоненты…")
        selected_provider = str(self.provider.currentData())
        selected_endpoint = self.endpoint.text().strip()

        def work() -> None:
            started = time.perf_counter()
            lines = []
            project_root = Path(__file__).resolve().parents[2]
            try:
                from utils.hardware_profiler import detect_hardware

                hardware = detect_hardware()
                lines.append(
                    f"✅ ПК: {hardware.tier_label}, score {hardware.score}/100, "
                    f"{hardware.logical_cores} потоков, {hardware.ram_gb:.1f} ГБ RAM"
                )
            except Exception as exc:
                lines.append(f"⚠ Тест ПК: {exc}")
            microphones = WakeWordListener.list_input_devices()
            lines.append(
                f"✅ Микрофоны: {len(microphones)}"
                if microphones
                else "❌ Микрофоны не найдены"
            )
            try:
                import piper  # noqa: F401

                voice = project_root / "models/piper/ru_RU-denis-medium.onnx"
                lines.append(
                    "✅ Piper: модель установлена"
                    if voice.is_file()
                    else "⚠ Piper: пакет есть, модель загрузит setup_venv.bat"
                )
            except Exception as exc:
                lines.append(f"❌ Piper: {exc}")
            try:
                import faster_whisper  # noqa: F401

                selected_whisper = str(self.whisper_model.currentData() or "tiny")
                whisper_model = (
                    project_root
                    / "models"
                    / "whisper"
                    / selected_whisper
                    / "model.bin"
                )
                lines.append(
                    f"✅ Whisper {selected_whisper}: локальная модель установлена"
                    if whisper_model.is_file()
                    else f"⚠ Whisper {selected_whisper}: модель не установлена"
                )
            except Exception as exc:
                lines.append(f"❌ Whisper: {exc}")
            vosk_model = project_root / str(
                self.store.get("assistant.vosk_model_path", "models/vosk-ru")
            )
            lines.append(
                "✅ Vosk: локальная модель найдена"
                if (vosk_model / "am/final.mdl").is_file()
                else "⚠ Vosk: локальная модель не найдена"
            )
            provider = selected_provider
            if provider == "none":
                lines.append("⚠ LLM: провайдер не выбран")
            else:
                try:
                    import requests

                    check_started = time.perf_counter()
                    if provider == "ollama":
                        url = selected_endpoint.rstrip("/") + "/api/tags"
                        response = requests.get(url, timeout=3)
                    elif provider == "openrouter":
                        response = requests.get(
                            "https://openrouter.ai/api/v1/models", timeout=4
                        )
                    else:
                        url = selected_endpoint
                        response = requests.get(url, timeout=3)
                    latency = (time.perf_counter() - check_started) * 1000
                    lines.append(
                        f"✅ LLM {provider}: HTTP {response.status_code}, {latency:.0f} мс"
                        if response.status_code < 500
                        else f"❌ LLM {provider}: HTTP {response.status_code}"
                    )
                except Exception as exc:
                    lines.append(f"❌ LLM {provider}: {exc}")
            try:
                import requests

                net_started = time.perf_counter()
                response = requests.get(
                    "https://www.msftconnecttest.com/connecttest.txt", timeout=3
                )
                net_latency = (time.perf_counter() - net_started) * 1000
                lines.append(
                    f"✅ Интернет: HTTP {response.status_code}, {net_latency:.0f} мс"
                )
            except Exception as exc:
                lines.append(f"⚠ Интернет: {exc}")
            lines.append(f"⏱ Диагностика: {(time.perf_counter() - started) * 1000:.0f} мс")
            self.signals.diagnostics_ready.emit("<br>".join(lines))

        threading.Thread(target=work, name="jarvis-diagnostics", daemon=True).start()

    def _show_diagnostics(self, value: str) -> None:
        self.diagnostics.setHtml(value)

    def _add_permission(self, permission: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(permission, bool):
            permission = None
        permission = permission or {"enabled": True, "name": "Новое приложение", "path": ""}
        row = self.permissions_table.rowCount()
        self.permissions_table.insertRow(row)
        enabled = QTableWidgetItem()
        enabled.setFlags(enabled.flags() | Qt.ItemIsUserCheckable)
        enabled.setCheckState(Qt.Checked if permission.get("enabled", True) else Qt.Unchecked)
        self.permissions_table.setItem(row, 0, enabled)
        self.permissions_table.setItem(row, 1, QTableWidgetItem(str(permission.get("name", ""))))
        self.permissions_table.setItem(row, 2, QTableWidgetItem(str(permission.get("path", ""))))
        self.permissions_table.selectRow(row)

    def _choose_permission_path(self) -> None:
        row = self.permissions_table.currentRow()
        if row < 0:
            self._add_permission()
            row = self.permissions_table.currentRow()
        path, _ = QFileDialog.getOpenFileName(self, "Выберите приложение", "", "Программы (*.exe)")
        if path:
            self.permissions_table.item(row, 2).setText(path)
            if self.permissions_table.item(row, 1).text() == "Новое приложение":
                self.permissions_table.item(row, 1).setText(Path(path).stem)

    def _load_commands(self, commands: List[Dict[str, Any]]) -> None:
        self.commands_table.setRowCount(0)
        for command in commands:
            self._add_command(command)

    def _add_command(self, command: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(command, bool):
            command = None
        command = command or {
            "enabled": True,
            "phrases": ["новая команда"],
            "type": "response",
            "target": "Команда настроена",
            "reply": "Готово",
        }
        row = self.commands_table.rowCount()
        self.commands_table.insertRow(row)
        enabled = QTableWidgetItem()
        enabled.setFlags(enabled.flags() | Qt.ItemIsUserCheckable)
        enabled.setCheckState(Qt.Checked if command.get("enabled", True) else Qt.Unchecked)
        self.commands_table.setItem(row, 0, enabled)
        phrases = command.get("phrases", [])
        if isinstance(phrases, list):
            phrases = " | ".join(str(item) for item in phrases)
        self.commands_table.setItem(row, 1, QTableWidgetItem(str(phrases)))
        kind = QComboBox()
        for value, label in ACTION_LABELS.items():
            kind.addItem(label, value)
        self._set_combo_data(kind, str(command.get("type", "response")))
        self.commands_table.setCellWidget(row, 2, kind)
        self.commands_table.setItem(row, 3, QTableWidgetItem(str(command.get("target", ""))))
        self.commands_table.setItem(row, 4, QTableWidgetItem(str(command.get("reply", ""))))
        self.commands_table.selectRow(row)

    def _choose_command_target(self) -> None:
        row = self.commands_table.currentRow()
        if row < 0:
            return
        combo = self.commands_table.cellWidget(row, 2)
        if not isinstance(combo, QComboBox) or combo.currentData() != "application":
            QMessageBox.information(
                self, "Тип действия", "Сначала выбери тип «Приложение»."
            )
            return
        path, _ = QFileDialog.getOpenFileName(self, "Выберите приложение", "", "Программы (*.exe)")
        if path:
            self.commands_table.item(row, 3).setText(path)

    @staticmethod
    def _remove_selected(table: QTableWidget) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _collect_permissions(self) -> List[Dict[str, Any]]:
        result = []
        for row in range(self.permissions_table.rowCount()):
            result.append(
                {
                    "enabled": self.permissions_table.item(row, 0).checkState() == Qt.Checked,
                    "name": self.permissions_table.item(row, 1).text().strip(),
                    "path": self.permissions_table.item(row, 2).text().strip(),
                }
            )
        return result

    def _collect_commands(self) -> List[Dict[str, Any]]:
        result = []
        for row in range(self.commands_table.rowCount()):
            kind = self.commands_table.cellWidget(row, 2)
            assert isinstance(kind, QComboBox)
            phrases = [
                part.strip()
                for part in self.commands_table.item(row, 1).text().split("|")
                if part.strip()
            ]
            result.append(
                {
                    "id": f"user_command_{row + 1}",
                    "enabled": self.commands_table.item(row, 0).checkState() == Qt.Checked,
                    "phrases": phrases,
                    "type": str(kind.currentData()),
                    "target": self.commands_table.item(row, 3).text().strip(),
                    "reply": self.commands_table.item(row, 4).text().strip(),
                }
            )
        return result

    def _save_all(self) -> None:
        permissions = self._collect_permissions()
        commands = self._collect_commands()
        scenarios = self._collect_scenarios()
        validator = ActionExecutor(permissions, scenarios=scenarios)
        try:
            for scenario in scenarios:
                validator.validate({"type": "scenario", "target": scenario["id"]})
            for command in commands:
                if command["enabled"]:
                    validator.validate(command)
        except Exception as exc:
            QMessageBox.warning(self, "Команды не сохранены", str(exc))
            return

        provider = str(self.provider.currentData())
        if provider in {"openrouter", "custom"}:
            self._draft_keys[provider] = self.api_key.text().strip()
        values: Dict[str, Any] = {
            "assistant.wake_phrases": self.wake_phrases.text().strip(),
            "assistant.stt_engine": str(self.stt_engine.currentData()),
            "assistant.whisper_model": str(self.whisper_model.currentData()),
            "assistant.microphone_index": int(self.microphone.currentData()),
            "assistant.microphone_gain": float(self.mic_gain.value()),
            "assistant.continuous_listening": self.continuous.isChecked(),
            "assistant.tts_enabled": self.tts_enabled.isChecked(),
            "assistant.tts_engine": str(self.tts_engine.currentData()),
            "assistant.tts_voice": str(self.tts_voice.currentData()),
            "assistant.tts_fallback_engine": str(self.tts_fallback.currentData()),
            "assistant.tts_profile": str(self.tts_profile.currentData()),
            "assistant.tts_rate": self.tts_rate.value(),
            "assistant.tts_pitch": self.tts_pitch.value(),
            "assistant.tts_volume": self.tts_volume.value() / 100.0,
            "assistant.llm_provider": provider,
            "assistant.max_tokens": self.max_tokens.value(),
            "assistant.system_prompt": self.system_prompt.toPlainText().strip(),
            "assistant.confirmation_level": str(self.confirmation_level.currentData()),
            "assistant.close_to_tray": self.close_to_tray.isChecked(),
            "assistant.push_to_talk_hotkey": self.push_to_talk_hotkey.text().strip(),
            "assistant.noise_suppression": self.noise_suppression.isChecked(),
            "assistant.echo_suppression": self.echo_suppression.isChecked(),
            "assistant.auto_microphone_switch": self.auto_mic_switch.isChecked(),
            "assistant.language_mode": str(self.language_mode.currentData()),
            "assistant.speaker_auth_enabled": self.speaker_auth.isChecked(),
            "assistant.documents_enabled": self.documents_enabled.isChecked(),
            "assistant.documents_folder": self.documents_folder.text().strip(),
            "assistant.overlay_enabled": self.overlay_enabled.isChecked(),
            "assistant.dry_run_commands": self.dry_run_commands.isChecked(),
            "assistant.adaptive_prosody": self.adaptive_prosody.isChecked(),
            "assistant.personal_voice_consent": self.personal_voice_consent.isChecked(),
        }
        if provider == "openrouter":
            values["assistant.openrouter_model"] = self.model.text().strip()
        elif provider == "ollama":
            values["assistant.ollama_url"] = self.endpoint.text().strip()
            values["assistant.ollama_model"] = self.model.text().strip()
        elif provider == "custom":
            values["assistant.custom_api_url"] = self.endpoint.text().strip()
            values["assistant.custom_model"] = self.model.text().strip()
        self.store.update_many(values)
        for key_provider in ("openrouter", "custom"):
            self.store.set_api_key(
                key_provider, self._draft_keys.get(key_provider, "")
            )
        self.store.replace_section("permissions", permissions)
        self.store.replace_section("commands", commands)
        self.store.replace_section("scenarios", scenarios)
        self.executor.set_permissions(permissions)
        self.executor.set_scenarios(scenarios)
        self.core.reload()
        self.listener.configure(
            wake_word=self.wake_phrases.text().strip(),
            model_path=str(self.store.get("assistant.vosk_model_path", "models/vosk-ru")),
            stt_engine=str(self.stt_engine.currentData()),
            mic_index=self._mic_index(self.microphone.currentData()),
            gain=self.mic_gain.value(),
            whisper_model=str(self.whisper_model.currentData()),
            noise_suppression=self.noise_suppression.isChecked(),
            echo_suppression=self.echo_suppression.isChecked(),
            language_mode=str(self.language_mode.currentData()),
        )
        self.tts.configure(
            engine=str(self.tts_engine.currentData()),
            voice_id=str(self.tts_voice.currentData()),
            rate=self.tts_rate.value(),
            volume=self.tts_volume.value() / 100.0,
            pitch=self.tts_pitch.value(),
            fallback_engine=str(self.tts_fallback.currentData()),
            profile=str(self.tts_profile.currentData()),
        )
        self._set_status("good", "Настройки применены")

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text or self._busy:
            return
        self.tts.stop()
        self.input.clear()
        self._append_message("user", text)
        self._process(text)

    def _process(self, text: str) -> None:
        self._busy = True
        self.send_button.setEnabled(False)
        self._set_status("warn", "Думаю…")

        def work() -> None:
            result = self.core.handle_text_stream(
                text, self.signals.stream_sentence.emit
            )
            self.signals.result_ready.emit(result)

        threading.Thread(target=work, name="jarvis-command", daemon=True).start()

    def _finish_result(self, result: AssistantResult) -> None:
        meta = ""
        if result.kind == "command":
            meta = f"команда · совпадение {result.score:.0f}%"
        elif result.kind == "fallback" and result.error:
            meta = "модель недоступна"
        elif result.kind == "error" and result.error:
            meta = f"действие заблокировано: {result.error}"
        elif result.kind == "streamed":
            meta = "потоковый ответ · озвучка началась по предложениям"
        elif result.kind == "preview":
            meta = "SIMULATION · действие не выполнялось"
        self._append_message("assistant", result.text, meta)
        if self.overlay_enabled.isChecked():
            self.mini_hud.show_message(result.kind, result.text)
        if self.adaptive_prosody.isChecked():
            profile = (
                "strict"
                if result.kind in {"error", "confirmation"}
                else ("emotional" if result.kind in {"fallback", "preview"} else "calm")
            )
            self.tts.configure(
                engine=str(self.tts_engine.currentData()),
                voice_id=str(self.tts_voice.currentData()),
                rate={"strict": 165, "emotional": 194}.get(profile, 175),
                volume=self.tts_volume.value() / 100.0,
                pitch={"strict": -18, "emotional": 3}.get(profile, -10),
                fallback_engine=str(self.tts_fallback.currentData()),
                profile=profile,
            )
        if self.tts_enabled.isChecked() and result.kind in {
            "answer",
            "fallback",
            "error",
            "command",
            "confirmation",
            "cancelled",
            "preview",
        }:
            self.tts.speak_stream(result.text)
        self._busy = False
        self.send_button.setEnabled(True)
        self._set_status("good", "Готов")

    def _stream_sentence(self, sentence: str) -> None:
        if self.tts_enabled.isChecked():
            self.tts.speak(sentence)
        self._set_status("warn", "Получаю ответ · озвучиваю по предложениям")

    def _append_message(self, role: str, text: str, meta: str = "") -> None:
        safe = html.escape(str(text)).replace("\n", "<br>")
        who = "Ты" if role == "user" else "Jarvis"
        css = "user" if role == "user" else "assistant"
        timestamp = time.strftime("%H:%M")
        meta_html = f"<div class='meta'>{html.escape(meta)}</div>" if meta else ""
        self.chat.append(
            f"<div class='row'><div class='who'>{who} · {timestamp}</div>"
            f"<div class='{css}'>{safe}</div>{meta_html}</div>"
        )
        bar = self.chat.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _toggle_listening(self, checked: bool) -> None:
        if checked:
            started = self.listener.start_listening(
                lambda text: self.signals.recognized.emit(text),
                self.continuous.isChecked(),
                lambda state, message: self.signals.listener_status.emit(state, message),
            )
            if not started:
                self.mic_button.setChecked(False)
                return
            self.mic_button.setText("Остановить микрофон")
        else:
            self.listener.stop_listening()
            self.mic_button.setText("Включить микрофон")

    def _handle_recognized(self, text: str) -> None:
        self.tts.stop()
        if self.speaker_auth.isChecked():
            score = self.listener.last_speaker_score
            if not self.voice_authenticator.enrolled:
                self._set_status(
                    "bad", "Проверка владельца включена, но голос не записан"
                )
                return
            if score is None or score < 0.78:
                self._append_message(
                    "assistant",
                    "Голос не подтверждён. Команда заблокирована.",
                    f"speaker score {0 if score is None else score:.0%}",
                )
                self._set_status("bad", "Голос владельца не подтверждён")
                return
        self._append_message("user", text, "голос")
        if not self._busy:
            self._process(text)

    def _listener_status(self, state: str, message: str) -> None:
        if state == "error":
            self.mic_button.setChecked(False)
            self.mic_button.setText("Включить микрофон")
            self._set_status("bad", message)
        elif state == "speech":
            self.tts.stop()
            self._set_status("warn", message)
        elif state in {"loading", "wake"}:
            self._set_status("warn", message)
        elif state == "ready":
            self._set_status("good", message)
        elif state == "idle":
            self._set_status("good", "Готов")

    def _set_status(self, level: str, text: str) -> None:
        names = {"good": "StatusGood", "warn": "StatusWarn", "bad": "StatusBad"}
        self.status.setObjectName(names.get(level, "StatusWarn"))
        self.status.setText(text)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status_changed.emit(level, text)
        if hasattr(self, "overlay_enabled") and self.overlay_enabled.isChecked():
            self.mini_hud.show_message(level, text)

    def apply_settings(self, _settings: Dict[str, Any]) -> None:
        self.core.reload()

    def push_to_talk(self) -> None:
        """Global hotkey target: interrupt speech and toggle microphone."""
        self.tts.stop()
        self.mic_button.setChecked(not self.mic_button.isChecked())
        self._toggle_listening(self.mic_button.isChecked())

    def cleanup(self) -> None:
        self.listener.stop_listening()
        self.tts.shutdown()
        self.mini_hud.close()
