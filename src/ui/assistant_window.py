from __future__ import annotations

import html
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from assistant.actions import ACTION_LABELS, ActionExecutor
from assistant.embedded_jarvis import AssistantResult, EmbeddedJarvis
from assistant.listener import WakeWordListener
from assistant.tts_engine import NEURAL_VOICES, TTSEngine, list_voices
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


class AssistantWindow(QWidget):
    """Chat, voice, permissions and command settings in one assistant tab."""

    status_changed = pyqtSignal(str, str)

    def __init__(self, store: ConfigStore | None = None, parent=None):
        super().__init__(parent)
        self.store = store or ConfigStore()
        self.executor = ActionExecutor(self.store.get("permissions", []))
        self.core = EmbeddedJarvis(self.store, executor=self.executor)
        assistant = self.store.get("assistant", {}) or {}
        self.tts = TTSEngine(
            engine=str(assistant.get("tts_engine", "edge")),
            voice_id=str(assistant.get("tts_voice", "ru-RU-DmitryNeural")),
            rate=int(assistant.get("tts_rate", 175)),
            volume=float(assistant.get("tts_volume", 0.9)),
            pitch=int(assistant.get("tts_pitch", -12)),
        )
        self.listener = WakeWordListener(
            wake_word=str(assistant.get("wake_phrases", "джарвис, аксиос")),
            model_path=str(assistant.get("vosk_model_path", "models/vosk-ru")),
            stt_engine=str(assistant.get("stt_engine", "vosk")),
            mic_index=self._mic_index(assistant.get("microphone_index", -1)),
            gain=float(assistant.get("microphone_gain", 1.0)),
        )
        self.signals = AssistantSignals()
        self.signals.result_ready.connect(self._finish_result)
        self.signals.recognized.connect(self._handle_recognized)
        self.signals.listener_status.connect(self._listener_status)
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
        title = QLabel("Помощник")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Jarvis встроен в приложение: чат, голос, разрешения, команды и модели"
        )
        subtitle.setObjectName("PageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.status = QLabel("Готов")
        self.status.setObjectName("StatusGood")
        header.addWidget(self.status)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_chat())
        splitter.addWidget(self._build_settings())
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([650, 550])
        root.addWidget(splitter, 1)

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
            .user{background:#245fcb;color:white;padding:10px;border-radius:10px}
            .assistant{background:#1b2943;color:#eef4ff;padding:10px;border-radius:10px}
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
        layout = QVBoxLayout(body)
        layout.setContentsMargins(2, 2, 8, 2)
        layout.setSpacing(12)
        layout.addWidget(self._voice_group())
        layout.addWidget(self._llm_group())
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
            .user{background:#245fcb;color:white;padding:10px;border-radius:10px}
            .assistant{background:#1b2943;color:#eef4ff;padding:10px;border-radius:10px}
            .meta{color:#8fa0bb;font-size:10px;margin-top:3px}
            """
        else:
            css = """
            .row{margin:8px 0}.who{font-size:10px;color:#64748b;margin-bottom:3px}
            .user{background:#2864d7;color:white;padding:10px;border-radius:10px}
            .assistant{background:#e9eff8;color:#172033;padding:10px;border-radius:10px}
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
        self.stt_engine.addItem("Google · онлайн", "google")
        form.addRow("Распознавание", self.stt_engine)

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
        self.tts_engine.addItem("Jarvis Neural · красивый онлайн-голос", "edge")
        self.tts_engine.addItem("Windows · системный офлайн-голос", "system")
        self.tts_engine.currentIndexChanged.connect(self._tts_engine_changed)
        form.addRow("Голосовой движок", self.tts_engine)

        voice_row = QHBoxLayout()
        self.tts_voice = QComboBox()
        for voice_id, label in NEURAL_VOICES:
            self.tts_voice.addItem(label, voice_id)
        test_voice = QPushButton("▶ Проверить")
        test_voice.setProperty("secondary", True)
        test_voice.clicked.connect(self._test_voice)
        voice_row.addWidget(self.tts_voice, 1)
        voice_row.addWidget(test_voice)
        form.addRow("Голос", voice_row)

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
        self.mic_gain.setValue(float(assistant.get("microphone_gain", 1.0)))
        self.continuous.setChecked(bool(assistant.get("continuous_listening", False)))
        self.tts_enabled.setChecked(bool(assistant.get("tts_enabled", True)))
        self._set_combo_data(self.tts_engine, assistant.get("tts_engine", "edge"))
        self._set_combo_data(
            self.tts_voice, assistant.get("tts_voice", "ru-RU-DmitryNeural")
        )
        self.tts_rate.setValue(int(assistant.get("tts_rate", 175)))
        self.tts_pitch.setValue(int(assistant.get("tts_pitch", -12)))
        self.tts_volume.setValue(int(float(assistant.get("tts_volume", 0.9)) * 100))
        self._tts_engine_changed()
        self._set_combo_data(self.provider, assistant.get("llm_provider", "none"))
        self.max_tokens.setValue(int(assistant.get("max_tokens", 350)))
        self.system_prompt.setPlainText(str(assistant.get("system_prompt", "")))
        self._active_provider = str(self.provider.currentData())
        self._refresh_microphones(int(assistant.get("microphone_index", -1)))
        self._load_permissions(self.store.get("permissions", []))
        self._load_commands(self.store.get("commands", []))
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

    def _tts_engine_changed(self, *_args) -> None:
        neural = str(self.tts_engine.currentData()) == "edge"
        selected = self.tts_voice.currentData()
        choices = NEURAL_VOICES if neural else list_voices()
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
        self.tts_pitch.setEnabled(neural)
        self.tts_engine.setToolTip(
            "Neural требует интернет и автоматически переключается на Windows при ошибке."
            if neural
            else "Системный голос работает без интернета."
        )

    def _test_voice(self) -> None:
        self.tts.stop()
        self.tts.configure(
            engine=str(self.tts_engine.currentData()),
            voice_id=str(self.tts_voice.currentData()),
            rate=self.tts_rate.value(),
            volume=self.tts_volume.value() / 100.0,
            pitch=self.tts_pitch.value(),
        )
        self.tts.speak(
            "Добрый вечер. Системы работают штатно. Я готов к вашим командам."
        )
        self._set_status("good", "Воспроизвожу тест голоса")

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
        validator = ActionExecutor(permissions)
        try:
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
            "assistant.microphone_index": int(self.microphone.currentData()),
            "assistant.microphone_gain": float(self.mic_gain.value()),
            "assistant.continuous_listening": self.continuous.isChecked(),
            "assistant.tts_enabled": self.tts_enabled.isChecked(),
            "assistant.tts_engine": str(self.tts_engine.currentData()),
            "assistant.tts_voice": str(self.tts_voice.currentData()),
            "assistant.tts_rate": self.tts_rate.value(),
            "assistant.tts_pitch": self.tts_pitch.value(),
            "assistant.tts_volume": self.tts_volume.value() / 100.0,
            "assistant.llm_provider": provider,
            "assistant.max_tokens": self.max_tokens.value(),
            "assistant.system_prompt": self.system_prompt.toPlainText().strip(),
        }
        if provider == "openrouter":
            values["assistant.openrouter_model"] = self.model.text().strip()
        elif provider == "ollama":
            values["assistant.ollama_url"] = self.endpoint.text().strip()
            values["assistant.ollama_model"] = self.model.text().strip()
        elif provider == "custom":
            values["assistant.custom_api_url"] = self.endpoint.text().strip()
            values["assistant.custom_model"] = self.model.text().strip()
        for key_provider in ("openrouter", "custom"):
            values[f"assistant.api_keys.{key_provider}"] = self._draft_keys.get(key_provider, "")
        self.store.update_many(values)
        self.store.replace_section("permissions", permissions)
        self.store.replace_section("commands", commands)
        self.executor.set_permissions(permissions)
        self.core.reload()
        self.listener.configure(
            wake_word=self.wake_phrases.text().strip(),
            model_path=str(self.store.get("assistant.vosk_model_path", "models/vosk-ru")),
            stt_engine=str(self.stt_engine.currentData()),
            mic_index=self._mic_index(self.microphone.currentData()),
            gain=self.mic_gain.value(),
        )
        self.tts.configure(
            engine=str(self.tts_engine.currentData()),
            voice_id=str(self.tts_voice.currentData()),
            rate=self.tts_rate.value(),
            volume=self.tts_volume.value() / 100.0,
            pitch=self.tts_pitch.value(),
        )
        self._set_status("good", "Настройки применены")

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text or self._busy:
            return
        self.input.clear()
        self._append_message("user", text)
        self._process(text)

    def _process(self, text: str) -> None:
        self._busy = True
        self.send_button.setEnabled(False)
        self._set_status("warn", "Думаю…")

        def work() -> None:
            result = self.core.handle_text(text)
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
        self._append_message("assistant", result.text, meta)
        if self.tts_enabled.isChecked() and result.kind in {"answer", "fallback", "error"}:
            self.tts.speak(result.text)
        self._busy = False
        self.send_button.setEnabled(True)
        self._set_status("good", "Готов")

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
        self._append_message("user", text, "голос")
        if not self._busy:
            self._process(text)

    def _listener_status(self, state: str, message: str) -> None:
        if state == "error":
            self.mic_button.setChecked(False)
            self.mic_button.setText("Включить микрофон")
            self._set_status("bad", message)
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

    def apply_settings(self, _settings: Dict[str, Any]) -> None:
        self.core.reload()

    def cleanup(self) -> None:
        self.listener.stop_listening()
        self.tts.shutdown()
