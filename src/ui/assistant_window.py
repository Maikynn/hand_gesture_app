"""Integrated assistant settings, permissions and chat page."""

from __future__ import annotations

import threading

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from assistant.action_executor import ACTION_LABELS, ActionExecutor
from assistant.jarvis_engine import JarvisEngine
from assistant.listener import WakeWordListener
from assistant.llm_client import LLMClient
from assistant.silero_tts import SileroTTSEngine, available as silero_available
from assistant.tts_engine import TTSEngine
from utils.config_store import load_config, save_config
from utils.logger import setup_logger


class AssistantSignals(QObject):
    answer = pyqtSignal(str)
    status = pyqtSignal(str)


class AssistantWindow(QWidget):
    """A complete assistant embedded into the main application tab."""

    config_saved = pyqtSignal(dict)

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.logger = setup_logger("AssistantWindow")
        self.config = config if config is not None else load_config()
        self.voice = self.config["voice"]
        self.is_listening = False
        self.signals = AssistantSignals()
        self.signals.answer.connect(self._show_answer)
        self.signals.status.connect(self._set_status)
        self.executor = ActionExecutor(self.voice.get("allowed_apps", []))
        self.tts = self._build_tts()
        self.listener = None
        self.jarvis = None
        self._build_ui()
        self._load_values()
        self._rebuild_services()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 20)
        title = QLabel("Голосовой помощник")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Настройте голос, доступы и способ ответа — затем общайтесь в чате."
        )
        subtitle.setObjectName("muted")
        root.addWidget(title)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._settings_panel())
        splitter.addWidget(self._chat_panel())
        splitter.setSizes([500, 780])
        root.addWidget(splitter, 1)

    def _settings_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 10, 0)

        voice_box = QGroupBox("Голос и активация")
        form = QFormLayout(voice_box)
        self.wake_word = QLineEdit()
        self.stt_engine = QComboBox()
        self.stt_engine.addItems(["vosk", "whisper"])
        self.tts_engine = QComboBox()
        self.tts_engine.addItems(["pyttsx3", "silero"])
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        form.addRow("Фраза активации", self.wake_word)
        form.addRow("Распознавание речи", self.stt_engine)
        form.addRow("Озвучивание", self.tts_engine)
        form.addRow("Громкость", self.volume)
        layout.addWidget(voice_box)

        ai_box = QGroupBox("Ответы на общие вопросы")
        ai_form = QFormLayout(ai_box)
        self.provider = QComboBox()
        self.provider.addItem("Без AI — веселая ошибка", "none")
        self.provider.addItem("OpenRouter", "openrouter")
        self.provider.addItem("Локальная Ollama", "ollama")
        self.provider.addItem("Другое OpenAI-compatible API", "custom")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.model = QLineEdit()
        self.api_url = QLineEdit()
        self.system_prompt = QTextEdit()
        self.system_prompt.setMaximumHeight(80)
        self.provider.currentIndexChanged.connect(self._provider_visibility)
        ai_form.addRow("Провайдер", self.provider)
        ai_form.addRow("API-ключ", self.api_key)
        ai_form.addRow("Модель", self.model)
        ai_form.addRow("URL API / Ollama", self.api_url)
        ai_form.addRow("Характер помощника", self.system_prompt)
        layout.addWidget(ai_box)

        app_box = QGroupBox("Разрешенные приложения")
        app_layout = QVBoxLayout(app_box)
        hint = QLabel("Помощник сможет запустить только приложения из этого списка.")
        hint.setObjectName("muted")
        app_layout.addWidget(hint)
        self.apps = QTableWidget(0, 2)
        self.apps.setHorizontalHeaderLabels(["Название", "Полный путь к программе"])
        self._prepare_table(self.apps)
        app_layout.addWidget(self.apps)
        buttons = QHBoxLayout()
        add_app = QPushButton("＋ Добавить программу")
        add_app.clicked.connect(self._add_app)
        remove_app = QPushButton("Удалить")
        remove_app.clicked.connect(lambda: self._remove_row(self.apps))
        buttons.addWidget(add_app)
        buttons.addWidget(remove_app)
        app_layout.addLayout(buttons)
        layout.addWidget(app_box)

        cmd_box = QGroupBox("Свои голосовые команды")
        cmd_layout = QVBoxLayout(cmd_box)
        self.commands = QTableWidget(0, 3)
        self.commands.setHorizontalHeaderLabels(
            ["Фразы через |", "Действие", "Путь / URL / клавиши"]
        )
        self._prepare_table(self.commands)
        cmd_layout.addWidget(self.commands)
        cmd_buttons = QHBoxLayout()
        add_cmd = QPushButton("＋ Добавить команду")
        add_cmd.clicked.connect(self._add_command)
        remove_cmd = QPushButton("Удалить")
        remove_cmd.clicked.connect(lambda: self._remove_row(self.commands))
        cmd_buttons.addWidget(add_cmd)
        cmd_buttons.addWidget(remove_cmd)
        cmd_layout.addLayout(cmd_buttons)
        layout.addWidget(cmd_box)

        save = QPushButton("Сохранить настройки помощника")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save_settings)
        layout.addWidget(save)
        layout.addStretch()
        scroll.setWidget(body)
        return scroll

    def _chat_panel(self):
        panel = QFrame()
        panel.setObjectName("chatPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        top = QHBoxLayout()
        self.status = QLabel("● Готов")
        self.listen = QPushButton("🎙 Начать слушать")
        self.listen.clicked.connect(self.toggle_listening)
        test = QPushButton("Проверить голос")
        test.clicked.connect(lambda: self.tts.speak("Привет! Я готов помогать."))
        top.addWidget(self.status)
        top.addStretch()
        top.addWidget(test)
        top.addWidget(self.listen)
        layout.addLayout(top)
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        self.chat.append(
            "<b>Акси</b><br>Привет! Я могу запустить разрешенную программу, выполнить вашу команду или ответить на вопрос."
        )
        layout.addWidget(self.chat, 1)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Напишите команду или вопрос…")
        self.input.returnPressed.connect(self.send_message)
        send = QPushButton("Отправить ➜")
        send.setObjectName("primaryButton")
        send.clicked.connect(self.send_message)
        row.addWidget(self.input, 1)
        row.addWidget(send)
        layout.addLayout(row)
        return panel

    @staticmethod
    def _prepare_table(table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setMinimumHeight(150)

    def _load_values(self):
        v = self.voice
        self.wake_word.setText(v.get("wake_word", "аксиос"))
        self.stt_engine.setCurrentText(v.get("stt_engine", "vosk"))
        self.tts_engine.setCurrentText(v.get("tts_engine", "pyttsx3"))
        self.volume.setValue(int(v.get("volume", 0.9) * 100))
        idx = self.provider.findData(v.get("llm_engine", "none"))
        self.provider.setCurrentIndex(max(0, idx))
        engine = v.get("llm_engine", "none")
        self.api_key.setText(
            v.get("custom_api_key" if engine == "custom" else "openrouter_key", "")
        )
        self.model.setText(
            v.get(
                "ollama_model"
                if engine == "ollama"
                else "custom_api_model"
                if engine == "custom"
                else "openrouter_model",
                "",
            )
        )
        self.api_url.setText(
            v.get("custom_api_url" if engine == "custom" else "ollama_url", "")
        )
        self.system_prompt.setPlainText(v.get("system_prompt", ""))
        for app in v.get("allowed_apps", []):
            self._append_items(self.apps, [app.get("name", ""), app.get("path", "")])
        for cmd in v.get("commands", []):
            self._add_command(cmd)
        self._provider_visibility()

    @staticmethod
    def _append_items(table, values):
        row = table.rowCount()
        table.insertRow(row)
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(str(value)))
        return row

    def _add_app(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите приложение")
        if path:
            name = path.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
            self._append_items(self.apps, [name, path])

    def _add_command(self, command=None):
        command = command or {"phrase": "", "action": "open_app", "value": ""}
        row = self._append_items(
            self.commands, [command.get("phrase", ""), "", command.get("value", "")]
        )
        combo = QComboBox()
        for key, label in ACTION_LABELS.items():
            combo.addItem(label, key)
        index = combo.findData(command.get("action", "none"))
        combo.setCurrentIndex(max(0, index))
        self.commands.setCellWidget(row, 1, combo)

    @staticmethod
    def _remove_row(table):
        if table.currentRow() >= 0:
            table.removeRow(table.currentRow())

    def _provider_visibility(self):
        engine = self.provider.currentData()
        self.api_key.setEnabled(engine in {"openrouter", "custom"})
        self.api_url.setEnabled(engine in {"ollama", "custom"})
        self.model.setEnabled(engine != "none")

    def save_settings(self):
        engine = self.provider.currentData()
        v = self.config["voice"]
        v.update(
            {
                "wake_word": self.wake_word.text().strip() or "аксиос",
                "stt_engine": self.stt_engine.currentText(),
                "tts_engine": self.tts_engine.currentText(),
                "volume": self.volume.value() / 100,
                "llm_engine": engine,
                "system_prompt": self.system_prompt.toPlainText().strip(),
            }
        )
        if engine == "openrouter":
            v.update(
                openrouter_key=self.api_key.text().strip(),
                openrouter_model=self.model.text().strip(),
            )
        elif engine == "ollama":
            v.update(
                ollama_url=self.api_url.text().strip(),
                ollama_model=self.model.text().strip(),
            )
        elif engine == "custom":
            v.update(
                custom_api_url=self.api_url.text().strip(),
                custom_api_key=self.api_key.text().strip(),
                custom_api_model=self.model.text().strip(),
            )
        v["allowed_apps"] = [
            {"name": self._cell(self.apps, r, 0), "path": self._cell(self.apps, r, 1)}
            for r in range(self.apps.rowCount())
            if self._cell(self.apps, r, 1)
        ]
        v["commands"] = [
            {
                "phrase": self._cell(self.commands, r, 0).casefold(),
                "action": self.commands.cellWidget(r, 1).currentData(),
                "value": self._cell(self.commands, r, 2),
            }
            for r in range(self.commands.rowCount())
            if self._cell(self.commands, r, 0)
        ]
        save_config(self.config)
        self.voice = v
        self.executor.update_allowed_apps(v["allowed_apps"])
        self._rebuild_services(rebuild_tts=True)
        self.config_saved.emit(self.config)
        QMessageBox.information(self, "Готово", "Настройки помощника сохранены.")

    @staticmethod
    def _cell(table, row, col):
        item = table.item(row, col)
        return item.text().strip() if item else ""

    def _build_tts(self):
        v = self.voice
        if v.get("tts_engine") == "silero" and silero_available():
            return SileroTTSEngine(
                speaker=v.get("tts_speaker", "xenia"),
                rate=v.get("tts_rate", 190),
                volume=v.get("volume", 0.9),
            )
        return TTSEngine(rate=v.get("tts_rate", 190), volume=v.get("volume", 0.9))

    def _rebuild_services(self, rebuild_tts=False):
        if self.listener:
            self.listener.stop_listening()
        self.listener = None
        v = self.voice
        if rebuild_tts:
            self.tts.shutdown()
            self.tts = self._build_tts()
        self.llm = LLMClient(
            engine=v.get("llm_engine", "none"),
            api_key=v.get("openrouter_key", ""),
            model=v.get("openrouter_model", ""),
            ollama_url=v.get("ollama_url", ""),
            ollama_model=v.get("ollama_model", ""),
            system_prompt=v.get("system_prompt", ""),
            custom_api_url=v.get("custom_api_url", ""),
            custom_api_key=v.get("custom_api_key", ""),
            custom_api_model=v.get("custom_api_model", ""),
        )
        if self.jarvis is None:
            self.jarvis = JarvisEngine(v, self.executor, self.llm)
        else:
            self.jarvis.update(v, self.executor, self.llm)

    def _ensure_listener(self):
        if self.listener is None:
            v = self.voice
            self.listener = WakeWordListener(
                wake_word=v.get("wake_word", "аксиос"),
                model_path=v.get("vosk_model_path", "models/vosk-ru"),
                stt_engine=v.get("stt_engine", "vosk"),
                whisper_model=v.get("whisper_model", ""),
                mic_index=v.get("mic_index"),
                gain=v.get("mic_gain", 2),
            )

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            try:
                self._ensure_listener()
                self.listener.start_listening(self._voice_command)
            except Exception as exc:
                self.is_listening = False
                self._set_status("⚠ Микрофон недоступен")
                QMessageBox.warning(self, "Не удалось начать прослушивание", str(exc))
                return
            self.listen.setText("■ Остановить")
            self._set_status(f"● Слушаю «{self.voice.get('wake_word')}»")
        else:
            if self.listener:
                self.listener.stop_listening()
            self.listen.setText("🎙 Начать слушать")
            self._set_status("● Готов")

    def _voice_command(self, text):
        self.signals.answer.emit(f"__USER__{text}")
        self.process_command(text)

    def send_message(self):
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._append_message("Вы", text, "user")
        self.process_command(text)

    def process_command(self, text):
        self.signals.status.emit("● Выполняю…")
        threading.Thread(target=self._process, args=(text,), daemon=True).start()

    def _process(self, text):
        try:
            reply = self.jarvis.process(text)
            self.signals.answer.emit(reply.text)
        except Exception as exc:
            self.logger.exception("Assistant command failed")
            self.signals.answer.emit(f"Ой, команда споткнулась об ошибку: {exc}")
        finally:
            self.signals.status.emit("● Слушаю" if self.is_listening else "● Готов")

    def _show_answer(self, text):
        if text.startswith("__USER__"):
            self._append_message("Вы", text[8:], "user")
            return
        self._append_message("Акси", text, "assistant")
        self.tts.speak(text)

    def _append_message(self, author, text, role):
        color = "#6c8cff" if role == "user" else "#36c993"
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        self.chat.append(
            f'<div style="margin:10px 0"><b style="color:{color}">{author}</b><br>{safe}</div>'
        )

    def _set_status(self, text):
        self.status.setText(text)

    def apply_settings(self, _settings):
        pass

    def cleanup(self):
        if self.listener:
            self.listener.stop_listening()
        self.tts.shutdown()
