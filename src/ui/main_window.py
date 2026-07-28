from __future__ import annotations

import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Load MediaPipe's native runtime before Qt. On Windows, loading Qt5 first can
# make _framework_bindings fail or crash later inside MSVCP140.dll.
from camera.skeleton_renderer import SkeletonRenderer

# Preload Piper/ONNX before Qt widgets for stable native DLL initialization.
from assistant import tts_engine as _tts_runtime  # noqa: F401
from faster_whisper import WhisperModel as _whisper_runtime  # noqa: F401

from PyQt5.QtCore import QEvent, QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QAbstractItemView,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from assistant.actions import ACTION_LABELS, ActionExecutor
from camera.advanced_features import (
    CustomGestureLibrary,
    ErrorClipRecorder,
    GestureHistory,
    GestureSequenceMatcher,
    apply_roi_mask,
    auto_enhance,
    normalized_landmarks,
    privacy_silhouette,
)
from camera.video_capture import VideoCapture
from hand_processing.gesture_recognizer import GestureRecognizer
from hand_processing.gesture_fusion import HAGRID_CLASSES
from hand_processing.static_yolo import DEFAULT_YOLO_MODEL, YOLOStaticModel
from ui.assistant_window import AssistantWindow
from ui.theme import apply_theme
from utils.active_window import foreground_process_name, match_profile, parse_profile_rules
from utils.config_store import ConfigStore
from utils.global_hotkey import GlobalHotkey
from utils.hardware_profiler import (
    AdaptiveLoadController,
    HardwareProfile,
    detect_hardware,
)
from utils.logger import setup_logger
from utils.model_manager import ModelManager


GESTURE_LABELS = {
    "unknown": "неизвестно",
    "fist": "кулак",
    "open_palm": "ладонь",
    "palm": "ладонь",
    "point": "один",
    "one": "один",
    "victory": "два",
    "two_up": "два",
    "thumbs_up": "палец вверх",
    "like": "палец вверх",
    "thumbs_down": "палец вниз",
    "dislike": "палец вниз",
    "ok_sign": "окей",
    "ok": "окей",
    "rock": "рок",
    "peace": "мир",
    "three": "три",
    "four": "четыре",
    "five": "пять",
    "call_me": "позвони",
    "call": "позвони",
    "gun": "пистолет",
    "pinch": "щипок",
    "no_gesture": "нет жеста",
}

CANONICAL_GESTURE = {
    "open_palm": "palm",
    "thumbs_up": "like",
    "thumbs_down": "dislike",
    "victory": "two_up",
    "ok_sign": "ok",
    "point": "one",
    "call_me": "call",
}

GESTURES = [
    "palm",
    "fist",
    "like",
    "dislike",
    "one",
    "two_up",
    "three",
    "four",
    "ok",
    "rock",
    "call",
    "gun",
    "pinch",
]


class ParameterWheelGuard(QObject):
    """Keep accidental wheel scrolling from changing focused controls."""

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Wheel and isinstance(
            watched, (QComboBox, QAbstractSpinBox, QSlider)
        ):
            parent = watched.parentWidget()
            while parent is not None and not isinstance(parent, QAbstractScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                delta = event.angleDelta().y()
                bar = parent.verticalScrollBar()
                direction = -1 if delta > 0 else 1
                steps = max(1, abs(delta) // 120)
                bar.setValue(bar.value() + direction * bar.singleStep() * 3 * steps)
            event.ignore()
            return True
        return super().eventFilter(watched, event)


class JarvisOrb(QWidget):
    """Small, lightweight HUD reactor drawn entirely by Qt."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(112, 112)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(45)

    def _tick(self) -> None:
        self._phase = (self._phase + 2) % 360
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(35, 224, 255, 22))
        painter.drawEllipse(center, 34, 34)
        painter.setBrush(QColor(67, 232, 255, 40))
        painter.drawEllipse(center, 22, 22)
        rings = (
            (QRectF(8, 8, 96, 96), self._phase * 16, 96 * 16, 2),
            (QRectF(18, 18, 76, 76), -self._phase * 16, 126 * 16, 3),
            (QRectF(29, 29, 54, 54), (self._phase * 2) * 16, 172 * 16, 2),
        )
        for bounds, start, span, width in rings:
            painter.setPen(QPen(QColor("#39dff5"), width))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(bounds, int(start), int(span))
            painter.drawArc(bounds, int(start + 190 * 16), int(span // 2))
        painter.setPen(QPen(QColor("#d8fbff"), 1))
        painter.drawEllipse(center, 8, 8)
        painter.setPen(QColor("#74edff"))
        painter.drawText(self.rect(), Qt.AlignCenter, "AI")


class ConfidenceGraph(QWidget):
    """Compact live graph for confidence and inference latency."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setObjectName("TelemetryGraph")
        self.confidence: deque[float] = deque([0.0] * 60, maxlen=60)
        self.latency: deque[float] = deque([0.0] * 60, maxlen=60)

    def add_sample(self, confidence: float, latency_ms: float) -> None:
        self.confidence.append(max(0.0, min(1.0, float(confidence))))
        self.latency.append(max(0.0, min(1.0, float(latency_ms) / 300.0)))
        if self.isVisible():
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#030b10"))
        width, height = self.width(), self.height()
        painter.setPen(QPen(QColor("#143942"), 1))
        for fraction in (0.25, 0.5, 0.75):
            y = round(height * fraction)
            painter.drawLine(0, y, width, y)

        def draw(values, color):
            points = list(values)
            if len(points) < 2:
                return
            painter.setPen(QPen(QColor(color), 2))
            for index in range(1, len(points)):
                x1 = round((index - 1) * (width - 1) / (len(points) - 1))
                x2 = round(index * (width - 1) / (len(points) - 1))
                y1 = round((1.0 - points[index - 1]) * (height - 18)) + 5
                y2 = round((1.0 - points[index]) * (height - 18)) + 5
                painter.drawLine(x1, y1, x2, y2)

        draw(self.latency, "#ffb454")
        draw(self.confidence, "#52e7f7")
        painter.setPen(QColor("#77aab3"))
        painter.drawText(10, height - 5, "CONFIDENCE")
        painter.setPen(QColor("#ffb454"))
        painter.drawText(width - 95, height - 5, "LATENCY")


class HandPreview(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setFixedSize(340, 150)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.image = QLabel("Рука не обнаружена")
        self.image.setObjectName("HandSurface")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setFixedSize(164, 124)
        self.text = QLabel(f"{title}: —")
        self.text.setStyleSheet("font-size: 13pt; font-weight: 650;")
        self.text.setWordWrap(True)
        layout.addWidget(self.image)
        layout.addWidget(self.text, 1)

    def clear_hand(self, title: str) -> None:
        self.image.clear()
        self.image.setText("Рука не обнаружена")
        self.text.setText(f"{title}: —")

    def update_hand(self, title: str, gesture: str, confidence: float, crop: np.ndarray) -> None:
        self.text.setText(
            f"{title}: {GESTURE_LABELS.get(gesture, gesture)}\n"
            f"<span style='font-size:9pt;color:#8f9bb3'>{confidence:.0%}</span>"
        )
        CameraWidget.set_image(self.image, crop, allow_upscale=False)

    def update_private(self, title: str, gesture: str, confidence: float) -> None:
        self.image.clear()
        self.image.setText("COORDS ONLY")
        self.text.setText(
            f"{title}: {GESTURE_LABELS.get(gesture, gesture)}\n"
            f"<span style='font-size:9pt;color:#78cbd5'>{confidence:.0%}</span>"
        )


class GestureBindingsPanel(QFrame):
    saved = pyqtSignal()

    def __init__(self, store: ConfigStore, executor: ActionExecutor, parent=None):
        super().__init__(parent)
        self.store = store
        self.executor = executor
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        header = QHBoxLayout()
        title = QLabel("Действия по жестам")
        title.setStyleSheet("font-size: 14pt; font-weight: 700;")
        header.addWidget(title)
        header.addStretch(1)
        add = QPushButton("+ Привязка")
        add.setProperty("secondary", True)
        add.clicked.connect(self.add_row)
        choose = QPushButton("Выбрать .exe")
        choose.setProperty("secondary", True)
        choose.clicked.connect(self.choose_application)
        remove = QPushButton("Удалить")
        remove.setProperty("danger", True)
        remove.clicked.connect(self.remove_selected)
        save = QPushButton("Сохранить")
        save.clicked.connect(self.save)
        for button in (add, choose, remove, save):
            header.addWidget(button)
        layout.addLayout(header)

        hint = QLabel(
            "Жест срабатывает один раз за удержание. Для запуска программы укажи полный путь "
            "и сначала разреши её во вкладке «Помощник»."
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Вкл.", "Жест", "Зона", "Действие", "Значение", "Точность", "Пауза, мс"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)
        for binding in store.get("gesture_bindings", []):
            self.add_row(binding)

    def load_bindings(self, bindings: List[Dict[str, Any]]) -> None:
        self.table.setRowCount(0)
        for binding in bindings:
            self.add_row(binding)

    @staticmethod
    def _combo(value_map: Dict[str, str], selected: str) -> QComboBox:
        combo = QComboBox()
        for value, label in value_map.items():
            combo.addItem(label, value)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(0, index))
        return combo

    def add_row(self, binding: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(binding, bool):
            binding = None
        binding = binding or {
            "enabled": True,
            "gesture": "palm",
            "type": "system",
            "target": "play_pause",
            "confidence": 0.8,
            "cooldown_ms": 1200,
        }
        row = self.table.rowCount()
        self.table.insertRow(row)
        enabled = QTableWidgetItem()
        enabled.setFlags(enabled.flags() | Qt.ItemIsUserCheckable)
        enabled.setCheckState(Qt.Checked if binding.get("enabled", True) else Qt.Unchecked)
        self.table.setItem(row, 0, enabled)
        gesture_map = {name: GESTURE_LABELS.get(name, name) for name in GESTURES}
        self.table.setCellWidget(row, 1, self._combo(gesture_map, str(binding.get("gesture", "palm"))))
        zone_map = {
            "any": "Любая",
            "left": "Слева",
            "center": "Центр",
            "right": "Справа",
            "top": "Сверху",
            "bottom": "Снизу",
        }
        self.table.setCellWidget(
            row, 2, self._combo(zone_map, str(binding.get("zone", "any")))
        )
        action_map = {
            key: label
            for key, label in ACTION_LABELS.items()
            if key in {"application", "url", "hotkey", "system"}
        }
        self.table.setCellWidget(row, 3, self._combo(action_map, str(binding.get("type", "system"))))
        self.table.setItem(row, 4, QTableWidgetItem(str(binding.get("target", ""))))
        self.table.setItem(row, 5, QTableWidgetItem(str(binding.get("confidence", 0.8))))
        self.table.setItem(row, 6, QTableWidgetItem(str(binding.get("cooldown_ms", 1200))))
        self.table.selectRow(row)

    def remove_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def choose_application(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        kind = self.table.cellWidget(row, 3)
        if not isinstance(kind, QComboBox) or kind.currentData() != "application":
            QMessageBox.information(self, "Тип действия", "Сначала выбери тип «Приложение».")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Выберите приложение", "", "Программы (*.exe)")
        if path:
            self.table.item(row, 4).setText(path)

    def values(self) -> List[Dict[str, Any]]:
        result = []
        for row in range(self.table.rowCount()):
            gesture = self.table.cellWidget(row, 1)
            zone = self.table.cellWidget(row, 2)
            kind = self.table.cellWidget(row, 3)
            assert (
                isinstance(gesture, QComboBox)
                and isinstance(zone, QComboBox)
                and isinstance(kind, QComboBox)
            )
            result.append(
                {
                    "enabled": self.table.item(row, 0).checkState() == Qt.Checked,
                    "gesture": str(gesture.currentData()),
                    "zone": str(zone.currentData()),
                    "type": str(kind.currentData()),
                    "target": self.table.item(row, 4).text().strip(),
                    "confidence": float(self.table.item(row, 5).text().replace(",", ".")),
                    "cooldown_ms": int(self.table.item(row, 6).text()),
                }
            )
        return result

    def save(self) -> None:
        try:
            values = self.values()
            for binding in values:
                if binding["enabled"]:
                    self.executor.validate(binding)
        except Exception as exc:
            QMessageBox.warning(self, "Привязки не сохранены", str(exc))
            return
        self.store.replace_section("gesture_bindings", values)
        self.saved.emit()


class CameraWidget(QWidget):
    status_changed = pyqtSignal(str, str)
    hardware_ready = pyqtSignal(object)

    def __init__(self, store: ConfigStore | None = None, parent=None):
        super().__init__(parent)
        self.store = store or ConfigStore()
        self.logger = setup_logger("CameraWidget")
        self.hardware_profile: HardwareProfile = detect_hardware()
        if bool(self.store.get("hardware.auto_tune", True)):
            self._apply_hardware_profile(self.hardware_profile, announce=False)
        self.settings = self.store.get("camera", {}) or {}
        self.camera_id = int(self.settings.get("device_index", 0))
        camera_source: int | str = self.camera_id
        if bool(self.settings.get("use_network_camera", False)):
            network_url = str(self.settings.get("network_url", "")).strip()
            if network_url:
                camera_source = network_url
        self.video_capture = VideoCapture(
            camera_source,
            width=int(
                self.settings.get(
                    "capture_width", self.hardware_profile.camera_width
                )
            ),
            height=int(
                self.settings.get(
                    "capture_height", self.hardware_profile.camera_height
                )
            ),
            fps=int(
                self.settings.get("capture_fps", self.hardware_profile.camera_fps)
            ),
        )
        self.renderer = SkeletonRenderer()
        self.model_manager = ModelManager(autoload=False)
        self.recognizer = GestureRecognizer(self.model_manager)
        self.yolo_model: Optional[YOLOStaticModel] = None
        self._model_loading = False
        self.executor = ActionExecutor(self.store.get("permissions", []))
        self.bindings = list(self.store.get("gesture_bindings", []))
        self.custom_gestures = CustomGestureLibrary()
        for custom_name in self.custom_gestures.names():
            if custom_name not in GESTURES:
                GESTURES.append(custom_name)
        self.history = GestureHistory()
        self.error_clips = ErrorClipRecorder()
        self._last_pixel_hands: List[List[Tuple[int, int]]] = []
        self._hand_zones = {"Left": "any", "Right": "any"}
        self._lighting_samples: Optional[List[Tuple[float, float, int]]] = None
        self._clip_saving = False
        self.sequence_matcher = GestureSequenceMatcher(
            self.store.get("gesture_sequences", [])
        )
        self.load_controller = AdaptiveLoadController(
            int(self.settings.get("inference_interval_ms", 100)),
            self.hardware_profile.tier,
        )
        self._adaptive_interval_ms = self.load_controller.current_interval_ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self._last_inference = 0.0
        self._processing = False
        self._camera_paused = False
        self._consecutive_errors = 0
        self._frame_times: List[float] = []
        self._last_hand_count = 0
        self._hand_state: Dict[str, Dict[str, Any]] = {
            "Left": {
                "gesture": "",
                "count": 0,
                "fired": False,
                "first_seen": 0.0,
                "logged": False,
            },
            "Right": {
                "gesture": "",
                "count": 0,
                "fired": False,
                "first_seen": 0.0,
                "logged": False,
            },
        }
        self._last_action: Dict[Tuple[str, str], float] = {}
        self._last_inference_ms = 0.0
        self._last_confidence = 0.0
        self._last_gesture = "none"
        self._last_cooldown_ms = 0
        self._pending_gesture_action: Optional[Tuple[Dict[str, Any], str, float]] = None
        self._recording_name = ""
        self._recording_side = ""
        self._recording_samples: List[List[float]] = []
        self._calibration_samples: Optional[List[float]] = None
        self._build_ui()
        self.hardware_ready.connect(self._hardware_retested)
        self.store.changed.connect(self._store_changed)
        self.profile_watch_timer = QTimer(self)
        self.profile_watch_timer.timeout.connect(self._auto_switch_profile)
        self.profile_watch_timer.start(1800)
        if self.model_combo.currentData() == "yolo":
            QTimer.singleShot(0, self._change_model)

    def _apply_hardware_profile(
        self, profile: HardwareProfile, *, announce: bool = True
    ) -> None:
        self.store.update_many(
            {
                "hardware.detected_tier": profile.tier,
                "hardware.score": profile.score,
                "hardware.last_tested": profile.measured_at,
                "camera.capture_width": profile.camera_width,
                "camera.capture_height": profile.camera_height,
                "camera.capture_fps": profile.camera_fps,
                "camera.inference_interval_ms": profile.inference_interval_ms,
                "camera.yolo_inference_interval_ms": profile.yolo_interval_ms,
                "camera.model": profile.camera_model,
                "assistant.whisper_model": profile.whisper_model,
                "assistant.local_llm_hint": profile.local_llm_hint,
                "assistant.ollama_model": profile.local_llm_hint,
            }
        )
        if announce and hasattr(self, "hardware_report"):
            self.hardware_report.setText(profile.summary())
            self._set_camera_status(
                "good", f"Применён профиль {profile.tier_label}"
            )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("CAMERA // LIVE")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Основной видеоканал и быстрые параметры распознавания")
        subtitle.setObjectName("PageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        layout.addLayout(header)

        quick = QHBoxLayout()
        self.performance_status = QLabel("Ожидание кадра")
        self.performance_status.setObjectName("StatusNeutral")
        self.performance_status.setMaximumWidth(180)
        quick.addWidget(self.performance_status)
        quick.addStretch(1)
        self.actions_toggle = QPushButton("Управление жестами: выкл")
        self.actions_toggle.setCheckable(True)
        self.actions_toggle.setChecked(bool(self.settings.get("actions_enabled", False)))
        self.actions_toggle.setProperty("secondary", True)
        self.actions_toggle.setToolTip(
            "Распознавание работает всегда. Действия выполняются только когда переключатель включён."
        )
        self.actions_toggle.toggled.connect(self._toggle_actions)
        self._set_actions_toggle_text(self.actions_toggle.isChecked())
        quick.addWidget(self.actions_toggle)
        self.pause_button = QPushButton("Пауза")
        self.pause_button.setProperty("secondary", True)
        self.pause_button.clicked.connect(self._toggle_camera_pause)
        quick.addWidget(self.pause_button)
        self.camera_status = QLabel("Камера подключена" if self.video_capture.is_open else "Нет камеры")
        self.camera_status.setObjectName(
            "StatusGood" if self.video_capture.is_open else "StatusBad"
        )
        self.camera_status.setMaximumWidth(280)
        self.camera_status.setWordWrap(True)
        quick.addWidget(self.camera_status)
        layout.addLayout(quick)

        self.video = QLabel("Подключаю камеру…")
        self.video.setObjectName("VideoSurface")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setFixedSize(820, 461)
        video_row = QHBoxLayout()
        video_row.addStretch(1)
        video_row.addWidget(self.video)
        video_row.addStretch(1)
        layout.addLayout(video_row)

        hands = QHBoxLayout()
        hands.setSpacing(14)
        self.left_hand = HandPreview("Левая")
        self.right_hand = HandPreview("Правая")
        hands.addStretch(1)
        hands.addWidget(self.left_hand)
        hands.addWidget(self.right_hand)
        hands.addStretch(1)
        layout.addLayout(hands)
        layout.addWidget(self._controls())
        layout.addStretch(1)

        # These panels live on the separate Camera Lab page, but remain owned by
        # CameraWidget so the recognition pipeline can use the same controls.
        self.model_details_card = self._model_details_controls()
        self.advanced_card = self._advanced_controls()
        self.bindings_panel = GestureBindingsPanel(self.store, self.executor)
        self.bindings_panel.saved.connect(self._reload_bindings)
        self.history_panel = self._history_panel()
        scroll.setWidget(body)
        root.addWidget(scroll)

    def _controls(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        row = QHBoxLayout(card)
        row.setContentsMargins(16, 16, 16, 16)
        row.setSpacing(22)
        form = QFormLayout()
        self.camera_combo = QComboBox()
        for index in range(6):
            self.camera_combo.addItem(f"Камера {index}", index)
        self.camera_combo.setCurrentIndex(max(0, self.camera_combo.findData(self.camera_id)))
        self.camera_combo.currentIndexChanged.connect(self._change_camera)
        form.addRow("Камера", self.camera_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItem("Быстрые правила · MediaPipe", "rules")
        self.model_combo.addItem("YOLO HaGRID · нейросеть", "yolo")
        self.model_combo.addItem("Встроенная ONNX/TFLite", "built_in")
        self.model_combo.addItem("Своя ONNX/TFLite", "custom")
        index = self.model_combo.findData(str(self.settings.get("model", "rules")))
        self.model_combo.setCurrentIndex(max(0, index))
        self.model_combo.currentIndexChanged.connect(self._model_selection_changed)
        form.addRow("Модель", self.model_combo)

        row.addLayout(form, 2)

        sliders = QFormLayout()
        self.brightness = QSlider(Qt.Horizontal)
        self.brightness.setRange(-100, 100)
        self.brightness.setValue(int(self.settings.get("brightness", 0)))
        self.brightness.valueChanged.connect(self._save_camera_controls)
        sliders.addRow("Яркость", self.brightness)
        row.addLayout(sliders, 2)

        toggles = QVBoxLayout()
        self.mirror = QCheckBox("Зеркальное изображение")
        self.mirror.setChecked(bool(self.settings.get("mirror", True)))
        self.mirror.toggled.connect(self._save_camera_controls)
        toggles.addWidget(self.mirror)
        basic_hint = QLabel("Остальные параметры —\nв CAMERA LAB")
        basic_hint.setProperty("muted", True)
        toggles.addWidget(basic_hint)
        row.addLayout(toggles, 1)
        return card

    def _model_details_controls(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Видеопоток и визуализация")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        row = QHBoxLayout()

        form = QFormLayout()
        model_path_row = QHBoxLayout()
        initial_model = str(self.model_combo.currentData())
        initial_path = (
            self.settings.get("yolo_model_path", DEFAULT_YOLO_MODEL)
            if initial_model == "yolo"
            else self.settings.get("custom_model_path", "")
        )
        self.model_path = QLineEdit(str(initial_path))
        self.choose_model_button = QPushButton("Выбрать…")
        self.choose_model_button.setProperty("secondary", True)
        self.choose_model_button.clicked.connect(self._choose_model)
        model_path_row.addWidget(self.model_path, 1)
        model_path_row.addWidget(self.choose_model_button)
        form.addRow("Файл модели", model_path_row)
        self.padding = QSlider(Qt.Horizontal)
        self.padding.setRange(5, 80)
        self.padding.setValue(int(self.settings.get("crop_padding", 34)))
        self.padding.valueChanged.connect(self._save_camera_controls)
        form.addRow("Отступ руки", self.padding)
        network_row = QHBoxLayout()
        self.network_camera_url = QLineEdit(
            str(self.settings.get("network_url", ""))
        )
        self.network_camera_url.setPlaceholderText(
            "http://192.168.1.20:8080/video или rtsp://..."
        )
        connect_network = QPushButton("Подключить")
        connect_network.setProperty("secondary", True)
        connect_network.clicked.connect(self._connect_network_camera)
        network_row.addWidget(self.network_camera_url, 1)
        network_row.addWidget(connect_network)
        form.addRow("Камера телефона", network_row)
        row.addLayout(form, 2)

        toggles = QVBoxLayout()
        self.show_hands = QCheckBox("Рисовать скелет рук")
        self.show_face = QCheckBox("Сетка лица")
        self.show_pose = QCheckBox("Скелет тела")
        self.privacy_mode_check = QCheckBox("Privacy: только координаты рук")
        self.error_clips_check = QCheckBox("Сохранять ролик после отметки ошибки")
        self.show_hands.setChecked(bool(self.settings.get("show_hands", True)))
        self.show_face.setChecked(bool(self.settings.get("show_face", False)))
        self.show_pose.setChecked(bool(self.settings.get("show_pose", False)))
        self.privacy_mode_check.setChecked(
            bool(self.settings.get("privacy_mode", False))
        )
        self.error_clips_check.setChecked(
            bool(self.settings.get("record_error_clips", True))
        )
        for widget in (
            self.show_hands,
            self.show_face,
            self.show_pose,
            self.privacy_mode_check,
            self.error_clips_check,
        ):
            widget.toggled.connect(self._save_camera_controls)
            toggles.addWidget(widget)
        row.addLayout(toggles, 1)
        layout.addLayout(row)
        self._sync_model_path_controls()
        return card

    def _advanced_controls(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Точная настройка и обучение")
        title.setStyleSheet("font-size: 13pt; font-weight: 700;")
        layout.addWidget(title)
        layout.addWidget(self._hardware_group())

        columns = QHBoxLayout()
        form = QFormLayout()
        self.profile_combo = QComboBox()
        profiles = self.store.get("camera_profiles", {}) or {}
        for profile_id, profile in profiles.items():
            self.profile_combo.addItem(str(profile.get("name", profile_id)), profile_id)
        self._set_combo_value(
            self.profile_combo, self.settings.get("active_profile", "work")
        )
        self.profile_combo.currentIndexChanged.connect(self._change_profile)
        save_profile = QPushButton("Сохранить текущие параметры в профиль")
        save_profile.setProperty("secondary", True)
        save_profile.clicked.connect(self._save_profile)
        form.addRow("Профиль", self.profile_combo)
        form.addRow("", save_profile)

        self.inference_device = QComboBox()
        self.inference_device.addItem("Авто: GPU → CPU", "auto")
        self.inference_device.addItem("Только CPU", "cpu")
        self.inference_device.addItem("NVIDIA CUDA", "cuda")
        self._set_combo_value(
            self.inference_device, self.settings.get("inference_device", "auto")
        )
        self.inference_device.currentIndexChanged.connect(self._save_advanced_controls)
        form.addRow("Ускорение YOLO", self.inference_device)

        self.hold_ms = QSpinBox()
        self.hold_ms.setRange(0, 3000)
        self.hold_ms.setSingleStep(100)
        self.hold_ms.setSuffix(" мс")
        self.hold_ms.setValue(int(self.settings.get("hold_ms", 450)))
        self.hold_ms.valueChanged.connect(self._save_advanced_controls)
        form.addRow("Удержание жеста", self.hold_ms)
        columns.addLayout(form, 2)

        options = QVBoxLayout()
        self.auto_enhance_check = QCheckBox("Автояркость, контраст и баланс белого")
        self.hud_check = QCheckBox("Подробный HUD поверх видео")
        self.roi_check = QCheckBox("Распознавать только внутри ROI")
        self.confirm_gesture_check = QCheckBox("Подтверждать действие вторым жестом")
        self.auto_profile_check = QCheckBox("Автопрофиль по активной программе")
        self.auto_enhance_check.setChecked(bool(self.settings.get("auto_enhance", False)))
        self.hud_check.setChecked(bool(self.settings.get("hud_enabled", True)))
        self.roi_check.setChecked(bool(self.settings.get("roi_enabled", False)))
        self.confirm_gesture_check.setChecked(
            bool(self.settings.get("confirmation_gesture_enabled", False))
        )
        self.auto_profile_check.setChecked(
            bool(self.settings.get("auto_profile_enabled", True))
        )
        for option in (
            self.auto_enhance_check,
            self.hud_check,
            self.roi_check,
            self.confirm_gesture_check,
            self.auto_profile_check,
        ):
            option.toggled.connect(self._save_advanced_controls)
            options.addWidget(option)
        self.confirm_gesture_combo = QComboBox()
        for gesture in GESTURES:
            self.confirm_gesture_combo.addItem(
                GESTURE_LABELS.get(gesture, gesture), gesture
            )
        self._set_combo_value(
            self.confirm_gesture_combo,
            self.settings.get("confirmation_gesture", "ok"),
        )
        self.confirm_gesture_combo.currentIndexChanged.connect(
            self._save_advanced_controls
        )
        options.addWidget(self.confirm_gesture_combo)
        self.profile_rules = QLineEdit(
            str(
                self.settings.get(
                    "profile_app_rules",
                    "powerpnt=presentation;spotify=music;steam=games;chrome=work",
                )
            )
        )
        self.profile_rules.setToolTip(
            "Формат: часть_имени.exe=профиль; следующая=профиль"
        )
        self.profile_rules.editingFinished.connect(self._save_advanced_controls)
        options.addWidget(self.profile_rules)
        columns.addLayout(options, 2)

        roi_form = QFormLayout()
        roi = list(self.settings.get("roi", [0.08, 0.08, 0.84, 0.84]))
        self.roi_sliders: List[QSlider] = []
        for label, value in zip(("ROI X", "ROI Y", "ROI ширина", "ROI высота"), roi):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(round(float(value) * 100))
            slider.valueChanged.connect(self._save_advanced_controls)
            self.roi_sliders.append(slider)
            roi_form.addRow(label, slider)
        columns.addLayout(roi_form, 2)
        layout.addLayout(columns)

        tools = QHBoxLayout()
        calibrate = QPushButton("Калибровать руки")
        calibrate.setProperty("secondary", True)
        calibrate.clicked.connect(self._start_calibration)
        lighting = QPushButton("Мастер камеры и света")
        lighting.setProperty("secondary", True)
        lighting.clicked.connect(self._start_lighting_wizard)
        record = QPushButton("Записать свой жест")
        record.setProperty("secondary", True)
        record.clicked.connect(self._start_custom_recording)
        self.sequence_gesture = QComboBox()
        for gesture in GESTURES:
            self.sequence_gesture.addItem(
                GESTURE_LABELS.get(gesture, gesture), gesture
            )
        add_step = QPushButton("+ Шаг")
        add_step.setProperty("secondary", True)
        add_step.clicked.connect(self._add_sequence_step)
        remove_step = QPushButton("− Шаг")
        remove_step.setProperty("secondary", True)
        remove_step.clicked.connect(self._remove_sequence_step)
        self.sequence_builder = QListWidget()
        self.sequence_builder.setFlow(QListWidget.LeftToRight)
        self.sequence_builder.setDragDropMode(QAbstractItemView.InternalMove)
        self.sequence_builder.setMaximumHeight(54)
        add_sequence = QPushButton("+ Последовательность")
        add_sequence.setProperty("secondary", True)
        add_sequence.clicked.connect(self._add_sequence)
        tools.addWidget(calibrate)
        tools.addWidget(lighting)
        tools.addWidget(record)
        tools.addWidget(self.sequence_gesture)
        tools.addWidget(add_step)
        tools.addWidget(remove_step)
        tools.addWidget(add_sequence)
        layout.addLayout(tools)
        layout.addWidget(self.sequence_builder)
        self.sequences_table = QTableWidget(0, 3)
        self.sequences_table.setHorizontalHeaderLabels(
            ["Последовательность", "Действие", "Пауза"]
        )
        self.sequences_table.horizontalHeader().setStretchLastSection(True)
        self.sequences_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sequences_table.setMaximumHeight(135)
        layout.addWidget(self.sequences_table)
        sequence_buttons = QHBoxLayout()
        remove_sequence = QPushButton("Удалить выбранную последовательность")
        remove_sequence.setProperty("danger", True)
        remove_sequence.clicked.connect(self._remove_sequence)
        sequence_buttons.addWidget(remove_sequence)
        sequence_buttons.addStretch(1)
        layout.addLayout(sequence_buttons)
        self.confidence_graph = ConfidenceGraph()
        layout.addWidget(self.confidence_graph)
        self._refresh_sequences()
        return card

    def _hardware_group(self) -> QGroupBox:
        group = QGroupBox("AUTO PERFORMANCE // тест конфигурации ПК")
        layout = QHBoxLayout(group)
        self.hardware_report = QLabel(self.hardware_profile.summary())
        self.hardware_report.setWordWrap(True)
        self.hardware_report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.hardware_report, 1)
        controls = QVBoxLayout()
        self.auto_tune_check = QCheckBox("Автоматически подстраивать качество и нагрузку")
        self.auto_tune_check.setChecked(
            bool(self.store.get("hardware.auto_tune", True))
        )
        self.auto_tune_check.toggled.connect(
            lambda value: self.store.set("hardware.auto_tune", value)
        )
        retest = QPushButton("Повторить тест железа")
        retest.setProperty("secondary", True)
        retest.clicked.connect(self._retest_hardware)
        apply_profile = QPushButton("Применить рекомендуемый профиль")
        apply_profile.clicked.connect(self._activate_hardware_profile)
        controls.addWidget(self.auto_tune_check)
        controls.addWidget(retest)
        controls.addWidget(apply_profile)
        controls.addStretch(1)
        layout.addLayout(controls)
        return group

    def _activate_hardware_profile(self) -> None:
        self.auto_tune_check.setChecked(True)
        self._hardware_retested(self.hardware_profile)

    def _history_panel(self) -> QGroupBox:
        group = QGroupBox("История распознавания и исправления")
        layout = QVBoxLayout(group)
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(
            ["Время", "Рука", "Жест", "Уверенность", "Модель", "Результат"]
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setMaximumHeight(190)
        layout.addWidget(self.history_table)
        row = QHBoxLayout()
        incorrect = QPushButton("Отметить ошибкой")
        incorrect.setProperty("danger", True)
        incorrect.clicked.connect(self._mark_history_incorrect)
        refresh = QPushButton("Обновить")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(self._refresh_history)
        row.addWidget(incorrect)
        row.addWidget(refresh)
        row.addStretch(1)
        layout.addLayout(row)
        self._refresh_history()
        return group

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(max(0, index))

    def _retest_hardware(self) -> None:
        self._set_camera_status("warn", "Тестирую CPU, RAM, GPU и видеопрофиль…")

        def work() -> None:
            self.hardware_ready.emit(detect_hardware(force=True))

        import threading

        threading.Thread(target=work, name="hardware-profiler", daemon=True).start()

    def _hardware_retested(self, profile: HardwareProfile) -> None:
        self.hardware_profile = profile
        self.hardware_report.setText(profile.summary())
        if self.auto_tune_check.isChecked():
            self._apply_hardware_profile(profile)
            self.video_capture.configure(
                profile.camera_width, profile.camera_height, profile.camera_fps
            )
            self.load_controller = AdaptiveLoadController(
                profile.inference_interval_ms, profile.tier
            )
            self._adaptive_interval_ms = profile.inference_interval_ms
            model_index = self.model_combo.findData(profile.camera_model)
            if model_index >= 0:
                self.model_combo.setCurrentIndex(model_index)
        else:
            self._set_camera_status(
                "good", f"Тест завершён: {profile.tier_label}"
            )

    def _auto_switch_profile(self) -> None:
        if not hasattr(self, "auto_profile_check") or not self.auto_profile_check.isChecked():
            return
        profile_id = match_profile(
            foreground_process_name(), parse_profile_rules(self.profile_rules.text())
        )
        if not profile_id or profile_id == self.profile_combo.currentData():
            return
        index = self.profile_combo.findData(profile_id)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)

    def _add_sequence_step(self) -> None:
        gesture = str(self.sequence_gesture.currentData())
        self.sequence_builder.addItem(
            f"{GESTURE_LABELS.get(gesture, gesture)} [{gesture}]"
        )

    def _remove_sequence_step(self) -> None:
        row = self.sequence_builder.currentRow()
        if row < 0:
            row = self.sequence_builder.count() - 1
        if row >= 0:
            self.sequence_builder.takeItem(row)

    def _start_lighting_wizard(self) -> None:
        self._lighting_samples = []
        self._set_camera_status(
            "warn",
            "Мастер камеры: сядь как обычно и покажи открытую ладонь 4 секунды",
        )

    def _save_advanced_controls(self, *_args) -> None:
        if not hasattr(self, "roi_sliders"):
            return
        selected_device = str(self.inference_device.currentData())
        reload_yolo = (
            self.yolo_model is not None
            and self.yolo_model.requested_device != selected_device
        )
        roi = [slider.value() / 100.0 for slider in self.roi_sliders]
        # Keep the rectangle usable even when sliders are dragged to zero.
        roi[2] = max(0.1, min(roi[2], 1.0 - roi[0]))
        roi[3] = max(0.1, min(roi[3], 1.0 - roi[1]))
        self.store.update_many(
            {
                "camera.auto_enhance": self.auto_enhance_check.isChecked(),
                "camera.hud_enabled": self.hud_check.isChecked(),
                "camera.roi_enabled": self.roi_check.isChecked(),
                "camera.roi": roi,
                "camera.inference_device": selected_device,
                "camera.hold_ms": self.hold_ms.value(),
                "camera.confirmation_gesture_enabled": (
                    self.confirm_gesture_check.isChecked()
                ),
                "camera.confirmation_gesture": str(
                    self.confirm_gesture_combo.currentData()
                ),
                "camera.auto_profile_enabled": self.auto_profile_check.isChecked(),
                "camera.profile_app_rules": self.profile_rules.text().strip(),
            }
        )
        if reload_yolo:
            self.yolo_model = None
            QTimer.singleShot(0, self._change_model)

    def _change_profile(self, *_args) -> None:
        profile_id = str(self.profile_combo.currentData() or "work")
        profile = (self.store.get("camera_profiles", {}) or {}).get(profile_id, {})
        values = {"camera.active_profile": profile_id}
        for key in ("stable_frames", "inference_interval_ms", "hold_ms"):
            if key in profile:
                values[f"camera.{key}"] = profile[key]
        self.store.update_many(values)
        if isinstance(profile.get("bindings"), list):
            self.store.replace_section("gesture_bindings", profile["bindings"])
            if hasattr(self, "bindings_panel"):
                self.bindings_panel.load_bindings(profile["bindings"])
        if "hold_ms" in profile:
            self.hold_ms.setValue(int(profile["hold_ms"]))
        if "inference_interval_ms" in profile:
            self.load_controller = AdaptiveLoadController(
                int(profile["inference_interval_ms"]), self.hardware_profile.tier
            )
            self._adaptive_interval_ms = self.load_controller.current_interval_ms
        self._set_camera_status("good", f"Профиль «{self.profile_combo.currentText()}» применён")

    def _save_profile(self) -> None:
        profile_id = str(self.profile_combo.currentData() or "work")
        profiles = self.store.get("camera_profiles", {}) or {}
        profile = dict(profiles.get(profile_id, {}))
        profile.update(
            {
                "name": self.profile_combo.currentText(),
                "stable_frames": int(self.store.get("camera.stable_frames", 3)),
                "inference_interval_ms": int(
                    self.store.get("camera.inference_interval_ms", 100)
                ),
                "hold_ms": self.hold_ms.value(),
                "bindings": self.bindings_panel.values()
                if hasattr(self, "bindings_panel")
                else self.bindings,
            }
        )
        profiles[profile_id] = profile
        self.store.replace_section("camera_profiles", profiles)
        self._set_camera_status("good", "Профиль обновлён")

    def _start_calibration(self) -> None:
        self._calibration_samples = []
        self._set_camera_status(
            "warn", "Калибровка: держи открытую ладонь в рабочем положении"
        )

    def _start_custom_recording(self) -> None:
        name, accepted = QInputDialog.getText(
            self, "Новый жест", "Название пользовательского жеста:"
        )
        if not accepted or not name.strip():
            return
        self._recording_name = " ".join(name.strip().lower().split())
        self._recording_side = ""
        self._recording_samples = []
        self._set_camera_status(
            "warn", f"Запись «{self._recording_name}»: покажи жест, нужно 30 кадров"
        )

    def _add_sequence(self) -> None:
        steps = []
        for index in range(self.sequence_builder.count()):
            text = self.sequence_builder.item(index).text()
            if "[" in text and text.endswith("]"):
                steps.append(text.rsplit("[", 1)[1][:-1].strip().lower())
        if len(steps) < 2:
            QMessageBox.warning(
                self,
                "Последовательность",
                "Добавь минимум два жеста кнопкой «+ Шаг».",
            )
            return
        labels = list(ACTION_LABELS.values())
        label, accepted = QInputDialog.getItem(
            self, "Действие последовательности", "Тип действия:", labels, 3, False
        )
        if not accepted:
            return
        kind = next(key for key, value in ACTION_LABELS.items() if value == label)
        if kind == "system":
            targets = sorted(ActionExecutor.SYSTEM_ACTIONS)
            target, accepted = QInputDialog.getItem(
                self, "Системное действие", "Действие:", targets, 0, False
            )
        else:
            target, accepted = QInputDialog.getText(
                self, "Цель действия", "Полный путь, URL, сочетание или текст:"
            )
        if not accepted or not str(target).strip():
            return
        action = {"type": kind, "target": str(target).strip()}
        try:
            self.executor.validate(action)
        except Exception as exc:
            QMessageBox.warning(self, "Действие не добавлено", str(exc))
            return
        sequences = list(self.store.get("gesture_sequences", []))
        sequences.append(
            {
                "id": f"sequence_{int(time.time())}",
                "name": " → ".join(steps),
                "steps": steps,
                "timeout_sec": 3.0,
                "enabled": True,
                **action,
            }
        )
        self.store.replace_section("gesture_sequences", sequences)
        self.sequence_matcher.configure(sequences)
        self.sequence_builder.clear()
        self._refresh_sequences()
        self._set_camera_status("good", "Последовательность добавлена и активна")

    def _refresh_sequences(self) -> None:
        if not hasattr(self, "sequences_table"):
            return
        self.sequences_table.setRowCount(0)
        for index, sequence in enumerate(self.store.get("gesture_sequences", [])):
            row = self.sequences_table.rowCount()
            self.sequences_table.insertRow(row)
            values = (
                " → ".join(str(step) for step in sequence.get("steps", [])),
                f"{sequence.get('type', '')}: {sequence.get('target', '')}",
                f"{float(sequence.get('timeout_sec', 3.0)):.1f} с",
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setData(Qt.UserRole, index)
                self.sequences_table.setItem(row, column, cell)

    def _remove_sequence(self) -> None:
        rows = {index.row() for index in self.sequences_table.selectedIndexes()}
        remove_indexes = set()
        for row in rows:
            cell = self.sequences_table.item(row, 0)
            if cell is not None:
                remove_indexes.add(int(cell.data(Qt.UserRole)))
        sequences = [
            sequence
            for index, sequence in enumerate(self.store.get("gesture_sequences", []))
            if index not in remove_indexes
        ]
        self.store.replace_section("gesture_sequences", sequences)
        self.sequence_matcher.configure(sequences)
        self._refresh_sequences()

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_table"):
            return
        self.history_table.setRowCount(0)
        start = max(0, len(self.history.items) - 50)
        for index, item in enumerate(self.history.items[start:], start=start):
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            values = (
                time.strftime("%H:%M:%S", time.localtime(item.timestamp)),
                item.side,
                item.gesture,
                f"{item.confidence:.0%}",
                item.model,
                "ошибка" if item.incorrect else (item.action or "распознано"),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setData(Qt.UserRole, index)
                self.history_table.setItem(row, column, cell)

    def _mark_history_incorrect(self) -> None:
        rows = {index.row() for index in self.history_table.selectedIndexes()}
        labels = []
        for row in rows:
            cell = self.history_table.item(row, 0)
            if cell is not None:
                item_index = int(cell.data(Qt.UserRole))
                self.history.mark_incorrect(item_index, True)
                if 0 <= item_index < len(self.history.items):
                    labels.append(self.history.items[item_index].gesture)
        self._refresh_history()
        clip = None
        if (
            rows
            and self.error_clips_check.isChecked()
            and not self.privacy_mode_check.isChecked()
        ):
            clip = self.error_clips.save_recent(
                "error_" + (labels[0] if labels else "gesture")
            )
        self._set_camera_status(
            "good", "Ошибка сохранена — запись готова для будущего переобучения"
            + (f" · ролик: {clip.name}" if clip else "")
        )

    def start(self) -> None:
        if not self.timer.isActive():
            self.timer.start(33)

    def stop(self) -> None:
        self.timer.stop()

    def _toggle_actions(self, enabled: bool) -> None:
        self._set_actions_toggle_text(enabled)
        self.store.set("camera.actions_enabled", enabled)
        if not enabled:
            for state in self._hand_state.values():
                state.update(
                    {
                        "gesture": "",
                        "count": 0,
                        "fired": False,
                        "first_seen": 0.0,
                        "logged": False,
                    }
                )
        self._set_camera_status(
            "good" if enabled else "warn",
            "Жестовые действия включены" if enabled else "Безопасный режим: действия выключены",
        )

    def _set_actions_toggle_text(self, enabled: bool) -> None:
        self.actions_toggle.setText(
            "Управление жестами: вкл" if enabled else "Управление жестами: выкл"
        )
        self.actions_toggle.setProperty("accent", enabled)
        self.actions_toggle.style().unpolish(self.actions_toggle)
        self.actions_toggle.style().polish(self.actions_toggle)

    def _toggle_camera_pause(self) -> None:
        self._camera_paused = not self._camera_paused
        if self._camera_paused:
            self.stop()
            self.pause_button.setText("Продолжить")
            self.performance_status.setText("Камера на паузе")
            self._set_camera_status("warn", "Камера приостановлена")
        else:
            self.pause_button.setText("Пауза")
            self.start()
            self._set_camera_status("good", "Камера продолжила работу")

    def _change_camera(self, *_args) -> None:
        new_id = int(self.camera_combo.currentData())
        if new_id == self.camera_id:
            return
        self.video_capture.release()
        self.camera_id = new_id
        self.video_capture = VideoCapture(
            new_id,
            width=int(self.store.get("camera.capture_width", 960)),
            height=int(self.store.get("camera.capture_height", 540)),
            fps=int(self.store.get("camera.capture_fps", 30)),
        )
        self.store.update_many(
            {"camera.device_index": new_id, "camera.use_network_camera": False}
        )
        self._set_camera_status(
            "good" if self.video_capture.is_open else "bad",
            f"Камера {new_id} подключена" if self.video_capture.is_open else "Камера недоступна",
        )

    def _connect_network_camera(self) -> None:
        url = self.network_camera_url.text().strip()
        if not url:
            QMessageBox.warning(
                self,
                "Камера телефона",
                "Укажи HTTP, HTTPS или RTSP адрес видеопотока.",
            )
            return
        if not url.casefold().startswith(("http://", "https://", "rtsp://")):
            QMessageBox.warning(
                self, "Камера телефона", "Разрешены только HTTP(S) и RTSP адреса."
            )
            return
        candidate = VideoCapture(
            url,
            width=int(self.store.get("camera.capture_width", 960)),
            height=int(self.store.get("camera.capture_height", 540)),
            fps=int(self.store.get("camera.capture_fps", 30)),
        )
        if not candidate.is_open:
            candidate.release()
            self._set_camera_status("bad", "Не удалось открыть поток телефона")
            return
        self.video_capture.release()
        self.video_capture = candidate
        self.store.update_many(
            {"camera.network_url": url, "camera.use_network_camera": True}
        )
        self._set_camera_status("good", "Беспроводная камера подключена")

    def _choose_model(self) -> None:
        yolo = self.model_combo.currentData() == "yolo"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель жестов",
            "",
            "YOLO (*.pt)" if yolo else "Модели (*.onnx *.tflite)",
        )
        if path:
            self.model_path.setText(path)
            if not yolo:
                self.model_combo.setCurrentIndex(self.model_combo.findData("custom"))
            self._change_model()

    def _model_selection_changed(self, *_args) -> None:
        model = str(self.model_combo.currentData())
        if model == "yolo":
            self.model_path.setText(
                str(self.store.get("camera.yolo_model_path", DEFAULT_YOLO_MODEL))
            )
        elif model == "custom":
            self.model_path.setText(
                str(self.store.get("camera.custom_model_path", ""))
            )
        self._sync_model_path_controls()
        self._change_model()

    def _sync_model_path_controls(self) -> None:
        selectable = self.model_combo.currentData() in {"yolo", "custom"}
        self.model_path.setEnabled(selectable)
        self.choose_model_button.setEnabled(selectable)

    def _change_model(self, *_args) -> None:
        model = str(self.model_combo.currentData())
        values = {"camera.model": model}
        if model == "yolo":
            values["camera.yolo_model_path"] = (
                self.model_path.text().strip() or DEFAULT_YOLO_MODEL
            )
        elif model == "custom":
            values["camera.custom_model_path"] = self.model_path.text().strip()
        self.store.update_many(values)
        if model == "rules":
            self.yolo_model = None
            self.recognizer.has_model = False
            self._set_camera_status("good", "Модель: быстрые правила")
            return
        if model == "yolo":
            path = str(values["camera.yolo_model_path"])
            if self.yolo_model is not None and self.yolo_model.loaded:
                resolved = self.yolo_model.resolved_path
                if resolved and Path(resolved).resolve() == Path(path).resolve():
                    self._set_camera_status("good", "YOLO HaGRID уже загружен")
                    return
            if self._model_loading:
                return
            self._model_loading = True
            self._set_camera_status("warn", "Загружаю YOLO HaGRID…")
            try:
                device = (
                    str(self.inference_device.currentData())
                    if hasattr(self, "inference_device")
                    else str(self.store.get("camera.inference_device", "auto"))
                )
                candidate = YOLOStaticModel(path, HAGRID_CLASSES, device=device)
                if candidate.ensure_loaded():
                    self.yolo_model = candidate
                    self.recognizer.has_model = False
                    resolved = candidate.resolved_path or path
                    self.model_path.setText(resolved)
                    self._set_camera_status("good", "YOLO HaGRID загружен")
                else:
                    self.yolo_model = None
                    self._set_camera_status(
                        "warn", "YOLO недоступен — временно использую правила MediaPipe"
                    )
            finally:
                self._model_loading = False
            return
        self.yolo_model = None
        if model == "built_in":
            if self.model_manager.get_current_model() is None:
                try:
                    self.model_manager.load_built_in_model()
                except (OSError, RuntimeError):
                    pass
            self.recognizer.model = self.model_manager.get_current_model()
            self.recognizer.has_model = self.recognizer.model is not None
        else:
            loaded = self.model_manager.load_external_model(self.model_path.text().strip())
            self.recognizer.model = self.model_manager.get_current_model()
            self.recognizer.has_model = loaded
        if self.recognizer.has_model:
            self._set_camera_status("good", "Нейросетевая модель загружена")
        else:
            self.model_combo.setCurrentIndex(self.model_combo.findData("rules"))
            self._set_camera_status("warn", "Модель недоступна — включены быстрые правила")

    def _save_camera_controls(self, *_args) -> None:
        self.store.update_many(
            {
                "camera.brightness": self.brightness.value(),
                "camera.crop_padding": self.padding.value(),
                "camera.mirror": self.mirror.isChecked(),
                "camera.show_hands": self.show_hands.isChecked(),
                "camera.show_face": self.show_face.isChecked(),
                "camera.show_pose": self.show_pose.isChecked(),
                "camera.privacy_mode": self.privacy_mode_check.isChecked(),
                "camera.record_error_clips": self.error_clips_check.isChecked(),
            }
        )

    def _store_changed(self, key: str, _value: Any) -> None:
        if key == "permissions":
            self.executor.set_permissions(self.store.get("permissions", []))
        elif key == "gesture_bindings":
            self._reload_bindings()
        elif key == "gesture_sequences":
            self.sequence_matcher.configure(self.store.get("gesture_sequences", []))
            self._refresh_sequences()

    def _reload_bindings(self) -> None:
        self.bindings = list(self.store.get("gesture_bindings", []))
        self.executor.set_permissions(self.store.get("permissions", []))
        self._set_camera_status("good", "Привязки жестов применены")

    def process_frame(self) -> None:
        if self._processing or self._camera_paused:
            return
        self._processing = True
        try:
            self._process_frame_once()
            self._consecutive_errors = 0
        except Exception as exc:
            self._consecutive_errors += 1
            self.logger.exception("Camera frame processing failed")
            self._set_camera_status("bad", f"Ошибка кадра: {exc}")
            if self._consecutive_errors >= 3:
                # A damaged native graph must not be reused indefinitely.
                try:
                    self.renderer.release()
                except Exception:
                    pass
                try:
                    self.renderer = SkeletonRenderer()
                except Exception as restart_error:
                    self.stop()
                    self._set_camera_status(
                        "bad", f"MediaPipe остановлен: {restart_error}"
                    )
                else:
                    self._consecutive_errors = 0
                    self._set_camera_status("warn", "MediaPipe перезапущен после ошибки")
        finally:
            self._processing = False

    def _process_frame_once(self) -> None:
        ok, frame = self.video_capture.read_frame()
        if not ok or frame is None:
            self.video.setText("Нет сигнала с камеры")
            return
        if self.mirror.isChecked():
            frame = cv2.flip(frame, 1)
        beta = self.brightness.value()
        if beta:
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
        if self.auto_enhance_check.isChecked():
            frame = auto_enhance(frame)
        if self.roi_check.isChecked():
            roi = [slider.value() / 100.0 for slider in self.roi_sliders]
            frame, _ = apply_roi_mask(frame, roi)
        if (
            self.error_clips_check.isChecked()
            and not self.privacy_mode_check.isChecked()
            and self._last_hand_count > 0
        ):
            self.error_clips.push(frame)
        self._collect_lighting_frame(frame)

        now = time.monotonic()
        interval_ms = (
            self._adaptive_interval_ms
            if bool(self.store.get("hardware.auto_tune", True))
            else int(self.store.get("camera.inference_interval_ms", 100))
        )
        interval = interval_ms / 1000.0
        if self.model_combo.currentData() == "yolo":
            interval = max(
                interval,
                int(self.store.get("camera.yolo_inference_interval_ms", 180))
                / 1000.0,
            )
        if now - self._last_inference >= interval:
            self._last_inference = now
            inference_started = time.perf_counter()
            frame, landmarks, _ = self.renderer.process_frame(
                frame,
                self.show_face.isChecked(),
                self.show_hands.isChecked(),
                self.show_pose.isChecked(),
            )
            self._last_pixel_hands = [
                list(points) for points in landmarks.get("hands", [])[:2]
            ]
            self._update_hands(frame, landmarks)
            self._last_inference_ms = (
                time.perf_counter() - inference_started
            ) * 1000.0
        if self.privacy_mode_check.isChecked():
            frame = privacy_silhouette(frame.shape, self._last_pixel_hands)
        if self.hud_check.isChecked():
            self._draw_hud(frame)
        self.set_image(self.video, frame)
        self._update_performance()

    def _update_performance(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        cutoff = now - 1.0
        self._frame_times = [stamp for stamp in self._frame_times if stamp >= cutoff]
        fps = max(0, len(self._frame_times) - 1)
        if bool(self.store.get("hardware.auto_tune", True)):
            self._adaptive_interval_ms = self.load_controller.update(
                fps, self._last_inference_ms, now
            )
        suffix = "рука" if self._last_hand_count == 1 else "руки"
        device = self.yolo_model.device.upper() if self.yolo_model else "CPU"
        self.performance_status.setText(
            f"{fps} FPS · {self._last_hand_count} {suffix} · {device} · "
            f"{self.hardware_profile.tier.upper()}"
        )
        if hasattr(self, "confidence_graph"):
            self.confidence_graph.add_sample(
                self._last_confidence, self._last_inference_ms
            )

    def _draw_hud(self, frame: np.ndarray) -> None:
        model = str(self.model_combo.currentData()).upper()
        lines = [
            f"{model} / {self.yolo_model.device.upper() if self.yolo_model else 'CPU'}",
            f"gesture: {self._last_gesture}  confidence: {self._last_confidence:.0%}",
            f"inference: {self._last_inference_ms:.1f} ms  cooldown: {self._last_cooldown_ms} ms",
            f"hold: {self.hold_ms.value()} ms  profile: {self.profile_combo.currentData()}",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (470, 108), (7, 12, 24), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (24, 34 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (140, 220, 255),
                1,
                cv2.LINE_AA,
            )
        height, width = frame.shape[:2]
        cyan = (80, 222, 242)
        length = max(22, min(width, height) // 18)
        margin = 10
        for x, y, sx, sy in (
            (margin, margin, 1, 1),
            (width - margin, margin, -1, 1),
            (margin, height - margin, 1, -1),
            (width - margin, height - margin, -1, -1),
        ):
            cv2.line(frame, (x, y), (x + sx * length, y), cyan, 1)
            cv2.line(frame, (x, y), (x, y + sy * length), cyan, 1)
        scan_y = int((time.monotonic() * 42) % max(1, height))
        scan_overlay = frame.copy()
        cv2.line(scan_overlay, (0, scan_y), (width, scan_y), (55, 170, 190), 1)
        cv2.addWeighted(scan_overlay, 0.18, frame, 0.82, 0, frame)
        center = (width // 2, height // 2)
        cv2.circle(frame, center, 18, (55, 125, 140), 1)
        cv2.line(frame, (center[0] - 28, center[1]), (center[0] - 9, center[1]), cyan, 1)
        cv2.line(frame, (center[0] + 9, center[1]), (center[0] + 28, center[1]), cyan, 1)

    def _update_hands(self, frame: np.ndarray, landmarks: Dict[str, Any]) -> None:
        seen = set()
        raw_hands = landmarks.get("raw_hands", [])
        handedness = landmarks.get("handedness", [])
        pixel_hands = landmarks.get("hands", [])
        self._last_hand_count = min(2, len(raw_hands))
        for index, raw_hand in enumerate(raw_hands[:2]):
            try:
                if not getattr(raw_hand, "landmark", None) or len(raw_hand.landmark) < 21:
                    continue
                side = str(handedness[index] if index < len(handedness) else "Unknown")
                if side not in {"Left", "Right"}:
                    side = "Left" if "Left" not in seen else "Right"
                seen.add(side)
                coords = pixel_hands[index] if index < len(pixel_hands) else []
                crop = self._crop(frame, coords, self.padding.value())
                if crop is None:
                    continue
                center_x = float(
                    np.mean([point.x for point in raw_hand.landmark[:21]])
                )
                center_y = float(
                    np.mean([point.y for point in raw_hand.landmark[:21]])
                )
                if center_y < 0.30:
                    zone = "top"
                elif center_y > 0.70:
                    zone = "bottom"
                elif center_x < 0.34:
                    zone = "left"
                elif center_x > 0.66:
                    zone = "right"
                else:
                    zone = "center"
                self._hand_zones[side] = zone
                prediction = self._recognize_crop(crop, raw_hand)
                gesture, confidence = prediction
                preview = self.left_hand if side == "Left" else self.right_hand
                label = "Левая" if side == "Left" else "Правая"
                if self.privacy_mode_check.isChecked():
                    preview.update_private(label, gesture, confidence)
                else:
                    preview.update_hand(label, gesture, confidence, crop)
                self._collect_calibration(raw_hand)
                self._collect_custom_sample(side, raw_hand)
                self._stabilize_and_fire(side, gesture, confidence)
            except Exception:
                self.logger.exception("Failed to process detected hand %s", index)
        if "Left" not in seen:
            self.left_hand.clear_hand("Левая")
            self._stabilize_and_fire("Left", "no_gesture", 0.0)
        if "Right" not in seen:
            self.right_hand.clear_hand("Правая")
            self._stabilize_and_fire("Right", "no_gesture", 0.0)

    def _recognize_crop(self, crop: np.ndarray, raw_hand: Any) -> Tuple[str, float]:
        custom = self.custom_gestures.predict(raw_hand)
        if custom is not None and custom[1] >= 0.45:
            return custom
        if self.model_combo.currentData() == "yolo" and self.yolo_model is not None:
            probabilities = self.yolo_model.predict_probs(
                crop, input_is_rgb=True
            )
            if probabilities is not None and probabilities.size:
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])
                if 0 <= index < len(HAGRID_CLASSES) and confidence >= 0.25:
                    gesture = HAGRID_CLASSES[index]
                    return CANONICAL_GESTURE.get(gesture, gesture), confidence
        gesture, _palm_side = self.recognizer.recognize_gesture(crop, raw_hand)
        gesture = CANONICAL_GESTURE.get(gesture, gesture)
        confidence = 0.9 if gesture not in {"unknown", "no_gesture"} else 0.4
        return gesture, confidence

    def _collect_lighting_frame(self, frame: np.ndarray) -> None:
        if self._lighting_samples is None:
            return
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(grey))
        contrast = float(np.std(grey))
        sharpness = float(cv2.Laplacian(grey, cv2.CV_32F).var())
        self._lighting_samples.append((brightness, contrast, round(sharpness)))
        count = len(self._lighting_samples)
        if count < 40:
            if count % 10 == 0:
                self._set_camera_status("warn", f"Анализ камеры: {count}/40")
            return
        values = np.asarray(self._lighting_samples, dtype=np.float32)
        mean_brightness, mean_contrast, mean_sharpness = values.mean(axis=0)
        advice = []
        if mean_brightness < 70:
            advice.append("добавь источник света перед собой")
        elif mean_brightness > 205:
            advice.append("уменьши яркость или убери свет из объектива")
        if mean_contrast < 28:
            advice.append("фон и рука слишком похожи по тону")
        if mean_sharpness < 45:
            advice.append("протри объектив и зафиксируй камеру")
        if self._last_hand_count == 0:
            advice.append("поставь камеру так, чтобы кисть полностью входила в кадр")
        message = (
            f"Свет {mean_brightness:.0f}/255 · контраст {mean_contrast:.0f} · "
            f"резкость {mean_sharpness:.0f}. "
            + ("; ".join(advice) if advice else "Положение и свет хорошие.")
        )
        self._lighting_samples = None
        QMessageBox.information(self, "Мастер камеры и света", message)
        self._set_camera_status("good", "Анализ света и положения завершён")

    def _collect_calibration(self, raw_hand: Any) -> None:
        if self._calibration_samples is None or len(self._calibration_samples) >= 40:
            return
        points = getattr(raw_hand, "landmark", None)
        if not points:
            return
        xy = np.asarray([[point.x, point.y] for point in points], dtype=np.float32)
        spread = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
        if spread > 0.03:
            self._calibration_samples.append(spread)
        if len(self._calibration_samples) == 40:
            median = float(np.median(self._calibration_samples))
            scale = max(0.5, min(2.0, 0.32 / max(0.05, median)))
            recommended_padding = max(12, min(70, round(34 * scale)))
            self.padding.setValue(recommended_padding)
            self.store.set("camera.calibration_scale", scale)
            self._set_camera_status(
                "good",
                f"Калибровка завершена: масштаб {scale:.2f}, отступ {recommended_padding}",
            )
            self._calibration_samples = None

    def _collect_custom_sample(self, side: str, raw_hand: Any) -> None:
        if not self._recording_name or len(self._recording_samples) >= 30:
            return
        if not self._recording_side:
            self._recording_side = side
        if side != self._recording_side:
            return
        vector = normalized_landmarks(raw_hand)
        if vector is None:
            return
        self._recording_samples.append(vector.tolist())
        count = len(self._recording_samples)
        self._set_camera_status(
            "warn", f"Запись «{self._recording_name}»: {count}/30 кадров"
        )
        if count == 30:
            name = self._recording_name
            try:
                saved = self.custom_gestures.train(name, self._recording_samples)
                if name not in GESTURES:
                    GESTURES.append(name)
                self._set_camera_status(
                    "good", f"Жест «{name}» обучен на {saved} кадрах"
                )
            except Exception as exc:
                self._set_camera_status("bad", f"Не удалось обучить жест: {exc}")
            finally:
                self._recording_name = ""
                self._recording_side = ""
                self._recording_samples = []

    @staticmethod
    def _crop(frame: np.ndarray, points: List[Tuple[int, int]], padding: int) -> Optional[np.ndarray]:
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        height, width = frame.shape[:2]
        x1, x2 = max(0, min(xs) - padding), min(width, max(xs) + padding)
        y1, y2 = max(0, min(ys) - padding), min(height, max(ys) + padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (210, 160), interpolation=cv2.INTER_AREA)

    def _stabilize_and_fire(self, side: str, gesture: str, confidence: float) -> None:
        state = self._hand_state[side]
        now = time.monotonic()
        if gesture in {"unknown", "no_gesture"}:
            state["count"] = max(0, int(state["count"]) - 1)
            if state["count"] == 0:
                state.update(
                    {
                        "gesture": "",
                        "fired": False,
                        "first_seen": 0.0,
                        "logged": False,
                    }
                )
            return
        if state["gesture"] == gesture:
            state["count"] += 1
        else:
            state.update(
                {
                    "gesture": gesture,
                    "count": 1,
                    "fired": False,
                    "first_seen": now,
                    "logged": False,
                }
            )
        stable_frames = int(self.store.get("camera.stable_frames", 3))
        held_ms = (now - float(state["first_seen"])) * 1000.0
        if state["count"] < stable_frames or held_ms < self.hold_ms.value():
            return
        self._last_gesture = gesture
        self._last_confidence = confidence
        if not state["logged"]:
            self.history.add(
                side,
                gesture,
                confidence,
                str(self.model_combo.currentData()),
            )
            state["logged"] = True
            self._refresh_history()
        sequence = self.sequence_matcher.feed(gesture, now)
        if sequence is not None and self.actions_toggle.isChecked():
            self._execute_or_confirm(sequence, side, f"цепочка {sequence.get('name', '')}")
            state["fired"] = True
            return
        if state["fired"]:
            return
        if not self.actions_toggle.isChecked():
            return
        confirmation = str(
            self.store.get("camera.confirmation_gesture", "ok")
        )
        if gesture == confirmation and self._pending_gesture_action is not None:
            action, pending_side, expires = self._pending_gesture_action
            self._pending_gesture_action = None
            if now <= expires:
                self._perform_gesture_action(action, pending_side, "подтверждено")
            else:
                self._set_camera_status("warn", "Подтверждение жеста истекло")
            state["fired"] = True
            return
        for binding in self.bindings:
            if not binding.get("enabled", True) or binding.get("gesture") != gesture:
                continue
            zone = str(binding.get("zone", "any"))
            if zone != "any" and zone != self._hand_zones.get(side, "any"):
                continue
            if confidence < float(binding.get("confidence", 0.8)):
                continue
            key = (side, gesture)
            cooldown = int(binding.get("cooldown_ms", 1200)) / 1000.0
            self._last_cooldown_ms = max(
                0, round((cooldown - (now - self._last_action.get(key, 0.0))) * 1000)
            )
            if now - self._last_action.get(key, 0.0) < cooldown:
                continue
            self._execute_or_confirm(binding, side, gesture)
            state["fired"] = True
            break

    def _execute_or_confirm(
        self, action: Dict[str, Any], side: str, label: str
    ) -> None:
        confirmation_enabled = bool(
            self.store.get("camera.confirmation_gesture_enabled", False)
        )
        confirmation = str(self.store.get("camera.confirmation_gesture", "ok"))
        if confirmation_enabled and label != confirmation:
            self._pending_gesture_action = (
                dict(action),
                side,
                time.monotonic() + 5.0,
            )
            self._set_camera_status(
                "warn", f"{label}: покажи «{GESTURE_LABELS.get(confirmation, confirmation)}»"
            )
            return
        self._perform_gesture_action(action, side, label)

    def _perform_gesture_action(
        self, action: Dict[str, Any], side: str, label: str
    ) -> None:
        try:
            message = self.executor.execute(action)
            gesture = str(action.get("gesture", label))
            self._last_action[(side, gesture)] = time.monotonic()
            self._last_cooldown_ms = int(action.get("cooldown_ms", 0))
            self.history.add(
                side,
                gesture,
                self._last_confidence,
                str(self.model_combo.currentData()),
                action=message,
            )
            self._refresh_history()
            self._set_camera_status("good", f"{side}: {message}")
        except Exception as exc:
            self._set_camera_status("bad", str(exc))

    def _set_camera_status(self, level: str, text: str) -> None:
        names = {"good": "StatusGood", "warn": "StatusWarn", "bad": "StatusBad"}
        self.camera_status.setObjectName(names.get(level, "StatusWarn"))
        self.camera_status.setText(text)
        self.camera_status.style().unpolish(self.camera_status)
        self.camera_status.style().polish(self.camera_status)
        self.status_changed.emit(level, text)

    @staticmethod
    def set_image(
        label: QLabel, rgb: np.ndarray, *, allow_upscale: bool = True
    ) -> None:
        rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888
        ).copy()
        target = label.size()
        if not allow_upscale:
            target.setWidth(min(target.width(), width))
            target.setHeight(min(target.height(), height))
        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                target, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def cleanup(self) -> None:
        self.stop()
        self.video_capture.release()
        self.renderer.release()


class CameraLabWidget(QWidget):
    """Advanced camera tools separated from the live monitoring view."""

    def __init__(self, camera: CameraWidget, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(14)
        title = QLabel("CAMERA LAB // CONFIG")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Модели, ROI, обучение, привязки жестов и журнал распознавания"
        )
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(camera.model_details_card)
        layout.addWidget(camera.advanced_card)
        layout.addWidget(camera.bindings_panel)
        layout.addWidget(camera.history_panel)
        layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger("MainWindow")
        self.store = ConfigStore()
        self.setWindowTitle("Axi Control · Жесты и Jarvis")
        self.resize(1440, 920)
        self.setMinimumSize(1080, 720)
        self._quitting = False
        self._cleaned = False

        self.camera_page = CameraWidget(self.store)
        self.assistant_page = AssistantWindow(self.store)
        self.camera_lab_page = CameraLabWidget(self.camera_page)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.camera_page)
        self.stack.addWidget(self.assistant_page)
        self.stack.addWidget(self.camera_lab_page)
        self.stack.addWidget(self.assistant_page.advanced_settings_page)
        self._build_shell()
        self._install_shortcuts()
        self._setup_tray()
        self._setup_global_hotkey()
        self.store.changed.connect(self._global_setting_changed)
        self.camera_page.status_changed.connect(self._status)
        self.assistant_page.status_changed.connect(self._status)
        self.camera_page.start()
        app = QApplication.instance()
        if app:
            apply_theme(app, str(self.store.get("ui.theme", "dark")))
        self.assistant_page.set_theme(str(self.store.get("ui.theme", "dark")))

    def _build_shell(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(208)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 18, 16, 18)
        side.setSpacing(7)
        brand = QLabel("J.A.R.V.I.S")
        brand.setObjectName("Brand")
        brand.setAlignment(Qt.AlignCenter)
        caption = QLabel("AXI // CONTROL SYSTEM")
        caption.setObjectName("BrandCaption")
        caption.setAlignment(Qt.AlignCenter)
        side.addWidget(brand)
        side.addWidget(caption)
        orb_row = QHBoxLayout()
        orb_row.addStretch(1)
        orb_row.addWidget(JarvisOrb())
        orb_row.addStretch(1)
        side.addLayout(orb_row)
        self.core_status = QLabel("● CORE ONLINE")
        self.core_status.setObjectName("CoreStatus")
        self.core_status.setAlignment(Qt.AlignCenter)
        side.addWidget(self.core_status)
        profile = self.camera_page.hardware_profile
        self.hardware_badge = QLabel(
            f"{profile.tier.upper()} // {profile.score:02d}\n"
            f"{profile.logical_cores}T · {profile.ram_gb:.0f}GB · "
            f"{'CUDA' if profile.cuda_available else 'CPU'}"
        )
        self.hardware_badge.setObjectName("HardwareBadge")
        self.hardware_badge.setAlignment(Qt.AlignCenter)
        self.hardware_badge.setToolTip(profile.summary())
        side.addWidget(self.hardware_badge)
        side.addSpacing(10)
        self.camera_button = QPushButton("01  CAMERA")
        self.assistant_button = QPushButton("02  ASSISTANT")
        self.camera_lab_button = QPushButton("03  CAMERA LAB")
        self.jarvis_core_button = QPushButton("04  JARVIS CORE")
        self._page_buttons = (
            self.camera_button,
            self.assistant_button,
            self.camera_lab_button,
            self.jarvis_core_button,
        )
        for button in self._page_buttons:
            button.setCheckable(True)
            button.setMinimumHeight(46)
            button.setObjectName("NavButton")
            side.addWidget(button)
        self.camera_button.setChecked(True)
        self.camera_button.clicked.connect(lambda: self._show_page(0))
        self.assistant_button.clicked.connect(lambda: self._show_page(1))
        self.camera_lab_button.clicked.connect(lambda: self._show_page(2))
        self.jarvis_core_button.clicked.connect(lambda: self._show_page(3))
        side.addStretch(1)
        self.clock_label = QLabel()
        self.clock_label.setObjectName("HudClock")
        self.clock_label.setAlignment(Qt.AlignCenter)
        side.addWidget(self.clock_label)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()
        shortcuts = QLabel(
            "CTRL 1—4  НАВИГАЦИЯ\nCTRL+T  ТЕМА\nCTRL+P  ПАУЗА"
        )
        shortcuts.setProperty("muted", True)
        shortcuts.setToolTip("Быстрые клавиши доступны из любой вкладки")
        shortcuts.setAlignment(Qt.AlignCenter)
        side.addWidget(shortcuts)
        self.theme_button = QPushButton("LIGHT MODE")
        self.theme_button.setProperty("secondary", True)
        self.theme_button.clicked.connect(self._toggle_theme)
        side.addWidget(self.theme_button)
        self.global_status = QLabel("Готово")
        self.global_status.setObjectName("StatusGood")
        self.global_status.setWordWrap(True)
        side.addWidget(self.global_status)
        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self._sync_theme_button()

    def _update_clock(self) -> None:
        self.clock_label.setText(time.strftime("%H:%M:%S\n%d.%m.%Y"))

    def _install_shortcuts(self) -> None:
        bindings = (
            ("Ctrl+1", lambda: self._show_page(0)),
            ("Ctrl+2", lambda: self._show_page(1)),
            ("Ctrl+3", lambda: self._show_page(2)),
            ("Ctrl+4", lambda: self._show_page(3)),
            ("Ctrl+T", self._toggle_theme),
            ("Ctrl+P", self.camera_page._toggle_camera_pause),
        )
        self._shortcuts = []
        for sequence, callback in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _setup_tray(self) -> None:
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray.setToolTip("Axi Control · жесты и Jarvis")
        menu = QMenu(self)
        show_action = QAction("Показать Axi Control", self)
        show_action.triggered.connect(self._show_from_tray)
        ptt_action = QAction("Push-to-Talk", self)
        ptt_action.triggered.connect(self.assistant_page.push_to_talk)
        quit_action = QAction("Выйти", self)
        quit_action.triggered.connect(self._quit_app)
        menu.addAction(show_action)
        menu.addAction(ptt_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._show_from_tray()
            if reason == QSystemTrayIcon.DoubleClick
            else None
        )
        self.tray.show()

    def _setup_global_hotkey(self) -> None:
        sequence = str(
            self.store.get("assistant.push_to_talk_hotkey", "ctrl+alt+j")
        ).strip()
        self.global_hotkey = GlobalHotkey(sequence, self)
        self.global_hotkey.activated.connect(self.assistant_page.push_to_talk)
        self.global_hotkey.status.connect(
            lambda ok, message: self._status("good" if ok else "warn", message)
        )
        self.global_hotkey.start()

    def _global_setting_changed(self, key: str, _value: Any) -> None:
        if key != "assistant.push_to_talk_hotkey" or self._cleaned:
            return
        self.global_hotkey.stop()
        self.global_hotkey.sequence = str(
            self.store.get("assistant.push_to_talk_hotkey", "ctrl+alt+j")
        ).strip()
        self.global_hotkey.start()

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit_app(self) -> None:
        self._quitting = True
        self._cleanup()
        self.tray.hide()
        app = QApplication.instance()
        if app:
            app.quit()

    def _cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.global_hotkey.stop()
        self.camera_page.cleanup()
        self.assistant_page.cleanup()

    def _show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self._page_buttons):
            button.setChecked(index == button_index)

    def _toggle_theme(self) -> None:
        current = str(self.store.get("ui.theme", "dark"))
        theme = "light" if current == "dark" else "dark"
        self.store.set("ui.theme", theme)
        app = QApplication.instance()
        if app:
            apply_theme(app, theme)
        self.assistant_page.set_theme(theme)
        self._sync_theme_button()

    def _sync_theme_button(self) -> None:
        dark = str(self.store.get("ui.theme", "dark")) == "dark"
        self.theme_button.setText("LIGHT MODE" if dark else "DARK MODE")

    def _status(self, level: str, text: str) -> None:
        names = {"good": "StatusGood", "warn": "StatusWarn", "bad": "StatusBad"}
        self.global_status.setObjectName(names.get(level, "StatusWarn"))
        self.global_status.setText(text)
        self.global_status.style().unpolish(self.global_status)
        self.global_status.style().polish(self.global_status)

    def closeEvent(self, event) -> None:
        if (
            not self._quitting
            and bool(self.store.get("assistant.close_to_tray", True))
            and self.tray.isVisible()
        ):
            self.hide()
            self.tray.showMessage(
                "Axi Control работает в фоне",
                "Ctrl+Alt+J включает Push-to-Talk. Выход — через меню трея.",
                QSystemTrayIcon.Information,
                2500,
            )
            event.ignore()
            return
        self._cleanup()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    wheel_guard = ParameterWheelGuard(app)
    app.installEventFilter(wheel_guard)
    app._parameter_wheel_guard = wheel_guard
    window = MainWindow()
    window.show()
    # Let Ctrl+C from run.bat close native camera/MediaPipe resources cleanly.
    signal.signal(signal.SIGINT, lambda *_args: window._quit_app())
    signal_pump = QTimer()
    signal_pump.timeout.connect(lambda: None)
    signal_pump.start(400)
    exit_code = app.exec_()
    window._cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
