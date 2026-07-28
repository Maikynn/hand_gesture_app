from __future__ import annotations

import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Load MediaPipe's native runtime before Qt. On Windows, loading Qt5 first can
# make _framework_bindings fail or crash later inside MSVCP140.dll.
from camera.skeleton_renderer import SkeletonRenderer

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from assistant.actions import ACTION_LABELS, ActionExecutor
from camera.video_capture import VideoCapture
from hand_processing.gesture_recognizer import GestureRecognizer
from ui.assistant_window import AssistantWindow
from ui.theme import apply_theme
from utils.config_store import ConfigStore
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


class HandPreview(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.image = QLabel("Рука не обнаружена")
        self.image.setObjectName("HandSurface")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setFixedSize(154, 124)
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
        CameraWidget.set_image(self.image, crop)


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
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Вкл.", "Жест", "Действие", "Значение", "Точность", "Пауза, мс"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)
        for binding in store.get("gesture_bindings", []):
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
        action_map = {
            key: label
            for key, label in ACTION_LABELS.items()
            if key in {"application", "url", "hotkey", "system"}
        }
        self.table.setCellWidget(row, 2, self._combo(action_map, str(binding.get("type", "system"))))
        self.table.setItem(row, 3, QTableWidgetItem(str(binding.get("target", ""))))
        self.table.setItem(row, 4, QTableWidgetItem(str(binding.get("confidence", 0.8))))
        self.table.setItem(row, 5, QTableWidgetItem(str(binding.get("cooldown_ms", 1200))))
        self.table.selectRow(row)

    def remove_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def choose_application(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        kind = self.table.cellWidget(row, 2)
        if not isinstance(kind, QComboBox) or kind.currentData() != "application":
            QMessageBox.information(self, "Тип действия", "Сначала выбери тип «Приложение».")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Выберите приложение", "", "Программы (*.exe)")
        if path:
            self.table.item(row, 3).setText(path)

    def values(self) -> List[Dict[str, Any]]:
        result = []
        for row in range(self.table.rowCount()):
            gesture = self.table.cellWidget(row, 1)
            kind = self.table.cellWidget(row, 2)
            assert isinstance(gesture, QComboBox) and isinstance(kind, QComboBox)
            result.append(
                {
                    "enabled": self.table.item(row, 0).checkState() == Qt.Checked,
                    "gesture": str(gesture.currentData()),
                    "type": str(kind.currentData()),
                    "target": self.table.item(row, 3).text().strip(),
                    "confidence": float(self.table.item(row, 4).text().replace(",", ".")),
                    "cooldown_ms": int(self.table.item(row, 5).text()),
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

    def __init__(self, store: ConfigStore | None = None, parent=None):
        super().__init__(parent)
        self.store = store or ConfigStore()
        self.logger = setup_logger("CameraWidget")
        self.settings = self.store.get("camera", {}) or {}
        self.camera_id = int(self.settings.get("device_index", 0))
        self.video_capture = VideoCapture(self.camera_id)
        self.renderer = SkeletonRenderer()
        self.model_manager = ModelManager()
        self.recognizer = GestureRecognizer(self.model_manager)
        self.executor = ActionExecutor(self.store.get("permissions", []))
        self.bindings = list(self.store.get("gesture_bindings", []))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self._last_inference = 0.0
        self._processing = False
        self._camera_paused = False
        self._consecutive_errors = 0
        self._frame_times: List[float] = []
        self._last_hand_count = 0
        self._hand_state: Dict[str, Dict[str, Any]] = {
            "Left": {"gesture": "", "count": 0, "fired": False},
            "Right": {"gesture": "", "count": 0, "fired": False},
        }
        self._last_action: Dict[Tuple[str, str], float] = {}
        self._build_ui()
        self.store.changed.connect(self._store_changed)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Камера")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Покажи руки — Jarvis распознает жесты и выполнит только разрешённые действия")
        subtitle.setObjectName("PageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.performance_status = QLabel("Ожидание кадра")
        self.performance_status.setObjectName("StatusNeutral")
        header.addWidget(self.performance_status)
        self.actions_toggle = QPushButton("Управление жестами: выкл")
        self.actions_toggle.setCheckable(True)
        self.actions_toggle.setChecked(bool(self.settings.get("actions_enabled", False)))
        self.actions_toggle.setProperty("secondary", True)
        self.actions_toggle.setToolTip(
            "Распознавание работает всегда. Действия выполняются только когда переключатель включён."
        )
        self.actions_toggle.toggled.connect(self._toggle_actions)
        self._set_actions_toggle_text(self.actions_toggle.isChecked())
        header.addWidget(self.actions_toggle)
        self.pause_button = QPushButton("Пауза")
        self.pause_button.setProperty("secondary", True)
        self.pause_button.clicked.connect(self._toggle_camera_pause)
        header.addWidget(self.pause_button)
        self.camera_status = QLabel("Камера подключена" if self.video_capture.is_open else "Нет камеры")
        self.camera_status.setObjectName(
            "StatusGood" if self.video_capture.is_open else "StatusBad"
        )
        header.addWidget(self.camera_status)
        layout.addLayout(header)

        self.video = QLabel("Подключаю камеру…")
        self.video.setObjectName("VideoSurface")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(760, 440)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video, 1)

        hands = QHBoxLayout()
        hands.setSpacing(14)
        self.left_hand = HandPreview("Левая")
        self.right_hand = HandPreview("Правая")
        hands.addWidget(self.left_hand)
        hands.addWidget(self.right_hand)
        layout.addLayout(hands)
        layout.addWidget(self._controls())
        self.bindings_panel = GestureBindingsPanel(self.store, self.executor)
        self.bindings_panel.saved.connect(self._reload_bindings)
        layout.addWidget(self.bindings_panel)
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
        self.model_combo.addItem("Встроенная ONNX/TFLite", "built_in")
        self.model_combo.addItem("Своя ONNX/TFLite", "custom")
        index = self.model_combo.findData(str(self.settings.get("model", "rules")))
        self.model_combo.setCurrentIndex(max(0, index))
        self.model_combo.currentIndexChanged.connect(self._change_model)
        form.addRow("Модель", self.model_combo)

        model_path_row = QHBoxLayout()
        self.model_path = QLineEdit(str(self.settings.get("custom_model_path", "")))
        choose = QPushButton("…")
        choose.setFixedWidth(42)
        choose.setProperty("secondary", True)
        choose.clicked.connect(self._choose_model)
        model_path_row.addWidget(self.model_path, 1)
        model_path_row.addWidget(choose)
        form.addRow("Файл модели", model_path_row)
        row.addLayout(form, 2)

        sliders = QFormLayout()
        self.brightness = QSlider(Qt.Horizontal)
        self.brightness.setRange(-100, 100)
        self.brightness.setValue(int(self.settings.get("brightness", 0)))
        self.brightness.valueChanged.connect(self._save_camera_controls)
        sliders.addRow("Яркость", self.brightness)
        self.padding = QSlider(Qt.Horizontal)
        self.padding.setRange(5, 80)
        self.padding.setValue(int(self.settings.get("crop_padding", 34)))
        self.padding.valueChanged.connect(self._save_camera_controls)
        sliders.addRow("Отступ руки", self.padding)
        row.addLayout(sliders, 2)

        toggles = QVBoxLayout()
        self.mirror = QCheckBox("Зеркальное изображение")
        self.show_hands = QCheckBox("Рисовать скелет рук")
        self.show_face = QCheckBox("Сетка лица")
        self.show_pose = QCheckBox("Скелет тела")
        self.mirror.setChecked(bool(self.settings.get("mirror", True)))
        self.show_hands.setChecked(bool(self.settings.get("show_hands", True)))
        self.show_face.setChecked(bool(self.settings.get("show_face", False)))
        self.show_pose.setChecked(bool(self.settings.get("show_pose", False)))
        for widget in (self.mirror, self.show_hands, self.show_face, self.show_pose):
            widget.toggled.connect(self._save_camera_controls)
            toggles.addWidget(widget)
        row.addLayout(toggles, 1)
        return card

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
                state.update({"gesture": "", "count": 0, "fired": False})
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
        self.video_capture = VideoCapture(new_id)
        self.store.set("camera.device_index", new_id)
        self._set_camera_status(
            "good" if self.video_capture.is_open else "bad",
            f"Камера {new_id} подключена" if self.video_capture.is_open else "Камера недоступна",
        )

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель жестов", "", "Модели (*.onnx *.tflite)"
        )
        if path:
            self.model_path.setText(path)
            self.model_combo.setCurrentIndex(self.model_combo.findData("custom"))
            self._change_model()

    def _change_model(self, *_args) -> None:
        model = str(self.model_combo.currentData())
        self.store.update_many(
            {
                "camera.model": model,
                "camera.custom_model_path": self.model_path.text().strip(),
            }
        )
        if model == "rules":
            self.recognizer.has_model = False
            self._set_camera_status("good", "Модель: быстрые правила")
            return
        if model == "built_in":
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
            }
        )

    def _store_changed(self, key: str, _value: Any) -> None:
        if key == "permissions":
            self.executor.set_permissions(self.store.get("permissions", []))
        elif key == "gesture_bindings":
            self._reload_bindings()

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

        now = time.monotonic()
        interval = int(self.store.get("camera.inference_interval_ms", 100)) / 1000.0
        if now - self._last_inference >= interval:
            self._last_inference = now
            frame, landmarks, _ = self.renderer.process_frame(
                frame,
                self.show_face.isChecked(),
                self.show_hands.isChecked(),
                self.show_pose.isChecked(),
            )
            self._update_hands(frame, landmarks)
        self.set_image(self.video, frame)
        self._update_performance()

    def _update_performance(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        cutoff = now - 1.0
        self._frame_times = [stamp for stamp in self._frame_times if stamp >= cutoff]
        fps = max(0, len(self._frame_times) - 1)
        suffix = "рука" if self._last_hand_count == 1 else "руки"
        self.performance_status.setText(f"{fps} FPS · {self._last_hand_count} {suffix}")

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
                gesture, _palm_side = self.recognizer.recognize_gesture(crop, raw_hand)
                gesture = CANONICAL_GESTURE.get(gesture, gesture)
                confidence = 0.9 if gesture not in {"unknown", "no_gesture"} else 0.4
                preview = self.left_hand if side == "Left" else self.right_hand
                preview.update_hand(
                    "Левая" if side == "Left" else "Правая", gesture, confidence, crop
                )
                self._stabilize_and_fire(side, gesture, confidence)
            except Exception:
                self.logger.exception("Failed to process detected hand %s", index)
        if "Left" not in seen:
            self.left_hand.clear_hand("Левая")
            self._stabilize_and_fire("Left", "no_gesture", 0.0)
        if "Right" not in seen:
            self.right_hand.clear_hand("Правая")
            self._stabilize_and_fire("Right", "no_gesture", 0.0)

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
        if gesture in {"unknown", "no_gesture"}:
            state["count"] = max(0, int(state["count"]) - 1)
            if state["count"] == 0:
                state["gesture"] = ""
                state["fired"] = False
            return
        if state["gesture"] == gesture:
            state["count"] += 1
        else:
            state.update({"gesture": gesture, "count": 1, "fired": False})
        stable_frames = int(self.store.get("camera.stable_frames", 3))
        if state["count"] < stable_frames or state["fired"]:
            return
        if not self.actions_toggle.isChecked():
            return
        for binding in self.bindings:
            if not binding.get("enabled", True) or binding.get("gesture") != gesture:
                continue
            if confidence < float(binding.get("confidence", 0.8)):
                continue
            key = (side, gesture)
            now = time.monotonic()
            cooldown = int(binding.get("cooldown_ms", 1200)) / 1000.0
            if now - self._last_action.get(key, 0.0) < cooldown:
                continue
            try:
                message = self.executor.execute(binding)
                self._last_action[key] = now
                state["fired"] = True
                self._set_camera_status("good", f"{side}: {message}")
            except Exception as exc:
                state["fired"] = True
                self._set_camera_status("bad", str(exc))
            break

    def _set_camera_status(self, level: str, text: str) -> None:
        names = {"good": "StatusGood", "warn": "StatusWarn", "bad": "StatusBad"}
        self.camera_status.setObjectName(names.get(level, "StatusWarn"))
        self.camera_status.setText(text)
        self.camera_status.style().unpolish(self.camera_status)
        self.camera_status.style().polish(self.camera_status)
        self.status_changed.emit(level, text)

    @staticmethod
    def set_image(label: QLabel, rgb: np.ndarray) -> None:
        rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888
        ).copy()
        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def cleanup(self) -> None:
        self.stop()
        self.video_capture.release()
        self.renderer.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger("MainWindow")
        self.store = ConfigStore()
        self.setWindowTitle("Axi Control · Жесты и Jarvis")
        self.resize(1440, 920)
        self.setMinimumSize(1080, 720)

        self.camera_page = CameraWidget(self.store)
        self.assistant_page = AssistantWindow(self.store)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.camera_page)
        self.stack.addWidget(self.assistant_page)
        self._build_shell()
        self._install_shortcuts()
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
        sidebar.setFixedWidth(230)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 22, 18, 20)
        side.setSpacing(9)
        brand = QLabel("AXI CONTROL")
        brand.setObjectName("Brand")
        caption = QLabel("Жесты + Jarvis")
        caption.setObjectName("BrandCaption")
        side.addWidget(brand)
        side.addWidget(caption)
        side.addSpacing(20)
        self.camera_button = QPushButton("◉  Камера\nжесты и действия")
        self.assistant_button = QPushButton("✦  Помощник\nчат и настройки")
        for button in (self.camera_button, self.assistant_button):
            button.setCheckable(True)
            button.setMinimumHeight(60)
            button.setProperty("secondary", True)
            side.addWidget(button)
        self.camera_button.setChecked(True)
        self.camera_button.clicked.connect(lambda: self._show_page(0))
        self.assistant_button.clicked.connect(lambda: self._show_page(1))
        side.addStretch(1)
        shortcuts = QLabel("Ctrl+1  Камера\nCtrl+2  Помощник\nCtrl+T  Тема\nCtrl+P  Пауза")
        shortcuts.setProperty("muted", True)
        shortcuts.setToolTip("Быстрые клавиши доступны из любой вкладки")
        side.addWidget(shortcuts)
        self.theme_button = QPushButton("☀ Светлая тема")
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

    def _install_shortcuts(self) -> None:
        bindings = (
            ("Ctrl+1", lambda: self._show_page(0)),
            ("Ctrl+2", lambda: self._show_page(1)),
            ("Ctrl+T", self._toggle_theme),
            ("Ctrl+P", self.camera_page._toggle_camera_pause),
        )
        self._shortcuts = []
        for sequence, callback in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.camera_button.setChecked(index == 0)
        self.assistant_button.setChecked(index == 1)

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
        self.theme_button.setText("☀ Светлая тема" if dark else "☾ Тёмная тема")

    def _status(self, level: str, text: str) -> None:
        names = {"good": "StatusGood", "warn": "StatusWarn", "bad": "StatusBad"}
        self.global_status.setObjectName(names.get(level, "StatusWarn"))
        self.global_status.setText(text)
        self.global_status.style().unpolish(self.global_status)
        self.global_status.style().polish(self.global_status)

    def closeEvent(self, event) -> None:
        self.camera_page.cleanup()
        self.assistant_page.cleanup()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    # Let Ctrl+C from run.bat close native camera/MediaPipe resources cleanly.
    signal.signal(signal.SIGINT, lambda *_args: window.close())
    signal_pump = QTimer()
    signal_pump.timeout.connect(lambda: None)
    signal_pump.start(400)
    exit_code = app.exec_()
    window.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
