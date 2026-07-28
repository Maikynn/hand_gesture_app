"""Modern two-tab interface for camera gestures and the integrated assistant."""

from __future__ import annotations

import os
import sys
import time

import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from assistant.action_executor import ACTION_LABELS, ActionExecutor
from camera.skeleton_renderer import SkeletonRenderer
from camera.video_capture import VideoCapture
from hand_processing.gesture_fusion import GestureFusion
from ui.assistant_window import AssistantWindow
from utils.config_store import load_config, save_config
from utils.logger import setup_logger


DARK_STYLE = """
QWidget { background:#11141b; color:#ecf1ff; font:10pt 'Segoe UI'; }
QMainWindow, QTabWidget::pane { background:#11141b; }
QTabWidget::pane { border:0; }
QTabBar::tab { background:#181d27; color:#8f9bb3; padding:12px 28px; margin:5px 3px; border-radius:8px; }
QTabBar::tab:selected { background:#27334f; color:#ffffff; }
QGroupBox, #chatPanel { background:#181d27; border:1px solid #293142; border-radius:12px; margin-top:12px; padding:14px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#aeb9d0; }
QLineEdit, QTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background:#10141c; border:1px solid #30394b; border-radius:7px; padding:7px; selection-background-color:#526fdd; }
QHeaderView::section { background:#202735; color:#aeb9d0; border:0; padding:7px; }
QPushButton { background:#252d3c; border:1px solid #354158; border-radius:7px; padding:8px 13px; }
QPushButton:hover { background:#303b50; }
QPushButton#primaryButton { background:#5877e8; border:0; color:white; font-weight:600; }
QLabel#pageTitle { font-size:22px; font-weight:700; }
QLabel#muted { color:#8f9bb3; }
QSlider::groove:horizontal { height:5px; background:#30394b; border-radius:2px; }
QSlider::handle:horizontal { width:16px; margin:-6px 0; border-radius:8px; background:#6c8cff; }
"""

LIGHT_STYLE = """
QWidget { background:#f4f6fb; color:#172033; font:10pt 'Segoe UI'; }
QTabWidget::pane { border:0; }
QTabBar::tab { background:#e8ecf5; color:#667085; padding:12px 28px; margin:5px 3px; border-radius:8px; }
QTabBar::tab:selected { background:#dbe4ff; color:#2349b5; }
QGroupBox, #chatPanel { background:white; border:1px solid #dce2ed; border-radius:12px; margin-top:12px; padding:14px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; }
QLineEdit, QTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background:#fbfcff; border:1px solid #cfd7e6; border-radius:7px; padding:7px; selection-background-color:#b8c8ff; }
QHeaderView::section { background:#edf1f8; border:0; padding:7px; }
QPushButton { background:#edf1f8; border:1px solid #d1d9e8; border-radius:7px; padding:8px 13px; }
QPushButton:hover { background:#e1e7f2; }
QPushButton#primaryButton { background:#526fdd; border:0; color:white; font-weight:600; }
QLabel#pageTitle { font-size:22px; font-weight:700; }
QLabel#muted { color:#667085; }
"""


class CameraWidget(QWidget):
    gesture_detected = pyqtSignal(str, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.logger = setup_logger("CameraWidget")
        self.config = config
        self.camera_cfg = config["camera"]
        self.gesture_cfg = config["gesture"]
        self.capture = VideoCapture(int(self.camera_cfg.get("device_index", 0)))
        self.renderer = SkeletonRenderer()
        # MediaPipe is the default backend.  Avoid loading PyTorch/torchvision and
        # allocating a neural model until the user explicitly selects one.
        self.fusion = GestureFusion(load_static=False)
        self.executor = ActionExecutor(config["voice"].get("allowed_apps", []))
        self.last_actions = {}
        self.last_inference = 0.0
        self.cached_gestures = {}
        self.gesture_candidates = {}
        self.active_gestures = {}
        self.is_running = True
        self._build_ui()
        self._load_bindings()
        self._apply_capture_settings()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(33)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 20)
        header = QHBoxLayout()
        texts = QVBoxLayout()
        title = QLabel("Камера и жесты")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Покажите руки камере и назначьте действия на удобные жесты.")
        subtitle.setObjectName("muted")
        texts.addWidget(title)
        texts.addWidget(subtitle)
        self.camera_state = QLabel("● Камера активна")
        self.toggle_camera = QPushButton("Остановить")
        self.toggle_camera.clicked.connect(self._toggle_camera)
        header.addLayout(texts)
        header.addStretch()
        header.addWidget(self.camera_state)
        header.addWidget(self.toggle_camera)
        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(16)
        preview_col = QVBoxLayout()
        self.video = QLabel("Камера недоступна\nВыберите другое устройство справа")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(720, 420)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet("background:#080a0f;border-radius:14px;color:#6f7a91;")
        preview_col.addWidget(self.video, 1)
        hands = QHBoxLayout()
        self.hand_views = {}
        for hand in ("Left", "Right"):
            card = QFrame()
            card.setObjectName("chatPanel")
            card_layout = QHBoxLayout(card)
            image = QLabel("Нет руки")
            image.setAlignment(Qt.AlignCenter)
            image.setFixedSize(150, 120)
            image.setStyleSheet("background:#0b0e14;border-radius:9px;color:#6f7a91;")
            label = QLabel(("Левая" if hand == "Left" else "Правая") + ": —")
            label.setStyleSheet("font-size:15px;font-weight:600;")
            card_layout.addWidget(image)
            card_layout.addWidget(label, 1)
            hands.addWidget(card)
            self.hand_views[hand] = (image, label)
        preview_col.addLayout(hands)
        content.addLayout(preview_col, 3)
        content.addWidget(self._controls(), 2)
        root.addLayout(content, 1)

    def _controls(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        source = QGroupBox("Источник и распознавание")
        form = QVBoxLayout(source)
        self.camera_combo = QComboBox()
        for index in range(5):
            self.camera_combo.addItem(f"Камера {index}", index)
        self.camera_combo.setCurrentIndex(int(self.camera_cfg.get("device_index", 0)))
        self.camera_combo.currentIndexChanged.connect(self._change_camera)
        self.model_combo = QComboBox()
        self.model_combo.addItem("MediaPipe — без модели", "mediapipe")
        self.model_combo.addItem("MobileNetV3 (HaGRID)", "mobilenet")
        self.model_combo.addItem("YOLOv8 (HaGRID)", "yolo")
        self.model_combo.addItem("MobileNetV3 + YOLO", "combined")
        idx = self.model_combo.findData(
            self.gesture_cfg.get("static_model_id", "mediapipe")
        )
        self.model_combo.setCurrentIndex(max(0, idx))
        self.model_combo.currentIndexChanged.connect(self._change_model)
        self._apply_model(self.model_combo.currentData())
        form.addWidget(QLabel("Камера"))
        form.addWidget(self.camera_combo)
        form.addWidget(QLabel("Модель жестов"))
        form.addWidget(self.model_combo)
        bright_row = QHBoxLayout()
        bright_row.addWidget(QLabel("Яркость"))
        self.brightness = QSlider(Qt.Horizontal)
        self.brightness.setRange(-100, 100)
        self.brightness.setValue(int(self.camera_cfg.get("brightness", 0)))
        self.brightness.valueChanged.connect(self._apply_brightness)
        self.brightness.sliderReleased.connect(self._save_camera)
        bright_row.addWidget(self.brightness)
        form.addLayout(bright_row)
        layout.addWidget(source)

        bindings = QGroupBox("Что делать при жесте")
        box = QVBoxLayout(bindings)
        tip = QLabel(
            "Для запуска программы сначала разрешите её на вкладке «Помощник»."
        )
        tip.setWordWrap(True)
        tip.setObjectName("muted")
        box.addWidget(tip)
        self.bindings = QTableWidget(0, 4)
        self.bindings.setHorizontalHeaderLabels(
            ["Рука", "Жест", "Действие", "Путь / URL"]
        )
        self.bindings.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bindings.verticalHeader().setVisible(False)
        self.bindings.setSelectionBehavior(QAbstractItemView.SelectRows)
        box.addWidget(self.bindings)
        buttons = QHBoxLayout()
        add = QPushButton("＋ Привязка")
        add.clicked.connect(self._add_binding)
        remove = QPushButton("Удалить")
        remove.clicked.connect(self._remove_binding)
        save = QPushButton("Сохранить")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save_bindings)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addWidget(save)
        box.addLayout(buttons)
        layout.addWidget(bindings, 1)
        return panel

    def _load_bindings(self):
        for binding in self.gesture_cfg.get("bindings", []):
            self._add_binding(binding)

    def _add_binding(self, binding=None):
        binding = binding or {
            "hand": "Любая",
            "gesture": "palm",
            "action": "none",
            "value": "",
        }
        row = self.bindings.rowCount()
        self.bindings.insertRow(row)
        hand = QComboBox()
        hand.addItems(["Любая", "Левая", "Правая"])
        hand.setCurrentText(binding.get("hand", "Любая"))
        gesture = QComboBox()
        gesture.addItems(self.fusion.classes)
        gesture.setCurrentText(binding.get("gesture", "palm"))
        action = QComboBox()
        for key, label in ACTION_LABELS.items():
            action.addItem(label, key)
        action.setCurrentIndex(max(0, action.findData(binding.get("action", "none"))))
        self.bindings.setCellWidget(row, 0, hand)
        self.bindings.setCellWidget(row, 1, gesture)
        self.bindings.setCellWidget(row, 2, action)
        self.bindings.setItem(row, 3, QTableWidgetItem(binding.get("value", "")))

    def _remove_binding(self):
        if self.bindings.currentRow() >= 0:
            self.bindings.removeRow(self.bindings.currentRow())

    def save_bindings(self):
        values = []
        for row in range(self.bindings.rowCount()):
            item = self.bindings.item(row, 3)
            values.append(
                {
                    "hand": self.bindings.cellWidget(row, 0).currentText(),
                    "gesture": self.bindings.cellWidget(row, 1).currentText(),
                    "action": self.bindings.cellWidget(row, 2).currentData(),
                    "value": item.text().strip() if item else "",
                }
            )
        self.gesture_cfg["bindings"] = values
        save_config(self.config)
        QMessageBox.information(self, "Готово", "Привязки жестов сохранены.")

    def _change_camera(self):
        self.camera_cfg["device_index"] = self.camera_combo.currentData()
        save_config(self.config)
        self.capture.release()
        self.capture = VideoCapture(self.camera_combo.currentData())
        self._apply_capture_settings()

    def _change_model(self):
        model = self.model_combo.currentData()
        self.gesture_cfg["static_model_id"] = model
        self._apply_model(model)
        save_config(self.config)

    def _apply_model(self, model):
        if model == "mediapipe":
            self.fusion.static_weight, self.fusion.mediapipe_weight = 0.0, 1.0
            return
        self.fusion.static_weight, self.fusion.mediapipe_weight = 0.8, 0.2
        if model == "yolo":
            self.fusion.use_yolo_static()
        elif model == "combined":
            self.fusion.use_combined_static()
        else:
            self.fusion.use_mobilenet_static()

    def _save_camera(self):
        self.camera_cfg["brightness"] = self.brightness.value()
        save_config(self.config)
        self._apply_capture_settings()

    def _apply_brightness(self, value):
        self.camera_cfg["brightness"] = value
        if self.capture.cap is not None:
            self.capture.cap.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def _apply_capture_settings(self):
        if self.capture.cap is None:
            return
        self.capture.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, self.camera_cfg.get("width", 1280)
        )
        self.capture.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, self.camera_cfg.get("height", 720)
        )
        self.capture.cap.set(
            cv2.CAP_PROP_BRIGHTNESS, self.camera_cfg.get("brightness", 0)
        )

    def _toggle_camera(self):
        self.is_running = not self.is_running
        self.toggle_camera.setText("Остановить" if self.is_running else "Запустить")
        self.camera_state.setText(
            "● Камера активна" if self.is_running else "○ Камера остановлена"
        )
        if not self.is_running:
            self.video.clear()
            self.video.setText("Камера остановлена")

    @staticmethod
    def _crop(frame, points, padding=30):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(xs) - padding), min(w, max(xs) + padding)
        y1, y2 = max(0, min(ys) - padding), min(h, max(ys) + padding)
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (180, 140)) if crop.size else None

    @staticmethod
    def _set_image(label, rgb, keep=True):
        h, w, channels = rgb.shape
        image = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888).copy()
        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                label.size(),
                Qt.KeepAspectRatio if keep else Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def process_frame(self):
        if not self.is_running:
            return
        success, frame = self.capture.read_frame()
        if not success:
            self.camera_state.setText("● Камера недоступна")
            return
        if self.camera_cfg.get("flip", True):
            frame = cv2.flip(frame, 1)
        processed, landmarks, _ = self.renderer.process_frame(frame, False, True, False)
        found = set()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        now = time.monotonic()
        inference_due = now - self.last_inference >= 0.10
        for index, raw in enumerate(landmarks["raw_hands"]):
            detected = (
                landmarks["handedness"][index]
                if index < len(landmarks["handedness"])
                else "Unknown"
            )
            hand = (
                ("Right" if detected == "Left" else "Left")
                if self.camera_cfg.get("flip", True)
                else detected
            )
            if hand not in self.hand_views:
                continue
            found.add(hand)
            if inference_due or hand not in self.cached_gestures:
                gesture, confidence, _ = self.fusion.predict(
                    frame_bgr, raw.landmark, w, h
                )
                self.cached_gestures[hand] = (gesture, confidence)
            else:
                gesture, confidence = self.cached_gestures[hand]
            crop = self._crop(
                frame,
                landmarks["hands"][index],
                int(self.gesture_cfg.get("hand_crop_padding", 30)),
            )
            image, label = self.hand_views[hand]
            if crop is not None:
                self._set_image(image, crop, False)
            russian_hand = "Левая" if hand == "Left" else "Правая"
            label.setText(f"{russian_hand}: {gesture}\nУверенность {confidence:.0%}")
            self.gesture_detected.emit(russian_hand, gesture)
            if inference_due and self._gesture_became_stable(hand, gesture):
                self._run_binding(russian_hand, gesture, confidence)
        if inference_due:
            self.last_inference = now
        for hand, (image, label) in self.hand_views.items():
            if hand not in found:
                image.clear()
                image.setText("Нет руки")
                label.setText(("Левая" if hand == "Left" else "Правая") + ": —")
                self.cached_gestures.pop(hand, None)
                self.gesture_candidates.pop(hand, None)
                self.active_gestures.pop(hand, None)
        self._set_image(self.video, processed)
        self.camera_state.setText(f"● Камера активна · рук: {len(found)}")

    def _run_binding(self, hand, gesture, confidence):
        if confidence < float(self.gesture_cfg.get("confidence_threshold", 0.65)):
            return
        now = time.monotonic()
        cooldown = float(self.gesture_cfg.get("action_cooldown", 2))
        for binding in self.gesture_cfg.get("bindings", []):
            if binding.get("gesture") != gesture or binding.get("hand") not in {
                "Любая",
                hand,
            }:
                continue
            key = f"{hand}:{gesture}:{binding.get('action')}:{binding.get('value')}"
            if now - self.last_actions.get(key, 0) < cooldown:
                continue
            self.last_actions[key] = now
            self.executor.update_allowed_apps(
                self.config["voice"].get("allowed_apps", [])
            )
            ok, message = self.executor.execute(
                binding.get("action", "none"), binding.get("value", "")
            )
            self.camera_state.setText(("✓ " if ok else "⚠ ") + message)

    def _gesture_became_stable(self, hand, gesture):
        """Trigger once after three equal samples; re-arm after gesture change."""
        previous, count = self.gesture_candidates.get(hand, (None, 0))
        count = count + 1 if previous == gesture else 1
        self.gesture_candidates[hand] = (gesture, count)
        if count < 3 or self.active_gestures.get(hand) == gesture:
            return False
        self.active_gestures[hand] = gesture
        return gesture != "no_gesture"

    def cleanup(self):
        self.timer.stop()
        self.capture.release()
        self.renderer.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.setWindowTitle("Axi Gesture — управление жестами и голосом")
        self.resize(1500, 920)
        self.setMinimumSize(1100, 720)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 6, 12, 12)
        top = QHBoxLayout()
        brand = QLabel("AXI GESTURE")
        brand.setStyleSheet("font-weight:800;font-size:16px;letter-spacing:2px;")
        self.theme = QPushButton(
            "☀ Светлая тема"
            if self.config["appearance"].get("theme") == "dark"
            else "☾ Темная тема"
        )
        self.theme.clicked.connect(self.toggle_theme)
        top.addWidget(brand)
        top.addStretch()
        top.addWidget(self.theme)
        layout.addLayout(top)
        self.tabs = QTabWidget()
        self.camera = CameraWidget(self.config)
        self.assistant = AssistantWindow(self.config)
        self.assistant.config_saved.connect(self._refresh_config)
        self.tabs.addTab(self.camera, "  Камера  ")
        self.tabs.addTab(self.assistant, "  Помощник  ")
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(container)
        self.apply_theme()

    def _refresh_config(self, config):
        """Apply assistant permissions to gesture actions without a restart."""
        self.config = config
        self.camera.config = config
        self.camera.camera_cfg = config["camera"]
        self.camera.gesture_cfg = config["gesture"]
        self.camera.executor.update_allowed_apps(
            config["voice"].get("allowed_apps", [])
        )

    def apply_theme(self):
        QApplication.instance().setStyleSheet(
            DARK_STYLE
            if self.config["appearance"].get("theme", "dark") == "dark"
            else LIGHT_STYLE
        )

    def toggle_theme(self):
        current = self.config["appearance"].get("theme", "dark")
        self.config["appearance"]["theme"] = "light" if current == "dark" else "dark"
        save_config(self.config)
        self.theme.setText("☀ Светлая тема" if current == "light" else "☾ Темная тема")
        self.apply_theme()

    def closeEvent(self, event):
        self.camera.cleanup()
        self.assistant.cleanup()
        event.accept()


def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
