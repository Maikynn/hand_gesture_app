#!/usr/bin/env python3
"""
Main UI window for the Hand Gesture Application.
Integrates camera feed, skeleton rendering, gesture recognition,
settings, and assistant functionality.
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Tuple, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QSlider, QGroupBox, QSizePolicy, QMessageBox,
    QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# Import project modules
from utils.logger import setup_logger
from utils.model_manager import ModelManager
from camera.video_capture import VideoCapture
from camera.skeleton_renderer import SkeletonRenderer
from hand_processing.hand_crop import HandCropProcessor
from hand_processing.gesture_recognizer import GestureRecognizer
from hand_processing.gesture_fusion import GestureFusion
from ui.settings_window import SettingsWindow
from ui.assistant_window import AssistantWindow

class CameraWidget(QWidget):
    """
    Widget that displays camera feed with skeleton overlay.
    Handles hand crop extraction and gesture recognition.

    The static gesture model is provided by :class:`GestureFusion`, whose
    backend (MobileNetV3 / YOLO) can be switched live from the UI without
    restarting the application.
    """
    gesture_detected = pyqtSignal(str, str)  # gesture_name, backend

    def __init__(self, camera_id: int = 0, parent=None):
        super().__init__(parent)
        self.logger = setup_logger("CameraWidget")
        self.camera_id = camera_id
        self.is_running = False
        self.parent_settings = None

        # Initialize components
        self.video_capture = VideoCapture(camera_id)
        self.skeleton_renderer = SkeletonRenderer()
        self.hand_crop_processor = HandCropProcessor()
        self.model_manager = ModelManager()
        self.gesture_recognizer = GestureRecognizer(self.model_manager)
        # Fusion module: switchable MobileNetV3 / YOLO static backends.
        self.fusion = GestureFusion()

        # UI elements
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")

        # Hand crop display
        self.crop_label = QLabel()
        self.crop_label.setFixedSize(150, 150)
        self.crop_label.setStyleSheet("border: 2px solid white;")

        # Gesture display
        self.gesture_label = QLabel("Gesture: None")
        self.gesture_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.gesture_label.setAlignment(Qt.AlignCenter)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)

        # Hand crop area
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(QLabel("Hand Crop:"))
        crop_layout.addWidget(self.crop_label)
        main_layout.addLayout(crop_layout)

        # Gesture display
        main_layout.addWidget(self.gesture_label)

        # Static gesture model backend switch (live, no restart)
        self.backend_group = QGroupBox("Статическая модель жестов (live)")
        backend_layout = QVBoxLayout()
        self.radio_mobilenet = QRadioButton("MobileNetV3 (HaGRID)")
        self.radio_yolo = QRadioButton("YOLO (HaGRID)")
        self.radio_mobilenet.setChecked(True)
        self.backend_button_group = QButtonGroup(self)
        self.backend_button_group.addButton(self.radio_mobilenet, 0)
        self.backend_button_group.addButton(self.radio_yolo, 1)
        self.backend_button_group.buttonClicked.connect(self._on_backend_selected)
        backend_layout.addWidget(self.radio_mobilenet)
        backend_layout.addWidget(self.radio_yolo)
        self.backend_status_label = QLabel("Активна: MobileNetV3 (HaGRID)")
        self.backend_status_label.setAlignment(Qt.AlignCenter)
        backend_layout.addWidget(self.backend_status_label)
        self.refresh_yolo_btn = QPushButton("Обновить YOLO")
        self.refresh_yolo_btn.clicked.connect(lambda: self._set_static_backend('yolo', force=True))
        backend_layout.addWidget(self.refresh_yolo_btn)
        self.backend_group.setLayout(backend_layout)
        main_layout.addWidget(self.backend_group)

        self.setLayout(main_layout)

        # Timer for video processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def start(self):
        """Start video processing."""
        if not self.is_running:
            self.is_running = True
            self.timer.start(30)  # ~33 FPS
            self.logger.info("Camera widget started")

    def stop(self):
        """Stop video processing."""
        if self.is_running:
            self.is_running = False
            self.timer.stop()
            self.logger.info("Camera widget stopped")

    def process_frame(self):
        """Process next video frame."""
        if not self.is_running:
            return

        success, frame = self.video_capture.read_frame()
        if not success:
            return

        # Keep original frame for drawing
        display_frame = frame.copy()

        # Get settings (with safe defaults)
        show_face = False
        show_hand = False
        show_pose = False
        if self.parent_settings is not None:
            show_face = getattr(self.parent_settings, 'show_face_skeleton', False)
            show_hand = getattr(self.parent_settings, 'show_hand_skeleton', False)
            show_pose = getattr(self.parent_settings, 'show_body_skeleton', False)

        # Process with skeleton renderer (now returns 3 values)
        processed_frame, landmarks, _ = self.skeleton_renderer.process_frame(
            frame, show_face, show_hand, show_pose
        )

        # Extract hand crop if hand detected
        if landmarks['hands']:
            try:
                hand_crop = self.hand_crop_processor.extract_hand_crop(
                    processed_frame, landmarks
                )
                if hand_crop is not None:
                    # Update crop label
                    h, w, ch = hand_crop.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(hand_crop.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.crop_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))

                    # Run gesture recognition via the fusion module
                    # (switchable MobileNetV3 / YOLO static backends + MediaPipe)
                    raw_hand = landmarks['raw_hands'][0] if landmarks['raw_hands'] else None
                    if raw_hand is not None:
                        lm_list = raw_hand.landmark
                        fh, fw, _ = frame.shape
                        # GestureFusion expects a BGR frame (it converts internally).
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        gesture_name, conf, _ = self.fusion.predict(
                            frame_bgr, lm_list, fw, fh, None
                        )
                        backend = self.fusion._static_backend
                        self.gesture_detected.emit(gesture_name, backend)
                        self.gesture_label.setText(
                            f"Gesture: {gesture_name} ({conf:.2f}) [{backend}]"
                        )
                    else:
                        self.gesture_label.setText("Gesture: None")
            except Exception as e:
                self.logger.error(f"Error in hand crop processing: {e}")
        else:
            # Clear crop label
            self.crop_label.clear()
            self.gesture_label.setText("Gesture: None")

        # Convert frame to QImage for display (RGB -> BGR for Qt)
        bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        h, w, ch = bgr_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(bgr_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit label while maintaining aspect ratio
        # Use label size if valid, otherwise use frame dimensions
        label_size = self.video_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            target_size = label_size
        else:
            target_size = pixmap.size()
        self.video_label.setPixmap(pixmap.scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    # ----------------------- static backend switching -----------------------
    def _on_backend_selected(self, button):
        if button is self.radio_yolo:
            self._set_static_backend('yolo')
        else:
            self._set_static_backend('mobilenet')

    def _set_static_backend(self, name: str, force: bool = False):
        """Switch the static gesture model backend live (no restart)."""
        try:
            if name == 'yolo':
                self.fusion.use_yolo_static()
            else:
                self.fusion.use_mobilenet_static()
        except Exception as e:
            self.logger.error(f"backend switch error: {e}")
        self._update_backend_status()

    def _update_backend_status(self):
        backend = self.fusion._static_backend
        if backend == 'yolo':
            yolo = self.fusion._yolo
            if yolo is not None and getattr(yolo, 'loaded', False):
                self.backend_status_label.setText("Активна: YOLO (HaGRID)")
            else:
                self.backend_status_label.setText(
                    "YOLO: модель не обучена — fallback на MobileNetV3"
                )
        else:
            self.backend_status_label.setText("Активна: MobileNetV3 (HaGRID)")

    def set_parent_settings(self, settings_obj):
        """Set reference to settings object for accessing user preferences."""
        self.parent_settings = settings_obj

    def cleanup(self):
        """Release resources."""
        self.stop()
        self.video_capture.release()
        self.skeleton_renderer.release()


class MainWindow(QMainWindow):
    """
    Main application window with three main sections:
    1. Settings
    2. Camera (with skeleton rendering)
    3. Assistant
    """
    def __init__(self):
        super().__init__()
        self.logger = setup_logger("MainWindow")

        # Initialize model manager
        self.model_manager = ModelManager()

        # Create UI components
        self.settings_window = SettingsWindow()
        self.camera_widget = CameraWidget(camera_id=0)
        self.assistant_window = AssistantWindow()

        # Set up main stack widget
        self.stack = QStackedWidget()
        self.stack.addWidget(self.settings_window)
        self.stack.addWidget(self.camera_widget)
        self.stack.addWidget(self.assistant_window)
        self.setCentralWidget(self.stack)

        # Set window properties
        self.setWindowTitle("Hand Gesture Application")
        self.resize(1200, 800)

        # Setup navigation buttons
        self.setup_navigation()

        # Connect signals
        self.settings_window.apply_settings.connect(self.apply_user_settings)
        self.camera_widget.gesture_detected.connect(self.handle_gesture)

        # Load default settings
        self.apply_user_settings(self.settings_window.settings)

        # Setup camera
        self.camera_widget.set_parent_settings(self.settings_window)

        # Load model (non-fatal if missing)
        try:
            if self.model_manager.get_current_model() is None:
                self.logger.warning("No gesture model loaded - using rule-based recognition")
        except Exception as e:
            self.logger.error(f"Model loading issue: {e}")

        # Start camera
        self.camera_widget.start()

    def setup_navigation(self):
        """Create navigation controls for switching between sections."""
        nav_widget = QWidget()
        nav_layout = QHBoxLayout()

        self.settings_btn = QPushButton("Settings")
        self.camera_btn = QPushButton("Camera")
        self.assistant_btn = QPushButton("Assistant")

        self.settings_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.settings_window))
        self.camera_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.camera_widget))
        self.assistant_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.assistant_window))

        nav_layout.addWidget(self.settings_btn)
        nav_layout.addWidget(self.camera_btn)
        nav_layout.addWidget(self.assistant_btn)
        nav_widget.setLayout(nav_layout)

        self.setMenuWidget(nav_widget)

    def apply_user_settings(self, settings: dict):
        """Apply user settings to appropriate components."""
        # Update camera widget settings reference
        if hasattr(self.camera_widget, 'set_parent_settings'):
            self.camera_widget.set_parent_settings(self.settings_window)

        # Update assistant settings if needed
        if hasattr(self.assistant_window, 'apply_settings'):
            try:
                self.assistant_window.apply_settings(settings)
            except Exception as e:
                self.logger.error(f"Error applying assistant settings: {e}")

        self.logger.info(f"Applied settings: {list(settings.keys())}")

    def handle_gesture(self, gesture_name: str, palm_side: str):
        """Handle recognized gesture."""
        self.logger.info(f"Detected gesture: {gesture_name} ({palm_side})")
        # Could trigger specific actions based on gesture

    def closeEvent(self, event):
        """Handle window close event."""
        self.logger.info("Main window closing")
        # Cleanup camera resources
        if hasattr(self, 'camera_widget'):
            self.camera_widget.cleanup()
        # Cleanup assistant resources
        if hasattr(self, 'assistant_window'):
            self.assistant_window.cleanup()
        event.accept()

def main():
    """Application entry point."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
