#!/usr/bin/env python3
"""
Settings Window for the Hand Gesture Application.
Provides UI for camera selection, skeleton toggles, calibration, and model loading.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QSlider, QPushButton, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

class SettingsWindow(QWidget):
    """Signal emitted when settings should be applied."""
    apply_settings = pyqtSignal(dict)
    
    SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.logger = logging.getLogger("SettingsWindow")
        self.settings: Dict[str, Any] = {}
        self.load_settings()
        self.init_ui()

    def load_settings(self):
        """Load settings from JSON file or create defaults."""
        if self.SETTINGS_FILE.exists():
            try:
                with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
            except Exception as e:
                self.logger.error("Failed to load settings: %s", e)
                self.settings = {}
        else:
            # Default settings
            self.settings = {
                "camera_id": 0,
                "show_face_skeleton": False,
                "show_hand_skeleton": False,
                "show_body_skeleton": False,
                "line_thickness": 2,
                "calibration_padding": 30,
                "custom_model_path": ""
            }

    def save_settings(self):
        """Save current settings to JSON file."""
        try:
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.logger.error("Failed to save settings: %s", e)

    def init_ui(self):
        """Create and layout UI elements."""
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Camera selection
        self.create_camera_section()
        # Skeleton toggles
        self.create_skeleton_section()
        # Calibration and model loading
        self.create_controls_section()
        # Apply/Save buttons
        self.create_action_buttons()

        self.setLayout(self.layout)

    def create_camera_section(self):
        """Camera device selector."""
        group = QGroupBox("Camera Settings")
        layout = QVBoxLayout()
        label = QLabel("Camera Device:")
        self.camera_combo = QComboBox()
        # Populate with available devices (placeholder)
        self.camera_combo.addItems(["0", "1", "2"])
        self.camera_combo.setCurrentText(str(self.settings.get("camera_id", 0)))
        layout.addWidget(label)
        layout.addWidget(self.camera_combo)
        group.setLayout(layout)
        self.layout.addWidget(group)

    def create_skeleton_section(self):
        """Checkboxes for skeleton visualization."""
        group = QGroupBox("Skeleton Visualization")
        layout = QVBoxLayout()
        self.face_check = QCheckBox("Show Face Skeleton")
        self.hand_check = QCheckBox("Show Hand Skeleton")
        self.body_check = QCheckBox("Show Body Skeleton")
        self.face_check.setChecked(self.settings.get("show_face_skeleton", False))
        self.hand_check.setChecked(self.settings.get("show_hand_skeleton", False))
        self.body_check.setChecked(self.settings.get("show_body_skeleton", False))
        layout.addWidget(self.face_check)
        layout.addWidget(self.hand_check)
        layout.addWidget(self.body_check)
        group.setLayout(layout)
        self.layout.addWidget(group)

    def create_controls_section(self):
        """Line thickness slider and calibration."""
        group = QGroupBox("Visualization Controls")
        layout = QVBoxLayout()
        # Line thickness slider
        thickness_label = QLabel("Skeleton Line Thickness:")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 10)
        self.thickness_slider.setValue(self.settings.get("line_thickness", 2))
        layout.addWidget(thickness_label)
        layout.addWidget(self.thickness_slider)
        # Calibration padding
        padding_label = QLabel("Hand Crop Padding:")
        self.padding_slider = QSlider(Qt.Horizontal)
        self.padding_slider.setRange(10, 100)
        self.padding_slider.setValue(self.settings.get("calibration_padding", 30))
        layout.addWidget(padding_label)
        layout.addWidget(self.padding_slider)
        group.setLayout(layout)
        self.layout.addWidget(group)

    def create_action_buttons(self):
        """Buttons for loading custom model and applying settings."""
        hbox = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Custom Model")
        self.load_model_btn.clicked.connect(self.load_custom_model)
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self.apply_settings_clicked)
        hbox.addWidget(self.load_model_btn)
        hbox.addWidget(self.apply_btn)
        self.layout.addLayout(hbox)

    @pyqtSlot()
    def load_custom_model(self):
        """Open file dialog to select a custom model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Model", "", "Model Files (*.tflite *.onnx);;All Files (*)"
        )
        if file_path:
            self.settings["custom_model_path"] = file_path
            self.logger.info("Selected custom model: %s", file_path)

    @pyqtSlot()
    def apply_settings_clicked(self):
        """Apply current UI settings to the application."""
        self.settings["camera_id"] = int(self.camera_combo.currentText())
        self.settings["show_face_skeleton"] = self.face_check.isChecked()
        self.settings["show_hand_skeleton"] = self.hand_check.isChecked()
        self.settings["show_body_skeleton"] = self.body_check.isChecked()
        self.settings["line_thickness"] = self.thickness_slider.value()
        self.settings["calibration_padding"] = self.padding_slider.value()
        self.save_settings()
        # Emit signal for parent to handle
        self.apply_settings.emit(self.settings)
