#!/usr/bin/env python3
"""
Video capture module for the Hand Gesture Application.
Handles webcam acquisition and frame retrieval with error handling.
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple


class VideoCapture:
    """
    Wrapper around OpenCV VideoCapture with robust error handling.
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        self._last_recovery_attempt = 0.0
        self.open_camera()

    def open_camera(self) -> bool:
        """
        Open camera device with fallback to default if specified ID fails.
        """
        if self.cap is not None:
            self.cap.release()

        # Try specified camera ID first
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            # Fallback to default camera if specified ID fails
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.is_open = False
                return False

        self.is_open = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with error recovery.
        """
        if not self.is_open or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            # Avoid repeatedly reopening a disconnected camera every 33 ms.
            now = time.monotonic()
            if now - self._last_recovery_attempt >= 1.0:
                self._last_recovery_attempt = now
                self.cap.release()
                self.open_camera()
            return False, None

        # Convert BGR to RGB for MediaPipe compatibility
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, frame_rgb

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get current frame dimensions.
        """
        if self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def release(self):
        """
        Release camera resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
