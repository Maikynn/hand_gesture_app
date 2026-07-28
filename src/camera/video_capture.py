#!/usr/bin/env python3
"""
Video capture module for the Hand Gesture Application.
Handles webcam acquisition and frame retrieval with error handling.
"""

import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

class VideoCapture:
    """
    Wrapper around OpenCV VideoCapture with robust error handling.
    """
    def __init__(
        self,
        camera_id: Union[int, str] = 0,
        *,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
    ):
        self.camera_id = camera_id
        self.width = max(0, int(width))
        self.height = max(0, int(height))
        self.fps = max(0, int(fps))
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        self._last_reconnect = 0.0
        self.open_camera()
        
    def open_camera(self) -> bool:
        """
        Open camera device with fallback to default if specified ID fails.
        """
        if self.cap is not None:
            self.cap.release()
        
        # Try specified camera ID first
        if isinstance(self.camera_id, str):
            candidates = [
                (self.camera_id, cv2.CAP_FFMPEG),
                (self.camera_id, cv2.CAP_ANY),
            ]
        else:
            candidates = [
                (self.camera_id, cv2.CAP_DSHOW),
                (self.camera_id, cv2.CAP_MSMF),
                (self.camera_id, cv2.CAP_ANY),
            ]
            if self.camera_id != 0:
                candidates.extend(
                    [
                        (0, cv2.CAP_DSHOW),
                        (0, cv2.CAP_MSMF),
                        (0, cv2.CAP_ANY),
                    ]
                )
        self.cap = None
        for source, backend in candidates:
            try:
                if isinstance(source, str) and backend == cv2.CAP_FFMPEG:
                    candidate = cv2.VideoCapture(
                        source,
                        backend,
                        [
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                            3000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                            2000,
                        ],
                    )
                else:
                    candidate = cv2.VideoCapture(source, backend)
            except cv2.error:
                continue
            if candidate.isOpened():
                self.cap = candidate
                break
            candidate.release()
        if self.cap is None:
            self.is_open = False
            return False
        self._apply_capture_profile()
        
        self.is_open = True
        return True

    def _apply_capture_profile(self) -> None:
        if self.cap is None:
            return
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if isinstance(self.camera_id, int):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def configure(self, width: int, height: int, fps: int) -> None:
        self.width = max(0, int(width))
        self.height = max(0, int(height))
        self.fps = max(0, int(fps))
        self._apply_capture_profile()
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with error recovery.
        """
        if not self.is_open or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            # Do not spin on a disconnected camera.
            now = time.monotonic()
            if now - self._last_reconnect >= 1.0:
                self._last_reconnect = now
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
