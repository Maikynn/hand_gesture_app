#!/usr/bin/env python3
"""Stable, lazy MediaPipe tracking for the camera page."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


EMPTY_LANDMARKS = {
    "face": [],
    "hands": [],
    "pose": [],
    "raw_hands": [],
    "handedness": [],
}


class SkeletonRenderer:
    """Owns MediaPipe graphs and guarantees serialized native inference.

    Only the hand graph is created at startup. Face and pose graphs are loaded
    on demand, which cuts native threads and memory use for the default mode.
    """

    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.face_mesh = None
        self.pose = None
        self._closed = False

    def _ensure_face(self):
        if self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
        return self.face_mesh

    def _ensure_pose(self):
        if self.pose is None:
            self.pose = self.mp_pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
        return self.pose

    def process_frame(
        self,
        frame: np.ndarray,
        show_face: bool = False,
        draw_hands: bool = True,
        show_pose: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        if self._closed:
            raise RuntimeError("MediaPipe resources are already closed")
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an RGB frame with three channels")

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = frame.shape[:2]
        landmarks: Dict[str, Any] = {
            key: list(value) for key, value in EMPTY_LANDMARKS.items()
        }

        # MediaPipe recommends a read-only buffer during native inference.
        frame.flags.writeable = False
        try:
            hand_results = self.hands.process(frame)
            face_results = self._ensure_face().process(frame) if show_face else None
            pose_results = self._ensure_pose().process(frame) if show_pose else None
        finally:
            frame.flags.writeable = True

        if hand_results and hand_results.multi_hand_landmarks:
            for index, hand_landmark in enumerate(hand_results.multi_hand_landmarks[:2]):
                points = [
                    (
                        min(width - 1, max(0, int(lm.x * width))),
                        min(height - 1, max(0, int(lm.y * height))),
                    )
                    for lm in hand_landmark.landmark
                ]
                landmarks["hands"].append(points)
                landmarks["raw_hands"].append(hand_landmark)
                if hand_results.multi_handedness and index < len(hand_results.multi_handedness):
                    side = hand_results.multi_handedness[index].classification[0].label
                else:
                    side = "Unknown"
                landmarks["handedness"].append(side)
                if draw_hands:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmark,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

        if face_results and face_results.multi_face_landmarks:
            for face_landmark in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmark,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    None,
                    self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                landmarks["face"] = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in face_landmark.landmark
                ]

        if pose_results and pose_results.pose_landmarks:
            pose_landmark = pose_results.pose_landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmark,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            landmarks["pose"] = [
                (int(lm.x * width), int(lm.y * height))
                for lm in pose_landmark.landmark
            ]

        return frame, landmarks, (face_results, hand_results, pose_results)

    def extract_hand_crop(
        self, frame: np.ndarray, landmarks: Dict[str, Any], padding: int = 30
    ) -> Optional[np.ndarray]:
        hands = landmarks.get("hands") or []
        if not hands:
            return None
        points = hands[0]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        height, width = frame.shape[:2]
        x1, x2 = max(0, min(xs) - padding), min(width, max(xs) + padding)
        y1, y2 = max(0, min(ys) - padding), min(height, max(ys) + padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (150, 150), interpolation=cv2.INTER_AREA)

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in (self.hands, self.face_mesh, self.pose):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        self.hands = None
        self.face_mesh = None
        self.pose = None
