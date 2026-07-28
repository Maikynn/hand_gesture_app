#!/usr/bin/env python3
"""
Gesture Fusion module.

Combines three information sources into a single robust gesture prediction:

  * Static branch (weight 0.8): a static gesture model run on the hand crop
    obtained from MediaPipe hand landmarks. The backend is switchable:
      - HaGRID MobileNetV3-Small (.pth, PyTorch)  [default]
      - YOLO (ultralytics) trained on HaGRID       [use_yolo_static()]
      - Combined (ensemble of both)                 [use_combined_static()]
  * MediaPipe branch (weight 0.2): a lightweight geometric classifier that maps the
    same MediaPipe landmarks directly to the HaGRID vocabulary.
  * Dynamic branch (optional, pluggable): an LSTM trained on Jester 21-landmark
    sequences. When attached and confident, dynamic predictions take priority over
    the static branch (dynamics > static rule).

Final static prediction = 0.8 * static_model_probs + 0.2 * mediapipe_probs
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# HaGRID class vocabulary (STRICTLY alphabetical, as seen by ImageFolder at train time)
# ---------------------------------------------------------------------------
HAGRID_CLASSES: List[str] = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'no_gesture', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]

# Default model / asset paths
DEFAULT_HAGRID_MODEL = r"F:\MODELS\Model_lerning\hagrid_mobilenetv3.pth"
DEFAULT_HAND_LANDMARKER = r"F:\MODELS\hand_landmarker.task"
# Trained YOLO static model (file appears here once training finishes)
DEFAULT_YOLO_STATIC_MODEL = r"F:\MODELS\hagrid_static_yolo-2"

# MediaPipe hand landmark indices (Tasks API returns 21 landmarks)
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 6, 10, 14, 18
THUMB_IP, THUMB_MCP = 3, 2


class MediaPipeStaticClassifier:
    """Geometric classifier that maps MediaPipe hand landmarks to HaGRID classes."""

    @staticmethod
    def _finger_extended(lm, tip_idx, pip_idx) -> bool:
        return lm[tip_idx].y < lm[pip_idx].y

    @staticmethod
    def _dist(a, b) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def predict(self, landmarks) -> str:
        lm = landmarks  # list of landmark objects with .x, .y, .z in [0, 1]
        fingers = [
            self._finger_extended(lm, INDEX_TIP, INDEX_PIP),
            self._finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP),
            self._finger_extended(lm, RING_TIP, RING_PIP),
            self._finger_extended(lm, PINKY_TIP, PINKY_PIP),
        ]
        thumb_ext = self._dist(lm[THUMB_TIP], lm[THUMB_MCP]) > 0.1
        num = sum(fingers)

        # OK: thumb-index circle, other fingers extended
        if self._dist(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.05 and fingers[1] and fingers[2] and fingers[3]:
            return 'ok'
        # Rock: index + pinky
        if fingers[0] and fingers[3] and not fingers[1] and not fingers[2]:
            return 'rock'
        # Call: thumb + pinky
        if thumb_ext and fingers[3] and not fingers[0] and not fingers[1] and not fingers[2]:
            return 'call'
        # Like: thumb up only
        if thumb_ext and num == 0 and lm[THUMB_TIP].y < lm[WRIST].y:
            return 'like'
        # Dislike: thumb down only
        if thumb_ext and num == 0 and lm[THUMB_TIP].y > lm[WRIST].y:
            return 'dislike'
        # One: index only
        if fingers[0] and num == 1:
            return 'one'
        # Two up: index + middle
        if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
            return 'two_up'
        # Three: index + middle + ring
        if fingers[0] and fingers[1] and fingers[2] and not fingers[3]:
            return 'three'
        # Four: all four fingers extended
        if all(fingers):
            return 'four'
        # Fist: nothing extended
        if num == 0 and not thumb_ext:
            return 'fist'
        # Palm / stop: all extended including thumb
        if all(fingers) and thumb_ext:
            return 'palm'
        return 'no_gesture'


class GestureFusion:
    """
    Fuses a static gesture model (MobileNetV3, YOLO, or Combined) with a MediaPipe
    geometric classifier and an optional dynamic LSTM model. Designed to be dropped
    into the hand_gesture_app recognition pipeline.
    """

    def __init__(self,
                 ha_grid_model_path: str = None,
                 hand_landmarker_path: str = DEFAULT_HAND_LANDMARKER,
                 static_weight: float = 0.8,
                 mediapipe_weight: float = 0.2,
                 device: Optional[str] = None,
                 dynamic_threshold: float = 0.6):
        # If ha_grid_model_path is None, fall back to DEFAULT_HAGRID_MODEL
        self.ha_grid_model_path = ha_grid_model_path or DEFAULT_HAGRID_MODEL
        self.hand_landmarker_path = hand_landmarker_path
        self.static_weight = static_weight
        self.mediapipe_weight = mediapipe_weight
        self.dynamic_threshold = dynamic_threshold
        self.device = device or self._default_device()

        self.classes = HAGRID_CLASSES
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self._model = None           # HaGRID PyTorch model
        self._transform = None
        self._mp_classifier = MediaPipeStaticClassifier()

        # Static backend state (switchable between MobileNetV3, YOLO, Combined)
        self._static_backend = 'mobilenet'   # 'mobilenet' | 'yolo' | 'combined'
        self._yolo = None                    # YOLOStaticModel instance
        self._yolo_path = None
        self._dynamic_model = None
        self._dynamic_classes: List[str] = []

        self._load_static_model()

    # --------------------------- static model ---------------------------
    @staticmethod
    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_static_model(self):
        if not os.path.isfile(self.ha_grid_model_path):
            print(
                "[GestureFusion] static weights are not configured; "
                "using the lightweight MediaPipe classifier"
            )
            self._model = None
            self._transform = None
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except Exception as e:
            print(f"[GestureFusion] torch/torchvision unavailable, static model disabled: {e}")
            return

        num_classes = len(self.classes)
        model = models.mobilenet_v3_small()
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )
        try:
            state = torch.load(self.ha_grid_model_path, map_location=self.device)
            model.load_state_dict(state)
            print(f"[GestureFusion] HaGRID weights loaded: {self.ha_grid_model_path}")
        except Exception as e:
            print(f"[GestureFusion] failed to load weights: {e}")
            self._model = None
            self._transform = None
            return
        model = model.to(self.device)
        model.eval()
        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # --------------------------- static backend switch ---------------------------
    def use_yolo_static(self, model_path: str = DEFAULT_YOLO_STATIC_MODEL):
        """
        Switch the static backend to a trained YOLO model (e.g. HaGRID YOLO).

        The model file is expected at ``model_path`` (default F:\\MODELS\\hagrid_static_yolo-2).
        If the file is not trained yet, it is loaded lazily on the first ``predict`` call,
        so you can switch now and the weights will be picked up automatically later.
        """
        from .static_yolo import YOLOStaticModel
        self._yolo = YOLOStaticModel(model_path, self.classes)
        self._yolo_path = model_path
        self._static_backend = 'yolo'
        self._yolo.ensure_loaded()  # attempts load now; warns if not trained yet
        print(f"[GestureFusion] static backend -> YOLO ({model_path})")

    def use_mobilenet_static(self):
        """Switch the static backend back to the HaGRID MobileNetV3 model."""
        self._static_backend = 'mobilenet'
        print("[GestureFusion] static backend -> MobileNetV3 (HaGRID)")

    def use_combined_static(self):
        """Switch the static backend to a combined MobileNetV3 + YOLO ensemble."""
        self._static_backend = 'combined'
        # Ensure YOLO is loaded (lazy) so the ensemble has both branches.
        if self._yolo is None:
            try:
                self.use_yolo_static()
            except Exception as e:
                print(f"[GestureFusion] combined mode: YOLO not available: {e}")
        print("[GestureFusion] static backend -> Combined (MobileNetV3 + YOLO)")


    def load_static_model_by_id(self, model_id: str, model_config: dict):
        """
        Load a static model by its ID from the model configuration.

        Args:
            model_id: The ID of the model to load (e.g., 'mobilenet_v3', 'yolo_v8', 'combined')
            model_config: Dictionary containing model configurations from model_config.json
        """
        static_models = model_config.get("static_models", {})
        if model_id in static_models:
            info = static_models[model_id]
            model_type = info.get("type", "mobilenet")
            path = info.get("path")
            if model_type == "yolo":
                if path and os.path.exists(path):
                    self.use_yolo_static(path)
                    print(f"[GestureFusion] Loaded YOLO model '{model_id}' from {path}")
                    return True
                else:
                    print(f"[GestureFusion] WARNING: YOLO path not found for '{model_id}': {path}")
                    return False
            elif model_type == "combined":
                self.use_combined_static()
                print(f"[GestureFusion] Loaded combined model '{model_id}'")
                return True
            else:  # mobilenet
                if path and os.path.exists(path):
                    self.ha_grid_model_path = path
                    self._load_static_model()
                    self._static_backend = 'mobilenet'
                    print(f"[GestureFusion] Loaded MobileNetV3 model '{model_id}' from {path}")
                    return True
                else:
                    print(f"[GestureFusion] WARNING: MobileNetV3 path not found for '{model_id}': {path}")
                    return False
        print(f"[GestureFusion] WARNING: Model ID '{model_id}' not found in config")
        return False

    # --------------------------- dynamic model ---------------------------
    def set_dynamic_model(self, model, dynamic_classes: Optional[List[str]] = None):
        """
        Attach a trained dynamic (LSTM) predictor.

        `model` must expose ``predict(sequence) -> (name, confidence)`` where
        ``sequence`` is a (SEQ_LEN, FEATURES) numpy array of buffered landmarks.
        """
        self._dynamic_model = model
        if dynamic_classes is not None:
            self._dynamic_classes = list(dynamic_classes)
        elif hasattr(model, "classes"):
            self._dynamic_classes = list(model.classes)

    @property
    def has_dynamic(self) -> bool:
        return self._dynamic_model is not None

    # --------------------------- inference ---------------------------
    def _crop_hand(self, frame: np.ndarray, landmarks, w: int, h: int) -> Optional[np.ndarray]:
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
        x2, y2 = min(w, x_max + pad_x), min(h, y_max + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _mobilenet_probs(self, frame, landmarks, w, h) -> np.ndarray:
        """MobileNetV3 (HaGRID) backend probabilities."""
        probs = np.zeros(len(self.classes), dtype=np.float32)
        if self._model is None or self._transform is None:
            return probs
        crop = self._crop_hand(frame, landmarks, w, h)
        if crop is None:
            return probs
        import torch
        from PIL import Image
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._model(tensor)
            p = torch.softmax(out[0], dim=0).cpu().numpy()
        return p.astype(np.float32)

    def _yolo_probs(self, frame, landmarks, w, h):
        """YOLO backend probabilities (None if unavailable)."""
        if self._yolo is None:
            return None
        crop = self._crop_hand(frame, landmarks, w, h)
        if crop is None:
            return None
        return self._yolo.predict_probs(crop)

    def _static_probs(self, frame, landmarks, w, h) -> np.ndarray:
        # Combined backend: average MobileNetV3 + YOLO probabilities
        if self._static_backend == 'combined':
            mn = self._mobilenet_probs(frame, landmarks, w, h)
            yl = self._yolo_probs(frame, landmarks, w, h)
            if yl is not None:
                return 0.5 * mn + 0.5 * yl
            return mn

        # YOLO backend (switched at runtime)
        if self._static_backend == 'yolo' and self._yolo is not None:
            crop = self._crop_hand(frame, landmarks, w, h)
            if crop is not None:
                probs = self._yolo.predict_probs(crop)
                if probs is not None:
                    return probs
            # fall through to MobileNetV3 if YOLO unavailable / not trained yet

        # MobileNetV3 (HaGRID) backend
        return self._mobilenet_probs(frame, landmarks, w, h)

    def _mediapipe_probs(self, landmarks) -> np.ndarray:
        probs = np.zeros(len(self.classes), dtype=np.float32)
        name = self._mp_classifier.predict(landmarks)
        idx = self._class_to_idx.get(name, self._class_to_idx['no_gesture'])
        probs[idx] = 1.0
        return probs

    def predict(self, frame: np.ndarray,
                landmarks: Any,
                w: int = None, h: int = None,
                dynamic_sequence: Any = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Full fusion prediction.

        Args:
            frame: BGR image (np.ndarray).
            landmarks: MediaPipe hand landmark list (normalized x,y,z in [0,1]).
            w, h: frame dimensions (inferred from frame if omitted).
            dynamic_sequence: optional buffered (SEQ_LEN, FEATURES) array for the
                              dynamic LSTM branch.

        Returns:
            (gesture_name, confidence, probability_dict)
        """
        if w is None or h is None:
            h, w = frame.shape[:2]

        # Dynamic branch takes priority when confident (dynamics > static)
        if self.has_dynamic and dynamic_sequence is not None:
            dyn_name, dyn_conf = self._dynamic_model.predict(dynamic_sequence)
            if dyn_conf >= self.dynamic_threshold and dyn_name != 'no_gesture':
                return dyn_name, float(dyn_conf), {dyn_name: float(dyn_conf)}

        # Static fusion: 0.8 * static_model + 0.2 * MediaPipe
        static_p = self._static_probs(frame, landmarks, w, h)
        mp_p = self._mediapipe_probs(landmarks)
        fused = self.static_weight * static_p + self.mediapipe_weight * mp_p
        fused = fused / (fused.sum() + 1e-8)

        idx = int(np.argmax(fused))
        name = self.classes[idx]
        conf = float(fused[idx])
        prob_dict = {c: float(fused[i]) for i, c in enumerate(self.classes)}
        return name, conf, prob_dict
