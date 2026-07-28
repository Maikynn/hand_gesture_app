#!/usr/bin/env python3
"""
YOLO static gesture backend (ultralytics).

Wraps a trained YOLO model (classification or detection) so it can replace the
HaGRID MobileNetV3 static backend inside GestureFusion.

The model is expected under the training directory
``F:\\MODELS\\hagrid_static_yolo-2`` (or ``F:\\MODELS\\Model_lerning\\
hagrid_static_yolo-2``) - typically at ``weights/best.pt`` once training
finishes. :meth:`ensure_loaded` resolves the actual ``.pt`` file robustly
(directory -> weights/best.pt, weights/last.pt, *.pt) and also checks a few
fallback candidate locations, so the exact save path does not matter.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import cv2


# Fallback locations where the trained YOLO weights may appear.
_FALLBACK_DIRS = [
    str(Path(__file__).resolve().parents[2] / "models" / "yolo"),
    r"F:\MODELS\Model_lerning\hagrid_static_yolo-2",
    r"F:\MODELS\hagrid_static_yolo-2",
]
DEFAULT_YOLO_MODEL = str(
    Path(__file__).resolve().parents[2] / "models" / "yolo" / "hagrid_best.pt"
)


class YOLOStaticModel:
    def __init__(self, model_path: str, classes: list, device: str = "auto"):
        self.model_path = model_path
        self.classes = list(classes)
        self.requested_device = device if device in {"auto", "cpu", "cuda"} else "auto"
        self.device = "cpu"
        self.fallback_reason = ""
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self._model = None
        self._loaded = False  # True once ensure_loaded has run at least once
        self._resolved_path = None

    # ----------------------------- path resolution -----------------------------
    def _resolve_model_path(self) -> Optional[str]:
        """Find the actual ``.pt`` weights from ``model_path`` or fallbacks."""
        candidates: list = []

        # 1) The configured path itself.
        configured = Path(self.model_path).expanduser()
        if not configured.is_absolute():
            configured = Path(__file__).resolve().parents[2] / configured
        configured_path = str(configured)
        if os.path.isfile(configured_path) and configured_path.endswith(".pt"):
            return configured_path
        if os.path.isdir(configured_path):
            candidates += [
                os.path.join(configured_path, "weights", "best.pt"),
                os.path.join(configured_path, "weights", "last.pt"),
                os.path.join(configured_path, "best.pt"),
                os.path.join(configured_path, "last.pt"),
            ]
            for f in sorted(os.listdir(configured_path)):
                if f.endswith(".pt"):
                    candidates.append(os.path.join(configured_path, f))

        # 2) Fallback directories (same search strategy).
        for d in _FALLBACK_DIRS:
            if os.path.isdir(d):
                candidates += [
                    os.path.join(d, "weights", "best.pt"),
                    os.path.join(d, "weights", "last.pt"),
                    os.path.join(d, "best.pt"),
                    os.path.join(d, "last.pt"),
                ]
                for f in sorted(os.listdir(d)):
                    if f.endswith(".pt"):
                        candidates.append(os.path.join(d, f))

        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    # ----------------------------- loading -----------------------------
    def ensure_loaded(self) -> bool:
        """Lazily load the YOLO model. Returns True if ready for inference."""
        if self._loaded:
            return self._model is not None
        self._loaded = True

        resolved = self._resolve_model_path()
        if not resolved:
            print(f"[YOLOStaticModel] weights not found yet for "
                  f"'{self.model_path}' (train the model first; it will be "
                  f"loaded automatically once weights/best.pt appears)")
            return False
        self._resolved_path = resolved

        try:
            from ultralytics import YOLO
        except Exception as e:
            print(f"[YOLOStaticModel] ultralytics not installed: {e}")
            return False
        try:
            self._model = YOLO(resolved)
            try:
                import torch

                if self.requested_device == "cuda" and not torch.cuda.is_available():
                    self.fallback_reason = "CUDA недоступна — используется CPU"
                self.device = (
                    "cuda"
                    if self.requested_device in {"auto", "cuda"}
                    and torch.cuda.is_available()
                    else "cpu"
                )
            except Exception:
                self.device = "cpu"
            print(f"[YOLOStaticModel] loaded: {resolved}")
            return True
        except Exception as e:
            print(f"[YOLOStaticModel] failed to load: {e}")
            return False

    @property
    def loaded(self) -> bool:
        """True when the YOLO model is ready for inference."""
        return self._model is not None

    def status(self) -> str:
        if self._model is not None:
            return f"loaded:{self._resolved_path}:{self.device}"
        if self._loaded:
            return "not_trained"
        return "pending"

    @property
    def resolved_path(self) -> Optional[str]:
        return self._resolved_path

    # ----------------------------- inference -----------------------------
    def predict_probs(self, crop: np.ndarray, *, input_is_rgb: bool = False):
        """
        Run inference on a BGR hand crop.

        Returns:
            np.ndarray of probabilities over self.classes, or None on failure.
        """
        if not self.ensure_loaded() or self._model is None:
            return None
        crop_rgb = (
            np.ascontiguousarray(crop)
            if input_is_rgb
            else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        )
        try:
            results = self._model(crop_rgb, verbose=False, device=self.device)
        except Exception as e:
            if self.device == "cuda":
                self.fallback_reason = f"CUDA inference failed: {e}"
                self.device = "cpu"
                try:
                    results = self._model(crop_rgb, verbose=False, device="cpu")
                except Exception as cpu_error:
                    print(f"[YOLOStaticModel] inference error: {cpu_error}")
                    return None
            else:
                print(f"[YOLOStaticModel] inference error: {e}")
                return None
        res = results[0]
        probs = np.zeros(len(self.classes), dtype=np.float32)
        names = self._model.names  # dict idx -> class name

        # Classification model: results[0].probs
        if getattr(res, "probs", None) is not None:
            data = res.probs.data.cpu().numpy().astype(np.float32)
            for idx, name in names.items():
                j = self._class_to_idx.get(str(name).lower(), None)
                if j is not None and idx < len(data):
                    probs[j] = data[idx]
            s = probs.sum()
            if s > 0:
                probs = probs / s
            return probs

        # Detection model: aggregate box confidences per class
        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32)
            for c, cf in zip(cls, conf):
                name = names[int(c)]
                j = self._class_to_idx.get(str(name).lower(), None)
                if j is not None:
                    probs[j] += cf
            s = probs.sum()
            if s > 0:
                probs = probs / s
            return probs

        return probs
