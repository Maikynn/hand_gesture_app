"""User-facing calibration, custom gestures, sequences and recognition history."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_DATA_DIR = PROJECT_ROOT / "user_data"
CUSTOM_GESTURES_PATH = USER_DATA_DIR / "custom_gestures.json"
GESTURE_HISTORY_PATH = USER_DATA_DIR / "gesture_history.json"
ERROR_CLIPS_DIR = USER_DATA_DIR / "error_clips"


def normalized_landmarks(raw_hand: Any) -> Optional[np.ndarray]:
    """Return translation/scale invariant xyz hand features."""
    points = getattr(raw_hand, "landmark", None)
    if not points or len(points) < 21:
        return None
    array = np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in points[:21]],
        dtype=np.float32,
    )
    array -= array[0]
    scale = float(np.max(np.linalg.norm(array[:, :2], axis=1)))
    if scale < 1e-5:
        return None
    return (array / scale).reshape(-1)


class CustomGestureLibrary:
    """Centroid classifier trained from samples recorded in the camera UI."""

    def __init__(self, path: Path = CUSTOM_GESTURES_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._gestures: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        with self._lock:
            self._gestures = data if isinstance(data, dict) else {}

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._gestures)

    def train(self, name: str, samples: Iterable[Sequence[float]]) -> int:
        clean_name = " ".join(str(name).strip().lower().split())
        vectors = np.asarray(list(samples), dtype=np.float32)
        if not clean_name:
            raise ValueError("Введите название жеста")
        if vectors.ndim != 2 or vectors.shape[0] < 8 or vectors.shape[1] != 63:
            raise ValueError("Для обучения нужно не меньше 8 корректных кадров руки")
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        threshold = max(0.18, min(0.75, float(np.percentile(distances, 95) * 1.8)))
        with self._lock:
            self._gestures[clean_name] = {
                "centroid": centroid.tolist(),
                "threshold": threshold,
                "samples": int(vectors.shape[0]),
                "updated_at": time.time(),
            }
            self._save()
        return int(vectors.shape[0])

    def predict(self, raw_hand: Any) -> Optional[Tuple[str, float]]:
        vector = normalized_landmarks(raw_hand)
        if vector is None:
            return None
        best: Optional[Tuple[str, float]] = None
        with self._lock:
            gestures = dict(self._gestures)
        for name, item in gestures.items():
            centroid = np.asarray(item.get("centroid", []), dtype=np.float32)
            if centroid.size != vector.size:
                continue
            distance = float(np.linalg.norm(vector - centroid))
            threshold = max(0.05, float(item.get("threshold", 0.35)))
            confidence = max(0.0, min(1.0, 1.0 - distance / threshold))
            if distance <= threshold and (best is None or confidence > best[1]):
                best = (name, confidence)
        return best

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(self._gestures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)


class GestureSequenceMatcher:
    """Matches gesture transitions inside a configurable time window."""

    def __init__(self, sequences: Iterable[Dict[str, Any]] = ()):
        self.configure(sequences)
        self.events: deque[Tuple[float, str]] = deque(maxlen=24)
        self._last_gesture = ""

    def configure(self, sequences: Iterable[Dict[str, Any]]) -> None:
        self.sequences = [
            dict(item)
            for item in sequences
            if isinstance(item, dict) and item.get("enabled", True)
        ]

    def feed(self, gesture: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not gesture or gesture in {"unknown", "no_gesture"}:
            return None
        if gesture == self._last_gesture:
            return None
        self._last_gesture = gesture
        stamp = time.monotonic() if now is None else float(now)
        self.events.append((stamp, gesture))
        for item in self.sequences:
            steps = [
                str(step).strip().lower()
                for step in item.get("steps", [])
                if str(step).strip()
            ]
            if not steps or len(self.events) < len(steps):
                continue
            tail = list(self.events)[-len(steps) :]
            timeout = max(0.5, float(item.get("timeout_sec", 3.0)))
            if [event[1] for event in tail] == steps and tail[-1][0] - tail[0][0] <= timeout:
                self.events.clear()
                return item
        return None


@dataclass
class GestureHistoryItem:
    timestamp: float
    side: str
    gesture: str
    confidence: float
    model: str
    action: str = ""
    incorrect: bool = False


class GestureHistory:
    """Bounded persistent recognition log with correction labels."""

    def __init__(self, path: Path = GESTURE_HISTORY_PATH, limit: int = 300):
        self.path = path
        self.limit = max(20, int(limit))
        self.items: List[GestureHistoryItem] = []
        self._load()

    def _load(self) -> None:
        try:
            values = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = [
                GestureHistoryItem(**item)
                for item in values[-self.limit :]
                if isinstance(item, dict)
            ]
        except (OSError, json.JSONDecodeError, TypeError):
            self.items = []

    def add(
        self,
        side: str,
        gesture: str,
        confidence: float,
        model: str,
        action: str = "",
    ) -> int:
        self.items.append(
            GestureHistoryItem(
                timestamp=time.time(),
                side=side,
                gesture=gesture,
                confidence=float(confidence),
                model=model,
                action=action,
            )
        )
        self.items = self.items[-self.limit :]
        self._save()
        return len(self.items) - 1

    def mark_incorrect(self, index: int, incorrect: bool = True) -> None:
        if 0 <= index < len(self.items):
            self.items[index].incorrect = bool(incorrect)
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps([asdict(item) for item in self.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)


def auto_enhance(frame: np.ndarray) -> np.ndarray:
    """Apply grey-world white balance and CLAHE contrast normalization."""
    image = np.ascontiguousarray(frame, dtype=np.uint8)
    means = image.reshape(-1, 3).mean(axis=0)
    target = float(np.mean(means))
    scale = np.clip(target / np.maximum(means, 1.0), 0.72, 1.38)
    balanced = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2RGB)


def apply_roi_mask(
    frame: np.ndarray, roi: Sequence[float], *, dim: float = 0.22
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Dim pixels outside a normalized ROI while preserving full-frame coordinates."""
    height, width = frame.shape[:2]
    x, y, w, h = [max(0.0, min(1.0, float(value))) for value in roi[:4]]
    x1, y1 = int(x * width), int(y * height)
    x2, y2 = int(min(1.0, x + max(0.05, w)) * width), int(
        min(1.0, y + max(0.05, h)) * height
    )
    masked = np.clip(frame.astype(np.float32) * dim, 0, 255).astype(np.uint8)
    masked[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    cv2.rectangle(masked, (x1, y1), (x2, y2), (73, 156, 255), 2)
    return masked, (x1, y1, x2, y2)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def privacy_silhouette(
    shape: Sequence[int], hands: Iterable[Sequence[Tuple[int, int]]]
) -> np.ndarray:
    """Render only anonymous hand landmarks on a black HUD surface."""

    height, width = int(shape[0]), int(shape[1])
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for points_value in hands:
        points = [(int(x), int(y)) for x, y in points_value]
        for start, end in HAND_EDGES:
            if start < len(points) and end < len(points):
                cv2.line(canvas, points[start], points[end], (65, 226, 245), 2)
        for point in points[:21]:
            cv2.circle(canvas, point, 4, (190, 250, 255), -1)
    cv2.putText(
        canvas,
        "PRIVACY MODE // LANDMARKS ONLY",
        (18, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (75, 210, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


class ErrorClipRecorder:
    """Keep a compressed rolling buffer and save it after user feedback."""

    def __init__(self, directory: Path = ERROR_CLIPS_DIR, seconds: float = 5.0):
        self.directory = directory
        self.seconds = max(2.0, float(seconds))
        self.frames: deque[Tuple[float, bytes]] = deque(maxlen=75)
        self._last_capture = 0.0
        self._lock = threading.RLock()

    def push(self, rgb: np.ndarray, now: Optional[float] = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        if timestamp - self._last_capture < 0.10:
            return
        self._last_capture = timestamp
        small = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 68],
        )
        if not ok:
            return
        with self._lock:
            self.frames.append((timestamp, encoded.tobytes()))
            cutoff = timestamp - self.seconds
            while self.frames and self.frames[0][0] < cutoff:
                self.frames.popleft()

    def save_recent(self, label: str = "gesture_error") -> Optional[Path]:
        with self._lock:
            items = list(self.frames)
        if not items:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(
            character
            for character in label
            if character.isalnum() or character in "_-"
        )
        path = self.directory / f"{safe or 'gesture_error'}_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360)
        )
        try:
            if not writer.isOpened():
                return None
            for _timestamp, payload in items:
                frame = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    writer.write(frame)
        finally:
            writer.release()
        return path if path.is_file() and path.stat().st_size > 0 else None
