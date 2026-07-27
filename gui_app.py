import sys, os
# Auto-bootstrap: re-exec inside the Python 3.12 venv (mediapipe 0.10.14 with mp.solutions).
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_PY = next((c for c in [
    os.path.join(_HERE, "venv", "Scripts", "python.exe"),
    os.path.join(_HERE, "hand_gesture_app", "venv", "Scripts", "python.exe"),
] if os.path.exists(c)), None)
if _VENV_PY and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PY):
    os.execv(_VENV_PY, [_VENV_PY, __file__] + sys.argv[1:])

#!/usr/bin/env python3
"""
Axi Gesture Assistant — main GUI application.
Terminal-style interface with three tabs: Camera, Assistant, Settings.
"""
import sys
import os
import json
import threading
import queue
import time
import re
import webbrowser
import subprocess
import datetime
from urllib.parse import quote

import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QLineEdit, QTextBrowser,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QPlainTextEdit,
    QProgressBar, QDialog, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

# Suppress MediaPipe/TFLite INFO/WARNING noise (e.g. "inference_feedback_manager ... single signature inference")
# while keeping real ERRORs visible. Set before importing mediapipe.
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make the local `src` package importable when run directly.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from src.assistant.tts_engine import TTSEngine, list_voices
    _TTS_OK = True
except Exception as _tts_err:
    TTSEngine = None
    list_voices = lambda: []
    _TTS_OK = False
    print("TTS unavailable:", _tts_err)

try:
    from src.assistant.vosk_stt import (
        vosk_available, list_microphones, VoskStream, get_default_model_path,
        list_vosk_models)
    _VOSK_OK = True
except Exception as _vosk_err:
    vosk_available = lambda: False
    list_microphones = lambda: []
    VoskStream = None
    get_default_model_path = lambda: ""
    list_vosk_models = lambda: []
    _VOSK_OK = False
    print("Vosk STT unavailable:", _vosk_err)

try:
    from src.assistant.llm_client import LLMClient
    _LLM_OK = True
except Exception as _llm_err:
    LLMClient = None
    _LLM_OK = False
    print("LLM client unavailable:", _llm_err)

_ASSISTANT_OK = _TTS_OK and _VOSK_OK and _LLM_OK

# Optional neural engines (Silero TTS v5, Whisper STT, GestureFusion backend).
try:
    from src.assistant.silero_tts import (
        SileroTTSEngine, available as silero_available,
        list_voices as silero_list_voices)
    _SILERO_OK = True
except Exception as _silero_err:
    SileroTTSEngine = None
    silero_available = lambda: False
    silero_list_voices = lambda: []
    _SILERO_OK = False
    print("Silero TTS unavailable:", _silero_err)

try:
    from src.assistant.whisper_stt import WhisperStream, whisper_available
    _WHISPER_OK = True
except Exception as _whisper_err:
    WhisperStream = None
    whisper_available = lambda: False
    _WHISPER_OK = False
    print("Whisper STT unavailable:", _whisper_err)

try:
    from src.hand_processing.gesture_fusion import GestureFusion
    _FUSION_OK = True
except Exception as _fusion_err:
    GestureFusion = None
    _FUSION_OK = False
    print("GestureFusion unavailable:", _fusion_err)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG = {
    "camera": {
        "device_index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "show_face": True,
        "show_hands": True,
        "show_eyes": True,
        "show_brows": True,
        "show_lips": True,
        "show_fingers": True,
        "show_palm": True,
        "brightness": 0,
        "contrast": 1.0,
        "flip": True
    },
    "gesture": {
        "model_path": "",
        "confidence_threshold": 0.85,
        "hand_crop_size": 150,
        "hand_crop_padding": 30,
        "palm_side_detection": True
    },
    "voice": {
        "enabled": True,
        "wake_word": "аксиос",
        "mic_index": -1,                 # -1 = system default
        "mic_gain": 2.0,                 # digital input boost (>=1.0)
        "stt_engine": "vosk",            # offline, streaming (reliable)
        "vosk_model_path": "",           # empty -> bundled models/vosk-ru
        "tts_engine": "pyttsx3",         # offline SAPI5
        "tts_voice": "",                 # voice id; empty -> default Russian
        "tts_rate": 160,
        "llm_engine": "openrouter",      # openrouter | ollama
        "openrouter_key": "",
        "openrouter_model": "meta-llama/llama-3.1-8b-instruct:free",
        # Alternative OpenRouter models tried in order when the primary
        # is rate-limited (429) or unavailable. Free-tier models share
        # one key, so rolling over mitigates the 429 limit errors.
        "openrouter_fallback_models": [
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free"
        ],
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3",
        "system_prompt": "Ты — голосовой помощник Акси. Отвечай кратко и по делу на русском.",
        "max_tokens": 200
    }
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk not in cfg[k]:
                            cfg[k][kk] = vv
            return cfg
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Local command execution + offline responses (voice assistant)
# ---------------------------------------------------------------------------
APP_MAP = {
    "блокнот": "notepad", "notepad": "notepad",
    "калькулятор": "calc", "calc": "calc", "calculator": "calc",
    "проводник": "explorer", "explorer": "explorer",
    "диспетчер задач": "taskmgr", "task manager": "taskmgr",
    "параметры": "ms-settings:", "settings": "ms-settings:",
    "командная строка": "cmd", "терминал": "cmd", "cmd": "cmd", "terminal": "cmd",
    "паинт": "mspaint", "paint": "mspaint",
    "powershell": "powershell",
    "word": "winword", "excel": "excel",
    "браузер": None,
}
SITE_MAP = {
    "ютуб": "https://youtube.com", "youtube": "https://youtube.com",
    "гугл": "https://google.com", "google": "https://google.com",
    "вк": "https://vk.com", "vk": "https://vk.com",
}


def _looks_like_url(s):
    s = s.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return True
    return ("." in s) and (" " not in s)


def open_target(target):
    """Open a URL or perform a web search. Returns a spoken response."""
    target = target.strip()
    if not target:
        return "Не указан адрес."
    if target in SITE_MAP:
        webbrowser.open(SITE_MAP[target])
        return f"Открываю {target}."
    if _looks_like_url(target):
        url = target if target.startswith("http") else "https://" + target
        webbrowser.open(url)
        return f"Открываю {url}."
    webbrowser.open("https://www.google.com/search?q=" + quote(target))
    return f"Ищу в интернете: {target}."


def open_app(name):
    """Open a known application. Returns a response string, or None if unhandled."""
    key = name.strip().lower()
    if key in APP_MAP:
        cmd = APP_MAP[key]
        if cmd is None:
            webbrowser.open("https://www.google.com")
            return "Открываю браузер."
        try:
            if cmd.endswith(":"):          # e.g. ms-settings:
                os.startfile(cmd)
            else:
                subprocess.Popen(cmd, shell=False)
            return f"Запускаю {name}."
        except Exception as e:
            return f"Не удалось запустить {name}: {e}"
    try:
        os.startfile(name.strip())
        return f"Открываю {name}."
    except Exception:
        return None


def execute_local_command(text):
    """Parse and execute a local command. Returns a response string or None."""
    t = text.lower().strip()
    for kw in ("открой ссылку", "открыть ссылку", "open link", "ссылку"):
        if kw in t:
            return open_target(t.split(kw, 1)[1])
    for kw in ("запусти", "запустить", "открой", "открыть", "open", "run"):
        if t.startswith(kw + " ") or (" " + kw + " ") in t:
            idx = t.find(kw)
            target = t[idx + len(kw):].strip()
            if target:
                res = open_app(target)
                if res:
                    return res
                return open_target(target)
    return None


def local_respond(text):
    """Tiny offline responder for common questions (no API key needed)."""
    t = text.lower()
    if any(w in t for w in ["который час", "время", "сколько времени"]):
        return "Сейчас " + datetime.datetime.now().strftime("%H:%M") + "."
    if any(w in t for w in ["какое число", "какая дата", "дата", "число"]):
        return "Сегодня " + datetime.datetime.now().strftime("%d.%m.%Y") + "."
    if "как тебя зовут" in t or "твоё имя" in t or "твое имя" in t:
        return "Меня зовут Акси."
    if t.strip() in ("привет", "здравствуй", "здравствуйте", "хай", "hello"):
        return "Привет! Чем могу помочь?"
    if "спасибо" in t:
        return "Пожалуйста!"
    return None


# ---------------------------------------------------------------------------
# Camera + Skeleton Worker
# ---------------------------------------------------------------------------
class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    hand_crop_ready = pyqtSignal(np.ndarray, str)
    gesture_detected = pyqtSignal(str, float)
    dynamic_gesture_detected = pyqtSignal(str, float)
    error = pyqtSignal(str)
    hands_state = pyqtSignal(bool, bool)
    face_state = pyqtSignal(bool, bool, bool, bool)
    landmarks_ready = pyqtSignal(object)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.running = False
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.hands = None
        self.face = None
        # Gesture fusion (switchable MobileNetV3 / YOLO static backend).
        self.fusion = None
        self.cap = None
        self._static_backend = "mobilenet"
        self._gesture_throttle = 0
        # Dynamic (LSTM) gesture buffer: rolling window of 126-dim frames.
        self._dyn_buffer = []
        self._dyn_seq_len = 24
        # Dynamic (Jester LSTM) model runs ALWAYS, in parallel with the
        # static backend. Set True once the model is successfully attached.
        self._dynamic_active = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.cfg["camera"]["device_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["camera"]["height"])
        cap.set(cv2.CAP_PROP_FPS, self.cfg["camera"]["fps"])
        self.cap = cap

        # Fail fast with a clear message if the camera cannot be opened at all
        # (wrong device index, device in use, no permission, etc.). This avoids
        # entering the read loop and keeps the thread lifecycle clean.
        if not cap.isOpened():
            dev = self.cfg["camera"]["device_index"]
            self.error.emit(
                f"Камера недоступна (устройство {dev}). Проверьте подключение "
                "или выберите другую камеру в настройках.")
            cap.release()
            return

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Gesture fusion: loads HaGRID MobileNetV3 by default; the backend can
        # be switched live to YOLO via set_static_backend() (no restart).
        if _FUSION_OK and GestureFusion is not None:
            try:
                self.fusion = GestureFusion()
                self._static_backend = self.fusion._static_backend
            except Exception as e:
                print("GestureFusion init failed:", e)
                self.fusion = None
        # Attach the dynamic (Jester LSTM) model as a parallel, always-on
        # branch so dynamic gestures are recognised continuously.
        if self.fusion is not None:
            try:
                self._dynamic_active = self.fusion.ensure_dynamic_loaded()
            except Exception as e:
                print("Dynamic LSTM attach failed:", e)
                self._dynamic_active = False

        read_failures = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Tolerate a few transient frame drops, but give up (and end the
                # thread cleanly) if the camera keeps failing to deliver frames.
                read_failures += 1
                if read_failures >= 10:
                    self.error.emit("Camera read failed")
                    break
                time.sleep(0.1)
                continue
            read_failures = 0

            brightness = self.cfg["camera"]["brightness"]
            contrast = self.cfg["camera"]["contrast"]
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            if self.cfg["camera"]["flip"]:
                frame = cv2.flip(frame, 1)

            frame = self._draw_skeleton(frame)
            self.frame_ready.emit(frame)
            time.sleep(0.01)

        cap.release()
        if self.hands:
            self.hands.close()
        if self.face:
            self.face.close()

    def _draw_skeleton(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        left_detected = False
        right_detected = False
        first_hand = None
        left_lm = None
        right_lm = None
        if self.cfg["camera"]["show_hands"]:
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                        res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness.classification[0].label
                    if not self.cfg["camera"]["flip"]:
                        label = "Right" if label == "Left" else "Left"
                    if label == "Left":
                        left_detected = True
                        left_lm = hand_landmarks
                    else:
                        right_detected = True
                        right_lm = hand_landmarks
                    self._draw_hand(frame, hand_landmarks, w, h)
                    self._emit_hand_crop(frame, hand_landmarks, w, h, label)
                    if first_hand is None:
                        first_hand = hand_landmarks
        self.hands_state.emit(left_detected, right_detected)

        # Live gesture recognition via GestureFusion.
        # 1) STATIC branch (selected backend: mobilenet / yolo / combined).
        if self.fusion is not None and first_hand is not None:
            backend = self._static_backend
            self._gesture_throttle += 1
            if self._gesture_throttle % 2 == 0:  # throttle to ~every 2nd frame
                try:
                    name, conf, _ = self.fusion.predict(
                        frame, first_hand.landmark, w, h, None)
                    self.gesture_detected.emit(f"{name} [{backend}]", float(conf))
                except Exception as e:
                    print("gesture predict error:", e)

        # 2) DYNAMIC branch (Jester LSTM) -- ALWAYS active in parallel with
        #    the static model, so dynamic gestures are recognised continuously.
        if self.fusion is not None and self._dynamic_active and first_hand is not None:
            feat = self._build_dynamic_feature(left_lm, right_lm)
            if feat is not None:
                self._dyn_buffer.append(feat)
                if len(self._dyn_buffer) > self._dyn_seq_len:
                    self._dyn_buffer.pop(0)
                if len(self._dyn_buffer) >= self._dyn_seq_len:
                    try:
                        seq = np.array(self._dyn_buffer, dtype=np.float32)
                        name, conf, _ = self.fusion.predict_dynamic(seq)
                        self.dynamic_gesture_detected.emit(f"{name}", float(conf))
                    except Exception as e:
                        print("dynamic predict error:", e)

        face_detected = False
        left_eye_open = right_eye_open = mouth_open = False
        if self.cfg["camera"]["show_face"]:
            res = self.face.process(rgb)
            if res.multi_face_landmarks:
                face_detected = True
                for lm in res.multi_face_landmarks:
                    self._draw_face(frame, lm, w, h)
                    left_eye_open, right_eye_open, mouth_open = self._compute_face_state(lm, w, h)
        self.face_state.emit(left_eye_open, right_eye_open, mouth_open, face_detected)

        return frame

    def _compute_face_state(self, landmarks, w, h):
        pts = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]

        def dist(a, b):
            return ((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2) ** 0.5

        left_ear = (dist(160, 144) + dist(158, 153)) / (2.0 * dist(33, 133) + 1e-6)
        right_ear = (dist(385, 380) + dist(387, 373)) / (2.0 * dist(362, 263) + 1e-6)
        mar = dist(13, 14) / (dist(61, 291) + 1e-6)

        return left_ear > 0.18, right_ear > 0.18, mar > 0.40

    def _draw_hand(self, frame, landmarks, w, h):
        connections = self.mp_hands.HAND_CONNECTIONS
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
        if self.cfg["camera"]["show_palm"] or self.cfg["camera"]["show_fingers"]:
            for conn in connections:
                if (conn[0] < 21 and conn[1] < 21):
                    cv2.line(frame, pts[conn[0]], pts[conn[1]], (0, 255, 0), 2)
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i in [4, 8, 12, 16, 20] else (255, 0, 0)
            cv2.circle(frame, p, 3, color, -1)

    def _draw_face(self, frame, landmarks, w, h):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
        if self.cfg["camera"]["show_eyes"]:
            for idx in [33, 133, 159, 145, 362, 263, 386, 374]:
                if idx < len(pts):
                    cv2.circle(frame, pts[idx], 2, (255, 255, 0), -1)
        if self.cfg["camera"]["show_brows"]:
            for idx in [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]:
                if idx < len(pts):
                    cv2.circle(frame, pts[idx], 2, (0, 255, 255), -1)
        if self.cfg["camera"]["show_lips"]:
            for idx in [13, 14, 78, 308, 61, 291, 0, 17]:
                if idx < len(pts):
                    cv2.circle(frame, pts[idx], 2, (255, 0, 255), -1)

    def _emit_hand_crop(self, frame, landmarks, w, h, side):
        xs = [int(lm.x * w) for lm in landmarks.landmark]
        ys = [int(lm.y * h) for lm in landmarks.landmark]
        x1, x2 = max(0, min(xs) - 30), min(w, max(xs) + 30)
        y1, y2 = max(0, min(ys) - 30), min(h, max(ys) + 30)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crop = cv2.resize(crop, (150, 150))
            self.hand_crop_ready.emit(crop, side)

    def stop(self):
        self.running = False
        self.quit()
        # Release the capture from the calling thread so a blocking cap.read()
        # in the worker unblocks and the loop can exit promptly.
        cap = getattr(self, "cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.wait(3000)

    def set_static_backend(self, name):
        """Live-switch the gesture model backend (no app restart).

        name: 'mobilenet' (HaGRID MobileNetV3), 'yolo' (HaGRID YOLO),
              'combined' (ensemble) or 'lstm' (dynamic Jester model).
        """
        if self.fusion is None:
            return
        # Reset the dynamic buffer on any switch to avoid mixing sequences.
        self._dyn_buffer = []
        if name == "yolo":
            self.fusion.use_yolo_static()
        elif name == "combined":
            self.fusion.use_combined_static()
        elif name == "lstm":
            self.fusion.use_dynamic_lstm()
        else:
            self.fusion.use_mobilenet_static()
        self._static_backend = self.fusion._static_backend

    def _build_dynamic_feature(self, left_lm, right_lm):
        """Build the 126-dim wrist-relative feature for the dynamic LSTM.

        Layout: [left_hand(21*3), right_hand(21*3)], landmark-major, each
        landmark stored relative to its wrist (landmark 0). This mirrors the
        training data produced by train_jester_lstm.py so live predictions
        align with the offline-trained model.
        """
        def hand_block(lm):
            if lm is None:
                return np.zeros(21 * 3, dtype=np.float32)
            arr = np.array([[p.x, p.y, p.z] for p in lm.landmark],
                           dtype=np.float32)
            rel = arr - arr[0]  # wrist-relative
            return rel.reshape(-1)

        left = hand_block(left_lm)
        right = hand_block(right_lm)
        return np.concatenate([left, right]).astype(np.float32)

    def load_static_model_by_id(self, model_id):
        """Load a static model by ID from model_config.json."""
        if self.fusion is None:
            return
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    model_cfg = json.load(f)
                self.fusion.load_static_model_by_id(model_id, model_cfg)
            except Exception as e:
                print("Failed to load model config:", e)


# ---------------------------------------------------------------------------
# Voice Assistant (Vosk streaming: wake-word -> STT -> command/LLM -> TTS)
# ---------------------------------------------------------------------------
class VoiceWorker(QThread):
    log = pyqtSignal(str, str)
    heard = pyqtSignal(str)
    partial = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.running = False
        self.tts = None
        self.llm = None
        self._armed = False
        self._cmd = ""
        self._last_final = 0.0
        self._deadline = 0.0
        # Deduplication / debounce state for spoken commands.
        self._last_cmd_text = ""
        self._last_cmd_time = 0.0
        self._busy = False
        self._tts_voice_id = None
        self._tts_engine = None
        self._tts_rate = None
        self._stt_mode = "vosk"

    def run(self):
        self.running = True
        v = self.cfg.get("voice", {})
        raw_wake = v.get("wake_word", "аксиос") or "аксиос"
        self._wake_list = [w.strip().lower() for w in
                           raw_wake.replace(",", " ").split() if w.strip()]
        self._wake = self._wake_list[0] if self._wake_list else "аксиос"
        mic_index = v.get("mic_index", -1)
        if mic_index is not None and mic_index < 0:
            mic_index = None

        if not _ASSISTANT_OK:
            self.log.emit("error", "Модули assistant недоступны (src/assistant).")
            return
        if not vosk_available():
            self.log.emit("error", "Vosk не установлен — распознавание недоступно.")
            return

        # TTS (engine chosen from config: silero v5 neural or pyttsx3 offline)
        self._rebuild_tts(v)

        # LLM client is built LIVE inside _llm_answer() from the current
        # config, so provider / key / model switches in Settings take
        # effect immediately (no restart required).

        if v.get("llm_engine") == "ollama":
            self.log.emit("info",
                f"Локальная модель Ollama: {v.get('ollama_model')} @ {v.get('ollama_url')}.")
        elif v.get("openrouter_key"):
            self.log.emit("info", f"OpenRouter подключён (модель: {v.get('openrouter_model')}).")
        else:
            self.log.emit("warn",
                "Ключ OpenRouter не задан — доступны только локальные команды и офлайн-ответы.")

        # STT engine chosen from config: Whisper (bond005/whisper-podlodka-turbo)
        # or Vosk (offline). Whisper runs its own capture thread and drives the
        # same on_partial / on_final callbacks, so no read_chunk polling is used.
        stt_engine = (v.get("stt_engine") or "vosk").lower()
        self._stt_mode = stt_engine
        try:
            if stt_engine == "whisper" and _WHISPER_OK and WhisperStream is not None:
                try:
                    stream = WhisperStream(
                        model_name=v.get("whisper_model") or "bond005/whisper-podlodka-turbo",
                        mic_index=mic_index,
                        gain=v.get("mic_gain", 2.0),
                        on_partial=self._on_partial,
                        on_final=self._on_final)
                    stream.start()
                    self.log.emit("info", "STT: Whisper (bond005/whisper-podlodka-turbo).")
                except Exception as we:
                    # Whisper model failed to load (e.g. corrupt cache) ->
                    # transparently fall back to the offline Vosk engine so
                    # the assistant always works.
                    self.log.emit("warn",
                        f"Whisper недоступен ({we}); переключаюсь на Vosk.")
                    stt_engine = "vosk"
                    self._stt_mode = "vosk"
            if stt_engine != "whisper":
                # Prefer the large bundled model if present; fall back to small.
                model_path = v.get("vosk_model_path") or get_default_model_path()
                stream = VoskStream(model_path=model_path, mic_index=mic_index,
                                   gain=v.get("mic_gain", 2.0))
                stream.on_partial = self._on_partial
                stream.on_final = self._on_final
                stream.start()
                if self._stt_mode == "vosk":
                    self.log.emit("info", "STT: Vosk (офлайн).")
        except Exception as e:
            self.log.emit("error", f"Микрофон / STT недоступен: {e}")
            return

        self.status.emit(f'Слушаю. Скажите «{self._wake}»...')
        self.speak("Голосовой помощник Акси готов. Скажите «аксиос».")

        try:
            while self.running:
                # Live TTS engine/voice switching: rebuild without a restart
                # when the user changes the engine, voice or rate in Settings.
                self._rebuild_tts(v)

                speaking = bool(self.tts and self.tts.is_speaking)
                if speaking:
                    # Drain the mic buffer (Vosk) while we talk so we don't
                    # echo ourselves. Whisper runs its own capture thread.
                    if self._stt_mode != "whisper" and hasattr(stream, "read_chunk"):
                        stream.read_chunk(feed=False)
                    self._armed = False
                    self._cmd = ""
                    time.sleep(0.05)
                    continue
                if self._stt_mode == "whisper":
                    # Whisper drives recognition via callbacks; just idle here.
                    time.sleep(0.1)
                else:
                    try:
                        stream.read_chunk(feed=True)
                    except Exception as e:
                        self.log.emit("error", f"Ошибка аудио: {e}")
                        break
                # Finish a captured command on silence or hard timeout.
                if self._armed and self._cmd:
                    now = time.time()
                    if (now - self._last_final > 1.2) or (now > self._deadline):
                        cmd = self._cmd
                        self._armed = False
                        self._cmd = ""
                        self._process_command(cmd)
                elif self._armed and time.time() > self._deadline:
                    self._armed = False
                    self.speak("Я вас не расслышал, повторите команду.")
                    self.status.emit(f'Слушаю. Скажите «{self._wake}»...')
        finally:
            try:
                stream.stop()
            except Exception:
                pass
            if self.tts:
                try:
                    self.tts.shutdown()
                except Exception:
                    pass
            self.log.emit("info", "Голосовой помощник остановлен.")

    # -- recognition callbacks (run in the reader thread) ------------------
    def _on_partial(self, text):
        if text:
            self.partial.emit(text)
        if not self._armed and self._wake_span(text):
            self._arm(text)

    def _on_final(self, text):
        if not text:
            return
        self.heard.emit(text)
        if not self._armed:
            if self._wake_span(text):
                self._arm(text)
            return
        self._accumulate(text)

    def _accumulate(self, text):
        """Append recognized text to the pending command, but keep only the
        part AFTER the wake word so 'аксиос' never leaks into the command."""
        span = self._wake_span(text)
        if span:
            self._cmd = text[span[1]:].strip()
        else:
            self._cmd = (self._cmd + " " + text).strip() if self._cmd else text
        self._last_final = time.time()

    def _arm(self, text):
        self._armed = True
        span = self._wake_span(text)
        after = text[span[1]:].strip() if span else ""
        self.status.emit("Слушаю команду...")
        self._cmd = after
        self._last_final = time.time()
        self._deadline = time.time() + 6.0

    # -- wake-word matching (exact + fuzzy) --------------------------------
    def _wake_span(self, text):
        """Return (start, end) char span of a wake word inside `text`, or
        None. Exact substring first, then fuzzy token matching so small
        STT errors (e.g. 'аксиюс' -> 'аксиос') still trigger, while
        unrelated words like 'такой' do not false-trigger."""
        if not text:
            return None
        low = text.lower()
        for w in self._wake_list:
            idx = low.find(w)
            if idx >= 0:
                return (idx, idx + len(w))
        for m in re.finditer(r"\S+", low):
            tok = m.group(0)
            for w in self._wake_list:
                if self._wake_close(tok, w):
                    return (m.start(), m.end())
        return None

    @staticmethod
    def _lev(a, b):
        """Levenshtein edit distance between two strings."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for ca in a:
            cur = [prev[0] + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + cost))
            prev = cur
        return prev[lb]

    def _wake_close(self, tok, w, max_dist=2):
        """True if `tok` is close enough to wake word `w` to count as a
        (mis)recognition. Length guard avoids matching short/unrelated
        words; edit-distance guard tolerates 1-2 character errors."""
        if abs(len(tok) - len(w)) > max_dist:
            return False
        return self._lev(tok, w) <= max_dist

    # -- command handling --------------------------------------------------
    def _process_command(self, cmd):
        # Debounce / deduplicate: a single spoken command can be emitted by
        # Vosk as several final results, and the same phrase must not fan
        # out into 2-10 duplicate LLM requests or actions. Drop an exact
        # repeat that arrives within a short window, and never run two
        # commands concurrently.
        norm = (cmd or "").strip().lower()
        now = time.time()
        if norm and norm == self._last_cmd_text and (now - self._last_cmd_time) < 3.0:
            self.log.emit("info", f"Пропуск дубликата команды: {cmd}")
            return
        if self._busy:
            self.log.emit("info", "Команда уже обрабатывается — пропуск.")
            return
        self._last_cmd_text = norm
        self._last_cmd_time = now
        self._busy = True
        try:
            self.heard.emit(cmd)
            self.status.emit("Обрабатываю...")
            self.log.emit("info", f"Команда: {cmd}")

            resp = execute_local_command(cmd)
            if resp is None:
                resp = local_respond(cmd)
            if resp is None:
                resp = self._llm_answer(cmd)
            # Guarantee: the assistant speaks EVERY response, in full.
            self._speak_all(resp)
            self.log.emit("info", resp)
            self.status.emit(f'Слушаю. Скажите «{self._wake}»...')
        finally:
            self._busy = False

    def _llm_answer(self, cmd):
        v = self.cfg.get("voice", {})
        # Build a FRESH client from the live config on every call so that
        # switching provider / editing the key / model in Settings takes
        # effect immediately. The client created at startup was frozen and
        # ignored later UI changes (requests kept going to the old provider).
        client = LLMClient(
            engine=v.get("llm_engine", "openrouter"),
            api_key=v.get("openrouter_key", ""),
            model=v.get("openrouter_model", "meta-llama/llama-3.1-8b-instruct:free"),
            ollama_url=v.get("ollama_url", "http://localhost:11434"),
            ollama_model=v.get("ollama_model", "llama3"),
            system_prompt=v.get("system_prompt", ""),
            max_tokens=v.get("max_tokens", 200),
            fallback_models=v.get("openrouter_fallback_models", []))
        if client.engine == "ollama":
            ans = client.send_message(cmd)
            if ans:
                return ans
            return "Не удалось получить ответ от локальной модели Ollama."
        if client.api_key:
            ans = client.send_message(cmd)
            if ans:
                return ans
            return "Не удалось получить ответ от модели."
        return ("Для ответа нужен ключ OpenRouter или локальная модель Ollama. "
                "Я выполняю команды, например: «открой блокнот» или "
                "«открой ссылку youtube.com».")

    def _rebuild_tts(self, v):
        """(Re)build the TTS engine from the live config without a restart.

        Supports Silero TTS v5 (neural, Russian) and the offline pyttsx3 engine.
        Only rebuilds when the engine / voice / rate actually changed.
        """
        cur_engine = (v.get("tts_engine") or "pyttsx3").lower()
        cur_voice = v.get("tts_voice") or None
        cur_rate = v.get("tts_rate", 160)
        if (cur_engine == self._tts_engine and cur_voice == self._tts_voice_id
                and cur_rate == self._tts_rate):
            return
        try:
            if self.tts is not None:
                self.tts.shutdown()
        except Exception:
            pass
        self._tts_engine = cur_engine
        self._tts_voice_id = cur_voice
        self._tts_rate = cur_rate
        try:
            if cur_engine == "silero" and _SILERO_OK and SileroTTSEngine is not None:
                speaker = cur_voice if cur_voice in (
                    "aidar", "baya", "kseniya", "xenia", "eugene") else "xenia"
                self.tts = SileroTTSEngine(speaker=speaker, rate=cur_rate, volume=0.9)
                self.log.emit("info", f"TTS: Silero v5 (speaker={speaker}).")
            else:
                self.tts = TTSEngine(use_offline=True, voice_id=cur_voice or None,
                                     rate=cur_rate)
        except Exception as e:
            self.log.emit("error", f"TTS недоступен: {e}")
            self.tts = None

    def speak(self, text):
        if self.tts and text and text.strip():
            self.tts.speak(text)

    def _speak_all(self, text):
        """Speak the entire response, splitting long LLM answers into
        sentences so that EVERY phrase is pronounced (not just part)."""
        if not text or not text.strip():
            return
        parts = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
        for p in parts:
            p = p.strip()
            if p:
                self.speak(p)
        if self.tts:
            try:
                self.tts.speech_queue.join()
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.wait()


# ---------------------------------------------------------------------------
# Microphone test (live transcription + level meter + gain)
# ---------------------------------------------------------------------------
class MicTestWorker(QThread):
    level = pyqtSignal(float)
    text = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, mic_index=None, model_path=None, gain=2.0):
        super().__init__()
        self.mic_index = mic_index
        self.model_path = model_path
        self.gain = gain
        self.running = False
        self.stream = None
        self._display = ""

    def run(self):
        try:
            self.stream = VoskStream(model_path=self.model_path, mic_index=self.mic_index,
                                     gain=self.gain)
            self.stream.on_partial = self._on_partial
            self.stream.on_final = self._on_final
            self.stream.start()
        except Exception as e:
            self.error.emit(str(e))
            return
        self.running = True
        while self.running:
            try:
                lvl = self.stream.read_chunk()
                self.level.emit(lvl)
            except Exception:
                break
        if self.stream:
            self.stream.stop()

    def _on_final(self, t):
        if not t:
            return
        self._display = (self._display + "\n" + t).strip()
        self.text.emit(self._display)

    def _on_partial(self, t):
        if not t:
            return
        live = (self._display + "\n> " + t) if self._display else "> " + t
        self.text.emit(live)


class MicTestDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Тест микрофона")
        self.setMinimumSize(540, 460)
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        mic_row = QHBoxLayout()
        mic_row.addWidget(QLabel("Микрофон:"))
        self.mic_combo = QComboBox()
        self._populate_mic()
        mic_row.addWidget(self.mic_combo, 1)
        layout.addLayout(mic_row)

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setStyleSheet(
            "QProgressBar { background: #222; border: 1px solid #444; text-align: center; }"
            "QProgressBar::chunk { background: #00cc66; }")
        layout.addWidget(QLabel("Уровень сигнала:"))
        layout.addWidget(self.level_bar)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Усиление микрофона:"))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(10, 100)  # *0.1 -> 1.0x .. 10.0x
        self.gain_slider.setValue(int(self.cfg["voice"].get("mic_gain", 2.0) * 10))
        self.gain_slider.valueChanged.connect(self._on_gain)
        gain_row.addWidget(self.gain_slider, 1)
        self.gain_label = QLabel(f"{self.cfg['voice'].get('mic_gain', 2.0):.1f}x")
        gain_row.addWidget(self.gain_label)
        layout.addLayout(gain_row)

        layout.addWidget(QLabel("Что слышит микрофон (распознавание в реальном времени):"))
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setFont(QFont("Consolas", 11))
        layout.addWidget(self.transcript, 1)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ Начать")
        self.stop_btn = QPushButton("■ Стоп")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.start_btn.clicked.connect(self.start_test)
        self.stop_btn.clicked.connect(self.stop_test)
        self.setLayout(layout)

    def _populate_mic(self):
        self.mic_combo.clear()
        self.mic_combo.addItem("Системный микрофон (по умолчанию)", -1)
        for idx, name in list_microphones():
            self.mic_combo.addItem(name, idx)
        cur = self.cfg["voice"].get("mic_index", -1)
        for i in range(self.mic_combo.count()):
            if self.mic_combo.itemData(i) == cur:
                self.mic_combo.setCurrentIndex(i)
                break

    def _on_gain(self, val):
        g = val / 10.0
        self.gain_label.setText(f"{g:.1f}x")
        self.cfg["voice"]["mic_gain"] = g
        if self.worker and getattr(self.worker, "stream", None):
            self.worker.stream.gain = max(1.0, g)

    def start_test(self):
        mic_index = self.mic_combo.currentData()
        if mic_index is not None and mic_index < 0:
            mic_index = None
        self.cfg["voice"]["mic_index"] = mic_index if mic_index is not None else -1
        gain = self.cfg["voice"].get("mic_gain", 2.0)
        model_path = self.cfg["voice"].get("vosk_model_path") or None
        self.transcript.setPlainText("")
        self.worker = MicTestWorker(mic_index=mic_index, model_path=model_path, gain=gain)
        self.worker.level.connect(lambda l: self.level_bar.setValue(int(l * 100)))
        self.worker.text.connect(self.transcript.setPlainText)
        self.worker.error.connect(lambda e: self.transcript.setPlainText("Ошибка: " + e))
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_test(self):
        if self.worker:
            self.worker.running = False
            self.worker.wait(2000)
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        self.stop_test()
        event.accept()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class CameraTab(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setStyleSheet("background-color: #0a0a0a; color: #dddddd;")

        self.video_label = QLabel("Camera feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #050505; color: #777777; border: 1px solid #222222;")
        layout.addWidget(self.video_label)

        crops_layout = QHBoxLayout()
        left_box = QVBoxLayout()
        self.left_crop_label = QLabel("нет")
        self.left_crop_label.setFixedSize(150, 150)
        self.left_crop_label.setStyleSheet(
            "border: 1px solid #444444; background-color: #050505; color: #777777;")
        self.left_crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_cap = QLabel("Левая рука")
        left_cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_cap.setStyleSheet("color: #cccccc;")
        left_box.addWidget(self.left_crop_label)
        left_box.addWidget(left_cap)

        right_box = QVBoxLayout()
        self.right_crop_label = QLabel("нет")
        self.right_crop_label.setFixedSize(150, 150)
        self.right_crop_label.setStyleSheet(
            "border: 1px solid #444444; background-color: #050505; color: #777777;")
        self.right_crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_cap = QLabel("Правая рука")
        right_cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_cap.setStyleSheet("color: #cccccc;")
        right_box.addWidget(self.right_crop_label)
        right_box.addWidget(right_cap)

        crops_layout.addLayout(left_box)
        crops_layout.addSpacing(20)
        crops_layout.addLayout(right_box)
        crops_layout.addStretch()
        layout.addLayout(crops_layout)

        self.gesture_label = QLabel("Жест: —")
        self.gesture_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #00ff88; "
            "background-color: #0a1a0a; border: 1px solid #00ff88; "
            "border-radius: 4px; padding: 6px;")
        self.gesture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gesture_label)

        # Timer resets the gesture text when no detection arrives for a while
        self._gesture_timer = QTimer()
        self._gesture_timer.setSingleShot(True)
        self._gesture_timer.timeout.connect(
            lambda: self.gesture_label.setText("Жест: не распознан"))

        # Model selection combo (moved from Settings to main camera screen)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Модель жестов:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("MobileNetV3 (HaGRID)", "mobilenet")
        self.model_combo.addItem("YOLO (HaGRID)", "yolo")
        self.model_combo.addItem("Объединённая (MobileNetV3 + YOLO)", "combined")
        self.model_combo.setCurrentIndex(0)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo, 1)
        layout.addLayout(model_row)

        # Status labels
        self.backend_status_label = QLabel("Активна: MobileNetV3 (HaGRID)")
        self.backend_status_label.setStyleSheet("font-size: 13px; color: #cccccc; padding: 2px;")
        self.backend_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.backend_status_label)

        # Dynamic model status: shows the live Jester LSTM prediction. The
        # model is always active in parallel with the static backend.
        self.dynamic_status_label = QLabel("Динамика: модель не загружена")
        self.dynamic_status_label.setStyleSheet("font-size: 13px; color: #ffaa00; padding: 2px;")
        self.dynamic_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.dynamic_status_label)
        self._dynamic_active = False
        self._dynamic_timer = QTimer()
        self._dynamic_timer.setSingleShot(True)
        self._dynamic_timer.timeout.connect(self._reset_dynamic_label)

        self._pending_backend = "mobilenet"

        layout.addStretch(1)

        self.face_status_label = QLabel("Лицо: —   |   Левый глаз: —   |   Правый глаз: —   |   Рот: —")
        self.face_status_label.setStyleSheet("font-size: 15px; color: #cccccc; padding: 4px;")
        self.face_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.face_status_label)

        indicators_layout = QHBoxLayout()
        indicators_layout.addStretch()

        left_box2 = QVBoxLayout()
        self.left_indicator = QLabel()
        self.left_indicator.setFixedSize(90, 90)
        self.left_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_indicator.setStyleSheet(
            "background-color: #330000; border: 2px solid #555555; border-radius: 6px;")
        left_label = QLabel("Левая")
        left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_label.setStyleSheet("color: #cccccc;")
        left_box2.addWidget(self.left_indicator)
        left_box2.addWidget(left_label)

        right_box2 = QVBoxLayout()
        self.right_indicator = QLabel()
        self.right_indicator.setFixedSize(90, 90)
        self.right_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_indicator.setStyleSheet(
            "background-color: #330000; border: 2px solid #555555; border-radius: 6px;")
        right_label = QLabel("Правая")
        right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_label.setStyleSheet("color: #cccccc;")
        right_box2.addWidget(self.right_indicator)
        right_box2.addWidget(right_label)

        indicators_layout.addLayout(left_box2)
        indicators_layout.addSpacing(40)
        indicators_layout.addLayout(right_box2)
        indicators_layout.addStretch()
        layout.addLayout(indicators_layout)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Старт")
        self.stop_btn = QPushButton("■ Стоп")
        self.cam_combo = QComboBox()
        self.cam_combo.addItems([f"Камера {i}" for i in range(3)])
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.cam_combo)
        layout.addLayout(btn_layout)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.setLayout(layout)

    def start_camera(self):
        # Never orphan a running worker: stop any previous instance first.
        # Otherwise a second Start click would spawn a second thread that
        # reloads the (heavy) GestureFusion model and, on cleanup, trigger
        # "QThread: Destroyed while thread is still running".
        if self.worker is not None:
            self.stop_camera()
        self.cfg["camera"]["device_index"] = self.cam_combo.currentIndex()
        self.worker = CameraWorker(self.cfg)
        # Apply the backend chosen in the UI before the stream starts.
        pending = getattr(self, "_pending_backend", "mobilenet")
        if pending in ("yolo", "combined", "lstm"):
            self.worker.set_static_backend(pending)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.hand_crop_ready.connect(self.update_hand_crop)
        self.worker.gesture_detected.connect(self.update_gesture)
        self.worker.dynamic_gesture_detected.connect(self.update_dynamic_gesture)
        self.worker.hands_state.connect(self.update_hands_state)
        self.worker.face_state.connect(self.update_face_state)
        self.worker.error.connect(lambda e: print(e))
        # Reflect the dynamic-model load state (set by the worker thread).
        QTimer.singleShot(1500, self._refresh_dynamic_status)
        self.worker.start()

    def _on_model_changed(self, index):
        model_id = self.model_combo.currentData()
        self._set_static_backend(model_id)

    def _set_static_backend(self, name, force=False):
        """Switch the static gesture backend live (no camera restart).
        The dynamic (Jester LSTM) model stays active in parallel regardless
        of the selected static backend."""
        if self.worker is not None and hasattr(self.worker, "set_static_backend"):
            self.worker.set_static_backend(name)
        self._pending_backend = name
        labels = {
            "mobilenet": "MobileNetV3 (HaGRID)",
            "yolo": "YOLO (HaGRID)",
            "combined": "Объединённая (MobileNetV3 + YOLO)",
        }
        self.backend_status_label.setText(f"Активна: {labels.get(name, name)}")
        # The dynamic (Jester LSTM) model is always active in parallel; the
        # real load state is set by the worker when the camera starts.
        self._reset_dynamic_label()

    def update_dynamic_gesture(self, name, conf):
        """Show the live dynamic (Jester LSTM) prediction."""
        self.dynamic_status_label.setText(f"Динамика: {name} ({conf:.2f})")
        self.dynamic_status_label.setStyleSheet(
            "font-size: 13px; color: #00ff88; padding: 2px;")
        self._dynamic_timer.start(1500)

    def _reset_dynamic_label(self):
        if getattr(self, "_dynamic_active", False):
            self.dynamic_status_label.setText("Динамика: Jester LSTM (активна)")
            self.dynamic_status_label.setStyleSheet(
                "font-size: 13px; color: #00ff88; padding: 2px;")
        else:
            self.dynamic_status_label.setText("Динамика: модель не загружена")
            self.dynamic_status_label.setStyleSheet(
                "font-size: 13px; color: #ffaa00; padding: 2px;")

    def _refresh_dynamic_status(self):
        self._dynamic_active = bool(
            getattr(self.worker, "_dynamic_active", False)) if self.worker else False
        self._reset_dynamic_label()

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_hand_crop(self, crop, side):
        h, w, ch = crop.shape
        qimg = QImage(crop.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        if side == "Left":
            self.left_crop_label.setPixmap(pix)
        else:
            self.right_crop_label.setPixmap(pix)

    def update_gesture(self, name, conf):
        self.gesture_label.setText(f"Жест: {name} ({conf:.2f})")
        # Restart the idle timer so "не распознан" only shows after 1.5s
        self._gesture_timer.start(1500)

    def update_hands_state(self, left, right):
        on = "background-color: #00cc44; border: 2px solid #00ff66; border-radius: 6px;"
        off = "background-color: #330000; border: 2px solid #555555; border-radius: 6px;"
        self.left_indicator.setStyleSheet(on if left else off)
        self.right_indicator.setStyleSheet(on if right else off)
        if not left:
            self.left_crop_label.clear()
            self.left_crop_label.setText("нет")
        if not right:
            self.right_crop_label.clear()
            self.right_crop_label.setText("нет")

    def update_face_state(self, left_eye, right_eye, mouth, face_detected):
        if not face_detected:
            self.face_status_label.setText(
                "Лицо: нет   |   Левый глаз: —   |   Правый глаз: —   |   Рот: —")
            return
        self.face_status_label.setText(
            f"Лицо: найдено   |   Левый глаз: {'открыт' if left_eye else 'закрыт'}   |   "
            f"Правый глаз: {'открыт' if right_eye else 'закрыт'}   |   "
            f"Рот: {'открыт' if mouth else 'закрыт'}")

    def _trigger_dynamics_processing(self, landmarks):
        """Trigger dynamics processing script with extracted landmarks."""
        try:
            # Import here to avoid circular imports
            from process_dynamics import extract_dynamics_from_landmarks, save_dynamics_data

            # Extract dynamics features
            features = extract_dynamics_from_landmarks(landmarks)

            # Save the data
            filepath = save_dynamics_data(features)

            print(f"[GUI] Triggered dynamics processing with landmarks from {filepath}")

        except Exception as e:
            print(f"[GUI] Failed to trigger dynamics processing: {e}")


class AssistantTab(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.worker = None
        self.mic_test_dialog = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        mic_row = QHBoxLayout()
        mic_row.addWidget(QLabel("Микрофон:"))
        self.mic_combo = QComboBox()
        self._populate_mic()
        self.mic_combo.currentIndexChanged.connect(
            lambda i: self.cfg["voice"].__setitem__(
                "mic_index",
                self.mic_combo.currentData() if self.mic_combo.currentData() is not None else -1))
        mic_row.addWidget(self.mic_combo, 1)
        self.test_mic_btn = QPushButton("🎤 Тест микрофона")
        self.test_mic_btn.clicked.connect(self.open_mic_test)
        mic_row.addWidget(self.test_mic_btn)
        layout.addLayout(mic_row)

        self.log_browser = QTextBrowser()
        self.log_browser.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log_browser, 1)

        self.status_label = QLabel("Статус: остановлен")
        self.status_label.setStyleSheet("font-size: 14px; color: #88ff88; padding: 2px;")
        layout.addWidget(self.status_label)

        self.partial_label = QLabel("Слышу (в реальном времени): —")
        self.partial_label.setStyleSheet("font-size: 12px; color: #66ccff;")
        layout.addWidget(self.partial_label)

        self.heard_label = QLabel("Последняя команда: —")
        layout.addWidget(self.heard_label)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Запустить помощника")
        self.stop_btn = QPushButton("■ Стоп")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.start_btn.clicked.connect(self.start_assistant)
        self.stop_btn.clicked.connect(self.stop_assistant)
        self.setLayout(layout)

    def _populate_mic(self):
        self.mic_combo.clear()
        self.mic_combo.addItem("Системный микрофон (по умолчанию)", -1)
        for idx, name in list_microphones():
            self.mic_combo.addItem(name, idx)
        cur = self.cfg["voice"].get("mic_index", -1)
        for i in range(self.mic_combo.count()):
            if self.mic_combo.itemData(i) == cur:
                self.mic_combo.setCurrentIndex(i)
                break

    def open_mic_test(self):
        if self.mic_test_dialog is None or not self.mic_test_dialog.isVisible():
            self.mic_test_dialog = MicTestDialog(self.cfg, self)
        self.mic_test_dialog.show()
        self.mic_test_dialog.raise_()

    def start_assistant(self):
        # Avoid stacking multiple workers (which would double-process every
        # command and multiply LLM requests). Stop any running instance.
        if self.worker is not None:
            try:
                self.worker.stop()
            except Exception:
                pass
            self.worker = None
        self.worker = VoiceWorker(self.cfg)
        self.worker.log.connect(self.append_log)
        self.worker.heard.connect(self.update_heard)
        self.worker.partial.connect(self.update_partial)
        self.worker.status.connect(self.update_status)
        self.worker.start()

    def stop_assistant(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.update_status("остановлен")

    def append_log(self, level, msg):
        color = {"info": "#00ff00", "warn": "#ffff00", "error": "#ff0000"}.get(level, "#ffffff")
        self.log_browser.append(f'<span style="color:{color}">[{level}] {msg}</span>')

    def update_heard(self, text):
        self.heard_label.setText(f"Последняя команда: {text}")

    def update_partial(self, text):
        self.partial_label.setText(f"Слышу (в реальном времени): {text}")

    def update_status(self, text):
        self.status_label.setText(f"Статус: {text}")


class SettingsTab(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Camera settings
        cam_group = QGroupBox("Камера")
        cam_form = QFormLayout()
        self.show_face = QCheckBox("Показывать лицо")
        self.show_hands = QCheckBox("Показывать руки")
        self.show_eyes = QCheckBox("Показывать глаза")
        self.show_brows = QCheckBox("Показывать брови")
        self.show_lips = QCheckBox("Показывать губы")
        self.show_fingers = QCheckBox("Показывать пальцы")
        self.show_palm = QCheckBox("Показывать ладонь")
        self.brightness = QSlider(Qt.Orientation.Horizontal)
        self.brightness.setRange(-100, 100)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.1, 3.0)
        self.flip = QCheckBox("Отзеркаливание")

        for cb, key in [(self.show_face, "show_face"), (self.show_hands, "show_hands"),
                        (self.show_eyes, "show_eyes"), (self.show_brows, "show_brows"),
                        (self.show_lips, "show_lips"), (self.show_fingers, "show_fingers"),
                        (self.show_palm, "show_palm"), (self.flip, "flip")]:
            cb.setChecked(self.cfg["camera"][key])
            cb.stateChanged.connect(lambda s, k=key, c=cb: self.cfg["camera"].__setitem__(k, c.isChecked()))

        self.brightness.setValue(self.cfg["camera"]["brightness"])
        self.brightness.valueChanged.connect(lambda v: self.cfg["camera"].__setitem__("brightness", v))
        self.contrast.setValue(self.cfg["camera"]["contrast"])
        self.contrast.valueChanged.connect(lambda v: self.cfg["camera"].__setitem__("contrast", v))

        cam_form.addRow(self.show_face)
        cam_form.addRow(self.show_hands)
        cam_form.addRow(self.show_eyes)
        cam_form.addRow(self.show_brows)
        cam_form.addRow(self.show_lips)
        cam_form.addRow(self.show_fingers)
        cam_form.addRow(self.show_palm)
        cam_form.addRow("Яркость", self.brightness)
        cam_form.addRow("Контраст", self.contrast)
        cam_form.addRow(self.flip)
        cam_group.setLayout(cam_form)
        layout.addWidget(cam_group)

        # NOTE: Static gesture model selection was moved to the Camera tab.

        # Voice settings
        voice_group = QGroupBox("Голосовой помощник")
        voice_form = QFormLayout()

        self.wake_word = QLineEdit(self.cfg["voice"]["wake_word"])
        self.wake_word.textChanged.connect(lambda t: self.cfg["voice"].__setitem__("wake_word", t))

        self.mic_combo = QComboBox()
        self._populate_mic()
        self.mic_combo.currentIndexChanged.connect(
            lambda i: self.cfg["voice"].__setitem__(
                "mic_index",
                self.mic_combo.currentData() if self.mic_combo.currentData() is not None else -1))

        self.mic_gain = QSlider(Qt.Orientation.Horizontal)
        self.mic_gain.setRange(10, 100)
        self.mic_gain.setValue(int(self.cfg["voice"].get("mic_gain", 2.0) * 10))
        self.mic_gain.valueChanged.connect(
            lambda v: self.cfg["voice"].__setitem__("mic_gain", v / 10.0))

        self.vosk_model_combo = QComboBox()
        self.vosk_model_combo.setToolTip(
            "Модель распознавания речи. Применяется после перезапуска "
            "помощника. Большая модель точнее.")
        self._populate_vosk_models()
        self.vosk_model_combo.currentIndexChanged.connect(
            lambda i: self.cfg["voice"].__setitem__(
                "vosk_model_path", self.vosk_model_combo.currentData() or ""))

        # Sub-header for AI model selection (intuitive grouping)
        ai_header = QLabel("🧠 Модели нейросетей (ИИ)")
        ai_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #66ccff; padding: 6px 0 2px 0;")
        voice_form.addRow(ai_header)

        # STT engine selector: Whisper (neural) or Vosk (offline).
        self.stt_engine_combo = QComboBox()
        self.stt_engine_combo.addItem("Vosk (офлайн, потоковый)", "vosk")
        self.stt_engine_combo.addItem("Whisper (bond005/whisper-podlodka-turbo)", "whisper")
        self.stt_engine_combo.setToolTip(
            "Движок распознавания речи:\n"
            "• Vosk — быстрый, работает без интернета.\n"
            "• Whisper — точнее, но загружает большую модель.")
        _si = self.stt_engine_combo.findData(
            self.cfg["voice"].get("stt_engine", "vosk"))
        if _si >= 0:
            self.stt_engine_combo.setCurrentIndex(_si)
        self.stt_engine_combo.currentIndexChanged.connect(self._on_stt_engine_changed)

        self.whisper_model = QLineEdit(
            self.cfg["voice"].get("whisper_model", "bond005/whisper-podlodka-turbo"))
        self.whisper_model.setToolTip(
            "Имя модели HuggingFace для Whisper STT. Применяется после "
            "перезапуска помощника.")
        self.whisper_model.textChanged.connect(
            lambda t: self.cfg["voice"].__setitem__("whisper_model", t.strip()))

        self.tts_engine_combo = QComboBox()
        self.tts_engine_combo.addItem("pyttsx3 (офлайн, SAPI5)", "pyttsx3")
        self.tts_engine_combo.addItem("Silero v5 (нейросеть, ru)", "silero")
        self.tts_engine_combo.setToolTip(
            "Голос помощника:\n"
            "• pyttsx3 — встроенный голос Windows, всегда работает.\n"
            "• Silero v5 — нейросеть, звучит естественнее.")
        _ti = self.tts_engine_combo.findData(
            self.cfg["voice"].get("tts_engine", "pyttsx3"))
        if _ti >= 0:
            self.tts_engine_combo.setCurrentIndex(_ti)
        self.tts_engine_combo.currentIndexChanged.connect(self._on_tts_engine_changed)

        self.tts_voice = QComboBox()
        self._populate_voices()
        self.tts_voice.currentIndexChanged.connect(
            lambda i: self.cfg["voice"].__setitem__("tts_voice", self.tts_voice.currentData() or ""))

        self.tts_rate = QSlider(Qt.Orientation.Horizontal)
        self.tts_rate.setRange(80, 320)
        self.tts_rate.setValue(self.cfg["voice"].get("tts_rate", 160))
        self.tts_rate.valueChanged.connect(lambda v: self.cfg["voice"].__setitem__("tts_rate", v))

        self.test_voice_btn = QPushButton("🔊 Проверить голос")
        self.test_voice_btn.clicked.connect(self.test_voice)

        self.test_mic_btn2 = QPushButton("🎤 Тест микрофона")
        self.test_mic_btn2.clicked.connect(self.open_mic_test)

        self.llm_combo = QComboBox()
        self.llm_combo.addItem("OpenRouter (облако)", "openrouter")
        self.llm_combo.addItem("Ollama (локально)", "ollama")
        self.llm_combo.setToolTip(
            "Модель для ответов на вопросы:\n"
            "• OpenRouter — облачный ИИ (нужен ключ).\n"
            "• Ollama — локальный ИИ на вашем ПК.")
        # Select by the DATA value ("openrouter"/"ollama"), never the
        # display text -- otherwise the wrong value is stored and routing
        # silently falls back to OpenRouter.
        _li = self.llm_combo.findData(self.cfg["voice"].get("llm_engine", "openrouter"))
        if _li >= 0:
            self.llm_combo.setCurrentIndex(_li)
        self.llm_combo.currentIndexChanged.connect(self._on_llm_changed)

        self.or_key = QLineEdit(self.cfg["voice"]["openrouter_key"])
        self.or_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.or_key.textChanged.connect(lambda t: self.cfg["voice"].__setitem__("openrouter_key", t))
        self.or_model = QLineEdit(self.cfg["voice"]["openrouter_model"])
        self.or_model.textChanged.connect(lambda t: self.cfg["voice"].__setitem__("openrouter_model", t))

        self.or_fallback = QLineEdit(
            ", ".join(self.cfg["voice"].get("openrouter_fallback_models", [])))
        self.or_fallback.setPlaceholderText("через запятую, напр. model1, model2")
        self.or_fallback.textChanged.connect(
            lambda t: self.cfg["voice"].__setitem__(
                "openrouter_fallback_models",
                [m.strip() for m in t.split(",") if m.strip()]))

        self.ollama_url = QLineEdit(self.cfg["voice"]["ollama_url"])
        self.ollama_url.textChanged.connect(lambda t: self.cfg["voice"].__setitem__("ollama_url", t))
        self.ollama_model = QLineEdit(self.cfg["voice"]["ollama_model"])
        self.ollama_model.textChanged.connect(lambda t: self.cfg["voice"].__setitem__("ollama_model", t))

        self.system_prompt = QPlainTextEdit(self.cfg["voice"]["system_prompt"])
        self.system_prompt.textChanged.connect(
            lambda: self.cfg["voice"].__setitem__("system_prompt", self.system_prompt.toPlainText()))

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(50, 2000)
        self.max_tokens.setValue(self.cfg["voice"].get("max_tokens", 200))
        self.max_tokens.valueChanged.connect(lambda v: self.cfg["voice"].__setitem__("max_tokens", v))

        voice_form.addRow("Wake word", self.wake_word)
        voice_form.addRow("Микрофон", self.mic_combo)
        voice_form.addRow("Усиление микрофона", self.mic_gain)
        voice_form.addRow("Движок STT", self.stt_engine_combo)
        voice_form.addRow("Модель Whisper", self.whisper_model)
        voice_form.addRow("Модель Vosk (STT)", self.vosk_model_combo)
        voice_form.addRow("Движок TTS", self.tts_engine_combo)
        voice_form.addRow("Голос (озвучка)", self.tts_voice)
        voice_form.addRow("Скорость TTS", self.tts_rate)
        voice_form.addRow("", self.test_voice_btn)
        voice_form.addRow("", self.test_mic_btn2)
        voice_form.addRow("Модель LLM", self.llm_combo)
        voice_form.addRow("OpenRouter Key", self.or_key)
        voice_form.addRow("OpenRouter Model", self.or_model)
        voice_form.addRow("OpenRouter Fallback Models", self.or_fallback)
        voice_form.addRow("Ollama URL", self.ollama_url)
        voice_form.addRow("Ollama Model", self.ollama_model)
        voice_form.addRow("System prompt", self.system_prompt)
        voice_form.addRow("Max tokens", self.max_tokens)
        voice_group.setLayout(voice_form)
        layout.addWidget(voice_group)

        self.save_btn = QPushButton("💾 Сохранить настройки")
        self.save_btn.clicked.connect(lambda: save_config(self.cfg))
        layout.addWidget(self.save_btn)

        self.setLayout(layout)
        self._on_llm_changed(self.llm_combo.currentIndex())
        self._on_stt_engine_changed(self.stt_engine_combo.currentIndex())

    def _populate_mic(self):
        self.mic_combo.clear()
        self.mic_combo.addItem("Системный микрофон (по умолчанию)", -1)
        for idx, name in list_microphones():
            self.mic_combo.addItem(name, idx)
        cur = self.cfg["voice"].get("mic_index", -1)
        for i in range(self.mic_combo.count()):
            if self.mic_combo.itemData(i) == cur:
                self.mic_combo.setCurrentIndex(i)
                break

    def _populate_voices(self):
        self.tts_voice.clear()
        engine = (self.cfg["voice"].get("tts_engine") or "pyttsx3").lower()
        if engine == "silero":
            # Silero TTS v5 speakers (Russian). Stored in tts_voice so the
            # VoiceWorker._rebuild_tts picks them up directly.
            for sid, label in [
                ("xenia", "xenia (жен.)"),
                ("aidar", "aidar (муж.)"),
                ("baya", "baya (жен.)"),
                ("kseniya", "kseniya (жен.)"),
                ("eugene", "eugene (муж.)"),
            ]:
                self.tts_voice.addItem(label, sid)
            cur = self.cfg["voice"].get("tts_voice", "xenia")
            if cur not in ("aidar", "baya", "kseniya", "xenia", "eugene"):
                cur = "xenia"
        else:
            self.tts_voice.addItem("По умолчанию (русский)", "")
            for vid, name in list_voices():
                self.tts_voice.addItem(name, vid)
            cur = self.cfg["voice"].get("tts_voice", "")
        for i in range(self.tts_voice.count()):
            if self.tts_voice.itemData(i) == cur:
                self.tts_voice.setCurrentIndex(i)
                break

    def _on_tts_engine_changed(self, index):
        engine = self.tts_engine_combo.currentData() or "pyttsx3"
        self.cfg["voice"]["tts_engine"] = engine
        # Repopulate the voice list for the chosen engine (Silero vs pyttsx3).
        self._populate_voices()

    def _on_stt_engine_changed(self, index):
        engine = self.stt_engine_combo.currentData() or "vosk"
        self.cfg["voice"]["stt_engine"] = engine
        is_whisper = engine == "whisper"
        self.whisper_model.setEnabled(is_whisper)
        self.vosk_model_combo.setEnabled(not is_whisper)

    def _populate_vosk_models(self):
        self.vosk_model_combo.clear()
        self.vosk_model_combo.addItem("Авто (большая, если есть)", "")
        for p, label in list_vosk_models():
            self.vosk_model_combo.addItem(label, p)
        cur = self.cfg["voice"].get("vosk_model_path", "")
        for i in range(self.vosk_model_combo.count()):
            if self.vosk_model_combo.itemData(i) == cur:
                self.vosk_model_combo.setCurrentIndex(i)
                break

    def _on_llm_changed(self, index):
        # index is the QComboBox current index; read the DATA value so we
        # store "openrouter"/"ollama" (not the human-readable display text).
        engine = self.llm_combo.currentData() or "openrouter"
        self.cfg["voice"]["llm_engine"] = engine
        is_or = engine == "openrouter"
        # The API key and model must stay EDITABLE at all times so the key
        # can be updated even while Ollama is selected, and switching back
        # to OpenRouter works without a restart.
        self.or_key.setEnabled(True)
        self.or_model.setEnabled(True)
        self.ollama_url.setEnabled(not is_or)
        self.ollama_model.setEnabled(not is_or)

    def test_voice(self):
        engine = (self.cfg["voice"].get("tts_engine") or "pyttsx3").lower()
        vid = self.tts_voice.currentData() or None
        rate = self.tts_rate.value()

        def _run():
            try:
                if engine == "silero" and _SILERO_OK and SileroTTSEngine is not None:
                    speaker = vid if vid in (
                        "aidar", "baya", "kseniya", "xenia", "eugene") else "xenia"
                    eng = SileroTTSEngine(speaker=speaker, rate=rate, volume=0.9)
                else:
                    eng = TTSEngine(voice_id=vid, rate=rate)
                eng.speak("Привет! Это проверка голоса Акси.")
                eng.speech_queue.join()
                eng.shutdown()
            except Exception as e:
                print("voice test error:", e)

        threading.Thread(target=_run, daemon=True).start()

    def open_mic_test(self):
        dlg = MicTestDialog(self.cfg, self)
        dlg.exec()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.setWindowTitle("Axi Gesture Assistant")
        self.setGeometry(100, 100, 1000, 700)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.camera_tab = CameraTab(self.cfg)
        self.assistant_tab = AssistantTab(self.cfg)
        self.settings_tab = SettingsTab(self.cfg)

        self.tabs.addTab(self.camera_tab, "Камера")
        self.tabs.addTab(self.assistant_tab, "Помощник")
        self.tabs.addTab(self.settings_tab, "Настройки")

    def closeEvent(self, event):
        self.camera_tab.stop_camera()
        self.assistant_tab.stop_assistant()
        save_config(self.cfg)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
