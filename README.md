# Hand Gesture Application

A desktop application for controlling a computer with hand gestures and the
integrated Axi voice assistant. The modern interface has two focused tabs:
**Camera** and **Assistant**; all related settings live next to the feature they
configure, so there is no separate settings page.

## Features

### 1. Camera tab
- Live webcam feed with real-time skeleton overlay
- Separate left/right hand previews with gesture names and confidence
- Camera, recognition model, and brightness controls
- Configurable actions for each hand/gesture pair, with safe defaults
- Rule-based recognition fallback (works without a trained model)
- Optional trained model support (TFLite / ONNX) for higher accuracy

### 2. Assistant tab ("Axi")
- Embedded command core based on the command/alias approach used by
  [Priler/Jarvis](https://github.com/Priler/jarvis); it runs inside this tab and
  is not launched as a separate application
- Editable wake phrase and Vosk/Whisper speech recognition
- Integrated chat and pyttsx3/Silero speech output
- OpenRouter, local Ollama, or any OpenAI-compatible API
- A friendly configurable fallback phrase when no AI provider is available
- Explicit application allow-list: arbitrary programs cannot be launched
- User-defined phrases mapped to application, URL, volume, or hotkey actions
- Dark and light themes available from the main header

> Security: API keys are blank in the repository. Add them through the Assistant
> tab on your own machine and never commit a populated `config.json`.
> Runtime settings and secrets are written atomically to the ignored
> `config.local.json`; the tracked `config.json` remains a safe template.

## Architecture

| Component | Responsibility | Key Libraries |
|-----------|----------------|---------------|
| `VideoCapture` | Webcam acquisition with fallback | `opencv-python` |
| `SkeletonRenderer` | Face/hand/body landmark drawing + raw landmark extraction | `mediapipe` |
| `HandCropProcessor` | 150×150 hand crop extraction with padding | `numpy`, `cv2` |
| `GestureRecognizer` | Rule-based + model-based gesture classification | MediaPipe landmarks / TFLite / ONNX |
| `ModelManager` | Model loading (TFLite/ONNX), inference, switching | `tensorflow`, `onnxruntime` |
| `SettingsWindow` | UI for camera, skeleton toggles, calibration | `PyQt5` |
| `AssistantWindow` | Wake-word, OpenRouter, TTS, commands | `PyQt5`, `requests`, `pyttsx3` |
| `WakeWordListener` | Offline speech recognition + wake-word | `vosk` / `speech_recognition` |
| `OpenRouterClient` | Cloud LLM API communication | `requests` |
| `TTSEngine` | Offline speech synthesis | `pyttsx3` |
| `Logger` | File + console logging | `logging` |

## Installation

> **IMPORTANT — Python version requirement**
> MediaPipe 0.10.35 on **Python 3.14** has **no working vision API** — both the
> legacy `mp.solutions` and the new `mediapipe.tasks.vision` were removed/omitted
> from that build, so hand/face/pose detection cannot run on 3.14.
> This project therefore runs inside a **Python 3.12 virtual environment** that
> uses `mediapipe==0.10.14` (the last release that still ships `mp.solutions`).

### Option A — Automated (Windows)

```bash
cd hand_gesture_app
setup_venv.bat      # creates the venv + installs mediapipe 0.10.14 and all deps
```

### Option B — Manual

```bash
# 1. Install Python 3.12 (if not present), e.g.:
winget install Python.Python.3.12

# 2. Create and activate a venv
cd hand_gesture_app
py -3.12 -m venv venv
venv\Scripts\activate.bat

# 3. Install dependencies
pip install "mediapipe==0.10.14" PyQt6 PyQt5 requests pyttsx3 onnxruntime SpeechRecognition
```

**Note for Windows:** `pyaudio` requires the PortAudio C library. If you need
microphone wake-word detection, install a prebuilt wheel from
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio. Without it, the assistant's
text-input mode still works (typing commands instead of speaking).

## Running

```bash
# Full application (hand_gesture_app):
run.bat                 # Windows — uses the venv automatically
# or:
venv\Scripts\activate.bat
python src/main.py

# Standalone gui_app.py (PyQt6 single-file app, in this folder):
python gui_app.py       # auto-switches into the venv (mediapipe 0.10.14)
# or use the launcher:
run_gui_app.bat
```

> `gui_app.py` contains a small bootstrap that automatically re-executes it inside
> the Python 3.12 venv, so the plain command `python gui_app.py` works directly
> (no manual venv activation needed). It is a lighter PyQt6 app whose
> camera/skeleton tab works with this venv. The full three-window assistant with
> wake-word + OpenRouter + TTS is `src/main.py` (hand_gesture_app).

## Gesture Recognition

The app supports two recognition modes:

1. **Rule-based (default, no model needed)**: Uses MediaPipe hand landmarks to classify common gestures:
   - `fist`, `open_palm`, `point`, `victory`, `thumbs_up`, `thumbs_down`
   - `ok_sign`, `rock`, `peace`, `call_me`, `gun`, `pinch`, `three`, `four`, `five`
   - Plus 25+ additional gestures (see `models/gesture_classes.txt`)
   - Palm-side classification (front/back) via landmark depth (z-coordinate)

2. **Model-based (optional)**: Place a trained `built_in.tflite` or `built_in.onnx` in `models/`, or load a custom model via Settings. The model must output 41 classes matching `models/gesture_classes.txt`.

### Training a Custom Model

```bash
python train_model.py --data_dir datasets --epochs 50
```

See `datasets/README.md` for data organization. The training script exports to `models/built_in.tflite`.

## Project Structure

```
hand_gesture_app/
├─ models/
│   ├─ built_in.tflite          # optional trained model (TFLite)
│   ├─ built_in.onnx            # optional trained model (ONNX)
│   ├─ gesture_classes.txt      # 40 gesture names + "unknown"
│   ├─ custom/                  # user-provided models
│   └─ README.md
├─ datasets/                    # training data (see datasets/README.md)
├─ logs/                        # application logs
├─ venv/                        # Python 3.12 virtual environment (created by setup_venv.bat)
├─ src/
│   ├─ main.py                  # entry point
│   ├─ ui/
│   │   ├─ main_window.py       # 3-window stacked UI
│   │   ├─ settings_window.py   # settings UI + apply_settings signal
│   │   └─ assistant_window.py  # assistant UI
│   ├─ camera/
│   │   ├─ video_capture.py     # OpenCV wrapper
│   │   └─ skeleton_renderer.py # MediaPipe drawing + landmark extraction
│   ├─ hand_processing/
│   │   ├─ hand_crop.py         # hand crop extraction
│   │   └─ gesture_recognizer.py# rule-based + model inference
│   ├─ assistant/
│   │   ├─ listener.py          # wake-word detection
│   │   ├─ openrouter_client.py # OpenRouter API
│   │   └─ tts_engine.py        # TTS engine
│   └─ utils/
│       ├─ logger.py            # logging setup
│       └─ model_manager.py     # model loading/inference
├─ gui_app.py                   # standalone PyQt6 app (auto-bootstraps into the venv)
├─ train_model.py               # training script
├─ test_gesture_recognition.py  # unit tests for recognizer
├─ hand_gesture_app.spec        # PyInstaller spec
├─ setup_venv.bat               # creates the Python 3.12 venv
├─ run.bat                      # launches src/main.py in the venv
├─ run_gui_app.bat              # launches gui_app.py in the venv
├─ requirements.txt
└─ README.md
```

## Testing

```bash
venv\Scripts\activate.bat
python test_gesture_recognition.py
```

Verifies rule-based recognition for fist, open_palm, point, and victory gestures.

## Packaging (Windows)

```bash
venv\Scripts\activate.bat
pip install pyinstaller
pyinstaller hand_gesture_app.spec
```

Output: `dist/HandGestureApp/HandGestureApp.exe`

## Notes

- All perception (hand/face/pose tracking) runs **offline** via MediaPipe.
- Assistant works **offline** with Vosk + pyttsx3; OpenRouter requires internet + API key.
- If no model is present, the app automatically uses rule-based recognition (no crash).
- Logs are written to `logs/app.log` for debugging.
- **Requires Python 3.12 (via the bundled `venv`)** — Python 3.14's MediaPipe build lacks the vision API needed for skeleton/gesture detection. `gui_app.py` auto-switches into the venv; for `src/main.py` use `run.bat` or activate the venv first.
