# Axi Control

A Windows desktop app that combines two-hand gesture control with an embedded
Jarvis voice assistant. The application has only two main pages:

- **Camera** — large live preview, separate left/right hand crops, gesture
  labels, camera/model/brightness controls, and gesture bindings.
- **Assistant** — chat, wake phrases, microphone and TTS controls, application
  permissions, editable commands, and LLM provider settings.

The assistant runs inside the Python/PyQt process. It does not open a separate
Jarvis window, Tauri app, Vite server, or background assistant executable.

## Quick start

Python 3.12 is required because the project pins the Windows-compatible
MediaPipe 0.10.21 build.

```bat
setup_venv.bat
run.bat
```

## Camera and gestures

The camera page provides:

- one large camera view;
- independent previews for the left and right hands;
- labels such as `Левая: кулак` and `Правая: три`;
- camera selection, brightness, mirroring, crop padding, and skeleton toggles;
- rule-based MediaPipe recognition by default;
- optional built-in or user-selected ONNX/TFLite models;
- bundled YOLO HaGRID model with a selectable `.pt` path;
- stable-frame filtering and one action per held gesture;
- a personal hand calibration wizard and in-app custom-gesture recording;
- gesture sequences such as `fist → palm → two_up`;
- Work, Games, Music, and Presentation control profiles;
- configurable recognition ROI and automatic light/contrast/white balance;
- an on-video HUD with FPS, confidence, latency, cooldown, model and device;
- YOLO CUDA acceleration with automatic CPU fallback;
- hold time and optional second-gesture confirmation against accidental actions;
- a persistent history where wrong recognitions can be marked for retraining;
- safe mode: gesture actions stay off until explicitly enabled;
- application, URL, hotkey, and safe built-in system actions.

Default bindings include play/pause on an open palm and volume control on
thumb-up/thumb-down. Potentially disruptive bindings start disabled.

## Embedded Jarvis

The assistant page contains:

- text chat and microphone control;
- multiple wake phrases separated by commas;
- offline Vosk, local Faster-Whisper, or online Google speech recognition;
- editable command phrases with fuzzy matching;
- a strict allowlist for executable files;
- original Priler/Jarvis Russian reaction sounds;
- local Piper Neural Russian speech (`Денис`) that stays offline after setup;
- optional Edge Neural and explicitly separate Windows SAPI speech;
- no silent Microsoft SAPI substitution: the fallback is a visible user choice;
- calm, strict, and emotional voice profiles plus sentence-level streaming TTS;
- barge-in: microphone speech and Push-to-Talk interrupt current playback;
- visible synthesis/playback/fallback/error status and repeat-safe voice tests;
- OpenRouter, local Ollama, or any OpenAI-compatible API;
- inspectable/deletable long-term memory and multi-step action scenarios;
- confirmation levels for important commands and `отмени последнее`;
- system tray operation and a global `Ctrl+Alt+J` Push-to-Talk hotkey;
- a diagnostics panel for microphone, STT, TTS, LLM and latency;
- configurable humorous fallback phrases when no model is available.

API keys and every choice made in the interface are never written to tracked
`config.json`. They are stored only in the git-ignored `config.local.json`.

## Configuration

- `config.json` — versioned defaults.
- `config.local.json` — all personal settings and API keys; created
  automatically and ignored.
- `config.local.example.json` — safe example file.

To open an application, add its absolute `.exe` path to **Assistant → Access
to applications**, then reference that same path in a voice command or gesture
binding. Arbitrary shell commands are intentionally unsupported.

## Structure

```text
src/
  assistant/
    actions.py           safe action executor
    embedded_jarvis.py   command router and Priler voice reactions
    listener.py          Vosk/Whisper/Google wake-word listener
    llm_client.py        OpenRouter/Ollama/custom API routing
  camera/                camera, MediaPipe and advanced gesture features
  hand_processing/       gesture recognition and optional models
  ui/
    main_window.py       shell, camera page, gesture bindings
    assistant_window.py  chat and assistant settings
    theme.py             dark/light themes
  utils/config_store.py  atomic public/local configuration
third_party/priler_jarvis/
  original source and voice/command resources
models/vosk-ru/
  offline Russian speech-recognition model
models/piper/
  downloaded local neural voice (ignored by Git)
models/whisper/
  downloaded Faster-Whisper model (ignored by Git)
```

## Tests

```bat
venv\Scripts\python.exe -m unittest discover -s tests -v
venv\Scripts\python.exe test_gesture_recognition.py
venv\Scripts\python.exe -m ruff check src tests
```

## License and attribution

This adapted distribution is provided under
[CC BY-NC-SA 4.0](LICENSE). It is for non-commercial use and must retain
attribution and ShareAlike terms.

Priler/Jarvis attribution and the exact integration changes are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
