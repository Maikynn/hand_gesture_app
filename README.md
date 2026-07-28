# Axi Control

A Windows desktop app that combines two-hand gesture control with an embedded
Jarvis voice assistant. The interface is split into four focused pages:

- **Camera** — fixed-size live preview, equal left/right hand cards and only
  the essential camera, model, brightness, mirror and action controls.
- **Assistant** — uncluttered text chat and microphone control.
- **Camera Lab** — model files, visualization, ROI, training, profiles,
  gesture bindings, sequences and recognition history.
- **Jarvis Core** — voice, wake phrases, speech recognition, LLM, memory,
  security, scenarios, permissions, commands and diagnostics.

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

- an automatic hardware benchmark that selects ECO, BALANCED, or PERFORMANCE
  quality and keeps adapting inference cadence to real FPS/latency;
- one large camera view;
- independent previews for the left and right hands;
- labels such as `Левая: кулак` and `Правая: три`;
- camera selection, brightness and mirroring on the live page;
- crop, skeleton and advanced recognition settings on Camera Lab;
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
- an optional compressed five-second error clip saved only after the user marks
  a recognition as wrong;
- a live confidence/latency telemetry graph and drag-and-drop sequence builder;
- gesture zones (left, centre, right, top, bottom);
- automatic control-profile switching by foreground application;
- a landmark-only privacy view that never writes camera images;
- an in-app lighting, sharpness, placement and framing wizard;
- HTTP/HTTPS/RTSP phone-camera support;
- safe mode: gesture actions stay off until explicitly enabled;
- application, URL, hotkey, and safe built-in system actions.

Default bindings include play/pause on an open palm and volume control on
thumb-up/thumb-down. Potentially disruptive bindings start disabled.

## Embedded Jarvis

The Assistant and Jarvis Core pages contain:

- text chat and microphone control;
- multiple wake phrases separated by commas;
- offline Vosk, local Faster-Whisper, or online Google speech recognition;
- editable command phrases with fuzzy matching;
- a strict allowlist for executable files;
- original Priler/Jarvis Russian reaction sounds;
- local Silero TTS v5.5 Russian speech (`Eugene`) as the default high-quality
  voice, with Piper (`Денис`) as a lighter offline alternative;
- optional consent-gated XTTS v2 reference recording and personal voice engine;
- optional Edge Neural and explicitly separate Windows SAPI speech;
- no silent Microsoft SAPI substitution: the fallback is a visible user choice;
- calm, strict, and emotional voice profiles plus sentence-level streaming TTS;
- barge-in: microphone speech and Push-to-Talk interrupt current playback;
- visible synthesis/playback/fallback/error status and repeat-safe voice tests;
- automatic noise gating, lightweight echo reduction, microphone failover and
  Russian/English auto mode;
- OpenRouter, local Ollama, or any OpenAI-compatible API;
- inspectable/deletable long-term memory and multi-step action scenarios;
- local TXT/Markdown/DOCX/PDF search with an explicitly untrusted-context
  boundary before excerpts reach the selected LLM;
- local speaker verification for voice commands;
- a simulation mode that previews commands without executing them;
- an optional always-on-top mini HUD and adaptive reply prosody;
- confirmation levels for important commands and `отмени последнее`;
- system tray operation and a global `Ctrl+Alt+J` Push-to-Talk hotkey;
- `Ctrl+1` through `Ctrl+4` navigation and wheel-safe parameter controls;
- a diagnostics panel for microphone, STT, TTS, LLM and latency;
- configurable humorous fallback phrases when no model is available.

API keys are stored in Windows Credential Manager. Other choices are never
written to tracked `config.json`; they live in the git-ignored
`config.local.json`.

Malformed model responses such as `ÐÑÐ¸Ð²ÐµÑ` are repaired at the provider
boundary for both normal and streamed responses.

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
  downloaded lightweight local neural voice (ignored by Git)
models/silero/
  downloaded high-quality Russian Silero voice (ignored by Git)
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
