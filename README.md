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
- stable-frame filtering and one action per held gesture;
- safe mode: gesture actions stay off until explicitly enabled;
- application, URL, hotkey, and safe built-in system actions.

Default bindings include play/pause on an open palm and volume control on
thumb-up/thumb-down. Potentially disruptive bindings start disabled.

## Embedded Jarvis

The assistant page contains:

- text chat and microphone control;
- multiple wake phrases separated by commas;
- offline Vosk or online Google speech recognition;
- editable command phrases with fuzzy matching;
- a strict allowlist for executable files;
- original Priler/Jarvis Russian reaction sounds;
- a deep Microsoft neural Russian voice with speed, pitch, and volume controls;
- automatic offline Windows voice fallback when the network is unavailable;
- OpenRouter, local Ollama, or any OpenAI-compatible API;
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
    listener.py          Vosk/Google wake-word listener
    llm_client.py        OpenRouter/Ollama/custom API routing
  camera/                camera and MediaPipe tracking
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
