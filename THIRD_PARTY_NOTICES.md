# Third-party notices

## Priler/Jarvis

The embedded assistant is adapted from ideas, command packs, voice reactions,
and source code from [Priler/Jarvis](https://github.com/Priler/jarvis), created
by Abraham Tugalov (Priler).

- Upstream revision: `520b98143fdcf72bad855f722e7b32932d61cd46`
- Upstream license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)
- The original license text is preserved in
  `third_party/priler_jarvis/LICENSE.txt` and at the repository root.
- A source snapshot is preserved in `third_party/priler_jarvis`.

### Modifications

The original Rust/Tauri assistant was adapted into the existing Python/PyQt
process. The new integration:

- does not launch the original Tauri UI or a second assistant executable;
- ports the fuzzy command matching and localized command concept to Python;
- exposes wake phrases, permissions, commands, providers, and chat in one tab;
- uses an explicit application allowlist instead of arbitrary CLI execution;
- uses the original Russian WAV reaction pack when available;
- uses the Russian Vosk model from the upstream resources for offline STT;
- adds OpenRouter, Ollama, and generic OpenAI-compatible provider routing.

Four upstream Git LFS objects in the two sentence-transformer model folders
could not be downloaded because the upstream repository's public LFS budget
was exhausted. GitHub rejects unknown LFS pointers in a new repository, so
their paths, sizes, and SHA-256 identifiers are preserved in
`third_party/priler_jarvis/UNAVAILABLE_LFS_OBJECTS.md`. The embedded Python
implementation does not require those objects.

## Vosk model

The Russian Vosk speech-recognition model under `models/vosk-ru` originated
from the Priler/Jarvis resource snapshot. Vosk models and toolkit components
retain their upstream notices and licensing terms.

## Ultralytics YOLO

The optional HaGRID neural gesture backend uses the separately installed
[Ultralytics](https://github.com/ultralytics/ultralytics) Python runtime.
Ultralytics is distributed under AGPL-3.0 unless covered by a separate
enterprise license. The bundled `models/yolo/hagrid_best.pt` contains the
project's trained gesture weights and is loaded only when YOLO is selected.

## Piper neural speech

The optional local neural speech backend uses
[OHF-Voice Piper](https://github.com/OHF-Voice/piper1-gpl) 1.6.0, distributed
under GPL-3.0-or-later. `setup_venv.bat` downloads the Russian
`ru_RU-denis-medium` voice from the public
[Piper voices collection](https://huggingface.co/rhasspy/piper-voices). The
model is installed locally and is intentionally not committed to this
repository.

## Silero Russian neural speech

The default high-quality Russian voice uses
[Silero TTS v5.5](https://github.com/snakers4/silero-models) with the
`v5_5_ru` model. The model supports Russian stress, homographs and question
intonation and is distributed under CC-NC-BY. It is downloaded by
`setup_venv.bat` into the Git-ignored `models/silero` directory.

## Coqui TTS / XTTS v2 (optional)

The consent-gated personal-voice option can install
[coqui-tts](https://pypi.org/project/coqui-tts/) and use XTTS v2 lazily.
The Python package is distributed under MPL-2.0. XTTS model terms are shown by
the upstream downloader and must be accepted by the user before the model is
used. No personal reference recording is bundled or committed.

## pypdf

Local PDF text extraction uses [pypdf](https://pypi.org/project/pypdf/),
distributed under the BSD-3-Clause license.

## Faster-Whisper

Local neural speech recognition uses
[SYSTRAN Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 and
CTranslate2 4.8.1, both distributed under the MIT license. The `tiny`
multilingual model is downloaded during setup into the Git-ignored
`models/whisper` directory.
