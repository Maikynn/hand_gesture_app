"""Isolated XTTS process.

Coqui's dependency range conflicts with MediaPipe's protobuf pin, so XTTS runs
inside ``models/xtts_env`` and only exchanges UTF-8 text and a WAV file with
the main application.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="ru")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("TTS_HOME", str(project_root / "models" / "xtts_cache"))
    from TTS.api import TTS

    model = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
    )
    text = Path(args.text_file).read_text(encoding="utf-8")
    model.tts_to_file(
        text=text,
        speaker_wav=args.reference,
        language=args.language,
        file_path=args.output,
        split_sentences=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
