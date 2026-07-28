#!/usr/bin/env python3
"""Main entry point for the Hand Gesture Application."""

import faulthandler
import os
import sys
from pathlib import Path


_CRASH_LOG = None


def enable_native_crash_log() -> None:
    """Keep the last Python stacks when a native camera/Qt module terminates."""
    global _CRASH_LOG
    try:
        log_dir = Path(__file__).resolve().parents[1] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _CRASH_LOG = (log_dir / "native_crash.log").open(
            "a", encoding="utf-8", buffering=1
        )
        faulthandler.enable(_CRASH_LOG, all_threads=True)
    except (OSError, RuntimeError):
        _CRASH_LOG = None

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import main

if __name__ == "__main__":
    enable_native_crash_log()
    main()
