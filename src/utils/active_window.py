from __future__ import annotations

import ctypes
import os
from pathlib import Path


def foreground_process_name() -> str:
    """Return the foreground executable name without optional dependencies."""

    if os.name != "nt":
        return ""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    process_id = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    process = kernel32.OpenProcess(0x1000, False, process_id.value)
    if not process:
        return ""
    try:
        size = ctypes.c_ulong(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(
            process, 0, buffer, ctypes.byref(size)
        ):
            return Path(buffer.value).name.casefold()
    finally:
        kernel32.CloseHandle(process)
    return ""


def parse_profile_rules(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in str(value).replace("\n", ";").split(";"):
        process, separator, profile = chunk.partition("=")
        if separator and process.strip() and profile.strip():
            result[process.strip().casefold()] = profile.strip()
    return result


def match_profile(process_name: str, rules: dict[str, str]) -> str:
    value = process_name.casefold()
    for fragment, profile in rules.items():
        if fragment in value:
            return profile
    return ""
