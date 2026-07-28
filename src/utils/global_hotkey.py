"""Small Windows RegisterHotKey wrapper used by global Push-to-Talk."""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes
from typing import Dict, Optional

from PyQt5.QtCore import QObject, pyqtSignal


class GlobalHotkey(QObject):
    activated = pyqtSignal()
    status = pyqtSignal(bool, str)

    MODIFIERS: Dict[str, int] = {
        "alt": 0x0001,
        "ctrl": 0x0002,
        "control": 0x0002,
        "shift": 0x0004,
        "win": 0x0008,
    }

    def __init__(self, sequence: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.sequence = sequence
        self._thread: Optional[threading.Thread] = None
        self._thread_id = 0
        self._stop = threading.Event()

    @classmethod
    def parse(cls, sequence: str) -> tuple[int, int]:
        parts = [
            value.strip().lower()
            for value in str(sequence).replace(" ", "").split("+")
            if value.strip()
        ]
        if not parts:
            raise ValueError("Пустая комбинация Push-to-Talk")
        modifiers = 0
        key = ""
        for part in parts:
            if part in cls.MODIFIERS:
                modifiers |= cls.MODIFIERS[part]
            else:
                key = part
        if len(key) == 1 and key.isalnum():
            virtual_key = ord(key.upper())
        elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
            virtual_key = 0x70 + int(key[1:]) - 1
        else:
            raise ValueError("Поддерживаются буква, цифра или F1–F24")
        return modifiers, virtual_key

    def start(self) -> None:
        if os.name != "nt":
            self.status.emit(False, "Глобальная клавиша поддерживается только в Windows")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="axi-global-hotkey", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            modifiers, virtual_key = self.parse(self.sequence)
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            self._thread_id = int(kernel32.GetCurrentThreadId())
            hotkey_id = 0xA71
            if not user32.RegisterHotKey(None, hotkey_id, modifiers, virtual_key):
                raise RuntimeError("комбинация уже занята другой программой")
            self.status.emit(True, f"Global Push-to-Talk: {self.sequence}")
            message = wintypes.MSG()
            try:
                while not self._stop.is_set():
                    result = int(user32.GetMessageW(ctypes.byref(message), None, 0, 0))
                    if result <= 0:
                        break
                    if message.message == 0x0312 and int(message.wParam) == hotkey_id:
                        self.activated.emit()
            finally:
                user32.UnregisterHotKey(None, hotkey_id)
        except Exception as exc:
            self.status.emit(False, f"Global Push-to-Talk: {exc}")
        finally:
            self._thread_id = 0

    def stop(self) -> None:
        self._stop.set()
        if self._thread_id and os.name == "nt":
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
