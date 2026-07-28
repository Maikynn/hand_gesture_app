"""Permission-aware execution of assistant commands and gesture actions."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, Tuple


ACTION_LABELS = {
    "none": "Ничего",
    "open_app": "Открыть приложение",
    "open_url": "Открыть сайт",
    "volume_up": "Громче",
    "volume_down": "Тише",
    "mute": "Без звука",
    "hotkey": "Сочетание клавиш",
}


class ActionExecutor:
    """Executes only explicit actions configured by the user."""

    def __init__(self, allowed_apps: Iterable[Dict[str, str]] = ()):
        self.allowed_apps = list(allowed_apps)

    def update_allowed_apps(self, apps: Iterable[Dict[str, str]]) -> None:
        self.allowed_apps = list(apps)

    def execute(self, action: str, value: str = "") -> Tuple[bool, str]:
        if action == "none":
            return True, "Действие отключено."
        if action == "open_url":
            url = value.strip()
            if not url:
                return False, "Не указан адрес сайта."
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            if not webbrowser.open(url):
                return False, "Системный браузер не смог открыть ссылку."
            return True, f"Открываю {url}"
        if action == "open_app":
            return self._open_allowed_app(value)
        if action in {"volume_up", "volume_down", "mute"}:
            return self._media_key(action)
        if action == "hotkey":
            return self._hotkey(value)
        return False, "Неизвестное действие."

    def _open_allowed_app(self, requested: str) -> Tuple[bool, str]:
        requested = requested.strip().casefold()
        app = next(
            (
                item
                for item in self.allowed_apps
                if requested
                in {item.get("name", "").casefold(), item.get("path", "").casefold()}
            ),
            None,
        )
        if not app:
            return (
                False,
                "Приложение не разрешено. Добавьте его полный путь во вкладке «Помощник».",
            )
        path = Path(app.get("path", "")).expanduser()
        if not path.is_file():
            return False, f"Файл приложения не найден: {path}"
        if os.name == "nt":
            try:
                os.startfile(str(path))
            except OSError as exc:
                return False, f"Не удалось запустить приложение: {exc}"
        else:
            try:
                subprocess.Popen([str(path)])
            except OSError as exc:
                return False, f"Не удалось запустить приложение: {exc}"
        return True, f"Запускаю {app.get('name') or path.name}."

    @staticmethod
    def _media_key(action: str) -> Tuple[bool, str]:
        try:
            if os.name == "nt":
                script = {
                    "volume_up": "$w=New-Object -ComObject WScript.Shell;$w.SendKeys([char]175)",
                    "volume_down": "$w=New-Object -ComObject WScript.Shell;$w.SendKeys([char]174)",
                    "mute": "$w=New-Object -ComObject WScript.Shell;$w.SendKeys([char]173)",
                }[action]
                subprocess.Popen(["powershell", "-NoProfile", "-Command", script])
            elif sys.platform == "darwin":
                return False, "Управление громкостью для macOS пока не настроено."
            else:
                subprocess.Popen(
                    [
                        "pactl",
                        "set-sink-mute" if action == "mute" else "set-sink-volume",
                        "@DEFAULT_SINK@",
                        "toggle"
                        if action == "mute"
                        else ("+5%" if action == "volume_up" else "-5%"),
                    ]
                )
        except OSError as exc:
            return False, f"Не удалось изменить громкость: {exc}"
        return True, {
            "volume_up": "Делаю громче.",
            "volume_down": "Делаю тише.",
            "mute": "Переключаю звук.",
        }[action]

    @staticmethod
    def _hotkey(value: str) -> Tuple[bool, str]:
        try:
            import pyautogui
        except ImportError:
            return False, "Для сочетаний клавиш установите pyautogui."
        keys = [key.strip().lower() for key in value.split("+") if key.strip()]
        if not keys:
            return False, "Укажите сочетание, например ctrl+shift+s."
        try:
            pyautogui.hotkey(*keys)
        except Exception as exc:
            return False, f"Не удалось нажать сочетание клавиш: {exc}"
        return True, f"Нажимаю {' + '.join(keys)}."
