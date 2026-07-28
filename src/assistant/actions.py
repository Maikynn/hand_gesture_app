from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List


class ActionError(RuntimeError):
    pass


class ActionExecutor:
    """Executes only explicit, user-visible action types.

    Application actions are denied unless their absolute executable path is in
    the enabled permission list. Arbitrary shell commands are intentionally not
    supported.
    """

    SYSTEM_ACTIONS = {
        "volume_up",
        "volume_down",
        "volume_mute",
        "play_pause",
        "next_track",
        "previous_track",
        "show_desktop",
        "screenshot",
        "lock",
    }

    def __init__(
        self,
        permissions: Iterable[Dict[str, Any]] | None = None,
        *,
        app_launcher: Callable[[str], Any] | None = None,
        url_opener: Callable[[str], Any] | None = None,
        hotkey_sender: Callable[[List[str]], Any] | None = None,
    ):
        self._app_launcher = app_launcher or self._default_app_launcher
        self._url_opener = url_opener or webbrowser.open
        self._hotkey_sender = hotkey_sender or self._default_hotkey_sender
        self.permissions: List[Dict[str, Any]] = []
        self.set_permissions(permissions or [])

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(os.path.expandvars(path.strip())))

    def set_permissions(self, permissions: Iterable[Dict[str, Any]]) -> None:
        self.permissions = [
            {
                "name": str(item.get("name", "")).strip(),
                "path": str(item.get("path", "")).strip(),
                "enabled": bool(item.get("enabled", True)),
            }
            for item in permissions
            if isinstance(item, dict)
        ]

    def allowed_paths(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for item in self.permissions:
            path = item["path"]
            if path and item["enabled"] and Path(path).is_absolute():
                result[self._norm(path)] = item
        return result

    def validate(self, action: Dict[str, Any]) -> None:
        kind = str(action.get("type", "")).strip().lower()
        target = str(action.get("target", "")).strip()
        if kind == "application":
            if not target or not Path(target).is_absolute():
                raise ActionError("Для приложения нужен полный абсолютный путь к .exe")
            if self._norm(target) not in self.allowed_paths():
                raise ActionError("Приложение не добавлено в список разрешённых")
            if not Path(target).is_file():
                raise ActionError("Файл приложения не найден")
        elif kind == "url":
            if not target.startswith(("https://", "http://")):
                raise ActionError("Ссылка должна начинаться с https:// или http://")
        elif kind == "hotkey":
            if not self._parse_hotkey(target):
                raise ActionError("Укажите клавишу или сочетание, например ctrl+shift+s")
        elif kind == "system":
            if target not in self.SYSTEM_ACTIONS:
                raise ActionError("Неизвестное системное действие")
        elif kind == "response":
            if not target:
                raise ActionError("Для ответа нужен текст")
        else:
            raise ActionError(
                "Поддерживаются: приложение, ссылка, горячая клавиша, системное действие, ответ"
            )

    def execute(self, action: Dict[str, Any]) -> str:
        self.validate(action)
        kind = str(action["type"]).lower()
        target = str(action["target"]).strip()
        if kind == "application":
            self._app_launcher(target)
            name = next(
                (
                    item["name"]
                    for item in self.permissions
                    if item["enabled"] and self._norm(item["path"]) == self._norm(target)
                ),
                Path(target).stem,
            )
            return f"Открываю {name}"
        if kind == "url":
            self._url_opener(target)
            return "Открываю ссылку"
        if kind == "hotkey":
            self._hotkey_sender(self._parse_hotkey(target))
            return f"Выполняю {target}"
        if kind == "response":
            return target
        self._execute_system(target)
        labels = {
            "volume_up": "Делаю громче",
            "volume_down": "Делаю тише",
            "volume_mute": "Переключаю звук",
            "play_pause": "Переключаю воспроизведение",
            "next_track": "Следующий трек",
            "previous_track": "Предыдущий трек",
            "show_desktop": "Показываю рабочий стол",
            "screenshot": "Открываю снимок экрана",
            "lock": "Блокирую компьютер",
        }
        return labels[target]

    @staticmethod
    def _parse_hotkey(value: str) -> List[str]:
        return [part.strip().lower() for part in value.replace(" ", "+").split("+") if part.strip()]

    @staticmethod
    def _default_app_launcher(path: str) -> None:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([path])

    @staticmethod
    def _default_hotkey_sender(keys: List[str]) -> None:
        try:
            import pyautogui
        except ImportError as exc:
            raise ActionError("Не установлен пакет pyautogui") from exc
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)

    def _execute_system(self, action: str) -> None:
        if action == "lock" and os.name == "nt":
            subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"])
            return
        mapping = {
            "volume_up": ["volumeup"],
            "volume_down": ["volumedown"],
            "volume_mute": ["volumemute"],
            "play_pause": ["playpause"],
            "next_track": ["nexttrack"],
            "previous_track": ["prevtrack"],
            "show_desktop": ["win", "d"],
            "screenshot": ["win", "shift", "s"],
        }
        keys = mapping.get(action)
        if not keys:
            raise ActionError("Системное действие недоступно")
        self._hotkey_sender(keys)


ACTION_LABELS = {
    "application": "Приложение",
    "url": "Ссылка",
    "hotkey": "Горячая клавиша",
    "system": "Системное действие",
    "response": "Только ответ",
}

ACTION_TYPES_BY_LABEL = {value: key for key, value in ACTION_LABELS.items()}
