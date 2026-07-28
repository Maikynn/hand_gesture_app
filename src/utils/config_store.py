from __future__ import annotations

import json
import os
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

from PyQt5.QtCore import QObject, pyqtSignal

from utils.credential_vault import CredentialVault

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CONFIG_PATH = PROJECT_ROOT / "config.json"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.json"


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class ConfigStore(QObject):
    """Thread-safe public/local settings store.

    ``config.json`` contains versioned defaults. UI choices are written to the
    git-ignored ``config.local.json``; API secrets use Windows Credential
    Manager in the normal application store. This keeps the repository clean
    and prevents updates from replacing personal settings.
    """

    changed = pyqtSignal(str, object)
    saved = pyqtSignal()

    def __init__(
        self,
        public_path: Path | None = None,
        local_path: Path | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.public_path = public_path or PUBLIC_CONFIG_PATH
        self.local_path = local_path or LOCAL_CONFIG_PATH
        self._lock = threading.RLock()
        self._vault = (
            CredentialVault() if public_path is None and local_path is None else None
        )
        self._public = _read_json(self.public_path)
        self._local = _read_json(self.local_path)
        migrated = False
        local_assistant = self._local.get("assistant")
        if isinstance(local_assistant, dict) and self._vault is not None:
            local_keys = local_assistant.get("api_keys")
            if isinstance(local_keys, dict):
                for provider, value in tuple(local_keys.items()):
                    if str(value).strip() and self._vault.set(provider, str(value)):
                        local_keys[provider] = ""
                        migrated = True
        self._data = _deep_merge(self._public, self._local)
        if migrated:
            _write_json_atomic(self.local_path, self._local)

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    def public_dict(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._public)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        with self._lock:
            value: Any = self._data
            for part in dotted_key.split("."):
                if not isinstance(value, dict) or part not in value:
                    return default
                value = value[part]
            return deepcopy(value)

    @staticmethod
    def _put(root: Dict[str, Any], dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = root
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = deepcopy(value)

    def set(self, dotted_key: str, value: Any, *, save: bool = True) -> None:
        with self._lock:
            self._put(self._local, dotted_key, value)
            self._put(self._data, dotted_key, value)
            if save:
                self._save_locked()
        self.changed.emit(dotted_key, deepcopy(value))

    def update_many(self, values: Dict[str, Any]) -> None:
        with self._lock:
            for key, value in values.items():
                self._put(self._local, key, value)
                self._put(self._data, key, value)
            self._save_locked()
        for key, value in values.items():
            self.changed.emit(key, deepcopy(value))

    def replace_section(self, section: str, value: Any, *, local: bool = True) -> None:
        with self._lock:
            target = self._local if local else self._public
            target[section] = deepcopy(value)
            self._data = _deep_merge(self._public, self._local)
            self._save_locked()
        self.changed.emit(section, deepcopy(value))

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        # Versioned defaults are read-only during normal application use.
        # ``save`` therefore cannot accidentally publish personal paths/keys.
        if self._local:
            _write_json_atomic(self.local_path, self._local)
        self.saved.emit()

    def api_key(self, provider: str) -> str:
        if self._vault is not None:
            secret = self._vault.get(provider)
            if secret:
                return secret
        return str(self.get(f"assistant.api_keys.{provider}", "") or "")

    def set_api_key(self, provider: str, value: str) -> None:
        secret = str(value).strip()
        if self._vault is not None and self._vault.set(provider, secret):
            self.set(f"assistant.api_keys.{provider}", "")
            return
        self.set(f"assistant.api_keys.{provider}", secret)

    def export_without_secrets(self) -> Dict[str, Any]:
        """Return a safe support snapshot suitable for logs or bug reports."""
        snapshot = self.as_dict()
        assistant = snapshot.get("assistant")
        if isinstance(assistant, dict):
            assistant["api_keys"] = {
                key: "***" if value else ""
                for key, value in assistant.get("api_keys", {}).items()
            }
        return snapshot

    def all_keys(self) -> Iterable[str]:
        def walk(prefix: str, node: Any):
            if isinstance(node, dict):
                for key, value in node.items():
                    next_key = f"{prefix}.{key}" if prefix else key
                    yield from walk(next_key, value)
            else:
                yield prefix

        return tuple(walk("", self.as_dict()))
