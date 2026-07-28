"""Local, inspectable long-term memory for the embedded assistant."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = PROJECT_ROOT / "user_data" / "assistant_memory.json"


class MemoryStore:
    def __init__(self, path: Path = MEMORY_PATH, limit: int = 100):
        self.path = path
        self.limit = max(10, int(limit))
        self.items: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = [dict(item) for item in data if isinstance(item, dict)][
                -self.limit :
            ]
        except (OSError, json.JSONDecodeError, TypeError):
            self.items = []

    def remember(self, text: str) -> Dict[str, Any]:
        value = " ".join(str(text).strip().split())
        if not value:
            raise ValueError("Нечего запоминать")
        item = {"text": value, "created_at": time.time()}
        self.items.append(item)
        self.items = self.items[-self.limit :]
        self.save()
        return item

    def delete(self, index: int) -> None:
        if 0 <= index < len(self.items):
            self.items.pop(index)
            self.save()

    def clear(self) -> None:
        self.items.clear()
        self.save()

    def prompt_context(self, count: int = 20) -> str:
        values = [str(item.get("text", "")).strip() for item in self.items[-count:]]
        values = [value for value in values if value]
        if not values:
            return ""
        return "Долговременная память пользователя:\n" + "\n".join(
            f"- {value}" for value in values
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(self.path)
