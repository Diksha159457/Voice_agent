"""Persistent chat history stored as a JSON file.

Writes are atomic (temp file + rename) and serialised with a lock, so a crash
or two concurrent requests can't leave a half-written, unreadable file.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

MAX_ENTRIES = 200
_lock = threading.Lock()


class HistoryStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def save(self, entries: list[dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".history-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entries[-MAX_ENTRIES:], fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            pass  # read-only filesystem (some hosts): history just isn't persisted

    def append(self, entry: dict) -> None:
        with _lock:
            entries = self.load()
            entries.append(entry)
            self.save(entries)

    def clear(self) -> None:
        with _lock:
            self.save([])
