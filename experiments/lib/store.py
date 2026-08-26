"""
Append-only JSONL checkpoint store.

The brief requires that a crash lose at most one trial and that a rerun
resume rather than restart. JSONL gives both cheaply: every completed unit is
one flushed line, and a torn final line from a hard kill is detected and
dropped on the next read instead of corrupting the whole file (which is what
would happen with a single json.dump at the end of a run).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class JsonlStore:
    def __init__(self, path: Path, key_fields: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = key_fields
        self._lock = threading.Lock()
        self._done: set[tuple] = set()
        self._load()

    def _key(self, row: dict) -> tuple:
        return tuple(row.get(f) for f in self.key_fields)

    def _load(self) -> None:
        if not self.path.exists():
            return
        good, dropped = [], 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1  # torn final line from a hard kill
        if dropped:
            # Rewrite without the torn line so the file stays clean.
            with open(self.path, "w", encoding="utf-8") as f:
                for row in good:
                    f.write(json.dumps(row) + "\n")
            print(f"  [store] dropped {dropped} torn line(s) from {self.path.name}")
        self._done = {self._key(r) for r in good}

    def has(self, row: dict) -> bool:
        return self._key(row) in self._done

    def append(self, row: dict) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
            self._done.add(self._key(row))

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def __len__(self) -> int:
        return len(self._done)
