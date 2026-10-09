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
from typing import Any


class JsonlStore:
    """Thread-safe, append-only JSONL file with an in-memory resume index.

    Each row is one JSON object on one line. On construction the file is read
    once and the key of every row already on disk is held in a set, so a
    caller can ask "is this unit of work done?" in constant time and skip it.

    Args:
        path: File to read from and append to. Missing parent directories are
            created; the file itself is created by the first append.
        key_fields: Names of the row fields that together identify one unit
            of work. Pass an empty list to use the store purely as a reader.

    Attributes:
        path: Location of the backing file.
        key_fields: The fields that make up a row's identity.
    """

    def __init__(self, path: Path | str, key_fields: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = key_fields
        self._lock = threading.Lock()
        self._done: set[tuple[Any, ...]] = set()
        self._load()

    def _key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        """Return the identity of a row; a missing key field reads as None."""
        return tuple(row.get(f) for f in self.key_fields)

    def _load(self) -> None:
        """Index the rows already on disk, repairing a torn file if needed.

        A process killed mid-write leaves a partial last line. Any line that
        does not parse as JSON is dropped, the file is rewritten without it,
        and the number dropped is reported on stdout. A missing file is an
        empty store.
        """
        if not self.path.exists():
            return
        good: list[dict[str, Any]] = []
        dropped = 0
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

    def has(self, row: dict[str, Any]) -> bool:
        """Report whether a row with the same key is already on disk.

        Args:
            row: Any mapping that carries the key fields. Other fields are
                ignored, so a partial "probe" row is enough.

        Returns:
            True if a row with this key was found on load or has been appended
            since. The check is on the key alone, not on the row's outcome: a
            trial recorded as failed is present too and is not re-attempted.
        """
        return self._key(row) in self._done

    def append(self, row: dict[str, Any]) -> None:
        """Write one row as a single line and add its key to the index.

        The write is serialised by a lock, so concurrent workers cannot
        interleave lines, and it is flushed before the lock is released, so a
        crash of this process loses at most the row being written. The data
        is flushed to the operating system, not fsynced.

        Args:
            row: A JSON-serialisable mapping.
        """
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
            self._done.add(self._key(row))

    def read_all(self) -> list[dict[str, Any]]:
        """Return every row on disk, in file order.

        Reads the file afresh instead of using the index, so rows that share
        a key are all returned. Blank and unparseable lines are skipped.
        """
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def __len__(self) -> int:
        """Return the number of distinct keys held, not the number of lines."""
        return len(self._done)
