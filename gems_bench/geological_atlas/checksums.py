"""Chunked SHA-256 with a persistent (path, size, mtime_ns) cache, so verified files are never re-hashed."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .cache import Ledger
from .reliability import retry

CHUNK = 16 << 20


def _hash(path: Path, chunk: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = CHUNK, attempts: int = 3) -> str:
    return retry(_hash, Path(path), chunk, attempts=attempts, where=f"hashing {Path(path).name}")


def file_key(path: Path) -> tuple[str, int, int]:
    st = os.stat(path)
    return str(Path(path)), int(st.st_size), int(st.st_mtime_ns)


class HashCache:
    """sha256 per (path, size, mtime_ns); stored in an append-only ledger."""

    def __init__(self, ledger_path: Path):
        self.ledger = Ledger(ledger_path, key="key")
        self._mem = {r["key"]: r["sha256"] for r in self.ledger.latest().values()}

    @staticmethod
    def key(path: Path) -> str:
        p, s, m = file_key(path)
        return f"{p}|{s}|{m}"

    def get(self, path: Path) -> str | None:
        return self._mem.get(self.key(path))

    def sha256(self, path: Path) -> str:
        k = self.key(path)
        if k not in self._mem:
            self._mem[k] = sha256_file(path)
            self.ledger.append({"key": k, "sha256": self._mem[k]})
        return self._mem[k]

    def put(self, path: Path, sha: str):
        k = self.key(path)
        if self._mem.get(k) != sha:
            self._mem[k] = sha
            self.ledger.append({"key": k, "sha256": sha})
