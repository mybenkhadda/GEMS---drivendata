"""Stage records: completion, provenance and fingerprints of every pipeline stage, persisted on Drive.

``environment/pipeline/stages/<stage>.json`` (one per stage, atomic writes, mirrored to Drive)::

    {"stage", "status": done | running | interrupted | failed,
     "origin": built | resumed | adopted_existing,
     "fingerprint": {...}, "fingerprint_hash", "outputs": {rel: {bytes, mtime_ns[, sha256]}},
     "git_commit", "package_version", "started", "finished", "seconds", "summary", "error", "recovery"}

``environment/pipeline/events.jsonl`` is an append-only log of stage transitions (one line per event).

A record never makes an artifact valid by itself: the executor always re-validates outputs. The record
answers the other question - *were these outputs produced under the current configuration?* - and keeps
provenance (code revision, inputs, timing) for reports. Artifacts that existed before the pipeline (legacy
uploads, earlier notebooks) are *adopted*: validated, fingerprinted, recorded with ``origin=adopted_existing``.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..artifacts.atomic import atomic_write_json

DIR = "environment/pipeline"
STAGE_DIR = f"{DIR}/stages"
EVENTS = f"{DIR}/events.jsonl"


def fp_hash(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:16]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class StageManifests:
    local_root: Path
    drive_root: Path | None = None

    def __post_init__(self) -> None:
        self.local_root = Path(self.local_root)
        self.drive_root = Path(self.drive_root) if self.drive_root else None

    def _rel(self, stage: str) -> str:
        return f"{STAGE_DIR}/{stage}.json"

    def load(self, stage: str) -> dict:
        """Local record, else the Drive copy (new Colab session), else {}."""
        for root in (self.local_root, self.drive_root):
            if root is None:
                continue
            p = root / self._rel(stage)
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
        return {}

    def save(self, stage: str, rec: dict) -> dict:
        rec = {**rec, "stage": stage, "updated": now()}
        atomic_write_json(self.local_root / self._rel(stage), rec)
        self._mirror(self._rel(stage))
        return rec

    def update(self, stage: str, **fields) -> dict:
        return self.save(stage, {**self.load(stage), **fields})

    def event(self, stage: str, event: str, **info) -> None:
        line = json.dumps({"time": now(), "stage": stage, "event": event, **info}, default=str)
        p = self.local_root / EVENTS
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() and self.drive_root and (self.drive_root / EVENTS).exists():
            p.write_text((self.drive_root / EVENTS).read_text(encoding="utf-8"), encoding="utf-8")
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._mirror(EVENTS)

    def _mirror(self, rel: str) -> None:
        if self.drive_root is None or not self.drive_root.exists():
            return
        try:
            src = self.local_root / rel
            dst = self.drive_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".partial")
            tmp.write_bytes(src.read_bytes())
            tmp.replace(dst)
        except OSError as e:                     # Drive hiccup: the local record stays authoritative
            print(f"[pipeline] record {rel} not mirrored to Drive: {e}")

    def all(self, names: list[str]) -> dict[str, dict]:
        return {n: self.load(n) for n in names}

    def events(self, last: int = 200):
        import pandas as pd

        for root in (self.local_root, self.drive_root):
            if root and (root / EVENTS).exists():
                lines = (root / EVENTS).read_text(encoding="utf-8").splitlines()[-last:]
                return pd.DataFrame([json.loads(x) for x in lines if x.strip()])
        return pd.DataFrame(columns=["time", "stage", "event"])


def output_fingerprints(root: Path, rels: list[str], sha_max_bytes: int = 8 << 20) -> dict:
    """{rel: {bytes, mtime_ns[, sha256]}} for files (sha256 only for small files: tables, JSON, small rasters;
    feature caches and maps are identified by size + mtime) / {files, bytes} for directories."""
    from ..artifacts.validation import fingerprint

    out = {}
    for rel in rels:
        p = Path(root) / rel
        if not p.exists():
            out[rel] = None
            continue
        small = p.is_file() and p.stat().st_size <= sha_max_bytes
        out[rel] = fingerprint(p, sha=small)
    return out
