"""Atomic artifacts, fingerprints and stage manifests (restart-safe, no legacy recovery).

A stage is *complete* only when its manifest exists, its fingerprint equals the current one, and every listed
output file exists with the recorded size (and, for small files, the recorded sha256). Anything else is
recomputed. Writes go to ``<name>.partial`` and are renamed, so an interrupted write never looks complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

HASH_LIMIT = 64 * 2**20  # sha256 files up to 64 MiB; larger files: size + sampled hash


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path, obj) -> Path:
    return atomic_write_bytes(path, json.dumps(obj, indent=2, default=_json_default).encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_save_npy(path: Path, arr: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
    return path


def atomic_save_table(path: Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def file_fingerprint(path: Path) -> dict:
    """size + sha256 (full for small files, head/middle/tail 4 MiB samples for large ones)."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= HASH_LIMIT:
            for chunk in iter(lambda: f.read(2**20), b""):
                h.update(chunk)
            kind = "sha256"
        else:
            for off in (0, size // 2, max(size - 4 * 2**20, 0)):
                f.seek(off)
                h.update(f.read(4 * 2**20))
            h.update(str(size).encode())
            kind = "sampled_sha256"
    return {"name": path.name, "size": size, "hash": h.hexdigest(), "hash_kind": kind}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def combine(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Workspace:
    """Local working tree (fast disk) mirrored to a persistent output tree (Drive).

    Artifacts are written locally, then copied to ``out`` with :meth:`persist`. ``stage_*`` methods keep one JSON
    manifest per stage under ``manifests/``.
    """

    SUBDIRS = (
        "manifests",
        "inventory",
        "splits",
        "features",
        "quality",
        "eda",
        "regimes",
        "models",
        "predictions",
        "ablation",
        "fusion",
        "topology",
        "dti_opt",
        "failure_modes",
        "submissions",
        "figures",
        "reports",
        "tables",
    )

    def __init__(self, work: str | Path, out: str | Path | None = None):
        self.root = Path(work)
        self.out = Path(out) if out else None
        for s in self.SUBDIRS:
            (self.root / s).mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ manifests
    def manifest_path(self, stage: str) -> Path:
        return self.root / "manifests" / f"{stage}.json"

    def read_manifest(self, stage: str) -> dict | None:
        for base in (self.root, self.out):
            if base is None:
                continue
            p = base / "manifests" / f"{stage}.json"
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return None
        return None

    def stage_complete(self, stage: str, fingerprint: str) -> tuple[bool, str]:
        m = self.read_manifest(stage)
        if m is None:
            return False, "no manifest"
        if m.get("status") != "complete":
            return False, f"status {m.get('status')}"
        if m.get("fingerprint") != fingerprint:
            return False, "fingerprint changed"
        for rel, fp in m.get("outputs", {}).items():
            p = self.root / rel
            if not p.exists() and self.out is not None and (self.out / rel).exists():
                self.restore(rel)
            if not p.exists():
                return False, f"missing output {rel}"
            if p.stat().st_size != fp["size"]:
                return False, f"size changed {rel}"
            if fp.get("hash_kind") == "sha256" and fp["size"] <= 4 * 2**20 and file_fingerprint(p)["hash"] != fp["hash"]:
                return False, f"content changed {rel}"
        return True, "verified"

    def write_manifest(
        self,
        stage: str,
        fingerprint: str,
        outputs: list[Path],
        summary: dict | None = None,
        status: str = "complete",
        seconds: float | None = None,
    ) -> dict:
        rec = {
            "stage": stage,
            "status": status,
            "fingerprint": fingerprint,
            "created_unix": time.time(),
            "seconds": seconds,
            "summary": summary or {},
            "outputs": {str(Path(p).relative_to(self.root)): file_fingerprint(p) for p in outputs if Path(p).exists()},
        }
        atomic_write_json(self.manifest_path(stage), rec)
        return rec

    # ------------------------------------------------------------------ persistence
    # large, recomputable trees are not mirrored by default; predictions are persisted selectively by the engine
    PERSIST_EXCLUDE = ("features", "cache", "predictions")

    def persist(self, rels: list[str | Path] | None = None, exclude: tuple = PERSIST_EXCLUDE) -> int:
        """Copy local artifacts to the persistent tree (skip unchanged by size + mtime). Returns files copied."""
        if self.out is None:
            return 0
        n = 0
        if rels:
            items = [self.root / r for r in rels]
        else:
            items = [p for p in self.root.rglob("*") if p.is_file() and p.relative_to(self.root).parts[0] not in exclude]
        for src in items:
            if not src.exists() or src.name.endswith(".partial") or ".partial." in src.name:
                continue
            files = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file()]
            for f in files:
                rel = f.relative_to(self.root)
                dst = self.out / rel
                if dst.exists() and dst.stat().st_size == f.stat().st_size and dst.stat().st_mtime >= f.stat().st_mtime:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = _tmp(dst)
                shutil.copy2(f, tmp)
                os.replace(tmp, dst)
                n += 1
        return n

    def restore(self, rel: str | Path) -> bool:
        if self.out is None:
            return False
        src = self.out / rel
        if not src.exists():
            return False
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp(dst)
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return True

    def restore_tree(self, rel: str) -> int:
        """Copy a persisted subtree back to the local work tree where files are missing (resume on a new runtime)."""
        if self.out is None or not (self.out / rel).exists():
            return 0
        n = 0
        for src in (self.out / rel).rglob("*"):
            if src.is_file() and not (self.root / src.relative_to(self.out)).exists() and ".partial" not in src.name:
                self.restore(src.relative_to(self.out))
                n += 1
        return n

    def log(self, stage: str, event: str, **info) -> None:
        rec = {"t": time.time(), "stage": stage, "event": event, **info}
        self.events.append(rec)
        with open(self.root / "reports" / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
