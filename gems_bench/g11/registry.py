"""Output provenance and resumable stages.

* :func:`atomic_*` writers (``.partial`` + ``os.replace``);
* :class:`RunContext`: output root, config hash, code commit, source hashes - stamped into every manifest;
* :class:`StageStore`: ``manifests/<stage>.json`` with fingerprint + outputs (path, size, sha256); a stage is
  COMPLETE only if the fingerprint matches and every output still has its recorded size and hash (idempotent,
  resumable).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import code_commit, config_hash, resolve_output_root, stable_hash

STAGE_DIRS = {"runtime": "00_runtime", "audit": "01_audit", "ingest": "02_ingestion", "features": "03_features",
              "eda": "04_eda", "splits": "05_splits", "train": "06_models", "predictions": "07_predictions",
              "infer": "08_candidates", "evaluate": "09_evaluation", "submit": "10_submissions", "reports": "11_reports",
              "checkpoints": "checkpoints", "logs": "logs"}
MODEL_DIRS = {"pixel": "xgboost", "patch": "segformer", "multires": "gated_fusion", "fusion": "fusion",
              "segment": "segment_ranker"}
LAYOUT = list(STAGE_DIRS.values()) + ["manifests"] + [f"06_models/{d}" for d in MODEL_DIRS.values()]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _partial(p: Path) -> Path:
    if p.suffix.lower() in (".tif", ".tiff", ".npy", ".gpkg"):
        return p.with_name(p.stem + ".partial" + p.suffix)
    return p.with_name(p.name + ".partial")


def atomic_write_bytes(path, data: bytes) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(p)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


def atomic_write_text(path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def atomic_write_parquet(path, df: pd.DataFrame) -> Path:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            vals = out[c].dropna()
            if len(vals) and vals.map(lambda v: isinstance(v, (bool, np.bool_))).all():
                out[c] = out[c].astype("boolean")                      # keep booleans boolean (not "false")
            elif len(vals) and vals.map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, bool)).all():
                out[c] = pd.to_numeric(out[c])
            else:
                out[c] = out[c].map(lambda v: json.dumps(v, default=_json_default)
                                    if isinstance(v, (dict, list, tuple, np.ndarray)) else (None if v is None else str(v)))
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    return atomic_write_bytes(path, buf.getvalue())


def atomic_write_npy(path, a: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(p)
    np.save(tmp, a)
    os.replace(tmp, p)
    return p


def read_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def sha256_file(path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class HashCache:
    """sha256 per (path, size, mtime_ns), persisted as JSONL - each file is hashed once."""

    def __init__(self, path):
        self.path = Path(path)
        self.mem: dict = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.mem[(r["p"], r["s"], r["m"])] = r["h"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def sha(self, path) -> str:
        p = Path(path)
        st = p.stat()
        k = (str(p), st.st_size, st.st_mtime_ns)
        if k not in self.mem:
            self.mem[k] = sha256_file(p)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"p": k[0], "s": k[1], "m": k[2], "h": self.mem[k]}) + "\n")
        return self.mem[k]


@dataclass
class RunContext:
    cfg: dict
    root: Path = field(init=False)
    cfg_hash: str = field(init=False)
    commit: str | None = field(init=False)
    source_hashes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.root = resolve_output_root(self.cfg)
        self.cfg_hash = config_hash(self.cfg)
        self.commit = code_commit()
        self.hashes = HashCache(self.root / "cache" / "sha256.jsonl")

    def d(self, stage: str, *parts) -> Path:
        p = self.root.joinpath(STAGE_DIRS.get(stage, stage), *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def ensure_layout(self) -> Path:
        for d in LAYOUT:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self.root

    def stamp(self, extra: dict | None = None) -> dict:
        return {"config_hash": self.cfg_hash, "code_commit": self.commit, "source_hashes": self.source_hashes,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})}

    def write_manifest(self, path, payload: dict) -> Path:
        return atomic_write_json(path, {**self.stamp(), **payload})


class StageStore:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.dir = ctx.root / "manifests"

    def fingerprint(self, stage: str, *inputs) -> str:
        return stable_hash([stage, self.ctx.cfg_hash, self.ctx.source_hashes, *inputs])

    def _p(self, stage: str) -> Path:
        """``manifests/<stage>.json``; a stage key may be namespaced (``<run namespace>/train_E0``)."""
        p = self.dir / f"{stage}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def outputs_ok(self, stage: str) -> tuple[bool, str]:
        """COMPLETE record whose every output still has its recorded size and sha256 (fingerprint not compared)."""
        r = read_json(self._p(stage))
        if r is None or r.get("status") != "COMPLETE":
            return False, "no COMPLETE record"
        for o in r.get("outputs", []):
            p = Path(o["path"])
            if not p.is_file() or p.stat().st_size != o["size"] or self.ctx.hashes.sha(p) != o["sha256"]:
                return False, f"{p.name} missing or changed"
        return True, "outputs verified"

    def status(self, stage: str, fp: str) -> tuple[bool, str]:
        r = read_json(self._p(stage))
        if r is None:
            return False, "MISSING"
        if r.get("status") != "COMPLETE":
            return False, f"INCOMPLETE ({r.get('status')})"
        if r.get("fingerprint") != fp:
            return False, "STALE (fingerprint changed)"
        for o in r.get("outputs", []):
            p = Path(o["path"])
            if not p.is_file() or p.stat().st_size != o["size"] or self.ctx.hashes.sha(p) != o["sha256"]:
                return False, f"STALE ({p.name} missing or changed)"
        return True, "COMPLETE"

    def begin(self, stage: str, fp: str) -> None:
        """Mark a stage RUNNING (start time kept until completion; a crash leaves an INCOMPLETE record)."""
        self._started = getattr(self, "_started", {})
        self._started[stage] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.ctx.write_manifest(self._p(stage), {"stage": stage, "fingerprint": fp, "status": "RUNNING",
                                                             "started_utc": self._started[stage], "outputs": []})

    def fail(self, stage: str, fp: str, error: BaseException) -> None:
        self.ctx.write_manifest(self._p(stage), {
            "stage": stage, "fingerprint": fp, "status": "FAILED", "outputs": [],
            "started_utc": getattr(self, "_started", {}).get(stage), "error": f"{type(error).__name__}: {error}"[:4000]})

    def record(self, stage: str, fp: str, outputs, stats: dict | None = None, status: str = "COMPLETE",
               validation: dict | None = None) -> dict:
        outs = [{"path": str(p), "size": Path(p).stat().st_size, "sha256": self.ctx.hashes.sha(p)}
                for p in outputs if Path(p).is_file()]
        rec = {"stage": stage, "fingerprint": fp, "status": status, "outputs": outs, "stats": stats or {},
               "validation": validation or {}, "started_utc": getattr(self, "_started", {}).get(stage),
               "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.ctx.write_manifest(self._p(stage), rec)
        return rec

    def load(self, stage: str) -> dict | None:
        return read_json(self._p(stage))
