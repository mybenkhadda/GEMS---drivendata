"""Artifacts, manifests and cache correctness.

A cached stage is reused only when ALL hold (``validate_cached_stage``):
1. its manifest fingerprint equals the current one (config section + upstream fingerprints + module source hash);
2. every required output exists, is readable and carries its required columns;
3. the manifest itself is readable and its status is COMPLETE;
4. the feature-schema hash matches (stages that consume features);
5. the grid matches (stages that write rasters);
6. the recorded provenance checks passed.

Writes are atomic (``gems_bench.artifacts.atomic``); an earlier successful output that is about to be replaced is
moved to ``history/<stage>/<old fingerprint>/`` first and indexed in ``history/index.csv`` - never silently
overwritten. Drive synchronisation reuses ``gems_bench.artifacts.sync.DriveSync``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

COMPLETE = "COMPLETE"
REUSED = "REUSED_VALID_CACHE"
FAILED = "FAILED"
BLOCKED_UPSTREAM = "BLOCKED_UPSTREAM"
BLOCKED_MISSING = "BLOCKED_MISSING_INPUT"
NOT_REQUESTED = "NOT_REQUESTED"
SKIPPED = "SKIPPED"
NOT_RUN = "NOT_RUN"
VALID = (COMPLETE, REUSED)


def stable_hash(*parts, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:n]


def code_identity(modules) -> dict:
    """{module: sha256(source)[:16]} of the modules a stage executes (a code change invalidates that stage)."""
    out = {}
    for m in modules:
        mod = importlib.import_module(m if m.startswith("gems_bench") else f"gems_bench.g06.{m}")
        p = Path(getattr(mod, "__file__", "") or "")
        out[m] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "unavailable"
    return out


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def artifact_fingerprint(path, mode: str = "stat") -> dict | None:
    """{bytes, mtime_ns[, sha256]} of a file (``mode='sha256'`` hashes the content); directories aggregate their
    files (sorted relative paths + sizes [+ hashes]). None when absent."""
    p = Path(path)
    if not p.exists():
        return None
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file() and not f.name.endswith(".partial"))
        parts = [(f.relative_to(p).as_posix(), f.stat().st_size, sha256_file(f) if mode == "sha256" else None)
                 for f in files]
        return {"bytes": int(sum(x[1] for x in parts)), "n_files": len(parts), "sha256": stable_hash(parts, n=64)}
    st = p.stat()
    out = {"bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    if mode == "sha256":
        out["sha256"] = sha256_file(p)
    return out


def atomic_write(path, writer) -> Path:
    """``writer(tmp_path)`` then an atomic rename onto ``path`` (format inferred from the kept suffix)."""
    from ..artifacts.atomic import atomic_save

    return atomic_save(Path(path), writer)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path, obj) -> Path:
    text = json.dumps(obj, indent=2, default=_jsonable, sort_keys=False)
    return atomic_write(path, lambda t: Path(t).write_text(text, encoding="utf-8"))


def read_json(path) -> dict | list | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda x: x if x is None or isinstance(x, str) else json.dumps(x, default=_jsonable))
    return out


def write_table(path, df: pd.DataFrame | None) -> Path:
    p = Path(path)
    df = df if df is not None else pd.DataFrame()
    if p.suffix == ".parquet":
        return atomic_write(p, lambda t: _parquet_safe(df).to_parquet(t, index=False))
    return atomic_write(p, lambda t: df.to_csv(t, index=False))


def read_table(path, columns=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix == ".parquet":
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(p, usecols=columns) if p.stat().st_size else pd.DataFrame()


# =============================================================================== stage manifests
def manifest_path(root, stage: str) -> Path:
    return Path(root) / "manifests" / "stages" / f"{stage}.json"


def save_stage_manifest(root, stage: str, status: str, fingerprint: str | None, outputs, info: dict | None = None,
                        **extra) -> dict:
    m = {"stage": stage, "status": status, "fingerprint": fingerprint,
         "required_outputs": [o if isinstance(o, str) else o[0] for o in (outputs or [])],
         "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {}), **extra}
    write_json(manifest_path(root, stage), m)
    return m


def load_cached_stage(root, stage: str) -> dict | None:
    m = read_json(manifest_path(root, stage))
    return m if isinstance(m, dict) else None


def check_outputs(root, outputs) -> tuple[list[str], list[str]]:
    """(valid, problems) for ``outputs`` = [(rel, required columns | None)] or [rel]: exists, readable, columns."""
    ok, bad = [], []
    for o in outputs or []:
        rel, cols = (o, None) if isinstance(o, str) else (o[0], o[1])
        p = Path(root) / rel
        if not p.exists():
            bad.append(f"{rel}: missing")
            continue
        try:
            if p.is_dir():
                if not any(p.iterdir()):
                    bad.append(f"{rel}: empty directory")
                    continue
                T = None
            elif p.suffix == ".csv":
                T = pd.read_csv(p, nrows=5) if p.stat().st_size else pd.DataFrame()
            elif p.suffix == ".parquet":
                import pyarrow.parquet as pq

                T = pd.DataFrame(columns=pq.read_schema(p).names)
            elif p.suffix == ".json":
                json.loads(p.read_text(encoding="utf-8"))
                T = None
            elif p.suffix in (".tif", ".tiff"):
                import rasterio

                with rasterio.open(p) as ds:
                    _ = ds.count
                T = None
            elif p.suffix == ".npz":
                with np.load(p) as z:
                    _ = z.files
                T = None
            else:
                T = None
        except Exception as e:
            bad.append(f"{rel}: unreadable ({type(e).__name__})")
            continue
        miss = [c for c in (cols or ()) if T is None or c not in T.columns]
        if miss:
            bad.append(f"{rel}: columns {miss} absent")
        else:
            ok.append(rel)
    return ok, bad


def validate_cached_stage(root, stage: str, fingerprint: str, outputs, schema_hash: str | None = None,
                          grid: dict | None = None) -> tuple[bool, list[str]]:
    """The six reuse conditions of the module docstring; (reusable, problems)."""
    m = load_cached_stage(root, stage)
    if m is None:
        return False, ["manifest absent or unreadable"]
    probs = []
    if m.get("status") != COMPLETE:
        probs.append(f"status {m.get('status')} (only COMPLETE is reusable)")
    if m.get("fingerprint") != fingerprint:
        probs.append("fingerprint differs")
    _, bad = check_outputs(root, outputs)
    probs += bad
    if schema_hash is not None and m.get("feature_schema_hash") not in (None, schema_hash):
        probs.append(f"feature schema {m.get('feature_schema_hash')} != {schema_hash}")
    if grid is not None and m.get("grid") not in (None, grid):
        probs.append("grid differs")
    if m.get("provenance_ok") is False:
        probs.append("recorded provenance checks failed")
    return not probs, probs


def archive_outputs(root, stage: str, rels, tag: str | None) -> list[str]:
    """Move existing ``rels`` to ``history/<stage>/<tag>/`` (index in history/index.csv); returns moved paths."""
    root = Path(root)
    tag = (tag or "untagged")[:10]
    moved = []
    for rel in rels or []:
        src = root / rel
        if not src.exists():
            continue
        dst = root / "history" / stage / tag / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        moved.append(rel)
    if moved:
        idx = root / "history" / "index.csv"
        row = pd.DataFrame([{"stage": stage, "fingerprint": tag, "archived": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "paths": ";".join(moved)}])
        old = read_table(idx)
        write_table(idx, pd.concat([old, row], ignore_index=True) if len(old) else row)
    return moved


def write_artifact_inventory(root, out_rel: str = "manifests/artifact_inventory.csv") -> pd.DataFrame:
    """Every file of the run directory (history excluded) with size, modification time and top-level section."""
    root = Path(root)
    rows = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name.endswith(".partial"):
            continue
        rel = f.relative_to(root).as_posix()
        if rel.startswith(("history/", "tiles/")) or rel == out_rel:
            continue
        st = f.stat()
        rows.append({"path": rel, "section": rel.split("/")[0], "bytes": int(st.st_size),
                     "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
                     "kind": f.suffix.lstrip(".")})
    T = pd.DataFrame(rows, columns=["path", "section", "bytes", "modified", "kind"])
    write_table(root / out_rel, T)
    return T


# =============================================================================== workspace (local run dir + Drive)
class Workspace:
    """Local run directory ``local_root/<output_subdir>/<run_id>`` mirrored to the same relative path on Drive."""

    def __init__(self, local_root, drive_root, subdir: str, sync: bool = True):
        self.local_root, self.drive_root = Path(local_root), Path(drive_root)
        self.subdir = subdir.strip("/")
        self.root = self.local_root / self.subdir
        self.root.mkdir(parents=True, exist_ok=True)
        self.sync = None
        if sync:
            from ..artifacts.sync import DriveSync

            self.sync = DriveSync(self.local_root, self.drive_root)

    def path(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def drive_path(self, rel: str = "") -> Path:
        return self.drive_root / self.subdir / rel

    def save_json(self, obj, rel: str) -> Path:
        return write_json(self.path(rel), obj)

    def read_json(self, rel: str):
        return read_json(self.root / rel)

    def save_table(self, df: pd.DataFrame | None, rel: str) -> Path:
        return write_table(self.path(rel), df)

    def load_table(self, rel: str, columns=None) -> pd.DataFrame:
        return read_table(self.root / rel, columns)

    def save_text(self, text: str, rel: str) -> Path:
        return atomic_write(self.path(rel), lambda t: Path(t).write_text(text, encoding="utf-8"))

    def save_npz(self, rel: str, **arrays) -> Path:
        return atomic_write(self.path(rel), lambda t: np.savez_compressed(t, **arrays))

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def stage_in(self, rels) -> None:
        """Copy earlier outputs from Drive (resume after a disconnect)."""
        if self.sync is None:
            return
        want = [f"{self.subdir}/{r}" for r in rels if (self.drive_path(r)).exists()]
        if want:
            self.sync.stage_in(want)

    def push(self, rels=None, exclude=None) -> None:
        """Mirror ``rels`` (default: the whole run directory) to Drive; tile caches excluded by default."""
        if self.sync is None:
            return
        targets = [f"{self.subdir}/{r}" for r in rels] if rels else [self.subdir]
        self.sync.push(targets, exclude=list(exclude or ["*/tiles/*"]))
