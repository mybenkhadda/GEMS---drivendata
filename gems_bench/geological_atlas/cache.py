"""Atomic writes, append-only ledgers and fingerprinted stage records.

* Every file is written as ``<name>.partial`` in its destination folder and renamed with ``os.replace``; readers
  ignore ``.partial`` files, so an interrupted write never looks complete.
* Ledgers (``*.jsonl``) are append-only with ``fsync`` per record; the latest record per key wins.
* A stage is complete when ``07_cache/manifests/<stage>.json`` exists with the expected fingerprint and every
  listed output exists with its recorded size.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np

from .reliability import raise_infrastructure

PARTIAL = ".partial"


def is_partial(p: Path) -> bool:
    return Path(p).name.endswith(PARTIAL) or Path(p).name.startswith(".tmp-")


def _partial(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + PARTIAL)


def _commit(tmp: Path, path: Path) -> Path:
    os.replace(tmp, path)
    return Path(path)


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    tmp = _partial(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return _commit(tmp, path)
    except OSError as exc:
        raise_infrastructure(exc, f"writing {path}")
        raise


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def atomic_write_json(path: Path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_default))


def read_json(path: Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_csv(path: Path, df) -> Path:
    return atomic_write_text(path, df.to_csv(index=False))


def atomic_write_parquet(path: Path, df) -> Path:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return atomic_write_bytes(path, buf.getvalue())


def atomic_save_npz(path: Path, **arrays) -> Path:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return atomic_write_bytes(path, buf.getvalue())


def atomic_write_geotiff(path: Path, arr: np.ndarray, profile: dict, band_names=None, tags: dict | None = None,
                         nodata=None) -> Path:
    import rasterio

    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    prof = {k: v for k, v in profile.items() if k in ("crs", "transform")}
    prof.update(driver="GTiff", count=a.shape[0], dtype=str(a.dtype), height=a.shape[1], width=a.shape[2],
                compress="deflate", predictor=3 if a.dtype.kind == "f" else 2, tiled=True, blockxsize=256,
                blockysize=256, BIGTIFF="IF_SAFER")
    prof["nodata"] = np.nan if a.dtype.kind == "f" and nodata is None else nodata
    tmp = _partial(path)
    try:
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(a)
            for i, n in enumerate(band_names or [], start=1):
                dst.set_band_description(i, str(n))
            if tags:
                dst.update_tags(**{k: str(v) for k, v in tags.items()})
        return _commit(tmp, path)
    except OSError as exc:
        raise_infrastructure(exc, f"writing {path}")
        raise


def read_geotiff(path: Path, band: int | None = None) -> np.ndarray:
    import rasterio

    if is_partial(path):
        raise FileNotFoundError(f"refusing partial file {path}")
    with rasterio.open(path) as s:
        a = s.read(band) if band else s.read()
    return a.astype(np.float32) if a.dtype.kind == "f" else a


# ---------------------------------------------------------------------------
class Ledger:
    """Append-only JSONL ledger; ``latest()`` gives the newest record per key."""

    def __init__(self, path: Path, key: str):
        self.path, self.key = Path(path), key
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: dict) -> dict:
        rec = {**rec, "logged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        line = json.dumps(rec, default=_default) + "\n"
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise_infrastructure(exc, f"ledger {self.path.name}")
            raise
        return rec

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:   # a torn last line after a crash: ignored, the next append is complete
                continue
        return out

    def latest(self) -> dict:
        out = {}
        for r in self.records():
            out[r[self.key]] = r
        return out


# ---------------------------------------------------------------------------
def stage_record_path(stage_dir: Path, stage: str) -> Path:
    return Path(stage_dir) / f"{stage}.json"


def write_stage(stage_dir: Path, root: Path, stage: str, fingerprint: str, outputs: list[Path],
                extra: dict | None = None) -> dict:
    recs = []
    for p in outputs:
        p = Path(p)
        recs.append({"path": p.resolve().relative_to(Path(root).resolve()).as_posix(), "bytes": p.stat().st_size})
    rec = {"stage": stage, "fingerprint": fingerprint, "status": "COMPLETE", "outputs": recs,
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})}
    atomic_write_json(stage_record_path(stage_dir, stage), rec)
    return rec


def stage_status(stage_dir: Path, root: Path, stage: str, fingerprint: str | None) -> tuple[str, str]:
    rec = read_json(stage_record_path(stage_dir, stage))
    if rec is None:
        return "MISSING", "never completed"
    if fingerprint is not None and rec.get("fingerprint") != fingerprint:
        return "STALE", f"fingerprint {rec.get('fingerprint')} != {fingerprint}"
    for o in rec.get("outputs", []):
        p = Path(root) / o["path"]
        if not p.exists():
            return "INCOMPLETE", f"missing {o['path']}"
        if p.stat().st_size != o["bytes"]:
            return "INCOMPLETE", f"size changed {o['path']}"
    return "COMPLETE", "ok"
