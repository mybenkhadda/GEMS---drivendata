"""Atomic writes, content hashes and fingerprinted stage records.

* every file goes to ``<name>.partial`` (GeoTIFF: ``<stem>.partial.tif``) and is moved with ``os.replace``;
* every write passes :meth:`RecoveryPaths.check_writable` - nothing is ever written into the V1 run or Atlas;
* a stage record ``manifests/<stage>.json`` stores fingerprint, outputs (path, size, sha256) and status;
  it is valid only if the fingerprint matches AND every output still has its recorded size and hash.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

_GUARD = None


def set_guard(paths) -> None:
    global _GUARD
    _GUARD = paths


def _target(path) -> Path:
    path = Path(path)
    if _GUARD is not None:
        _GUARD.check_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _partial(path: Path) -> Path:
    if path.suffix.lower() in (".tif", ".tiff"):
        return path.with_name(path.stem + ".partial" + path.suffix)
    return path.with_name(path.name + ".partial")


def atomic_write_bytes(path, data: bytes) -> Path:
    path = _target(path)
    tmp = _partial(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def read_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def atomic_write_csv(path, df: pd.DataFrame) -> Path:
    return atomic_write_text(path, df.to_csv(index=False))


def atomic_write_parquet(path, df: pd.DataFrame) -> Path:
    buf = io.BytesIO()
    out = df.copy()
    for c in out.columns:                                   # mixed object columns -> strings (pyarrow-safe)
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: v if v is None or isinstance(v, str) else json.dumps(v, default=_jsonable))
    out.to_parquet(buf, index=False)
    return atomic_write_bytes(path, buf.getvalue())


def atomic_write_npz(path, **arrays) -> Path:
    path = _target(path)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_geotiff(path, array: np.ndarray, profile: dict, descriptions=None, nodata="auto") -> Path:
    """Write a 2-D or 3-D array with the grid ``profile``; reopen and check shape / CRS / transform."""
    import rasterio

    path = _target(path)
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    p = {k: v for k, v in profile.items() if k in ("crs", "transform", "width", "height")}
    is_float = np.issubdtype(a.dtype, np.floating)
    if is_float:
        a = a.astype(np.float32)
    p.update(driver="GTiff", count=a.shape[0], dtype=a.dtype.name, tiled=True, blockxsize=256, blockysize=256,
             compress="deflate", predictor=3 if is_float else 2, height=a.shape[1], width=a.shape[2],
             BIGTIFF="IF_SAFER")
    if nodata == "auto":
        p["nodata"] = np.nan if is_float else None
    else:
        p["nodata"] = nodata
    tmp = _partial(path)
    with rasterio.open(tmp, "w", **p) as ds:
        ds.write(a)
        for i, d in enumerate(descriptions or [], start=1):
            ds.set_band_description(i, str(d))
    with rasterio.open(tmp) as ds:
        if ds.count != a.shape[0] or ds.shape != a.shape[1:] or ds.transform != p["transform"] or ds.crs != p["crs"]:
            raise RuntimeError(f"GeoTIFF readback mismatch: {tmp}")
    os.replace(tmp, path)
    return path


def read_raster(path, band: int = 1, out_shape=None, resampling=None) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        if out_shape is None:
            a = ds.read(band, masked=True)
        else:
            a = ds.read(band, masked=True, out_shape=out_shape, resampling=resampling)
        return a.astype(np.float32).filled(np.nan)


def sha256_file(path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def array_sha(a: np.ndarray, n: int = 16) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode() + a.dtype.str.encode()).hexdigest()[:n]


class HashCache:
    """SHA-256 per (path, size, mtime_ns) in an append-only JSONL file - a file is hashed once."""

    def __init__(self, path):
        self.path = Path(path)
        self.mem: dict = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.mem[(r["path"], r["size"], r["mtime_ns"])] = r["sha256"]
                except (json.JSONDecodeError, KeyError):
                    continue                                   # torn last line

    def sha(self, path) -> str:
        p = Path(path)
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
        if key not in self.mem:
            self.mem[key] = sha256_file(p)
            _target(self.path)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"path": key[0], "size": key[1], "mtime_ns": key[2], "sha256": self.mem[key]}) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return self.mem[key]


class StageStore:
    def __init__(self, manifest_dir, hashes: HashCache):
        self.dir = Path(manifest_dir)
        self.hashes = hashes

    def path(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def record(self, stage: str, fingerprint: str, outputs, stats: dict | None = None, inputs: dict | None = None,
               status: str = "COMPLETE") -> dict:
        outs = []
        for o in outputs:
            o = Path(o)
            if o.is_file():
                outs.append({"path": str(o), "size": o.stat().st_size, "sha256": self.hashes.sha(o)})
        rec = {"stage": stage, "fingerprint": fingerprint, "status": status, "outputs": outs,
               "stats": stats or {}, "inputs": inputs or {}, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_write_json(self.path(stage), rec)
        return rec

    def valid(self, stage: str, fingerprint: str) -> tuple[bool, str]:
        rec = read_json(self.path(stage))
        if rec is None:
            return False, "MISSING"
        if rec.get("status") != "COMPLETE":
            return False, f"INCOMPLETE ({rec.get('status')})"
        if rec.get("fingerprint") != fingerprint:
            return False, "STALE (fingerprint changed)"
        for o in rec.get("outputs", []):
            p = Path(o["path"])
            if not p.is_file():
                return False, f"INCOMPLETE (missing {p.name})"
            if p.stat().st_size != o["size"] or self.hashes.sha(p) != o["sha256"]:
                return False, f"STALE (content changed: {p.name})"
        return True, "COMPLETE"

    def load(self, stage: str) -> dict | None:
        return read_json(self.path(stage))
