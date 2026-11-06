"""Atomic writes, checksums and fingerprint-keyed stage manifests.

Every file is written to ``.tmp-<name>`` in its destination folder and moved into place with ``os.replace``,
so an interrupted session leaves either the previous complete file or a ``.tmp-`` file that every reader
ignores. A stage is complete only when its manifest exists, its fingerprint equals the expected one and every
listed output exists with the recorded size.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

TMP_PREFIX = ".tmp-"


class StaleArtifact(RuntimeError):
    """An artifact exists but was produced from other inputs / code than the current ones."""


class MissingArtifact(RuntimeError):
    """A required upstream artifact is absent or incomplete."""


def is_temporary(path: Path) -> bool:
    n = Path(path).name
    return n.startswith(TMP_PREFIX) or n.endswith((".partial", ".tmp", ".crdownload"))


def _tmp(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{TMP_PREFIX}{os.getpid()}-{path.name}")


def _commit(tmp: Path, path: Path) -> Path:
    os.replace(tmp, path)
    return Path(path)


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return _commit(tmp, path)


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def atomic_write_json(path: Path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_default))


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_save_npy(path: Path, arr: np.ndarray) -> Path:
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    return _commit(tmp, path)


def atomic_save_npz(path: Path, **arrays) -> Path:
    tmp = _tmp(path)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    return _commit(tmp, path)


def atomic_write_csv(path: Path, df) -> Path:
    return atomic_write_text(path, df.to_csv(index=False))


def atomic_write_geotiff(path: Path, arr: np.ndarray, profile: dict, band_names: list[str] | None = None) -> Path:
    import rasterio

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    prof = dict(profile)
    prof.update(driver="GTiff", count=arr.shape[0], dtype=str(arr.dtype), height=arr.shape[1], width=arr.shape[2],
                compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    if np.issubdtype(arr.dtype, np.floating):
        prof["nodata"] = np.nan
    tmp = _tmp(path)
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        for i, n in enumerate(band_names or [], start=1):
            dst.set_band_description(i, n)
    return _commit(tmp, path)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def quick_sha256(path: Path, edge: int = 1 << 20) -> str:
    """sha256 of (size, first MiB, last MiB) - used for the large DEM tiles; labelled as such in the inventory."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(edge))
        if size > edge:
            f.seek(max(size - edge, edge))
            h.update(f.read(edge))
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(f"{a.dtype}|{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# stage manifests
# ---------------------------------------------------------------------------
def output_record(path: Path, root: Path, hash_limit_bytes: int = 256 << 20) -> dict:
    path = Path(path)
    size = path.stat().st_size
    rec = {"path": Path(path).resolve().relative_to(Path(root).resolve()).as_posix(), "bytes": size}
    if size <= hash_limit_bytes:
        rec["sha256"] = sha256_file(path)
    return rec


def write_manifest(path: Path, *, stage: str, fingerprint: str, outputs: list[Path], root: Path,
                   inputs: dict | None = None, extra: dict | None = None) -> dict:
    man = {"stage": stage, "fingerprint": fingerprint, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "inputs": inputs or {}, "outputs": [output_record(p, root) for p in outputs], **(extra or {})}
    atomic_write_json(path, man)
    return man


def manifest_status(path: Path, fingerprint: str, root: Path, verify_hash: bool = False) -> tuple[str, str]:
    """('complete' | 'missing' | 'stale' | 'incomplete', reason)."""
    man = read_json(path)
    if man is None:
        return "missing", f"no manifest {Path(path).name}"
    if man.get("fingerprint") != fingerprint:
        return "stale", f"fingerprint {man.get('fingerprint')} != expected {fingerprint}"
    for rec in man.get("outputs", []):
        p = Path(root) / rec["path"]
        if not p.exists():
            return "incomplete", f"output missing: {rec['path']}"
        if p.stat().st_size != rec["bytes"]:
            return "incomplete", f"size mismatch: {rec['path']}"
        if verify_hash and "sha256" in rec and sha256_file(p) != rec["sha256"]:
            return "incomplete", f"checksum mismatch: {rec['path']}"
    return "complete", "ok"


def load_npy(path: Path, mmap: bool = False) -> np.ndarray:
    path = Path(path)
    if is_temporary(path):
        raise MissingArtifact(f"refusing temporary file {path}")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
