"""Local-first, verified persistence to Google Drive.

Every expensive artifact is written to LOCAL scratch first, validated there, then:

1. copied to a temporary name on Drive (``<name>.partial<ext>``),
2. re-read from Drive and checked (size + sha256 equal to the local file),
3. finalised with an atomic rename,
4. given a completion manifest (sidecar JSON or a manifest row),
5. registered as complete.

Only artifacts that passed all five steps are skipped on restart. A crash at any point leaves either nothing or a
``.partial`` file, which recovery classifies as INCOMPLETE (never as complete).
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np

from .registry import atomic_write_json, sha256_file


class CommitError(OSError):
    pass


def _partial_name(p: Path) -> Path:
    return p.with_name(p.stem + ".partial" + p.suffix)


def commit_file(local, dest, verify: bool = True) -> dict:
    """Copy ``local`` -> ``dest`` through a verified temporary file; returns size / sha256 / mtime of ``dest``."""
    src, dst = Path(local), Path(dest)
    dst.parent.mkdir(parents=True, exist_ok=True)
    size, sha = src.stat().st_size, sha256_file(src)
    tmp = _partial_name(dst)
    shutil.copyfile(src, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    if verify:
        st = tmp.stat().st_size
        if st != size:
            tmp.unlink(missing_ok=True)
            raise CommitError(f"{dst.name}: size on destination {st} != local {size}")
        if sha256_file(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise CommitError(f"{dst.name}: sha256 on destination differs from the local file")
    os.replace(tmp, dst)
    return {"path": str(dst), "size": int(size), "sha256": sha, "mtime_ns": int(dst.stat().st_mtime_ns)}


def commit_layer(arr: np.ndarray, features_root, relpath: str, local_dir, row: dict, pool, valid=None) -> dict:
    """Write one full-grid float32 layer locally, validate it, commit it to ``features_root/relpath`` and register
    its completion-manifest row in ``pool`` (:class:`feature_versions.LayerManifest`, flushed by the caller)."""
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    if np.isinf(a).any():
        raise CommitError(f"{relpath}: infinite values")
    ld = Path(local_dir)
    ld.mkdir(parents=True, exist_ok=True)
    loc = ld / (Path(relpath).name)
    tmp = loc.with_name(loc.stem + ".partial.npy")
    np.save(tmp, a)
    tmp.replace(loc)
    v = valid if valid is not None and valid.shape == a.shape else np.ones(a.shape, bool)
    info = commit_file(loc, Path(features_root) / relpath)
    loc.unlink(missing_ok=True)
    fin = np.isfinite(a)
    rec = {**row, "relpath": relpath, "size": info["size"], "sha256": info["sha256"], "mtime_ns": info["mtime_ns"],
           "finite_fraction": float(fin[v].mean()) if v.any() else 0.0,
           "nodata_outside_valid": int((fin & ~v).sum()), "status": "VERIFIED_COMPLETE",
           "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pool.upsert([rec])
    return rec


def write_completion(path, payload: dict) -> Path:
    """Completion manifest (written only after the artifact itself was committed)."""
    return atomic_write_json(path, {**payload, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
