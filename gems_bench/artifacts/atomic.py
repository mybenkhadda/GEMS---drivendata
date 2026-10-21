"""Atomic file writes and integrity checks.

Every durable artifact is written to a temporary sibling (same directory, so ``os.replace`` is an atomic
rename on local disks and on the Colab Drive FUSE mount) and only then moved into place. A crash or a
Colab disconnect therefore leaves either the previous complete file or the new complete file, never a
truncated one. Temporaries are named ``.<name>.tmp-<pid>-<rand>`` and are ignored by synchronisation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable

TMP_MARK = ".tmp-"


def is_temporary(path: Path) -> bool:
    n = Path(path).name
    return n.startswith(".") and TMP_MARK in n or n.endswith(".partial")


def _tmp_for(path: Path, suffix: str = "") -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}{TMP_MARK}{os.getpid()}-{uuid.uuid4().hex[:8]}{suffix}")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fsync(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:                       # some FUSE mounts refuse fsync on read-only handles
        pass


def atomic_save(path: Path, writer: Callable[[Path], None], keep_suffix: bool = True) -> Path:
    """Call ``writer(tmp_path)`` then rename onto ``path``. ``keep_suffix`` keeps the extension on the
    temporary (drivers such as GTiff / Parquet infer the format from it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path, path.suffix if keep_suffix else "")
    try:
        writer(tmp)
        _fsync(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    return atomic_save(path, lambda t: Path(t).write_bytes(data))


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_save(path, lambda t: Path(t).write_text(text, encoding="utf-8"))


def atomic_write_json(path: Path, obj, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, default=_jsonable))


def atomic_torch_save(obj, path: Path) -> Path:
    import torch

    return atomic_save(path, lambda t: torch.save(obj, t))


def atomic_copy(src: Path, dst: Path, verify: str = "size") -> dict:
    """Copy ``src`` to ``dst`` through ``dst.partial``; the destination is replaced only after verification.

    ``verify``: ``"size"`` (cheap, default) or ``"sha256"`` (re-reads both files; use for checkpoints and
    final products). Raises ``IOError`` on a mismatch and leaves any previous ``dst`` untouched.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, part)
    try:
        shutil.copystat(src, part)
    except OSError:
        pass
    s_size, p_size = src.stat().st_size, part.stat().st_size
    info = {"src": str(src), "dst": str(dst), "bytes": s_size, "verify": verify}
    if s_size != p_size:
        part.unlink(missing_ok=True)
        raise IOError(f"size mismatch copying {src} -> {dst}: {s_size} vs {p_size}")
    if verify == "sha256":
        a, b = sha256_file(src), sha256_file(part)
        if a != b:
            part.unlink(missing_ok=True)
            raise IOError(f"checksum mismatch copying {src} -> {dst}")
        info["sha256"] = a
    os.replace(part, dst)
    return info


def _jsonable(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
