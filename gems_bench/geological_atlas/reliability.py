"""Google Drive reliability: health check, disconnect detection, bounded retries.

``OSError: [Errno 107] Transport endpoint is not connected`` means the Drive FUSE mount died. It is an
infrastructure failure, never evidence that a raster is corrupt or missing: it is converted to
``DriveDisconnected`` and stops the stage (completed artifacts stay valid; re-run after remounting).
Quota errors (ENOSPC 28, EDQUOT 122) become ``DriveQuotaExceeded``. Only short transient errors (EIO 5,
ETIMEDOUT 110, EAGAIN 11, connection resets) are retried, a bounded number of times.
"""
from __future__ import annotations

import errno
import os
import time
import uuid
from pathlib import Path

REMOUNT_HELP = (
    "Google Drive is disconnected (Errno 107). Completed artifacts are safe.\n"
    "  1. Run:  from google.colab import drive; drive.flush_and_unmount()\n"
    "  2. Run:  drive.mount('/content/drive', force_remount=True)\n"
    "  3. If it fails again: Runtime > Restart session, then re-run sections 00-01 and the stage you were in.\n"
    "Nothing is re-copied or re-hashed: verified files are recorded in the ledgers."
)
TRANSIENT = {errno.EIO, errno.EAGAIN, getattr(errno, "ETIMEDOUT", 110), getattr(errno, "ECONNRESET", 104)}
QUOTA = {errno.ENOSPC, getattr(errno, "EDQUOT", 122)}
DISCONNECT = {getattr(errno, "ENOTCONN", 107)}


class DriveDisconnected(RuntimeError):
    pass


class DriveQuotaExceeded(RuntimeError):
    pass


def is_disconnect(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and (exc.errno in DISCONNECT or "Transport endpoint is not connected" in str(exc))


def is_quota(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno in QUOTA


def is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError)) or (isinstance(exc, OSError) and exc.errno in TRANSIENT)


def classify(exc: BaseException) -> str:
    if is_disconnect(exc):
        return "DRIVE_DISCONNECTED"
    if is_quota(exc):
        return "QUOTA_EXCEEDED"
    if is_transient(exc):
        return "TRANSIENT_IO"
    if isinstance(exc, FileNotFoundError):
        return "NOT_FOUND"
    if isinstance(exc, PermissionError):
        return "PERMISSION_DENIED"
    return "IO_ERROR" if isinstance(exc, OSError) else type(exc).__name__


def raise_infrastructure(exc: BaseException, where: str = ""):
    """Re-raise Drive disconnects / quota errors as their dedicated exceptions; otherwise do nothing."""
    if is_disconnect(exc):
        raise DriveDisconnected(f"{where}: {exc}\n{REMOUNT_HELP}") from exc
    if is_quota(exc):
        raise DriveQuotaExceeded(f"{where}: Drive quota / disk full ({exc}). Free space or raise the quota; "
                                 "verified files are kept.") from exc


def retry(fn, *args, attempts: int = 3, wait_s: float = 5.0, where: str = "", **kw):
    """Call ``fn`` retrying only transient errors; disconnect / quota errors raise immediately."""
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kw)
        except Exception as exc:  # classified below, never swallowed
            raise_infrastructure(exc, where)
            if not is_transient(exc) or i == attempts - 1:
                raise
            last = exc
            time.sleep(wait_s * (i + 1))
    raise last  # pragma: no cover


def drive_health_check(drive_root: Path, probe_dir: Path | None = None) -> dict:
    """stat + listdir of the Drive root, then a write / fsync / read / delete probe. Raises on failure."""
    t0 = time.time()
    root = Path(drive_root)
    try:
        if not root.exists():
            return {"ok": False, "reason": f"{root} does not exist - is Drive mounted?", "seconds": 0.0}
        os.listdir(root)
        if probe_dir is not None:
            probe_dir.mkdir(parents=True, exist_ok=True)
            p = probe_dir / f".health-{uuid.uuid4().hex}"
            payload = os.urandom(4096)
            with open(p, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            ok = p.read_bytes() == payload
            p.unlink()
            if not ok:
                return {"ok": False, "reason": "probe read-back differs", "seconds": time.time() - t0}
    except Exception as exc:
        raise_infrastructure(exc, "drive health check")
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "seconds": time.time() - t0}
    return {"ok": True, "reason": "stat, list, write, fsync, read, delete ok", "seconds": round(time.time() - t0, 2)}
