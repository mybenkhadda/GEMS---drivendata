"""Verified raw-data snapshot into ``00_raw_snapshot/``.

Per file:
1. ledger says VERIFIED for the same (source size, mtime_ns) and the destination exists with that size -> skip
   (no re-hash);
2. destination exists without a ledger record -> hash source and destination; equal -> VERIFIED (adopted);
   different -> recopy (the existing file is replaced only after the new copy is verified);
3. otherwise copy in ``snapshot.chunk_mib`` chunks to ``<dest>.partial`` (fsync), hashing the source stream, re-read
   and hash the ``.partial``; equal -> ``os.replace`` onto the destination and append VERIFIED to the ledger;
   different -> CHECKSUM_MISMATCH (the ``.partial`` is removed, the destination untouched).
Errors: Errno 107 -> ``DriveDisconnected`` (stage stops, everything verified so far stays verified); ENOSPC /
EDQUOT -> ``DriveQuotaExceeded`` recorded as QUOTA_EXCEEDED and the stage stops with the incomplete-copy report;
a source that cannot be read is SOURCE_INACCESSIBLE (never "missing" / "invalid"). Originals are opened read-only.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from .cache import Ledger, atomic_write_csv, atomic_write_json
from .checksums import HashCache
from .reliability import DriveDisconnected, DriveQuotaExceeded, classify, is_disconnect, is_quota, is_transient

REQUIRED_ROLES = ("geodawn_features", "fault_labels", "official_template")


def capacity_report(plan: pd.DataFrame, ledger: dict, atlas_root: Path, margin: float) -> dict:
    todo = plan[~plan.source_path.isin({k for k, v in ledger.items() if v.get("status") == "VERIFIED"})]
    need = int(todo.size_bytes.fillna(0).sum())
    total = int(plan.size_bytes.fillna(0).sum())
    try:
        du = shutil.disk_usage(atlas_root)
        free, reported = int(du.free), True
    except OSError:
        free, reported = None, False
    return {"sources": int(len(plan)), "total_source_bytes": total, "already_verified_bytes": total - need,
            "bytes_to_copy": need, "duplicate_storage_bytes": total,
            "destination_free_bytes": free, "free_space_reported": reported,
            "free_space_note": "shutil.disk_usage on the Drive FUSE mount; Google Drive quota can differ",
            "fits": None if free is None else bool(free >= need * margin)}


def _copy_verify(src: Path, dst: Path, chunk: int) -> tuple[str, str]:
    part = dst.with_name(dst.name + ".partial")
    dst.parent.mkdir(parents=True, exist_ok=True)
    hs = hashlib.sha256()
    with open(src, "rb") as fi, open(part, "wb") as fo:
        for b in iter(lambda: fi.read(chunk), b""):
            hs.update(b)
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())
    hd = hashlib.sha256()
    with open(part, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            hd.update(b)
    return hs.hexdigest(), hd.hexdigest()


def snapshot(sources: pd.DataFrame, atlas_root: Path, cfg, verify: bool = True, progress=print) -> dict:
    """Copy + verify every accessible source. Returns the report (also written to 01_inventory/manifests)."""
    sc = cfg["snapshot"]
    chunk = int(sc["chunk_mib"]) << 20
    led = Ledger(atlas_root / "07_cache" / "copy_state" / "copy_ledger.jsonl", key="source_path")
    hc = HashCache(atlas_root / "07_cache" / "copy_state" / "hash_cache.jsonl")
    plan = sources[sources.snapshot_rel.notna() & sources.source_path.notna()].copy()
    latest = led.latest()
    cap = capacity_report(plan[plan.access_status == "OK"], latest, atlas_root, sc["capacity_margin"])
    results, stop_reason = [], None
    t0 = time.time()
    for i, r in enumerate(plan.itertuples(), start=1):
        src, dst = Path(r.source_path), atlas_root / r.snapshot_rel
        base = {"source_path": str(src), "snapshot_path": str(dst), "role": r.role, "source_bytes": r.size_bytes,
                "source_mtime_ns": r.mtime_ns}
        if stop_reason:
            results.append({**base, "status": "NOT_ATTEMPTED", "detail": stop_reason})
            continue
        if r.access_status != "OK":
            results.append(led.append({**base, "status": "SOURCE_INACCESSIBLE", "detail": r.access_status}))
            continue
        prev = latest.get(str(src))
        if (prev and prev.get("status") == "VERIFIED" and prev.get("source_bytes") == r.size_bytes
                and prev.get("source_mtime_ns") == r.mtime_ns and dst.exists() and dst.stat().st_size == r.size_bytes):
            results.append({**prev, "status": "VERIFIED", "detail": "reused verified copy (ledger)"})
            continue
        try:
            if dst.exists() and dst.stat().st_size == r.size_bytes and verify:
                s_sha, d_sha = hc.sha256(src), hc.sha256(dst)
                if s_sha == d_sha:
                    results.append(led.append({**base, "status": "VERIFIED", "sha256": s_sha,
                                               "detail": "existing destination verified by checksum"}))
                    continue
            attempt = 0
            while True:
                try:
                    s_sha, d_sha = _copy_verify(src, dst, chunk)
                    break
                except OSError as exc:
                    if is_transient(exc) and attempt < int(sc["retries"]) - 1:
                        attempt += 1
                        time.sleep(float(sc["retry_wait_s"]) * attempt)
                        continue
                    raise
            part = dst.with_name(dst.name + ".partial")
            if s_sha != d_sha:
                part.unlink(missing_ok=True)
                results.append(led.append({**base, "status": "CHECKSUM_MISMATCH", "detail": f"{s_sha} != {d_sha}"}))
                continue
            os.replace(part, dst)
            hc.put(dst, d_sha)
            hc.put(src, s_sha)
            results.append(led.append({**base, "status": "VERIFIED", "sha256": s_sha, "detail": "copied + verified"}))
        except Exception as exc:
            if is_disconnect(exc):
                led.append({**base, "status": "INTERRUPTED_DRIVE_DISCONNECTED", "detail": str(exc)[:200]})
                _write_report(atlas_root, plan, results, cap, "DRIVE_DISCONNECTED", t0)
                raise DriveDisconnected(f"snapshot interrupted at {src.name}: {exc}") from exc
            if is_quota(exc):
                results.append(led.append({**base, "status": "QUOTA_EXCEEDED", "detail": str(exc)[:200]}))
                stop_reason = "stopped after QUOTA_EXCEEDED"
                continue
            results.append(led.append({**base, "status": f"COPY_FAILED:{classify(exc)}", "detail": str(exc)[:200]}))
        if i % 25 == 0:
            progress(f"[snapshot] {i}/{len(plan)} files, {time.time() - t0:.0f}s")
    rep = _write_report(atlas_root, plan, results, cap, stop_reason or "DONE", t0)
    if stop_reason:
        raise DriveQuotaExceeded(f"snapshot incomplete ({stop_reason}); see 01_inventory/manifests/snapshot_report.json")
    return rep


def _write_report(atlas_root: Path, plan, results, cap, end_state, t0) -> dict:
    res = pd.DataFrame(results)
    inv = atlas_root / "01_inventory"
    atomic_write_csv(inv / "manifests" / "snapshot_status.csv", res)
    ver = res[res.status == "VERIFIED"] if len(res) else res
    if len(ver):
        atomic_write_csv(inv / "source_checksums.csv",
                         ver[["role", "source_path", "snapshot_path", "source_bytes", "sha256"]].rename(
                             columns={"source_bytes": "size_bytes"}))
    req_missing = [r for r in REQUIRED_ROLES if not ((res.role == r) & (res.status == "VERIFIED")).any()] if len(res) \
        else list(REQUIRED_ROLES)
    dem = res[res.role == "dem_tile"] if len(res) else res
    rep = {"end_state": end_state, "seconds": round(time.time() - t0, 1), "capacity": cap,
           "counts": res.status.value_counts().to_dict() if len(res) else {},
           "required_missing_or_unverified": req_missing,
           "dem_tiles_planned": int((plan.role == "dem_tile").sum()),
           "dem_tiles_verified": int((dem.status == "VERIFIED").sum()) if len(dem) else 0,
           "complete": end_state == "DONE" and not req_missing and bool(len(res)) and bool((res.status == "VERIFIED").all())}
    atomic_write_json(inv / "manifests" / "snapshot_report.json", rep)
    return rep


def snapshot_state(atlas_root: Path) -> dict:
    from .cache import read_json

    return read_json(atlas_root / "01_inventory" / "manifests" / "snapshot_report.json", {}) or {}
