"""Opt-in acquisition of additional USGS 3DEP 1 m DEM tiles (``enable_new_dem_downloads``, default False).

Selection is label-free and coverage-first: catalog tiles (snapshot copy of the TNM catalog) not yet present
as VALID / DUPLICATE tiles are ranked greedily by the number of GeoDAWN-footprint cells they newly cover (nominal
10 km tile footprint from the USGS tile name, NAD83 / UTM zone), one tile per 10 km tile id (newest project),
with explicit tile-count and GiB budgets. Only URLs on ``acquisition.allowed_host`` under ``allowed_prefix`` are
used. Downloads go straight into ``00_raw_snapshot/dem_1m/<project_dir>/`` via ``<file>.partial`` with HTTP Range
resume; a tile is accepted only after its header and a decimated data read pass, then its SHA-256, URL, UTC
timestamp and licence are appended to ``07_cache/download_state/downloads.jsonl``.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .cache import Ledger
from .checksums import sha256_file
from .reliability import raise_infrastructure


def plan_downloads(catalog: pd.DataFrame, dem_inv: pd.DataFrame, grid, footprint: np.ndarray, coverage: np.ndarray,
                   cfg) -> pd.DataFrame:
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    from .geospatial import window_of_bounds

    a = cfg["acquisition"]
    have = set(dem_inv.loc[dem_inv.status.isin(["VALID", "DUPLICATE_OVERLAP"]), "tile_id"].dropna())
    cat = catalog.copy()
    ok_url = cat.url.map(lambda u: urlparse(u).scheme == "https" and urlparse(u).hostname == a["allowed_host"]
                         and urlparse(u).path.startswith(a["allowed_prefix"]))
    cat = cat[ok_url & ~cat.tile_id.isin(have)]
    cat["year"] = cat.project.str.extract(r"((?:19|20)\d{2})", expand=False).astype(float).fillna(0)
    cat = cat.sort_values(["tile_id", "year"], ascending=[True, False]).drop_duplicates("tile_id")
    covered = coverage > 0
    cand = []
    for r in cat.itertuples():
        try:
            b = transform_bounds(CRS.from_epsg(26900 + int(r.zone)), grid.crs, r.west, r.south, r.east, r.north,
                                 densify_pts=21)
        except Exception:
            continue
        w = window_of_bounds(b, grid)
        if w is not None:
            cand.append((r, w))
    chosen, plan_cov = [], covered.copy()
    budget = float(a["max_gib"]) * 2 ** 30
    used = 0.0
    while cand and len(chosen) < int(a["max_new_tiles"]):
        best, best_n = None, 0
        for i, (r, (r0, r1, c0, c1)) in enumerate(cand):
            n = int((footprint[r0:r1, c0:c1] & ~plan_cov[r0:r1, c0:c1]).sum())
            if n > best_n:
                best, best_n = i, n
        if best is None or best_n < int(a["min_new_cells"]):
            break
        r, (r0, r1, c0, c1) = cand.pop(best)
        size = float(r.size_bytes) if pd.notna(r.size_bytes) and float(r.size_bytes) > 0 else 3.5e8
        if used + size > budget:
            break
        used += size
        plan_cov[r0:r1, c0:c1] |= footprint[r0:r1, c0:c1]
        chosen.append({"file": r.file, "url": r.url, "project_dir": r.project_dir, "tile_id": r.tile_id,
                       "new_footprint_cells": best_n, "size_estimate_bytes": size})
    return pd.DataFrame(chosen)


def _download(url: str, dst: Path, timeout: float) -> int:
    part = dst.with_name(dst.name + ".partial")
    dst.parent.mkdir(parents=True, exist_ok=True)
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        mode = "ab" if start and resp.status == 206 else "wb"
        with open(part, mode) as f:
            while True:
                b = resp.read(8 << 20)
                if not b:
                    break
                f.write(b)
            f.flush()
            os.fsync(f.fileno())
    return part.stat().st_size


def _validate(path: Path) -> str | None:
    import rasterio

    try:
        with rasterio.open(path) as s:
            if not (s.crs and s.crs.is_projected) or not (0.5 <= abs(s.transform.a) <= 3.0):
                return "header check failed (CRS / resolution)"
            a = s.read(1, out_shape=(max(s.height // 50, 1), max(s.width // 50, 1)), masked=True)
            if np.ma.count(a) == 0:
                return "no finite elevation in a decimated read"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def run(plan: pd.DataFrame, snapshot_dem: Path, state_dir: Path, cfg, progress=print) -> pd.DataFrame:
    a = cfg["acquisition"]
    led = Ledger(state_dir / "downloads.jsonl", key="url")
    prev = led.latest()
    rows = []
    for r in plan.itertuples():
        dst = snapshot_dem / r.project_dir / r.file
        if prev.get(r.url, {}).get("status") == "VERIFIED" and dst.exists():
            rows.append({**prev[r.url], "status": "VERIFIED", "detail": "already downloaded"})
            continue
        try:
            n = _download(r.url, dst, float(a["timeout_s"]))
            part = dst.with_name(dst.name + ".partial")
            err = _validate(part)
            if err:
                rows.append(led.append({"url": r.url, "file": r.file, "status": "INVALID_DOWNLOAD", "detail": err,
                                        "bytes": n}))
                continue
            sha = sha256_file(part)
            os.replace(part, dst)
            rows.append(led.append({"url": r.url, "file": r.file, "path": str(dst), "status": "VERIFIED",
                                    "sha256": sha, "bytes": n, "license": a["license"],
                                    "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            progress(f"[acquire] {r.file} {n / 2 ** 20:.0f} MiB verified")
        except Exception as exc:
            raise_infrastructure(exc, f"downloading {r.file}")
            rows.append(led.append({"url": r.url, "file": r.file, "status": "DOWNLOAD_FAILED",
                                    "detail": f"{type(exc).__name__}: {str(exc)[:200]} (resumable: re-run)"}))
    return pd.DataFrame(rows)
