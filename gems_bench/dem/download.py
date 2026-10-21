"""Explicit, resumable download of *selected* DEM tiles (never automatic; guarded by count and size limits).

Files land in ``data/raw/dem_1m/<project>/<file>`` on the workspace (Colab local disk), are verified (HTTP
Content-Length and a GDAL open), then pushed to the same path on Drive. Partial downloads use ``.partial``
and resume with HTTP Range requests.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import pandas as pd


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download_one(url: str, dst: Path, chunk: int = 8 << 20, retries: int = 3) -> dict:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    if dst.exists() and (total is None or dst.stat().st_size == total):
        return {"url": url, "path": str(dst), "status": "exists", "bytes": dst.stat().st_size}
    part = dst.with_name(dst.name + ".partial")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                      # server ignored Range: restart
                    f.seek(0)
                    f.truncate()
                while True:
                    b = r.read(chunk)
                    if not b:
                        break
                    f.write(b)
            if total is not None and part.stat().st_size != total:
                raise IOError(f"incomplete: {part.stat().st_size} / {total}")
            _check_raster(part)
            os.replace(part, dst)
            return {"url": url, "path": str(dst), "status": "downloaded", "bytes": dst.stat().st_size, "attempts": attempt}
        except Exception as e:
            err = str(e)
            time.sleep(min(30, 5 * attempt))
    return {"url": url, "path": str(dst), "status": "failed", "error": err}


def _check_raster(p: Path) -> None:
    import rasterio

    with rasterio.open(p) as src:
        _ = src.profile


def download_selected(sel: pd.DataFrame, root: Path, *, max_tiles: int = 30, max_gib: float = 15.0, sync=None) -> pd.DataFrame:
    """Download the pilot tiles (refuses to exceed ``max_tiles`` / ``max_gib``); pushes each tile to Drive."""
    if len(sel) > max_tiles:
        raise ValueError(f"{len(sel)} tiles selected > max_tiles={max_tiles}: reduce the pilot or raise the limit explicitly")
    est = sel.size_bytes.fillna(350e6).astype(float).sum() / 2**30 if "size_bytes" in sel else len(sel) * 0.33
    if est > max_gib:
        raise ValueError(f"estimated {est:.1f} GiB > max_gib={max_gib}")
    rows = []
    for r in sel.itertuples():
        dst = Path(root) / "data" / "raw" / "dem_1m" / r.project_dir / r.file
        res = download_one(r.url, dst)
        rows.append({**res, "tile_id": r.tile_id})
        if sync is not None and res["status"] in ("downloaded", "exists"):
            sync.push([dst.resolve().relative_to(Path(sync.local_root).resolve()).as_posix()])
        print(f"[dem] {r.tile_id}: {res['status']}")
    return pd.DataFrame(rows)
