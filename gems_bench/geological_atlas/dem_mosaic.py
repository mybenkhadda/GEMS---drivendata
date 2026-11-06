"""DEM source preference, per-tile work-resolution caches and windowed mosaics on the analysis-aligned work grid.

Work grid: same origin as the 100 m analysis grid, pixel = ``work_res_m`` (10 m FAST, 2 m DETAILED), so every
100 m cell is exactly (100 / res)^2 work pixels (exact area-weighted aggregation).

Source preference (deterministic): VALID tiles first, then DUPLICATE_OVERLAP tiles (they only fill pixels the
preferred tiles leave empty); within each group newer project year, then project name, then file name.

FAST_10M: each tile is warped once onto the 10 m work grid with GDAL ``average`` (1 m -> 10 m block mean over
valid source pixels, vertical unit converted to metres) and cached in ``07_cache/dem_windows/FAST_10M/tiles``.
A window is assembled from those caches by integer pasting (first source wins). DETAILED_2M warps the 1 m tiles
straight onto the window's 2 m grid (``average``). No full-region DEM array is ever held in memory.
Seams: work pixels whose 4-neighbour comes from another DEM project.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .cache import atomic_write_geotiff, is_partial
from .inventory import UNIT_FACTORS
from .reliability import raise_infrastructure

ELEV_RANGE_M = (-500.0, 9000.0)
DEFAULT_NODATA = -999999.0


def preference(dem_inv: pd.DataFrame) -> pd.DataFrame:
    use = dem_inv[dem_inv.status.isin(["VALID", "DUPLICATE_OVERLAP"])].copy()
    use["group"] = (use.status != "VALID").astype(int)
    use = use.sort_values(["group", "year", "project", "file"], ascending=[True, False, True, True])
    use = use.reset_index(drop=True)
    use["priority"] = np.arange(len(use))
    use["project_id"] = pd.factorize(use.project)[0]
    return use


def work_transform(grid, res: float, r_cell: int, c_cell: int, halo_px: int):
    from affine import Affine

    a, _, c, _, e, f = grid.transform[:6]
    return Affine(res, 0.0, c + a * c_cell - halo_px * res, 0.0, -res, f + e * r_cell + halo_px * res)


def _warp(path: str, dst_crs, tf, w: int, h: int, unit: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(path) as src:
        nod = src.nodata if src.nodata is not None else DEFAULT_NODATA
        with WarpedVRT(src, crs=dst_crs, transform=tf, width=w, height=h, resampling=Resampling.average,
                       src_nodata=nod, nodata=nod) as vrt:
            a = vrt.read(1, out_dtype="float32", masked=True)
    z = np.ma.filled(a.astype(np.float32), np.nan) * np.float32(UNIT_FACTORS[unit])
    z[(z < ELEV_RANGE_M[0]) | (z > ELEV_RANGE_M[1])] = np.nan
    return z


def tile_cache_path(cache_dir: Path, tile_file: str) -> Path:
    return Path(cache_dir) / f"{Path(tile_file).stem}_10m.tif"


def build_tile_cache(rec: dict, grid, cache_dir: Path) -> dict:
    """FAST_10M: warp one tile onto the 10 m work grid (aligned), save it, return its placement."""
    out = tile_cache_path(cache_dir, rec["file"])
    res = 10.0
    a, _, c, _, e, f = grid.transform[:6]
    l, b, r, t = rec["left"], rec["bottom"], rec["right"], rec["top"]
    c0 = math.floor((l - c) / res)
    c1 = math.ceil((r - c) / res)
    r0 = math.floor((f - t) / res)
    r1 = math.ceil((f - b) / res)
    from affine import Affine

    tf = Affine(res, 0, c + c0 * res, 0, -res, f - r0 * res)
    if not out.exists():
        try:
            z = _warp(rec["snapshot_path"], grid.crs, tf, c1 - c0, r1 - r0, rec["vertical_unit"])
        except Exception as exc:
            raise_infrastructure(exc, f"warping {rec['file']}")
            raise
        atomic_write_geotiff(out, z, {"crs": grid.crs, "transform": tf}, ["elevation_m"],
                             tags={"source": rec["file"], "vertical_unit_source": rec["vertical_unit_source"],
                                   "resampling": "GDAL average 1 m -> 10 m"})
    return {"file": rec["file"], "cache": str(out), "row0": int(r0), "col0": int(c0), "rows": int(r1 - r0),
            "cols": int(c1 - c0), "priority": int(rec["priority"]), "project_id": int(rec["project_id"])}


def window_plan(grid, footprint: np.ndarray, coverage: np.ndarray, cells: int) -> pd.DataFrame:
    rows = []
    for i in range(math.ceil(grid.height / cells)):
        for j in range(math.ceil(grid.width / cells)):
            r0, c0 = i * cells, j * cells
            nr, nc = min(cells, grid.height - r0), min(cells, grid.width - c0)
            sl = (slice(r0, r0 + nr), slice(c0, c0 + nc))
            if (footprint[sl] & (coverage[sl] > 0)).any():
                rows.append({"window_id": f"w{i:03d}_{j:03d}", "row0": r0, "col0": c0, "n_rows": nr, "n_cols": nc,
                             "coverage_mean": float(coverage[sl].mean())})
    return pd.DataFrame(rows)


def read_window_fast(win: dict, placements: list[dict], halo_m: float, res: float = 10.0) -> dict:
    import rasterio

    f = int(round(100.0 / res))
    hp = int(math.ceil(halo_m / res))
    R0 = win["row0"] * f - hp
    C0 = win["col0"] * f - hp
    h, w = win["n_rows"] * f + 2 * hp, win["n_cols"] * f + 2 * hp
    z = np.full((h, w), np.nan, np.float32)
    proj = np.full((h, w), -1, np.int32)
    used = []
    for p in sorted(placements, key=lambda q: q["priority"]):
        rs, cs = max(R0, p["row0"]), max(C0, p["col0"])
        re_, ce = min(R0 + h, p["row0"] + p["rows"]), min(C0 + w, p["col0"] + p["cols"])
        if re_ <= rs or ce <= cs or is_partial(Path(p["cache"])):
            continue
        from rasterio.windows import Window

        with rasterio.open(p["cache"]) as s:
            a = s.read(1, window=Window(cs - p["col0"], rs - p["row0"], ce - cs, re_ - rs)).astype(np.float32)
        dst = (slice(rs - R0, re_ - R0), slice(cs - C0, ce - C0))
        fill = np.isnan(z[dst]) & np.isfinite(a)
        if fill.any():
            z[dst][fill] = a[fill]
            proj[dst][fill] = p["project_id"]
            used.append(p["file"])
    return {"z": z, "project": proj, "halo_px": hp, "factor": f, "res": res, "tiles_used": used}


def read_window_detailed(win: dict, sources: list[dict], grid, halo_m: float, res: float = 2.0) -> dict:
    f = int(round(100.0 / res))
    hp = int(math.ceil(halo_m / res))
    h, w = win["n_rows"] * f + 2 * hp, win["n_cols"] * f + 2 * hp
    tf = work_transform(grid, res, win["row0"], win["col0"], hp)
    x0, y1 = tf.c, tf.f
    x1, y0 = x0 + w * res, y1 - h * res
    z = np.full((h, w), np.nan, np.float32)
    proj = np.full((h, w), -1, np.int32)
    used = []
    for s in sorted(sources, key=lambda q: q["priority"]):
        if not (s["right"] > x0 and s["left"] < x1 and s["top"] > y0 and s["bottom"] < y1):
            continue
        a = _warp(s["snapshot_path"], grid.crs, tf, w, h, s["vertical_unit"])
        fill = np.isnan(z) & np.isfinite(a)
        if fill.any():
            z[fill] = a[fill]
            proj[fill] = s["project_id"]
            used.append(s["file"])
        if not np.isnan(z).any():
            break
    return {"z": z, "project": proj, "halo_px": hp, "factor": f, "res": res, "tiles_used": used, "transform": tf}


def seams_and_distances(z: np.ndarray, proj: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seam = np.zeros(z.shape, bool)
    for ax in (0, 1):
        p1 = np.roll(proj, -1, axis=ax)
        s = (proj >= 0) & (p1 >= 0) & (proj != p1)
        if ax == 0:
            s[-1] = False
        else:
            s[:, -1] = False
        seam |= s | np.roll(s, 1, axis=ax)
    miss = ~np.isfinite(z)
    d_miss = (ndimage.distance_transform_edt(~miss) * res).astype(np.float32) if miss.any() else \
        np.full(z.shape, np.inf, np.float32)
    d_seam = (ndimage.distance_transform_edt(~seam) * res).astype(np.float32) if seam.any() else \
        np.full(z.shape, np.inf, np.float32)
    return seam, d_miss, d_seam
