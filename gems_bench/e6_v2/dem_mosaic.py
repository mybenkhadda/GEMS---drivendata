"""Windowed DEM mosaic on the competition CRS at the 2 m working resolution.

The 100 m grid is cut into windows of ``dem.window_cells`` x ``dem.window_cells`` cells. A window is read with a
halo of ``dem.halo_m`` (>= the largest 2 m operator support), so features in its core never see an artificial
window edge. For every intersecting tile, in the deterministic overlap order of ``dem_sources.overlap_order``,
GDAL warps the 1 m source onto the window's 2 m grid with ``average`` resampling (this IS the 1 m -> 2 m
averaging; nodata source pixels are excluded from each average), converts the vertical unit to metres and fills
only pixels still empty (first tile wins). Pixels whose 4-neighbour comes from a different DEM project form the
seam set; features near seams are excluded (``dem.seam_guard_m``).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage

from .dem_sources import UNIT_FACTORS

ELEV_RANGE_M = (-500.0, 9000.0)
SOURCE_NODATA_DEFAULT = -999999.0


def plan_windows(grid_dict: dict, valid: np.ndarray, tiles: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per window that intersects the data footprint and at least one usable DEM tile."""
    wc, halo = int(cfg["dem"]["window_cells"]), float(cfg["dem"]["halo_m"])
    a, _, c, _, e, f = grid_dict["transform"]
    H, W = grid_dict["height"], grid_dict["width"]
    tl, tb, tr, tt = (tiles[k].to_numpy(float) for k in ("left", "bottom", "right", "top"))
    rows = []
    for i in range(math.ceil(H / wc)):
        for j in range(math.ceil(W / wc)):
            r0, c0 = i * wc, j * wc
            nr, nc = min(wc, H - r0), min(wc, W - c0)
            if not valid[r0:r0 + nr, c0:c0 + nc].any():
                continue
            x0, x1 = c + a * c0 - halo, c + a * (c0 + nc) + halo
            y1, y0 = f + e * r0 + halo, f + e * (r0 + nr) - halo
            hit = np.flatnonzero((tr > x0) & (tl < x1) & (tt > y0) & (tb < y1))
            if len(hit) == 0:
                continue
            pri = tiles.priority.to_numpy()[hit]
            rows.append({"window_id": f"w{i:03d}_{j:03d}", "row0": r0, "col0": c0, "n_rows": nr, "n_cols": nc,
                         "n_tiles": len(hit), "tile_priorities": "|".join(str(int(p)) for p in np.sort(pri))})
    return pd.DataFrame(rows)


def window_transform(grid_dict: dict, win: dict, res: float, halo_px: int):
    from affine import Affine

    a, _, c, _, e, f = grid_dict["transform"]
    x0 = c + a * win["col0"] - halo_px * res
    y0 = f + e * win["row0"] + halo_px * res
    return Affine(res, 0.0, x0, 0.0, -res, y0)


def read_window(win: dict, tiles: list[dict], grid_dict: dict, crs_wkt: str, cfg: dict) -> dict:
    """Mosaic for one window: z (m, NaN = no data), source priority, project id, transform, halo."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    d = cfg["dem"]
    res = float(d["work_res_m"])
    f = int(round(grid_dict["pixel_m"] / res))
    halo_px = int(math.ceil(float(d["halo_m"]) / res))
    h2, w2 = win["n_rows"] * f + 2 * halo_px, win["n_cols"] * f + 2 * halo_px
    tf = window_transform(grid_dict, win, res, halo_px)
    dst_crs = CRS.from_wkt(crs_wkt)
    z = np.full((h2, w2), np.nan, np.float32)
    src_pri = np.full((h2, w2), -1, np.int32)
    proj = np.full((h2, w2), -1, np.int32)
    used = []
    for t in sorted(tiles, key=lambda t: t["priority"]):
        with rasterio.open(t["path"]) as src:
            nod = src.nodata if src.nodata is not None else SOURCE_NODATA_DEFAULT
            with WarpedVRT(src, crs=dst_crs, transform=tf, width=w2, height=h2, resampling=Resampling.average,
                           src_nodata=nod, nodata=nod) as vrt:
                arr = vrt.read(1, out_dtype="float32", masked=True)
        a = np.ma.filled(arr.astype(np.float32), np.nan)
        a *= UNIT_FACTORS[t["vertical_unit"]]
        a[(a < ELEV_RANGE_M[0]) | (a > ELEV_RANGE_M[1])] = np.nan
        fill = np.isnan(z) & np.isfinite(a)
        n = int(fill.sum())
        if n:
            z[fill] = a[fill]
            src_pri[fill] = int(t["priority"])
            proj[fill] = int(t["project_id"])
            used.append({"name": t["name"], "priority": int(t["priority"]), "px_filled": n})
        if not np.isnan(z).any():
            break
    seam = np.zeros_like(z, bool)
    for ax in (0, 1):
        p0 = proj
        p1 = np.roll(proj, -1, axis=ax)
        s = (p0 >= 0) & (p1 >= 0) & (p0 != p1)
        if ax == 0:
            s[-1, :] = False
        else:
            s[:, -1] = False
        seam |= s | np.roll(s, 1, axis=ax)
    return {"z": z, "src_priority": src_pri, "project": proj, "seam": seam, "transform": tf, "halo_px": halo_px,
            "factor": f, "res": res, "tiles_used": used}


def distance_fields(z: np.ndarray, seam: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """(distance to missing data, distance to a project seam) in metres; inf where there is none."""
    miss = ~np.isfinite(z)
    d_miss = ndimage.distance_transform_edt(~miss) * res if miss.any() else np.full(z.shape, np.inf, np.float32)
    d_seam = ndimage.distance_transform_edt(~seam) * res if seam.any() else np.full(z.shape, np.inf, np.float32)
    return d_miss.astype(np.float32), d_seam.astype(np.float32)
