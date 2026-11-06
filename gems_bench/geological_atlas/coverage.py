"""DEM coverage on the 100 m grid measured from real raster values (not file rectangles).

Per VALID tile a decimated read (GDAL ``average`` to ~10 m, nodata excluded) gives the valid-pixel mask; it is
reprojected to the analysis grid with ``average`` resampling (area-weighted share of valid DEM pixels per 100 m
cell). Outputs on the grid: coverage (max over tiles, i.e. union share), overlap count (tiles with share > 0),
preferred-project id (newest year wins per cell), seam mask (cells where two different projects meet), and the
multimodal coverage bitmask (1 GeoDAWN footprint, 2 label-raster footprint, 4 DEM coverage > 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reliability import raise_infrastructure


def tile_coverage(path: str, grid, decim_m: float = 10.0) -> tuple[tuple, np.ndarray] | None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from .geospatial import window_of_bounds

    with rasterio.open(path) as s:
        f = max(int(round(decim_m / abs(s.transform.a))), 1)
        h, w = max(s.height // f, 1), max(s.width // f, 1)
        a = s.read(1, out_shape=(h, w), masked=True, resampling=Resampling.average)
        valid = (~np.ma.getmaskarray(a)).astype(np.float32)
        tf = s.transform * s.transform.scale(s.width / w, s.height / h)
        from rasterio.warp import transform_bounds

        b = transform_bounds(s.crs, grid.crs, *s.bounds, densify_pts=21)
        win = window_of_bounds(b, grid, pad=1)
        if win is None:
            return None
        r0, r1, c0, c1 = win
        from affine import Affine

        g = grid.transform
        dst_tf = Affine(g[0], 0, g[2] + g[0] * c0, 0, g[4], g[5] + g[4] * r0)
        out = np.zeros((r1 - r0, c1 - c0), np.float32)
        reproject(valid, out, src_transform=tf, src_crs=s.crs, dst_transform=dst_tf, dst_crs=grid.crs,
                  resampling=Resampling.average, src_nodata=None, dst_nodata=None, init_dest_nodata=False)
        # cells partially outside the tile get share < 1 automatically (average of the covered fraction);
        # cells entirely outside stay 0
    return win, np.clip(out, 0.0, 1.0)


def build_coverage(dem_inv: pd.DataFrame, grid, footprint: np.ndarray, label_foot: np.ndarray, progress=print) -> dict:
    H, W = grid.height, grid.width
    cov = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.int16)
    proj = np.full((H, W), -1, np.int32)
    proj_year = np.full((H, W), -1, np.int32)
    ok = dem_inv[dem_inv.status == "VALID"].reset_index(drop=True)
    pid = {p: i for i, p in enumerate(sorted(ok.project.unique()))} if len(ok) else {}
    rows = []
    for i, r in enumerate(ok.itertuples(), start=1):
        try:
            res = tile_coverage(r.snapshot_path, grid)
        except Exception as exc:
            raise_infrastructure(exc, f"coverage {r.file}")
            rows.append({"file": r.file, "covered_cells": 0, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if res is None:
            continue
        (r0, r1, c0, c1), share = res
        sl = (slice(r0, r1), slice(c0, c1))
        cov[sl] = np.maximum(cov[sl], share)
        count[sl] += (share > 0).astype(np.int16)
        better = (share > 0) & (r.year > proj_year[sl])
        proj[sl] = np.where(better, pid[r.project], proj[sl])
        proj_year[sl] = np.where(better, r.year, proj_year[sl])
        rows.append({"file": r.file, "project": r.project, "covered_cells": int((share > 0).sum()),
                     "covered_km2": float(share.sum() * grid.pixel_m ** 2 / 1e6)})
        if i % 50 == 0:
            progress(f"[coverage] {i}/{len(ok)} tiles")
    seam = np.zeros((H, W), bool)
    for ax in (0, 1):
        p1 = np.roll(proj, -1, axis=ax)
        s = (proj >= 0) & (p1 >= 0) & (proj != p1)
        if ax == 0:
            s[-1] = False
        else:
            s[:, -1] = False
        seam |= s | np.roll(s, 1, axis=ax)
    multi = (footprint.astype(np.uint8) * 1) | (label_foot.astype(np.uint8) * 2) | ((cov > 0).astype(np.uint8) * 4)
    stats = {"tiles_valid": int(len(ok)), "projects": len(pid),
             "covered_cells_in_footprint": int(((cov > 0) & footprint).sum()),
             "covered_share_of_footprint": float(((cov > 0) & footprint).sum() / max(footprint.sum(), 1)),
             "area_weighted_coverage_km2": float((cov * footprint).sum() * grid.pixel_m ** 2 / 1e6),
             "overlap_cells": int((count > 1).sum()), "seam_cells": int(seam.sum())}
    return {"coverage": cov, "count": count, "project": proj, "seam": seam, "multimodal": multi,
            "project_ids": pid, "tile_table": pd.DataFrame(rows), "stats": stats}
