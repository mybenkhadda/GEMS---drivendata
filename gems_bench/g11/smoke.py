"""SMOKE_TEST subset: a square window of the master grid with BOTH mapped faults and DEM coverage.

The window is chosen from metadata only (fault label raster + DEM tile header footprints; no DEM pixels are read):
the ``window_km`` square maximising (fault pixels with DEM coverage) + 0.1 x (DEM-covered cells), searched on a
half-window stride. The chosen window is saved and reused so a restart selects the same subset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def dem_header_coverage(dem_files, grid) -> np.ndarray:
    """Union of DEM tile footprints (header bounds, reprojected) rasterised on ``grid``."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    from shapely.geometry import box, mapping, shape

    geoms = []
    for p in dem_files:
        try:
            with rasterio.open(p) as ds:
                if ds.crs is None:
                    continue
                geoms.append(shape(transform_geom(ds.crs, grid.crs, mapping(box(*ds.bounds)))))
        except Exception:                                          # noqa: BLE001 - unreadable tiles are skipped here
            continue
    if not geoms:
        return np.zeros(grid.shape, bool)
    return rasterize([(g, 1) for g in geoms], out_shape=grid.shape, transform=grid.affine, fill=0,
                     dtype="uint8").astype(bool)


def choose_window(fault: np.ndarray, dem_cov: np.ndarray, valid: np.ndarray, px_m: float, window_km: float):
    n = max(int(round(window_km * 1000 / px_m)), 16)
    h, w = fault.shape
    n = min(n, h, w)
    score_px = (fault & dem_cov & valid).astype(np.float64) + 0.1 * (dem_cov & valid)
    S = np.pad(score_px.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, arg = -1.0, (0, n, 0, n)
    step = max(n // 2, 1)
    for r in range(0, h - n + 1, step):
        for c in range(0, w - n + 1, step):
            s = S[r + n, c + n] - S[r, c + n] - S[r + n, c] + S[r, c]
            if s > best:
                best, arg = s, (r, r + n, c, c + n)
    r0, r1, c0, c1 = arg
    return {"window": [int(r0), int(r1), int(c0), int(c1)], "window_km": n * px_m / 1000, "score": float(best),
            "fault_px_with_dem": int((fault & dem_cov & valid)[r0:r1, c0:c1].sum()),
            "fault_px": int(fault[r0:r1, c0:c1].sum()), "dem_px": int((dem_cov & valid)[r0:r1, c0:c1].sum())}


def select_smoke_window(fault_path, dem_files, full_grid, window_km: float, out_json) -> dict:
    import json

    from .fault_labels import label_raster

    p = Path(out_json)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    fault = label_raster(fault_path, full_grid)
    cov = dem_header_coverage(dem_files, full_grid)
    rec = choose_window(fault, cov, full_grid.valid, full_grid.res[0], window_km)
    if rec["fault_px_with_dem"] == 0:
        rec["warning"] = "no fault pixel with DEM coverage found: the smoke window has faults or DEM, not both"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    tmp.replace(p)
    return rec
