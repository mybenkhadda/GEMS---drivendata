"""Coverage EDA: DEM vs geophysics vs mapped faults vs splits (lengths in km, areas in km^2)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_summary(valid: np.ndarray, dem_covered: np.ndarray, fault: np.ndarray, segments, dem_index, grid,
                     split_code: np.ndarray | None = None) -> pd.DataFrame:
    px_km2 = grid.res[0] * grid.res[1] / 1e6
    rows = [("geophysics_area_km2", float(valid.sum() * px_km2)),
            ("dem_area_km2", float((dem_covered & valid).sum() * px_km2)),
            ("dem_fraction_of_geophysics_area", float((dem_covered & valid).sum() / max(valid.sum(), 1))),
            ("mapped_fault_pixels", int(fault.sum())),
            ("mapped_fault_pixels_with_dem", int((fault & dem_covered).sum())),
            ("fraction_fault_pixels_with_dem", float((fault & dem_covered).sum() / max(fault.sum(), 1)))]
    if segments is not None and len(segments) and dem_index is not None and dem_index.tree is not None:
        from shapely.ops import unary_union

        cover = unary_union(list(dem_index.cat.geometry))
        L = segments.geometry.length
        inside = segments.geometry.intersection(cover).length
        rows += [("fault_length_km", float(L.sum() / 1000)), ("fault_length_with_dem_km", float(inside.sum() / 1000)),
                 ("fraction_fault_length_with_dem", float(inside.sum() / max(L.sum(), 1e-9))),
                 ("segments_with_usable_dem_(>=80%_length)", int((inside >= 0.8 * L).sum())),
                 ("segments_total", int(len(segments)))]
    if split_code is not None:
        from ..spatial_splits import CODES

        for k, v in CODES.items():
            if v:
                m = split_code == v
                rows += [(f"{k}_area_km2", float(m.sum() * px_km2)), (f"{k}_dem_fraction", float((m & dem_covered).sum()
                                                                                              / max(m.sum(), 1)))]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def tiles_per_region(catalog, regions: np.ndarray, grid) -> pd.DataFrame:
    from ..grid import inside, world_to_pixel

    if not len(catalog):
        return pd.DataFrame(columns=["region", "n_tiles"])
    cx = catalog.geometry.centroid
    r, c = world_to_pixel(cx.x.to_numpy(), cx.y.to_numpy(), grid)
    ok = inside(r, c, grid)
    reg = np.full(len(catalog), -1)
    reg[ok] = regions[r[ok], c[ok]]
    return pd.Series(reg).value_counts().rename_axis("region").reset_index(name="n_tiles")
