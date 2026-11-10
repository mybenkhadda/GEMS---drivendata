"""Turn ranked candidate segments into the official raster representation.

The competition submission is a continuous [0, 1] raster on the template grid (``official_adapter``), so a
segment set is rasterised: each selected segment burns its calibrated score along its trace (``all_touched``);
overlapping segments keep the maximum. Optionally the pixel model probability is kept elsewhere at a reduced
weight (``background_weight``; 0 = segments only).
"""
from __future__ import annotations

import numpy as np


def rasterize_segments(gdf, scores, grid, valid: np.ndarray, background: np.ndarray | None = None,
                       background_weight: float = 0.0) -> np.ndarray:
    from rasterio.features import rasterize

    out = np.zeros(grid.shape, np.float32)
    order = np.argsort(np.asarray(scores))                       # max wins: burn ascending
    for i in order:
        g, s = gdf.geometry.iloc[i], float(np.clip(scores[i], 0, 1))
        m = rasterize([(g, 1)], out_shape=grid.shape, transform=grid.affine, fill=0, dtype="uint8",
                      all_touched=True).astype(bool)
        out[m] = np.maximum(out[m], s)
    if background is not None and background_weight > 0:
        out = np.maximum(out, np.clip(np.nan_to_num(background) * background_weight, 0, 1))
    out[~valid] = np.nan
    return out
