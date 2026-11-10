"""Candidate-level metrics against mapped faults (incomplete reference).

* candidate (length) recall: share of mapped-fault pixels within ``tol_m`` of a selected segment;
* length precision: share of selected length within ``tol_m`` of a mapped fault;
* fragmentation: selected segments per recalled mapped fault component;
* topology: share of recalled mapped components covered by a single selected segment;
* total predicted length; curves versus length budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def _seg_mask(gdf, grid):
    from rasterio.features import rasterize

    if not len(gdf):
        return np.zeros(grid.shape, bool), np.zeros(grid.shape, np.int32)
    ids = rasterize([(g, i + 1) for i, g in enumerate(gdf.geometry)], out_shape=grid.shape, transform=grid.affine,
                    fill=0, dtype="int32", all_touched=True)
    return ids > 0, ids


def candidate_metrics(gdf, truth: np.ndarray, region: np.ndarray, grid, tol_m: float) -> dict:
    px = grid.res[0]
    m, ids = _seg_mask(gdf, grid)
    m &= region
    t = truth & region
    d_seg = ndimage.distance_transform_edt(~m, sampling=px) if m.any() else np.full(truth.shape, np.inf)
    d_tru = ndimage.distance_transform_edt(~truth, sampling=px)
    rec = float((d_seg[t] <= tol_m).mean()) if t.any() else np.nan
    prec = float((d_tru[m] <= tol_m).mean()) if m.any() else np.nan
    lab, n = ndimage.label(t, structure=np.ones((3, 3), bool))
    frag, single = [], 0
    near_ids = ndimage.grey_dilation(ids, size=2 * int(np.ceil(tol_m / px)) + 1)
    for k in range(1, n + 1):
        comp = lab == k
        if not (d_seg[comp] <= tol_m).any():
            continue
        segs = np.unique(near_ids[comp])
        segs = segs[segs > 0]
        frag.append(len(segs))
        single += int(len(segs) == 1)
    return {"length_recall": rec, "length_precision": prec, "predicted_km": float(gdf.geometry.length.sum() / 1000.0)
            if len(gdf) else 0.0, "n_segments": int(len(gdf)),
            "fragmentation": float(np.mean(frag)) if frag else np.nan,
            "topology_single_segment_share": single / max(len(frag), 1), "n_truth_components": int(n)}


def budget_curve(gdf, scores, truth, region, grid, budgets_km, tol_m) -> pd.DataFrame:
    from ..inference.ranking import select_by_length_budget

    L = gdf.geometry.length.to_numpy()
    rows = []
    for b in budgets_km:
        sel = select_by_length_budget(np.asarray(scores), L, b)
        rows.append({"budget_km": b, **candidate_metrics(gdf[sel], truth, region, grid, tol_m)})
    return pd.DataFrame(rows)
