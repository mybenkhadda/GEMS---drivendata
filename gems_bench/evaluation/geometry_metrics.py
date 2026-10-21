"""Experiment C scientific metrics at a matched area budget (one binary top-a mask per fold, as the scorecard).

Per model:
* weighted recall (exact TP support of a binary mask, k = max(0, 1 - d/300 m)) on trace strata from
  ``geology.skeleton.trace_strata``: < 1 km, 1-2 km, 2-5 km, >= 5 km, endpoints, intersections, interior;
* long-fault continuity: for skeleton components >= ``long_km``, covered fraction and the longest covered
  connected run as a fraction of the component (a fragmented detection scores low even at equal recall);
* false-positive footprint: mask area > ``far_km`` from any mapped fault (km2, share, number of components);
* orientation error: axial difference (deg) between the orientation head and the label strike where the
  label direction is defined (only for models that exported ``orient`` rasters);
* distance to withheld traces: distribution of withheld-pixel distances to the discovery mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .. import phase4_eval
from ..geology.skeleton import skeletonize, trace_strata


def _support(d_m: np.ndarray, radius_m: float = 300.0) -> np.ndarray:
    return np.clip(1 - d_m / radius_m, 0, 1)


def _fold_distance(ctx, mask: np.ndarray) -> np.ndarray:
    """Distance (m) from every pixel to the mask of its own fold zone (inf where the fold has no mask)."""
    g = ctx.data.grid
    px = (g.pixel_km_y * 1000, g.pixel_km_x * 1000)
    out = np.full(mask.shape, np.inf, np.float32)
    for k in ctx.fold_ids:
        z = ctx.folds.fold_map == k
        mk = mask & z
        if mk.any():
            out[z] = ndimage.distance_transform_edt(~mk, sampling=px)[z]
    return out


def geometry_scorecard(ctx, maps: dict[str, np.ndarray], frac: float = 0.05, long_km: float = 5.0,
                       far_km: float = 1.0) -> pd.DataFrame:
    lab = ctx.data.labels & (ctx.folds.fold_map >= 0)
    strata = trace_strata(lab, ctx.data.grid)
    sk = skeletonize(lab)
    comp, n = ndimage.label(sk, structure=np.ones((3, 3), bool))
    counts = np.bincount(comp.ravel(), minlength=n + 1)
    comp_km = counts * ctx.data.grid.pixel_km * 1.12             # same length rule as component_length_km
    long_ids = np.nonzero((comp_km >= long_km) & (np.arange(n + 1) > 0))[0]
    far = ctx.dist_to_labels_km() > far_km
    pix_km2 = ctx.data.grid.pixel_km ** 2
    rows = []
    for m, p in maps.items():
        mask = phase4_eval.fold_masks(ctx, p, frac)
        d = _fold_distance(ctx, mask)
        k = _support(d)
        r = {"model": m, "frac": frac}
        for s, sm in strata.items():
            r[f"recall_w_{s}"] = float(k[sm].mean()) if sm.any() else np.nan
            r[f"n_px_{s}"] = int(sm.sum())
        cov = (d <= 300) & sk
        cov_frac = np.bincount(comp[cov], minlength=n + 1) / np.maximum(counts, 1)
        cov_lab, n_cov = ndimage.label(cov, structure=np.ones((3, 3), bool))
        longest = np.zeros(n + 1)
        if n_cov:                                    # covered runs lie inside one skeleton component each
            run_size = np.bincount(cov_lab.ravel())[1:]
            run_comp = np.asarray(ndimage.maximum(comp, cov_lab, index=np.arange(1, n_cov + 1)), int)
            np.maximum.at(longest, run_comp, run_size)
        run_frac = longest / np.maximum(counts, 1)
        r.update(long_fault_covered=float(cov_frac[long_ids].mean()) if len(long_ids) else np.nan,
                 long_fault_longest_run=float(run_frac[long_ids].mean()) if len(long_ids) else np.nan,
                 n_long_faults=int(len(long_ids)))
        fm = mask & far
        _, n_far = ndimage.label(fm, structure=np.ones((3, 3), bool))
        r.update(fp_far_km2=float(fm.sum() * pix_km2), fp_far_share=float(fm.sum() / max(mask.sum(), 1)),
                 fp_far_components=int(n_far))
        rows.append(r)
    return pd.DataFrame(rows)


def orientation_error(ctx, orient_rasters: dict[str, tuple[np.ndarray, np.ndarray]], *, scale_km: float = 0.3,
                      min_coherence: float = 0.7) -> pd.DataFrame:
    """Axial error (deg) of predicted strike at label pixels with a defined direction (OOF folds only)."""
    from ..geology.orientation import axial_angle, axial_difference
    from ..geology.skeleton import orientation_targets

    lab = ctx.data.labels & (ctx.folds.fold_map >= 0)
    t = orientation_targets(lab, ctx.data.grid, scale_km=scale_km, min_coherence=min_coherence)
    ok = (t["ovalid"] > 0) & lab
    truth = axial_angle(t["ocos"], t["osin"])
    rows = []
    for m, (c2, s2) in orient_rasters.items():
        sel = ok & np.isfinite(c2) & np.isfinite(s2)
        err = axial_difference(axial_angle(c2[sel], s2[sel]), truth[sel])
        rows.append({"model": m, "n_px": int(sel.sum()), "orient_err_median_deg": float(np.median(err)) if err.size else np.nan,
                     "orient_err_mean_deg": float(np.mean(err)) if err.size else np.nan,
                     "orient_within_15deg": float((err <= 15).mean()) if err.size else np.nan})
    return pd.DataFrame(rows)


def withheld_distance(ctx, models: list[str], frac: float = 0.05) -> pd.DataFrame:
    """Distance from withheld fault pixels to the model's top-frac mask inside each discovery evaluation zone."""
    from ..p3.budget import binary_mask

    g = ctx.data.grid
    px = (g.pixel_km_y * 1000, g.pixel_km_x * 1000)
    rows = []
    for m in models:
        ds = []
        for d in range(ctx.splits.n_splits):
            p = ctx.discovery_pred(m, d)
            if p is None:
                continue
            ez = ctx.splits.eval_zone(d) & ctx.data.valid
            mask = binary_mask(np.where(ez, p, np.nan), ez, None, frac)
            if mask.any():
                ds.append(ndimage.distance_transform_edt(~mask, sampling=px)[ctx.splits.withheld(d)])
        if ds:
            x = np.concatenate(ds)
            rows.append({"model": m, "frac": frac, "median_m": float(np.median(x)), "p90_m": float(np.quantile(x, 0.9)),
                         "recall@300m": float((x <= 300).mean()), "recall@1km": float((x <= 1000).mean())})
    return pd.DataFrame(rows)
