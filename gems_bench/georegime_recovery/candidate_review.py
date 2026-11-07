"""Section 13 - audit of the Atlas candidate lineaments against model confidence (NO pseudo-labelling).

Label-free priority = the V1 review formula ``0.5 continuity + 0.3 clip(orientation_agreement) +
0.2 clip(multiscale_persistence)`` (``GeoRegimeExperiment.candidate_proposals``), computed for EVERY
proposal (V1 pre-filtered by evidence_count / agreement / continuity; here the filter is reported as a flag).
Model confidence along each trace: mean / q90 / max of the V1 full-model raster over the rasterised line
(all_touched), and its percentile rank among eligible non-mapped pixels. ``mapped_fault_distance_m`` comes from
the Atlas and is a diagnostic annotation only. Candidates never become training labels or submission inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("candidate_id", "geometry", "length_km", "continuity", "orientation_agreement",
            "multiscale_persistence", "evidence_count")


def line_pixels(geom, transform, shape):
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import transform as wtransform

    if geom is None or geom.is_empty:
        return np.array([], int), np.array([], int)
    x0, y0, x1, y1 = geom.bounds
    pad = 2 * abs(transform.a)
    win = from_bounds(x0 - pad, y0 - pad, x1 + pad, y1 + pad, transform).round_offsets().round_lengths()
    try:
        win = win.intersection(Window(0, 0, shape[1], shape[0]))
    except Exception:                                       # rasterio WindowError: no overlap with the grid
        return np.array([], int), np.array([], int)
    if win.width <= 0 or win.height <= 0:
        return np.array([], int), np.array([], int)
    t = wtransform(win, transform)
    r = rasterize([(geom, 1)], out_shape=(int(win.height), int(win.width)), transform=t, fill=0, dtype="uint8",
                  all_touched=True)
    rr, cc = np.nonzero(r)
    return rr + int(win.row_off), cc + int(win.col_off)


def review(lin, prob, valid, labels, transform, extra_probs: dict | None = None) -> tuple[pd.DataFrame, dict]:
    miss = [c for c in REQUIRED if c not in lin.columns]
    if miss:
        raise ValueError(f"Atlas lineament table lacks {miss}")
    elig = valid & ~labels & np.isfinite(prob)
    ref = np.sort(prob[elig])
    df = lin.drop(columns="geometry").copy()
    df["label_free_support"] = (0.5 * df.continuity.fillna(0) + 0.3 * df.orientation_agreement.clip(0, 1).fillna(0)
                                + 0.2 * df.multiscale_persistence.clip(0, 1).fillna(0)).clip(0, 1)
    df["v1_review_filter_pass"] = ((df.evidence_count >= 2) & (df.orientation_agreement >= 0.5)
                                   & (df.continuity >= 0.65))
    stats = {k: [] for k in ("model_mean_p", "model_q90_p", "model_max_p", "n_px")}
    extra = {f"{k}_mean_p": [] for k in (extra_probs or {})}
    pix = {}
    for cid, g in zip(lin.candidate_id, lin.geometry):
        rr, cc = line_pixels(g, transform, prob.shape)
        ok = valid[rr, cc] if rr.size else np.array([], bool)
        rr, cc = rr[ok], cc[ok]
        pix[cid] = (rr, cc)
        v = prob[rr, cc]
        v = v[np.isfinite(v)]
        stats["n_px"].append(int(v.size))
        stats["model_mean_p"].append(float(v.mean()) if v.size else np.nan)
        stats["model_q90_p"].append(float(np.quantile(v, 0.9)) if v.size else np.nan)
        stats["model_max_p"].append(float(v.max()) if v.size else np.nan)
        for k, arr in (extra_probs or {}).items():
            w = arr[rr, cc]
            w = w[np.isfinite(w)]
            extra[f"{k}_mean_p"].append(float(w.mean()) if w.size else np.nan)
    for k, v in {**stats, **extra}.items():
        df[k] = v
    df["model_pct_rank"] = np.searchsorted(ref, df.model_mean_p.fillna(-1).to_numpy()) / max(len(ref), 1)
    sup_hi = df.label_free_support >= df.label_free_support.quantile(0.75)
    mod_hi = df.model_pct_rank >= 0.99
    df["agreement_class"] = np.select(
        [sup_hi & mod_hi, sup_hi & ~mod_hi, ~sup_hi & mod_hi], ["geology_and_model_high", "geology_high_model_low",
                                                               "model_high_geology_moderate"],
        default="both_moderate_or_low")
    if "limitations" not in df.columns:
        df["limitations"] = ""
    df["status"] = "UNVERIFIED geological structure (no pseudo-label)"
    st = {"n_candidates": int(len(df)), "n_v1_filter_pass": int(df.v1_review_filter_pass.sum()),
          "n_geology_and_model_high": int((df.agreement_class == "geology_and_model_high").sum()),
          "n_geology_high_model_low": int((df.agreement_class == "geology_high_model_low").sum()),
          "spearman_support_vs_model": float(df[["label_free_support", "model_mean_p"]].corr("spearman").iloc[0, 1])
          if len(df) > 3 else np.nan,
          "median_model_pct_rank": float(df.model_pct_rank.median()) if len(df) else np.nan}
    return df, {"stats": st, "pixels": pix}
