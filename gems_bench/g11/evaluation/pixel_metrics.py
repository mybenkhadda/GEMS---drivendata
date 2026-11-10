"""Pixel metrics on spatially held-out regions (mapped faults as an INCOMPLETE reference).

* PR-AUC (average precision) and ROC-AUC on block-sampled pixels;
* tolerant recall at a fixed predicted-positive area fraction (a mapped pixel counts as recalled if a predicted
  pixel lies within ``tol_m``) - the area budget makes models with different score scales comparable;
* expected calibration error; all with spatial block-bootstrap CIs when requested.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def pr_auc(y, p) -> float:
    from sklearn.metrics import average_precision_score

    y, p = np.asarray(y), np.asarray(p)
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        raise ValueError("PR-AUC needs both classes")
    return float(average_precision_score(y[ok] > 0.5, p[ok]))


def roc_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(y), np.asarray(p)
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        raise ValueError("ROC-AUC needs both classes")
    return float(roc_auc_score(y[ok] > 0.5, p[ok]))


def recall_at_area(p: np.ndarray, truth: np.ndarray, region: np.ndarray, area_fraction: float, tol_m: float,
                   px_m: float = 100.0) -> dict:
    v = p[region & np.isfinite(p)]
    if not v.size or not (truth & region).any():
        return {"area_fraction": area_fraction, "recall": np.nan, "precision": np.nan}
    thr = np.quantile(v, 1 - area_fraction)
    pred = region & (np.nan_to_num(p, nan=-np.inf) >= thr)
    d_pred = ndimage.distance_transform_edt(~pred, sampling=px_m) if pred.any() else np.full(p.shape, np.inf)
    d_truth = ndimage.distance_transform_edt(~truth, sampling=px_m)
    t = truth & region
    return {"area_fraction": area_fraction, "threshold": float(thr), "recall": float((d_pred[t] <= tol_m).mean()),
            "precision": float((d_truth[pred] <= tol_m).mean()) if pred.any() else np.nan,
            "n_predicted": int(pred.sum())}


def ece(y, p, bins: int = 10) -> float:
    from ..training.calibration import reliability

    return reliability(np.asarray(p, float), np.asarray(y, float), bins)[1]


def evaluate_region(p: np.ndarray, truth: np.ndarray, region: np.ndarray, sample_rows, sample_cols, blocks,
                    cfg: dict, bootstrap: bool = True) -> dict:
    """Headline pixel metrics for one held-out region (with block-bootstrap CI on PR-AUC)."""
    from .spatial_bootstrap import block_bootstrap

    y = truth[sample_rows, sample_cols].astype(float)
    s = p[sample_rows, sample_cols]
    out = {"n_samples": int(len(y)), "prevalence": float(np.mean(y)) if len(y) else np.nan}
    try:
        out["pr_auc"], out["roc_auc"] = pr_auc(y, s), roc_auc(y, s)
    except ValueError as e:
        out["pr_auc"] = out["roc_auc"] = np.nan
        out["note"] = str(e)
    out["ece"] = ece(y, s)
    ev = cfg["evaluation"]
    for af in ev["area_fractions"]:
        r = recall_at_area(p, truth, region, af, ev["pixel_tolerance_m"])
        out[f"recall_at_{af:g}"] = r["recall"]
        out[f"precision_at_{af:g}"] = r["precision"]
    if bootstrap and len(y) and np.isfinite(out["pr_auc"]):
        bb = block_bootstrap(lambda idx: pr_auc(y[idx], s[idx]), blocks, ev["n_bootstrap"])
        out.update(pr_auc_ci_low=bb["ci_low"], pr_auc_ci_high=bb["ci_high"], n_blocks=bb["n_blocks"])
    return out
