"""Pixel-level model fusion (Part 9 of the G11 spec).

Members: G11 experiment maps only (E0-E4). Each member was fitted on the TRAIN split and checkpoint-selected on
VALIDATION, so its scores on the STACK split are out-of-sample: supervised fusion (weights, logistic stacking,
terrain-conditioned stacking) is FITTED on stack samples and the method is SELECTED on validation. External maps
(G07 / M9 / P6) are not fusion members: their original split cannot be verified against the G11 split.

Methods: best_single, mean, median, max, calibrated_weighted (Platt-calibrate each member on stack, weights
proportional to stack average precision), logistic_stacking, terrain_conditioned (stacking with DEM availability
and terrain roughness interactions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _X(members: dict, names, idx, cond: dict | None = None, interact: bool = False):
    cols = [_logit(np.nan_to_num(members[n].ravel()[idx], nan=0.0)) for n in names]
    if cond:
        for k, a in cond.items():
            c = np.nan_to_num(a.ravel()[idx], nan=0.0)
            cols.append(c)
            if interact:
                cols += [x * c for x in cols[: len(names)]]
    return np.stack(cols, 1)


def fit_fusions(members: dict, truth: np.ndarray, stack_idx: np.ndarray, cond: dict | None = None) -> dict:
    """Fit every supervised fusion on stack sample indices (flat). Returns {method: fitted object or weights}."""
    from sklearn.linear_model import LogisticRegression

    from .evaluation.pixel_metrics import pr_auc
    from .training.calibration import Calibrator

    names = sorted(members)
    y = truth.ravel()[stack_idx].astype(int)
    out = {"names": names}
    cal = {n: Calibrator("platt").fit(members[n].ravel()[stack_idx], y.astype(float), "stack") for n in names}
    ap = {}
    for n in names:
        try:
            ap[n] = pr_auc(y, members[n].ravel()[stack_idx])
        except ValueError:
            ap[n] = 0.0
    w = np.array([max(ap[n], 0.0) for n in names])
    out["calibrated_weighted"] = {"calibrators": cal, "weights": w / max(w.sum(), EPS)}
    out["best_single"] = max(ap, key=ap.get)
    out["stack_ap"] = ap
    if len(np.unique(y)) == 2:
        out["logistic_stacking"] = LogisticRegression(max_iter=1000, C=1.0).fit(_X(members, names, stack_idx), y)
        if cond:
            out["terrain_conditioned"] = LogisticRegression(max_iter=2000, C=1.0).fit(
                _X(members, names, stack_idx, cond, interact=True), y)
    return out


def apply_fusion(method: str, fit: dict, members: dict, valid: np.ndarray, cond: dict | None = None,
                 chunk: int = 2_000_000) -> np.ndarray:
    names = fit["names"]
    out = np.full(valid.shape, np.nan, np.float32)
    flat = np.flatnonzero(valid.ravel())
    for i in range(0, flat.size, chunk):
        idx = flat[i:i + chunk]
        P = np.stack([np.nan_to_num(members[n].ravel()[idx], nan=0.0) for n in names], 1)
        if method == "mean":
            v = P.mean(1)
        elif method == "median":
            v = np.median(P, 1)
        elif method == "max":
            v = P.max(1)
        elif method == "best_single":
            v = P[:, names.index(fit["best_single"])]
        elif method == "calibrated_weighted":
            cw = fit["calibrated_weighted"]
            v = sum(wt * cw["calibrators"][n].transform(P[:, k]) for k, (n, wt) in enumerate(zip(names, cw["weights"])))
        elif method == "logistic_stacking":
            v = fit["logistic_stacking"].predict_proba(_X(members, names, idx))[:, 1]
        elif method == "terrain_conditioned":
            v = fit["terrain_conditioned"].predict_proba(_X(members, names, idx, cond, interact=True))[:, 1]
        else:
            raise ValueError(method)
        out.ravel()[idx] = np.asarray(v, np.float32)
    return out


METHODS = ("best_single", "mean", "median", "max", "calibrated_weighted", "logistic_stacking", "terrain_conditioned")


def compare(members: dict, truth: np.ndarray, valid: np.ndarray, stack_idx, val_idx, cond=None) -> tuple[pd.DataFrame, dict, dict]:
    """Fit on stack, score every method on validation samples. Returns (table, fit, maps)."""
    from .evaluation.pixel_metrics import pr_auc, roc_auc

    fit = fit_fusions(members, truth, stack_idx, cond)
    yv = truth.ravel()[val_idx].astype(int)
    rows, maps = [], {}
    for m in METHODS:
        if m in ("logistic_stacking", "terrain_conditioned") and m not in fit:
            rows.append({"method": m, "status": "SKIPPED (single class on stack or no conditioning layers)"})
            continue
        mp = apply_fusion(m, fit, members, valid, cond)
        maps[m] = mp
        s = mp.ravel()[val_idx]
        try:
            rows.append({"method": m, "status": "OK", "validation_pr_auc": pr_auc(yv, s), "validation_roc_auc": roc_auc(yv, s)})
        except ValueError as e:
            rows.append({"method": m, "status": f"NO_SCORE: {e}"})
    t = pd.DataFrame(rows)
    return t, fit, maps
