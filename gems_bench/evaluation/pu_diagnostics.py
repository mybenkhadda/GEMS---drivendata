"""Experiment B safeguards: is a PU model finding unmapped structure, or just predicting "fault" more often?

Every number is reported next to the B0 reference at the same area budget, because a larger footprint alone
raises recall of withheld faults (and lowers DTI through FP). High-confidence predictions far from mapped
faults are *candidates*, never verified faults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import phase4_eval


def confidence_mass(ctx, maps: dict[str, np.ndarray], far_km: float = 1.0, frac: float = 0.05) -> pd.DataFrame:
    far = ctx.dist_to_labels_km() > far_km
    v = ctx.data.valid & (ctx.folds.fold_map >= 0)
    rows = []
    for m, p in maps.items():
        q = np.where(v, np.nan_to_num(p), 0.0)
        mask = phase4_eval.fold_masks(ctx, p, frac)
        rows.append({"model": m, "mean_p_valid": float(q[v].mean()), "mean_p_unlabeled": float(q[v & ~ctx.data.labels].mean()),
                     "mass_share_far": float(q[far & v].sum() / max(q[v].sum(), 1e-12)),
                     "area_p_ge_0.5_km2": float(((q >= 0.5) & v).sum() * ctx.data.grid.pixel_km ** 2),
                     "far_share_at_budget": float((mask & far).sum() / max(mask.sum(), 1)), "budget_frac": frac})
    return pd.DataFrame(rows)


def training_collapse(history: pd.DataFrame) -> pd.DataFrame:
    """Per unit: final / max mean p on unlabeled training pixels and whether the collapse guard fired."""
    if history is None or not len(history) or "loss_unlabeled_mean_p" not in history:
        return pd.DataFrame()
    g = history.groupby("unit")
    return pd.DataFrame({"unlabeled_p_last": g.loss_unlabeled_mean_p.last(), "unlabeled_p_max": g.loss_unlabeled_mean_p.max(),
                         "collapsed": g.collapsed.max() if "collapsed" in history else False}).reset_index()
