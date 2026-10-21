"""Experiment G: discovery-aware ensembling with honest (nested) weight selection.

Members enter as label-free per-fold rank quantiles (``ensemble3.member_quantiles``). Geographic scores use a
nested scheme: for held-out fold k, weights *and* area budget are chosen on folds != k.

Pseudo-discovery weights are never fitted on the withheld systems they are scored on:
``geo_masked``  weights for split d chosen on the geographic OOF maps with split d's withheld systems + ring
                removed from truth and from the scored area (they are labels in the geographic folds!);
``loso``        weights for split d chosen on the other discovery splits' predictions (leave-one-split-out).

Methods: ``mean`` (probabilities), ``rank_mean``, ``weighted_rank`` (simplex grid), ``convex_stack``
(non-negative weights summing to 1, fitted by a regularised logistic objective on training-side pixels),
``region_weighted`` (per-region simplex weights; only meaningful if region stability supports it) and
``uncertainty`` (rank mean penalised by member disagreement; refused unless calibration was verified).
An ensemble is promoted only through ``evaluation.scoreboard.promotion_decision`` against its best member.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..p3 import ensemble3
from ..p3.curves import DEFAULT_AREA_FRACS, crossfit_select, dti_from, zone_curves
from ..p3.transforms import quantile_map


def quantiles(ctx, members: list[str]) -> dict[str, np.ndarray]:
    return ensemble3.member_quantiles(ctx, {m: ctx.oof(m) for m in members})


def _combine(U: dict[str, np.ndarray], w: dict[str, float], valid: np.ndarray) -> np.ndarray:
    e = sum(float(w.get(m, 0.0)) * np.nan_to_num(U[m]) for m in U)
    return np.where(valid, e, np.nan).astype(np.float32)


def geo_simple(ctx, U: dict[str, np.ndarray], maps: dict[str, np.ndarray] | None, kind: str) -> np.ndarray:
    v = ctx.data.valid
    if kind == "mean":
        return np.where(v, np.nanmean(np.stack([maps[m] for m in maps]), 0), np.nan).astype(np.float32)
    return np.where(v, np.nanmean(np.stack([U[m] for m in U]), 0), np.nan).astype(np.float32)


def geo_weighted(ctx, U: dict[str, np.ndarray], step: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Stitched OOF map: fold k uses the (weights, budget) chosen on folds != k."""
    wt = ensemble3.weighted_table(ctx, U, step)
    per, _ = crossfit_select(wt, ["w", "target_frac"])
    out = np.full(ctx.data.shape, np.nan, np.float32)
    for r in per.itertuples():
        z = (ctx.folds.fold_map == r.zone) & ctx.data.valid
        out[z] = _combine(U, json.loads(r.w), ctx.data.valid)[z]
    return out, per.rename(columns={"w": "weights"})


def geo_convex_stack(ctx, U: dict[str, np.ndarray], rng, n_fit: int = 300_000, l2: float = 1e-3) -> tuple[np.ndarray, pd.DataFrame]:
    """Simplex-constrained weights (softmax parametrisation) of a logistic model fitted on folds != k."""
    from scipy import ndimage
    from scipy.optimize import minimize

    v, fm, lab = ctx.data.valid, ctx.folds.fold_map, ctx.data.labels
    names = list(U)
    kd = np.clip(1 - ndimage.distance_transform_edt(~lab, sampling=(100.0, 100.0)) / 300.0, 0, None)
    y_all = (kd >= 2.0 / 3.0).ravel()
    out = np.full(v.shape, np.nan, np.float32)
    rows = []
    for k in ctx.fold_ids:
        tr = np.flatnonzero((v & (fm >= 0) & (fm != k)).ravel())
        tr = rng.choice(tr, min(n_fit, len(tr)), replace=False)
        X = np.stack([np.nan_to_num(U[m].ravel()[tr]) for m in names], 1)
        y = y_all[tr].astype(float)
        pw = (1 - y.mean()) / max(y.mean(), 1e-6)            # class balance

        def nll(theta):
            w = np.exp(theta[:-2] - theta[:-2].max())
            w = w / w.sum()
            z = theta[-2] * (X @ w) + theta[-1]
            ll = pw * y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)
            return ll.mean() + l2 * np.sum(theta[:-2] ** 2)

        res = minimize(nll, np.r_[np.zeros(len(names)), 5.0, -2.5], method="L-BFGS-B")
        w = np.exp(res.x[:-2] - res.x[:-2].max())
        w = dict(zip(names, (w / w.sum()).round(4).tolist()))
        z = (fm == k) & v
        out[z] = _combine(U, w, v)[z]
        rows.append({"zone": k, "weights": json.dumps(w), "converged": bool(res.success)})
    return out, pd.DataFrame(rows)


def geo_uncertainty(ctx, U: dict[str, np.ndarray], lam: float = 0.5, calibration_ok: bool = False) -> np.ndarray:
    if not calibration_ok:
        raise RuntimeError("uncertainty-aware fusion needs verified member calibration (calibration audit); refused")
    S = np.stack([U[m] for m in U])
    return np.where(ctx.data.valid, np.nanmean(S, 0) - lam * np.nanstd(S, 0), np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# pseudo-discovery maps with honest weights
# ---------------------------------------------------------------------------
def disc_quantiles(ctx, members: list[str], d: int) -> dict[str, np.ndarray] | None:
    ez = ctx.splits.eval_zone(d) & ctx.data.valid
    out = {}
    for m in members:
        p = ctx.discovery_pred(m, d)
        if p is None:
            return None
        out[m] = quantile_map(np.where(ez, p, np.nan), ez, None, "average")
    return out


def weights_geo_masked(ctx, U: dict[str, np.ndarray], d: int, step: float, fracs=DEFAULT_AREA_FRACS) -> dict:
    """Best (weights, budget) on geographic OOF with split d's withheld systems + ring removed."""
    from ..p3.ensemble3 import simplex

    ign = ctx.splits.ignore(d)
    v = ctx.data.valid & ~ign
    lab = ctx.data.labels & ~ign
    names, best = list(U), (-1.0, None)
    for w in simplex(len(names), step):
        wd = dict(zip(names, w.tolist()))
        zc = zone_curves(_combine(U, wd, v), lab, ctx.folds.fold_map, v, fracs=fracs)
        s = zc.groupby("target_frac").apply(dti_from).max()
        if s > best[0]:
            best = (float(s), wd)
    return best[1]


def weights_loso(ctx, members: list[str], d: int, step: float, policy: str = "excl") -> dict:
    """Best weights on the other discovery splits (pooled cross-fitted area under ``policy``)."""
    from ..p3 import discovery3
    from ..p3.ensemble3 import simplex

    others = [x for x in range(ctx.splits.n_splits) if x != d]
    Q = {x: disc_quantiles(ctx, members, x) for x in others}
    best = (-1.0, None)
    for w in simplex(len(members), step):
        wd = dict(zip(members, w.tolist()))
        extra = {("ens", x): _combine(Q[x], wd, ctx.splits.eval_zone(x) & ctx.data.valid) for x in others if Q[x]}
        rows = []
        for x in others:
            m = discovery3.split_masks(ctx, x)
            truth, ex = discovery3.policy_truth_exclude(m, policy)
            from ..p3.curves import BinaryCurve

            c = BinaryCurve(extra[("ens", x)], truth, m["ez"], ex, far_mask=m["far"])
            rows.append(c.at_area(DEFAULT_AREA_FRACS).assign(zone=x))
        s = pd.concat(rows).groupby("target_frac").apply(dti_from).max()
        if s > best[0]:
            best = (float(s), wd)
    return best[1]


def discovery_maps(ctx, members: list[str], weights_by_split: dict[int, dict]) -> dict[int, np.ndarray]:
    out = {}
    for d, w in weights_by_split.items():
        Q = disc_quantiles(ctx, members, d)
        if Q is not None:
            out[d] = _combine(Q, w, ctx.splits.eval_zone(d) & ctx.data.valid)
    return out


def region_stability(ctx, maps: dict[str, np.ndarray]) -> pd.DataFrame:
    from ..geo_eval import dti_by_region, region_map

    reg = region_map(ctx.data)
    return pd.concat([dti_by_region(p, ctx.data, reg).assign(model=m) for m, p in maps.items()], ignore_index=True)
