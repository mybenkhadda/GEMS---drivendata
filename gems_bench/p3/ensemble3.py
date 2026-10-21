"""Nested ensemble validation (Phase 3G) and the decomposition of Phase-2's rank-ensemble 0.1125.

Protocol: every free choice — member weights, fusion fit, area budget — is made on the folds
other than the scored one, then applied to it. Members are always compared under the *same*
output rule (binary per-fold top-a mask), so an ensemble can only win by ranking pixels better.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from ..ensemble import rank_normalise
from ..gbm import RANK_TRANSFORM_GRID
from ..metric import apply_output_transform
from .context import P3Context
from .curves import DEFAULT_AREA_FRACS, crossfit_select, dti_from, zone_curves
from .transforms import FoldScorers, compare_families, family_table, quantile_map


# ---------------------------------------------------------------------------
# 1. why was rank_mean 0.1125?
# ---------------------------------------------------------------------------
def rank_grid_on_values(name: str, p: np.ndarray, scorers: FoldScorers) -> pd.DataFrame:
    """Phase-2's exact evaluation of rank_mean: RANK grid applied to the map values themselves."""
    rows = []
    for g, s, f in itertools.product(*RANK_TRANSFORM_GRID.values()):
        q = apply_output_transform(p, g, s, f)
        rows += scorers.table(q, family="p2_rank_grid_on_values", param=json.dumps({"gamma": g, "scale": s, "floor": f}),
                              gamma=g, scale=s, floor=f)
    df = pd.DataFrame(rows)
    df.insert(0, "model", name)
    return df


def decompose_rank_ensemble(ctx: P3Context, scorers: FoldScorers, members=("G3", "G4", "G2")) -> pd.DataFrame:
    """Rows: Phase-2 reproduction, and every member / ensemble under the same transform families."""
    v, fm, lab = ctx.data.valid, ctx.folds.fold_map, ctx.data.labels
    ranks_ord = {m: rank_normalise(ctx.oof(m), v, "ordinal") for m in members}
    ranks_avg = {m: rank_normalise(ctx.oof(m), v, "average") for m in members}
    ens_ord = np.nanmean(np.stack(list(ranks_ord.values())), 0)
    ens_avg = np.nanmean(np.stack(list(ranks_avg.values())), 0)
    tabs = [rank_grid_on_values("ens_rank_mean[ordinal]", np.where(v, ens_ord, np.nan), scorers),
            rank_grid_on_values("ens_rank_mean[average]", np.where(v, ens_avg, np.nan), scorers)]
    for m in members:                                      # single members under the *same* RANK grid
        tabs.append(rank_grid_on_values(f"{m}[ordinal rank]", ranks_ord[m], scorers))
    for name, p in [("ens_rank_mean[average]", ens_avg)] + [(m, ctx.oof(m)) for m in members]:
        tabs.append(family_table(name, np.where(v, p, np.nan), lab, fm, v, scorers, families=["hard_threshold"]))
    summ, _ = compare_families(pd.concat(tabs, ignore_index=True))
    p2 = ctx.bench.ensembles / "ensemble_summary.csv"
    if p2.exists():
        r = pd.read_csv(p2).set_index("ensemble").loc["rank_mean"]
        summ = pd.concat([pd.DataFrame([{"model": "ens_rank_mean (Phase-2 reported)", "family": "p2_rank_grid_on_values",
                                         "dti_crossfit": r.dti_crossfit_tf, "fold_min": r.fold_min, "fold_max": r.fold_max}]),
                          summ], ignore_index=True)
    return summ


# ---------------------------------------------------------------------------
# 2. nested ensemble evaluation on geographic folds
# ---------------------------------------------------------------------------
def simplex(n: int, step: float):
    k = int(round(1 / step))
    for c in itertools.product(range(k + 1), repeat=n):
        if sum(c) == k:
            yield np.array(c, float) / k


def member_quantiles(ctx: P3Context, maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-fold average-rank quantiles: label-free and scale-free inputs for fusion."""
    return {m: quantile_map(p, ctx.data.valid, ctx.folds.fold_map, "average") for m, p in maps.items()}


def weighted_table(ctx: P3Context, U: dict[str, np.ndarray], step: float, fracs=DEFAULT_AREA_FRACS) -> pd.DataFrame:
    names = list(U)
    stack = np.stack([np.nan_to_num(U[m]) for m in names])
    rows = []
    for w in simplex(len(names), step):
        e = np.where(ctx.data.valid, np.tensordot(w, stack, axes=1), np.nan).astype(np.float32)
        zc = zone_curves(e, ctx.data.labels, ctx.folds.fold_map, ctx.data.valid, fracs=fracs)
        zc["w"] = json.dumps(dict(zip(names, w.round(3).tolist())))
        rows.append(zc)
    return pd.concat(rows, ignore_index=True)


def fitted_fusion_rows(ctx: P3Context, U: dict[str, np.ndarray], kind: str, rng, fracs=DEFAULT_AREA_FRACS,
                       n_fit: int = 400_000) -> pd.DataFrame:
    """'stack_lr' (logistic stacking on member quantiles) or 'iso_mean' (mean of per-member isotonic credit).

    For outer fold k the fusion is fitted on folds != k; the area budget is chosen on folds != k from
    the same fitted map (training-side), then fold k is scored once.
    """
    from scipy import ndimage
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    v, fm, lab = ctx.data.valid, ctx.folds.fold_map, ctx.data.labels
    names = list(U)
    kd = np.clip(1 - ndimage.distance_transform_edt(~lab, sampling=(100.0, 100.0)) / 300.0, 0, None)
    near = kd >= 2.0 / 3.0                                   # within 100 m: full-ish TP credit
    out = []
    for k in ctx.fold_ids:
        tr = np.flatnonzero((v & (fm >= 0) & (fm != k)).ravel())
        tr = rng.choice(tr, min(n_fit, len(tr)), replace=False)
        X = np.stack([np.nan_to_num(U[m].ravel()[tr]) for m in names], 1)
        allX = np.stack([np.nan_to_num(U[m][v]) for m in names], 1)
        F = np.full(v.shape, np.nan, np.float32)
        if kind == "stack_lr":
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(X, near.ravel()[tr])
            F[v] = lr.decision_function(allX).astype(np.float32)
            info = dict(zip(names, lr.coef_[0].round(3).tolist()))
        else:
            cal = [IsotonicRegression(out_of_bounds="clip").fit(X[:, j], kd.ravel()[tr]) for j in range(len(names))]
            F[v] = np.mean([c.predict(allX[:, j]) for j, c in enumerate(cal)], 0).astype(np.float32)
            info = {}
        zc = zone_curves(F, lab, fm, v, fracs=fracs)
        train_side = zc[zc.zone != k]
        best = train_side.groupby("target_frac").apply(dti_from).idxmax()
        held = zc[(zc.zone == k) & (zc.target_frac == best)]
        out.append({"method": kind, "zone": k, "target_frac": best, "tp_w": held.tp_w.sum(), "fp_w": held.fp_w.sum(),
                    "fn_w": held.fn_w.sum(), "dti_heldout": dti_from(held), "fit_info": json.dumps(info)})
    return pd.DataFrame(out)


def nested_ensemble_eval(ctx: P3Context, maps: dict[str, np.ndarray], *, step: float | None = None,
                         fused=("stack_lr", "iso_mean"), rng=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold held-out DTI of members and ensembles; every choice made on the other folds."""
    rng = rng or np.random.default_rng(ctx.cfg.seed)
    v, fm, lab = ctx.data.valid, ctx.folds.fold_map, ctx.data.labels
    U = member_quantiles(ctx, maps)
    per = []
    # members alone (budget cross-fitted)
    for m, p in maps.items():
        p_, _ = crossfit_select(zone_curves(p, lab, fm, v), ["target_frac"])
        per.append(p_.assign(method=f"single:{m}"))
    # best single member chosen on the training folds (the fair baseline for an ensemble)
    single = pd.concat([zone_curves(p, lab, fm, v).assign(member=m) for m, p in maps.items()], ignore_index=True)
    p_, _ = crossfit_select(single, ["member", "target_frac"])
    per.append(p_.assign(method="best_single_nested"))
    # probability mean / rank mean (no weights to fit)
    for name, e in (("prob_mean", np.nanmean(np.stack([maps[m] for m in maps]), 0)),
                    ("rank_mean", np.nanmean(np.stack([U[m] for m in U]), 0))):
        p_, _ = crossfit_select(zone_curves(np.where(v, e, np.nan), lab, fm, v), ["target_frac"])
        per.append(p_.assign(method=name))
    # weighted rank mean: weights and budget chosen jointly on the other folds
    step = step or (0.1 if len(maps) <= 3 else 0.25)
    wt = weighted_table(ctx, U, step)
    p_, _ = crossfit_select(wt, ["w", "target_frac"])
    per.append(p_.assign(method=f"weighted_rank(step={step})"))
    for kind in fused:
        f_ = fitted_fusion_rows(ctx, U, kind, rng)
        per.append(f_)
    per = pd.concat(per, ignore_index=True)
    per["method"] = per["method"].fillna("")
    summ = []
    base = per[per.method == "best_single_nested"].set_index("zone").dti_heldout
    for m, g in per.groupby("method", sort=False):
        g = g.set_index("zone")
        d = g.dti_heldout - base.reindex(g.index)
        summ.append({"method": m, "dti_pooled": dti_from(g), "fold_min": g.dti_heldout.min(), "fold_max": g.dti_heldout.max(),
                     "folds_beating_best_single": int((d > 0).sum()), "mean_delta_vs_best_single": float(d.mean()),
                     **{f"dti_fold{z}": g.dti_heldout.get(z) for z in ctx.fold_ids}})
    return pd.DataFrame(summ).sort_values("dti_pooled", ascending=False), per


# ---------------------------------------------------------------------------
# 3. pseudo-discovery maps for ensembles (same rule, inside each evaluation zone)
# ---------------------------------------------------------------------------
def discovery_ensemble_preds(ctx: P3Context, members, weights: dict[str, float] | None = None,
                             name: str = "ens") -> dict:
    """{(name, d): map}: weighted mean of member quantiles computed inside the split's evaluation zone."""
    out = {}
    for d in range(ctx.splits.n_splits):
        ez = ctx.splits.eval_zone(d) & ctx.data.valid
        parts, ws = [], []
        for m in members:
            p = ctx.discovery_pred(m, d)
            if p is None:
                break
            parts.append(quantile_map(np.where(ez, p, np.nan), ez, None, "average"))
            ws.append(1.0 if weights is None else weights.get(m, 0.0))
        else:
            ws = np.asarray(ws) / max(sum(ws), 1e-9)
            e = np.tensordot(ws, np.stack([np.nan_to_num(x) for x in parts]), axes=1)
            out[(name, d)] = np.where(ez, e, np.nan).astype(np.float32)
    return out


def promotion_decision(summary: pd.DataFrame, disc: pd.DataFrame | None, method: str, min_gain: float = 0.003,
                       min_folds: int = 2) -> dict:
    """Promote an ensemble only if it beats the nested best single model in >= min_folds folds, by >= min_gain
    pooled, and is not worse on pseudo-discovery (retained_fp policy, cross-fitted area)."""
    r = summary.set_index("method")
    if method not in r.index or "best_single_nested" not in r.index:
        return {"method": method, "promote": False, "reason": "missing rows"}
    gain = float(r.loc[method, "dti_pooled"] - r.loc["best_single_nested", "dti_pooled"])
    folds = int(r.loc[method, "folds_beating_best_single"])
    ok_disc, disc_note = True, "no discovery rows"
    if disc is not None and len(disc):
        dd = disc[(disc.row == "crossfit_area") & (disc.policy == "retained_fp") & (disc.variant == "model")].set_index("model").dti
        if "ens" in dd.index:
            best_member = dd.drop("ens").max()
            ok_disc = bool(dd["ens"] >= best_member - 1e-4)
            disc_note = f"discovery ens {dd['ens']:.4f} vs best member {best_member:.4f}"
    promote = gain >= min_gain and folds >= min_folds and ok_disc
    return {"method": method, "promote": promote, "pooled_gain": gain, "folds_beating": folds, "discovery": disc_note}
