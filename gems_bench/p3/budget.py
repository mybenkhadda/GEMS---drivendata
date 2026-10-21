"""Recall-area-DTI analysis at matched prediction-area budgets (Phase 3B).

High recall is only meaningful at a fixed predicted area: a map that is >= 0.5 everywhere has
recall 1. Every comparison here is therefore made between *binary top-a masks* of each model
(per-fold quantile cut), which is also the DTI-optimal output form (curves.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metric import DTI_ALPHA, DTI_BETA, DTI_EPS
from .context import P3Context, model_color
from .curves import DEFAULT_AREA_FRACS, crossfit_select, pooled, zone_curves

MATCHED_FRACS = (0.01, 0.02, 0.05, 0.10, 0.20)


def binary_mask(p: np.ndarray, valid: np.ndarray, zones: np.ndarray | None, frac: float) -> np.ndarray:
    """Top-``frac`` pixels of ``p`` inside every zone (or over ``valid`` when zones is None)."""
    out = np.zeros(p.shape, bool)
    ids = [None] if zones is None else [int(z) for z in np.unique(zones[valid & (zones >= 0)])]
    for z in ids:
        m = valid if z is None else valid & (zones == z)
        v = np.nan_to_num(p[m], nan=0.0)
        thr = np.sort(v)[len(v) - max(int(np.ceil(frac * len(v))), 1)]
        out |= m & (np.nan_to_num(p, nan=-1.0) >= thr)
    return out


def geo_curves(ctx: P3Context, names, fracs=DEFAULT_AREA_FRACS, modes=("per_zone",), far_km: float = 1.0,
               maps: dict | None = None) -> pd.DataFrame:
    """Per-fold binary-mask curves on the geographic folds (truth = all mapped faults)."""
    far = ctx.dist_to_labels_km() > far_km
    rows = []
    for n in names:
        p = maps[n] if maps and n in maps else ctx.oof(n)
        for mode in modes:
            df = zone_curves(p, ctx.data.labels, ctx.folds.fold_map, ctx.data.valid, far_mask=far, fracs=fracs, mode=mode)
            df.insert(0, "model", n)
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def pool_models(curves: pd.DataFrame) -> pd.DataFrame:
    return pooled(curves, ["model", "mode", "target_frac"])


def crossfit_area(curves: pd.DataFrame, mode: str = "per_zone") -> pd.DataFrame:
    """Area fraction chosen on the other folds, scored on the held-out fold, per model."""
    rows = []
    for m, g in curves[curves["mode"] == mode].groupby("model", sort=False):
        per, s = crossfit_select(g, ["target_frac"])
        rows.append({"model": m, "dti_crossfit_area": s["dti_crossfit"], "dti_insample_best": s["dti_insample"],
                     "optimism": s["optimism"], "fold_min": s["fold_min"], "fold_max": s["fold_max"],
                     "frac_all_folds": s["params_all"]["target_frac"],
                     **{f"frac_fold{r.zone}": r.target_frac for r in per.itertuples()},
                     **{f"dti_fold{r.zone}": r.dti_heldout for r in per.itertuples()}})
    return pd.DataFrame(rows)


def matched_budget_table(pooled_df: pd.DataFrame, fracs=MATCHED_FRACS, mode: str = "per_zone") -> pd.DataFrame:
    cols = ["model", "target_frac", "area_km2", "dti", "recall_w", "recall@100m", "recall@200m", "recall@300m",
            "tp_w", "fp_w", "far_area_km2"]
    d = pooled_df[(pooled_df["mode"] == mode) & pooled_df.target_frac.isin(fracs)]
    return d[[c for c in cols if c in d]].sort_values(["target_frac", "dti"], ascending=[True, False]).reset_index(drop=True)


def pairwise_operating_range(curves: pd.DataFrame, a: str, b: str, mode: str = "per_zone") -> pd.DataFrame:
    """DTI and recall differences a - b per area fraction, pooled and per fold (where does a beat b?)."""
    c = curves[curves["mode"] == mode]
    pa, pb = c[c.model == a].set_index(["zone", "target_frac"]), c[c.model == b].set_index(["zone", "target_frac"])
    per = (pa[["dti", "recall_w"]] - pb[["dti", "recall_w"]]).reset_index()
    wide = per.pivot(index="target_frac", columns="zone", values="dti")
    wide.columns = [f"d_dti_fold{z}" for z in wide.columns]
    pool = pool_models(c[c.model.isin([a, b])]).set_index(["model", "target_frac"])
    out = wide.reset_index()
    out["d_dti_pooled"] = (pool.loc[a].dti - pool.loc[b].dti).reindex(out.target_frac).values
    out["d_recall_pooled"] = (pool.loc[a].recall_w - pool.loc[b].recall_w).reindex(out.target_frac).values
    fold_cols = [c for c in out if c.startswith("d_dti_fold")]
    out["a_wins_all_folds"] = (out[fold_cols] > 0).all(1)
    out.insert(0, "comparison", f"{a} - {b}")
    return out


def pareto_front(pooled_df: pd.DataFrame, x: str = "fp_w", y: str = "recall_w") -> pd.DataFrame:
    """Points not dominated in (lower x, higher y) across all models and budgets."""
    d = pooled_df.sort_values([x, y], ascending=[True, False]).reset_index(drop=True)
    best, keep = -np.inf, []
    for i, r in d.iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    d["pareto"] = False
    d.loc[keep, "pareto"] = True
    return d


def density_sensitivity(pooled_df: pd.DataFrame, rhos=(1.0, 0.5, 0.25, 0.1, 0.05), mode: str = "per_zone") -> pd.DataFrame:
    """Best area budget if the hidden truth were a random rho-thinning of the mapped-fault density.

    Expected components under random thinning (ratio of expectations): TP -> rho TP, n -> rho n,
    FP -> FP + (1 - rho) * near_mass (mask pixels that lose their nearby truth become full FP).
    New faults are not a random thinning of USGS faults, so this only shows how the optimal area
    shrinks with truth density; leaderboard probes must settle the actual operating point.
    """
    rows = []
    d = pooled_df[pooled_df["mode"] == mode]
    for m, g in d.groupby("model"):
        for rho in rhos:
            tp, fp = rho * g.tp_w, g.fp_w + (1 - rho) * g.near_mass
            fn = rho * g.n_truth - tp
            dti = tp / (tp + DTI_ALPHA * fp + DTI_BETA * fn + DTI_EPS)
            i = int(np.argmax(dti.values))
            rows.append({"model": m, "rho": rho, "best_frac": float(g.target_frac.iloc[i]), "best_dti": float(dti.iloc[i]),
                         "dti_at_frac": dict(zip(g.target_frac.round(4), dti.round(5)))})
    return pd.DataFrame(rows)


def segment_recall_at_budget(ctx: P3Context, maps: dict[str, np.ndarray], frac: float,
                             zones: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    """Per-segment recovery of binary top-frac masks (Phase-1 segment attributes)."""
    from .. import geo_eval

    zones = ctx.folds.fold_map if zones is None else zones
    out = {}
    for n, p in maps.items():
        mask = binary_mask(p, ctx.data.valid, zones, frac).astype(np.float32)
        out[n] = geo_eval.segment_recovery(mask, ctx.data, ctx.seg, ctx.seg_tab, ctx.folds.fold_map, threshold=0.5)
    return out


# ---------------------------------------------------------------------------
# figures (one y-axis per panel; colour fixed per model)
# ---------------------------------------------------------------------------
def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_budget_curves(pooled_df: pd.DataFrame, curves: pd.DataFrame, mode: str = "per_zone", logx: bool = True):
    """1) recall vs area, 2) DTI vs area, 3) per-fold DTI vs area (dashed = folds)."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    d = pooled_df[pooled_df["mode"] == mode]
    for m, g in d.groupby("model", sort=False):
        c = model_color(m)
        axs[0].plot(g.target_frac * 100, g.recall_w, color=c, lw=2, marker="o", ms=4, label=m)
        axs[1].plot(g.target_frac * 100, g.dti, color=c, lw=2, marker="o", ms=4, label=m)
        for z, gz in curves[(curves.model == m) & (curves["mode"] == mode)].groupby("zone"):
            axs[2].plot(gz.target_frac * 100, gz.dti, color=c, lw=1.2, ls=(0, (3, 2)), alpha=0.8)
        axs[1].annotate(m, (g.target_frac.iloc[-1] * 100, g.dti.iloc[-1]), fontsize=7, color="#52514e",
                        xytext=(3, 0), textcoords="offset points")
    _style(axs[0], "predicted area (% of evaluated footprint)", "distance-weighted recall", "Recall vs predicted area")
    _style(axs[1], "predicted area (%)", "DTI (pooled over folds)", "DTI vs predicted area")
    _style(axs[2], "predicted area (%)", "DTI per held-out fold", "Fold-specific DTI curves")
    for ax in axs:
        if logx:
            ax.set_xscale("log")
    axs[0].legend(fontsize=8, frameon=False)
    return fig


def fig_dti_vs_scale(fam_table: pd.DataFrame, family: str = "linear"):
    """DTI vs confidence scale s in q = clip(s p) (pooled over folds, from the transform table)."""
    import json

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    t = fam_table[fam_table.family == family].copy()
    t["s"] = [json.loads(x)["s"] for x in t.param]
    for m, g in t.groupby("model", sort=False):
        pc = g.groupby("s").apply(lambda x: x.tp_w.sum() / (x.tp_w.sum() + DTI_ALPHA * x.fp_w.sum() + DTI_BETA * x.fn_w.sum()))
        ax.plot(pc.index, pc.values, color=model_color(m), lw=2, marker="o", ms=4, label=m)
    _style(ax, "confidence scale s  (q = clip(s p, 0, 1))", "DTI (pooled, in-sample per s)", "DTI vs probability scale")
    ax.legend(fontsize=8, frameon=False)
    return fig


def fig_matched_bars(table: pd.DataFrame, metric: str = "dti"):
    import matplotlib.pyplot as plt

    fracs = sorted(table.target_frac.unique())
    models = list(dict.fromkeys(table.model))
    fig, axs = plt.subplots(1, len(fracs), figsize=(3.2 * len(fracs), 3.8), sharey=True, constrained_layout=True)
    for ax, f in zip(np.atleast_1d(axs), fracs):
        g = table[table.target_frac == f].set_index("model").reindex(models)
        ax.bar(range(len(models)), g[metric], color=[model_color(m) for m in models], width=0.7, edgecolor="white", linewidth=2)
        ax.set_xticks(range(len(models)), models, rotation=60, fontsize=7)
        _style(ax, "", metric if f == fracs[0] else "", f"area {f * 100:g}%")
    return fig


def fig_pareto(front: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    for m, g in front.groupby("model", sort=False):
        ax.plot(g.fp_w, g.recall_w, color=model_color(m), lw=1.5, marker="o", ms=4, label=m)
    p = front[front.pareto].sort_values("fp_w")
    ax.plot(p.fp_w, p.recall_w, color="#0b0b0b", lw=1, ls=(0, (2, 2)), label="Pareto front")
    ax.set_xscale("log")
    _style(ax, "false-positive mass FP_w (log)", "distance-weighted recall", "Recovery vs false-positive mass")
    ax.legend(fontsize=8, frameon=False)
    return fig
