"""M6.3: area-budget and score calibration of existing OOF predictions (no retraining).

Every label-dependent choice is cross-fitted: for held-out fold k the parameter is selected on the pooled
components of the other folds (``p3.curves.crossfit_select``) and only then scored on fold k. Each result row
records its selection zones and its held-out zone (``selection_audit``).

Strategies (families of ``p3.transforms`` where they already exist)
--------------------------------------------------------------------
raw                identity (soft map, clipped to [0, 1])                                      label-free
rank               per-fold average-rank quantile as a soft map                                label-free
robust_quantile    per-fold (p - q_lo) / (q_hi - q_lo), clipped                                label-free
linear / temperature   parametric probability rescaling, parameter cross-fitted                 dev labels
isotonic           p -> E[kernel credit] fitted on the other folds' pixels (ML calibration)     dev labels
hard_threshold     binary per-fold top-a mask, a cross-fitted (the protocol's area budget)      dev labels
hard_threshold_rank   the same on the rank map: must equal hard_threshold (invariance check)
absolute_threshold one score cut chosen on the other folds, applied unchanged to fold k         dev labels
                   (``threshold_transfer``: tests whether raw scores transfer across regions)

Why the area budget dominates: for a fixed truth set the DTI-optimal map is binary (``p3.curves`` docstring), so
monotone recalibration cannot change a per-fold binary mask. Probability calibration matters only for soft
exports or a single global cut; both are measured here rather than assumed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..p3.curves import DEFAULT_AREA_FRACS, BinaryCurve, dti_from, pooled, zone_curves
from ..p3.transforms import FoldScorers, compare_families, family_table, quantile_map

LABEL_FREE = ("raw", "rank", "robust_quantile")
DEFAULT_FAMILIES = ("raw", "rank", "robust_quantile", "linear", "temperature", "isotonic", "hard_threshold")


def robust_quantile_map(p: np.ndarray, valid: np.ndarray, zones: np.ndarray, lo: float = 0.05,
                        hi: float = 0.995) -> np.ndarray:
    """Per-zone robust min-max scaling between the ``lo`` and ``hi`` score quantiles (label-free)."""
    out = np.full(p.shape, np.nan, np.float32)
    for z in np.unique(zones[valid & (zones >= 0)]):
        m = valid & (zones == z)
        v = np.nan_to_num(p[m], nan=0.0)
        a, b = np.quantile(v, [lo, hi])
        out[m] = np.clip((v - a) / max(b - a, 1e-9), 0.0, 1.0).astype(np.float32)
    return out


def calibration_table(ctx, model: str, p: np.ndarray, *, families=DEFAULT_FAMILIES, scorers: FoldScorers | None = None,
                      fracs=DEFAULT_AREA_FRACS, rng=None) -> pd.DataFrame:
    """Per-fold DTI components of every (family, parameter); the existing families come from ``family_table``."""
    v, fm, lab = ctx.data.valid, ctx.folds.fold_map, ctx.data.labels
    scorers = scorers or FoldScorers(lab, fm, v)
    known = [f for f in families if f not in ("rank", "robust_quantile")]
    tab = family_table(model, np.where(v, p, np.nan), lab, fm, v, scorers, families=known, rng=rng,
                       area_fracs=fracs) if known else pd.DataFrame()
    extra = []
    u = quantile_map(p, v, fm, "average")
    if "rank" in families:
        extra += scorers.table(u, family="rank", param="{}")
    if "robust_quantile" in families:
        extra += scorers.table(robust_quantile_map(p, v, fm), family="robust_quantile", param="{}")
    if "hard_threshold" in families:                  # invariance check: identical masks from the rank map
        zc = zone_curves(u, lab, fm, v, fracs=fracs, mode="per_zone")
        for r in zc.itertuples():
            extra.append({"family": "hard_threshold_rank", "param": json.dumps({"a": float(r.target_frac)}),
                          "a": float(r.target_frac), "zone": int(r.zone), "tp_w": r.tp_w, "fp_w": r.fp_w,
                          "fn_w": r.fn_w, "mass": r.area_px})
    if extra:
        e = pd.DataFrame(extra)
        e.insert(0, "model", model)
        tab = pd.concat([tab, e], ignore_index=True)
    return tab


def calibration_summary(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per model x family, per-fold choices) with the selection zones of every held-out fold."""
    summ, per = compare_families(table)
    zones = sorted(table.zone.unique())
    per["selection_zones"] = per.zone.map(lambda z: ",".join(str(x) for x in zones if x != z))
    per["heldout_zone"] = per.zone
    per["uses_heldout_labels"] = False                # crossfit_select never reads the held-out zone's rows
    summ["label_free"] = summ.family.isin(LABEL_FREE)
    return summ, per


def invariance_check(table: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """hard_threshold on raw vs on per-fold rank: identical per-zone masks must give identical components."""
    a = table[table.family == "hard_threshold"].set_index(["model", "zone", "param"])[["tp_w", "fp_w", "fn_w"]]
    b = table[table.family == "hard_threshold_rank"].set_index(["model", "zone", "param"])[["tp_w", "fp_w", "fn_w"]]
    j = a.join(b, rsuffix="_rank", how="inner")
    if not len(j):
        return pd.DataFrame()
    j["max_abs_diff"] = np.max(np.abs(j[["tp_w", "fp_w", "fn_w"]].to_numpy() -
                                      j[["tp_w_rank", "fp_w_rank", "fn_w_rank"]].to_numpy()), axis=1)
    out = j.reset_index().groupby("model").max_abs_diff.max().rename("max_abs_diff").reset_index()
    out["status"] = np.where(out.max_abs_diff <= tol * np.maximum(1, j.tp_w.max()), "IDENTICAL",
                             "DIFFERENT (ties: rank averaging changed the mask)")
    return out


# =============================================================================== threshold transfer
def threshold_transfer(ctx, model: str, p: np.ndarray, fracs=DEFAULT_AREA_FRACS, far_km: float = 1.0) -> pd.DataFrame:
    """Absolute score cut vs per-fold area fraction, both chosen on the other folds and applied to fold k.

    A well-calibrated score transfers: the absolute cut chosen on two regions selects a similar area and DTI on the
    third. A poorly calibrated one selects a very different area there; the per-fold area rule is immune to that.
    """
    v, fm, lab = ctx.data.valid, ctx.folds.fold_map, ctx.data.labels
    far = ctx.dist_to_labels_km() > far_km
    curves = {k: BinaryCurve(p, lab, v & (fm == k), far_mask=far) for k in ctx.fold_ids}
    rows = []
    for k in ctx.fold_ids:
        dev = [j for j in ctx.fold_ids if j != k]
        devv = np.sort(np.nan_to_num(p[v & np.isin(fm, dev)], nan=0.0))
        n = len(devv)
        thr = devv[n - np.clip(np.ceil(np.asarray(fracs) * n).astype(int), 1, n)]
        dev_tab = pd.concat([curves[j].at(thr).assign(zone=j, cand=np.arange(len(thr))) for j in dev])
        s = dev_tab.groupby("cand").apply(dti_from)
        best = int(s.idxmax())
        t = float(thr[best])
        held = curves[k].at([t]).iloc[0]
        area_dev = dev_tab[dev_tab.cand == best]
        rows.append(_transfer_row(model, k, dev, "absolute_threshold", {"threshold": t}, float(s.max()),
                                  float(area_dev.area_px.sum() / area_dev.eval_px.sum()), held))
        # per-fold area fraction (the protocol): the same fraction quantile-cut inside fold k
        per = pd.concat([curves[j].at_area(fracs).assign(zone=j) for j in dev])
        sf = per.groupby("target_frac").apply(dti_from)
        f = float(sf.idxmax())
        held = curves[k].at_area([f]).iloc[0]
        rows.append(_transfer_row(model, k, dev, "area_fraction", {"target_frac": f}, float(sf.max()), f, held))
    return pd.DataFrame(rows)


def _transfer_row(model, k, dev, policy, prm, dev_dti, dev_area, held) -> dict:
    return {"model": model, "heldout_fold": k, "selection_zones": ",".join(map(str, dev)), "policy": policy,
            "parameter": json.dumps(prm), "dev_dti": dev_dti, "dev_area_frac": dev_area,
            "heldout_dti": float(held.dti), "heldout_area_frac": float(held.area_frac),
            "area_ratio_heldout_vs_dev": float(held.area_frac) / max(dev_area, 1e-12),
            "heldout_recall_w": float(held.recall_w), "heldout_recall300": float(held.get("recall@300m", np.nan)),
            "heldout_far_share": float(held.get("far_area_km2", np.nan)) / max(float(held.area_km2), 1e-9),
            "uses_heldout_labels": False}


def score_distribution(ctx, maps: dict[str, np.ndarray], qs=(0.5, 0.9, 0.95, 0.99, 0.999)) -> pd.DataFrame:
    """Label-free score quantiles per (model, fold): do the folds see comparable score scales?"""
    rows = []
    for m, p in maps.items():
        for k in ctx.fold_ids:
            x = np.nan_to_num(p[ctx.data.valid & (ctx.folds.fold_map == k)], nan=0.0)
            rows.append({"model": m, "fold": k, "n_px": len(x), "mean": float(x.mean()),
                         **{f"q{q:g}": float(np.quantile(x, q)) for q in qs}})
    return pd.DataFrame(rows)


def area_budget_curves(sc: dict, models: list[str]) -> pd.DataFrame:
    """Long table behind ``area_budget_curves.png``: pooled and per-unit DTI / recall vs area, both tasks."""
    from ..p3 import budget

    g = sc["geo_curves"]
    g = g[g.model.isin(models) & (g["mode"] == "per_zone")]
    out = [g.assign(task="geographic", unit=g.zone.astype(str)),
           budget.pool_models(g).assign(task="geographic", unit="pooled")]
    dc = sc.get("disc_curves", pd.DataFrame())
    if len(dc):
        dc = dc[dc.model.isin(models) & (dc.variant == "model")]
        out += [dc.assign(task="discovery_" + dc.policy, unit=dc.zone.astype(str)),
                pooled(dc, ["model", "policy", "target_frac"]).assign(unit="pooled").pipe(
                    lambda x: x.assign(task="discovery_" + x.policy))]
    keep = ["task", "unit", "model", "target_frac", "area_frac", "dti", "recall_w", "recall@300m", "tp_w", "fp_w",
            "fn_w", "far_area_km2", "area_km2"]
    t = pd.concat(out, ignore_index=True)
    return t[[c for c in keep if c in t.columns]]


def selection_audit(per: pd.DataFrame, strategy_col: str = "family") -> pd.DataFrame:
    """One row per (model, strategy, held-out zone): which zones chose the parameter (never the held-out one)."""
    cols = ["model", strategy_col, "heldout_zone", "selection_zones", "param", "dti_heldout", "uses_heldout_labels"]
    t = per.copy()
    if "heldout_zone" not in t:
        t["heldout_zone"] = t.zone
    t["selection_ok"] = [str(h) not in str(s).split(",") for h, s in zip(t.heldout_zone, t.selection_zones)]
    return t[[c for c in cols if c in t.columns] + ["selection_ok"]]


# =============================================================================== figures
def fig_area_budget_curves(curves: pd.DataFrame, models: list[str], crossfit: dict | None = None):
    """Pooled DTI and recall vs area (geographic, discovery per policy); dashed = per-unit curves."""
    import matplotlib.pyplot as plt

    from .comparative import _colors, _style

    col = _colors(models)
    tasks = list(dict.fromkeys(curves.task))
    fig, axs = plt.subplots(2, len(tasks), figsize=(4.6 * len(tasks), 6.6), squeeze=False, constrained_layout=True)
    for j, t in enumerate(tasks):
        for m in models:
            g = curves[(curves.task == t) & (curves.model == m)]
            pool = g[g.unit == "pooled"]
            axs[0, j].plot(pool.target_frac * 100, pool.dti, color=col[m], lw=1.8, marker="o", ms=3, label=m)
            axs[1, j].plot(pool.target_frac * 100, pool.recall_w, color=col[m], lw=1.8, marker="o", ms=3)
            for _, gu in g[g.unit != "pooled"].groupby("unit"):
                axs[0, j].plot(gu.target_frac * 100, gu.dti, color=col[m], lw=0.8, ls=(0, (3, 2)), alpha=0.7)
            if crossfit and (t, m) in crossfit:
                axs[0, j].axvline(crossfit[(t, m)] * 100, color=col[m], lw=0.7, alpha=0.6)
        for i in range(2):
            axs[i, j].set_xscale("log")
        _style(axs[0, j], "", "DTI", t)
        _style(axs[1, j], "predicted area (%)", "weighted recall", "")
    axs[0, 0].legend(fontsize=7, frameon=False)
    return fig


def fig_calibration_summary(summ: pd.DataFrame, transfer: pd.DataFrame | None = None):
    import matplotlib.pyplot as plt

    from .comparative import _colors, _style

    models = list(dict.fromkeys(summ.model))
    col = _colors(models)
    fams = list(dict.fromkeys(summ.family))
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.32 * len(fams) * len(models) + 2), constrained_layout=True)
    w = 0.8 / max(len(models), 1)
    y = np.arange(len(fams))
    for i, m in enumerate(models):
        g = summ[summ.model == m].set_index("family").reindex(fams)
        axs[0].barh(y + (i - (len(models) - 1) / 2) * w, g.dti_crossfit, height=w, color=col[m], label=m)
    axs[0].set_yticks(y, fams, fontsize=8)
    _style(axs[0], "cross-fitted geographic DTI", "", "Score calibration strategies")
    axs[0].legend(fontsize=7, frameon=False)
    if transfer is not None and len(transfer):
        for m in models:
            g = transfer[transfer.model == m]
            for pol, mk in (("absolute_threshold", "s"), ("area_fraction", "o")):
                x = g[g.policy == pol]
                axs[1].scatter(x.heldout_area_frac * 100, x.heldout_dti, marker=mk, color=col[m], s=30,
                               label=f"{m} {pol}")
        axs[1].set_xscale("log")
        _style(axs[1], "held-out predicted area (%)", "held-out DTI", "Absolute cut vs area fraction transfer")
        axs[1].legend(fontsize=6, frameon=False)
    else:
        axs[1].axis("off")
    return fig
