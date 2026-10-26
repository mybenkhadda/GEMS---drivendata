"""Candidate ranking, scored as rasters with the official scorer at matched budgets.

Methods
-------
``evidence``  label-free: mean per-unit rank of the member scores (mean along the line), the number of agreeing
              sources and every |z| evidence column. No training.
``lgbm``      LightGBM binary classifier. For held-out unit z, training rows come from the OTHER units of the same
              task, minus every row whose midpoint lies inside unit z's evaluation zone dilated by
              ``exclude_eval_buffer_km`` (spatial separation; disc splits overlap geographically). Unlabeled rows
              (-1) are dropped; negatives get ``negative_weight`` (mapped faults are incomplete).

Rasterisation: each candidate is drawn as a 1-px line carrying its score (max where lines cross); the top-``frac``
share of the unit's evaluation pixels among candidate pixels is predicted 1 (binary: DTI-optimal for fixed truth).
The area fraction is cross-fitted with ``metric_opt.nested_select``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .candidates import CandidateSet
from .descriptors import feature_columns
from .metric_opt import nested_select
from .official import Unit
from .representation import top_mask


def evidence_score(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.endswith("_mean") and not c.startswith("dist")] + \
           [c for c in df.columns if c.startswith("ev_")] + ["n_agree"]
    cols = [c for c in dict.fromkeys(cols) if c in df and c != "kd_mean"]
    if not cols:
        return np.zeros(len(df))
    r = df.groupby(["task", "zone"])[cols].rank(pct=True)
    return r.mean(axis=1).fillna(0).to_numpy()


def _eval_zone_full(u: Unit, shape, buffer_km: float) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[u.sl] = u.ev
    if buffer_km > 0:
        m = ndimage.distance_transform_edt(~m, sampling=(0.1, 0.1)) <= buffer_km
    return m


def lgbm_scores(table: pd.DataFrame, units: list[Unit], shape, cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Leave-one-unit-out LightGBM scores (one model per held-out unit) + a training audit."""
    import lightgbm as lgb

    rc = cfg["ranker"]
    prm = dict(rc["lgbm"])
    seed = prm.pop("seed", 2026)
    feats = feature_columns(table)
    out = np.full(len(table), np.nan)
    audit = []
    for u in units:
        held = (table.task == u.task) & (table.zone == u.zone)
        if not held.any():
            continue
        ez = _eval_zone_full(u, shape, rc["exclude_eval_buffer_km"])
        inside = ez[table.r_full.to_numpy(), table.c_full.to_numpy()]
        tr = (table.task == u.task) & (table.zone != u.zone) & ~inside & (table.label >= 0)
        n_pos = int((table.label[tr] == 1).sum())
        audit.append({"task": u.task, "zone": u.zone, "n_train": int(tr.sum()), "n_pos": n_pos,
                      "n_dropped_inside_eval": int(((table.task == u.task) & (table.zone != u.zone) & inside).sum()),
                      "n_heldout": int(held.sum()), "features": len(feats)})
        if n_pos < 5 or tr.sum() - n_pos < 5:
            audit[-1]["status"] = "too few labelled rows: evidence score used"
            out[held.to_numpy()] = evidence_score(table[held])
            continue
        y = table.label[tr].to_numpy()
        w = np.where(y == 1, 1.0, rc["negative_weight"])
        model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **prm)
        model.fit(table.loc[tr, feats], y, sample_weight=w)
        out[held.to_numpy()] = model.predict_proba(table.loc[held, feats])[:, 1]
        audit[-1]["status"] = "trained"
    return out, pd.DataFrame(audit)


def fit_full(table: pd.DataFrame, cfg: dict):
    """One model on every labelled row (for the full-region candidate ranking)."""
    import lightgbm as lgb

    rc = cfg["ranker"]
    prm = dict(rc["lgbm"])
    seed = prm.pop("seed", 2026)
    feats = feature_columns(table)
    tr = table.label >= 0
    y = table.label[tr].to_numpy()
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **prm)
    model.fit(table.loc[tr, feats], y, sample_weight=np.where(y == 1, 1.0, rc["negative_weight"]))
    imp = pd.DataFrame({"feature": feats, "gain": model.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False)
    return model, feats, imp


def score_raster(cs: CandidateSet, scores: np.ndarray, shape) -> np.ndarray:
    s = np.full(shape, -np.inf, np.float32)
    for p, v in zip(cs.paths, scores):
        np.maximum.at(s, (p[:, 0], p[:, 1]), np.float32(v))
    return s


def unit_curve(u: Unit, cs: CandidateSet, scores: np.ndarray, fracs, method: str) -> pd.DataFrame:
    """Official components of the ranked candidate raster at each area fraction."""
    s = score_raster(cs, scores, u.ev.shape)
    on = np.isfinite(s) & u.ev
    rows = []
    for f in fracs:
        k = int(np.ceil(f * u.ev.sum()))
        frac_c = min(1.0, k / max(int(on.sum()), 1))
        q = top_mask(np.where(on, s, -np.inf), on, frac_c).astype(np.float32) if on.any() else np.zeros(u.ev.shape, np.float32)
        c = u.components(q)
        rows.append({"task": u.task, "zone": u.zone, "method": method, "frac": float(f), **c.as_dict(),
                     "area_px": int(q.sum()), "eval_px": int(u.ev.sum())})
    return pd.DataFrame(rows)


def evaluate(table: pd.DataFrame, sets: dict, units: list[Unit], shape, cfg: dict) -> dict:
    fracs = cfg["ranker"]["area_fracs"]
    table = table.copy()
    table["score_evidence"] = evidence_score(table)
    lg, audit = lgbm_scores(table, units, shape, cfg)
    table["score_lgbm"] = lg
    rows = []
    for u in units:
        cs = sets.get(u.key)
        sel = ((table.task == u.task) & (table.zone == u.zone)).to_numpy()
        if cs is None or not sel.any():
            continue
        sub = table[sel].set_index("cand_id").reindex(cs.table.cand_id)
        for m in ("evidence", "lgbm"):
            rows.append(unit_curve(u, cs, sub[f"score_{m}"].fillna(-1).to_numpy(), fracs, m))
        rows.append(unit_curve(u, cs, np.ones(len(cs)), fracs, "unranked"))
    curves = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summ = []
    for (task, m), g in curves.groupby(["task", "method"]) if len(curves) else []:
        per, s = nested_select(g, ["frac"])
        summ.append({"task": task, "method": m, "dti_crossfit": s["dti_crossfit"], "dti_insample": s["dti_insample"],
                     "fold_min": s["fold_min"], "frac_all_zones": s["params_all"]["frac"],
                     **{f"dti_zone{r.zone}": r.dti_heldout for r in per.itertuples()}})
    return {"table": table, "curves": curves, "summary": pd.DataFrame(summ), "audit": audit}
