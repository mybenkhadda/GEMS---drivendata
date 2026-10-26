"""Milestone C gate: do the cached 1 m DEM aggregates add held-out evidence beyond B0 / B1? (no download)

Scaling the DEM pipeline (dem.download / dem.morphology) is justified only if, on DEM-covered pixels of each
held-out geographic fold, a small LightGBM on [rank B0, rank B1] + DEM features beats the same model on
[rank B0, rank B1] alone (average precision of "within 200 m of a mapped fault" vs "farther than 1 km").
Training pixels come from the other folds' covered area. Coverage and the covered share of every fold are
reported, because DEM availability must not act as a geographic shortcut: the comparison never leaves the
covered footprint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dem_feature_names(ctx, cfg: dict) -> list[str]:
    names = cfg["dem_gate"].get("features") or [n for n in ctx.store.specs if n.startswith("dem") and n != "dem:coverage"]
    return [n for n in names if n in ctx.store.specs]


def run(ctx, models: list[str], cfg: dict) -> dict[str, pd.DataFrame]:
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    from ..dem.benchmark import coverage_mask
    from .representation import rank01

    dg = cfg["dem_gate"]
    covered = coverage_mask(ctx, dg["min_coverage"]) & ctx.data.valid
    feats = dem_feature_names(ctx, cfg)
    if not feats:
        raise FileNotFoundError("no cached DEM features (dem:*) in the feature store: build the DEM mosaic first")
    fm = ctx.folds.fold_map
    d = ctx.dist_to_labels_km()
    near, far = d <= 0.2, d >= 1.0
    rng = np.random.default_rng(2026)
    base = {}
    for m in models:
        r = np.zeros(fm.shape, np.float32)
        for k in ctx.fold_ids:
            z = ctx.data.valid & (fm == k)
            r[z] = rank01(np.asarray(ctx.oof(m)), z)[z]
        base[m] = r
    dem = {n: np.asarray(ctx.store.get(n), np.float32) for n in feats}
    cov_rows, rows = [], []
    for k in ctx.fold_ids:
        v = ctx.data.valid & (fm == k)
        cov_rows.append({"fold": k, "covered_frac": float(covered[v].mean()) if v.any() else np.nan,
                         "covered_near_px": int((covered & v & near).sum())})
    lab_px = covered & (near | far) & (fm >= 0)
    idx = np.flatnonzero(lab_px.ravel())
    idx = rng.choice(idx, min(len(idx), int(dg["sample_px"])), replace=False) if len(idx) else idx
    y = near.ravel()[idx].astype(int)
    folds = fm.ravel()[idx]
    Xb = np.column_stack([base[m].ravel()[idx] for m in models])
    Xd = np.column_stack([Xb] + [dem[n].ravel()[idx] for n in feats])
    for k in ctx.fold_ids:
        tr, te = folds != k, folds == k
        if y[te].sum() < 10 or y[tr].sum() < 10:
            rows.append({"fold": k, "status": "too few covered positives"})
            continue
        res = {"fold": k, "n_train": int(tr.sum()), "n_test": int(te.sum()), "pos_test": int(y[te].sum())}
        for name, X in (("b0_b1", Xb), ("b0_b1_dem", Xd)):
            mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=2026)
            mdl.fit(X[tr], y[tr])
            res[f"ap_{name}"] = float(average_precision_score(y[te], mdl.predict_proba(X[te])[:, 1]))
        res["ap_gain"] = res["ap_b0_b1_dem"] - res["ap_b0_b1"]
        res["status"] = "ok"
        rows.append(res)
    res = pd.DataFrame(rows)
    ok = res[res.status == "ok"] if len(res) else res
    gain = float(ok.ap_gain.mean()) if len(ok) else np.nan
    worst = float(ok.ap_gain.min()) if len(ok) else np.nan
    dec = pd.DataFrame([{"mean_ap_gain": gain, "worst_fold_gain": worst, "n_features": len(feats),
                         "decision": "CONTINUE" if np.isfinite(gain) and gain >= dg["min_ap_gain"] and worst > 0 else "STOP",
                         "rule": f"mean held-out AP gain >= {dg['min_ap_gain']} and positive in every fold"}])
    return {"coverage": pd.DataFrame(cov_rows), "pilot_results": res, "decision": dec,
            "features": pd.DataFrame({"feature": feats})}
