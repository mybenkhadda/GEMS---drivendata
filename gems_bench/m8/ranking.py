"""Section 11: lightweight, spatially validated candidate ranking (controlled ablations, no hyper-parameter search).

Methods (``ranking.ablations``)::

    label_free     required control: per-unit rank mean of label-free geometry + physics + worm + agreement +
                   profile evidence (no labels, no neural support)
    cnn_only       required control: per-unit rank mean of the B0 / B1 support along the line (no training)
    physics_only   LightGBM on geometry + M7 geophysics + M8 physics / worms / agreement / profiles / provenance
    physics_mc     + Monte Carlo stability
    physics_dem    + terrain / 1 m DEM evidence (NaN outside the DEM footprint, with has_dem)
    physics_cnn    + neural support
    all            + Monte Carlo + DEM + neural + network + context

Training: M7's leave-one-zone-out loop (``m7.ranking.loo_scores``): rows of the other zones of the same task with a
PU label (1 / 0; -1 dropped), negatives down-weighted (mapped faults are incomplete); rows inside the held-out zone +
buffer AND every row of a linked network touching it are dropped. Normalisation is per-unit ranking only (label-free)
and LightGBM needs none; the budget fraction is nested. Optional: CatBoost benchmark, PU negative-weight sensitivity.
Rankers output a SCORE per candidate, not a calibrated probability.

Gate 5: the best learned ranker must beat max(label_free, cnn_only) by ``ranking.gate.min_gain_over_controls``
(cross-fitted DTI on the primary task); otherwise model complexity is not increased.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ablation_features, groups

CONTROLS = ("label_free", "cnn_only")


def label_free_score(T: pd.DataFrame) -> np.ndarray:
    from ..m7.cand_features import label_free_score as lf7

    g = groups(T)
    cols = [c for c in g["physics"] if c.endswith("_mean")] + \
           [c for c in g["worms"] if c.endswith(("_persistence", "_near_share", "_orient_stability"))] + \
           [c for c in g["agreement"] if c.endswith("_mean")] + \
           [c for c in g["profiles"] if c.endswith(("_continuity", "_sign_consistency", "_contrast_q75"))]
    cols = [c for c in dict.fromkeys(cols) if c in T]
    base = lf7(T)
    if not cols:
        return base
    r = T.groupby("unit")[cols].rank(pct=True).mean(axis=1).fillna(0).to_numpy()
    return 0.5 * base + 0.5 * r


def run(m8, tables: dict, sets: dict) -> dict:
    from ..gems3.metric_opt import nested_select
    from ..m7 import ranking as RK
    from ..m7.cand_features import neural_score
    from ..m7.rasterize import predict as raster_predict
    from ..m7.rasterize import save_heldout

    rc7 = m8.sh["ranking"]
    rk = m8.cfg["ranking"]
    shape = m8.ctx.data.shape
    fracs = m8.sh["rasterization"]["area_fracs"]
    units = [u for t in ("geo", m8.prim) for u in m8.units_of(t) if u.key in tables]
    T = pd.concat([tables[u.key] for u in units], ignore_index=True)
    T["score_label_free"] = label_free_score(T)
    T["score_cnn_only"] = neural_score(T)
    feats = ablation_features(T)
    audits, methods = [], [a for a in rk["ablations"]]
    for a in [x for x in rk["ablations"] if x not in CONTROLS]:
        f = [c for c in feats[a] if c in T]
        s, au = RK.loo_scores(T, units, f, rc7, shape, neg_w=float(rc7["negative_weight"]))
        T[f"score_{a}"] = s
        audits.append(au.assign(method=a, n_features=len(f)))
        if rc7.get("use_catboost"):
            try:
                s2, au2 = RK.loo_scores(T, units, f, rc7, shape, neg_w=float(rc7["negative_weight"]), catboost=True)
                T[f"score_{a}_catboost"] = s2
                audits.append(au2.assign(method=f"{a}_catboost"))
                methods.append(f"{a}_catboost")
            except ImportError:
                m8.say("s11 catboost not installed: skipped")
    if m8.profile.get("run_pu_sensitivity"):
        f = [c for c in feats["all"] if c in T]
        for w in rk["pu_negative_weights"]:
            s, au = RK.loo_scores(T, units, f, rc7, shape, neg_w=float(w))
            T[f"score_pu_w{w:g}"] = s
            audits.append(au.assign(method=f"pu_w{w:g}"))
            methods.append(f"pu_w{w:g}")
    rows = []
    for u in units:
        cs = sets[u.key]
        sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
        for mth in methods:
            col = f"score_{mth}"
            if col in sub:
                rows += RK.unit_rows(m8.view, u, cs, sub[col].fillna(-1).to_numpy(), mth, fracs)
    curves = pd.DataFrame(rows)
    summ, per_all = [], []
    for (task, mth), g in curves.groupby(["task", "method"]) if len(curves) else []:
        pz, s = nested_select(g, ["frac"])
        summ.append({"task": task, "method": mth, "dti_crossfit": s["dti_crossfit"], "dti_insample": s["dti_insample"],
                     "optimism": s["optimism"], "fold_min": s["fold_min"], "frac_all_zones": s["params_all"]["frac"],
                     **{f"dti_zone{r.zone}": r.dti_heldout for r in pz.itertuples()}})
        per_all.append(pz.assign(task=task, method=mth))
    summary = pd.DataFrame(summ)
    per = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    for r in per.itertuples() if len(per) else []:
        u = m8.unit(f"{r.task}:{r.zone}")
        cs = sets[u.key]
        sc = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")[f"score_{r.method}"]
        q, _ = raster_predict(u, cs, sc.fillna(-1).to_numpy(), {"kind": "centerline"}, float(r.frac), 0, m8.elig)
        save_heldout(m8.ws, f"m8_rank_{r.method}", u, q, m8.pkey(u))
    imp = RK.full_importance(T, {k: v for k, v in feats.items()}, rc7)
    fg = pd.DataFrame([{"group": k, "feature": c} for k, v in groups(T).items() for c in v])
    return {"table": T, "curves": curves, "summary": summary, "per_zone": per,
            "audit": pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(), "importance": imp,
            "feature_groups": fg, "gate": gate5(m8, summary)}


def best_learned(summary: pd.DataFrame, task: str) -> str | None:
    s = summary[(summary.task == task) & ~summary.method.isin(CONTROLS)] if len(summary) else summary
    s = s[~s.method.str.startswith("pu_")] if len(s) else s
    return str(s.sort_values("dti_crossfit", ascending=False).method.iloc[0]) if len(s) else None


def gate5(m8, summary: pd.DataFrame) -> pd.DataFrame:
    g = m8.cfg["ranking"]["gate"]
    prim = m8.prim
    s = summary[summary.task == prim].set_index("method") if len(summary) else pd.DataFrame()
    best = best_learned(summary, prim)
    ctrl = [c for c in CONTROLS if c in s.index]
    if best is None or not ctrl:
        return pd.DataFrame([{"gate": "G5_ranking", "task": prim, "decision": "INCONCLUSIVE",
                              "reason": "no learned ranker or no control evaluated"}])
    ref = max(ctrl, key=lambda c: s.loc[c, "dti_crossfit"])
    gain = float(s.loc[best, "dti_crossfit"] - s.loc[ref, "dti_crossfit"])
    worst = float(s.loc[best, "fold_min"] - s.loc[ref, "fold_min"])
    ok = gain >= float(g["min_gain_over_controls"])
    return pd.DataFrame([{"gate": "G5_ranking", "task": prim, "best_learned": best, "best_control": ref,
                          "dti_best_learned": float(s.loc[best, "dti_crossfit"]),
                          "dti_best_control": float(s.loc[ref, "dti_crossfit"]), "gain": gain,
                          "worst_zone_delta": worst, "decision": "PASS" if ok else "FAIL",
                          "rule": f"gain >= {g['min_gain_over_controls']} over max(label_free, cnn_only)",
                          "profile": m8.profile["name"], "provisional": bool(m8.profile["provisional"])}])


def score_column(m8, summary: pd.DataFrame) -> str:
    """Score used downstream: the best learned ranker if Gate 5 passed, else the better control."""
    g = gate5(m8, summary)
    if len(g) and g.decision.iloc[0] == "PASS":
        return f"score_{g.best_learned.iloc[0]}"
    s = summary[summary.task == m8.prim].set_index("method") if len(summary) else pd.DataFrame()
    ctrl = [c for c in CONTROLS if c in s.index]
    return f"score_{max(ctrl, key=lambda c: s.loc[c, 'dti_crossfit'])}" if ctrl else "score_label_free"
