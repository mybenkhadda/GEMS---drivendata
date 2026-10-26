"""Section 11: candidate ranking (lightweight, controlled ablations; no hyper-parameter search).

Methods::

    label_free               required control: rank mean of geometry / geophysics / topography evidence (no labels)
    neural_support           rank mean of the B0 / B1 support along the line (no training)
    geom_geophys             LightGBM on geometry + geophysics
    geom_geophys_dem         + topography (100 m terrain and 1 m DEM where covered; NaN elsewhere = missing)
    geom_geophys_dem_neural  + neural support + network context
    plus_completion          the previous feature set on a pool that also contains the section-08 completion
                             hypotheses (map-completion candidates)

Training (leave one zone out, per task): rows of the other zones of the same task with a label (1 / 0; -1 =
unlabelled is dropped, never a negative); negatives weighted ``negative_weight`` (mapped faults are incomplete:
positive-unlabelled data). Spatial separation: rows whose midpoint lies within the held-out evaluation zone +
``exclude_eval_buffer_km``, AND every row of a linked network (``network_id``) that touches that buffer, are dropped
(linked fragments never straddle train and test). Optional PU sensitivity: ``pu_negative_weights``.

Scoring: candidates rasterised as 1-px centerlines, top-``frac`` of eligible pixels (``rasterize``), frac chosen on
the other zones (``nested_select``). Section 12 then explores representations of the best rankers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.metric_opt import nested_select
from .cand_features import ablation_features, label_free_score, neural_score
from .protocol import score_row
from .rasterize import predict as raster_predict
from .rasterize import save_heldout


def _zone_mask(u, shape, buffer_km: float) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[u.sl] = u.ev
    if buffer_km > 0:
        m = ndimage.distance_transform_edt(~m, sampling=(0.1, 0.1)) <= buffer_km
    return m


def train_rows(table: pd.DataFrame, u, shape, buffer_km: float) -> tuple[np.ndarray, int]:
    """Boolean index of the training rows for held-out unit ``u`` and the number dropped by network grouping."""
    ez = _zone_mask(u, shape, buffer_km)
    rr = np.clip(table.r_full.to_numpy().astype(int), 0, shape[0] - 1)
    cc = np.clip(table.c_full.to_numpy().astype(int), 0, shape[1] - 1)
    inside = ez[rr, cc]
    same = (table.task == u.task).to_numpy()
    other = same & (table.zone != u.zone).to_numpy()
    bad_nets = set(table.loc[same & inside, "network_id"].astype(str)) if "network_id" in table else set()
    in_bad = table.network_id.astype(str).isin(bad_nets).to_numpy() if "network_id" in table else np.zeros(len(table), bool)
    tr = other & ~inside & ~in_bad & (table.label.to_numpy() >= 0)
    return tr, int((other & ~inside & in_bad).sum())


def _model(rc: dict, catboost: bool):
    if catboost:
        from catboost import CatBoostClassifier

        p = dict(rc["catboost"])
        return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"],
                                  random_seed=p.get("seed", 2026), verbose=False)
    import lightgbm as lgb

    p = dict(rc["lgbm"])
    seed = p.pop("seed", 2026)
    return lgb.LGBMClassifier(random_state=seed, verbose=-1, **p)


def loo_scores(table: pd.DataFrame, units: list, feats: list[str], rc: dict, shape, *, neg_w: float,
               catboost: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    out = np.full(len(table), np.nan)
    audit = []
    for u in units:
        held = ((table.task == u.task) & (table.zone == u.zone)).to_numpy()
        if not held.any():
            continue
        tr, n_net = train_rows(table, u, shape, float(rc["exclude_eval_buffer_km"]))
        y = table.label.to_numpy()[tr]
        n_pos = int((y == 1).sum())
        a = {"task": u.task, "zone": u.zone, "n_train": int(tr.sum()), "n_pos": n_pos, "n_neg": int((y == 0).sum()),
             "n_dropped_network_grouping": n_net, "n_heldout": int(held.sum()), "n_features": len(feats),
             "negative_weight": neg_w}
        if n_pos < 10 or (y == 0).sum() < 10 or not feats:
            a["status"] = "too few labelled rows: label-free score used"
            out[held] = label_free_score(table[held])
            audit.append(a)
            continue
        m = _model(rc, catboost)
        X = table.loc[tr, feats].astype(float)
        m.fit(X, y, sample_weight=np.where(y == 1, 1.0, neg_w))
        out[held] = m.predict_proba(table.loc[held, feats].astype(float))[:, 1]
        a["status"] = "trained"
        audit.append(a)
    return out, pd.DataFrame(audit)


def unit_rows(m7, u, cs, scores: np.ndarray, method: str, fracs, rep=None) -> list[dict]:
    rows = []
    rep = rep or {"kind": "centerline"}
    for f in fracs:
        q, _ = raster_predict(u, cs, scores, rep, float(f), 0, m7.elig)
        rows.append(score_row(u, q, m7.elig(u, 0), m7.pkey(u), method=method, frac=float(f), rep=rep["kind"]))
    return rows


def run(m7, tables: dict, sets: dict, comp_sets: dict | None = None) -> dict:
    """``tables`` = {unit key: feature table}; ``sets`` = {unit key: CandidateSet} (same row order as the table);
    ``comp_sets`` = {unit key: (CandidateSet, table)} of the pool that includes the completion hypotheses."""
    cfg = m7.cfg
    rc = cfg["ranking"]
    shape = m7.ctx.data.shape
    fracs = cfg["rasterization"]["area_fracs"]
    tasks = ["geo", cfg["evaluation"]["primary_task"]]
    units = [u for t in tasks for u in m7.units_of(t) if u.key in tables]
    T = pd.concat([tables[u.key] for u in units], ignore_index=True)
    T["score_label_free"] = label_free_score(T)
    T["score_neural_support"] = neural_score(T)
    feats = ablation_features(T)
    audits, imps = [], []
    methods = [a for a in rc["ablations"] if a != "plus_completion"]
    for a in methods:
        if a in ("label_free", "neural_support"):
            continue
        s, au = loo_scores(T, units, [f for f in feats[a] if f in T], rc, shape, neg_w=float(rc["negative_weight"]),
                           catboost=False)
        T[f"score_{a}"] = s
        audits.append(au.assign(method=a))
        if rc.get("use_catboost"):
            try:
                s2, au2 = loo_scores(T, units, [f for f in feats[a] if f in T], rc, shape,
                                     neg_w=float(rc["negative_weight"]), catboost=True)
                T[f"score_{a}_catboost"] = s2
                audits.append(au2.assign(method=f"{a}_catboost"))
                methods.append(f"{a}_catboost")
            except ImportError:
                print("[m7] s11 catboost not installed: skipped")
    if rc.get("run_pu_sensitivity"):
        a = "geom_geophys_dem_neural"
        for w in rc["pu_negative_weights"]:
            s, au = loo_scores(T, units, [f for f in feats[a] if f in T], rc, shape, neg_w=float(w))
            T[f"score_pu_w{w:g}"] = s
            audits.append(au.assign(method=f"pu_w{w:g}"))
            methods.append(f"pu_w{w:g}")
    rows = []
    for u in units:
        cs = sets[u.key]
        sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
        for mth in methods:
            col = f"score_{mth}"
            if col not in sub:
                continue
            rows += unit_rows(m7, u, cs, sub[col].fillna(-1).to_numpy(), mth, fracs)
    CT = None
    if "plus_completion" in rc["ablations"] and comp_sets:
        CT = pd.concat([comp_sets[u.key][1] for u in units if u.key in comp_sets], ignore_index=True)
        CT["score_label_free"] = label_free_score(CT)
        f = [x for x in ablation_features(CT)["plus_completion"] if x in CT]
        s, au = loo_scores(CT, [u for u in units if u.key in comp_sets], f, rc, shape, neg_w=float(rc["negative_weight"]))
        CT["score_plus_completion"] = s
        audits.append(au.assign(method="plus_completion"))
        for u in units:
            if u.key not in comp_sets:
                continue
            cs, _ = comp_sets[u.key]
            sub = CT[(CT.task == u.task) & (CT.zone == u.zone)].sort_values("cand_id")
            rows += unit_rows(m7, u, cs, sub.score_plus_completion.fillna(-1).to_numpy(), "plus_completion", fracs)
    curves = pd.DataFrame(rows)
    summ, per = [], []
    for (task, mth), g in curves.groupby(["task", "method"]) if len(curves) else []:
        pz, s = nested_select(g, ["frac"])
        summ.append({"task": task, "method": mth, "dti_crossfit": s["dti_crossfit"], "dti_insample": s["dti_insample"],
                     "optimism": s["optimism"], "fold_min": s["fold_min"], "frac_all_zones": s["params_all"]["frac"],
                     **{f"dti_zone{r.zone}": r.dti_heldout for r in pz.itertuples()}})
        per.append(pz.assign(task=task, method=mth))
    summary = pd.DataFrame(summ)
    per = pd.concat(per, ignore_index=True) if per else pd.DataFrame()
    # held-out maps of every ranker (its own per-zone frac) for section 13
    for r in per.itertuples() if len(per) else []:
        u = m7.unit(f"{r.task}:{r.zone}")
        if r.method == "plus_completion":
            cs = comp_sets[u.key][0]
            sc = CT[(CT.task == u.task) & (CT.zone == u.zone)].sort_values("cand_id").score_plus_completion
        else:
            cs = sets[u.key]
            sc = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")[f"score_{r.method}"]
        q, _ = raster_predict(u, cs, sc.fillna(-1).to_numpy(), {"kind": "centerline"}, float(r.frac), 0, m7.elig)
        save_heldout(m7.ws, f"rank_{r.method}", u, q, m7.pkey(u))
    importance = full_importance(T, feats, rc)
    return {"table": T, "completion_table": CT if CT is not None else pd.DataFrame(), "curves": curves,
            "summary": summary, "per_zone": per,
            "audit": pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(), "importance": importance}


def full_importance(T: pd.DataFrame, feats: dict, rc: dict) -> pd.DataFrame:
    """Gain importance of one model per ablation fitted on every labelled row of the primary pool (descriptive)."""
    rows = []
    lab = T[T.label >= 0]
    if (lab.label == 1).sum() < 10:
        return pd.DataFrame()
    for a, f in feats.items():
        f = [c for c in f if c in lab]
        if not f or a == "plus_completion":
            continue
        m = _model(rc, False)
        y = lab.label.to_numpy()
        m.fit(lab[f].astype(float), y, sample_weight=np.where(y == 1, 1.0, float(rc["negative_weight"])))
        rows += [{"method": a, "feature": c, "gain": float(g)} for c, g in zip(f, m.booster_.feature_importance("gain"))]
    return pd.DataFrame(rows).sort_values(["method", "gain"], ascending=[True, False]) if rows else pd.DataFrame()


def fit_full(T: pd.DataFrame, method: str, rc: dict, task: str):
    """One model on every labelled row of ``task`` (full-region ranking for submission); returns (model, features)."""
    f = [c for c in ablation_features(T).get(method, []) if c in T]
    lab = T[(T.task == task) & (T.label >= 0)]
    m = _model(rc, False)
    y = lab.label.to_numpy()
    m.fit(lab[f].astype(float), y, sample_weight=np.where(y == 1, 1.0, float(rc["negative_weight"])))
    return m, f
