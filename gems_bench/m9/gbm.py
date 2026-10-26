"""Section 15: XGBoost vs LightGBM vs CatBoost candidate verification on IDENTICAL inputs.

Identical for every library: the candidate table (section 14), feature set, leave-one-zone-out assignment with the
2 km buffer and linked-network exclusion (``m7.ranking.train_rows`` rule), supervision (label 1 / 0; -1 unlabelled
rows dropped, never negatives), sample weights (negatives ``negative_weight``: mapped faults are incomplete), seeds,
and a comparable budget (``n_estimators`` trees of depth ``max_depth``; LightGBM num_leaves = 2^depth - 1;
subsample / colsample / L2 matched where the library exposes them; XGBoost ``min_child_weight`` is in hessian units
and is left at its default: documented, not hidden). When a held-out unit has no other zone of its own task
(PILOT), rows of the other task outside its buffer are used and the audit says so.

Evaluation per (library, feature set): candidate AUC-PR (ROC-AUC reported, never optimised), recall of positive
candidates in the top 10 %, system recall at the gate density when candidates are ordered by the score, false
(label 0) candidate density, rasterised official-style DTI (1-px centerlines, eligible budget, frac chosen on the
other zones; held-out maps saved as ``m9_struct_<library>``), ranking stability (Spearman of held-out scores across
two seeds). Choice: the best pooled cross-fitted DTI on the primary task; every library within ``choose.tolerance``
of it competes on its worst zone (a steadier model beats a marginal aggregate gain). Optional XGBoost LambdaMART
with regional-block groups (``lambdamart.enabled``).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def make_model(lib: str, gc: dict, seed: int):
    d = int(gc["max_depth"])
    if lib == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=int(gc["n_estimators"]), learning_rate=float(gc["learning_rate"]), max_depth=d,
                             subsample=float(gc["subsample"]), colsample_bytree=float(gc["colsample"]),
                             reg_lambda=float(gc["reg_lambda"]), tree_method="hist", random_state=seed, n_jobs=-1,
                             eval_metric="logloss", verbosity=0)
    if lib == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=int(gc["n_estimators"]), learning_rate=float(gc["learning_rate"]),
                                  num_leaves=2 ** d - 1, max_depth=d, min_child_samples=int(gc["min_child"]),
                                  subsample=float(gc["subsample"]), subsample_freq=1, colsample_bytree=float(gc["colsample"]),
                                  reg_lambda=float(gc["reg_lambda"]), random_state=seed, verbose=-1)
    if lib == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(iterations=int(gc["n_estimators"]), learning_rate=float(gc["learning_rate"]), depth=d,
                                  l2_leaf_reg=float(gc["reg_lambda"]), rsm=float(gc["colsample"]),
                                  bootstrap_type="Bernoulli", subsample=float(gc["subsample"]), random_seed=seed,
                                  verbose=False, allow_writing_files=False)
    raise KeyError(lib)


def available(lib: str) -> tuple[bool, str]:
    try:
        __import__(lib)
        return True, "ok"
    except ImportError as e:
        return False, f"not installed ({e})"


def train_mask(table: pd.DataFrame, u, shape, buffer_km: float) -> tuple[np.ndarray, str]:
    """Training rows for held-out unit ``u`` (same task, other zones; else the other task outside the buffer)."""
    from ..m7.ranking import _zone_mask

    ez = _zone_mask(u, shape, buffer_km)
    rr = np.clip(table.r_full.to_numpy().astype(int), 0, shape[0] - 1)
    cc = np.clip(table.c_full.to_numpy().astype(int), 0, shape[1] - 1)
    inside = ez[rr, cc]
    lab = table.label.to_numpy() >= 0
    held = ((table.task == u.task) & (table.zone == u.zone)).to_numpy()
    bad = set(table.loc[inside, "network_id"].astype(str)) if "network_id" in table else set()
    in_bad = table.network_id.astype(str).isin(bad).to_numpy() if "network_id" in table else np.zeros(len(table), bool)
    same = (table.task == u.task).to_numpy() & ~held & ~inside & ~in_bad & lab
    if (table.label.to_numpy()[same] == 1).sum() >= 10:
        return same, "same task, other zones"
    other = ~held & ~inside & ~in_bad & lab
    return other, "cross-task rows outside the buffer (no other zone of this task)"


def loo(T: pd.DataFrame, units: list, feats: list[str], lib: str, gc: dict, shape, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    if lib == "xgbrank":
        return lambdamart(T, units, feats, gc, shape, seed)
    out = np.full(len(T), np.nan)
    audit = []
    for u in units:
        held = ((T.task == u.task) & (T.zone == u.zone)).to_numpy()
        if not held.any():
            continue
        tr, how = train_mask(T, u, shape, float(gc["exclude_buffer_km"]))
        y = T.label.to_numpy()[tr]
        a = {"library": lib, "task": u.task, "zone": u.zone, "n_train": int(tr.sum()), "n_pos": int((y == 1).sum()),
             "n_neg": int((y == 0).sum()), "n_heldout": int(held.sum()), "rows": how, "n_features": len(feats)}
        if a["n_pos"] < 10 or a["n_neg"] < 10 or not feats:
            a["status"] = "too few labelled rows: label-free confidence used"
            out[held] = T.loc[held, "confidence_lf"].to_numpy() if "confidence_lf" in T else 0.0
            audit.append(a)
            continue
        m = make_model(lib, gc, seed)
        X = T.loc[tr, feats].astype(float).to_numpy()
        m.fit(X, y, sample_weight=np.where(y == 1, 1.0, float(gc["negative_weight"])))
        out[held] = m.predict_proba(T.loc[held, feats].astype(float).to_numpy())[:, 1]
        a["status"] = "trained"
        audit.append(a)
    return out, pd.DataFrame(audit)


def lambdamart(T: pd.DataFrame, units: list, feats: list[str], gc: dict, shape, seed: int):
    """XGBoost LambdaMART: relevance = label (1 / 0), query groups = regional blocks of the training rows."""
    from xgboost import XGBRanker

    blk = max(int(round(float(gc["lambdamart"]["group_block_km"]) / 0.1)), 10)
    out = np.full(len(T), np.nan)
    audit = []
    for u in units:
        held = ((T.task == u.task) & (T.zone == u.zone)).to_numpy()
        tr, how = train_mask(T, u, shape, float(gc["exclude_buffer_km"]))
        sub = T[tr].copy()
        sub["_g"] = (sub.r_full // blk).astype(int) * 100000 + (sub.c_full // blk).astype(int)
        sub = sub.sort_values("_g")
        ok = sub.groupby("_g").label.transform(lambda s: s.nunique() > 1)
        sub = sub[ok]
        a = {"library": "xgbrank", "task": u.task, "zone": u.zone, "n_train": len(sub), "rows": how,
             "n_groups": int(sub._g.nunique()) if len(sub) else 0}
        if len(sub) < 50 or a["n_groups"] < 3:
            a["status"] = "too few ranked groups: label-free confidence used"
            out[held] = T.loc[held, "confidence_lf"].to_numpy() if "confidence_lf" in T else 0.0
            audit.append(a)
            continue
        m = XGBRanker(objective="rank:ndcg", n_estimators=int(gc["n_estimators"]), learning_rate=float(gc["learning_rate"]),
                      max_depth=int(gc["max_depth"]), subsample=float(gc["subsample"]),
                      colsample_bytree=float(gc["colsample"]), random_state=seed, tree_method="hist")
        m.fit(sub[feats].astype(float).to_numpy(), sub.label.to_numpy(), group=sub.groupby("_g", sort=False).size().to_numpy())
        out[held] = m.predict(T.loc[held, feats].astype(float).to_numpy())
        a["status"] = "trained"
        audit.append(a)
    return out, pd.DataFrame(audit)


def candidate_metrics(T: pd.DataFrame, score: np.ndarray, lib: str, fset: str) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for (task, zone), g in T.assign(_s=score).groupby(["task", "zone"]):
        lab = g[g.label >= 0]
        y = (lab.label == 1).to_numpy()
        r = {"library": lib, "feature_set": fset, "task": task, "zone": zone, "n": len(g), "n_pos": int(y.sum())}
        if y.any() and (~y).any():
            r["auc_pr"] = float(average_precision_score(y, lab._s))
            r["roc_auc"] = float(roc_auc_score(y, lab._s))
            k = max(int(0.1 * len(g)), 1)
            top = g.nlargest(k, "_s")
            r["recall_top10pct"] = float((top.label == 1).sum() / max((g.label == 1).sum(), 1))
            r["false_share_top10pct"] = float((top.label == 0).mean())
        rows.append(r)
    return rows


def system_recall_at(m9, u, cs, score: np.ndarray, density: float) -> dict:
    from ..m7.protocol import system_hits

    area = max(u.ev.sum() * 0.01 / 100.0, 1e-9)
    order = np.argsort(-np.nan_to_num(score, nan=-1), kind="stable")
    cum = np.cumsum(cs.table.length_km.to_numpy()[order])
    k = int(np.searchsorted(cum, density * area, side="right"))
    q = cs.subset(order[:k]).raster(u.ev.shape) if k else np.zeros(u.ev.shape, np.float32)
    h = system_hits(m9.ctx, u, np.where(u.ev, q, 0.0), m9.sh)
    return {"system_recall": float(h.recovered.mean()) if len(h) else np.nan, "systems_recovered": int(h.recovered.sum())
            if len(h) else 0, "n_systems": int(len(h)), "n_candidates": k}


def raster_rows(m9, u, cs, score: np.ndarray, method: str) -> list[dict]:
    from ..m7.protocol import score_row
    from ..m7.rasterize import predict as raster_predict

    rows = []
    for f in m9.cfg["validation"]["area_fracs"]:
        q, _ = raster_predict(u, cs, np.nan_to_num(score, nan=0.0), {"kind": "centerline"}, float(f), 0, m9.elig)
        rows.append(score_row(u, q, m9.elig(u, 0), m9.pkey(u), method=method, frac=float(f), rep="centerline"))
    return rows


def save_selected(m9, method: str, per: pd.DataFrame, sets: dict, T: pd.DataFrame, col: str) -> None:
    from ..m7.rasterize import predict as raster_predict
    from ..m7.rasterize import save_heldout

    for r in per.itertuples():
        u = m9.unit(f"{r.task}:{int(r.zone)}")
        cs = sets.get(u.key)
        if cs is None:
            continue
        s = T[(T.task == r.task) & (T.zone == r.zone)].sort_values("cand_id")[col].to_numpy()
        q, _ = raster_predict(u, cs, np.nan_to_num(s, nan=0.0), {"kind": "centerline"}, float(r.frac), 0, m9.elig)
        save_heldout(m9.ws, method, u, q, m9.pkey(u))


def run(m9, T: pd.DataFrame, sets: dict) -> dict:
    """Benchmark every available library on every feature set; returns all tables (scores stored in ``T``)."""
    from scipy.stats import spearmanr

    from ..gems3.metric_opt import nested_select
    from .candidates import feature_sets

    gc = m9.cfg["gbm"]
    shape = m9.ctx.data.shape
    units = [m9.unit(k) for k in sets]
    units = [u for u in units if u is not None]
    fsets = feature_sets(T)
    seed = int(m9.cfg["seed"])
    T = T.copy()
    cand, sysr, grid, audit, avail, stab = [], [], [], [], [], []
    dens = float(m9.cfg["tracing"]["oracle"]["gate_density"])
    methods = {"label_free": "confidence_lf"} if "confidence_lf" in T else {}
    libs = list(gc["libraries"]) + (["xgbrank"] if gc["lambdamart"].get("enabled") else [])
    for lib in libs:
        ok, why = available("xgboost" if lib == "xgbrank" else lib)
        avail.append({"library": lib, "available": ok, "note": why})
        if not ok:
            continue
        for fs, feats in fsets.items():
            if lib == "xgbrank" and fs != "all":
                continue
            col = f"score_{lib}_{fs}"
            T[col], a = loo(T, units, feats, lib, gc, shape, seed)
            audit.append(a.assign(feature_set=fs))
            methods[f"{lib}_{fs}"] = col
            if fs == "all":
                s2, _ = loo(T, units, feats, lib, gc, shape, seed + 17)
                ok2 = np.isfinite(T[col]) & np.isfinite(s2)
                stab.append({"library": lib, "feature_set": fs,
                             "seed_spearman": float(spearmanr(T[col][ok2], s2[ok2]).correlation) if ok2.sum() > 10 else np.nan})
    for name, col in methods.items():
        lib, fs = (name.split("_", 1) + ["-"])[:2] if name != "label_free" else ("label_free", "control")
        cand += candidate_metrics(T, T[col].to_numpy(), lib, fs)
        for u in units:
            cs = sets[u.key]
            sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
            sysr.append({"library": lib, "feature_set": fs, "task": u.task, "zone": u.zone,
                         **system_recall_at(m9, u, cs, sub[col].to_numpy(), dens)})
            grid += [{**r, "library": lib, "feature_set": fs} for r in raster_rows(m9, u, cs, sub[col].to_numpy(), name)]
    G = pd.DataFrame(grid)
    summ, per_all = [], []
    for name in methods:
        for task in G.task.unique() if len(G) else []:
            g = G[(G.method == name) & (G.task == task)]
            if not len(g):
                continue
            per, s = nested_select(g, ["frac"])
            if not len(per):                                   # single zone: frac borrowed from the dense channel
                from .validation import borrowed_choice

                ch = borrowed_choice(m9, task)
                fr = np.asarray(sorted(g.frac.unique()))
                f0 = float(fr[np.argmin(np.abs(fr - (ch["frac"] if ch else 0.02)))])
                per = g[np.isclose(g.frac, f0)][["zone", "frac", "tp_w", "fp_w", "fn_w", "n_truth", "dti"]].rename(
                    columns={"dti": "dti_heldout"})
                s = {"dti_crossfit": float((per.tp_w.sum()) / (per.tp_w.sum() + 0.2 * per.fp_w.sum() + 0.8 * per.fn_w.sum()
                                                               + 1e-6)) if len(per) else np.nan,
                     "fold_min": float(per.dti_heldout.min()) if len(per) else np.nan, "selection": "borrowed",
                     "params_all": {"frac": f0}}
            per_all.append(per.assign(method=name, task=task))
            summ.append({"method": name, "task": task, "dti_crossfit": s.get("dti_crossfit"),
                         "fold_min": s.get("fold_min", float(per.dti_heldout.min()) if len(per) else np.nan),
                         "selection": s.get("selection", "nested"),
                         "params_all": json.dumps(s.get("params_all") or {}, default=float)})
    S = pd.DataFrame(summ)
    P = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    choice = choose(m9, S)
    for name in [n for n in methods if n == "label_free" or n.endswith("_all")]:
        mname = f"m9_struct_{name.replace('_all', '')}"
        if len(P):
            save_selected(m9, mname, P[P.method == name], sets, T, methods[name])
    if choice.get("method"):
        save_selected(m9, "m9_structured", P[P.method == choice["method"]], sets, T, methods[choice["method"]])
    return {"table": T, "candidate_metrics": pd.DataFrame(cand), "system_recall": pd.DataFrame(sysr), "grid": G,
            "summary": S, "per_zone": P, "audit": pd.concat(audit, ignore_index=True) if audit else pd.DataFrame(),
            "availability": pd.DataFrame(avail), "stability": pd.DataFrame(stab), "choice": pd.DataFrame([choice]),
            "methods": methods}


def choose(m9, S: pd.DataFrame) -> dict:
    tol = float(m9.cfg["gbm"]["choose"]["tolerance"])
    s = S[(S.task == m9.prim) & S.method.str.endswith("_all")] if len(S) else S
    if not len(s):
        return {"method": None, "reason": "no library produced a primary-task score"}
    best = float(s.dti_crossfit.max())
    near = s[s.dti_crossfit >= best - tol].sort_values(["fold_min", "dti_crossfit"], ascending=False)
    r = near.iloc[0]
    return {"method": r.method, "library": r.method.split("_")[0], "dti_crossfit": float(r.dti_crossfit),
            "fold_min": float(r.fold_min), "best_dti": best, "n_within_tolerance": int(len(near)),
            "reason": f"best pooled DTI within {tol:g}, then the highest worst-zone DTI"}


def fit_full(T: pd.DataFrame, lib: str, feats: list[str], gc: dict, task: str, seed: int):
    """Refit on every labelled row of ``task`` (full-region scoring); None if too few labels."""
    tr = (T.task == task).to_numpy() & (T.label.to_numpy() >= 0)
    y = T.label.to_numpy()[tr]
    if (y == 1).sum() < 10 or (y == 0).sum() < 10:
        return None
    m = make_model(lib, gc, seed)
    m.fit(T.loc[tr, feats].astype(float).to_numpy(), y, sample_weight=np.where(y == 1, 1.0, float(gc["negative_weight"])))
    return m


def importance(model, feats: list[str]) -> pd.DataFrame:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame()
    return pd.DataFrame({"feature": feats, "importance": np.asarray(imp, float)}).sort_values("importance", ascending=False)
