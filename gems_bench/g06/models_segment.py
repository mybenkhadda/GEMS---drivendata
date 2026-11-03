"""Geological lineament (segment) classifiers and rankers, cross-fitted over validation units.

Tracks:
    S1  geology only   geometry + geology (store means, cross-trace field step) + orientation agreement + permitted
                       known-fault context (+ the G06 geology pixel prediction when available)
    S2  W2-aware       S1 + honest W2 / CNN segment evidence + W2 uncertainty

Cross-fitting (G05.1 / W2 convention): held-out unit u = (scenario s, zone z) is scored by a model trained on the
candidates of the OTHER zones of s, minus candidates whose centroid lies in u's evaluation zone dilated by
``segment_buffer_km``. Targets are the training units' own near-truth classes (evaluation side of THOSE units).

Strictness: for every training unit k the context columns that could carry outer truth are replaced by their
NESTED version (known context without outer z's truth; G06 pixel evidence from the nested fit ``s:z|k``) through
``unit_columns(unit_key, outer)``. A model is ``STRICT_NESTED`` only if (a) it uses no W2 / CNN column (S1),
(b) its candidate pools are label-free and nested (``physics`` / ``g06``), (c) nested columns were available for
every training unit. Otherwise it is ``SCREENING_ONLY`` and is never reported as nested DTI.

``segment_to_raster`` keeps pixel variation: inside candidates score = (1 - lam) pixel rank + lam segment rank,
outside = (1 - lam) pixel rank (no segment mean overwrites a pixel prediction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import LeakageError, assert_label_free_columns
from .datasets import segment_family

STRICT_POOLS = {"physics", "g06"}
MODEL_EVIDENCE_POOLS = {"w2", "cnn"}


def column_family(col: str) -> str:
    if col.startswith(("ev_g06", "on_g06")):
        return "geology_model"
    if col.startswith("on_"):
        return "model_evidence" if col[3:] in MODEL_EVIDENCE_POOLS else "geometry"
    if col == "unc_w2_rank_std":
        return "model_evidence"
    return segment_family(col)


def segment_feature_columns(T: pd.DataFrame, track: str, families=None) -> list[str]:
    allow = set(families or (["geometry", "geology", "orientation", "context_known", "uncertainty",
                              "geology_model"] + (["model_evidence"] if track == "S2" else [])))
    if track == "S1":
        allow.discard("model_evidence")
    cols = [c for c in T.columns if column_family(c) in allow and pd.api.types.is_numeric_dtype(T[c])]
    assert_label_free_columns(cols, f"segment {track} features")
    if track == "S1" and any(column_family(c) == "model_evidence" for c in cols):
        raise LeakageError("S1 (geology only) received W2 / CNN evidence columns")
    return cols


def class_weights(T: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Length x label confidence, then positives and negatives balanced to equal totals."""
    w = np.maximum(T.seg_length_km.to_numpy(float), 0.05) * np.maximum(T.label_conf.to_numpy(float), 0.1)
    half = 0.5 * w.sum()                       # fixed before rescaling (g07.1: classes now end exactly equal)
    for v in (0, 1):
        m = y == v
        if m.any():
            w[m] *= half / max(w[m].sum(), 1e-12)
    return w.astype(np.float32)


# =============================================================================== estimators
def train_segment_classifier(kind: str, X, y, w, params: dict, seed: int):
    if kind == "lightgbm":
        import lightgbm as lgb

        est = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "early_stopping_rounds"},
                                 random_state=int(seed), verbose=-1)
        est.fit(X, y, sample_weight=w)
        return est
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        est = RandomForestClassifier(**params, random_state=int(seed))
        est.fit(np.where(np.isfinite(X), X, np.nan_to_num(med)[None]), y, sample_weight=w)
        est.g06_impute = np.nan_to_num(med)
        return est
    if kind == "catboost":
        from catboost import CatBoostClassifier

        est = CatBoostClassifier(**params, random_seed=int(seed), verbose=False)
        est.fit(X, y, sample_weight=w)
        return est
    raise KeyError(kind)


def train_segment_ranker(X, y, groups: np.ndarray, params: dict, seed: int):
    """LightGBM lambdarank; ``groups`` = query id per row (rows must be sorted by group)."""
    import lightgbm as lgb

    order = np.argsort(groups, kind="stable")
    _, sizes = np.unique(groups[order], return_counts=True)
    est = lgb.LGBMRanker(**params, random_state=int(seed), verbose=-1)
    est.fit(X[order], y[order].astype(int), group=sizes)
    return est


def _predict(est, X) -> np.ndarray:
    if hasattr(est, "g06_impute"):
        X = np.where(np.isfinite(X), X, est.g06_impute[None])
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.predict(X)


# =============================================================================== cross-fitting
def crossfit_segment_model(exp_id: str, exp_cfg: dict, D: pd.DataFrame, zone_masks: dict, params: dict, seed: int,
                           unit_columns=None, pools_used=(), progress=print) -> tuple[pd.DataFrame, dict]:
    """Held-out segment scores of every unit in ``D`` (features + labels merged on unit / label).

    ``zone_masks``: {unit key: full-grid evaluation zone dilated by the buffer};
    ``unit_columns(unit_key, outer)``: DataFrame [label + override columns] - nested context for training units
    (outer = the held-out zone) and the held-out unit's own context (outer = None); None = no override."""
    track, kind = exp_cfg["track"], exp_cfg["model"]
    rows, info = [], {"units": {}, "nested_complete": True}
    base_cols = None
    for key in sorted(D.unit.unique()):
        sc, z = key.split(":")
        z = int(z)
        test = D[D.unit == key].copy()
        tr_units = sorted(u for u in D.unit.unique() if u.split(":")[0] == sc and int(u.split(":")[1]) != z)
        parts = []
        for k in tr_units:
            Tk = D[D.unit == k].copy()
            if unit_columns is not None:
                ov = unit_columns(k, z)
                if ov is None:
                    info["nested_complete"] = False
                else:
                    Tk = _override(Tk, ov)
            parts.append(Tk)
        if not parts:
            continue
        TR = pd.concat(parts, ignore_index=True)
        if unit_columns is not None:
            ov = unit_columns(key, None)
            if ov is not None:
                test = _override(test, ov)
        zm = zone_masks[key]
        inside = zm[TR.r_full.to_numpy().astype(int), TR.c_full.to_numpy().astype(int)]
        TR = TR[~inside & TR.target.notna()]
        y = TR.target.to_numpy(int)
        if np.unique(y).size < 2:
            info["units"][key] = {"status": "FAILED", "detail": "single class in training rows"}
            continue
        cols = segment_feature_columns(pd.concat([TR, test], ignore_index=True), track, exp_cfg.get("families"))
        base_cols = base_cols or cols
        X = TR[cols].to_numpy(np.float32)
        w = class_weights(TR, y)
        if kind == "ranker":
            est = train_segment_ranker(X, y, TR.unit.astype("category").cat.codes.to_numpy(), params, seed)
        else:
            est = train_segment_classifier(kind, X, y, w, params, seed)
        s = _predict(est, test[cols].to_numpy(np.float32))
        rows.append(pd.DataFrame({"unit": key, "label": test.label.to_numpy(), "score": s.astype(np.float32),
                                  "experiment": exp_id}))
        info["units"][key] = {"status": "COMPLETE", "n_train": int(len(TR)), "n_pos": int(y.sum()),
                              "n_test": int(len(test)), "train_units": tr_units, "n_features": len(cols)}
        if hasattr(est, "feature_importances_"):
            info["units"][key]["importance"] = dict(zip(cols, np.asarray(est.feature_importances_, float).tolist()))
        progress and progress(f"  {exp_id} {key}: train {len(TR)} rows ({int(y.sum())} pos), {len(test)} scored")
    pools_ok = set(pools_used) <= STRICT_POOLS
    strict = track == "S1" and pools_ok and unit_columns is not None and info["nested_complete"]
    info["strictness"] = "STRICT_NESTED" if strict else "SCREENING_ONLY"
    info["strictness_reason"] = ("S1 + label-free nested pools + nested context for every training unit" if strict
                                 else f"track {track}; pools {sorted(pools_used)} (strict needs subset of "
                                      f"{sorted(STRICT_POOLS)}); nested context complete: "
                                      f"{info['nested_complete'] and unit_columns is not None}")
    info["features"] = base_cols or []
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["unit", "label", "score",
                                                                               "experiment"])), info


def _override(T: pd.DataFrame, ov: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ov.columns if c not in ("unit", "label", "outer")]
    out = T.drop(columns=[c for c in cols if c in T.columns]).merge(ov[["label"] + cols], on="label", how="left")
    return out


# =============================================================================== maps
def segment_to_raster(lab: np.ndarray, scores: pd.DataFrame, pixel_rank: np.ndarray | None, ev: np.ndarray,
                      lam: float = 0.5) -> np.ndarray:
    """Segment-assisted score map that preserves the pixel ranking inside and outside the candidates."""
    from ..gems3.representation import rank01

    lab = np.asarray(lab, np.int32)
    n = int(lab.max())
    seg = np.zeros(n + 1, np.float32)
    if n and len(scores):
        r = pd.Series(scores.score.to_numpy(), index=scores.label.to_numpy())
        rk = r.rank(pct=True)
        seg[rk.index.to_numpy().astype(int)] = rk.to_numpy(np.float32)
    s_map = seg[lab]
    if pixel_rank is None:
        raise ValueError("segment_to_raster needs a pixel score (a constant background would make the top-k "
                         "selection pick ties arbitrarily)")
    pr = rank01(pixel_rank, ev)
    out = (1.0 - lam) * pr + lam * s_map
    return np.where(ev, out, np.nan).astype(np.float32)
