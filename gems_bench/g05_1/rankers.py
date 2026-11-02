"""Stage G: labels and residual rankers on top of the frozen W2 ranking. Pure (numpy / pandas / LightGBM).

Labels (training units only; the held-out unit's truth never enters a fit):
  positive       y_near_truth >= labels.positive_share
  ambiguous      labels.ambiguous_share <= y_near_truth < positive_share   -> excluded (weight 0)
  hard negative  y_near_truth < ambiguous_share and W2 unit rank >= labels.hard_negative_w2_q
  unlabelled     everything else: PU convention, down-weighted (labels.unlabeled_weight)
Sample weight = class weight x segment length (px).

Models (score = W2 logit + lambda x correction; lambda from ``rankers.lambda_grid`` chosen on INNER training zones):
  G0  frozen W2 (segment W2 unit rank; no fit)                      - the matched-construction reference
  G1  regularised residual LightGBM (binary, init_score = W2 logit), registry features without graph
  G2  pairwise LightGBM ranker (lambdarank, groups = pair_group_km blocks), correction standardised
  G3  gated G1: correction only where W2 rank <= gate.w2_q_max AND evidence rank >= gate.evidence_q_min
  G4  region-aware G1: spatial domain as a categorical feature, doubled regularisation (UNAVAILABLE without domains)
  G5  graph-informed G1: registry + Stage-F graph features
  S1  STRICT label-free fixed blend: (1 - b) W2 rank + b evidence rank, b = rankers.strict_fixed_blend (predeclared)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEARNED = ("G1", "G2", "G3", "G4", "G5")
EVIDENCE_COLUMNS = ("g_support", "disp_rtp_excess", "disp_iso_grav_anom_excess", "mean_pf_persistence",
                    "mean_mg_cross", "mean_g02c", "mean_g02a", "mean_g01_persistence", "g_regional_agreement")


# =============================================================================== labels
def make_labels(T: pd.DataFrame, lc: dict) -> pd.DataFrame:
    """(y, w, cls) per row from ``y_near_truth`` and the W2 unit rank."""
    s = T.y_near_truth.to_numpy(float)
    r = T.w2_unit_rank.to_numpy(float) if "w2_unit_rank" in T else np.full(len(T), np.nan)
    pos = s >= float(lc["positive_share"])
    amb = (s >= float(lc["ambiguous_share"])) & ~pos
    hard = ~pos & ~amb & (np.nan_to_num(r, nan=0.0) >= float(lc["hard_negative_w2_q"]))
    cls = np.where(pos, "positive", np.where(amb, "ambiguous", np.where(hard, "hard_negative", "unlabeled")))
    cw = np.where(pos, 1.0, np.where(amb, 0.0, np.where(hard, float(lc["hard_negative_weight"]),
                                                         float(lc["unlabeled_weight"]))))
    L = T.n_px.to_numpy(float) if "n_px" in T else np.ones(len(T))
    return pd.DataFrame({"y": pos.astype(np.int8), "w": cw * L, "cls": cls}, index=T.index)


def logit(p, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(np.nan_to_num(np.asarray(p, float), nan=0.5), eps, 1 - eps)
    return np.log(p / (1 - p))


def unit_rank_by(T: pd.DataFrame, col: str) -> np.ndarray:
    """Average-rank ECDF of ``col`` inside each unit (label-free)."""
    out = np.full(len(T), np.nan)
    v = T[col].to_numpy(float) if col in T else np.full(len(T), np.nan)
    for _, ix in T.groupby("unit").indices.items():
        x = v[ix]
        ok = np.isfinite(x)
        if ok.any():
            out[ix[ok]] = pd.Series(x[ok]).rank(method="average").to_numpy() / ok.sum()
    return out


def evidence_rank(T: pd.DataFrame) -> np.ndarray:
    """Unit rank of the mean of the available label-free evidence ranks (EVIDENCE_COLUMNS)."""
    cols = [c for c in EVIDENCE_COLUMNS if c in T]
    if not cols:
        return np.full(len(T), np.nan)
    R = np.column_stack([unit_rank_by(T, c) for c in cols])
    with np.errstate(invalid="ignore"):
        m = np.nanmean(R, 1)
    tmp = T[["unit"]].assign(_e=m)
    return unit_rank_by(tmp, "_e")


def strict_blend(T: pd.DataFrame, b: float) -> np.ndarray:
    """S1: (1 - b) W2 unit rank + b evidence rank; W2 alone where no evidence exists (label-free, no fit)."""
    w = T.w2_unit_rank.to_numpy(float)
    e = evidence_rank(T)
    return np.where(np.isfinite(e), (1 - float(b)) * w + float(b) * e, w)


# =============================================================================== LightGBM fits
def _params(p: dict, seed: int, strong: bool = False) -> dict:
    q = {"n_estimators": int(p["n_estimators"]), "learning_rate": float(p["learning_rate"]),
         "num_leaves": int(p["num_leaves"]), "min_child_samples": int(p["min_child_samples"]),
         "subsample": float(p["subsample"]), "subsample_freq": 1, "colsample_bytree": float(p["colsample_bytree"]),
         "reg_lambda": float(p["reg_lambda"]), "random_state": int(seed), "n_jobs": -1, "verbose": -1,
         "deterministic": True, "force_row_wise": True}
    if strong:
        q["min_child_samples"] *= 2
        q["reg_lambda"] *= 2
    return q


def fit_residual(X: pd.DataFrame, y, w, base, p: dict, seed: int, strong: bool = False, cat=None):
    from lightgbm import LGBMClassifier

    m = LGBMClassifier(**_params(p, seed, strong))
    m.fit(X, np.asarray(y), sample_weight=np.asarray(w, float), init_score=np.asarray(base, float),
          categorical_feature=cat if cat else "auto")
    return m


def fit_pairwise(X: pd.DataFrame, y, groups, base, p: dict, seed: int):
    from lightgbm import LGBMRanker

    o = np.argsort(np.asarray(groups), kind="stable")
    g = np.asarray(groups)[o]
    _, sizes = np.unique(g, return_counts=True)
    m = LGBMRanker(objective="lambdarank", **_params(p, seed))
    m.fit(X.iloc[o], np.asarray(y).astype(int)[o], group=sizes, init_score=np.asarray(base, float)[o])
    return m


def correction(m, X: pd.DataFrame) -> np.ndarray:
    """Raw model margin WITHOUT the init score (LightGBM never adds init_score at predict time)."""
    return np.asarray(m.predict(X, raw_score=True), float)


def block_groups(T: pd.DataFrame, km: float) -> np.ndarray:
    px = max(int(round(float(km) / 0.1)), 1)
    return (T.unit.astype(str) + "|" + (T.row_full // px).astype(int).astype(str) + "|" +
            (T.col_full // px).astype(int).astype(str)).to_numpy()


def weighted_ap(score, y, w) -> float:
    from ..geomoe.gating import weighted_ap as wap

    return wap(score, y, w)


def gate_mask(T: pd.DataFrame, gc: dict) -> np.ndarray:
    w = T.w2_unit_rank.to_numpy(float)
    e = evidence_rank(T)
    return (np.nan_to_num(w, nan=1.0) <= float(gc["w2_q_max"])) & (np.nan_to_num(e, nan=0.0) >= float(gc["evidence_q_min"]))


def model_spec(name: str, R: pd.DataFrame, feats_nograph: list, feats_graph: list) -> dict | None:
    """Feature set / options of one learned model; None when its inputs are unavailable."""
    if name in ("G1", "G3"):
        return {"features": feats_nograph, "kind": "residual", "gated": name == "G3"}
    if name == "G2":
        return {"features": feats_nograph, "kind": "pairwise"}
    if name == "G4":
        if "domain" not in R or not np.isfinite(R.domain.to_numpy(float)).any():
            return None
        return {"features": [*feats_nograph, "domain"] if "domain" not in feats_nograph else feats_nograph,
                "kind": "residual", "strong": True, "cat": ["domain"]}
    if name == "G5":
        return {"features": feats_graph, "kind": "residual"}
    raise KeyError(name)


def _fit_predict(spec: dict, Tr: pd.DataFrame, Lr: pd.DataFrame, Th: pd.DataFrame, rk: dict, seed: int):
    f = spec["features"]
    use = Lr.w.to_numpy() > 0
    Xtr, Xh = Tr.loc[use, f].astype(float), Th[f].astype(float)
    if spec.get("cat"):
        for c in spec["cat"]:
            Xtr[c] = Xtr[c].fillna(-1).astype(int).astype("category")
            Xh[c] = pd.Categorical(Xh[c].fillna(-1).astype(int), categories=Xtr[c].cat.categories)
    base_tr = logit(Tr.w2_unit_rank.to_numpy()[use])
    if spec["kind"] == "pairwise":
        keep = Lr.cls.to_numpy()[use] != "ambiguous"
        m = fit_pairwise(Xtr[keep], Lr.y.to_numpy()[use][keep], block_groups(Tr.loc[use][keep], rk["pair_group_km"]),
                         base_tr[keep], rk["lgbm"], seed)
        ctr = correction(m, Xtr)
        sd = float(np.std(ctr)) or 1.0
        return m, correction(m, Xh) / sd
    m = fit_residual(Xtr, Lr.y.to_numpy()[use], Lr.w.to_numpy()[use], base_tr, rk["lgbm"], seed,
                     bool(spec.get("strong")), spec.get("cat"))
    return m, correction(m, Xh)


def choose_lambda(spec: dict, T: pd.DataFrame, L: pd.DataFrame, rk: dict, gc: dict, seed: int) -> dict:
    """lambda maximising the mean inner-zone weighted AP (leave-one-TRAINING-zone-out). Fallback (fewer than two
    training zones): the middle of the grid, flagged."""
    grid = [float(x) for x in rk["lambda_grid"]]
    zones = sorted(T.zone.unique())
    if len(zones) < 2:
        return {"lambda": grid[len(grid) // 2], "status": "LAMBDA_FALLBACK (one training zone)", "inner": {}}
    ap = {lam: [] for lam in grid}
    for z in zones:
        tr, va = (T.zone != z).to_numpy(), (T.zone == z).to_numpy()
        if L.y.to_numpy()[tr].sum() < 1 or L.y.to_numpy()[va].sum() < 1:
            continue
        _, c = _fit_predict(spec, T[tr], L[tr], T[va], rk, seed)
        if spec.get("gated"):
            c = c * gate_mask(T[va], gc)
        base = logit(T.w2_unit_rank.to_numpy()[va])
        for lam in grid:
            ap[lam].append(weighted_ap(base + lam * c, L.y.to_numpy()[va], L.w.to_numpy()[va]))
    mean = {lam: float(np.nanmean(v)) if v and np.isfinite(v).any() else np.nan for lam, v in ap.items()}
    ok = {k: v for k, v in mean.items() if np.isfinite(v)}
    if not ok:
        return {"lambda": grid[len(grid) // 2], "status": "LAMBDA_FALLBACK (no inner zone with positives)",
                "inner": mean}
    best = max(ok.values())
    lam = min(k for k, v in ok.items() if v >= best - 1e-12)     # ties -> the weaker correction
    return {"lambda": lam, "status": "INNER_ZONE_SELECTED", "inner": mean}
