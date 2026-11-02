"""Does G06 complement W2? Cross-fitted meta-models on honest held-out predictions (G06-M1).

Inputs per unit (ranked inside the unit's evaluation zone, label-free): honest W2 (P6 held-out map), CNN B0 / B1
(OOF / discovery maps), the G06 geology pixel prediction and, optionally, the G06 segment map; disagreement
|W2 - G06|; the unsupervised geological domain.

Models compared (same units, same fixed policy):
    W2                frozen reference (no fit)
    rank_average      predeclared weights, no fit                          -> STRICT_LABEL_FREE (promotable)
    residual_lgbm     init score = logit(W2 rank), small regularised LightGBM correction
    moe               per-domain L2 logistic regressions shrunk to a global one (n_d / (n_d + tau))

Cross-fitting: held-out unit u (scenario s, zone z) <- rows of the other units of s. For a training unit k the G06
pixel input is the NESTED prediction of fit ``s:z|k`` (never saw z's truth); W2 / CNN maps of k were trained with
z's truth among their labels (P6 / M9 protocols), so every learned model that uses W2 / CNN is ``SCREENING_ONLY``.
A learned model on G06 inputs only, with nested inputs for every training unit, is ``STRICT_NESTED``. A G06 segment
map of a training unit came from a model that saw z: it is accepted only in screening mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import assert_label_free_columns

UPSTREAM_SUPERVISED = {"w2", "cnn_b0", "cnn_b1", "g06_segment"}


def logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_oof_prediction_matrix(maps: dict, ev: np.ndarray, domain: np.ndarray | None = None) -> dict:
    """Rank maps of the available inputs + disagreement (+ domain id): {column: crop array}."""
    from ..gems3.representation import rank01

    out = {}
    for k, m in maps.items():
        if m is not None:
            out[k] = rank01(np.nan_to_num(np.asarray(m, np.float32), nan=0.0), ev)
    if "w2" in out and "g06_pixel" in out:
        out["dis_w2_g06"] = np.abs(out["w2"] - out["g06_pixel"])
    if domain is not None:
        out["domain"] = np.asarray(domain, np.float32)
    return out


def meta_rows(vu, R: dict, lab_crop: np.ndarray, cfg: dict, seed: int) -> pd.DataFrame:
    """Training rows of one (training) unit: its scored truth as positives, scored pixels far from every mapped fault
    as negatives (W2 sampling convention, ``p6.stack``); columns = rank inputs."""
    from ..p6.stack import sample_weights, training_masks

    u = vu.unit
    ign = None
    pos, neg = training_masks(u.eval_truth(), u.scored_mask(), lab_crop, ign, int(cfg.get("neg_buffer_px", 5)))
    amb = np.zeros(pos.shape, bool)
    idx, y, w = sample_weights(pos, neg, amb, np.random.default_rng(int(seed)), int(cfg.get("neg_per_unit", 80000)),
                               int(cfg.get("pos_cap_per_unit", 20000)), float(cfg.get("negative_weight", 1.0)), 1.0)
    r, c = np.unravel_index(idx, pos.shape)
    T = pd.DataFrame({k: np.asarray(v)[r, c] for k, v in R.items()})
    T["y"], T["w"], T["unit"] = y, w, vu.key
    T["r_full"], T["c_full"] = r + u.sl[0].start, c + u.sl[1].start
    return T


class MetaModel:
    def __init__(self, kind: str, columns: list[str], est=None, weights: dict | None = None, extra: dict | None = None):
        self.kind, self.columns, self.est = kind, list(columns), est
        self.weights, self.extra = weights or {}, extra or {}

    def score(self, R: dict, ev: np.ndarray) -> np.ndarray:
        """Score map over the crop (NaN outside ``ev``)."""
        out = np.full(ev.shape, np.nan, np.float32)
        if self.kind == "w2":
            out[ev] = R["w2"][ev]
            return out
        if self.kind == "rank_average":
            tot = sum(self.weights.values())
            s = sum(w * np.nan_to_num(R[k], nan=0.0) for k, w in self.weights.items() if k in R)
            out[ev] = (s / max(tot, 1e-12))[ev]
            return out
        X = np.stack([np.asarray(R[c])[ev] for c in self.columns], 1).astype(np.float32)
        if self.kind == "residual_lgbm":
            base = logit(R["w2"][ev])
            out[ev] = 1 / (1 + np.exp(-(base + self.est.predict(X, raw_score=True))))
            return out
        if self.kind == "moe":
            out[ev] = moe_predict(self.est, X, np.asarray(R.get("domain", np.zeros(ev.shape)))[ev],
                                  float(self.extra.get("tau", 5000)))
            return out
        raise KeyError(self.kind)


def fit_residual_ranker(X: np.ndarray, y: np.ndarray, w: np.ndarray, w2_col: int, params: dict, seed: int):
    """LightGBM correction on top of logit(W2 rank) (init score), small and regularised."""
    import lightgbm as lgb

    est = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "early_stopping_rounds"},
                             random_state=int(seed), verbose=-1)
    est.fit(X, y, sample_weight=w, init_score=logit(X[:, w2_col]))
    return est


def fit_moe(X: np.ndarray, y: np.ndarray, w: np.ndarray, domain: np.ndarray, C: float, seed: int) -> dict:
    """{'global': model, 'domains': {d: (model, n_d)}} of L2 logistic regressions."""
    from sklearn.linear_model import LogisticRegression

    def lr():
        return LogisticRegression(C=float(C), max_iter=500, random_state=int(seed))

    ok = np.all(np.isfinite(X), 1)
    g = lr().fit(X[ok], y[ok], sample_weight=w[ok])
    doms = {}
    for d in np.unique(domain[ok & np.isfinite(domain)]):
        m = ok & (domain == d)
        if np.unique(y[m]).size == 2 and m.sum() >= 50:
            doms[float(d)] = (lr().fit(X[m], y[m], sample_weight=w[m]), int(m.sum()))
    return {"global": g, "domains": doms}


def moe_predict(model: dict, X: np.ndarray, domain: np.ndarray, tau: float) -> np.ndarray:
    Xf = np.nan_to_num(X, nan=0.0)
    p = model["global"].predict_proba(Xf)[:, 1]
    for d, (m, n) in model["domains"].items():
        sel = domain == d
        if sel.any():
            lam = n / (n + tau)
            p[sel] = lam * m.predict_proba(Xf[sel])[:, 1] + (1 - lam) * p[sel]
    return p.astype(np.float32)


def train_regularized_meta_model(kind: str, T: pd.DataFrame, columns: list[str], cfg: dict, seed: int) -> MetaModel:
    assert_label_free_columns(columns, "meta inputs")
    X = T[columns].to_numpy(np.float32)
    y, w = T.y.to_numpy(int), T.w.to_numpy(np.float32)
    if kind == "residual_lgbm":
        est = fit_residual_ranker(X, y, w, columns.index("w2"), cfg.get("residual_lgbm", {}), seed)
        return MetaModel(kind, columns, est)
    if kind == "moe":
        mc = cfg.get("moe", {})
        cols = [c for c in columns if c != "domain"]
        dom = T["domain"].to_numpy(float) if "domain" in T else np.zeros(len(T))
        est = fit_moe(T[cols].to_numpy(np.float32), y, w, dom, float(mc.get("C", 0.1)), seed)
        return MetaModel(kind, cols, est, extra={"tau": float(mc.get("tau", 5000))})
    raise KeyError(kind)


def strictness_of(kind: str, inputs, nested_ok: bool) -> str:
    if kind == "w2":
        return "REFERENCE"
    if kind == "rank_average":
        return "STRICT_LABEL_FREE"
    if set(inputs) & UPSTREAM_SUPERVISED:
        return "SCREENING_ONLY"
    return "STRICT_NESTED" if nested_ok else "SCREENING_ONLY"


def crossfit_meta(units: list, unit_inputs, lab_full: np.ndarray, cfg: dict, seed: int, progress=print
                  ) -> tuple[dict, dict]:
    """{model kind: {unit key: score map}}, info. ``unit_inputs(vu, outer)`` -> rank dict (outer = held-out zone
    for training units, None for the held-out unit itself) or None when an input is missing."""
    kinds = list(cfg.get("models", ["w2", "rank_average", "residual_lgbm", "moe"]))
    inputs = list(cfg.get("inputs", ["w2", "cnn_b0", "cnn_b1", "g06_pixel"]))
    maps: dict = {k: {} for k in kinds}
    info: dict = {"units": {}, "nested_ok": True}
    for vu in units:
        R = unit_inputs(vu, None)
        if R is None:
            info["units"][vu.key] = {"status": "BLOCKED_MISSING_INPUT"}
            continue
        ev = vu.unit.ev
        for k in kinds:
            if k == "w2":
                maps[k][vu.key] = MetaModel("w2", ["w2"]).score(R, ev)
            elif k == "rank_average":
                wts = {a: b for a, b in (cfg.get("rank_average_weights") or {}).items() if a in R}
                maps[k][vu.key] = MetaModel("rank_average", list(wts), weights=wts).score(R, ev)
        learned = [k for k in kinds if k in ("residual_lgbm", "moe")]
        if not learned:
            continue
        parts = []
        for ku in units:
            if ku.scenario != vu.scenario or ku.zone == vu.zone:
                continue
            Rk = unit_inputs(ku, vu.zone)
            if Rk is None:
                info["nested_ok"] = False
                continue
            parts.append(meta_rows(ku, Rk, lab_full[ku.sl], cfg.get("rows", {}), seed + 7 * ku.zone))
        if not parts:
            info["units"][vu.key] = {"status": "FAILED", "detail": "no training unit"}
            continue
        T = pd.concat(parts, ignore_index=True)
        from ..m7.ranking import _zone_mask

        zm = _zone_mask(vu.unit, lab_full.shape, float(cfg.get("rows", {}).get("exclude_buffer_km", 5.0)))
        T = T[~zm[T.r_full.to_numpy(), T.c_full.to_numpy()]]
        cols = [c for c in inputs + ["dis_w2_g06", "domain"] if c in T.columns and c in R]
        for kd in learned:
            if kd == "residual_lgbm" and "w2" not in cols:
                continue
            m = train_regularized_meta_model(kd, T, cols, cfg, seed)
            maps[kd][vu.key] = m.score(R, ev)
        info["units"][vu.key] = {"status": "COMPLETE", "n_train": int(len(T)), "n_pos": int(T.y.sum()),
                                 "columns": cols}
        progress and progress(f"  meta {vu.key}: {len(T)} training rows, inputs {cols}")
    info["strictness"] = {k: strictness_of(k, inputs, info["nested_ok"]) for k in kinds}
    return maps, info


def evaluate_model_complementarity(maps: dict, vu, frac: float, sys_support: float, sys_frac: float,
                                   ref: str = "w2") -> pd.DataFrame:
    """Per model vs ``ref`` on one unit: rank correlation (sample), Jaccard of the top-``frac`` masks, systems
    recovered by each / both / only the model / only the reference."""
    from scipy.stats import spearmanr

    from ..gems3.representation import top_mask
    from ..geomoe.scenarios import system_recovery

    u = vu.unit
    elig = u.scored_mask()
    rows = []
    if ref not in maps or maps[ref] is None:
        return pd.DataFrame()
    Mr = top_mask(maps[ref], elig, frac)
    Sr = system_recovery(u, Mr.astype(np.float32), vu.sysmap, sys_support, sys_frac)
    rec_r = set(Sr.system_id[Sr.recovered]) if len(Sr) else set()
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(elig.ravel())
    pick = rng.choice(idx, min(200000, idx.size), replace=False) if idx.size else idx
    for name, m in maps.items():
        if m is None or name == ref:
            continue
        M = top_mask(m, elig, frac)
        S = system_recovery(u, M.astype(np.float32), vu.sysmap, sys_support, sys_frac)
        rec = set(S.system_id[S.recovered]) if len(S) else set()
        inter, union = int((M & Mr).sum()), int((M | Mr).sum())
        rho = spearmanr(np.nan_to_num(m.ravel()[pick]), np.nan_to_num(maps[ref].ravel()[pick])).correlation \
            if pick.size > 2 else np.nan
        rows.append({"unit": vu.key, "model": name, "ref": ref, "frac": frac, "spearman": float(rho),
                     "jaccard_top": inter / max(union, 1), "n_rec_model": len(rec), "n_rec_ref": len(rec_r),
                     "n_rec_both": len(rec & rec_r), "n_new_vs_ref": len(rec - rec_r),
                     "n_lost_vs_ref": len(rec_r - rec), "new_ids": ";".join(map(str, sorted(rec - rec_r))),
                     "lost_ids": ";".join(map(str, sorted(rec_r - rec)))})
    return pd.DataFrame(rows)
