"""Experiment D: static fusion (Model B), predeclared rule gate (Model C), learned GeoMoE gate (Model D).

All three combine the SAME normalised expert scores s_i(c) in [0, 1] of the SAME candidates:

    S(c) = sum_i g_i(c) s_i(c),   g_i >= 0,   sum over AVAILABLE experts = 1,   g_i = 0 where expert i is missing

* B: one weight vector for every candidate, chosen on the training zones only from a predeclared simplex grid by the
  length-weighted average precision of the positive-unlabelled candidate targets (``pu_targets``).
* C: B's weights modified per candidate by predeclared observable rules (R1-R5, config ``rules``); each rule is ablated
  (C_all, C_minus_<rule>). Thresholds are training-zone ECDF quantiles; nothing is fitted.
* D: g(z) = softmax(V z + b) over the available experts, p = sigmoid(a S + c) fitted by weighted logistic loss with
  L2 on V and a KL penalty toward B's weights (the gate is a learned DEVIATION from static fusion: no collapse onto
  one source without evidence). Gate weights are routing weights, not calibrated probabilities.

Targets (PU): a training candidate is positive when >= ``positive_share`` of it lies within 300 m of its training
zone's scored truth; every other candidate is UNLABELLED (down-weighted negative, ``pu_unlabeled_weight``), never a
confirmed negative. ``training_rows`` refuses rows of the held-out zone and drops rows within ``train_buffer_km`` of
its evaluation zone.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .experts import FAMILIES

RULES = ("R1_terrain_scarp", "R2_structural_displacement", "R3_tip_continuation", "R4_magnetic_edge_only",
         "R5_short_multisource")
E_IDX = {f: i for i, f in enumerate(FAMILIES)}
GATE_DESCRIPTORS = ("mean_cnn_disagree", "mean_mag_hgrad", "mean_grav_hgrad", "mean_mg_orient_agree",
                    "mean_pf_persistence", "mean_g02c", "mean_g02a", "mean_g01_persistence", "length_km",
                    "straightness", "linearity", "connectivity", "mean_known_dist_km", "min_tip_dist_km",
                    "mean_tip_along_cos", "mean_known_density", "mean_dem", "mean_dem_observed", "n_proposers")


class LeakageError(RuntimeError):
    """A held-out row (or truth) was about to enter a fit."""


# =============================================================================== pure: targets / rows
def pu_targets(T: pd.DataFrame, positive_share: float, unlabeled_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """(y, w): positive = near-truth share >= ``positive_share``; weights = segment length, unlabelled down-weighted."""
    y = (T.y_near_truth.to_numpy(float) >= float(positive_share)).astype(np.int8)
    L = T.n_px.to_numpy(float)
    return y, np.where(y == 1, L, float(unlabeled_weight) * L)


def training_rows(T: pd.DataFrame, held_zone: int, zone_mask_full: np.ndarray | None) -> np.ndarray:
    """Training candidates for held-out zone ``held_zone`` of one scenario: other zones only, minus candidates whose
    centroid lies inside ``zone_mask_full`` (the held-out evaluation zone dilated by the buffer)."""
    m = (T.zone.to_numpy() != int(held_zone))
    if zone_mask_full is not None and m.any():
        r = np.clip(np.round(T.row_full.to_numpy()).astype(int), 0, zone_mask_full.shape[0] - 1)
        c = np.clip(np.round(T.col_full.to_numpy()).astype(int), 0, zone_mask_full.shape[1] - 1)
        m &= ~zone_mask_full[r, c]
    return m


def assert_disjoint(train_zones, held_zone: int) -> None:
    if int(held_zone) in {int(z) for z in train_zones}:
        raise LeakageError(f"held-out zone {held_zone} is among the training zones {sorted(set(train_zones))}")


# =============================================================================== pure: combination
def combine(S: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(score, effective weights): weights of missing experts set to 0 and the rest renormalised per candidate.
    ``W`` is (E,) or (n, E), nonnegative. A candidate with no available expert scores NaN."""
    S = np.asarray(S, float)
    W = np.broadcast_to(np.asarray(W, float), S.shape).copy()
    if (W < 0).any():
        raise ValueError("expert weights must be nonnegative")
    avail = np.isfinite(S)
    W[~avail] = 0.0
    tot = W.sum(1, keepdims=True)
    eff = np.where(tot > 0, W / np.where(tot > 0, tot, 1.0), 0.0)
    none = ~avail.any(1) | (tot[:, 0] <= 0)
    eff[none] = 0.0
    sc = np.where(none, np.nan, (eff * np.nan_to_num(S)).sum(1))
    return sc, eff


def simplex_grid(n: int, step: float) -> np.ndarray:
    k = int(round(1.0 / float(step)))
    pts = [c for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]
    return np.asarray(pts, float) / k


def weighted_ap(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted average precision (score descending; NaN scores last)."""
    s = np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    o = np.argsort(-s, kind="stable")
    y, w = np.asarray(y, bool)[o], np.asarray(w, float)[o]
    tp = np.cumsum(w * y)
    if tp[-1] <= 0:
        return float("nan")
    prec = tp / np.maximum(np.cumsum(w), 1e-12)
    return float((prec * w * y).sum() / tp[-1])


def fit_static(S: np.ndarray, y: np.ndarray, w: np.ndarray, step: float) -> dict:
    """Simplex point with the best weighted AP on the training candidates (ties -> closest to uniform)."""
    G = simplex_grid(S.shape[1], step)
    uni = np.full(S.shape[1], 1.0 / S.shape[1])
    best, rows = None, []
    for g in G:
        ap = weighted_ap(combine(S, g)[0], y, w)
        rows.append({"weights": g.tolist(), "ap": ap})
        key = (np.nan_to_num(ap, nan=-1.0), -float(np.abs(g - uni).sum()))
        if best is None or key > best[0]:
            best = (key, g)
    return {"weights": best[1].tolist(), "ap": float(best[0][0]), "grid": rows}


# =============================================================================== pure: rule gate
def rule_masks(D: pd.DataFrame, rc: dict) -> dict:
    """Boolean condition of every rule on the candidates (``D``: training-ECDF quantiles ``q_*`` + raw descriptors;
    a missing descriptor makes its condition False)."""
    g = (lambda c: D[c].to_numpy(float) if c in D else np.full(len(D), np.nan))
    with np.errstate(invalid="ignore"):
        obs = np.nan_to_num(g("mean_dem_observed"), nan=0.0) >= 0.5
        r = {"R1_terrain_scarp": obs & (g("q_mean_dem") >= rc["R1_terrain_scarp"]["dem_q"]),
             "R2_structural_displacement": ((g("q_mean_mag_hgrad") >= rc["R2_structural_displacement"]["pf_q"]) |
                                            (g("q_mean_grav_hgrad") >= rc["R2_structural_displacement"]["pf_q"])) &
             (g("q_mean_g02a") >= rc["R2_structural_displacement"]["g02a_q"]),
             "R3_tip_continuation": (g("min_tip_dist_km") <= rc["R3_tip_continuation"]["tip_km"]) &
             (g("mean_tip_along_cos") >= rc["R3_tip_continuation"]["tip_cos"]),
             "R4_magnetic_edge_only": (g("q_mean_mag_hgrad") >= rc["R4_magnetic_edge_only"]["mag_q"]) &
             (g("q_mean_grav_hgrad") < rc["R4_magnetic_edge_only"]["grav_q_max"]) &
             (np.nan_to_num(g("q_mean_g02c"), nan=0.0) < rc["R4_magnetic_edge_only"]["g02c_q_max"]),
             "R5_short_multisource": g("length_km") <= rc["R5_short_multisource"]["short_km"]}
    return {k: np.asarray(v, bool) for k, v in r.items()}


def rule_gate(S: np.ndarray, base_w, masks: dict, active, rc: dict) -> tuple[np.ndarray, np.ndarray]:
    """(score, effective weights) of Model C with the ``active`` rules (R5 acts on the combined score)."""
    f = float(rc["factor"])
    W = np.tile(np.asarray(base_w, float), (S.shape[0], 1))
    up = {"R1_terrain_scarp": "E4_terrain_network", "R2_structural_displacement": "E3_structural",
          "R3_tip_continuation": "E4_terrain_network"}
    for r, fam in up.items():
        if r in active:
            W[masks[r], E_IDX[fam]] *= f
    if "R4_magnetic_edge_only" in active:
        W[masks["R4_magnetic_edge_only"], E_IDX["E2_potential"]] /= f
    W = np.where(W.sum(1, keepdims=True) > 0, W, 1.0)          # a zero base vector falls back to uniform
    sc, eff = combine(S, W)
    if "R5_short_multisource" in active:
        p = rc["R5_short_multisource"]
        hi = (np.nan_to_num(S, nan=-1.0) >= float(p["agree_q"])).sum(1) >= int(p["min_agree"])
        top2 = np.sort(np.nan_to_num(S, nan=-np.inf), 1)[:, -2:]
        t2 = np.where(np.isfinite(top2).all(1), top2.mean(1), np.nan)
        m = masks["R5_short_multisource"] & hi & np.isfinite(t2)
        sc = np.where(m, np.fmax(sc, t2), sc)
    return sc, eff


def rule_variants(active_all=RULES) -> dict:
    out = {"C_all": tuple(active_all)}
    for r in active_all:
        out[f"C_minus_{r.split('_')[0]}"] = tuple(x for x in active_all if x != r)
    return out


# =============================================================================== pure: learned gate
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def gate_forward(theta: np.ndarray, Z: np.ndarray, S: np.ndarray, prior: np.ndarray):
    """(p, S_mix, g, logits) for parameters theta = [V (E x d), b (E), a, c]."""
    n, d = Z.shape
    E = S.shape[1]
    V = theta[:E * d].reshape(E, d)
    b = theta[E * d:E * d + E]
    a, c = theta[-2], theta[-1]
    avail = np.isfinite(S)
    L = Z @ V.T + b
    L = np.where(avail, L, -np.inf)
    mx = np.max(np.where(avail, L, -1e300), 1, keepdims=True)
    ex = np.where(avail, np.exp(L - mx), 0.0)
    den = ex.sum(1, keepdims=True)
    g = np.where(den > 0, ex / np.where(den > 0, den, 1.0), 0.0)
    Sm = (g * np.nan_to_num(S)).sum(1)
    return _sig(a * Sm + c), Sm, g, L


def gate_objective(theta, Z, S, y, w, prior, l2: float, kl: float):
    """Weighted logistic loss + l2 ||V||^2 + kl * mean KL(g || prior renormalised over available experts); analytic
    gradient."""
    n, d = Z.shape
    E = S.shape[1]
    p, Sm, g, _ = gate_forward(theta, Z, S, prior)
    avail = np.isfinite(S)
    W = float(w.sum())
    eps = 1e-12
    loss = -(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / W
    pr = np.where(avail, prior[None, :], 0.0)
    pr = pr / np.maximum(pr.sum(1, keepdims=True), eps)
    lg = np.where(g > 0, np.log(np.maximum(g, eps) / np.maximum(pr, eps)), 0.0)
    klr = (g * lg).sum(1)
    V = theta[:E * d].reshape(E, d)
    a = theta[-2]
    J = loss + kl * klr.mean() + l2 * float((V ** 2).sum())
    r = w * (p - y) / W                                     # dloss / dlogit(p)
    dS = r * a
    dL = dS[:, None] * g * (np.nan_to_num(S) - Sm[:, None])
    dL += (kl / n) * g * (lg - klr[:, None])
    dL = np.where(avail, dL, 0.0)
    gV = dL.T @ Z + 2.0 * l2 * V
    gb = dL.sum(0)
    ga = float((r * Sm).sum())
    gc = float(r.sum())
    return J, np.concatenate([gV.ravel(), gb, [ga, gc]])


def standardize_fit(Z: np.ndarray) -> dict:
    mu = np.nanmean(Z, 0)
    sd = np.nanstd(Z, 0)
    mu, sd = np.nan_to_num(mu, nan=0.0), np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return {"mean": mu.tolist(), "std": sd.tolist()}


def standardize_apply(Z: np.ndarray, st: dict) -> np.ndarray:
    return np.nan_to_num((Z - np.asarray(st["mean"])) / np.asarray(st["std"]), nan=0.0)


def fit_gate(Z: np.ndarray, S: np.ndarray, y: np.ndarray, w: np.ndarray, prior, lc: dict, seed: int = 2026) -> dict:
    """L-BFGS fit from the static-fusion starting point (V = 0, b = log prior)."""
    from scipy.optimize import minimize

    n, d = Z.shape
    E = S.shape[1]
    prior = np.maximum(np.asarray(prior, float), 1e-3)
    prior = prior / prior.sum()
    rng = np.random.default_rng(seed)
    th0 = np.concatenate([rng.normal(0, 1e-3, E * d), np.log(prior), [4.0, -2.0]])
    res = minimize(gate_objective, th0, args=(Z, S, y.astype(float), w.astype(float), prior, float(lc["l2"]),
                                              float(lc["kl_to_static"])), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(lc["max_iter"])})
    return {"theta": res.x.tolist(), "E": E, "d": d, "prior": prior.tolist(), "converged": bool(res.success),
            "n_iter": int(res.nit), "objective": float(res.fun), "message": str(res.message)}


def gate_predict(model: dict, Z: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mixture score S(c), gate weights g) of a fitted gate; candidates without any expert -> NaN."""
    _, Sm, g, _ = gate_forward(np.asarray(model["theta"]), Z, S, np.asarray(model["prior"]))
    none = ~np.isfinite(S).any(1)
    return np.where(none, np.nan, Sm), g


def collapse(g: np.ndarray, S: np.ndarray, cc: dict) -> dict:
    """Routing collapse diagnostic (label-free): mean max weight and the share of candidates routed mainly to the
    most frequent expert; COLLAPSED when both exceed the configured limits."""
    ok = np.isfinite(S).any(1)
    if not ok.any():
        return {"status": "NO_CANDIDATES", "collapsed": False}
    gg = g[ok]
    mx = gg.max(1)
    am = gg.argmax(1)
    dom = int(np.bincount(am, minlength=gg.shape[1]).argmax())
    share = float((am == dom).mean())
    col = bool(mx.mean() >= float(cc["mean_max_weight"]) and share >= float(cc["dominant_share"]))
    return {"status": "COLLAPSED" if col else "OK", "collapsed": col, "mean_max_weight": float(mx.mean()),
            "dominant_expert": FAMILIES[dom] if dom < len(FAMILIES) else dom, "dominant_share": share,
            "mean_weights": dict(zip(FAMILIES, gg.mean(0).round(4).tolist()))}


def expert_agreement(S: np.ndarray, q: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """(number of experts with normalised score >= q, dispersion of the available expert scores)."""
    with np.errstate(invalid="ignore"):
        return (np.nan_to_num(S, nan=-1.0) >= q).sum(1), np.nanstd(S, 1)
