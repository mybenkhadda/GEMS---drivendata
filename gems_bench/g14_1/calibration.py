"""Soft probability calibration fitted out-of-fold (Phase G4) and the cross-fitted calibration x representation
sweep.

Every calibrator for held-out zone z is fitted on the OTHER evaluation units of the same task (their held-out maps
and their truth); z's labels never touch its own calibration. Monotone transforms (temperature, power, quantile,
isotonic, logistic) leave binary top-a masks unchanged but change the continuous values the metric integrates
(FP = sum p (1 - k)), so they matter for soft and sparse-soft submissions.

Kinds:

``identity``      p
``temperature``   sigmoid(logit(p) / T), T on a grid maximising pooled raw DTI of the other zones
``power``         p ** g, g on a grid maximising pooled raw DTI of the other zones
``logistic``      Platt scaling on logit(p) against truth pixels of the other zones (sampled)
``isotonic``      isotonic regression of truth on p (other zones, sampled)
``quantile``      within-zone rank mapped onto the pooled score distribution of the other zones (removes zone-level
                  scale drift)
``regime``        temperature per label-free regime (falls back to the global temperature for unseen regimes)

UNTESTED: written without execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
G_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def temperature(p, T):
    return _sig(_logit(p) / T).astype(np.float32)


def _pooled_raw_dti(crops: list[tuple], f) -> float:
    from ..g13.evaluation import dti_of

    tp = fp = fn = 0.0
    for u, s in crops:
        c = u.components(np.where(u.ev, f(np.nan_to_num(s)), 0.0))
        tp, fp, fn = tp + c.tp, fp + c.fp, fn + c.fn
    return dti_of(tp, fp, fn)


def _samples(crops: list[tuple], n_per_unit: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for u, s in crops:
        m = u.scored_mask() & np.isfinite(s)
        t = u.eval_truth()[m]
        v = np.asarray(s, np.float64)[m]
        pos, neg = np.flatnonzero(t), np.flatnonzero(~t)
        take = np.concatenate([pos, rng.choice(neg, min(neg.size, n_per_unit), replace=False)]) if neg.size else pos
        w = np.ones(take.size)
        if neg.size > n_per_unit:                       # importance weight restores the true class ratio
            w[pos.size:] = neg.size / n_per_unit
        xs.append(np.c_[v[take], w])
        ys.append(t[take])
    a = np.vstack(xs) if xs else np.zeros((0, 2))
    return a, (np.concatenate(ys) if ys else np.zeros(0, bool))


def fit(kind: str, crops: list[tuple], regimes_crops: list | None = None, seed: int = 0):
    """Calibrator fitted on ``crops`` = [(unit, cropped held-out score), ...] of the OTHER zones.
    Returns (callable(score, regime_crop=None) -> calibrated, params dict)."""
    if kind == "identity":
        return (lambda s, r=None: s), {}
    if kind == "temperature":
        best = max(T_GRID, key=lambda T: _pooled_raw_dti(crops, lambda x: temperature(x, T)))
        return (lambda s, r=None: temperature(s, best)), {"T": best}
    if kind == "power":
        best = max(G_GRID, key=lambda g: _pooled_raw_dti(crops, lambda x: np.clip(x, 0, 1) ** g))
        return (lambda s, r=None: (np.clip(s, 0, 1) ** best).astype(np.float32)), {"gamma": best}
    if kind in ("logistic", "isotonic"):
        X, y = _samples(crops, seed=seed)
        if y.size == 0 or y.all() or not y.any():
            return (lambda s, r=None: s), {"status": "degenerate sample: identity"}
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            m = LogisticRegression(C=1e3, max_iter=200).fit(_logit(X[:, :1]), y, sample_weight=X[:, 1])
            a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
            return (lambda s, r=None: _sig(a * _logit(s) + b).astype(np.float32)), {"a": a, "b": b}
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(X[:, 0], y, sample_weight=X[:, 1])
        return (lambda s, r=None: iso.predict(np.nan_to_num(s).ravel()).reshape(np.shape(s)).astype(np.float32)), \
            {"n_thresholds": int(len(iso.X_thresholds_))}
    if kind == "quantile":
        ref = np.concatenate([np.asarray(s, np.float64)[u.scored_mask() & np.isfinite(s)] for u, s in crops])
        qs = np.quantile(ref, np.linspace(0, 1, 2001)) if ref.size else np.linspace(0, 1, 2001)

        def f(s, r=None):
            from scipy.stats import rankdata

            out = np.zeros(np.shape(s), np.float32)
            fin = np.isfinite(s)
            if fin.any():
                rk = (rankdata(np.asarray(s)[fin]) - 1) / max(fin.sum() - 1, 1)
                out[fin] = np.interp(rk, np.linspace(0, 1, 2001), qs)
            return out
        return f, {"n_ref": int(ref.size)}
    if kind == "regime":
        if regimes_crops is None:
            raise ValueError("regime calibration needs regime crops")
        glob_f, gp = fit("temperature", crops)
        per = {}
        allr = np.unique(np.concatenate([np.unique(r[u.ev]) for (u, _), r in zip(crops, regimes_crops)]))
        for g in allr:
            sub = [(u, np.where(r == g, s, np.nan)) for (u, s), r in zip(crops, regimes_crops)]
            per[int(g)] = max(T_GRID, key=lambda T: _pooled_raw_dti(sub, lambda x: temperature(x, T)))

        def f(s, r=None):
            if r is None:
                return glob_f(s)
            out = glob_f(s)
            for g, T in per.items():
                m = r == g
                out[m] = temperature(np.asarray(s)[m], T)
            return out
        return f, {"T_global": gp["T"], "T_by_regime": per}
    raise KeyError(kind)


def crossfit_calibration(maps: dict, units: list, kinds, reps=("raw", "sparse_soft"), fracs=(0.005, 0.01, 0.02, 0.05),
                         regimes: np.ndarray | None = None) -> pd.DataFrame:
    """Rows (task, zone, calibration, rep, target_frac, components) where every calibration for a zone is fitted on
    the other zones of its task. ``maps``: {(task, zone): full-grid held-out score}."""
    from .representations import apply_rep

    by_task: dict[str, list] = {}
    for u in units:
        if (u.task, u.zone) in maps:
            by_task.setdefault(u.task, []).append(u)
    rows = []
    for task, us in by_task.items():
        crops = {u.zone: np.asarray(u.crop(maps[(u.task, u.zone)]), np.float32) for u in us}
        rcrops = {u.zone: u.crop(regimes) for u in us} if regimes is not None else None
        for u in us:
            other = [(o, crops[o.zone]) for o in us if o.zone != u.zone] or [(u, crops[u.zone])]
            for kind in kinds:
                if kind == "regime" and rcrops is None:
                    continue
                f, prm = fit(kind, other, [rcrops[o.zone] for o, _ in other] if kind == "regime" else None)
                s = f(crops[u.zone], rcrops[u.zone] if rcrops is not None else None)
                known = u.exclude
                for rep in reps:
                    for fr in (fracs if rep != "raw" else [None]):
                        q, acc = apply_rep(rep, s, u.ev, known, u.scored_mask(), fr, "pixel_exact")
                        c = u.components(q)
                        rows.append({"task": task, "zone": u.zone, "calibration": kind, "params": str(prm),
                                     "rep": rep, "target_frac": -1.0 if fr is None else fr, "mass": acc["mass"],
                                     "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "dti": c.dti,
                                     "fit_on_zones": ",".join(str(o.zone) for o, _ in other),
                                     "in_sample": len(us) == 1})
            u.release()
    return pd.DataFrame(rows)
