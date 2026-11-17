"""Ensembles of independently validated models (Phase 11).

Members are honest map sets {(task, zone): full grid} (OOF for geo zones, the split's discovery map for disc zones)
plus an optional full-region map. Every learned parameter for zone z is fitted on the OTHER zones' out-of-fold
predictions only, so the selector never sees validation labels of the region it predicts.

Methods: ``mean`` (rank mean), ``weighted`` (cross-fitted simplex weights, via ``g13.fusion.fixed_weight_search``
logic), ``oof_blend`` (logistic regression on member ranks), ``regime_gate`` (per-regime weights chosen on other
zones), ``spatial_moe`` (LightGBM gate on member ranks + label-free context, fitted per zone on the other zones).

UNTESTED: written without execution.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..gems3.representation import rank01


def zone_rank(p: np.ndarray, zone: np.ndarray) -> np.ndarray:
    out = np.full(p.shape, np.nan, np.float32)
    m = zone & np.isfinite(p)
    if m.any():
        out[m] = rank01(np.where(m, p, 0), m)[m]
    return out


def ranked_members(members: dict[str, dict], system) -> dict[str, dict]:
    """member -> {(task, zone): rank inside that unit's evaluation zone}."""
    zones = {(u.task, u.zone): np.zeros(system.data.shape, bool) for u in system.eval_units()}
    for u in system.eval_units():
        zones[(u.task, u.zone)][u.sl] = u.ev
        u.release()
    out = {}
    for name, maps in members.items():
        out[name] = {k: zone_rank(np.asarray(v, np.float32), zones[k]) for k, v in maps.items() if k in zones}
    return out


def _weights_grid(n: int, step: float):
    return [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=n) if abs(sum(w) - 1) < 1e-9]


def combine(ranks: dict, names: list[str], key, w) -> np.ndarray:
    return sum(np.nan_to_num(ranks[n][key]) * wi for n, wi in zip(names, w)).astype(np.float32)


def weighted_crossfit(ranks: dict, names: list[str], system, cfg, task: str) -> tuple[dict, pd.DataFrame]:
    """For each zone of ``task``: weights + area chosen on the other zones; returns the fused maps and the choices."""
    from ..g13.evaluation import area_table

    units = [u for u in system.eval_units((task,)) if all((task, u.zone) in ranks[n] for n in names)]
    grid = _weights_grid(len(names), cfg.ensemble.weight_step)
    rows = []
    for w in grid:
        for u in units:
            t = area_table(combine(ranks, names, (task, u.zone), w), [u], cfg.eval.area_fracs)
            t["w"] = str(tuple(round(x, 3) for x in w))
            rows.append(t)
    tab = pd.concat(rows, ignore_index=True)
    fused, choice = {}, []
    for u in units:
        other = tab[tab.zone != u.zone].groupby(["w", "target_frac"])[["tp_w", "fp_w", "fn_w"]].sum()
        s = other.tp_w / (other.tp_w + 0.2 * other.fp_w + 0.8 * other.fn_w + 1e-6)
        wbest, fbest = s.idxmax()
        w = tuple(float(x) for x in wbest.strip("()").split(",") if x.strip())
        fused[(task, u.zone)] = combine(ranks, names, (task, u.zone), w)
        choice.append({"task": task, "zone": u.zone, "weights": wbest, "frac": fbest, "dti_other": float(s.max())})
    return fused, pd.DataFrame(choice)


def oof_blend(ranks: dict, names: list[str], system, task: str, seed: int = 0, learner: str = "logistic",
              context: dict[str, np.ndarray] | None = None) -> dict:
    """Blend for zone z fitted on samples of the other zones (their truth = held-out truth of THOSE zones, whose maps
    are themselves out-of-fold). ``learner='lgbm'`` + label-free ``context`` layers = spatial mixture of experts."""
    from sklearn.linear_model import LogisticRegression

    ctx = list((context or {}).values())
    rng = np.random.default_rng(seed)
    units = [u for u in system.eval_units((task,)) if all((task, u.zone) in ranks[n] for n in names)]
    samples = {}
    for u in units:
        truth = np.zeros(system.data.shape, bool)
        truth[u.sl] = u.eval_truth()
        ev = np.zeros(system.data.shape, bool)
        ev[u.sl] = u.scored_mask()
        pos = np.flatnonzero((truth & ev).ravel())
        neg = np.flatnonzero((ev & ~truth).ravel())
        neg = rng.choice(neg, min(len(neg), 20 * max(len(pos), 1)), replace=False) if len(neg) else neg
        ids = np.r_[pos, neg]
        X = np.stack([np.nan_to_num(ranks[n][(task, u.zone)].ravel()[ids]) for n in names]
                     + [np.nan_to_num(np.asarray(c, np.float32).ravel()[ids]) for c in ctx], 1)
        samples[u.zone] = (X, np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        u.release()
    fused = {}
    for u in units:
        Xs = [samples[z][0] for z in samples if z != u.zone]
        ys = [samples[z][1] for z in samples if z != u.zone]
        if not Xs:
            continue
        if learner == "lgbm":
            from lightgbm import LGBMClassifier

            clf = LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.05, min_child_samples=50,
                                 random_state=seed, deterministic=True, force_row_wise=True, verbose=-1,
                                 class_weight="balanced")
        else:
            clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(np.vstack(Xs), np.concatenate(ys))
        R = np.stack([np.nan_to_num(ranks[n][(task, u.zone)]) for n in names]
                     + [np.nan_to_num(np.asarray(c, np.float32)) for c in ctx], -1)
        flat = R.reshape(-1, R.shape[-1])
        p = np.concatenate([clf.predict_proba(flat[i:i + 1_000_000])[:, 1] for i in range(0, len(flat), 1_000_000)])
        fused[(task, u.zone)] = p.reshape(system.data.shape).astype(np.float32)
    return fused


def spatial_moe(ranks: dict, names: list[str], system, task: str, context: dict[str, np.ndarray], seed: int = 0) -> dict:
    """Spatially varying mixture of experts: a LightGBM gate on member ranks + label-free context (regime, data
    reliability, coherence), fitted per held-out zone on the other zones only."""
    return oof_blend(ranks, names, system, task, seed, learner="lgbm", context=context)


def regime_gate(ranks: dict, names: list[str], system, cfg, task: str, regimes: np.ndarray) -> dict:
    """Per-regime member weights; for zone z each regime's weights are chosen on the other zones' pixels of that
    regime (regimes are label-free, fitted on training pixels in G13)."""
    from ..g13.evaluation import dti_of

    units = [u for u in system.eval_units((task,)) if all((task, u.zone) in ranks[n] for n in names)]
    grid = _weights_grid(len(names), cfg.ensemble.weight_step)
    fused = {}
    for u in units:
        out = np.zeros(system.data.shape, np.float32)
        for reg in sorted(set(np.unique(regimes)) - {-1}):
            best, bw = -1.0, grid[0]
            for w in grid:
                tp = fp = fn = 0.0
                for v in units:
                    if v.zone == u.zone:
                        continue
                    s = combine(ranks, names, (task, v.zone), w)
                    s = np.where(regimes == reg, s, 0.0)
                    sc = v.crop(s)
                    # fixed 5 % budget inside the regime for the weight choice (documented simplification)
                    c = v.components((sc >= np.nanquantile(sc[v.ev], 0.95)).astype(np.float64) * v.ev)
                    tp, fp, fn = tp + c.tp, fp + c.fp, fn + c.fn
                d = dti_of(tp, fp, fn)
                if d > best:
                    best, bw = d, w
            m = regimes == reg
            out[m] = combine(ranks, names, (task, u.zone), bw)[m]
        fused[(task, u.zone)] = out
    return fused


def save(name: str, fused: dict, choices: pd.DataFrame | None, out_dir: str | Path, provenance: dict) -> None:
    root = Path(out_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    for (task, zone), a in fused.items():
        np.save(root / f"{task}_{zone}.npy", a)
    if choices is not None:
        choices.to_csv(root / "choices.csv", index=False)
    (root / "provenance.json").write_text(json.dumps(provenance, indent=1, default=str))
