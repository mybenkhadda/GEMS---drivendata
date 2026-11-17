"""Tier-1 tabular models, incomplete-label sampling strategies and honest map production (sections 22-23).

One *experiment* = (feature list, model kind, sampling strategy). It produces, with no label leakage:

* ``oof``    out-of-fold map on the CV folds (model k trained outside fold k and its embargo buffer),
* ``disc``   one map per discovery split (model trained without the withheld systems; ignore ring weight 0),
* ``full``   the submission model (trained on the whole CV region, TEST excluded) applied to the full grid; its
             TEST-fold values are the once-only held-out evaluation.

Sampling strategies (unlabeled pixels are *unlabeled*, not proven negatives)::

    standard              uniform negatives beyond the ignore ring (100-300 m ring weight 0)
    distance_stratified   equal negatives per distance band 0.3-1, 1-3, 3-10, > 10 km (near-field discrimination)
    soft_negative         uniform negatives with weight rising from 0.2 at 300 m to 1 at 3 km
    hard_negative         half of the negatives from label-free confounder categories (ridges, drainages, strong
                          potential-field edges, nodata boundaries), weight 0.5 because some may be unmapped faults
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .cache import atomic_save_npy, atomic_write_json, combine

DIST_BANDS_KM = (0.3, 1.0, 3.0, 10.0, 1e9)


# =============================================================================== sampling
def distance_px(labels: np.ndarray) -> np.ndarray:
    return (
        ndimage.distance_transform_edt(~labels).astype(np.float32)
        if labels.any()
        else np.full(labels.shape, 1e6, np.float32)
    )


@dataclass
class Sample:
    ids: np.ndarray
    y: np.ndarray
    w: np.ndarray
    info: dict = field(default_factory=dict)


def draw_sample(
    train: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    strategy: str = "standard",
    positive_radius_px: int = 1,
    ignore_ring_px: int = 3,
    negative_ratio: float = 6.0,
    max_positives: int = 120_000,
    ignore: np.ndarray | None = None,
    hard_categories: dict[str, np.ndarray] | None = None,
) -> Sample:
    """Positive / negative pixel sample inside ``train`` (flat indices), with weights."""
    d = distance_px(labels)
    allowed = train.copy()
    if ignore is not None:
        allowed &= ~ignore
    pos = np.flatnonzero((allowed & (d <= positive_radius_px)).ravel())
    if pos.size == 0:
        raise ValueError("no positive pixels in the training mask")
    if pos.size > max_positives:
        pos = rng.choice(pos, max_positives, replace=False)
    wpos = (1.0 - d.ravel()[pos] / 3.0).astype(np.float32)  # DTI kernel weight of a 0 / 1 px positive
    neg_pool = allowed & (d > ignore_ring_px)
    n_neg = int(min(negative_ratio * pos.size, neg_pool.sum()))
    dk = d * 0.1
    info = {"strategy": strategy, "n_pos": int(pos.size)}
    if strategy == "standard":
        neg = rng.choice(np.flatnonzero(neg_pool.ravel()), n_neg, replace=False)
        wneg = np.ones(n_neg, np.float32)
    elif strategy == "distance_stratified":
        parts = []
        bands = list(zip(DIST_BANDS_KM[:-1], DIST_BANDS_KM[1:]))
        per = n_neg // len(bands)
        for lo, hi in bands:
            pool = np.flatnonzero((neg_pool & (dk > lo) & (dk <= hi)).ravel())
            if pool.size:
                parts.append(rng.choice(pool, min(per, pool.size), replace=False))
            info[f"neg_{lo:g}_{hi:g}km"] = int(min(per, pool.size))
        neg = np.concatenate(parts) if parts else np.empty(0, np.int64)
        wneg = np.ones(len(neg), np.float32)
    elif strategy == "soft_negative":
        neg = rng.choice(np.flatnonzero(neg_pool.ravel()), n_neg, replace=False)
        wneg = np.clip((dk.ravel()[neg] - 0.3) / 2.7, 0.2, 1.0).astype(np.float32)
    elif strategy == "hard_negative":
        if not hard_categories:
            raise ValueError("hard_negative needs hard_categories")
        half = n_neg // 2
        parts, wparts = [], []
        per = max(half // len(hard_categories), 1)
        far = neg_pool & (dk > 1.0)
        for name, m in hard_categories.items():
            pool = np.flatnonzero((far & m).ravel())
            k = min(per, pool.size)
            if k:
                parts.append(rng.choice(pool, k, replace=False))
                wparts.append(np.full(k, 0.5, np.float32))
            info[f"hard_{name}"] = int(k)
        rest = n_neg - sum(len(p) for p in parts)
        parts.append(rng.choice(np.flatnonzero(neg_pool.ravel()), rest, replace=False))
        wparts.append(np.ones(rest, np.float32))
        neg, wneg = np.concatenate(parts), np.concatenate(wparts)
    else:
        raise KeyError(f"unknown sampling strategy {strategy}")
    ids = np.concatenate([pos, neg]).astype(np.int64)
    y = np.r_[np.ones(len(pos), np.uint8), np.zeros(len(neg), np.uint8)]
    w = np.r_[wpos, wneg].astype(np.float32)
    order = rng.permutation(len(ids))
    info["n_neg"] = int(len(neg))
    return Sample(ids[order], y[order], w[order], info)


def hard_negative_categories(store, data, top_q: float = 0.95) -> dict[str, np.ndarray]:
    """Label-free confounder masks (top 5 % responses) used for hard-negative sampling and the error taxonomy."""
    cats = {}
    cand = {
        "terrain_ridge": "terrain:det_elev:ridge_s400",
        "drainage_valley": "terrain:det_elev:valley_s400",
        "magnetic_edge": "magnetic:rtp:thd",
        "gravity_edge": "gravity:iso_grav_anom:thd",
    }
    for k, n in cand.items():
        if store.has(n):
            a = store.get(n)
            v = data.valid & np.isfinite(a)
            if v.any():
                cats[k] = v & (a >= np.quantile(a[v][:: max(v.sum() // 300_000, 1)], top_q))
    if store.has("validity:all:boundary_dist_km"):
        raw_km = store.physical("validity:all:boundary_dist_km")
        cats["nodata_boundary"] = data.valid & np.isfinite(raw_km) & (raw_km < 1.0)
    if not cats:  # raw-only fallback: gradient of raw terrain / RTP
        from .signal import fill_nodata, gaussian_gradient

        for k, n in (("terrain_edge", "det_elev"), ("magnetic_edge", "rtp")):
            if data.has(n):
                x = data.band(n)
                v = data.valid & np.isfinite(x)
                g = np.hypot(*gaussian_gradient(fill_nodata(x, v), 2.0))
                cats[k] = v & (g >= np.quantile(g[v], top_q))
    return cats


# =============================================================================== model wrappers
def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def make_model(kind: str, cfg, seed: int):
    if kind == "xgboost":
        from xgboost import XGBClassifier

        kw = dict(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=2.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=cfg.n_jobs,
            random_state=seed,
            eval_metric="logloss",
            missing=np.nan,
            max_bin=256,
        )
        if _cuda():
            kw["device"] = "cuda"
        return XGBClassifier(**kw)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=cfg.n_estimators,
            num_leaves=2 ** min(cfg.max_depth, 8) - 1,
            learning_rate=cfg.learning_rate,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_lambda=1.0,
            n_jobs=cfg.n_jobs,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
    raise KeyError(kind)


def fit(kind: str, cfg, X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int):
    m = make_model(kind, cfg, seed)
    m.fit(X, y, sample_weight=w)
    return m


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)


def save_model(model, kind: str, path: Path, names: list[str], meta: dict) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "xgboost":
        mp = path.with_suffix(".xgb.json")
        model.save_model(mp)
    else:
        mp = path.with_suffix(".lgb.txt")
        model.booster_.save_model(str(mp))
    jp = atomic_write_json(path.with_suffix(".meta.json"), {"kind": kind, "features": names, **meta})
    return [mp, jp]


def load_model(path: Path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if meta["kind"] == "xgboost":
        from xgboost import XGBClassifier

        m = XGBClassifier()
        m.load_model(path.with_suffix(".xgb.json"))
    else:
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path.with_suffix(".lgb.txt")))
        m = _BoosterWrap(booster)
    return m, meta


class _BoosterWrap:
    def __init__(self, b):
        self.b = b

    def predict_proba(self, X):
        p = self.b.predict(X)
        return np.c_[1 - p, p]


def predict_mask(model, store, names: list[str], mask: np.ndarray, chunk_rows: int = 128) -> np.ndarray:
    """Probability on ``mask`` pixels (row-chunked reads of the feature memmaps); NaN elsewhere."""
    h, w = mask.shape
    out = np.full((h, w), np.nan, np.float32)
    for r0 in range(0, h, chunk_rows):
        r1 = min(h, r0 + chunk_rows)
        m = mask[r0:r1]
        if not m.any():
            continue
        ids = (np.flatnonzero(m.ravel()) + r0 * w).astype(np.int64)
        X = store.matrix(ids, names)
        out.ravel()[ids] = predict_proba(model, X)
    return out


# =============================================================================== experiments
@dataclass
class Experiment:
    name: str
    features: list
    kind: str = "xgboost"
    strategy: str = "standard"
    disc: bool = True  # also produce discovery-split maps
    full: bool = True  # also fit the submission model and predict the full grid

    def fingerprint(self, cfg) -> str:
        return combine(
            self.name,
            self.features,
            self.kind,
            self.strategy,
            self.disc,
            self.full,
            cfg.fingerprint(
                "seed",
                "positive_radius_px",
                "ignore_ring_px",
                "negative_ratio",
                "max_positives",
                "n_estimators",
                "learning_rate",
                "max_depth",
            ),
        )


def run_experiment(exp: Experiment, store, data, sp, cfg, ws, *, hard=None, log=print) -> dict:
    """Train / predict the CV-fold, discovery and full models of one experiment (resumable per unit)."""
    root = ws.path("predictions", exp.name, "x").parent
    fp = exp.fingerprint(cfg)
    stamp_p = root / "experiment.json"
    stamp = json.loads(stamp_p.read_text()) if stamp_p.exists() else {}
    if stamp.get("fingerprint") != fp:
        stamp = {"fingerprint": fp, "units": {}, "features": exp.features, "kind": exp.kind, "strategy": exp.strategy}
    seeds = np.random.SeedSequence([cfg.seed, len(exp.name)])
    unit_seeds = {
        u: int(s.generate_state(1)[0])
        for u, s in zip(
            ["oof"] + [f"cv{k}" for k in sp.cv_folds] + [f"disc{d}" for d in range(sp.n_splits)] + ["full"],
            seeds.spawn(2 + len(sp.cv_folds) + sp.n_splits),
        )
    }
    timings = {}

    def done(u):
        rec = stamp["units"].get(u)
        return rec is not None and Path(rec["file"]).exists()

    def fit_unit(u, train, labels, ignore=None):
        rng = np.random.default_rng(unit_seeds[u])
        smp = draw_sample(
            train,
            labels,
            rng,
            strategy=exp.strategy,
            positive_radius_px=cfg.positive_radius_px,
            ignore_ring_px=cfg.ignore_ring_px,
            negative_ratio=cfg.negative_ratio,
            max_positives=cfg.max_positives,
            ignore=ignore,
            hard_categories=hard,
        )
        X = store.matrix(smp.ids, exp.features)
        t0 = time.perf_counter()
        m = fit(exp.kind, cfg, X, smp.y, smp.w, unit_seeds[u])
        timings[f"{u}_fit_s"] = time.perf_counter() - t0
        save_model(
            m,
            exp.kind,
            ws.path("models", exp.name, u),
            exp.features,
            {"unit": u, "sample": smp.info, "fingerprint": fp, "seed": unit_seeds[u]},
        )
        return m, smp

    # ---- CV folds -> OOF
    oof_p = root / "oof.npy"
    if not done("oof"):
        oof = np.full(data.shape, np.nan, np.float32)
        for k in sp.cv_folds:
            m, _ = fit_unit(f"cv{k}", sp.train(k), sp.labels)
            t0 = time.perf_counter()
            pk = predict_mask(m, store, exp.features, sp.val(k) & data.valid)
            timings[f"cv{k}_predict_s"] = time.perf_counter() - t0
            sel = np.isfinite(pk)
            oof[sel] = pk[sel]
        atomic_save_npy(oof_p, oof)
        stamp["units"]["oof"] = {"file": str(oof_p)}
        atomic_write_json(stamp_p, stamp)
    # ---- discovery splits
    if exp.disc:
        for d in range(sp.n_splits):
            u = f"disc{d}"
            p = root / f"{u}.npy"
            if done(u):
                continue
            m, _ = fit_unit(u, sp.disc_train(d), sp.disc_labels(d), ignore=sp.ignore(d))
            pd_ = predict_mask(m, store, exp.features, sp.eval_zone(d) & data.valid)
            atomic_save_npy(p, pd_)
            stamp["units"][u] = {"file": str(p)}
            atomic_write_json(stamp_p, stamp)
    # ---- submission model (TEST excluded from training)
    if exp.full and not done("full"):
        m, smp = fit_unit("full", sp.train_final(), sp.labels)
        t0 = time.perf_counter()
        full = predict_mask(m, store, exp.features, data.valid)
        timings["full_predict_s"] = time.perf_counter() - t0
        p = root / "full.npy"
        atomic_save_npy(p, full)
        stamp["units"]["full"] = {"file": str(p)}
        if hasattr(m, "feature_importances_"):
            imp = pd.DataFrame({"feature": exp.features, "importance": m.feature_importances_})
            imp.sort_values("importance", ascending=False).to_csv(root / "importance.csv", index=False)
        atomic_write_json(stamp_p, stamp)
    stamp["timings"] = {**stamp.get("timings", {}), **timings}
    atomic_write_json(stamp_p, stamp)
    log(f"[model] {exp.name}: done ({', '.join(f'{k}={v:.0f}s' for k, v in timings.items()) or 'reused'})")
    return load_maps(root, sp)


def load_maps(root: Path, sp) -> dict:
    root = Path(root)
    out = {
        "oof": np.load(root / "oof.npy") if (root / "oof.npy").exists() else None,
        "disc": {d: np.load(root / f"disc{d}.npy") for d in range(sp.n_splits) if (root / f"disc{d}.npy").exists()},
        "full": np.load(root / "full.npy", mmap_mode="r") if (root / "full.npy").exists() else None,
    }
    return out
