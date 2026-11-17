"""Model disagreement and adaptive geological mixture-of-experts (sections 27-28).

Experts enter as per-zone rank maps (scale-free). Three fusions are compared under identical units:

* ``rank_mean``      plain average of expert ranks,
* ``fixed_weights``  a weight vector on a simplex grid, cross-fitted (chosen on the other zones),
* ``gated``          a small LightGBM gate on [expert ranks, context] trained on *out-of-fold* expert predictions of
                     the other CV folds (geo) or on all OOF with the split's withheld systems removed from
                     supervision and their ignore ring dropped (disc). The gate is kept only if it beats both
                     simpler fusions on held-out DTI.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..gems3.representation import rank01
from .evaluation import area_table, crossfit
from .models import draw_sample


def zone_rank(p: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    out = np.full(p.shape, np.nan, np.float32)
    m = zone_mask & np.isfinite(p)
    if m.any():
        out[m] = rank01(np.where(m, p, 0), m)[m]
    return out


def rank_maps(experts: dict, sp, data) -> dict:
    """expert -> {'oof': rank per CV fold zone, 'disc': {d: rank on eval zone}, 'full': rank on valid}."""
    out = {}
    for name, m in experts.items():
        r = {"oof": None, "disc": {}, "full": None}
        if m.get("oof") is not None:
            oof = np.full(data.shape, np.nan, np.float32)
            for k in sp.cv_folds:
                z = sp.val(k) & data.valid
                rk = zone_rank(m["oof"], z)
                oof[z] = rk[z]
            r["oof"] = oof
        for d, p in (m.get("disc") or {}).items():
            r["disc"][d] = zone_rank(p, sp.eval_zone(d) & data.valid)
        if m.get("full") is not None:
            r["full"] = zone_rank(np.asarray(m["full"]), data.valid)
        out[name] = r
    return out


def disagreement(ranks: dict, key: str = "full") -> np.ndarray:
    stack = np.stack([r[key] for r in ranks.values() if r.get(key) is not None])
    return np.nanstd(stack, 0).astype(np.float32)


def weighted(ranks: dict, names: list[str], w: tuple, key: str, d: int | None = None) -> np.ndarray:
    acc = None
    for n, wi in zip(names, w):
        a = ranks[n][key] if d is None else ranks[n]["disc"].get(d)
        a = np.nan_to_num(a, nan=0.0) * wi
        acc = a if acc is None else acc + a
    return acc.astype(np.float32)


def simplex(n: int, step: float = 0.25) -> list[tuple]:
    pts = [p for p in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=n) if abs(sum(p) - 1) < 1e-9]
    return [tuple(float(x) for x in p) for p in pts]


def fixed_weight_search(ranks: dict, names: list[str], sp, fracs, step: float = 0.25) -> dict:
    """Cross-fitted (weights, area) choice per task. Returns tables and the chosen weights per task."""
    units_geo, units_disc = sp.units(("geo",)), sp.units(("disc_official",))
    res = {}
    W = simplex(len(names), step)
    for task, units in (("geo", units_geo), ("disc_official", units_disc)):
        rows = []
        for w in W:
            if task == "geo":
                if any(ranks[n]["oof"] is None for n in names):
                    break
                s = weighted(ranks, names, w, "oof")
                t = area_table(s, units, fracs)
            else:
                parts = []
                for u in units:
                    if any(ranks[n]["disc"].get(u.zone) is None for n in names):
                        continue
                    parts.append(area_table(weighted(ranks, names, w, "disc", u.zone), [u], fracs))
                if not parts:
                    break
                t = pd.concat(parts)
            t["weights"] = str(w)
            rows.append(t)
        if not rows:
            continue
        tab = pd.concat(rows, ignore_index=True)
        # identity-first ordering: equal weights first so ties prefer the simplest fusion
        eq = str(tuple(1.0 / len(names) for _ in names))
        tab["_o"] = (tab.weights != eq).astype(int)
        tab = tab.sort_values(["_o", "weights", "target_frac"]).drop(columns="_o")
        per, summ = crossfit(tab, ["weights", "target_frac"])
        res[task] = {"table": tab, "per_zone": per, "summary": summ}
    return res


def context_layers(store, data, regimes: np.ndarray | None) -> dict[str, np.ndarray]:
    ctx = {}
    for nm in (
        "magnetic:rtp:coh_s1600",
        "gravity:iso_grav_anom:coh_s1600",
        "cross_modal:all:agreement_frac",
        "validity:all:boundary_dist_km",
        "regional_norm:rtp:localz_8km",
    ):
        if store.has(nm):
            ctx[nm] = store.get(nm)
    if regimes is not None:
        ctx["regime"] = regimes.astype(np.float32)
    return ctx


def gate_fit(
    rank_layers: list[np.ndarray],
    ctx: dict,
    labels: np.ndarray,
    train: np.ndarray,
    cfg,
    rng,
    ignore: np.ndarray | None = None,
):
    from lightgbm import LGBMClassifier

    smp = draw_sample(
        train,
        labels,
        rng,
        strategy="standard",
        positive_radius_px=cfg.positive_radius_px,
        ignore_ring_px=cfg.ignore_ring_px,
        negative_ratio=cfg.negative_ratio,
        max_positives=min(cfg.max_positives, 60_000),
        ignore=ignore,
    )
    X = _gate_X(rank_layers, ctx, smp.ids)
    m = LGBMClassifier(
        n_estimators=150,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=int(rng.integers(1 << 30)),
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    m.fit(X, smp.y, sample_weight=smp.w)
    return m


def _gate_X(rank_layers, ctx, ids):
    cols = [np.asarray(r, np.float32).ravel()[ids] for r in rank_layers]
    st = np.stack(cols, 1)
    cols += [np.nanstd(st, 1), np.nanmax(st, 1)]
    cols += [np.asarray(a, np.float32).ravel()[ids] for a in ctx.values()]
    return np.stack(cols, 1)


def gate_predict(m, rank_layers, ctx, mask) -> np.ndarray:
    out = np.full(mask.shape, np.nan, np.float32)
    ids = np.flatnonzero(mask.ravel())
    for s in range(0, len(ids), 1_000_000):
        i = ids[s : s + 1_000_000]
        out.ravel()[i] = m.predict_proba(_gate_X(rank_layers, ctx, i))[:, 1]
    return out


def gated_maps(ranks: dict, names: list[str], ctx: dict, sp, data, cfg, rng) -> dict:
    """Leakage-safe gated fusion maps (OOF per CV fold, per discovery split, and full)."""
    oof = np.full(data.shape, np.nan, np.float32)
    layers_oof = [ranks[n]["oof"] for n in names]
    have_oof = np.ones(data.shape, bool)
    for r in layers_oof:
        have_oof &= np.isfinite(r)
    for k in sp.cv_folds:
        tr = have_oof & ~sp.val(k) & ~sp.buffer(k) & sp.cv_region()
        g = gate_fit(layers_oof, ctx, sp.labels, tr, cfg, rng)
        z = sp.val(k) & have_oof
        oof[z] = gate_predict(g, layers_oof, ctx, z)[z]
    disc = {}
    for d in range(sp.n_splits):
        if any(ranks[n]["disc"].get(d) is None for n in names):
            continue
        g = gate_fit(layers_oof, ctx, sp.disc_labels(d), have_oof & sp.cv_region(), cfg, rng, ignore=sp.ignore(d))
        layers = [ranks[n]["disc"][d] for n in names]
        z = sp.eval_zone(d) & data.valid
        disc[d] = gate_predict(g, layers, ctx, z)
    full = None
    if all(ranks[n]["full"] is not None for n in names):
        g = gate_fit(layers_oof, ctx, sp.labels, have_oof & sp.cv_region(), cfg, rng)
        full = gate_predict(g, [ranks[n]["full"] for n in names], ctx, data.valid)
    return {"oof": oof, "disc": disc, "full": full}


def disagreement_candidates(dis: np.ndarray, data, q: float = 0.98, min_px: int = 8) -> pd.DataFrame:
    """Connected regions of the top-(1-q) expert disagreement (ranked, with UTM centroid)."""
    from scipy import ndimage

    v = data.valid & np.isfinite(dis)
    if not v.any():
        return pd.DataFrame()
    m = v & (dis >= np.quantile(dis[v], q))
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return pd.DataFrame()
    idx = np.arange(1, n + 1)
    size = ndimage.sum(m, lab, idx)
    mean = ndimage.mean(dis, lab, idx)
    com = ndimage.center_of_mass(m, lab, idx)
    t = data.transform
    df = pd.DataFrame(
        {
            "region": idx,
            "n_px": size,
            "mean_disagreement": mean,
            "x_utm": [t.c + (c + 0.5) * t.a for _, c in com],
            "y_utm": [t.f + (r + 0.5) * t.e for r, _ in com],
        }
    )
    return df[df.n_px >= min_px].sort_values("mean_disagreement", ascending=False).reset_index(drop=True)
