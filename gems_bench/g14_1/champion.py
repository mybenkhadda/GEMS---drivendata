"""Champion comparison, complementarity and residual recovery (Phase J).

Compatibility rules (enforced):

* scored comparisons only between maps of the SAME split backend and units (``g14.evaluation.load_g13_model``
  refuses the phase5 backend; ``load_unit_maps`` refuses crops of other units);
* full-region maps are compared label-free only (correlation, rank correlation, top-area overlap, placement);
* every blend / gate weight for zone z is chosen on the OTHER zones of the task (cross-fitted), and the champion
  alone (weight 0) is always a candidate, so a blend can only be selected if it beats the champion out of zone.

UNTESTED: written without execution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HIT = 0.5          # truth pixel counted as recovered when its TP support (max p k) >= 0.5 (within ~150 m of p = 1)


# =============================================================================== loading
def read_map(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, mmap_mode="r")
    import rasterio

    with rasterio.open(p) as s:
        return s.read(1, masked=True).astype(np.float32).filled(np.nan)


def g13_champions(system, g13_models: dict, scoreboard: str | Path | None, top: int = 3,
                  metric: str = "disc_proxy_dti") -> dict[str, dict]:
    """Top-``top`` G13 models by their own scoreboard, loaded on the g13 backend units."""
    from ..g14.evaluation import load_g13_model

    names = list(g13_models)
    if scoreboard is not None and Path(scoreboard).exists():
        b = pd.read_csv(scoreboard)
        if metric in b:
            names = [m for m in b.sort_values(metric, ascending=False).model if m in g13_models][:top]
    out = {}
    for m in names[:top]:
        any_path = next(iter(g13_models[m].values()))
        out[f"G13_{m}"] = load_g13_model(Path(any_path).parent, system)
    return out


def unit_maps_from_paths(paths: dict[str, Path]) -> dict:
    """{'geo_1': path, ...} -> {(task, zone): array} (G14 / G14.1 held-out map files)."""
    out = {}
    for k, p in paths.items():
        stem = k.split("__")[-1]
        task, zone = stem.rsplit("_", 1)
        out[(task, int(zone))] = np.asarray(read_map(p), np.float32)
    return out


# =============================================================================== label-free comparison
def label_free(a: np.ndarray, b: np.ndarray, valid: np.ndarray, known: np.ndarray, top_fracs=(0.005, 0.02, 0.08),
               n_sample: int = 2_000_000, seed: int = 0) -> dict:
    from scipy.stats import spearmanr

    m = valid & np.isfinite(a) & np.isfinite(b) & ~known
    x, y = np.asarray(a, np.float64)[m], np.asarray(b, np.float64)[m]
    if x.size > n_sample:
        i = np.random.default_rng(seed).choice(x.size, n_sample, replace=False)
        xs, ys = x[i], y[i]
    else:
        xs, ys = x, y
    out = {"n_px": int(m.sum()), "pearson": float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 2 else np.nan,
           "spearman": float(spearmanr(xs, ys)[0]) if xs.size > 2 else np.nan}
    for f in top_fracs:
        k = max(int(np.ceil(f * x.size)), 1)
        ta = x >= np.partition(x, x.size - k)[x.size - k]
        tb = y >= np.partition(y, y.size - k)[y.size - k]
        out[f"top{f}_jaccard"] = float((ta & tb).sum() / max((ta | tb).sum(), 1))
        out[f"top{f}_share_of_a_in_b"] = float((ta & tb).sum() / max(ta.sum(), 1))
    return out


# =============================================================================== scored complementarity
def _rank(s, ev):
    from ..gems3.representation import rank01

    return rank01(np.where(ev, np.nan_to_num(s), 0.0), ev)


def pixel_classes(u, champ_crop: np.ndarray, g14_crop: np.ndarray, frac: float, pixel_m: float = 100.0) -> dict:
    """Truth-pixel and prediction-pixel classes at the same top-``frac`` budget for both models."""
    from .representations import apply_rep

    sc = u.scorer
    qa, _ = apply_rep("top", champ_crop, u.ev, u.exclude, u.scored_mask(), frac)
    qb, _ = apply_rep("top", g14_crop, u.ev, u.exclude, u.scored_mask(), frac)
    sa, sb = sc.support(qa.astype(np.float64)), sc.support(qb.astype(np.float64))
    ha, hb = sa >= HIT, sb >= HIT
    truth = u.eval_truth()
    far = (ndimage.distance_transform_edt(~truth) * pixel_m > 300.0) if truth.any() else np.ones(truth.shape, bool)
    fa, fb = (qa > 0) & far, (qb > 0) & far
    out = {"task": u.task, "zone": u.zone, "frac": frac, "n_truth": int(len(sa)),
           "both_tp": int((ha & hb).sum()), "champ_tp_g14_miss": int((ha & ~hb).sum()),
           "champ_miss_g14_tp": int((~ha & hb).sum()), "both_miss": int((~ha & ~hb).sum()),
           "g14_only_fp_px": int((fb & ~fa).sum()), "champ_only_fp_px": int((fa & ~fb).sum()),
           "both_fp_px": int((fa & fb).sum())}
    # fault-system recall overlap (truth components)
    lab, n = ndimage.label(truth, structure=np.ones((3, 3), bool))
    if n:
        gr, gc = sc.g_rc
        ids = lab[gr, gc]
        ra = np.array([sa[ids == i].mean() for i in range(1, n + 1)])
        rb = np.array([sb[ids == i].mean() for i in range(1, n + 1)])
        out.update(systems=int(n), systems_both=int(((ra >= 0.3) & (rb >= 0.3)).sum()),
                   systems_champ_only=int(((ra >= 0.3) & (rb < 0.3)).sum()),
                   systems_g14_only=int(((ra < 0.3) & (rb >= 0.3)).sum()))
    u.release()
    return out


def residual_maps(champ_crop, g14_crop, ev, w: float, kind: str) -> np.ndarray:
    """Champion-first combinations (ranks inside the unit's evaluation mask, label-free)."""
    rc, rg = _rank(champ_crop, ev), _rank(g14_crop, ev)
    if kind == "blend":
        return ((1 - w) * rc + w * rg).astype(np.float32)
    if kind == "confirm":                                # G14 can only re-order inside the champion's ranking scale
        return (rc * (1 + w * (rg - 0.5))).astype(np.float32)
    if kind == "rerank_pool":                            # only the champion's top 10 % pool can be re-ordered
        pool = rc >= 0.90
        return np.where(pool, 0.9 + 0.1 * ((1 - w) * (rc - 0.9) / 0.1 + w * rg), rc).astype(np.float32)
    if kind == "reject_fp":                              # champion components with weak G14 support are demoted
        cand = rc >= 0.95
        lab, n = ndimage.label(cand, structure=np.ones((3, 3), bool))
        out = rc.copy()
        if n:
            sup = ndimage.mean(rg, lab, np.arange(1, n + 1))
            weak = np.zeros(n + 1, bool)
            weak[1:] = sup < np.quantile(sup, min(w, 0.99)) if n > 1 else False
            out = np.where(weak[lab], rc * 0.9, rc)
        return out.astype(np.float32)
    raise KeyError(kind)


def blend_sweep(champ: dict, g14: dict, units: list, weights, kinds=("blend", "confirm", "rerank_pool"),
                fracs=(0.005, 0.01, 0.02, 0.05), rep: str = "top") -> pd.DataFrame:
    """Components for every (kind, weight, frac) on every unit where both maps exist."""
    from .representations import apply_rep

    rows = []
    for u in units:
        k = (u.task, u.zone)
        if k not in champ or k not in g14:
            continue
        a, b = np.asarray(u.crop(champ[k]), np.float32), np.asarray(u.crop(g14[k]), np.float32)
        for kind in kinds:
            for w in weights:
                if w == 0 and kind != kinds[0]:
                    continue                               # weight 0 = champion alone, scored once
                s = residual_maps(a, b, u.ev, w, kind)
                for fr in fracs:
                    q, acc = apply_rep(rep, s, u.ev, u.exclude, u.scored_mask(), fr)
                    c = u.components(q)
                    rows.append({"task": u.task, "zone": u.zone, "kind": kind if w else "champion", "w": w,
                                 "target_frac": fr, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "dti": c.dti})
        u.release()
    return pd.DataFrame(rows)


def crossfit_blend(table: pd.DataFrame) -> dict:
    """Cross-fitted (kind, w, frac) per task, plus the cross-fitted champion-only reference and the paired delta."""
    from ..g13.evaluation import crossfit

    out = {}
    for task, t in table.groupby("task"):
        per, summ = crossfit(t, ["kind", "w", "target_frac"])
        ref = t[t.kind == "champion"]
        per0, summ0 = crossfit(ref, ["target_frac"]) if len(ref) else (None, {"dti_crossfit": np.nan})
        out[task] = {"per_zone": per, "summary": summ, "champion_crossfit": summ0["dti_crossfit"],
                     "delta_vs_champion": summ["dti_crossfit"] - summ0["dti_crossfit"],
                     "selected_weights": per[["zone", "kind", "w", "target_frac"]].to_dict("records")}
    return out


def context_strata(rasters: dict[str, np.ndarray], valid: np.ndarray, n_bins: int = 3) -> dict[str, np.ndarray]:
    """Label-free conditioning strata (tertiles) for J3: magnetic / gravity / terrain dominance, conductivity
    evidence, structural coherence, distance to known faults, model uncertainty..."""
    out = {}
    for k, r in rasters.items():
        a = np.asarray(r, np.float32)
        m = valid & np.isfinite(a)
        if not m.any():
            continue
        qs = np.quantile(a[m], np.linspace(0, 1, n_bins + 1)[1:-1])
        s = np.full(a.shape, -1, np.int16)
        s[m] = np.digitize(a[m], qs).astype(np.int16)
        out[k] = s
    return out


def conditional_blend(champ: dict, g14: dict, units: list, strata: np.ndarray, weights, frac: float) -> pd.DataFrame:
    """J3: one G14 weight per stratum, chosen for zone z on the other zones only (never z's labels); the blended
    map of z then gets its top-``frac`` budget. Returns per-zone held-out components and the chosen weights."""
    from ..g13.evaluation import dti_of
    from ..metric import dti_components
    from .representations import _top, apply_rep

    rows = []
    cache = {}
    for u in units:
        k = (u.task, u.zone)
        if k in champ and k in g14:
            cache[k] = (u, _rank(u.crop(champ[k]), u.ev), _rank(u.crop(g14[k]), u.ev), u.crop(strata))
    by_task: dict = {}
    for k, v in cache.items():
        by_task.setdefault(k[0], []).append(v)
    for task, items in by_task.items():
        # components per (zone, stratum, weight) of the stratum-restricted top-frac mask
        comp = []
        for u, rc, rg, st in items:
            for g in np.unique(st[u.ev & (st >= 0)]):
                ev_g = u.ev & (st == g)
                for w in weights:
                    s = (1 - w) * rc + w * rg
                    q = _top(s, ev_g & u.scored_mask(), max(int(np.ceil(frac * ev_g.sum())), 1)).astype(np.float64)
                    c = dti_components(q, u.truth, eval_mask=ev_g, exclude_mask=u.exclude, crop=False,
                                       exclude_supports_tp=False)
                    comp.append({"zone": u.zone, "stratum": int(g), "w": w, "tp": c.tp, "fp": c.fp, "fn": c.fn})
        C = pd.DataFrame(comp)
        for u, rc, rg, st in items:
            other = C[C.zone != u.zone] if (C.zone != u.zone).any() else C
            agg = other.groupby(["stratum", "w"])[["tp", "fp", "fn"]].sum().reset_index()
            agg["dti"] = [dti_of(a, b, c) for a, b, c in zip(agg.tp, agg.fp, agg.fn)]
            best = agg.loc[agg.groupby("stratum").dti.idxmax()].set_index("stratum").w.to_dict()
            wmap = np.zeros(st.shape, np.float32)
            for g, w in best.items():
                wmap[st == g] = w
            s = (1 - wmap) * rc + wmap * rg
            q, _ = apply_rep("top", s, u.ev, u.exclude, u.scored_mask(), frac)
            c = u.components(q)
            rows.append({"task": task, "zone": u.zone, "frac": frac, "weights_by_stratum": str(best),
                         "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "dti": c.dti})
            u.release()
    return pd.DataFrame(rows)
