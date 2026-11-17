"""Failure-mode and error taxonomy on held-out predictions (section 17).

Every category is a *measured* quantity on the validation units at the selected representation; remedies are
linked to the tournament operation that tests them, so "does the remedy work" is answered by the cross-fitted
tournament table, not asserted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..geology.skeleton import branch_points, skeletonize

S8 = np.ones((3, 3), bool)

REMEDIES = {
    "missed_faults": ("lower area budget threshold / multiscale features", "top|frac grid"),
    "missed_weak_faults": ("persistence and cross-modal features", "ablation A6/A8"),
    "broken_continuity": ("orientation-aware gap linking", "top_link4"),
    "displaced_predictions": ("crest (NMS) localisation", "crest_w1"),
    "thick_ribbons": ("crest lines + width 1", "crest_w1"),
    "isolated_noise": ("minimum-size filtering", "top_remove_small6"),
    "fp_terrain_ridge": ("hard-negative sampling", "pu_learning hard_negative"),
    "fp_drainage_valley": ("hard-negative sampling", "pu_learning hard_negative"),
    "fp_nodata_boundary": ("validity features / boundary audit", "quality boundary_dependence"),
    "fp_near_known": ("known-fault ring", "ring=1/2"),
    "spurious_junctions": ("topology-aware ranking", "top_link4 / remove_small"),
    "regional_shift": ("regime-balanced budgets", "top_regime_balanced"),
}


def taxonomy(units, q_by_unit: dict, archetype_map: np.ndarray | None = None, hard: dict | None = None) -> pd.DataFrame:
    """``q_by_unit``: unit key -> cropped submitted map of the chosen representation."""
    rows = []
    for u in units:
        q = np.asarray(q_by_unit[u.key], np.float64)
        sc = u.scorer
        p = q.copy()
        p[sc.zero] = 0.0
        sup = sc.support(p, cleaned=True)
        gr, gc = sc.g_rc
        n = len(sup)
        pred = p > 0
        fpw = p * sc.fp_w
        fp_tot = float(fpw.sum())
        r = {"task": u.task, "zone": u.zone, "n_truth": n, "fp_w": fp_tot}
        if n:
            r["missed_faults"] = float((sup == 0).mean())
            r["displaced_predictions"] = float(((sup > 0) & (sup < 0.99)).mean())
            if archetype_map is not None:
                a = u.crop(archetype_map)[gr, gc]
                weak = a == 6
                r["missed_weak_faults"] = float((sup[weak] == 0).mean()) if weak.any() else np.nan
            tl, tn = ndimage.label(u.eval_truth(), structure=S8)
            pl, _ = ndimage.label(pred, structure=S8)
            if tn:
                frag = []
                for i, sl in enumerate(ndimage.find_objects(tl), 1):
                    if sl is None:
                        continue
                    t = tl[sl] == i
                    near = ndimage.binary_dilation(t, S8, iterations=2)
                    ids = np.unique(pl[sl][near & (pl[sl] > 0)])
                    if t.sum() >= 10 and len(ids):
                        frag.append(len(ids))
                r["broken_continuity"] = float(np.mean(np.array(frag) > 1)) if frag else np.nan
        if pred.any():
            pl, pn = ndimage.label(pred, structure=S8)
            sizes = np.bincount(pl.ravel())[1:]
            small = np.isin(pl, np.flatnonzero(sizes < 4) + 1)
            r["isolated_noise"] = float(fpw[small].sum() / max(fp_tot, 1e-12))
            sk = skeletonize(pred)
            width = pred.sum() / max(sk.sum(), 1)
            r["mean_ribbon_width_px"] = float(width)
            r["thick_ribbons"] = float(width > 2.5)
            junc = branch_points(sk)
            if n:
                dt = ndimage.distance_transform_edt(~u.eval_truth())
                r["spurious_junctions"] = float((junc & (dt > 10)).sum() / max(junc.sum(), 1))
            for name, m in (hard or {}).items():
                mm = u.crop(m)
                r[f"fp_{name}"] = float(fpw[mm].sum() / max(fp_tot, 1e-12))
            if u.known is not None and u.known.any():
                dk = ndimage.distance_transform_edt(~u.known)
                r["fp_near_known"] = float(fpw[dk <= 10].sum() / max(fp_tot, 1e-12))
        u.release()
        rows.append(r)
    per = pd.DataFrame(rows)
    return per


def summary(per: pd.DataFrame) -> pd.DataFrame:
    if per.empty:
        return per
    cats = [c for c in per.columns if c not in ("task", "zone", "n_truth", "fp_w", "mean_ribbon_width_px")]
    rows = []
    for task, g in per.groupby("task"):
        for c in cats:
            if g[c].notna().any():
                w = g.n_truth if not c.startswith(("fp_", "isolated", "thick", "spurious")) else g.fp_w
                v = np.average(g[c].fillna(0), weights=np.maximum(w, 1e-9))
                rem = REMEDIES.get(c, REMEDIES.get(c.replace("fp_", "fp_"), ("", "")))
                rows.append(
                    {
                        "task": task,
                        "category": c,
                        "value": float(v),
                        "zones": int(g[c].notna().sum()),
                        "remedy": rem[0],
                        "tested_by": rem[1],
                    }
                )
        rows.append(
            {
                "task": task,
                "category": "regional_shift",
                "value": np.nan,
                "zones": len(g),
                "remedy": REMEDIES["regional_shift"][0],
                "tested_by": REMEDIES["regional_shift"][1],
            }
        )
    return pd.DataFrame(rows)


def example_windows(units, q_by_unit: dict, k: int = 3, size: int = 80) -> list[dict]:
    """Windows with the largest missed-truth and false-positive mass (for panels)."""
    out = []
    for u in units:
        q = np.asarray(q_by_unit[u.key], np.float64)
        sc = u.scorer
        p = q.copy()
        p[sc.zero] = 0
        sup = sc.support(p, cleaned=True)
        miss = np.zeros(q.shape)
        if len(sup):
            miss[sc.g_rc] = 1 - sup
        fp = p * sc.fp_w
        for kind, a in (("missed", miss), ("false_positive", fp)):
            agg = ndimage.uniform_filter(a, size)
            for _ in range(k):
                i = int(np.argmax(agg))
                r, c = np.unravel_index(i, agg.shape)
                if agg[r, c] <= 0:
                    break
                out.append(
                    {
                        "task": u.task,
                        "zone": u.zone,
                        "kind": kind,
                        "row": int(r + u.sl[0].start),
                        "col": int(c + u.sl[1].start),
                        "mass": float(agg[r, c] * size * size),
                    }
                )
                agg[max(0, r - size) : r + size, max(0, c - size) : c + size] = 0
        u.release()
    return out
