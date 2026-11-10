"""Fault contrast: near-fault vs LOCAL control (ring 1-3 km from the same faults, same region), cross-fault
asymmetry, within/between-region heterogeneity, with spatial block-bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.spatial_bootstrap import block_bootstrap


def contrast_table(store, names, fault, dist_m, valid, regions, near_m: float, ring_m=(1000.0, 3000.0),
                   block_px: int = 100, n_boot: int = 300, seed: int = 2026, max_n: int = 60000) -> pd.DataFrame:
    from .spatial import block_sample

    near = valid & (dist_m <= near_m)
    ctrl = valid & (dist_m > ring_m[0]) & (dist_m <= ring_m[1])
    rn, cn, bn = block_sample(near, max(block_px // 10, 1), seed, max_n)
    rc, cc, bc = block_sample(ctrl, max(block_px // 10, 1), seed + 1, max_n)
    coarse = lambda r, c: (r // block_px) * 10000 + c // block_px          # noqa: E731 - bootstrap blocks
    rows = []
    for n in names:
        a = store.get(n)
        xn, xc = np.asarray(a[rn, cn], np.float64), np.asarray(a[rc, cc], np.float64)
        v = np.r_[xn, xc]
        lab = np.r_[np.ones(xn.size), np.zeros(xc.size)]
        blk = np.r_[coarse(rn, cn), coarse(rc, cc)]
        ok = np.isfinite(v)
        if ok.sum() < 50 or lab[ok].min() == lab[ok].max():
            continue

        def stat(idx, v=v[ok], lab=lab[ok]):
            a_, b_ = v[idx][lab[idx] == 1], v[idx][lab[idx] == 0]
            if not a_.size or not b_.size:
                raise ValueError("empty group")
            s = np.sqrt((np.var(a_) + np.var(b_)) / 2) + 1e-12
            return (np.median(a_) - np.median(b_)) / s

        bb = block_bootstrap(stat, blk[ok], n_boot, seed=seed)
        reg_vals = []
        rid = np.r_[regions[rn, cn], regions[rc, cc]][ok]
        for r in np.unique(rid[rid >= 0]):
            m = rid == r
            try:
                reg_vals.append(stat(np.flatnonzero(m)))
            except ValueError:
                pass
        rows.append({"feature": n, "near_vs_local_control": bb["estimate"], "ci_low": bb["ci_low"],
                     "ci_high": bb["ci_high"], "n_blocks": bb["n_blocks"],
                     "between_region_sd": float(np.std(reg_vals)) if len(reg_vals) > 1 else np.nan,
                     "sign_consistency": float(np.mean(np.sign(reg_vals) == np.sign(bb["estimate"]))) if reg_vals else np.nan})
    return pd.DataFrame(rows)


def cross_fault_asymmetry(store, names, segments, grid, offset_m: float = 500.0, step_m: float = 200.0) -> pd.DataFrame:
    """Median |left - right| difference of each feature sampled at +-offset along segment normals."""
    from ..grid import inside, world_to_pixel

    rows = []
    for n in names:
        a = store.get(n)
        diffs = []
        for g in segments.geometry:
            L = g.length
            for t in np.arange(step_m / 2, L, step_m):
                p, q = g.interpolate(max(t - 50, 0)), g.interpolate(min(t + 50, L))
                tx, ty = q.x - p.x, q.y - p.y
                nn = np.hypot(tx, ty) + 1e-12
                nx, ny = ty / nn, -tx / nn
                c = g.interpolate(t)
                xs, ys = np.array([c.x + offset_m * nx, c.x - offset_m * nx]), np.array([c.y + offset_m * ny,
                                                                                         c.y - offset_m * ny])
                r, cc = world_to_pixel(xs, ys, grid)
                if inside(r, cc, grid).all():
                    v = np.asarray(a[r, cc], np.float64)
                    if np.isfinite(v).all():
                        diffs.append(abs(v[0] - v[1]))
        rows.append({"feature": n, "cross_fault_abs_diff_median": float(np.median(diffs)) if diffs else np.nan,
                     "n_stations": len(diffs)})
    return pd.DataFrame(rows)
