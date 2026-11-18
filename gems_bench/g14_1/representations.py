"""Representation catalogue with exact area accounting and explicit known-fault masking order (Phases G2, G3, G5).

Every representation maps a held-out score crop to the prediction raster the official scorer would see and returns an
accounting record, so that "8 % area" always says *which* 8 %:

``requested_frac``  budget asked for (share of the eligible domain)
``k_requested``     ceil(requested_frac x eligible pixels)
``selected_px``     pixels selected before any post-operation (e.g. crest pixels before dilation)
``final_px``        pixels with a positive value after post-operations (before the scorer zeroes known faults)
``scored_px``       positive pixels the scorer counts (known faults and nodata removed)
``on_known_px``     positive pixels that fall on pixel-exact known faults (wasted, zeroed by the scorer)
``final_frac``      scored_px / eligible pixels      ``mass``  sum of values on scored pixels

Masking orders (G5):

``pixel_exact``  G14 behaviour: representation on the unmasked score, known pixels excluded from the budget only
``mask_before``  known pixels set to 0 in the score before any representation (crests recomputed on the masked map)
``mask_after``   representation on the unmasked score with known pixels competing for the budget; zeroed afterwards
``mask_both``    mask_before + zero known pixels again after post-operations (dilation cannot re-enter them)
``ring1``        mask_before + a 1-pixel ring around known faults removed from the budget (justification required:
                 the official mask is pixel-exact, a ring only removes FP mass, never TP support on the ring itself)

UNTESTED: written without execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

SOFT_UNBUDGETED = ("raw", "power2", "power4")
BUDGETED = ("top", "crest_w1", "crest_w0", "nms", "skeleton", "soft_skeleton", "sharpen", "hysteresis",
            "segment_rank", "sparse_soft", "p6_skel_w1")
ALL = SOFT_UNBUDGETED + BUDGETED
ORDERS = ("pixel_exact", "mask_before", "mask_after", "mask_both", "ring1")


def _k(frac: float, n: int) -> int:
    return max(int(np.ceil(frac * n)), 1)


def _top(s: np.ndarray, dom: np.ndarray, k: int) -> np.ndarray:
    v = s[dom]
    if not v.size:
        return np.zeros(s.shape, bool)
    k = min(k, v.size)
    thr = np.partition(v, v.size - k)[v.size - k]
    return dom & (s >= thr)


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def hysteresis(s: np.ndarray, dom: np.ndarray, k_hi: int, k_lo: int) -> np.ndarray:
    hi, lo = _top(s, dom, k_hi), _top(s, dom, k_lo)
    lab, n = ndimage.label(lo, structure=np.ones((3, 3), bool))
    if n == 0:
        return hi
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[hi])] = True
    keep[0] = False
    return keep[lab]


def segment_rank(s: np.ndarray, dom: np.ndarray, k: int, pool_factor: float = 2.5) -> np.ndarray:
    """Candidate components of the top (pool_factor x k) pixels ranked by mean score x sqrt(length); kept in order
    until the budget k is used."""
    cand = _top(s, dom, int(pool_factor * k))
    lab, n = ndimage.label(cand, structure=np.ones((3, 3), bool))
    if n == 0:
        return cand
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(s), lab, idx)
    mean = ndimage.mean(s, lab, idx)
    order = np.argsort(-(mean * np.sqrt(size)))
    keep, used = np.zeros(n + 1, bool), 0
    for o in order:
        if used >= k:
            break
        keep[o + 1] = True
        used += int(size[o])
    return keep[lab]


def apply_rep(name: str, score: np.ndarray, ev: np.ndarray, known: np.ndarray | None, scored: np.ndarray,
              frac: float | None, order: str = "pixel_exact", crest: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """One representation on one (cropped) unit. ``scored``: pixels the scorer counts (ev minus known faults)."""
    from ..gems3.representation import crest_map
    from ..geology.skeleton import skeletonize

    if name not in ALL:
        raise KeyError(f"unknown representation {name!r}; known {ALL}")
    if order not in ORDERS:
        raise KeyError(order)
    if name in BUDGETED and frac is None:
        raise ValueError(f"representation {name!r} needs an area fraction")
    known = np.zeros(ev.shape, bool) if known is None else np.asarray(known, bool)
    s = np.where(ev, np.nan_to_num(np.asarray(score, np.float32), nan=0.0), 0.0).astype(np.float32)
    if order in ("mask_before", "mask_both", "ring1"):
        s = np.where(known, 0.0, s).astype(np.float32)
        crest = None                                            # crests must be recomputed on the masked score
    dom = (ev & ~known) if order != "mask_after" else ev.copy()
    if order == "ring1":
        dom &= ~(ndimage.binary_dilation(known, structure=_disk(1)) & ~known)
    dom &= scored | known if order == "mask_after" else scored
    n_dom = int(dom.sum())
    k = _k(frac, n_dom) if frac is not None and name in BUDGETED else None
    sel = None
    if name == "raw":
        q = np.clip(s, 0, 1)
    elif name == "power2":
        q = np.clip(s, 0, 1) ** 2
    elif name == "power4":
        q = np.clip(s, 0, 1) ** 4
    elif name == "top":
        q = _top(s, dom, k).astype(np.float32)
    elif name in ("crest_w1", "crest_w0", "nms"):
        c = crest if (crest is not None and name != "nms") else crest_map(s, ev, 1.0 if name != "nms" else 0.0)
        cd = dom & c
        sel = _top(s, cd, min(k, int(cd.sum())) if cd.any() else 1) if cd.any() else np.zeros(s.shape, bool)
        q = sel.astype(np.float32)
        if name == "crest_w1":
            q = ndimage.binary_dilation(sel, structure=_disk(1)).astype(np.float32)
    elif name == "skeleton":
        sel = _top(s, dom, k)
        q = skeletonize(sel).astype(np.float32)
    elif name == "soft_skeleton":
        sel = _top(s, dom, k)
        q = (skeletonize(sel) * np.clip(s, 0, 1)).astype(np.float32)
    elif name == "sharpen":
        sh = np.clip(s + 1.0 * (s - ndimage.gaussian_filter(s, 2.0)), 0, 1)
        sel = _top(sh, dom, k)
        q = np.where(sel, sh, 0.0).astype(np.float32)
    elif name == "hysteresis":
        q = hysteresis(s, dom, k, int(2.5 * k)).astype(np.float32)
    elif name == "segment_rank":
        q = segment_rank(s, dom, k).astype(np.float32)
    elif name == "sparse_soft":
        sel = _top(s, dom, k)
        q = np.where(sel, np.clip(s, 0, 1), 0.0).astype(np.float32)
    elif name == "p6_skel_w1":
        sel = _top(s, dom, k)
        q = ndimage.binary_dilation(skeletonize(sel), structure=_disk(1)).astype(np.float32)
    else:  # pragma: no cover
        raise KeyError(name)
    q = np.where(ev, q, 0.0).astype(np.float32)
    pos_all = q > 0
    on_known = int((pos_all & known).sum())
    # the official scorer zeroes pixel-exact known faults; doing it here changes no score and keeps exports honest
    q = np.where(known, 0.0, q).astype(np.float32)
    pos = q > 0
    acc = {"rep": name, "order": order, "requested_frac": np.nan if k is None else float(frac),
           "k_requested": k if k is not None else -1, "eligible_px": n_dom,
           "selected_px": int(sel.sum()) if sel is not None else int(pos_all.sum()),
           "final_px": int(pos_all.sum()), "on_known_px": on_known, "scored_px": int((pos & scored).sum()),
           "final_frac": float((pos & scored).sum() / max(n_dom, 1)), "mass": float(q[scored].sum()),
           "dilation_factor": float(pos_all.sum() / max(int(sel.sum()), 1)) if sel is not None else np.nan}
    return q, acc


def sweep_unit(u, s_full: np.ndarray, reps, fracs, orders) -> pd.DataFrame:
    """Official components for every (representation, area, masking order) on one gems3 Unit."""
    s = np.asarray(u.crop(s_full), np.float32)
    known = u.exclude if u.exclude is not None else (u.known if u.known is not None else None)
    scored = u.scored_mask()
    rows = []
    for order in orders:
        crest = None
        if order in ("pixel_exact", "mask_after") and any(r in ("crest_w1", "crest_w0") for r in reps):
            from ..gems3.representation import crest_map

            crest = crest_map(np.where(u.ev, np.nan_to_num(s), 0.0), u.ev)
        for rep in reps:
            for fr in (fracs if rep in BUDGETED else [None]):
                q, acc = apply_rep(rep, s, u.ev, known, scored, fr, order, crest)
                c = u.components(q)
                rows.append({"task": u.task, "zone": u.zone, **acc, "target_frac": -1.0 if fr is None else fr,
                             "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "n_truth": c.n_truth, "dti": c.dti})
    u.release()
    return pd.DataFrame(rows)


def crossfit_reps(table: pd.DataFrame, params=("rep", "order", "target_frac")) -> dict:
    """Cross-fitted choice per task (G13 ``crossfit``: chosen on the other zones, scored on the zone)."""
    from ..g13.evaluation import crossfit

    out = {}
    for task, t in table.groupby("task"):
        per, summ = crossfit(t.assign(rep=t.rep.astype(str), order=t.order.astype(str)), list(params))
        out[task] = {"per_zone": per, "summary": summ}
    return out


def best_per_rep(table: pd.DataFrame) -> pd.DataFrame:
    """Cross-fitted DTI of each representation family separately (is the ranking destroyed by crest_w1?)."""
    from ..g13.evaluation import crossfit

    rows = []
    for (task, rep, order), t in table.groupby(["task", "rep", "order"]):
        per, summ = crossfit(t, ["target_frac"])
        rows.append({"task": task, "rep": rep, "order": order, "dti_crossfit": summ["dti_crossfit"],
                     "zone_min": summ["zone_min"], "optimism": summ["optimism"],
                     "frac_all": summ["params_all"]["target_frac"],
                     "frac_on_edge": summ["params_all"]["target_frac"] in (t.target_frac.min(), t.target_frac.max())
                     and t.target_frac.nunique() > 1})
    return pd.DataFrame(rows).sort_values(["task", "dti_crossfit"], ascending=[True, False])
