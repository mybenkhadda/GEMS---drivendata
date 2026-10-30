"""Complementarity of a feature with a frozen baseline prediction (W2 / M9), at IDENTICAL realised area budgets.

Both rankings select the top ``budget`` share of the evaluated valid pixels. A mapped fault pixel is "recovered" by
a selection when a selected pixel lies within ``radius_px`` (GEMS: 3 px = 300 m, the official tolerance radius); a
mapped fault COMPONENT (8-connected) is recovered when >= ``component_share`` of its pixels are. This is a ranking
diagnostic: it is NOT the official DTI (use the existing scorer for any score claim), and a held-out claim needs a
baseline map whose provenance says it was produced without these pixels (``provenance()``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

CATEGORY = {0: "background", 1: "fault: both", 2: "fault: baseline only", 3: "fault: feature only",
            4: "fault: missed by both", 5: "unlabelled: baseline only", 6: "unlabelled: feature only",
            7: "unlabelled: both"}


def top_mask(score, valid, budget):
    s = np.where(valid, np.asarray(score, np.float64), -np.inf)
    k = max(1, int(math.ceil(budget * int(valid.sum()))))
    idx = np.argpartition(-s.ravel(), k - 1)[:k]
    m = np.zeros(s.size, bool)
    m[idx] = True
    return m.reshape(s.shape) & valid


def provenance(pred_path: str) -> dict:
    """Read ``<prediction>.provenance.json`` next to the raster. HELD_OUT_VERIFIED only when it states
    ``held_out: true`` with a fold / zone definition; otherwise UNVERIFIED (treat as in-sample)."""
    p = Path(str(pred_path) + ".provenance.json")
    alt = Path(pred_path).with_suffix(".provenance.json")
    q = p if p.exists() else alt
    if not q.exists():
        return {"status": "UNVERIFIED", "reason": f"no provenance file ({p.name})"}
    try:
        d = json.loads(q.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "UNVERIFIED", "reason": f"unreadable provenance: {e}"}
    ok = bool(d.get("held_out")) and bool(d.get("folds") or d.get("zones") or d.get("zone_map"))
    return {"status": "HELD_OUT_VERIFIED" if ok else "UNVERIFIED", "reason": "" if ok else
            "provenance lacks held_out=true with folds / zones", **d}


def compare(baseline, feature, faults, valid=None, budget=0.01, radius_px=3, component_share=0.5,
            feature_direction="high") -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(summary table, category raster, per-component table)."""
    b = np.asarray(baseline, np.float32)
    x = np.asarray(feature, np.float32)
    f = np.asarray(faults, bool)
    v = np.isfinite(b) & np.isfinite(x) & (np.asarray(valid, bool) if valid is not None else True)
    xs = x if feature_direction == "high" else -x
    sb, sf = top_mask(b, v, budget), top_mask(xs, v, budget)
    from scipy.stats import rankdata

    rb = np.full(b.shape, np.nan)
    rf = np.full(b.shape, np.nan)
    rb[v], rf[v] = rankdata(b[v]) / v.sum(), rankdata(xs[v]) / v.sum()
    s_avg = top_mask(np.where(v, 0.5 * (rb + rf), -np.inf), v, budget)                 # rank-average fusion
    half = top_mask(b, v, budget / 2)                                                     # baseline's best half
    k_rest = max(1, int(math.ceil(budget * int(v.sum()))) - int(half.sum()))
    s_half = half | top_mask(xs, v & ~half, k_rest / max(int((v & ~half).sum()), 1))      # + feature's best rest
    st = ndimage.generate_binary_structure(2, 2)
    near = lambda s: ndimage.binary_dilation(s, st, iterations=radius_px) if s.any() else s  # noqa: E731
    nb, nf, na, nh = near(sb), near(sf), near(s_avg), near(s_half)
    fv = f & v
    P = int(fv.sum())
    rows = []
    for name, sel, nr in (("baseline", sb, nb), ("feature", sf, nf), ("rank_average", s_avg, na),
                          ("half_and_half", s_half, nh)):
        rows.append({"selection": name, "area_px": int(sel.sum()), "area_share": float(sel.sum() / v.sum()),
                     "fault_px_recovered": int((nr & fv).sum()), "pixel_recall": (nr & fv).sum() / P if P else np.nan,
                     "unlabelled_selected_px": int((sel & ~near(f)).sum())})
    T = pd.DataFrame(rows)
    cat = np.zeros(b.shape, np.int8)
    cat[fv & nb & nf] = 1
    cat[fv & nb & ~nf] = 2
    cat[fv & ~nb & nf] = 3
    cat[fv & ~nb & ~nf] = 4
    far = ~near(f)
    cat[sb & ~sf & far] = 5
    cat[sf & ~sb & far] = 6
    cat[sb & sf & far] = 7
    lab, n = ndimage.label(fv, st)
    comp = []
    if n:
        idx = np.arange(1, n + 1)
        size = ndimage.sum(np.ones_like(lab), lab, idx)
        rb_ = ndimage.sum(nb & fv, lab, idx) / size >= component_share
        rf_ = ndimage.sum(nf & fv, lab, idx) / size >= component_share
        comp = pd.DataFrame({"component": idx, "n_px": size.astype(int), "baseline": rb_, "feature": rf_})
        comp["category"] = np.select([comp.baseline & comp.feature, comp.baseline & ~comp.feature,
                                      ~comp.baseline & comp.feature], ["both", "baseline only", "feature only"],
                                     "missed by both")
    comp = pd.DataFrame(comp)
    if len(comp):
        cnt = comp.category.value_counts()
        T.attrs["components"] = {k: int(cnt.get(k, 0)) for k in ("both", "baseline only", "feature only",
                                                                 "missed by both")}
        T.attrs["unique_feature_components"] = int(cnt.get("feature only", 0))
    T.attrs.update(budget=budget, radius_px=radius_px, n_fault_px=P, n_valid=int(v.sum()),
                   marginal_pixel_recall_rank_average=float(T.pixel_recall[2] - T.pixel_recall[0]) if P else np.nan,
                   marginal_pixel_recall_half=float(T.pixel_recall[3] - T.pixel_recall[0]) if P else np.nan,
                   note="ranking diagnostic at equal area; NOT DTI")
    return T, cat, comp
