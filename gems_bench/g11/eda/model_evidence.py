"""EDA of EXISTING model predictions (G07 / M9 / P6 / GeoRegime) that the audit placed on the master grid.

Only COMPATIBLE maps (documented placement, nested grid) are used. Each map is analysed on its OWN finite support
(fold-masked arrays cover only their original held-out region). Statistics (block-sampled):
  pairwise Spearman on the joint support, mean |difference| (disagreement), mapped-fault recall at the top-1 % of
  the map's own support (tolerance 300 m), share of its top-1 % pixels > 2 km from any mapped fault (high-scoring
  unmapped regions - a FALSE-POSITIVE PROXY, not verified false positives), share of its top-1 % pixels with strong
  terrain support (terrain_scarp_rank >= 0.8; NaN without DEM).
Provenance is carried through: OOF_FOLD_MASKED vs FULL_DOMAIN_INFERENCE are never pooled as if comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_compatible(compat_csv, max_maps: int = 12) -> dict:
    """Aligned COMPATIBLE maps (G07 first, then by family), at most ``max_maps`` (49 MB each on the full grid)."""
    import os

    import rasterio

    try:
        t = pd.read_csv(compat_csv) if compat_csv and os.path.isfile(compat_csv) else pd.DataFrame()
    except pd.errors.EmptyDataError:                      # no external prediction discovered
        t = pd.DataFrame()
    out = {}
    if not len(t) or "aligned_path" not in t:
        return out
    t = t[t.status.astype(str).str.startswith("COMPATIBLE") & t.aligned_path.notna()]
    t = t.assign(_o=(~t.family.astype(str).str.startswith("g07")).astype(int)).sort_values(["_o", "family", "name"])
    for r in t.head(max_maps).itertuples():
        with rasterio.open(r.aligned_path) as ds:
            out[f"{r.family}:{r.name}"] = {"array": ds.read(1).astype(np.float32), "provenance": r.provenance,
                                           "source_split": r.source_split}
    return out


def complementarity(maps: dict, fault: np.ndarray, dist_m: np.ndarray, valid: np.ndarray, terrain_rank=None,
                    block_px: int = 10, seed: int = 2026) -> pd.DataFrame:
    from scipy.stats import spearmanr

    from ..evaluation.pixel_metrics import recall_at_area
    from .spatial import block_sample

    rows = []
    names = list(maps)
    for n in names:
        a = maps[n]["array"]
        sup = valid & np.isfinite(a)
        rec = recall_at_area(a, fault, sup, 0.01, 300.0) if sup.any() else {"recall": np.nan, "precision": np.nan}
        v = a[sup]
        thr = np.quantile(v, 0.99) if v.size else np.inf
        top = sup & (a >= thr)
        tr = terrain_rank[top] if terrain_rank is not None and top.any() else np.array([])
        rows.append({"model_a": n, "model_b": "", "provenance": maps[n]["provenance"],
                     "source_split": maps[n]["source_split"], "support_px": int(sup.sum()),
                     "mapped_recall_top1pct": rec["recall"], "precision_top1pct_300m": rec["precision"],
                     "top1pct_far_from_mapped_share": float((dist_m[top] > 2000).mean()) if top.any() else np.nan,
                     "top1pct_terrain_supported_share": float(np.nanmean(tr >= 0.8)) if np.isfinite(tr).any() else np.nan})
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = maps[a]["array"], maps[b]["array"]
            joint = valid & np.isfinite(A) & np.isfinite(B)
            if joint.sum() < 100:
                rows.append({"model_a": a, "model_b": b, "joint_support_px": int(joint.sum()),
                             "note": "insufficient joint support"})
                continue
            r, c, _ = block_sample(joint, block_px, seed, 50_000)
            rho = spearmanr(A[r, c], B[r, c]).correlation
            rows.append({"model_a": a, "model_b": b, "joint_support_px": int(joint.sum()), "spearman": float(rho),
                         "mean_abs_disagreement": float(np.mean(np.abs(A[r, c] - B[r, c]))),
                         "same_provenance": maps[a]["provenance"] == maps[b]["provenance"]})
    return pd.DataFrame(rows)


def disagreement_map(maps: dict, valid: np.ndarray) -> np.ndarray | None:
    arrs = [m["array"] for m in maps.values()]
    if len(arrs) < 2:
        return None
    S = np.stack(arrs)
    n = np.isfinite(S).sum(0)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(S, 0)
    return np.where(valid & (n >= 2), sd, np.nan).astype(np.float32)
