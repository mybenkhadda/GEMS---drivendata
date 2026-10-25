"""M6.5 geological error analysis: which mapped faults each model misses, and attributes that may explain it.

``segment_attributes`` describes every mapped segment inside a geographic fold zone (one row per Phase-1
``seg_id``): length, strike, sinuosity, junction context, local fault density, distance to the fold-zone edge,
how common its strike is among the *other* folds' faults (orientation rarity: what the fold's model was trained
on) and the normalised gradient contrast of selected geophysical bands along the trace.

``miss_attribution`` compares segments missed by both models with segments both recover. It is descriptive: a
higher missed share among short, rare-strike or low-contrast faults is a hypothesis about the error, not a
causal estimate. Nothing here reads predictions except through the per-pixel TP support passed in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

DEFAULT_CONTRAST_PATTERNS = ("mag", "grav", "bouguer")


def pick_bands(band_names: list[str], requested=None, patterns=DEFAULT_CONTRAST_PATTERNS, limit: int = 2) -> list[str]:
    """Explicit band names (validated) or the first ``limit`` bands whose name contains a pattern."""
    if requested:
        miss = [b for b in requested if b not in band_names]
        if miss:
            raise KeyError(f"bands not in the feature raster: {miss}")
        return list(requested)
    out = [b for b in band_names if any(p in b.lower() for p in patterns)]
    return out[:limit]


def strike_from_pixels(r: np.ndarray, c: np.ndarray) -> float:
    """Axial strike (deg clockwise from north, [0, 180)) of a pixel set's principal axis (rows grow south)."""
    if len(r) < 2:
        return np.nan
    X = np.stack([-(r - r.mean()), c - c.mean()], 1).astype(float)       # (north, east)
    w, v = np.linalg.eigh(X.T @ X)
    n, e = v[:, int(np.argmax(w))]
    return float(np.degrees(np.arctan2(e, n)) % 180.0)


def _edge_distance_km(fold_map: np.ndarray, valid: np.ndarray, grid) -> np.ndarray:
    """Distance (km) of every pixel to the boundary of its own fold zone (buffer / outside)."""
    out = np.zeros(fold_map.shape, np.float32)
    for k in np.unique(fold_map[valid & (fold_map >= 0)]):
        z = valid & (fold_map == k)
        out[z] = ndimage.distance_transform_edt(z, sampling=(grid.pixel_km_y, grid.pixel_km_x))[z]
    return out


def segment_attributes(ctx, *, contrast_bands=(), density_km: float = 2.0, rarity_tol_deg: float = 15.0) -> pd.DataFrame:
    from ..geology.orientation import axial_difference
    from ..phase4_eval import label_pixel_table

    lpt = label_pixel_table(ctx)
    lpt = lpt[lpt.seg_id >= 0]
    g = ctx.data.grid
    dens_px = max(int(round(density_km / g.pixel_km)), 3)
    dens = ndimage.uniform_filter(ctx.data.labels.astype(np.float32), dens_px)
    edge = _edge_distance_km(ctx.folds.fold_map, ctx.data.valid, g)
    t = lpt.assign(density=dens[lpt.r, lpt.c], edge_km=edge[lpt.r, lpt.c],
                   at_intersection=(lpt.structure == "intersection"), at_endpoint=(lpt.structure == "endpoint"))
    agg = {"n_px": ("r", "size"), "fold": ("fold", lambda x: int(pd.Series(x).mode().iloc[0])),
           "r_mid": ("r", "median"), "c_mid": ("c", "median"), "local_density": ("density", "mean"),
           "edge_km": ("edge_km", "min"), "share_intersection": ("at_intersection", "mean"),
           "share_endpoint": ("at_endpoint", "mean")}
    for c in ("length_km", "length_bin", "sinuosity_bin", "junction", "primary_archetype", "region", "short_fault"):
        if c in t:
            agg[c] = (c, "first")
    seg = t.groupby("seg_id").agg(**agg).reset_index()
    tab = ctx.seg_tab.set_index("seg_id") if ctx.seg_tab is not None and "azimuth" in ctx.seg_tab else None
    if tab is not None:
        seg["strike_deg"] = seg.seg_id.map(tab.azimuth % 180.0)
        seg["strike_source"] = "segment_table"
    else:
        st = t.groupby("seg_id").apply(lambda x: strike_from_pixels(x.r.to_numpy(), x.c.to_numpy()))
        seg["strike_deg"] = seg.seg_id.map(st)
        seg["strike_source"] = "pixel_pca"
    seg["strike_sector"] = pd.cut(seg.strike_deg, np.arange(0, 181, 30), include_lowest=True).astype(str)
    # orientation rarity: share of the OTHER folds' fault pixels within +-tol of this strike
    rar = np.full(len(seg), np.nan)
    for k in seg.fold.unique():
        other = seg[seg.fold != k]
        if not len(other):
            continue
        tot = float(other.n_px.sum())
        idx = np.flatnonzero(seg.fold.to_numpy() == k)
        for i in idx:
            dd = axial_difference(other.strike_deg.to_numpy(), seg.strike_deg.iloc[i])
            rar[i] = float(other.n_px[dd <= rarity_tol_deg].sum() / max(tot, 1.0))
    seg["strike_share_other_folds"] = rar
    for b in contrast_bands:
        arr = ctx.data.read_band(b)
        gy, gx = np.gradient(np.nan_to_num(arr, nan=np.nanmedian(arr)))
        gm = np.hypot(gx, gy)
        ref = float(np.nanmedian(gm[ctx.data.valid])) or 1.0
        val = pd.Series(gm[lpt.r, lpt.c] / ref, index=lpt.index)
        seg[f"contrast_{b}"] = seg.seg_id.map(val.groupby(lpt.seg_id).mean())
    return seg


def recovery_classes(seg: pd.DataFrame, support: dict[str, np.ndarray], lpt: pd.DataFrame, a: str, b: str,
                     k_thr: float = 0.5, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per segment: mean TP support of a and b (+ extra models) and the class both / a_only / b_only / neither."""
    t = lpt[["seg_id"]].assign(**{f"k_{m}": support[m] for m in (a, b) + tuple(extra) if m in support})
    k = t[t.seg_id >= 0].groupby("seg_id").mean()
    out = seg.merge(k.reset_index(), on="seg_id", how="left")
    ra, rb = out[f"k_{a}"] >= k_thr, out[f"k_{b}"] >= k_thr
    out["class"] = np.select([ra & rb, ra & ~rb, ~ra & rb], ["both", f"{a}_only", f"{b}_only"], "neither")
    return out


def class_summary(cls: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Counts, trace km and shares of each recovery class per stratum value."""
    rows = []
    for col in by:
        if col not in cls:
            continue
        for val, g in cls.groupby(col, observed=True, dropna=False):
            km = g.length_km.sum() if "length_km" in g else np.nan
            for c, h in g.groupby("class"):
                rows.append({"stratum": col, "value": str(val), "class": c, "n_segments": len(h),
                             "trace_km": float(h.length_km.sum()) if "length_km" in h else np.nan,
                             "share_of_stratum_km": float(h.length_km.sum() / km) if km else np.nan})
    return pd.DataFrame(rows)


ATTRIBUTE_HYPOTHESES = {
    "length_km": "short traces are harder (fewer pixels, below the receptive field)",
    "strike_share_other_folds": "strikes rare in the training folds are missed (orientation prior)",
    "local_density": "isolated faults lack the context of dense fault networks",
    "share_intersection": "junction-rich traces are fragmented or merged",
    "edge_km": "faults near the fold-zone edge lose context at tile / zone borders",
}


def miss_attribution(cls: pd.DataFrame) -> pd.DataFrame:
    """Missed-by-both vs recovered-by-both: medians and ratio for each attribute (descriptive hypotheses)."""
    miss, hit = cls[cls["class"] == "neither"], cls[cls["class"] == "both"]
    rows = []
    attrs = dict(ATTRIBUTE_HYPOTHESES)
    attrs.update({c: "low geophysical contrast along the trace" for c in cls.columns if c.startswith("contrast_")})
    for c, why in attrs.items():
        if c not in cls or not len(miss) or not len(hit):
            continue
        a, b = float(np.nanmedian(miss[c])), float(np.nanmedian(hit[c]))
        rows.append({"attribute": c, "hypothesis": why, "median_missed_by_both": a, "median_recovered_by_both": b,
                     "ratio_missed_over_recovered": a / b if b else np.nan, "n_missed": len(miss), "n_recovered": len(hit),
                     "note": "descriptive comparison of segment groups; not a causal estimate"})
    return pd.DataFrame(rows)
