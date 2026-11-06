"""Cross-modal geological evidence (descriptive, label-free; not a fault probability).

Independent modality groups: MAGNETIC (rtp), GRAVITY (isostatic anomaly), TOPOGRAPHY (100 m terrain and the
USGS DEM together - two views of the same surface, never counted twice). Edge strength per group is a
percentile rank over the footprint (DEM: over covered cells):
  magnetic  = mag_boundary_strength (mean THD rank over continuation levels)
  gravity   = grav_boundary_strength
  terrain   = rank(terrain_slope_1000m);  dem = rank(dem_edge_p90);  topography = max(terrain, dem where covered)
Pairwise edge agreement = sqrt(rank_a * rank_b) where both exist (NaN otherwise: missing data is missing
evidence, never disagreement). Orientation agreement = cos 2(a - b) of the axial strikes where both coherences
>= ``evidence.min_coherence``. Evidence count = groups with rank >= ``evidence.strong_rank``; available count =
groups with data. Continuity = mean combined evidence along the local strike over +- ``continuity_half_length_m``.
Disagreement = largest |rank_a - rank_b| among available pairs where one of them is strong.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .circular_statistics import agreement
from .feature_registry import pct_rank

GROUPS = ("magnetic", "gravity", "topography")


def build(load, footprint: np.ndarray, dem_cov: np.ndarray | None, cfg, writer, px: float) -> dict:
    """``load(feature_id) -> array`` reads registry features (aliases resolved)."""
    ev = cfg["evidence"]
    strong, mincoh = float(ev["strong_rank"]), float(ev["min_coherence"])
    rank = {"magnetic": load("mag_boundary_strength"), "gravity": load("grav_boundary_strength"),
            "terrain": pct_rank(load("terrain_slope_1000m"), footprint)}
    have_dem = dem_cov is not None
    if have_dem:
        covered = footprint & (dem_cov > 0)
        rank["dem"] = pct_rank(load("dem_edge_p90"), covered)
        rank["topography"] = np.where(np.isfinite(rank["dem"]), np.fmax(rank["terrain"], rank["dem"]), rank["terrain"])
    else:
        rank["topography"] = rank["terrain"]
    orient = {"magnetic": (load("mag_strike_c2_1000m"), load("mag_strike_s2_1000m"), load("mag_coherence_1000m")),
              "gravity": (load("grav_strike_c2_1000m"), load("grav_strike_s2_1000m"), load("grav_coherence_1000m")),
              "terrain": (load("terrain_strike_c2_1000m"), load("terrain_strike_s2_1000m"),
                          load("terrain_coherence_1000m"))}
    if have_dem:
        orient["dem"] = (load("dem_orient_c2"), load("dem_orient_s2"), load("dem_orient_R"))

    def put(fid, arr, parents, op, meaning, unit="rank"):
        writer.put(fid, arr, family="cross_modal", parent="+".join(parents), unit=unit, operator=op, meaning=meaning)

    pairs = [("magnetic", "gravity"), ("magnetic", "terrain"), ("gravity", "terrain")]
    if have_dem:
        pairs += [("magnetic", "dem"), ("gravity", "dem")]
    for a, b in pairs:
        ra, rb = rank[a], rank[b]
        put(f"xm_edge_{a[:4]}_{b[:4]}", np.sqrt(ra * rb), [a, b], "sqrt(rank_a * rank_b)", f"{a}-{b} edge agreement")
        ca, sa, wa = orient[a]
        cb, sb, wb = orient[b]
        ok = (wa >= mincoh) & (wb >= mincoh)
        put(f"xm_orient_{a[:4]}_{b[:4]}", np.where(ok, agreement(ca, sa, cb, sb), np.nan), [a, b],
            f"cos 2(a-b), both coherences >= {mincoh}", f"{a}-{b} strike agreement (+1 parallel, -1 perpendicular)",
            unit="cos2dtheta")
    G = np.stack([rank[g] for g in GROUPS])
    avail = np.isfinite(G).sum(0).astype(np.float32)
    count = np.where(np.isfinite(G), G >= strong, False).sum(0).astype(np.float32)
    mean_ev = np.where(avail > 0, np.nanmean(np.where(np.isfinite(G), G, np.nan), 0), np.nan)
    put("xm_available_modalities", avail, list(GROUPS), "count of groups with data", "evidence available (0-3)", "count")
    put("xm_evidence_count", np.where(avail > 0, count, np.nan), list(GROUPS), f"groups with rank >= {strong}",
        "independent strong-edge evidence count", "count")
    put("xm_evidence_fraction", np.where(avail > 0, count / np.maximum(avail, 1), np.nan), list(GROUPS),
        "evidence_count / available", "share of available groups with a strong edge", "fraction")
    put("xm_evidence_mean", mean_ev, list(GROUPS), "mean rank of available groups", "combined edge evidence")
    pers = [load("mag_edge_persistence"), load("grav_edge_persistence")]
    put("xm_multiscale_persistence", np.nanmean(np.stack(pers), 0), ["magnetic", "gravity"],
        "mean of magnetic and gravity edge persistence", "multiscale boundary persistence", "fraction")
    # disagreement: one group strong, another weak, both observed
    dis = np.zeros(footprint.shape, np.float32)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = G[i], G[j]
            d = np.where(np.isfinite(a) & np.isfinite(b) & ((a >= strong) | (b >= strong)), np.abs(a - b), 0)
            dis = np.maximum(dis, d)
    put("xm_disagreement", np.where(avail >= 2, dis, np.nan), list(GROUPS),
        "max |rank_a - rank_b| over available pairs with one strong edge", "structural disagreement")
    # continuity along the local strike of the most coherent orientation source
    cohs = np.stack([np.nan_to_num(orient[k][2], nan=-1) for k in ("magnetic", "gravity", "terrain")])
    best = np.argmax(cohs, 0)
    c2 = np.choose(best, [np.nan_to_num(orient[k][0]) for k in ("magnetic", "gravity", "terrain")])
    s2 = np.choose(best, [np.nan_to_num(orient[k][1]) for k in ("magnetic", "gravity", "terrain")])
    az = 0.5 * np.arctan2(s2, c2)                              # strike azimuth (rad, clockwise from north)
    dr, dc = -np.cos(az), np.sin(az)                           # unit step along strike in (row, col)
    E = np.nan_to_num(mean_ev, nan=0.0).astype(np.float32)
    n = int(round(float(ev["continuity_half_length_m"]) / px))
    rr, cc = np.indices(E.shape, dtype=np.float32)
    acc = np.zeros_like(E)
    for k in range(-n, n + 1):
        acc += ndimage.map_coordinates(E, [rr + k * dr, cc + k * dc], order=1, mode="constant", cval=0.0)
    put("xm_continuity", np.where(avail > 0, acc / (2 * n + 1), np.nan), list(GROUPS),
        f"mean combined evidence along strike +- {ev['continuity_half_length_m']:g} m", "structural continuity")
    return {"ranks": rank, "orient": orient, "evidence_mean": mean_ev, "available": avail, "count": count}
