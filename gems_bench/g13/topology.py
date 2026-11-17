"""Fault topology, graph reconstruction and the unmapped-lineament discovery atlas (sections 25-26).

Mask operations (gap linking, small-component removal, NMS crests) are reused from
``gems_bench.postprocess.topology`` and ``gems_bench.gems3.representation``; this module adds the segment graph,
graph-based confidence propagation and the candidate classification relative to mapped faults.

A candidate is a *model prediction*. Its class says how it relates to mapped traces, never that it is a fault:
``mapped_trace`` / ``known_extension`` / ``parallel`` / ``intersecting`` / ``unverified_network`` / ``isolated``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..geology.skeleton import branch_points, endpoints, neighbour_count, skeletonize
from ..postprocess.topology import link_endpoints, remove_small
from .signal import axial_diff, structure_tensor

S8 = np.ones((3, 3), bool)


def segment_graph(
    mask: np.ndarray, score: np.ndarray | None = None, pixel_m: float = 100.0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Decompose a binary mask into skeleton segments between endpoints / junctions.

    Returns (segments, edges, segment-id raster on the skeleton). Lengths are geodesic along the skeleton (orthogonal
    steps 1 px, diagonal sqrt(2) px). Tortuosity = path length / endpoint chord; curvature = mean absolute change of
    local strike along the segment (deg/km)."""
    sk = skeletonize(mask)
    if not sk.any():
        return pd.DataFrame(), pd.DataFrame(), np.full(mask.shape, -1, np.int32)
    nb = neighbour_count(sk)
    junction = sk & (nb >= 3)
    jl, nj = ndimage.label(ndimage.binary_dilation(junction, S8) & sk, structure=S8)
    body = sk & (jl == 0)
    seg, ns = ndimage.label(body, structure=S8)
    _, th_sk, _ = structure_tensor(sk.astype(np.float32), 1.0, 2.0)
    jl_near = ndimage.maximum_filter(jl, size=3)  # junction id adjacent to each skeleton pixel
    end_mask = endpoints(sk)
    rows, edges = [], []
    objs = ndimage.find_objects(seg)
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = seg[sl] == i
        rr, cc = np.nonzero(m)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        n = len(rr)
        # geodesic length: orthogonal neighbour pairs count 1, diagonal sqrt(2), each pair once
        pts = set(zip(rr.tolist(), cc.tolist()))
        orth = sum((r, c + 1) in pts for r, c in pts) + sum((r + 1, c) in pts for r, c in pts)
        diag = sum((r + 1, c + 1) in pts for r, c in pts) + sum((r + 1, c - 1) in pts for r, c in pts)
        length = (orth + np.sqrt(2) * diag) * pixel_m / 1000 if n > 1 else pixel_m / 1000
        X = np.c_[cc.astype(float), -rr.astype(float)]
        if n >= 3:
            Xc = X - X.mean(0)
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            strike = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180)
            proj = Xc @ vt[0]
            chord = (proj.max() - proj.min()) * pixel_m / 1000
        else:
            strike, chord = np.nan, length
        th = th_sk[rr, cc]
        order = np.argsort(Xc @ vt[0]) if n >= 3 else np.arange(n)
        dth = np.degrees(axial_diff(th[order][1:], th[order][:-1])) if n >= 3 else np.zeros(0)
        nodes = np.unique(jl_near[rr, cc])
        nodes = nodes[nodes > 0]
        for a in nodes:
            edges.append({"segment_id": i, "junction_id": int(a)})
        s = np.asarray(score)[rr, cc] if score is not None else np.full(n, np.nan)
        rows.append(
            {
                "segment_id": i,
                "n_px": n,
                "length_km": float(length),
                "chord_km": float(chord),
                "tortuosity": float(length / max(chord, 1e-6)),
                "strike_deg": strike,
                "curvature_deg_per_km": float(dth.sum() / max(length, 1e-6)) if len(dth) else 0.0,
                "row_c": float(rr.mean()),
                "col_c": float(cc.mean()),
                "n_junctions": int(len(nodes)),
                "is_endpoint_segment": bool(end_mask[rr, cc].any()),
                "score_mean": float(np.nanmean(s)) if n else np.nan,
                "score_max": float(np.nanmax(s)) if n else np.nan,
            }
        )
    segs = pd.DataFrame(rows)
    return segs, pd.DataFrame(edges), np.where(seg > 0, seg, -1).astype(np.int32)


def propagate_confidence(segs: pd.DataFrame, edges: pd.DataFrame, iterations: int = 2, mix: float = 0.5) -> pd.Series:
    """Segments sharing a junction pull each other's confidence up: c <- max(c, mix c + (1 - mix) mean(neighbours))."""
    if segs.empty:
        return pd.Series(dtype=float)
    c = segs.set_index("segment_id").score_mean.copy()
    if edges.empty:
        return c
    by_j = edges.groupby("junction_id").segment_id.apply(list)
    neigh = {s: set() for s in c.index}
    for lst in by_j:
        for a in lst:
            neigh[a].update(x for x in lst if x != a)
    for _ in range(iterations):
        new = c.copy()
        for s, ns in neigh.items():
            if ns:
                new[s] = max(c[s], mix * c[s] + (1 - mix) * np.mean([c[x] for x in ns]))
        c = new
    return c


def extract_candidates(
    score: np.ndarray,
    domain: np.ndarray,
    frac: float,
    *,
    min_px: int = 4,
    link_gap_px: int = 4,
    link_angle_deg: float = 30.0,
) -> np.ndarray:
    """Binary candidate mask: top-``frac`` of ``domain``, small components removed, collinear gaps linked."""
    from ..gems3.representation import top_mask

    m = top_mask(score, domain, frac)
    m = remove_small(m, min_px)
    if link_gap_px > 0:
        m = link_endpoints(m, link_gap_px, link_angle_deg) & domain
    return m


def classify_candidates(
    segs: pd.DataFrame,
    seg_raster: np.ndarray,
    known: np.ndarray,
    *,
    pixel_m: float = 100.0,
    extension_km: float = 1.0,
    parallel_km: float = 3.0,
    angle_tol_deg: float = 20.0,
) -> pd.DataFrame:
    """Relation of every segment to the mapped faults (fixed geometric rules, documented in the module docstring)."""
    if segs.empty:
        return segs
    kd = ndimage.distance_transform_edt(~known) * pixel_m / 1000 if known.any() else np.full(known.shape, 1e9)
    kd_idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True) if known.any() else None
    _, kth, _ = structure_tensor(known.astype(np.float32), 1.0, 3.0)
    ksk = skeletonize(known)
    kend = endpoints(ksk) if known.any() else known
    kend_d = ndimage.distance_transform_edt(~kend) * pixel_m / 1000 if kend.any() else np.full(known.shape, 1e9)
    n_seg_in_comp = _network_size(seg_raster)
    pix = segment_pixels(seg_raster)
    kd_f, kend_f = kd.ravel(), kend_d.ravel()
    out = []
    for r in segs.itertuples():
        ids = pix.get(r.segment_id, np.zeros(0, np.int64))
        if not len(ids):
            continue
        d = kd_f[ids]
        overlap = float((d <= pixel_m / 1000 + 1e-9).mean())
        dmin = float(d.min())
        if kd_idx is not None:
            i = ids[int(np.argmin(d))]
            rr, cc = np.unravel_index(i, known.shape)
            nr, nc = kd_idx[0][rr, cc], kd_idx[1][rr, cc]
            kstrike = float(np.degrees(kth[nr, nc]))
        else:
            kstrike = np.nan
        dang = (
            float(np.degrees(axial_diff(np.radians(r.strike_deg), np.radians(kstrike))))
            if np.isfinite(r.strike_deg) and np.isfinite(kstrike)
            else np.nan
        )
        end_near = float(kend_f[ids].min())
        if overlap >= 0.5:
            cls = "mapped_trace"
        elif end_near <= extension_km and np.isfinite(dang) and dang <= angle_tol_deg + 5:
            cls = "known_extension"
        elif dmin <= 0.3 and np.isfinite(dang) and dang > 30:
            cls = "intersecting"
        elif dmin <= parallel_km and np.isfinite(dang) and dang <= angle_tol_deg:
            cls = "parallel"
        elif n_seg_in_comp.get(r.segment_id, 1) >= 3:
            cls = "unverified_network"
        else:
            cls = "isolated"
        out.append(
            {
                "segment_id": r.segment_id,
                "known_overlap": overlap,
                "dist_known_km": dmin,
                "known_strike_deg": kstrike,
                "angle_to_known_deg": dang,
                "dist_known_tip_km": end_near,
                "network_segments": n_seg_in_comp.get(r.segment_id, 1),
                "class": cls,
            }
        )
    return segs.merge(pd.DataFrame(out), on="segment_id")


def segment_pixels(seg_raster: np.ndarray) -> dict[int, np.ndarray]:
    """segment id -> flat pixel indices (one pass, no per-segment full-raster scans)."""
    f = seg_raster.ravel()
    idx = np.flatnonzero(f >= 0)
    if not len(idx):
        return {}
    ids = f[idx]
    order = np.argsort(ids, kind="stable")
    idx, ids = idx[order], ids[order]
    cut = np.flatnonzero(np.diff(ids)) + 1
    return {int(g[0]): i for g, i in zip(np.split(ids, cut), np.split(idx, cut))}


def _network_size(seg_raster: np.ndarray) -> dict:
    comp, n = ndimage.label(ndimage.binary_dilation(seg_raster >= 0, S8, iterations=2), structure=S8)
    s = seg_raster >= 0
    df = pd.DataFrame({"seg": seg_raster[s], "comp": comp[s]}).drop_duplicates()
    size = df.groupby("comp").seg.nunique()
    return dict(zip(df.seg, df.comp.map(size)))


def candidate_table(
    score: np.ndarray,
    data,
    store,
    *,
    frac: float,
    evidence: dict[str, np.ndarray] | None = None,
    region_confidence: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sortable candidate table (UTM coordinates, length, strike, scores, evidence, uncertainty)."""
    from ..gems3.representation import top_mask  # noqa: F401  (documented reuse)

    m = extract_candidates(score, data.valid, frac)
    segs, edges, seg_r = segment_graph(m, score)
    if segs.empty:
        return segs, seg_r
    segs["score_propagated"] = segs.segment_id.map(propagate_confidence(segs, edges))
    segs = classify_candidates(segs, seg_r, data.labels)
    t = data.transform
    segs["x_utm"] = t.c + (segs.col_c + 0.5) * t.a
    segs["y_utm"] = t.f + (segs.row_c + 0.5) * t.e
    pix = segment_pixels(seg_r)
    empty = np.zeros(0, np.int64)
    for name, arr in (evidence or {}).items():
        a = np.asarray(arr, np.float32).ravel()
        segs[f"ev_{name}"] = [
            float(np.nanmean(a[pix.get(i, empty)])) if len(pix.get(i, empty)) else np.nan for i in segs.segment_id
        ]
    if region_confidence is not None:
        a = np.asarray(region_confidence, np.float32).ravel()
        segs["region_model_dti"] = [
            float(np.nanmean(a[pix.get(i, empty)])) if len(pix.get(i, empty)) else np.nan for i in segs.segment_id
        ]
    ev_cols = [c for c in segs.columns if c.startswith("ev_")]
    if ev_cols:
        E = segs[ev_cols].rank(pct=True)
        segs["evidence_support"] = E.mean(axis=1)
        segs["evidence_spread"] = E.std(axis=1)  # uncertainty proxy: disagreement between evidence lines
    segs["novelty_km"] = segs.dist_known_km
    return segs.sort_values(["class", "score_propagated"], ascending=[True, False]).reset_index(drop=True), seg_r


def network_summary(segs: pd.DataFrame) -> pd.DataFrame:
    if segs.empty:
        return segs
    return (
        segs.groupby("class")
        .agg(
            n=("segment_id", "size"),
            length_km=("length_km", "sum"),
            median_len_km=("length_km", "median"),
            mean_score=("score_mean", "mean"),
            median_tortuosity=("tortuosity", "median"),
        )
        .reset_index()
    )


def junction_count(mask: np.ndarray) -> int:
    sk = skeletonize(mask)
    return int(branch_points(sk).sum())
