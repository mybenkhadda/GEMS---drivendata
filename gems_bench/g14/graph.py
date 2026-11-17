"""Fault-segment graph refinement (Phase 9).

Nodes: candidate segments of a binary operating-point mask (``g13.topology.segment_graph``: skeleton segments
between endpoints / junctions with length, strike, tortuosity and score). Edges: pairs of segment ENDPOINTS within
``max_gap_km`` whose geometry is compatible. Edge features: endpoint distance, strike agreement of both segments,
collinearity of the gap with both strikes, curvature, mean / min evidence along the gap (magnetic, gravity,
structural coherence, terrain), candidate confidence, regime equality.

Rule-based baseline (default): accept an edge only if it is collinear (both angle terms <= ``max_angle_deg``) AND
the physical evidence along the gap is above its ``evidence_quantile`` among all candidate gaps AND the model's own
mean probability along the gap is above the scene's ``prob_quantile``. Proximity alone never links. An optional
logistic edge classifier can be trained on out-of-fold geometry (edges whose gap lies on a VISIBLE mapped trace are
positives); it is never trained on held-out truth. A GNN is not implemented (documented as future work).

Outputs: accepted links, edge table, candidate network graph (networkx-free adjacency), reconstructed raster
(original mask probability + linked gaps at a confidence below the weakest linked end).

UNTESTED: written without execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..g13.signal import axial_diff
from ..g13.topology import segment_graph, segment_pixels


def segment_endpoints(seg_raster: np.ndarray) -> pd.DataFrame:
    """Endpoint pixels of every segment with the local direction pointing out of the segment."""
    from ..geology.skeleton import endpoints

    sk = seg_raster >= 0
    ep = endpoints(sk)
    rr, cc = np.nonzero(ep)
    rows = []
    pix = segment_pixels(seg_raster)
    W = seg_raster.shape[1]
    for r, c in zip(rr, cc):
        sid = int(seg_raster[r, c])
        ids = pix.get(sid)
        if ids is None or len(ids) < 2:
            continue
        pr, pc = ids // W, ids % W
        near = np.hypot(pr - r, pc - c) <= 5
        dr, dc = r - pr[near].mean(), c - pc[near].mean()
        n = np.hypot(dr, dc) or 1.0
        rows.append({"segment_id": sid, "row": int(r), "col": int(c), "dir_r": dr / n, "dir_c": dc / n})
    return pd.DataFrame(rows)


def _line(r0, c0, r1, c1):
    from skimage.draw import line

    return line(int(r0), int(c0), int(r1), int(c1))


def candidate_edges(segs: pd.DataFrame, ends: pd.DataFrame, evidence: dict[str, np.ndarray], prob: np.ndarray,
                    regimes: np.ndarray | None, max_gap_km: float = 2.0, pixel_km: float = 0.1) -> pd.DataFrame:
    if ends.empty:
        return pd.DataFrame()
    from scipy.spatial import cKDTree

    pts = ends[["row", "col"]].to_numpy(float)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(max_gap_km / pixel_km)
    S = segs.set_index("segment_id")
    rows = []
    for i, j in pairs:
        a, b = ends.iloc[i], ends.iloc[j]
        if a.segment_id == b.segment_id:
            continue
        dr, dc = b.row - a.row, b.col - a.col
        gap = float(np.hypot(dr, dc))
        if gap < 1:
            continue
        ur, uc = dr / gap, dc / gap
        out_a = a.dir_r * ur + a.dir_c * uc                  # a's tip points towards b
        out_b = -(b.dir_r * ur + b.dir_c * uc)               # b's tip points towards a
        gap_theta = np.mod(np.arctan2(-dr, dc), np.pi)
        sa, sb = S.loc[a.segment_id], S.loc[b.segment_id]
        th_a = np.radians(sa.strike_deg) if np.isfinite(sa.strike_deg) else np.nan
        th_b = np.radians(sb.strike_deg) if np.isfinite(sb.strike_deg) else np.nan
        rr, cc = _line(a.row, a.col, b.row, b.col)
        rec = {"end_a": i, "end_b": j, "seg_a": int(a.segment_id), "seg_b": int(b.segment_id),
               "gap_km": gap * pixel_km, "tip_a_cos": float(out_a), "tip_b_cos": float(out_b),
               "strike_diff_deg": float(np.degrees(axial_diff(np.array(th_a), np.array(th_b)))),
               "gap_vs_a_deg": float(np.degrees(axial_diff(np.array(gap_theta), np.array(th_a)))),
               "gap_vs_b_deg": float(np.degrees(axial_diff(np.array(gap_theta), np.array(th_b)))),
               "curvature_a": float(sa.curvature_deg_per_km), "curvature_b": float(sb.curvature_deg_per_km),
               "conf_a": float(sa.score_mean), "conf_b": float(sb.score_mean),
               "gap_prob_mean": float(np.nanmean(prob[rr, cc])), "gap_prob_min": float(np.nanmin(prob[rr, cc])),
               "same_regime": float(regimes[a.row, a.col] == regimes[b.row, b.col]) if regimes is not None else np.nan}
        for k, ev in evidence.items():
            v = ev[rr, cc]
            rec[f"ev_{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            rec[f"ev_{k}_min"] = float(np.nanmin(v)) if np.isfinite(v).any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def rule_accept(edges: pd.DataFrame, max_angle_deg: float = 25.0, evidence_quantile: float = 0.6,
                prob_quantile: float = 0.5, min_tip_cos: float = 0.5) -> pd.Series:
    """Collinear AND evidence-supported AND model-supported. Never proximity alone."""
    if edges.empty:
        return pd.Series(dtype=bool)
    geo = (edges.gap_vs_a_deg <= max_angle_deg) & (edges.gap_vs_b_deg <= max_angle_deg) & \
          (edges.tip_a_cos >= min_tip_cos) & (edges.tip_b_cos >= min_tip_cos)
    ev_cols = [c for c in edges if c.startswith("ev_") and c.endswith("_mean")]
    if ev_cols:
        ev = edges[ev_cols].rank(pct=True).mean(axis=1)
        phys = ev >= ev.quantile(evidence_quantile)
    else:
        phys = pd.Series(False, index=edges.index)        # no evidence available -> no link
    model = edges.gap_prob_mean >= edges.gap_prob_mean.quantile(prob_quantile)
    return geo & phys & model


def edge_labels_from_visible(edges: pd.DataFrame, ends: pd.DataFrame, visible: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Training labels for the optional classifier: the gap lies (>= frac of its pixels within 1 px) on VISIBLE
    mapped faults of the unit. Never held-out truth."""
    near = ndimage.binary_dilation(visible, iterations=1)
    y = []
    for r in edges.itertuples():
        a, b = ends.iloc[r.end_a], ends.iloc[r.end_b]
        rr, cc = _line(a.row, a.col, b.row, b.col)
        y.append(float(near[rr, cc].mean() >= frac))
    return np.asarray(y, np.float32)


def fit_edge_classifier(edges: pd.DataFrame, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    cols = [c for c in edges.columns if c not in ("end_a", "end_b", "seg_a", "seg_b")]
    X = edges[cols].fillna(edges[cols].median()).to_numpy()
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y)
    clf.feature_names_ = cols
    return clf


def refine(prob: np.ndarray, binary: np.ndarray, evidence: dict[str, np.ndarray], regimes: np.ndarray | None = None,
           classifier=None, **rule_kw) -> dict:
    """Refine one operating-point mask. Returns edges, accepted links, adjacency and the reconstructed raster."""
    segs, _, seg_r = segment_graph(binary, prob)
    if segs.empty:
        return {"edges": pd.DataFrame(), "accepted": pd.DataFrame(), "raster": binary.astype(np.float32) * prob,
                "graph": {}, "segments": segs}
    ends = segment_endpoints(seg_r)
    edges = candidate_edges(segs, ends, evidence, prob, regimes)
    if edges.empty:
        acc = pd.Series(dtype=bool)
    elif classifier is not None:
        X = edges[classifier.feature_names_].fillna(edges[classifier.feature_names_].median()).to_numpy()
        edges["p_link"] = classifier.predict_proba(X)[:, 1]
        acc = (edges.p_link >= 0.5) & rule_accept(edges, **{**rule_kw, "evidence_quantile": 0.0, "prob_quantile": 0.0})
    else:
        acc = rule_accept(edges, **rule_kw)
    edges["accepted"] = acc.to_numpy() if len(edges) else []
    out = np.where(binary, np.nan_to_num(prob), 0.0).astype(np.float32)
    graph: dict[int, set] = {int(s): set() for s in segs.segment_id}
    accepted = edges[edges.accepted] if len(edges) else edges
    for r in accepted.itertuples():
        a, b = ends.iloc[r.end_a], ends.iloc[r.end_b]
        rr, cc = _line(a.row, a.col, b.row, b.col)
        conf = 0.9 * min(r.conf_a, r.conf_b)
        out[rr, cc] = np.maximum(out[rr, cc], conf)
        graph[r.seg_a].add(r.seg_b)
        graph[r.seg_b].add(r.seg_a)
    return {"edges": edges, "accepted": accepted, "raster": out, "graph": {k: sorted(v) for k, v in graph.items()},
            "segments": segs, "segment_raster": seg_r}
