"""G02-D geological network compatibility: label-free lineament graph extracted from an evidence map.

Extraction
    hysteresis threshold of the evidence at the unit's top ``hi_area_frac`` / ``lo_area_frac`` of valid pixels (area
    fractions, never labels), one-pixel skeleton, components shorter than ``min_length_m`` dropped.
Graph
    nodes   endpoints (crossing number 1) and junction clusters (crossing number >= 3, with their 8-neighbours)
    edges   skeleton components after removing junction clusters = candidate lineament segments (traced polylines)
Segment measurements
    length, chord / length (straightness), principal-axis azimuth (axial), linearity (1 - l2 / l1), turning curvature
    (deg / km), mean evidence and mean of any attached map (G02-A, G02-B, hypothesis supports ...), junction degree and
    the largest intersection angle at its nodes, continuation support (another segment end within ``gap_m`` that is
    co-axial within ``max_angle_deg`` AND either ahead of the end within ``max_bearing_deg`` (colinear continuation /
    branch) or laterally within ``stepover_m`` (en-echelon array / stepover)), regional trend (length-weighted
    doubled-angle mean of the OTHER segments within ``regional_radius_m``) and alignment with it, local fragmentation
    (segments per 100 km^2).
Score (fixed, nothing learned)
    s = mean evidence * (1 - exp(-L / L0)) * (0.5 + 0.5 straightness) * (1 - c + c * continuation),  c = 0.25
    An isolated straight segment keeps >= 75 % of its score: connectivity is rewarded, never forced. Regional alignment
    is measured, not scored (faults may cut the regional grain).
Raster (G02-D)
    each pixel takes the score of its nearest scored skeleton pixel, discounted by exp(-d^2 / (2 spread^2)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .geology_truncation import remove_short, skeleton_nodes

STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))   # 4-neighbours first


def hysteresis_skeleton(evidence: np.ndarray, valid: np.ndarray, pixel_m, hi_frac: float, lo_frac: float,
                        min_len_m: float) -> tuple[np.ndarray, dict]:
    from skimage.filters import apply_hysteresis_threshold
    from skimage.morphology import skeletonize

    e = np.where(np.asarray(valid, bool) & np.isfinite(evidence), evidence, np.nan).astype(np.float32)
    vals = e[np.isfinite(e)]
    if vals.size < 100:
        return np.zeros(e.shape, bool), {"hi": np.nan, "lo": np.nan, "n_valid": int(vals.size)}
    hi = float(np.quantile(vals, 1.0 - float(hi_frac)))
    lo = float(np.quantile(vals, 1.0 - float(lo_frac)))
    if not hi > lo:                                            # flat evidence: no lineament can be told apart
        return np.zeros(e.shape, bool), {"hi": hi, "lo": lo, "n_valid": int(vals.size)}
    m = apply_hysteresis_threshold(np.nan_to_num(e, nan=lo - 1.0), lo, hi)
    skel = remove_short(skeletonize(m), pixel_m, min_len_m)
    return skel, {"hi": hi, "lo": lo, "n_valid": int(vals.size)}


def _trace(pix: set, start: tuple) -> list:
    path, seen, cur = [start], {start}, start
    while True:
        nxt = None
        for dr, dc in STEPS:
            p = (cur[0] + dr, cur[1] + dc)
            if p in pix and p not in seen:
                nxt = p
                break
        if nxt is None:
            return path
        path.append(nxt)
        seen.add(nxt)
        cur = nxt


def _axial_diff(a, b):
    return np.abs((np.asarray(a, np.float64) - np.asarray(b, np.float64) + 90.0) % 180.0 - 90.0)


def segment_graph(skel: np.ndarray, pixel_m) -> dict:
    """Segments (traced paths), junction nodes and incidence of a one-pixel skeleton (window pixel coordinates)."""
    skel = np.asarray(skel, bool)
    eight = np.ones((3, 3), bool)
    ep, jn = skeleton_nodes(skel)
    jcl = ndimage.binary_dilation(jn, eight) & skel
    seg_mask = skel & ~jcl
    seg_lab, nseg = ndimage.label(seg_mask, structure=eight)
    node_lab, nnode = ndimage.label(jcl, structure=eight)
    py, px = float(pixel_m[0]), float(pixel_m[1])
    paths = []
    if nseg:
        idx = np.flatnonzero(seg_lab.ravel())
        lab = seg_lab.ravel()[idx]
        order = np.argsort(lab, kind="stable")
        idx, lab = idx[order], lab[order]
        cuts = np.flatnonzero(np.diff(lab)) + 1
        W = skel.shape[1]
        nb = ndimage.convolve(seg_mask.astype(np.int8), eight.astype(np.int8), mode="constant") - seg_mask
        for grp in np.split(idx, cuts):
            rr, cc = grp // W, grp % W
            pix = set(zip(rr.tolist(), cc.tolist()))
            ends = [(r, c) for r, c in zip(rr.tolist(), cc.tolist()) if nb[r, c] == 1]
            start = min(ends) if ends else (int(rr[0]), int(cc[0]))
            p = np.asarray(_trace(pix, start), np.int64)
            paths.append({"rows": p[:, 0], "cols": p[:, 1], "n_px": int(grp.size)})
    inc = pd.DataFrame(columns=["segment", "node"])
    if nseg and nnode:
        ndil = ndimage.grey_dilation(node_lab, size=(3, 3))
        m = (seg_lab > 0) & (ndil > 0)
        if m.any():
            pr = np.unique(np.stack([seg_lab[m], ndil[m]], 1), axis=0)
            inc = pd.DataFrame({"segment": pr[:, 0] - 1, "node": pr[:, 1]})
    com = ndimage.center_of_mass(jcl, node_lab, list(range(1, nnode + 1))) if nnode else []
    nodes = pd.DataFrame([{"node": k + 1, "row": float(r), "col": float(c)} for k, (r, c) in enumerate(com)],
                         columns=["node", "row", "col"])
    if len(nodes) and len(inc):
        nodes = nodes.merge(inc.groupby("node").size().rename("degree").reset_index(), on="node", how="left")
    return {"paths": paths, "seg_lab": seg_lab.astype(np.int32), "node_lab": node_lab.astype(np.int32),
            "incidence": inc, "nodes": nodes,
            "endpoints": ep, "pixel_m": (py, px)}


def segment_attributes(g: dict, maps: dict, cfg: dict) -> pd.DataFrame:
    """One row per segment: geometry, attached-map means, continuation, intersections, regional trend, score."""
    from scipy.spatial import cKDTree

    nc = cfg["network"]
    py, px = g["pixel_m"]
    rows = []
    for k, p in enumerate(g["paths"]):
        r, c = p["rows"].astype(np.float64), p["cols"].astype(np.float64)
        E, N = c * px, -r * py
        steps = np.hypot(np.diff(E), np.diff(N))
        length = float(steps.sum()) + 0.5 * (px + py)
        chord = float(np.hypot(E[-1] - E[0], N[-1] - N[0])) + 0.5 * (px + py)
        if len(E) >= 3:
            cov = np.cov(np.stack([E, N]))
            w, V = np.linalg.eigh(cov)
            vE, vN = V[0, 1], V[1, 1]
            lin = float(1.0 - w[0] / w[1]) if w[1] > 0 else 0.0
        else:
            vE, vN, lin = E[-1] - E[0], N[-1] - N[0], 1.0
        az = float(np.degrees(np.arctan2(vE, vN)) % 180.0)
        kt = max(1, min(len(E) - 1, 5))
        t_end = np.array([E[-1] - E[-1 - kt], N[-1] - N[-1 - kt]]) if len(E) > 1 else np.array([vE, vN])
        t_start = np.array([E[0] - E[kt], N[0] - N[kt]]) if len(E) > 1 else -np.array([vE, vN])
        sub = max(1, int(round(500.0 / max(px, py))))         # turning angles on a 500 m resampled path
        Es, Ns = E[::sub], N[::sub]
        if len(Es) >= 3:
            h = np.degrees(np.arctan2(np.diff(Es), np.diff(Ns)))
            turn = np.abs((np.diff(h) + 180.0) % 360.0 - 180.0)
            curv = float(turn.sum() / max(length / 1000.0, 1e-6))
        else:
            curv = 0.0
        row = {"segment": k, "n_px": p["n_px"], "length_m": length, "straightness": min(chord / length, 1.0),
               "azimuth": az, "linearity": lin, "curvature_deg_per_km": curv,
               "row0": int(p["rows"][0]), "col0": int(p["cols"][0]), "row1": int(p["rows"][-1]),
               "col1": int(p["cols"][-1]), "cE": float(E.mean()), "cN": float(N.mean()),
               "endE": (float(E[0]), float(E[-1])), "endN": (float(N[0]), float(N[-1])),
               "tan": (t_start / (np.linalg.norm(t_start) + 1e-9), t_end / (np.linalg.norm(t_end) + 1e-9))}
        ir, ic = p["rows"], p["cols"]
        for name, a in maps.items():
            v = np.asarray(a, np.float32)[ir, ic]
            row[f"mean_{name}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows.append(row)
    S = pd.DataFrame(rows)
    if not len(S):
        return pd.DataFrame(columns=["segment", "length_m", "straightness", "azimuth", "score"])
    # ---- continuation (colinear branch or stepover) -----------------------------------------------------------
    ends = np.array([(e, n_, i, j) for i, (es, ns) in enumerate(zip(S.endE, S.endN)) for j, (e, n_) in
                     enumerate(zip(es, ns))], np.float64)
    tree = cKDTree(ends[:, :2])
    sup = np.zeros((len(S), 2), bool)
    kind = [["", ""] for _ in range(len(S))]
    azs = S.azimuth.to_numpy()
    for q, (e, n_, i, j) in enumerate(ends):
        i, j = int(i), int(j)
        t = S.tan.iloc[i][j]
        ax = np.array([np.sin(np.radians(azs[i])), np.cos(np.radians(azs[i]))])
        for o in tree.query_ball_point([e, n_], float(nc["gap_m"])):
            oi = int(ends[o, 2])
            if oi == i or _axial_diff(azs[i], azs[oi]) > float(nc["max_angle_deg"]):
                continue
            w = ends[o, :2] - np.array([e, n_])
            dist = float(np.hypot(*w))
            if dist < 1e-6:
                sup[i, j], kind[i][j] = True, "touching"
                break
            bearing = np.degrees(np.arccos(np.clip(np.dot(w / dist, t), -1, 1)))
            lateral = abs(float(w[0] * ax[1] - w[1] * ax[0]))
            if bearing <= float(nc["max_bearing_deg"]):
                sup[i, j], kind[i][j] = True, "colinear"
                break
            if lateral <= float(nc["stepover_m"]):
                sup[i, j], kind[i][j] = True, "stepover"
    S["continuation"] = sup.mean(1)
    S["continuation_kind"] = ["/".join(x for x in k if x) for k in kind]
    # ---- intersections at junction nodes ----------------------------------------------------------------------
    inc = g["incidence"]
    S["n_junctions"] = 0
    S["max_intersection_deg"] = np.nan
    if len(inc):
        for node, grp in inc.groupby("node"):
            segs = [int(s) for s in grp.segment if 0 <= int(s) < len(S)]
            for s in segs:
                S.loc[s, "n_junctions"] += 1
            if len(segs) >= 2:
                for s in segs:
                    others = [o for o in segs if o != s]
                    ang = float(np.max(_axial_diff(azs[s], azs[others])))
                    cur = S.loc[s, "max_intersection_deg"]
                    S.loc[s, "max_intersection_deg"] = ang if not np.isfinite(cur) else max(cur, ang)
    # ---- regional trend + fragmentation ------------------------------------------------------------------------
    ctree = cKDTree(np.stack([S.cE, S.cN], 1))
    Rr = float(nc["regional_radius_m"])
    trend, res, align, frag = [], [], [], []
    L = S.length_m.to_numpy()
    for i, (e, n_) in enumerate(zip(S.cE, S.cN)):
        nb = [o for o in ctree.query_ball_point([e, n_], Rr) if o != i]
        frag.append(len(nb) * 1e8 / (np.pi * Rr * Rr))
        if not nb:
            trend.append(np.nan)
            res.append(0.0)
            align.append(0.5)
            continue
        a2 = np.radians(2.0 * azs[nb])
        C, Sn = float((L[nb] * np.cos(a2)).sum()), float((L[nb] * np.sin(a2)).sum())
        R = float(np.hypot(C, Sn) / max(L[nb].sum(), 1e-9))
        tr = float(np.degrees(0.5 * np.arctan2(Sn, C)) % 180.0)
        al = 0.5 * (1.0 + np.cos(np.radians(2.0 * (azs[i] - tr))))
        trend.append(tr)
        res.append(R)
        align.append(R * al + (1.0 - R) * 0.5)
    S["regional_trend_deg"], S["regional_coherence"], S["regional_alignment"] = trend, res, align
    S["segments_per_100km2"] = frag
    ev = S["mean_evidence"] if "mean_evidence" in S else pd.Series(np.nan, index=S.index)
    cw = float(nc["continuation_weight"])
    S["score"] = (np.nan_to_num(ev.to_numpy(), nan=0.0) * (1.0 - np.exp(-L / float(nc["length_scale_m"])))
                  * (0.5 + 0.5 * S.straightness.to_numpy()) * (1.0 - cw + cw * S.continuation.to_numpy()))
    return S.drop(columns=["endE", "endN", "tan"])


def network_raster(g: dict, S: pd.DataFrame, valid: np.ndarray, pixel_m, spread_m: float) -> np.ndarray:
    """G02-D: nearest scored skeleton pixel's segment score x exp(-d^2 / (2 spread^2)); 0 without any segment."""
    v = np.asarray(valid, bool)
    img = np.zeros(v.shape, np.float32)
    if len(S) and "score" in S:
        sc = np.concatenate([[0.0], S.sort_values("segment").score.to_numpy(np.float64)]).astype(np.float32)
        lab = g["seg_lab"]
        img = sc[np.clip(lab, 0, len(sc) - 1)]
        inc = g["incidence"]
        if len(inc):                                           # junction clusters take their best incident segment
            best = inc.assign(s=sc[inc.segment.to_numpy() + 1]).groupby("node").s.max()
            nl = g["node_lab"]
            nimg = np.zeros(int(nl.max()) + 1, np.float32)
            nimg[best.index.to_numpy()] = best.to_numpy()
            img = np.maximum(img, nimg[nl])
    on = img > 0
    if not on.any():
        return np.where(v, 0.0, np.nan).astype(np.float32)
    d, (ir, ic) = ndimage.distance_transform_edt(~on, sampling=(float(pixel_m[0]), float(pixel_m[1])),
                                                 return_indices=True)
    out = img[ir, ic] * np.exp(-0.5 * (d / float(spread_m)) ** 2)
    return np.where(v, out, np.nan).astype(np.float32)


def build_network(evidence: np.ndarray, valid: np.ndarray, pixel_m, cfg: dict, attach: dict | None = None) -> dict:
    """Full G02-D pipeline on one window: skeleton, graph, segment table, raster."""
    nc = cfg["network"]
    skel, th = hysteresis_skeleton(evidence, valid, pixel_m, float(nc["hi_area_frac"]), float(nc["lo_area_frac"]),
                                   float(nc["min_length_m"]))
    g = segment_graph(skel, pixel_m)
    maps = {"evidence": evidence, **(attach or {})}
    S = segment_attributes(g, maps, cfg)
    return {"skeleton": skel, "graph": g, "segments": S, "thresholds": th,
            "raster": network_raster(g, S, valid, pixel_m, float(nc["spread_m"]))}


def segment_features(g: dict, S: pd.DataFrame, rc_to_xy, r0: int = 0, c0: int = 0, extra: dict | None = None) -> list:
    """GeoJSON-ready features (CRS coordinates of pixel centres; ``r0, c0`` = window offset in the full grid)."""
    feats = []
    if not len(S):
        return feats
    keep = [c for c in S.columns if c not in ("continuation_kind",)]
    by = S.set_index("segment")
    for k, p in enumerate(g["paths"]):
        if k not in by.index:
            continue
        x, y = rc_to_xy(p["rows"] + r0, p["cols"] + c0)
        props = {}
        for c, v in by.loc[k, [c for c in keep if c != "segment"]].items():
            v = v.item() if hasattr(v, "item") else v
            props[c] = None if (isinstance(v, float) and not np.isfinite(v)) else v
        props.update({"segment": int(k), "continuation_kind": by.loc[k, "continuation_kind"], **(extra or {})})
        feats.append({"coords": list(zip(np.asarray(x, float).tolist(), np.asarray(y, float).tolist())),
                      "properties": props})
    return feats
