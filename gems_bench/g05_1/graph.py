"""Stage F: lightweight candidate graph (no GNN prerequisite). Pure numpy / scipy.

Nodes = candidate segments of one unit; edges are label-free:
  * continuity: an endpoint of A lies within ``max_gap_km`` of an endpoint of B, A's outward end direction points at
    B (angle <= ``max_angle_deg``) and the two strikes are compatible (axial difference <= ``max_angle_deg``);
  * branch: an endpoint of A lies within ``branch_km`` of a pixel of B with a strike difference inside
    ``branch_angle_deg`` (a splay / branch geometry).
Per-node features: degrees, continuity component (union-find) size / length, confidence-weighted neighbour support
(confidence = a label-free score such as the W2 rank, weighted exp(-gap / scale)), best neighbour confidence, mean
strike compatibility, regional orientation agreement (length-weighted doubled-angle agreement with every segment
centroid within ``regional_radius_km``) and the distance from the nearest endpoint to a permitted known-fault tip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import axial_diff_deg

PX_KM = 0.1
GRAPH_COLUMNS = ("g_deg_cont", "g_deg_branch", "g_component", "g_component_size", "g_component_length_km",
                 "g_support", "g_best_neighbor_conf", "g_compat_mean", "g_regional_agreement",
                 "g_known_tip_dist_km", "g_isolated")


class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return int(a)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _bearing_deg(dr, dc) -> np.ndarray:
    return np.degrees(np.arctan2(np.asarray(dc, float), -np.asarray(dr, float))) % 360.0


def _circ_diff(a, b) -> np.ndarray:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def edges(G: pd.DataFrame, pixels: dict | None, gc: dict, px_km: float = PX_KM) -> pd.DataFrame:
    """Edge list (src, dst, kind, gap_km, compat_deg) from a geometry table ``G`` (label, end_r0 / c0 / r1 / c1,
    dir0_deg / dir1_deg, strike_deg) and optionally ``pixels`` {label: (rr, cc)} for branch edges."""
    cols = ["src", "dst", "kind", "gap_km", "compat_deg"]
    G = G.dropna(subset=["end_r0", "end_c0", "end_r1", "end_c1"]).reset_index(drop=True)
    if len(G) < 2:
        return pd.DataFrame(columns=cols)
    lab = G.label.to_numpy(int)
    E = np.r_[G[["end_r0", "end_c0"]].to_numpy(float), G[["end_r1", "end_c1"]].to_numpy(float)]
    Ed = np.r_[G.dir0_deg.to_numpy(float), G.dir1_deg.to_numpy(float)]
    Eo = np.r_[np.arange(len(G)), np.arange(len(G))]
    strike = G.strike_deg.to_numpy(float)
    gap_px = float(gc["max_gap_km"]) / px_km
    amax = float(gc["max_angle_deg"])
    rows = []
    tree = cKDTree(E)
    for i, j in tree.query_pairs(gap_px):
        a, b = Eo[i], Eo[j]
        if a == b:
            continue
        v = E[j] - E[i]
        gap = float(np.hypot(*v))
        comp = float(axial_diff_deg(strike[a], strike[b]))
        if gap > 0:
            ok_a = _circ_diff(Ed[i], _bearing_deg(v[0], v[1])) <= amax
            ok_b = _circ_diff(Ed[j], _bearing_deg(-v[0], -v[1])) <= amax
        else:
            ok_a = ok_b = True
        if (ok_a or ok_b) and comp <= amax:
            rows.append((int(lab[a]), int(lab[b]), "continuity", gap * px_km, comp))
    if pixels:
        P = [(k, *pixels[k]) for k in lab if k in pixels]
        if P:
            pl = np.concatenate([np.full(len(r), k) for k, r, _ in P])
            pts = np.c_[np.concatenate([r for _, r, _ in P]), np.concatenate([c for _, _, c in P])].astype(float)
            pt = cKDTree(pts)
            br = float(gc["branch_km"]) / px_km
            lo, hi = (float(x) for x in gc["branch_angle_deg"])
            pos = {int(k): n for n, k in enumerate(lab)}
            for i, nbrs in enumerate(pt.query_ball_point(E, br)):
                a = int(lab[Eo[i]])
                cand = ({int(pl[k]) for k in nbrs} - {a}) & set(pos)
                for b in cand:
                    comp = float(axial_diff_deg(strike[pos[a]], strike[pos[b]]))
                    if lo <= comp <= hi:
                        d = pts[[k for k in nbrs if pl[k] == b]] - E[i]
                        rows.append((a, b, "branch", float(np.hypot(d[:, 0], d[:, 1]).min()) * px_km, comp))
    D = pd.DataFrame(rows, columns=cols)
    if len(D):                                     # one edge per (pair, kind): the shortest gap
        key = np.where(D.src < D.dst, D.src.astype(str) + "-" + D.dst.astype(str),
                       D.dst.astype(str) + "-" + D.src.astype(str))
        D = D.assign(_k=key).sort_values("gap_km").drop_duplicates(["_k", "kind"]).drop(columns="_k")
    return D.reset_index(drop=True)


def node_features(G: pd.DataFrame, D: pd.DataFrame, conf: dict, gc: dict, known_tips: np.ndarray | None = None,
                  px_km: float = PX_KM) -> pd.DataFrame:
    """Per-label graph features (GRAPH_COLUMNS). ``conf`` = {label: label-free confidence in [0, 1]};
    ``known_tips`` = (n, 2) crop coordinates of permitted known-fault endpoints (None = not available)."""
    lab = G.label.to_numpy(int)
    pos = {int(k): n for n, k in enumerate(lab)}
    n = len(lab)
    uf = _UF(n)
    dc, db = np.zeros(n, int), np.zeros(n, int)
    sup, best = np.zeros(n), np.full(n, np.nan)
    cs, cn = np.zeros(n), np.zeros(n, int)
    scale = float(gc["scale_km"])
    for r in D.itertuples():
        a, b = pos.get(int(r.src)), pos.get(int(r.dst))
        if a is None or b is None:
            continue
        for x, y in ((a, b), (b, a)):
            c = float(conf.get(int(lab[y]), np.nan))
            if np.isfinite(c):
                sup[x] += c * np.exp(-float(r.gap_km) / scale)
                best[x] = c if not np.isfinite(best[x]) else max(best[x], c)
            cs[x] += float(r.compat_deg)
            cn[x] += 1
            if r.kind == "continuity":
                dc[x] += 1
            else:
                db[x] += 1
        if r.kind == "continuity":
            uf.union(a, b)
    comp = np.asarray([uf.find(i) for i in range(n)])
    L = G.chain_length_km.to_numpy(float) if "chain_length_km" in G else np.zeros(n)
    csize = pd.Series(comp).map(pd.Series(comp).value_counts()).to_numpy()
    clen = pd.Series(comp).map(pd.Series(np.nan_to_num(L)).groupby(comp).sum()).to_numpy()
    out = pd.DataFrame({"label": lab, "g_deg_cont": dc, "g_deg_branch": db, "g_component": comp,
                        "g_component_size": csize, "g_component_length_km": clen, "g_support": sup,
                        "g_best_neighbor_conf": best, "g_compat_mean": np.where(cn > 0, cs / np.maximum(cn, 1),
                                                                                 np.nan)})
    out["g_regional_agreement"] = regional_agreement(G, float(gc["regional_radius_km"]), px_km)
    if known_tips is not None and len(known_tips) and len(G):
        t = cKDTree(np.asarray(known_tips, float))
        e0 = G[["end_r0", "end_c0"]].to_numpy(float)
        e1 = G[["end_r1", "end_c1"]].to_numpy(float)
        ok = np.isfinite(e0).all(1) & np.isfinite(e1).all(1)
        d = np.full(n, np.nan)
        if ok.any():
            d[ok] = np.minimum(t.query(e0[ok])[0], t.query(e1[ok])[0]) * px_km
        out["g_known_tip_dist_km"] = d
    else:
        out["g_known_tip_dist_km"] = np.nan
    out["g_isolated"] = (dc + db) == 0
    return out


def regional_agreement(G: pd.DataFrame, radius_km: float, px_km: float = PX_KM) -> np.ndarray:
    """Length-weighted doubled-angle agreement in [-1, 1] between a segment's strike and the strikes of the other
    segments whose centroid lies within ``radius_km`` of its centroid (NaN without neighbours)."""
    n = len(G)
    out = np.full(n, np.nan)
    if n < 2 or "strike_deg" not in G:
        return out
    cr = ((G.end_r0 + G.end_r1) / 2).to_numpy(float)
    cc = ((G.end_c0 + G.end_c1) / 2).to_numpy(float)
    s = G.strike_deg.to_numpy(float)
    w = np.nan_to_num(G.get("chain_length_km", pd.Series(1.0, index=G.index)).to_numpy(float), nan=0.0)
    ok = np.isfinite(cr) & np.isfinite(cc) & np.isfinite(s)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return out
    t = cKDTree(np.c_[cr[idx], cc[idx]])
    for k, nb in zip(idx, t.query_ball_point(np.c_[cr[idx], cc[idx]], radius_km / px_km)):
        j = idx[[x for x in nb if idx[x] != k]]
        if len(j) and w[j].sum() > 0:
            out[k] = float((w[j] * np.cos(np.radians(2.0 * (s[j] - s[k])))).sum() / w[j].sum())
    return out


def known_tips(known: np.ndarray) -> np.ndarray:
    """(n, 2) endpoints of the skeleton of a permitted known-fault map (crop coordinates)."""
    from ..p6.stack import endpoints
    from ..postprocess.topology import skeleton

    k = np.asarray(known, bool)
    if not k.any():
        return np.zeros((0, 2))
    return np.argwhere(endpoints(skeleton(k)))
