"""Fault-label morphology: skeleton graph, segment geometry, label-suitability diagnostics.

The analysis unit is the *skeleton segment* (node-to-node polyline), not the
raster connected component: components are often merged networks (many faults
joined at intersections) or broken fragments of a single trace. Both effects are
quantified by :func:`component_diagnostics` and :func:`endpoint_gaps`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .io import Grid
from .stats import axial_diff, axial_hist, axial_mean

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_EIGHT = np.ones((3, 3), bool)


@dataclass
class SkeletonGraph:
    paths: list[np.ndarray]          # ordered (row, col) int arrays
    seg_nodes: np.ndarray            # (n_segments, 2) node ids, -1 for closed loops
    nodes: pd.DataFrame              # node_id, r, c, degree, kind


def _neighbor_table(skel: np.ndarray):
    rr, cc = np.nonzero(skel)
    idx = -np.ones(skel.shape, np.int64)
    idx[rr, cc] = np.arange(len(rr))
    pad = np.pad(idx, 1, constant_values=-1)
    nb = np.stack([pad[rr + 1 + dr, cc + 1 + dc] for dr, dc in _OFFSETS], axis=1)
    return rr, cc, nb


def _path_len_px(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    d = np.diff(path, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def trace_skeleton(skel: np.ndarray, spur_px: float = 3.0) -> SkeletonGraph:
    """Trace a one-pixel skeleton into node-to-node segments.

    Junction pixels are clustered into single nodes; tiny self-loops (staircase
    artefacts) and short spurs are pruned, and pass-through (degree-2) nodes are
    merged so that each segment runs between true endpoints / junctions.
    """
    rr, cc, nb = _neighbor_table(skel)
    if len(rr) == 0:
        return SkeletonGraph([], np.zeros((0, 2), int), pd.DataFrame(columns=["node_id", "r", "c", "degree", "kind"]))
    deg = (nb >= 0).sum(1)
    is_node = deg != 2
    node_mask = np.zeros(skel.shape, bool)
    node_mask[rr[is_node], cc[is_node]] = True
    lab, n_nodes = ndimage.label(node_mask, structure=_EIGHT)
    node_of = np.where(is_node, lab[rr, cc] - 1, -1)

    visited = np.zeros(len(rr), bool)
    raw: list[list] = []
    seen_pairs: set[tuple[int, int]] = set()
    for p in np.flatnonzero(is_node):
        for q in nb[p]:
            if q < 0:
                continue
            if is_node[q]:
                if node_of[q] != node_of[p]:
                    key = (min(p, q), max(p, q))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        raw.append([node_of[p], node_of[q], [p, q]])
                continue
            if visited[q]:
                continue
            path, prev, cur = [p, q], p, q
            visited[q] = True
            end = -1
            while True:
                nxt = [r for r in nb[cur] if r >= 0 and r != prev]
                if not nxt:
                    break
                r = nxt[0]
                path.append(r)
                if is_node[r]:
                    end = node_of[r]
                    break
                if visited[r]:
                    break
                visited[r] = True
                prev, cur = cur, r
            raw.append([node_of[p], end, path])
    # isolated closed loops (no node pixel)
    for start in np.flatnonzero(~is_node & ~visited):
        if visited[start]:
            continue
        path, prev, cur = [start], -1, start
        visited[start] = True
        while True:
            nxt = [r for r in nb[cur] if r >= 0 and r != prev and not visited[r]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            visited[cur] = True
            path.append(cur)
        path.append(start)
        raw.append([-1, -1, path])

    segs = {i: [a, b, np.column_stack([rr[s], cc[s]])] for i, (a, b, s) in enumerate(raw)}
    # drop staircase self-loops
    segs = {i: s for i, s in segs.items() if not (s[0] >= 0 and s[0] == s[1] and _path_len_px(s[2]) <= 4.5)}

    def degrees() -> dict[int, int]:
        d: dict[int, int] = defaultdict(int)
        for a, b, _ in segs.values():
            if a >= 0:
                d[a] += 1
            if b >= 0:
                d[b] += 1
        return d

    # prune short spurs hanging off junctions
    d = degrees()
    for i, (a, b, path) in list(segs.items()):
        if _path_len_px(path) < spur_px and ({d.get(a, 0), d.get(b, 0)} & {1}) and max(d.get(a, 0), d.get(b, 0)) >= 3:
            del segs[i]

    # merge through degree-2 nodes
    ends: dict[int, set[int]] = defaultdict(set)
    for i, (a, b, _) in segs.items():
        if a >= 0:
            ends[a].add(i)
        if b >= 0:
            ends[b].add(i)
    next_id = max(segs, default=-1) + 1
    for n in [n for n, s in ends.items() if len(s) == 2]:
        s = list(ends.get(n, ()))
        if len(s) != 2:
            continue
        i, j = s
        A, B = segs[i], segs[j]
        if A[1] != n:
            A = [A[1], A[0], A[2][::-1]]
        if B[0] != n:
            B = [B[1], B[0], B[2][::-1]]
        if A[0] == n or B[1] == n:   # loop through n; leave as is
            continue
        segs[next_id] = [A[0], B[1], np.vstack([A[2], B[2]])]
        del segs[i], segs[j]
        ends[n] = set()
        for node, old in ((A[0], i), (B[1], j)):
            if node >= 0:
                ends[node].discard(old)
                ends[node].add(next_id)
        next_id += 1

    paths = [s[2] for s in segs.values()]
    seg_nodes = np.array([[s[0], s[1]] for s in segs.values()], dtype=int).reshape(-1, 2)
    d = degrees()
    com = ndimage.center_of_mass(node_mask, lab, np.arange(1, n_nodes + 1)) if n_nodes else []
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes),
        "r": [c[0] for c in com],
        "c": [c[1] for c in com],
        "degree": [d.get(k, 0) for k in range(n_nodes)],
    })
    nodes["kind"] = np.select([nodes.degree == 1, nodes.degree >= 3], ["endpoint", "junction"], "pass")
    return SkeletonGraph(paths, seg_nodes, nodes)


def path_geometry(path: np.ndarray, grid: Grid, step_km: float | None = None) -> dict:
    """Length, chord, sinuosity, PCA strike, linearity and curvature of a pixel path."""
    e, n = grid.rc_to_km(path[:, 0], path[:, 1])
    de, dn = np.diff(e), np.diff(n)
    length = float(np.hypot(de, dn).sum())
    chord = float(np.hypot(e[-1] - e[0], n[-1] - n[0]))
    out = {"length_km": length, "chord_km": chord, "sinuosity": length / chord if chord > 0 else np.nan}
    if len(path) >= 3:
        cov = np.cov(np.vstack([e, n]))
        w, v = np.linalg.eigh(cov)
        out["azimuth"] = float(np.degrees(np.arctan2(v[0, 1], v[1, 1])) % 180)
        out["linearity"] = float(1 - w[0] / w[1]) if w[1] > 0 else np.nan
    else:
        out["azimuth"] = float(np.degrees(np.arctan2(e[-1] - e[0], n[-1] - n[0])) % 180)
        out["linearity"] = 1.0
    step = step_km or 3 * grid.pixel_km
    if length > 4 * step and len(path) >= 5:
        es, ns = gaussian_filter1d(e, 2, mode="nearest"), gaussian_filter1d(n, 2, mode="nearest")
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(es), np.diff(ns)))])
        t = np.arange(0, s[-1], step)
        er, nr = np.interp(t, s, es), np.interp(t, s, ns)
        head = np.arctan2(np.diff(nr), np.diff(er))
        turn = np.abs(np.angle(np.exp(1j * np.diff(head))))
        k = np.degrees(turn) / step
        out["curv_mean_deg_km"] = float(k.mean()) if k.size else np.nan
        out["curv_p90_deg_km"] = float(np.percentile(k, 90)) if k.size else np.nan
    else:
        out["curv_mean_deg_km"] = np.nan
        out["curv_p90_deg_km"] = np.nan
    return out


@dataclass
class FaultNetwork:
    mask: np.ndarray
    skeleton: np.ndarray
    components: np.ndarray           # 8-connected component labels of the raw mask
    graph: SkeletonGraph
    segments: pd.DataFrame
    distance_km: np.ndarray          # distance to nearest skeleton pixel

    @property
    def paths(self) -> list[np.ndarray]:
        return self.graph.paths

    @property
    def nodes(self) -> pd.DataFrame:
        return self.graph.nodes


def region_ids(rows, cols, grid: Grid, region_grid: tuple[int, int]) -> np.ndarray:
    nr, nc = region_grid
    rb = np.clip((np.asarray(rows) * nr / grid.height).astype(int), 0, nr - 1)
    cb = np.clip((np.asarray(cols) * nc / grid.width).astype(int), 0, nc - 1)
    return rb * nc + cb


def region_name(rid: int, region_grid: tuple[int, int]) -> str:
    nr, nc = region_grid
    rows = ["N", "C", "S"] if nr == 3 else [f"R{i}" for i in range(nr)]
    cols = ["W", "C", "E"] if nc == 3 else [f"C{i}" for i in range(nc)]
    return f"{rows[rid // nc]}{cols[rid % nc]}"


def build_fault_network(mask: np.ndarray, grid: Grid, region_grid=(3, 3)) -> FaultNetwork:
    mask = mask.astype(bool)
    skel = skeletonize(mask)
    comps, _ = ndimage.label(mask, structure=_EIGHT)
    width_px = 2 * ndimage.distance_transform_edt(mask)[skel].mean() - 1 if skel.any() else 1.0
    graph = trace_skeleton(skel, spur_px=max(3.0, 2 * width_px))

    rows = []
    for k, (path, (a, b)) in enumerate(zip(graph.paths, graph.seg_nodes)):
        g = path_geometry(path, grid)
        mid = path[len(path) // 2]
        rows.append({
            "seg_id": k, "start_node": a, "end_node": b, "n_px": len(path),
            **g,
            "component": int(comps[mid[0], mid[1]]),
            "r_mid": int(mid[0]), "c_mid": int(mid[1]),
            "touches_edge": bool((path[:, 0].min() <= 1) | (path[:, 1].min() <= 1)
                                 | (path[:, 0].max() >= grid.height - 2) | (path[:, 1].max() >= grid.width - 2)),
        })
    seg = pd.DataFrame(rows)
    if len(seg):
        deg = graph.nodes.set_index("node_id")["degree"]
        seg["start_deg"] = seg.start_node.map(deg).fillna(0).astype(int)
        seg["end_deg"] = seg.end_node.map(deg).fillna(0).astype(int)
        seg["is_loop"] = seg.start_node == seg.end_node
        seg["n_junction_ends"] = (seg.start_deg >= 3).astype(int) + (seg.end_deg >= 3).astype(int)
        seg["region"] = region_ids(seg.r_mid, seg.c_mid, grid, region_grid)
        x, y = grid.rc_to_xy(seg.r_mid, seg.c_mid)
        seg["x_mid"], seg["y_mid"] = x, y
    dist = ndimage.distance_transform_edt(~skel, sampling=(grid.pixel_km_y, grid.pixel_km_x)).astype(np.float32)
    return FaultNetwork(mask, skel, comps, graph, seg, dist)


def component_diagnostics(net: FaultNetwork, grid: Grid, short_km: float) -> pd.DataFrame:
    """Per raster component: is it a clean single trace, a network, a fragment or a blob?"""
    labels = net.components
    n = labels.max()
    if n == 0:
        return pd.DataFrame()
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(labels), labels, idx)
    skel_px = ndimage.sum(net.skeleton, labels, idx)
    seg = net.segments
    per = seg.groupby("component").agg(
        n_segments=("seg_id", "size"), length_km=("length_km", "sum"),
        n_junction_ends=("n_junction_ends", "sum"), has_loop=("is_loop", "any"),
    ).reindex(idx)
    df = pd.DataFrame({"component": idx, "area_px": area, "skeleton_px": skel_px})
    df = df.join(per, on="component")
    df["n_segments"] = df.n_segments.fillna(0).astype(int)
    df["length_km"] = df.length_km.fillna(0.0)
    df["mean_width_px"] = df.area_px / np.maximum(df.skeleton_px, 1)
    df["kind"] = np.select(
        [df.mean_width_px > 4, df.length_km < short_km, df.n_segments > 1,
         df.has_loop.fillna(False).astype(bool)],
        ["blob", "fragment", "network", "loop"], "single_trace",
    )
    return df


def endpoint_gaps(net: FaultNetwork, grid: Grid, max_gap_km: float, max_misalign_deg: float) -> pd.DataFrame:
    """Endpoint pairs from *different* components that look like one interrupted trace."""
    recs = []
    deg = net.nodes.set_index("node_id")["degree"] if len(net.nodes) else pd.Series(dtype=int)
    k = 5
    for sid, path, (a, b) in zip(net.segments.seg_id, net.paths, net.graph.seg_nodes):
        if len(path) < 3:
            continue
        for node, end, back in ((a, path[0], path[min(k, len(path) - 1)]), (b, path[-1], path[max(-k - 1, -len(path))])):
            if node >= 0 and deg.get(node, 0) == 1:
                e0, n0 = grid.rc_to_km(end[0], end[1])
                e1, n1 = grid.rc_to_km(back[0], back[1])
                recs.append((sid, node, float(e0), float(n0), float(e0 - e1), float(n0 - n1),
                             int(net.components[end[0], end[1]]), int(end[0]), int(end[1])))
    if len(recs) < 2:
        return pd.DataFrame()
    ep = pd.DataFrame(recs, columns=["seg_id", "node", "e", "n", "de", "dn", "component", "r", "c"])
    tree = cKDTree(ep[["e", "n"]].to_numpy())
    pairs = tree.query_pairs(max_gap_km, output_type="ndarray")
    out = []
    for i, j in pairs:
        A, B = ep.iloc[i], ep.iloc[j]
        if A.component == B.component:
            continue
        v = np.array([B.e - A.e, B.n - A.n])
        gap = np.hypot(*v)
        if gap == 0:
            continue

        def ang(u, w):
            return np.degrees(np.arccos(np.clip(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12), -1, 1)))

        a1 = ang(np.array([A.de, A.dn]), v)
        a2 = ang(np.array([B.de, B.dn]), -v)
        if a1 <= max_misalign_deg and a2 <= max_misalign_deg:
            out.append({"seg_a": int(A.seg_id), "seg_b": int(B.seg_id), "gap_km": gap,
                        "misalign_a": a1, "misalign_b": a2,
                        "r_a": int(A.r), "c_a": int(A.c), "r_b": int(B.r), "c_b": int(B.c)})
    return pd.DataFrame(out)


def trace_density(net: FaultNetwork, grid: Grid, radius_km: float) -> np.ndarray:
    """Kernel density of mapped trace length (km of trace per 100 km^2)."""
    s = grid.km_to_px(radius_km)
    dens = ndimage.gaussian_filter(net.skeleton.astype(np.float32) * grid.pixel_km, s, mode="constant")
    return (dens / (grid.pixel_km_x * grid.pixel_km_y) * 100.0).astype(np.float32)


def regional_summary(net: FaultNetwork, grid: Grid, valid: np.ndarray, region_grid=(3, 3)) -> pd.DataFrame:
    seg = net.segments
    rr, cc = np.nonzero(valid)
    area = np.bincount(region_ids(rr, cc, grid, region_grid), minlength=region_grid[0] * region_grid[1])
    area = area * grid.pixel_km_x * grid.pixel_km_y
    jn = net.nodes[net.nodes.kind == "junction"]
    jreg = np.bincount(region_ids(jn.r.astype(int), jn.c.astype(int), grid, region_grid),
                       minlength=len(area)) if len(jn) else np.zeros(len(area))
    rows = []
    for rid in range(len(area)):
        s = seg[seg.region == rid]
        mean_az, R = axial_mean(s.azimuth, s.length_km) if len(s) else (np.nan, np.nan)
        rows.append({
            "region": rid, "region_name": region_name(rid, region_grid), "area_km2": area[rid],
            "n_segments": len(s), "trace_km": s.length_km.sum(),
            "density_km_per_100km2": 100 * s.length_km.sum() / area[rid] if area[rid] else np.nan,
            "junctions_per_1000km2": 1000 * jreg[rid] / area[rid] if area[rid] else np.nan,
            "median_length_km": s.length_km.median(), "median_sinuosity": s.sinuosity.median(),
            "median_curv_deg_km": s.curv_mean_deg_km.median(),
            "mean_strike": mean_az, "strike_R": R,
        })
    return pd.DataFrame(rows)


def length_weighted_rose(seg: pd.DataFrame, bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    return axial_hist(seg.azimuth, seg.length_km, bins)


def strike_families(seg: pd.DataFrame, bins: int = 36, n_max: int = 3) -> list[float]:
    """Peaks of the length-weighted rose (circularly smoothed) = dominant fault families."""
    centers, h = length_weighted_rose(seg, bins)
    hs = ndimage.gaussian_filter1d(h, 1.0, mode="wrap")
    peaks = [i for i in range(bins) if hs[i] >= hs[i - 1] and hs[i] >= hs[(i + 1) % bins] and hs[i] > 1.0 / bins]
    peaks = sorted(peaks, key=lambda i: -hs[i])[:n_max]
    fams = []
    for i in peaks:
        if all(axial_diff(centers[i], f) > 20 for f in fams):
            fams.append(float(centers[i]))
    return fams
