"""Candidate topology: endpoint linking with angular constraints and conservative deduplication.

Linking joins two segments when their facing endpoints are within ``link_radius_m`` AND the connecting chord is
within ``link_max_angle_deg`` (axial) of both segments' local end directions - a gap in a straight trace is
bridged, a crossing / branching geometry is kept as separate segments.
Deduplication removes a segment only if >= ``dedup_overlap`` of its length lies within one cell of a LONGER
segment with compatible strike (``dedup_angle_deg``); branches with a different strike survive.
Candidate IDs are stable: sha1 of the rounded WKT.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _end_dir(line, at_start: bool, d: float = 300.0):
    L = line.length
    p = line.interpolate(0 if at_start else L)
    q = line.interpolate(min(d, L) if at_start else max(L - d, 0))
    v = np.array([p.x - q.x, p.y - q.y])
    return np.array([p.x, p.y]), v / (np.linalg.norm(v) + 1e-12)       # outward direction


def _axial_angle(u, v) -> float:
    c = abs(float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


def link_segments(gdf, link_radius_m: float, max_angle_deg: float, max_iter: int = 5):
    """Greedy endpoint linking (repeated until no link is added or ``max_iter``)."""
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    geoms = list(gdf.geometry)
    attrs = gdf.drop(columns="geometry").to_dict("records")
    for _ in range(max_iter):
        tree = STRtree(geoms)
        used, merged, new_attrs, changed = set(), [], [], False
        for i, g in enumerate(geoms):
            if i in used:
                continue
            best = None
            for si in (True, False):
                pi, di = _end_dir(g, si)
                for j in tree.query(g.buffer(link_radius_m)):
                    if j == i or j in used:
                        continue
                    for sj in (True, False):
                        pj, dj = _end_dir(geoms[j], sj)
                        gap = pj - pi
                        dist = float(np.linalg.norm(gap))
                        if dist > link_radius_m or dist < 1e-6:
                            continue
                        if _axial_angle(gap, di) <= max_angle_deg and _axial_angle(gap, dj) <= max_angle_deg \
                                and np.dot(gap, di) > 0 and np.dot(-gap, dj) > 0:
                            if best is None or dist < best[0]:
                                best = (dist, j, si, sj)
            if best is None:
                continue
            _, j, si, sj = best
            a = list(g.coords)[::-1] if si else list(g.coords)
            b = list(geoms[j].coords) if sj else list(geoms[j].coords)[::-1]
            merged.append(LineString(a + b))
            new_attrs.append({**attrs[i], "linked_from": f"{i}+{j}"})
            used |= {i, j}
            changed = True
        keep = [k for k in range(len(geoms)) if k not in used]
        geoms = [geoms[k] for k in keep] + merged
        attrs = [attrs[k] for k in keep] + new_attrs
        if not changed:
            break
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs=gdf.crs)


def deduplicate(gdf, overlap: float, angle_deg: float, tol_m: float = 100.0):
    from ..features.segment import geometry_attributes

    g = gdf.copy()
    g["_len"] = g.geometry.length
    g = g.sort_values("_len", ascending=False).reset_index(drop=True)
    az = np.array([geometry_attributes(x)["azimuth_deg"] for x in g.geometry]) if len(g) else np.array([])
    keep = np.ones(len(g), bool)
    from shapely.strtree import STRtree

    tree = STRtree(list(g.geometry)) if len(g) else None
    for i in range(len(g)):
        if not keep[i]:
            continue
        for j in tree.query(g.geometry[i].buffer(tol_m)):
            if j <= i or not keep[j]:
                continue
            d = abs(az[i] - az[j]) % 180
            if min(d, 180 - d) > angle_deg:
                continue
            inter = g.geometry[j].intersection(g.geometry[i].buffer(tol_m)).length
            if inter >= overlap * g.geometry[j].length:
                keep[j] = False
    return g[keep].drop(columns="_len").reset_index(drop=True)


def stable_ids(gdf, prefix: str = "C") -> list[str]:
    """Deterministic id per geometry: sha1 of its coordinates rounded to 0.1 m (order-sensitive)."""
    return [f"{prefix}{hashlib.sha1(np.round(np.asarray(g.coords, np.float64), 1).tobytes()).hexdigest()[:12]}"
            for g in gdf.geometry]
