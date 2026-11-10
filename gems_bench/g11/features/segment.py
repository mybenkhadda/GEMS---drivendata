"""Candidate-segment attributes (Representation C: segment-centred, linking 100 m rasters and the 1 m DEM).

Geometry (metres): length, azimuth (axial, deg from north), sinuosity = length / chord, linearity = PCA
l1/(l1+l2) of the vertices, mean absolute turning angle per km (curvature). Raster evidence is sampled every half
cell along the line. Topology: endpoint connectivity (other segments' endpoints within ``link_radius_m``), branch
count (segments touching the interior) and local density (segment length within 2 km per km^2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..grid import inside, world_to_pixel


def _sample_points(line, step_m: float):
    n = max(int(line.length // step_m) + 1, 2)
    return np.array([line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)])


def geometry_attributes(line) -> dict:
    xy = np.asarray(line.coords)
    chord = float(np.hypot(*(xy[-1] - xy[0])))
    L = float(line.length)
    X = xy - xy.mean(0)
    if len(xy) >= 3:
        w = np.linalg.eigvalsh(np.cov(X.T))
        lin = float(w[1] / (w.sum() + 1e-12))
    else:
        lin = 1.0
    seg = np.diff(xy, axis=0)
    ang = np.arctan2(seg[:, 0], seg[:, 1])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi) if len(ang) > 1 else np.zeros(1)
    return {"length_m": L, "azimuth_deg": float(np.degrees(np.arctan2(*(xy[-1] - xy[0]))) % 180.0),
            "sinuosity": L / max(chord, 1e-6), "linearity": lin,
            "curvature_deg_per_km": float(np.degrees(turn.sum()) / max(L / 1000.0, 1e-6))}


def raster_attributes(line, rasters: dict, grid, step_m: float) -> dict:
    pts = _sample_points(line, step_m)
    r, c = world_to_pixel(pts[:, 0], pts[:, 1], grid)
    ok = inside(r, c, grid)
    out = {}
    for name, a in rasters.items():
        if a is None:
            continue
        v = np.asarray(a[r[ok], c[ok]], np.float32) if ok.any() else np.array([], np.float32)
        v = v[np.isfinite(v)]
        if name.startswith("prob"):
            q = np.quantile(v, [0.1, 0.5, 0.9]) if v.size else [np.nan] * 3
            out.update({f"{name}_mean": float(v.mean()) if v.size else np.nan,
                        f"{name}_max": float(v.max()) if v.size else np.nan,
                        f"{name}_q10": float(q[0]), f"{name}_q50": float(q[1]), f"{name}_q90": float(q[2])})
        else:
            out[f"{name}_mean"] = float(v.mean()) if v.size else np.nan
        out.setdefault("raster_support_fraction", float(ok.mean()))
    return out


def topology_attributes(gdf, link_radius_m: float) -> pd.DataFrame:
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    ends = [(Point(g.coords[0]), Point(g.coords[-1])) for g in geoms]
    rows = []
    for i, g in enumerate(geoms):
        conn = 0
        for p in ends[i]:
            near = tree.query(p.buffer(link_radius_m))
            conn += int(any(j != i and min(ends[j][0].distance(p), ends[j][1].distance(p)) <= link_radius_m
                            for j in near))
        inter = tree.query(g, predicate="intersects")
        branches = sum(1 for j in inter if j != i and not (g.boundary.distance(geoms[j]) < 1e-6))
        local = tree.query(g.centroid.buffer(2000.0))
        dens = sum(geoms[j].intersection(g.centroid.buffer(2000.0)).length for j in local) / 1000.0 / (np.pi * 4.0)
        rows.append({"endpoint_connectivity": conn, "branch_count": branches, "local_density_km_per_km2": dens})
    return pd.DataFrame(rows)


def segment_features(gdf, rasters: dict, grid, cfg: dict, dem_reader=None, crs=None) -> pd.DataFrame:
    """One row per candidate: geometry + raster evidence + topology (+ cross-fault DEM features if a reader is
    given and the segment is DEM-covered). ``rasters`` keys starting with 'prob' get quantile statistics."""
    from .structural import extract_cross_fault_features, sample_cross_fault_profiles

    cg = cfg["candidate_generation"]
    step = grid.res[0] / 2.0
    rows = []
    for g in gdf.geometry:
        r = geometry_attributes(g)
        r.update(raster_attributes(g, rasters, grid, step))
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(gdf):
        df = pd.concat([df, topology_attributes(gdf, cg["link_radius_m"])], axis=1)
    if dem_reader is not None and len(gdf):
        cf = cfg["feature_engineering"]["cross_fault"]
        xf = []
        for g in gdf.geometry:
            try:
                p = sample_cross_fault_profiles(g, dem_reader, cf["offsets_m"], cf["spacing_m"], crs or grid.crs,
                                                cf["resolution_m"])
                d = extract_cross_fault_features(p, cf["inner_m"])
                d["dem_tiles"] = ";".join(p.tile_ids)
            except Exception as e:                                       # noqa: BLE001 - recorded per segment
                d = {"profile_valid_fraction": 0.0, "dem_error": f"{type(e).__name__}: {e}"}
            xf.append(d)
        df = pd.concat([df, pd.DataFrame(xf)], axis=1)
    prob_cols = [c for c in df.columns if c.startswith("prob") and c.endswith("_mean")]
    if len(prob_cols) > 1:
        df["model_agreement"] = 1.0 - df[prob_cols].std(axis=1) / (df[prob_cols].mean(axis=1).abs() + 1e-6)
        df["uncertainty"] = df[prob_cols].std(axis=1)
    return df
