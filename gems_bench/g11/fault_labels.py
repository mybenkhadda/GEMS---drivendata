"""Fault geometries and multi-resolution label targets.

Label semantics: mapped absence is NOT verified absence. States (``labels.states``): OUTSIDE (no survey),
BACKGROUND (unlabeled, not a confirmed negative), NEAR (within ``near_band_m`` of a mapped fault), FAULT, EXCLUDED.
Targets built here (``Y_*``) are TRAINING TARGETS / EVALUATION references; distance or proximity layers derived from
labels are registered ``label_derived=True`` and refused as predictors by the feature store.

When no fault vector file exists, segments are traced from the label raster skeleton
(``segments_from_raster``) and every geometry carries ``source='raster_skeleton'`` - attributes are never invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


class FaultGeometryError(ValueError):
    pass


def load_fault_geometries(path, target_crs):
    """Read a line layer and reproject to ``target_crs``; keeps every original attribute + ``orig_fid``."""
    import geopandas as gpd

    g = gpd.read_parquet(path) if str(path).endswith(".parquet") else gpd.read_file(path)
    if g.crs is None:
        raise FaultGeometryError(f"{path}: fault layer has no CRS")
    bad = set(g.geom_type.dropna()) - {"LineString", "MultiLineString"}
    if bad:
        raise FaultGeometryError(f"{path}: unsupported geometry types {sorted(bad)}")
    g = g.to_crs(target_crs)
    g["orig_fid"] = np.arange(len(g))
    g["source"] = g.get("source", str(path))
    return g


def repair_fault_geometries(gdf):
    """make_valid, explode multi-parts, drop empty / zero-length and exact duplicates; adds length_m, azimuth_deg."""
    from shapely import make_valid

    g = gdf.copy()
    g["geometry"] = g.geometry.map(lambda x: make_valid(x) if x is not None and not x.is_valid else x)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g = g.explode(index_parts=False).reset_index(drop=True)
    g = g[g.geom_type == "LineString"]
    g = g[g.geometry.length > 0]
    g = g[~g.geometry.to_wkb().duplicated()].reset_index(drop=True)
    g["length_m"] = g.geometry.length
    g["azimuth_deg"] = g.geometry.map(line_azimuth)
    return g


def line_azimuth(line) -> float:
    """Axial azimuth (deg clockwise from north, 0-180) of the end-to-end chord."""
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return float(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 180.0)


def build_fault_spatial_index(gdf):
    from shapely.strtree import STRtree

    return STRtree(list(gdf.geometry))


def label_raster(path, grid, min_value: float = 1) -> np.ndarray:
    """Mapped-fault mask on the master grid (or an exactly nested sub-grid, e.g. the SMOKE window)."""
    import rasterio

    from .grid import GridError, raster_window_on_grid

    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise FaultGeometryError(f"{path}: label raster has no CRS")
        try:
            win = raster_window_on_grid(ds, grid)
        except GridError as e:
            raise FaultGeometryError(f"{path}: label raster grid differs from the master grid ({e})") from e
        a = ds.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
    return np.nan_to_num(a, nan=0.0) >= min_value


def rasterize_faults(gdf, grid, all_touched: bool = True) -> np.ndarray:
    from rasterio.features import rasterize

    if not len(gdf):
        return np.zeros(grid.shape, bool)
    return rasterize([(g, 1) for g in gdf.geometry], out_shape=grid.shape, transform=grid.affine, fill=0,
                     dtype="uint8", all_touched=all_touched).astype(bool)


def segments_from_raster(mask: np.ndarray, grid, min_length_m: float, simplify_m: float):
    """Trace junction-free skeleton branches of the label raster into LineStrings (source='raster_skeleton')."""
    import geopandas as gpd
    from shapely.geometry import LineString
    from skimage.morphology import skeletonize

    sk = skeletonize(mask)
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    branches = sk & (nb <= 2)
    lab, n = ndimage.label(branches, structure=np.ones((3, 3), bool))
    a, _, c, _, e, f = grid.transform[:6]
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lab[sl] == i)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        if len(rr) < 2:
            continue
        order = _order_path(rr, cc)
        xs, ys = c + a * (cc[order] + 0.5), f + e * (rr[order] + 0.5)
        line = LineString(np.c_[xs, ys]).simplify(simplify_m)
        if line.length >= min_length_m:
            rows.append({"orig_fid": i, "source": "raster_skeleton", "geometry": line})
    if not rows:
        return gpd.GeoDataFrame({"orig_fid": [], "source": [], "length_m": [], "azimuth_deg": []}, geometry=[],
                                crs=grid.crs)
    return repair_fault_geometries(gpd.GeoDataFrame(rows, geometry="geometry", crs=grid.crs))


def _order_path(rr, cc) -> np.ndarray:
    """Order 8-connected branch pixels from one endpoint to the other (greedy nearest neighbour)."""
    pts = np.c_[rr, cc].astype(float)
    n = len(pts)
    d = np.abs(pts[:, None] - pts[None]).max(-1)
    deg = (d <= 1).sum(1) - 1
    start = int(np.argmin(deg))
    order, seen = [start], np.zeros(n, bool)
    seen[start] = True
    for _ in range(n - 1):
        dd = np.where(seen, np.inf, d[order[-1]])
        j = int(np.argmin(dd))
        if not np.isfinite(dd[j]):
            break
        order.append(j)
        seen[j] = True
    return np.asarray(order)


def distance_to_fault_m(mask: np.ndarray, grid) -> np.ndarray:
    return (ndimage.distance_transform_edt(~mask, sampling=grid.res[::-1]) if mask.any()
            else np.full(mask.shape, np.inf)).astype(np.float32)


def orientation_raster(gdf, grid) -> tuple[np.ndarray, np.ndarray]:
    """Axial (cos 2a, sin 2a) of the nearest segment's local azimuth on fault pixels (NaN elsewhere)."""
    from rasterio.features import rasterize

    c2 = np.full(grid.shape, np.nan, np.float32)
    s2 = np.full(grid.shape, np.nan, np.float32)
    for g in gdf.geometry:
        coords = np.asarray(g.coords)
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            from shapely.geometry import LineString

            az = np.radians(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 180.0)
            m = rasterize([(LineString([(x0, y0), (x1, y1)]), 1)], out_shape=grid.shape, transform=grid.affine,
                          fill=0, dtype="uint8", all_touched=True).astype(bool)
            c2[m], s2[m] = np.cos(2 * az), np.sin(2 * az)
    return c2, s2


def label_states(mask: np.ndarray, valid: np.ndarray, dist_m: np.ndarray, cfg: dict,
                 excluded: np.ndarray | None = None) -> np.ndarray:
    S = cfg["labels"]["states"]
    st = np.full(mask.shape, S["OUTSIDE"], np.uint8)
    st[valid] = S["BACKGROUND"]
    st[valid & (dist_m <= cfg["labels"]["near_band_m"])] = S["NEAR"]
    st[valid & mask] = S["FAULT"]
    if excluded is not None:
        st[valid & excluded] = S["EXCLUDED"]
    return st


def make_targets(mask: np.ndarray, valid: np.ndarray, grid, cfg: dict, gdf=None) -> dict:
    """Target variants (float32, NaN outside the survey): Y_TRACE_100M, Y_BUFFER_<d>M for every configured buffer,
    Y_DISTANCE (m), Y_ORIENTATION_C2 / _S2 (only with geometries). Y_CONFIDENCE only if a confidence attribute
    exists in the source layer."""
    dist = distance_to_fault_m(mask, grid)
    out = {"Y_TRACE_100M": mask.astype(np.float32)}
    for b in cfg["labels"]["buffers_m"]:
        out[f"Y_BUFFER_{int(b)}M"] = (dist <= b).astype(np.float32)
    out["Y_DISTANCE"] = dist
    if gdf is not None and len(gdf):
        c2, s2 = orientation_raster(gdf, grid)
        out["Y_ORIENTATION_C2"], out["Y_ORIENTATION_S2"] = c2, s2
        conf = [c for c in gdf.columns if c.lower() in ("confidence", "conf", "certainty")]
        if conf:
            from rasterio.features import rasterize

            vals = pd.to_numeric(gdf[conf[0]], errors="coerce")
            ok = vals.notna()
            out["Y_CONFIDENCE"] = rasterize([(g, float(v)) for g, v in zip(gdf.geometry[ok], vals[ok])],
                                            out_shape=grid.shape, transform=grid.affine, fill=np.nan,
                                            dtype="float32", all_touched=True)
    for k in out:
        out[k] = np.where(valid, out[k], np.nan).astype(np.float32)
    return out


def segment_overlap_labels(segments, mask: np.ndarray, grid, tol_m: float) -> pd.DataFrame:
    """Fraction of each candidate segment's length within ``tol_m`` of a mapped fault pixel (evaluation only)."""
    dist = distance_to_fault_m(mask, grid)
    from .grid import inside, world_to_pixel

    rows = []
    for i, g in enumerate(segments.geometry):
        n = max(int(g.length // (grid.res[0] / 2)), 2)
        pts = [g.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
        r, c = world_to_pixel([p.x for p in pts], [p.y for p in pts], grid)
        ok = inside(r, c, grid)
        frac = float((dist[r[ok], c[ok]] <= tol_m).mean()) if ok.any() else np.nan
        rows.append({"segment": i, "overlap_fraction": frac, "label_positive": bool(frac >= 0.5) if ok.any() else None})
    return pd.DataFrame(rows)
