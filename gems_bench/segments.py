"""Phase-1 fault segments on the native grid.

Phase-1 segments (``outputs/eda/datasets/known_fault_segments.{gpkg,parquet}``) were traced on
the 200 m EDA grid. To reuse their IDs and attributes at 100 m, the LineStrings are burned
onto the native grid and every *native* label pixel is assigned to the nearest burned segment
within ``max_dist_m`` (default 300 m). Native label pixels are the scoring units; segments are
only an attribute lookup. Unassigned label pixels keep seg_id = -1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .data import NativeData

GEOM_COLS = ["seg_id", "length_km", "azimuth", "sinuosity", "linearity", "curv_mean_deg_km",
             "n_junction_ends", "region", "x_mid", "y_mid", "component", "touches_edge"]


def load_segment_table(eda_dir: Path) -> pd.DataFrame:
    seg = pd.read_parquet(eda_dir / "datasets" / "known_fault_segments.parquet")
    seg = seg[[c for c in GEOM_COLS if c in seg.columns]].copy()
    arch_p = eda_dir / "datasets" / "segment_archetypes.parquet"
    if arch_p.exists():
        arch = pd.read_parquet(arch_p)
        keep = [c for c in arch.columns if c == "seg_id" or c == "primary_archetype" or c.startswith("ev_")
                or c.startswith("is_") or c.startswith("geom_")]
        seg = seg.merge(arch[keep], on="seg_id", how="left")
    if "primary_archetype" not in seg:
        seg["primary_archetype"] = np.nan
    seg["primary_archetype"] = seg.primary_archetype.fillna("unassigned")
    seg["length_bin"] = pd.cut(seg.length_km, [0, 1, 2, 5, 10, np.inf], labels=["<1", "1-2", "2-5", "5-10", ">10"])
    seg["strike_sector"] = pd.cut(seg.azimuth % 180, np.arange(0, 181, 30), include_lowest=True).astype(str)
    seg["sinuosity_bin"] = pd.cut(seg.sinuosity, [0, 1.05, 1.15, 1.3, np.inf], labels=["straight", "gentle", "curved", "very_curved"])
    seg["curv_bin"] = pd.qcut(seg.curv_mean_deg_km.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"]) \
        if seg.curv_mean_deg_km.notna().sum() >= 8 else "n/a"
    seg["junction"] = np.where(seg.get("n_junction_ends", 0) > 0, "touches_junction", "isolated")
    return seg


def segment_id_raster(data: NativeData, max_dist_m: float = 300.0) -> np.ndarray:
    """int32 (H, W): Phase-1 seg_id for each native label pixel, -1 elsewhere."""
    import geopandas as gpd
    from rasterio.features import rasterize

    gdf = gpd.read_file(data.paths.eda / "datasets" / "known_fault_segments.gpkg")
    if gdf.crs is not None and data.grid.crs is not None and gdf.crs != data.grid.crs:
        gdf = gdf.to_crs(data.grid.crs)
    shapes = [(g, int(s) + 1) for g, s in zip(gdf.geometry, gdf.seg_id) if g is not None and not g.is_empty]
    if not shapes:                                      # rasterize() raises an opaque ValueError on no shapes
        raise ValueError(f"{data.paths.eda / 'datasets' / 'known_fault_segments.gpkg'} has no non-empty geometry")
    burned = rasterize(shapes,
                       out_shape=data.shape, transform=data.grid.transform, fill=0, dtype="int32",
                       all_touched=True)
    dist, (ir, ic) = ndimage.distance_transform_edt(
        burned == 0, sampling=(data.grid.pixel_km_y * 1000, data.grid.pixel_km_x * 1000), return_indices=True)
    seg = burned[ir, ic] - 1
    out = np.where(data.labels & (dist <= max_dist_m), seg, -1).astype(np.int32)
    return out


def save_segment_raster(data: NativeData, seg_raster: np.ndarray, path: Path) -> Path:
    from gems_eda.io import write_geotiff

    return write_geotiff(path, {"seg_id": np.where(seg_raster >= 0, seg_raster, np.nan).astype(np.float32)}, data.grid)


def load_segment_raster(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        a = src.read(1)
    return np.where(np.isfinite(a), a, -1).astype(np.int32)
