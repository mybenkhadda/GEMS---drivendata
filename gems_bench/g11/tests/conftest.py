"""Synthetic fixtures (no real data): 12 x 10 km EPSG:32611 grid at 100 m, a GeoDAWN-like 3-band raster,
a fault raster + vector, the official-like template and two overlapping 2 m DEM tiles with a 3 m scarp."""
from __future__ import annotations

import copy

import numpy as np
import pytest

X0, Y0, PX, H, W = 500000.0, 4000000.0, 100.0, 120, 100
CRS = "EPSG:32611"
FAULT = ((501000.0, 3998000.0), (509000.0, 3990000.0))      # NW-SE line, azimuth 135 deg


def _write(path, arr, transform, descriptions=None, nodata=None, crs=CRS):
    import rasterio

    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                       dtype=a.dtype.name, crs=crs, transform=transform, nodata=nodata) as ds:
        ds.write(a)
        for i, d in enumerate(descriptions or [], 1):
            ds.set_band_description(i, d)
    return path


def fault_distance_grid():
    """Signed distance (m) of every master-grid pixel centre to the FAULT line (positive = NE side)."""
    rr, cc = np.indices((H, W))
    x, y = X0 + PX * (cc + 0.5), Y0 - PX * (rr + 0.5)
    (x0, y0), (x1, y1) = FAULT
    t = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)
    n = np.array([t[1], -t[0]])
    return (x - x0) * n[0] + (y - y0) * n[1]


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    from rasterio.transform import from_origin

    root = tmp_path_factory.mktemp("g11data")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    T = from_origin(X0, Y0, PX, PX)
    rng = np.random.default_rng(0)
    d = fault_distance_grid()
    rtp = (50 * np.tanh(d / 400.0) + rng.normal(0, 1, (H, W))).astype(np.float32)
    grav = (5 * np.tanh(d / 800.0) + rng.normal(0, 0.1, (H, W))).astype(np.float32)
    elev = (1000 + 0.01 * (np.arange(W)[None] * PX) + 3 * (d > 0) + rng.normal(0, 0.2, (H, W))).astype(np.float32)
    stack = np.stack([rtp, grav, elev])
    stack[:, :, :3] = np.nan                                                  # invalid western margin
    _write(raw / "gems-geodawn-numerical-features.tif", stack, T, ["rtp", "iso_grav_anom", "det_elev"], nodata=np.nan)
    fault = (np.abs(d) <= 75).astype(np.uint8)
    _write(raw / "existing_faults.tif", fault, T)
    tpl = np.where(np.isfinite(stack[0]), 0.0, np.nan).astype(np.float32)
    _write(raw / "example_submission.tif", tpl, T, nodata=np.nan)
    import geopandas as gpd
    from shapely.geometry import LineString

    gpd.GeoDataFrame({"name": ["F1", "F1dup"]}, geometry=[LineString(FAULT), LineString(FAULT)], crs=CRS).to_file(
        raw / "faults.gpkg", driver="GPKG")
    # DEM tiles: 2 m, 2 km x 2 km, overlapping by 1 km; tile B is newer and offset by +0.8 m (seam)
    dem = root / "data" / "dem_1m"
    dem.mkdir(parents=True)
    tiles = []
    for name, (tx, ty), off in (("USGS_1M_11_x50y399_NV_A_2016", (504000.0, 3995000.0), 0.0),
                                ("USGS_1M_11_x51y399_NV_B_2020", (505000.0, 3995000.0), 0.8)):
        n = 1000
        Tt = from_origin(tx, ty, 2.0, 2.0)
        cc, rr = np.meshgrid(np.arange(n), np.arange(n))
        x, y = tx + 2.0 * (cc + 0.5), ty - 2.0 * (rr + 0.5)
        (x0, y0), (x1, y1) = FAULT
        t = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)
        sd = (x - x0) * t[1] - (y - y0) * t[0]
        z = (1000 + 0.01 * (x - X0) + 3.0 * (sd > 0) + off).astype(np.float32)
        _write(dem / f"{name}.tif", z, Tt, nodata=-9999.0)
        tiles.append(dem / f"{name}.tif")
    from gems_bench.g11.config import load_config

    cfg = load_config(overrides={"data_root": str(root), "output_root": str(root / "out"),
                                 "geophysical_grid": {"expected_shape": [H, W], "expected_band_count": 3},
                                 "source_artifacts": {"optional_patterns": {"g07_arrays": ["outputs/**/*foldmasked*.npy"]}},
                                 "dem": {"native_res_range_m": [0.25, 3.0], "aggregate_chunk_cells": 30,
                                         "context_scales_m": [25, 100], "max_window_mpix": 64.0},
                                 "spatial_splits": {"block_km": 2.0, "autocorrelation_range_m": 300.0, "embargo_m": 300.0},
                                 "feature_engineering": {"geophysics": {"scales_m": [200, 400, 800]}}})
    return {"root": root, "raw": raw, "dem_dir": dem, "tiles": tiles, "cfg": cfg, "signed_d": d}


@pytest.fixture()
def cfg(synth):
    return copy.deepcopy(synth["cfg"])


@pytest.fixture(scope="session")
def grid(synth):
    from gems_bench.g11.grid import load_master_grid

    return load_master_grid(synth["raw"] / "gems-geodawn-numerical-features.tif")
