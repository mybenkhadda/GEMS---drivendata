"""G07 FAST COARSE research track: cached 10 m DEM -> small 100 m feature stack.

This code never opens/downloads the native 1 m DEM, runs no neural models,
and uses no fault labels for tile selection or feature construction.
The DoG measure is a bandpass proxy, not a continuous wavelet transform.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import persistence as PS
from .dem_io import DemFeatureStore, analysis_window, mosaic_windows, write_window

FEATURES = (
    "dem_fast_slope_s30",
    "dem_fast_scarp_s60",
    "dem_fast_curv_s60",
    "dem_fast_rough_s60",
    "dem_fast_dircoh_s60",
    "dem_fast_hs_range",
    "dem_fast_dog_s60",
)
DIRECTIONS = {
    "dem_fast_slope_s30": "high",
    "dem_fast_scarp_s60": "high",
    "dem_fast_curv_s60": "high",
    "dem_fast_rough_s60": "high",
    "dem_fast_dircoh_s60": "high",
    "dem_fast_hs_range": "high",
    "dem_fast_dog_s60": "high",
}


def _smooth(a: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    """NaN-normalized Gaussian blur: nodata never becomes an artificial edge."""
    v = valid.astype(np.float32)
    numerator = ndimage.gaussian_filter(np.where(valid, a, 0.0).astype(np.float32),
                                        float(sigma_px), mode="constant", cval=0.0)
    denominator = ndimage.gaussian_filter(v, float(sigma_px), mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator >= 0.70, numerator / np.maximum(denominator, 1e-6), np.nan)
    return result.astype(np.float32)


def extract_fast_features(z: np.ndarray, resolution_m: float = 10.0) -> dict[str, np.ndarray]:
    """Seven compact geological proxies, derived exclusively from elevation.

    Smoothing scales are 30/60/150 m; no 1-10 m microfault inference claimed.
    All band edge filters use a minimum-support guard against artificial seams.
    """
    if not 5.0 <= resolution_m <= 25.0:
        raise ValueError("Fast extractor expects a cached 5–25 m coarse DEM, not raw 1 m tiles")
    z = np.asarray(z, dtype=np.float32)
    valid = np.isfinite(z)
    if not valid.any():
        return {k: np.full(z.shape, np.nan, np.float32) for k in FEATURES}

    p30 = max(30.0 / resolution_m, 0.8)
    p60 = max(60.0 / resolution_m, 0.8)
    p150 = max(150.0 / resolution_m, 0.8)
    a30 = _smooth(z, valid, p30)
    a60 = _smooth(z, valid, p60)
    a150 = _smooth(z, valid, p150)
    gy, gx = np.gradient(a30, resolution_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    gy60, gx60 = np.gradient(a60, resolution_m)
    gyy, _ = np.gradient(gy60, resolution_m)
    _, gxx = np.gradient(gx60, resolution_m)
    curvature = np.abs(gxx + gyy).astype(np.float32)

    # A terrain-slope break, not a classifier or a validated tectonic fault.
    gy150, gx150 = np.gradient(a150, resolution_m)
    scarp = np.abs(np.hypot(gx, gy) - np.hypot(gx150, gy150)).astype(np.float32)

    # Multiscale difference-of-Gaussians: cheap wavelet-like bandpass PROXY.
    dog = np.abs(a30 - a150).astype(np.float32)
    rough = np.sqrt(np.maximum(_smooth(z * z, valid, p60) - a60 * a60, 0)).astype(np.float32)

    jxx = _smooth(gx60 * gx60, np.isfinite(gx60), p60)
    jyy = _smooth(gy60 * gy60, np.isfinite(gy60), p60)
    jxy = _smooth(gx60 * gy60, np.isfinite(gx60) & np.isfinite(gy60), p60)
    coherence = np.sqrt(np.maximum((jxx - jyy)**2 + 4*jxy*jxy, 0)) / np.maximum(jxx + jyy, 1e-8)
    coherence = np.clip(coherence, 0, 1).astype(np.float32)

    # Four azimuths reduce single-light-direction artefacts.
    sun_alt = np.deg2rad(45.)
    sl = np.arctan(np.hypot(gx, gy))
    aspects = np.arctan2(-gy, -gx)
    hs = []
    for az in (315., 45., 135., 225.):
        azrad = np.deg2rad(360. - az + 90.)
        h = (np.sin(sun_alt)*np.cos(sl) +
             np.cos(sun_alt)*np.sin(sl)*np.cos(azrad - aspects))
        hs.append(np.clip(h, 0, 1))
    hs_range = (np.max(hs, axis=0) - np.min(hs, axis=0)).astype(np.float32)

    out = {
        "dem_fast_slope_s30": slope,
        "dem_fast_scarp_s60": scarp,
        "dem_fast_curv_s60": curvature,
        "dem_fast_rough_s60": rough,
        "dem_fast_dircoh_s60": coherence,
        "dem_fast_hs_range": hs_range,
        "dem_fast_dog_s60": dog,
    }
    # 3 sigma safety margin protects missing-data/tile edges, but retains
    # interior evidence. The 150 m DoG requires the largest edge exclusion.
    dist = ndimage.distance_transform_edt(valid)
    px_guard = {
        "dem_fast_slope_s30": int(np.ceil(3*p30)) + 2,
        "dem_fast_scarp_s60": int(np.ceil(3*p150)) + 2,
        "dem_fast_curv_s60": int(np.ceil(3*p60)) + 2,
        "dem_fast_rough_s60": int(np.ceil(3*p60)) + 2,
        "dem_fast_dircoh_s60": int(np.ceil(4*p60)) + 2,
        "dem_fast_hs_range": int(np.ceil(3*p30)) + 2,
        "dem_fast_dog_s60": int(np.ceil(3*p150)) + 2,
    }
    for name, x in out.items():
        out[name] = np.where(valid & (dist > px_guard[name]) & np.isfinite(x), x, np.nan).astype(np.float32)
    return out


def cached_parts(P) -> pd.DataFrame:
    """List already staged, validated cache products only; no source downloads."""
    import rasterio
    coarse = P.root / "dem_features" / "coarse_parts"
    valid_ids = {str(s.tile_id) for s in P._usable_sources(P.dem_sources()[1])}
    rows = []
    if not coarse.exists():
        return pd.DataFrame(columns=["tile_id", "path", "resolution_m", "bytes"])
    for path in sorted(coarse.glob("*/*.tif")):
        tile_id = path.stem.rsplit("_", 1)[0]
        if tile_id not in valid_ids:
            continue
        try:
            with rasterio.open(path) as ds:
                if ds.count != 1 or not ds.crs or not 5 <= abs(ds.res[0]) <= 25:
                    continue
                # Check minimal real data support; avoids using only header geometry.
                sample = ds.read(1, window=((0, min(10, ds.height)), (0, min(10, ds.width))))
                if not np.isfinite(sample).any():
                    continue
                rows.append(dict(tile_id=tile_id, path=str(path), resolution_m=float(abs(ds.res[0])),
                                 bytes=path.stat().st_size, bounds=tuple(ds.bounds), crs=str(ds.crs)))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["tile_id", "path", "resolution_m", "bytes"])
    return pd.DataFrame(rows).sort_values(["tile_id", "bytes"], ascending=[True, False]).drop_duplicates("tile_id")


def build_fast_coarse(P, *, max_tiles: int = 10, draw_figures: bool = True) -> dict:
    """Build a compact DEM-derived 100m raster from existing 10m cached parts.

    Produces an *independent screening artifact*. Does not change normal
    DEM_FEATURES or feasibility gates and cannot train/promote models.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    if not 1 <= int(max_tiles) <= 30:
        raise ValueError("FAST_COARSE max_tiles must be 1..30; batch intentionally bounded")
    candidates = cached_parts(P)
    if candidates.empty:
        print("[G07 FAST] No cached 5–25m coarse parts are present locally. "
              "Run the original coarse DEM builder on a small pilot ONCE, or restore "
              "the saved dem_features/coarse_parts folder from Drive. Fine tier is not needed.")
        return {"status": "MISSING_COARSE_CACHE", "tiles": 0, "store": None}

    grid = P.grid()
    study = np.asarray(P.engine().valid(), bool)  # data footprint; NO held-out fault labels
    priorities = []
    for row in candidates.itertuples():
        win = analysis_window(row.bounds, row.crs, grid, pad=0)
        if win is None:
            continue
        r0, r1, c0, c1 = win
        pixels = int(study[r0:r1, c0:c1].sum())
        if pixels:
            priorities.append((pixels, row, win))
    priorities.sort(key=lambda x: (-x[0], str(x[1].tile_id)))
    priorities = priorities[:max_tiles]
    if not priorities:
        return {"status": "NO_STUDY_INTERSECTION", "tiles": 0, "store": None}

    fp = PS.stable_hash("G07_FAST_COARSE_V1", [(x[1].tile_id, int(x[1].bytes)) for x in priorities],
                        FEATURES, (30, 60, 150), n=12)
    target = P.root / "dem_features" / "fast_coarse" / fp / "dem_features_100m_fast.tif"
    products = target.parent / "tiles"
    products.mkdir(parents=True, exist_ok=True)
    logs, finished = [], []
    gridT = grid.affine()

    for j, (px, row, win) in enumerate(priorities, 1):
        tile_file = products / f"fast_{row.tile_id}_{row.bytes}.tif"
        if tile_file.exists() and tile_file.with_suffix(".json").exists():
            finished.append(tile_file)
            logs.append({"tile_id": row.tile_id, "status": "CACHED", "nominal_study_px": px})
            continue

        r0, r1, c0, c1 = win
        height, width = r1 - r0, c1 - c0
        dstT = Affine(gridT.a, gridT.b, gridT.c + c0 * gridT.a,
                      gridT.d, gridT.e, gridT.f + r0 * gridT.e)
        try:
            with rasterio.open(row.path) as ds:
                z = ds.read(1, masked=True).filled(np.nan).astype(np.float32)
                if ds.nodata is not None and np.isfinite(ds.nodata):
                    z[z == ds.nodata] = np.nan
                srcT, srcCRS = ds.transform, ds.crs
            valid = np.isfinite(z)
            if not valid.any():
                raise ValueError("DEM part contains no finite elevation values")
            bands = extract_fast_features(z, row.resolution_m)
            agg = {}
            coverage = np.zeros((height, width), np.float32)
            reproject(
                valid.astype(np.float32), coverage,
                src_transform=srcT, src_crs=srcCRS,
                dst_transform=dstT, dst_crs=grid.crs,
                resampling=Resampling.average, dst_nodata=0,
            )
            agg["dem_coverage"] = np.clip(coverage, 0, 1)
            for name in FEATURES:
                dst = np.full((height, width), np.nan, np.float32)
                reproject(
                    bands[name], dst,
                    src_transform=srcT, src_crs=srcCRS, src_nodata=np.nan,
                    dst_transform=dstT, dst_crs=grid.crs, dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
                agg[name] = np.where(coverage > 0, dst, np.nan)
            write_window(
                tile_file, agg, ["dem_coverage", *FEATURES], grid, win,
                {"track": "G07_FAST_COARSE_V1", "tile_id": row.tile_id,
                 "cached_source": row.path, "resolution_m": row.resolution_m,
                 "fingerprint": fp, "strictness": "SCREENING_ONLY"},
            )
            P.push_files(tile_file, tile_file.with_suffix(".json"))
            finished.append(tile_file)
            logs.append({"tile_id": row.tile_id, "status": "COMPLETE", "nominal_study_px": px})
            print(f"[G07 FAST] {j}/{len(priorities)} {row.tile_id}: 10m -> 100m saved")
        except Exception as e:
            logs.append({"tile_id": row.tile_id, "status": "FAILED", "reason": f"{type(e).__name__}: {e}"})
            print(f"[G07 FAST] {row.tile_id}: {type(e).__name__}: {e}")

    if not finished:
        return {"status": "NO_VALID_PRODUCTS", "tiles": 0, "store": None, "logs": pd.DataFrame(logs)}

    # Mosaic only small 100m tile feature rasters; never reprocess original 1m.
    if not target.exists():
        mosaic_windows(finished, grid, ["dem_coverage", *FEATURES], target,
                       coverage_paths=finished)
        P.push_files(target)
    store = DemFeatureStore(target, float(P.cfg["dem"].get("min_coverage", 0.5)), fp)
    log = pd.DataFrame(logs)
    P.ws.save_table(log, "dem_features/fast_coarse/tiles_processed.csv")
    manifest = {
        "status": "SCREENING_ONLY",
        "mode": "FAST_CACHED_COARSE_10M",
        "fingerprint": fp,
        "feature_names": list(FEATURES),
        "n_complete": len(finished),
        "n_candidates": len(priorities),
        "source_processing": "cached 10m coarse_parts only, no native 1m, no GPU",
        "selection": "largest eligible data footprint, label-free",
        "limitations": [
            "No guaranteed independent fault-system coverage",
            "DoG is not a continuous wavelet transform",
            "Missing tile-edge feature support is deliberately masked",
            "Not proof of leaderboard improvement",
        ],
        "output": str(target.relative_to(P.root)),
    }
    P.ws.save_json(manifest, "dem_features/fast_coarse/manifest.json")
    P.push(["dem_features/fast_coarse"])
    if draw_figures:
        _figures(P, store)
    return {"status": "SCREENING_ONLY", "tiles": len(finished),
            "store": store, "manifest": manifest, "logs": log}


def _figures(P, store):
    import matplotlib.pyplot as plt

    n = 8
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, name in zip(axs.flat, FEATURES):
        a = np.asarray(store.load([name])[0][::n, ::n], np.float32)
        im = ax.imshow(np.ma.masked_invalid(a), cmap="viridis", interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=.02)
    axs.flat[-1].axis("off")
    fig.suptitle("G07 FAST: cached 10m DEM structural proxies | SCREENING_ONLY")
    folder = P.figs("fast_coarse")
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(folder / f"fast_coarse_features.{ext}", dpi=150)
    plt.close(fig)
    P.push(["figures/fast_coarse"])
