"""E6 production data staging: cached 10m DEM parts -> aligned 100m channels.

No new download, no native 1m/2m extraction, no tests and no mirrors of raw DEM.
Intermediate 100m tile rasters and final mosaic are saved to an immutable,
fingerprinted cache and reused across interrupted Colab runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def _parts(paths, max_tiles):
    import rasterio
    found = {}
    # The existing G07 pilot already saved the individual 10m DEM parts.
    for root in (Path(paths.root), Path(paths.drive) if paths.drive else Path(paths.root)):
        folder = root / "outputs/g07_multiscale_fault_discovery"
        for p in sorted(folder.glob("g07-*/dem_features/coarse_parts/*/*.tif")):
            tile = p.stem.rsplit("_", 1)[0]
            if tile in found:
                continue
            try:
                with rasterio.open(p) as ds:
                    if ds.count != 1 or not ds.crs or not 5 <= abs(ds.res[0]) <= 25:
                        continue
                    if not (ds.width >= 100 and ds.height >= 100):
                        continue
                    found[tile] = p
            except (OSError, ValueError):
                continue
    return list(sorted(found.items()))[:max_tiles]


def _grid(ctx):
    from ..g06.contracts import GridSpec
    t = ctx.data.profile["transform"]
    return GridSpec(ctx.data.shape[0], ctx.data.shape[1],
                    str(ctx.data.profile["crs"]), tuple(float(v) for v in t[:6]),
                    abs(float(t.a)))


def stage_cached_dem(exp, ctx, *, max_tiles=32) -> dict:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from ..g07.fast_coarse import FEATURES, extract_fast_features
    from ..g07.dem_io import analysis_window, mosaic_windows, write_window

    if not 1 <= int(max_tiles) <= 100:
        raise ValueError("E6 DEM cache tile limit outside 1..100")
    parts = _parts(exp.paths, int(max_tiles))
    if not parts:
        raise FileNotFoundError(
            "E6 requires saved G07 10m coarse parts; no native 1m/2m "
            "processing will be started automatically. Expected "
            "GEMS_2/outputs/g07_multiscale_fault_discovery/g07-*/"
            "dem_features/coarse_parts/*/*.tif")
    source = [(tile, str(p), p.stat().st_size) for tile, p in parts]
    identity = hashlib.sha256(json.dumps({"sources": source, "operators": list(FEATURES),
                                         "version": "E6_DEM10_100_V2"}, sort_keys=True).encode()).hexdigest()[:16]
    local = Path(exp.paths.root) / "outputs/e6_pipeline/cache" / identity
    drive = (Path(exp.paths.drive) / "outputs/e6_pipeline/cache" / identity) if exp.paths.drive else None
    out = local / "dem_features_100m.tif"
    manifest_file = local / "manifest.json"
    local.mkdir(parents=True, exist_ok=True)
    if not out.exists() and drive and (drive / out.name).is_file():
        # Reuse Drive cache *in place*. No duplicate full DEM mosaic.
        out = drive / out.name
        manifest_file = drive / "manifest.json"

    grid = _grid(ctx)
    records = []
    if not out.is_file():
        per = local / "tiles"
        per.mkdir(parents=True, exist_ok=True)
        products = []
        for tile, path in parts:
            cached = per / f"{tile}.tif"
            if cached.is_file() and cached.with_suffix(".json").is_file():
                products.append(cached)
                records.append({"tile": tile, "status": "CACHED"})
                continue
            try:
                with rasterio.open(path) as ds:
                    window = analysis_window(tuple(ds.bounds), ds.crs, grid, pad=0)
                    if window is None:
                        records.append({"tile": tile, "status": "OUTSIDE"}); continue
                    z = ds.read(1, masked=True).filled(np.nan).astype(np.float32)
                    step = abs(float(ds.res[0]))
                    tr, crs = ds.transform, ds.crs
                valid = np.isfinite(z)
                if not valid.any():
                    records.append({"tile": tile, "status": "EMPTY"}); continue
                features = extract_fast_features(z, step)
                r0, r1, c0, c1 = window
                t = grid.affine()
                from rasterio.transform import Affine
                target_transform = Affine(t.a, t.b, t.c + t.a*c0,
                                          t.d, t.e, t.f + t.e*r0)
                shape = (r1-r0, c1-c0)
                coverage = np.zeros(shape, np.float32)
                reproject(valid.astype(np.float32), coverage,
                          src_transform=tr, src_crs=crs,
                          dst_transform=target_transform, dst_crs=grid.crs,
                          dst_nodata=0, resampling=Resampling.average)
                arr = {"dem_coverage": np.clip(coverage, 0, 1)}
                for key, band in features.items():
                    result = np.full(shape, np.nan, np.float32)
                    reproject(band, result, src_transform=tr, src_crs=crs,
                              src_nodata=np.nan, dst_transform=target_transform,
                              dst_crs=grid.crs, dst_nodata=np.nan,
                              resampling=Resampling.average)
                    arr[key] = np.where(coverage > 0, result, np.nan)
                write_window(cached, arr, ["dem_coverage", *FEATURES], grid, window,
                             {"tile":tile, "source":"cached_10m", "version":"E6_DEM10_100_V2"})
                products.append(cached)
                records.append({"tile": tile, "status": "PROCESSED"})
                print(f"[E6 DEM] {len(products)}/{len(parts)} cached-10m tiles -> 100m")
            except Exception as e:
                records.append({"tile": tile, "status": "FAILED", "error": str(e)[:300]})
                print(f"[E6 DEM] Tile {tile}: {type(e).__name__}: {e}")
        if not products:
            raise RuntimeError("No E6 coarse DEM tiles processed; no fabricated DEM values allowed")
        mosaic_windows(products, grid, ["dem_coverage", *FEATURES], out, coverage_paths=products)
        import pandas as pd
        pd.DataFrame(records).to_csv(local / "tile_log.csv", index=False)
        manifest = {"version": "E6_DEM10_100_V2", "fingerprint": identity,
                    "source_tiles": source, "processed_tiles": len(products),
                    "features": list(FEATURES), "output": str(out),
                    "raw_dem_mirrored": False, "no_original_1m_reads": True}
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if exp.drive:
            # Sync only derived compressed feature cache, never the originals.
            exp.drive.push([exp.paths.rel(local)])
    return {"path": str(out), "folder": str(out.parent), "fingerprint": identity,
            "n_source_tiles": len(parts), "records": records}


def register_e6_dem(ctx, dem_folder: Path, fingerprint: str, local_root: Path):
    """Attach the computed mosaic to the existing P5 FeatureStore without reopening context.

    The feature cache is fingerprint-scoped; a later feature-bank update cannot
    silently reuse old .npy bands from a different DEM mosaic.
    """
    from ..features.phase5_store import dem_specs
    store = ctx.store
    store.dem_dir = Path(dem_folder)
    store.p5_dir = Path(local_root) / "outputs/e6_pipeline/features" / fingerprint
    store.p5_dir.mkdir(parents=True, exist_ok=True)
    new = dem_specs(store.dem_dir)
    if not new or "dem:dem_coverage" not in [s.name for s in new]:
        raise RuntimeError("E6 DEM 100m raster lacks a valid coverage band")
    for spec in new:
        store.specs[spec.name] = spec
        if spec.name not in store.p5_names:
            store.p5_names.append(spec.name)
        store._p5_set.add(spec.name)
    return [s.name for s in new]
