"""DEM stage: mode dispatch, per-window processing with atomic resume, assembly onto the 100 m grid, and the 2 m
example windows for figures.

Per window ``07_cache/dem_windows/<MODE>/<id>.npz`` (statistics, coverage, support) is written first and
``<id>.json`` (fingerprint, sources, seconds) second; a window is done only when both exist and the fingerprint
matches. Outputs (on the 100 m grid):
* ``03_features/dem_multiscale/<MODE>/dem_<stat>.tif`` (one GeoTIFF per statistic, registry family ``dem``),
* ``02_geospatial/coverage/dem_coverage_<MODE>.tif``, ``dem_support_<MODE>.tif`` (one band per base feature),
* ``02_geospatial/dem_mosaic/dem_mean_elevation_100m_<MODE>.tif`` (area-weighted mean elevation of the mosaic).
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import dem_aggregation as agg
from . import dem_features, dem_mosaic
from .cache import atomic_save_npz, atomic_write_csv, atomic_write_geotiff, atomic_write_json, read_json
from .config import module_hash, stable_hash

log = logging.getLogger("atlas.dem")


def mode_fingerprint(cfg) -> str:
    return stable_hash([cfg.dem_mode, cfg.dem_params, module_hash("dem", "dem_mosaic", "dem_features",
                                                                  "dem_aggregation", "fft_operators")])


def _process(win: dict, srcs: list, grid_d: dict, cfg_d: dict, mode: str, out_dir: str, fp: str,
             keep_example: bool) -> dict:
    from .config import AtlasConfig
    from .geospatial import Grid

    t0 = time.time()
    cfg = AtlasConfig(cfg_d)
    p = cfg.dem_params
    grid = Grid.from_dict(grid_d)
    detailed = mode == "DETAILED_2M"
    if detailed:
        m = dem_mosaic.read_window_detailed(win, srcs, grid, p["halo_m"], p["work_res_m"])
    else:
        m = dem_mosaic.read_window_fast(win, srcs, p["halo_m"], p["work_res_m"])
    z, hp, f, res = m["z"], m["halo_px"], m["factor"], m["res"]
    nr, nc = win["n_rows"], win["n_cols"]
    sl = (slice(hp, hp + nr * f), slice(hp, hp + nc * f))
    nch, nb = len(agg.channels(p, detailed)), len(agg.bases(p, detailed))
    if not np.isfinite(z[sl]).any():
        stats = np.full((nch, nr, nc), np.nan, np.float32)
        cov = np.zeros((nr, nc), np.float32)
        sup = np.zeros((nb, nr, nc), np.float32)
        seam_frac = np.zeros((nr, nc), np.float32)
    else:
        seam, d_miss, d_seam = dem_mosaic.seams_and_distances(z, m["project"], res)
        F, G = dem_features.compute(z, res, p, detailed)
        stats, cov, sup = agg.summarise(F, G, z, d_miss, d_seam, sl, nr, nc, f, p, detailed,
                                        f"coherence_{p['structure_scales_m'][1]}m")
        seam_frac = seam[sl].reshape(nr, f, nc, f).mean(axis=(1, 3)).astype(np.float32)
        if keep_example:
            from affine import Affine

            a, _, c, _, e, ff = grid.transform[:6]
            tf = Affine(res, 0, c + a * win["col0"], 0, -res, ff + e * win["row0"])
            q = ["elevation", "slope", "edge", f"coherence_{p['structure_scales_m'][1]}m", "scarp",
                 "profile_curvature"]
            atomic_write_geotiff(Path(out_dir) / "examples" / f"{win['window_id']}.tif",
                                 np.stack([np.where(d_miss > G[k], F[k], np.nan)[sl] for k in q]),
                                 {"crs": grid.crs, "transform": tf}, q, tags={"mode": mode, "res_m": res})
        del F
    atomic_save_npz(Path(out_dir) / f"{win['window_id']}.npz", stats=stats, coverage=cov, support=sup,
                    seam_frac=seam_frac)
    rec = {"window_id": win["window_id"], "fingerprint": fp, "seconds": round(time.time() - t0, 1),
           "sources": m["tiles_used"], "coverage_mean": float(cov.mean()), "example": bool(keep_example)}
    atomic_write_json(Path(out_dir) / f"{win['window_id']}.json", rec)
    return rec


def run(paths, cfg, grid, footprint: np.ndarray, dem_inv: pd.DataFrame, coverage_hint: np.ndarray, writer,
        mode: str | None = None, window_ids: list[str] | None = None, progress=print) -> dict:
    """Process pending windows of ``mode`` (default cfg.dem_mode); assemble when all are done."""
    mode = mode or cfg.dem_mode
    cfg_d = cfg.as_dict()
    cfg_d["dem_mode"] = mode
    from .config import AtlasConfig

    mcfg = AtlasConfig(cfg_d)
    p = mcfg.dem_params
    detailed = mode == "DETAILED_2M"
    wdir = paths.d("07_cache", "dem_windows") / mode
    wdir.mkdir(parents=True, exist_ok=True)
    pref = dem_mosaic.preference(dem_inv)
    if pref.empty:
        raise RuntimeError("no VALID DEM tile - the DEM stage has no input (see 01_inventory/dem_inventory.csv)")
    atomic_write_csv(paths.d("02_geospatial", "dem_mosaic") / "source_preference.csv",
                     pref[["priority", "file", "project", "project_id", "year", "status", "vertical_unit",
                           "vertical_unit_source", "res_x", "crs_epsg"]])
    if detailed:
        srcs = pref.to_dict("records")
    else:
        tdir = paths.d("07_cache", "dem_windows") / "FAST_10M" / "tiles"
        srcs = []
        for i, rec in enumerate(pref.to_dict("records"), start=1):
            srcs.append(dem_mosaic.build_tile_cache(rec, grid, tdir))
            if i % 25 == 0:
                progress(f"[dem] 10 m tile caches {i}/{len(pref)}")
        atomic_write_csv(paths.d("02_geospatial", "dem_mosaic") / "tile_placements_10m.csv", pd.DataFrame(srcs))
    plan = dem_mosaic.window_plan(grid, footprint, coverage_hint, int(p["window_cells"]))
    if window_ids is not None:
        plan = plan[plan.window_id.isin(window_ids)]
    atomic_write_csv(paths.d("02_geospatial", "dem_mosaic") / f"window_plan_{mode}.csv", plan)
    fp_mode = mode_fingerprint(mcfg)
    names = {s["file"]: s for s in srcs}
    todo, done = [], []
    for w in plan.to_dict("records"):
        wfp = stable_hash([fp_mode, w["window_id"], sorted(names)])
        js = read_json(wdir / f"{w['window_id']}.json")
        need_example = window_ids is not None and not (wdir / "examples" / f"{w['window_id']}.tif").exists()
        if js and js.get("fingerprint") == wfp and (wdir / f"{w['window_id']}.npz").exists() and not need_example:
            done.append(js)
        else:
            todo.append((w, wfp))
    n_ex = len(todo) if window_ids is not None else (int(cfg["dem"]["detailed_example_windows"]) if detailed else 0)
    progress(f"[dem {mode}] windows: {len(plan)} planned, {len(done)} done, {len(todo)} to process")
    workers = int(p["workers"]) or min(4, os.cpu_count() or 1)
    budget = float(p["budget_minutes"]) * 60
    t0, stopped = time.time(), False
    args = [(w, srcs, grid.as_dict(), cfg_d, mode, str(wdir), wfp, i < n_ex) for i, (w, wfp) in enumerate(todo)]
    if args:
        if workers <= 1:
            for a in args:
                if budget and time.time() - t0 > budget:
                    stopped = True
                    break
                done.append(_process(*a))
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork") if hasattr(os, "fork") else None
            it = iter(args)
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                pend = {ex.submit(_process, *a) for a in [next(it) for _ in range(min(2 * workers, len(args)))]}
                while pend:
                    fut = next(as_completed(pend))
                    pend.remove(fut)
                    done.append(fut.result())
                    if len(done) % 25 == 0:
                        progress(f"[dem {mode}] {len(done)}/{len(plan)} windows ({(time.time() - t0) / 60:.1f} min)")
                    if budget and time.time() - t0 > budget:
                        stopped = True
                        continue
                    a = next(it, None)
                    if a is not None:
                        pend.add(ex.submit(_process, *a))
                stopped = stopped or next(it, None) is not None
    if stopped:
        return {"status": "INCOMPLETE", "mode": mode, "windows_done": len(done), "windows_left": len(plan) - len(done)}
    if window_ids is not None:
        return {"status": "EXAMPLES_COMPLETE", "mode": mode, "windows": [d["window_id"] for d in done]}
    return assemble(paths, mcfg, grid, footprint, plan, wdir, writer, mode, fp_mode, done)


def assemble(paths, cfg, grid, footprint, plan, wdir: Path, writer, mode: str, fp_mode: str, done: list) -> dict:
    p = cfg.dem_params
    detailed = mode == "DETAILED_2M"
    ch = agg.channels(p, detailed)
    bs = agg.bases(p, detailed)
    H, W = grid.height, grid.width
    stats = np.full((len(ch), H, W), np.nan, np.float32)
    cov = np.zeros((H, W), np.float32)
    sup = np.zeros((len(bs), H, W), np.float32)
    seam = np.zeros((H, W), np.float32)
    for w in plan.to_dict("records"):
        with np.load(wdir / f"{w['window_id']}.npz") as z:
            sl = (slice(w["row0"], w["row0"] + w["n_rows"]), slice(w["col0"], w["col0"] + w["n_cols"]))
            stats[(slice(None),) + sl] = z["stats"]
            cov[sl] = z["coverage"]
            sup[(slice(None),) + sl] = z["support"]
            seam[sl] = z["seam_frac"]
    cov[~footprint] = 0
    covered = footprint & (cov > 0)
    meaning = {"edge": "directional edge energy", "slope": "slope (degree)", "scarp": "height step across strike (m)",
               "relief": "local relief (m)", "roughness": "terrain roughness (m)", "tpi": "topographic position (m)"}
    from .feature_registry import AXIAL

    for i, (name, base, stat) in enumerate(ch):
        writer.put(f"dem_{name}", stats[i], family="dem", parent=f"usgs_1m_dem:{mode}", unit=_unit(base),
                   scale_m=p["work_res_m"], operator=f"{base} at {p['work_res_m']:g} m work resolution",
                   aggregation=f"{stat} over supporting work pixels of each 100 m cell (exact area weights)",
                   nodata="NaN where support < min_cell_support or no DEM", sub=mode, mask=covered,
                   orientation=AXIAL if name in ("orient_c2", "orient_s2") else "",
                   meaning=next((v for k, v in meaning.items() if base.startswith(k)), base))
    cdir = paths.d("02_geospatial", "coverage")
    outs = [atomic_write_geotiff(cdir / f"dem_coverage_{mode}.tif", cov, grid.profile(), ["dem_coverage"]),
            atomic_write_geotiff(cdir / f"dem_support_{mode}.tif", sup, grid.profile(), [f"support_{b}" for b in bs]),
            atomic_write_geotiff(cdir / f"dem_seam_fraction_{mode}.tif", seam, grid.profile(), ["seam_fraction"]),
            atomic_write_geotiff(paths.d("02_geospatial", "dem_mosaic") / f"dem_mean_elevation_100m_{mode}.tif",
                                 stats[[n for n, _, _ in ch].index("elevation_mean")], grid.profile(),
                                 ["mean_elevation_m"])]
    log_df = pd.DataFrame([{k: v for k, v in d.items() if k != "sources"} | {"n_sources": len(d.get("sources", []))}
                           for d in done])
    outs.append(atomic_write_csv(paths.d("07_cache", "aggregations") / f"dem_window_log_{mode}.csv", log_df))
    st = {"mode": mode, "windows": int(len(plan)), "covered_cells": int(covered.sum()),
          "covered_share_of_footprint": float(covered.sum() / max(footprint.sum(), 1)),
          "mean_coverage_covered": float(cov[covered].mean()) if covered.any() else 0.0,
          "seam_cells": int((seam > 0).sum()), "fingerprint": fp_mode,
          "mean_seconds_per_window": float(log_df.seconds.mean()) if len(log_df) else 0.0}
    atomic_write_json(paths.d("07_cache", "aggregations") / f"dem_summary_{mode}.json", st)
    return {"status": "COMPLETE", **st, "outputs": [str(o) for o in outs]}


def _unit(base: str) -> str:
    if base in ("slope",):
        return "degree"
    if base in ("scarp", "relief", "roughness", "elevation") or base.startswith("tpi"):
        return "m"
    if base in ("profile_curvature", "plan_curvature", "ridge", "valley", "crest", "toe"):
        return "1/m"
    if base == "edge":
        return "m/m"
    return "1"


def pick_example_windows(paths, mode: str, n: int) -> list[str]:
    """Deterministic, label-free: the windows of ``mode`` with the highest mean coverage x edge strength."""
    plan_p = paths.d("02_geospatial", "dem_mosaic") / f"window_plan_{mode}.csv"
    if not plan_p.exists():
        return []
    plan = pd.read_csv(plan_p)
    wdir = paths.d("07_cache", "dem_windows") / mode
    scores = []
    for w in plan.window_id:
        npz = wdir / f"{w}.npz"
        if npz.exists():
            with np.load(npz) as z:
                e = z["stats"][1]
                scores.append((float(np.nanmean(e)) * float(z["coverage"].mean()) if np.isfinite(e).any() else 0.0, w))
    return [w for _, w in sorted(scores, reverse=True)[:n]]
