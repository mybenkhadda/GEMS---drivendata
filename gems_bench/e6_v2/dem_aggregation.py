"""DEM stage: per-window mosaic -> 2 m features -> 100 m statistics + measured coverage, with atomic per-window
resume, then assembly of the full-grid ``features_100m`` / ``coverage_100m`` rasters.

Per window ``cache/dem/windows/<id>.npz`` (features (22, r, c), coverage (r, c)) is written first and
``<id>.json`` (fingerprint, tiles used, timings) second; a window is done only when both exist and the json
fingerprint matches. An interrupted session therefore loses at most the windows in flight.

dem_coverage of a 100 m cell = share of its 2 m pixels that hold DEM data farther than the gradient guard from
missing data and farther than ``seam_guard_m`` from a project seam. A statistic is NaN in a cell where fewer than
``min_cell_valid_frac`` of the pixels are usable for that statistic's own (larger) guard.
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import dem_features, dem_mosaic
from .cache import (atomic_save_npy, atomic_save_npz, atomic_write_csv, atomic_write_geotiff, atomic_write_json,
                    is_temporary, manifest_status, read_json, write_manifest)
from .config import module_hash, section_hash, stable_hash
from .contracts import DEM_CHANNELS
from .dem_sources import overlap_order
from .normalization import robust_stats
from .paths import E6V2Paths

log = logging.getLogger("e6v2.dem")


def _blocks(a: np.ndarray, usable: np.ndarray, sl, nr: int, nc: int, f: int) -> np.ndarray:
    x = np.where(usable, a, np.nan)[sl].astype(np.float32)
    return x.reshape(nr, f, nc, f).transpose(0, 2, 1, 3).reshape(nr, nc, f * f)


def summarise(feats: dict, guard: dict, d_miss, d_seam, z, sl, nr, nc, f, cfg) -> tuple[np.ndarray, np.ndarray]:
    d = cfg["dem"]
    finite = np.isfinite(z)
    base_ok = finite & (d_seam > d["seam_guard_m"])
    use = {k: base_ok & (d_miss > g) for k, g in guard.items()}
    cov_mask = base_ok & (d_miss > guard["slope"])
    cov = cov_mask[sl].reshape(nr, f, nc, f).mean(axis=(1, 3)).astype(np.float32)
    minf = float(d["min_cell_valid_frac"])
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        def B(name, values=None):
            return _blocks(feats[name] if values is None else values, use[name], sl, nr, nc, f)

        def gate(name, stat):
            frac = np.isfinite(B(name)).mean(-1)
            return np.where(frac >= minf, stat, np.nan).astype(np.float32)

        b = B("edge_energy")
        out["edge_energy_p50"] = gate("edge_energy", np.nanpercentile(b, 50, axis=-1))
        out["edge_energy_p90"] = gate("edge_energy", np.nanpercentile(b, 90, axis=-1))
        out["edge_energy_max"] = gate("edge_energy", np.nanmax(b, axis=-1))
        out["edge_anisotropy"] = gate("edge_anisotropy", np.nanmean(B("edge_anisotropy"), -1))
        s_small, s_large = d["structure_scales_m"][0], d["structure_scales_m"][-1]
        out["coherence_p90_5m"] = gate(f"coherence_{s_small}m", np.nanpercentile(B(f"coherence_{s_small}m"), 90, -1))
        out["coherence_p90_20m"] = gate(f"coherence_{s_large}m", np.nanpercentile(B(f"coherence_{s_large}m"), 90, -1))
        w = B(f"coherence_{s_large}m")
        c = B("strike_cos2")
        s = B("strike_sin2")
        ws = np.nansum(w, -1)
        mc = np.nansum(w * c, -1) / np.maximum(ws, 1e-12)
        ms = np.nansum(w * s, -1) / np.maximum(ws, 1e-12)
        ok_w = ws > 0
        out["orientation_cos2"] = gate("strike_cos2", np.where(ok_w, mc, np.nan))
        out["orientation_sin2"] = gate("strike_cos2", np.where(ok_w, ms, np.nan))
        out["orientation_R"] = gate("strike_cos2", np.where(ok_w, np.hypot(mc, ms), np.nan))
        b = B("slope")
        out["slope_p50"] = gate("slope", np.nanpercentile(b, 50, -1))
        out["slope_p90"] = gate("slope", np.nanpercentile(b, 90, -1))
        out["slope_std"] = gate("slope", np.nanstd(b, -1))
        out["curvature_min"] = gate("kmin", np.nanpercentile(B("kmin"), 10, -1))
        out["curvature_max"] = gate("kmax", np.nanpercentile(B("kmax"), 90, -1))
        r_small, r_large = d["tpi_radii_m"][0], d["tpi_radii_m"][-1]
        out["tpi_50m"] = gate(f"tpi_{r_small}m", np.nanpercentile(np.abs(B(f"tpi_{r_small}m")), 90, -1))
        out["tpi_200m"] = gate(f"tpi_{r_large}m", np.nanpercentile(np.abs(B(f"tpi_{r_large}m")), 90, -1))
        out["scarp_strength"] = gate("scarp", np.nanpercentile(B("scarp"), 90, -1))
        out["crest_response"] = gate("crest", np.nanpercentile(B("crest"), 90, -1))
        out["toe_response"] = gate("toe", np.nanpercentile(B("toe"), 90, -1))
        out["edge_fraction"] = gate("edge", np.nanmean(B("edge"), -1))
        b = B("roughness")
        out["roughness"] = gate("roughness", np.nanmean(b, -1))
        out["roughness_p90"] = gate("roughness", np.nanpercentile(b, 90, -1))
    if list(out) != list(DEM_CHANNELS):
        raise RuntimeError(f"DEM summary channels {list(out)} != schema {list(DEM_CHANNELS)}")
    return np.stack([out[k] for k in DEM_CHANNELS]), cov


def process_window(win: dict, tiles: list[dict], grid_dict: dict, crs_wkt: str, cfg: dict, dem_root: str,
                   fingerprint: str, keep_2m: bool) -> dict:
    t0 = time.time()
    root = Path(dem_root)
    m = dem_mosaic.read_window(win, tiles, grid_dict, crs_wkt, cfg)
    z, res, f, hp = m["z"], m["res"], m["factor"], m["halo_px"]
    nr, nc = win["n_rows"], win["n_cols"]
    sl = (slice(hp, hp + nr * f), slice(hp, hp + nc * f))
    if not np.isfinite(z[sl]).any():
        feats100 = np.full((len(DEM_CHANNELS), nr, nc), np.nan, np.float32)
        cov = np.zeros((nr, nc), np.float32)
    else:
        d_miss, d_seam = dem_mosaic.distance_fields(z, m["seam"], res)
        feats, guard = dem_features.compute(z, res, cfg)
        feats100, cov = summarise(feats, guard, d_miss, d_seam, z, sl, nr, nc, f, cfg)
        if keep_2m:
            from rasterio.crs import CRS
            from affine import Affine

            tf = m["transform"]
            core_tf = Affine(res, 0, tf.c + hp * res, 0, -res, tf.f - hp * res)
            prof = {"crs": CRS.from_wkt(crs_wkt), "transform": core_tf}
            atomic_write_geotiff(root / "dem_2m" / f"{win['window_id']}.tif", z[sl], prof, ["elevation_m"])
            qa = ["slope", "edge_energy", f"coherence_{cfg['dem']['structure_scales_m'][-1]}m",
                  f"tpi_{cfg['dem']['tpi_radii_m'][-1]}m", "scarp"]
            atomic_write_geotiff(root / "features_2m" / f"{win['window_id']}.tif",
                                 np.stack([np.where(d_miss > guard[q], feats[q], np.nan)[sl] for q in qa]), prof, qa)
        del feats
    wid = win["window_id"]
    atomic_save_npz(root / "windows" / f"{wid}.npz", features=feats100.astype(np.float32), coverage=cov)
    rec = {"window_id": wid, "fingerprint": fingerprint, "seconds": round(time.time() - t0, 1),
           "tiles_used": m["tiles_used"], "mean_coverage": float(cov.mean()), "kept_2m": bool(keep_2m)}
    atomic_write_json(root / "windows" / f"{wid}.json", rec)
    return rec


def _tile_checksums(inv: dict) -> dict:
    return {r["path"]: r.get("checksum") for r in inv["rows"] if r.get("type") == "dem_1m"}


def dem_code_fingerprint(cfg: dict) -> str:
    return stable_hash([section_hash(cfg, "dem"), module_hash("dem_sources", "dem_mosaic", "dem_features",
                                                              "dem_aggregation", "fft_features")])


def prepare_dem(paths: E6V2Paths, cfg: dict, inv: dict, base: dict, force: bool = False,
                assemble: bool = True) -> dict:
    """Process every pending window, then (``assemble``) build the 100 m rasters. Returns the manifest,
    {'status': 'windows_complete'} when ``assemble`` is False, or {'status': 'incomplete', ...} when the session
    budget ran out (re-run the cell: finished windows are kept)."""
    from .geophysics import base_fingerprint

    out_feat = paths.dem_dir("features_100m") / "dem_features_100m.npy"
    out_cov = paths.dem_dir("coverage_100m") / "dem_coverage_100m.npy"
    man = paths.dem_dir("manifests") / "dem_manifest.json"
    src_csv = paths.dem_dir("source_inventory") / "dem_sources.csv"
    if not src_csv.exists():
        from .cache import MissingArtifact

        raise MissingArtifact("cache/dem/source_inventory/dem_sources.csv missing - run sections 02-03")
    tiles = overlap_order(pd.read_csv(src_csv))
    cks = _tile_checksums(inv)
    code_fp = dem_code_fingerprint(cfg)
    tiles_fp = stable_hash(sorted((t, cks.get(t)) for t in tiles.path))
    stage_fp = stable_hash([code_fp, tiles_fp, base_fingerprint(cfg, inv)])
    if not force:
        st, why = manifest_status(man, stage_fp, paths.root)
        if st == "complete":
            return read_json(man)
        log.info("dem: %s (%s)", st, why)

    grid = base["grid"]
    gd = grid.as_dict()
    plan = dem_mosaic.plan_windows(gd, base["valid"], tiles, cfg)
    atomic_write_csv(paths.dem_dir("mosaic") / "window_plan.csv", plan)
    atomic_write_csv(paths.dem_dir("mosaic") / "overlap_order.csv",
                     tiles[["priority", "name", "project", "project_id", "year", "vertical_unit", "vertical_unit_source",
                            "crs_epsg", "res_x", "res_y"]])
    recs = tiles.to_dict("records")
    by_pri = {int(r["priority"]): r for r in recs}
    wdir = paths.dem_dir("windows")
    todo, done = [], []
    for w in plan.to_dict("records"):
        pri = [int(p) for p in str(w["tile_priorities"]).split("|") if p != ""]
        wt = [by_pri[p] for p in pri]
        wfp = stable_hash([code_fp, gd, w["window_id"], sorted((t["name"], cks.get(t["path"])) for t in wt)])
        js = read_json(wdir / f"{w['window_id']}.json")
        if js and js.get("fingerprint") == wfp and (wdir / f"{w['window_id']}.npz").exists():
            done.append(js)
        else:
            todo.append((w, wt, wfp))
    log.info("dem windows: %d planned, %d done, %d to process", len(plan), len(done), len(todo))
    n_qa = max(int(cfg["dem"]["qa_windows"]) - sum(1 for j in done if j.get("kept_2m")), 0)
    workers = int(cfg["dem"]["workers"]) or min(4, os.cpu_count() or 1)
    budget = float(cfg["dem"]["budget_minutes"]) * 60.0
    t0 = time.time()
    stopped = False
    if todo:
        args = [(w, wt, gd, grid.crs_wkt, cfg, str(paths.dem), wfp, i < n_qa) for i, (w, wt, wfp) in enumerate(todo)]
        if workers <= 1:
            for a in args:
                if budget and time.time() - t0 > budget:
                    stopped = True
                    break
                done.append(process_window(*a))
                _progress(len(done), len(plan), t0)
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork") if hasattr(os, "fork") else None
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                pending = set()
                it = iter(args)
                for _ in range(workers * 2):
                    a = next(it, None)
                    if a is not None:
                        pending.add(ex.submit(process_window, *a))
                while pending:
                    fut = next(as_completed(pending))
                    pending.remove(fut)
                    done.append(fut.result())          # a failing window raises here with its traceback
                    _progress(len(done), len(plan), t0)
                    if budget and time.time() - t0 > budget:
                        stopped = True
                        continue
                    a = next(it, None)
                    if a is not None:
                        pending.add(ex.submit(process_window, *a))
                stopped = stopped or next(it, None) is not None
    if stopped:
        n_left = len(plan) - len(done)
        log.warning("dem: session budget reached, %d windows left - re-run this cell to continue", n_left)
        return {"status": "incomplete", "windows_done": len(done), "windows_left": n_left}
    if not assemble:
        return {"status": "windows_complete", "windows": int(len(plan)),
                "mean_seconds_per_window": float(np.mean([j["seconds"] for j in done])) if done else 0.0}

    H, W = grid.height, grid.width
    feats = np.full((len(DEM_CHANNELS), H, W), np.nan, np.float32)
    cov = np.zeros((H, W), np.float32)
    for w in plan.to_dict("records"):
        p = wdir / f"{w['window_id']}.npz"
        if is_temporary(p):
            continue
        with np.load(p) as z:
            r0, c0, nr, nc = w["row0"], w["col0"], w["n_rows"], w["n_cols"]
            feats[:, r0:r0 + nr, c0:c0 + nc] = z["features"]
            cov[r0:r0 + nr, c0:c0 + nc] = z["coverage"]
    valid = base["valid"]
    feats[:, ~valid] = np.nan
    cov[~valid] = 0.0
    stats_mask = valid & (cov > 0)
    norm = robust_stats(feats, list(DEM_CHANNELS), stats_mask, cfg, seed=cfg["splits"]["seed"])
    win_df = pd.DataFrame([{k: v for k, v in j.items() if k != "tiles_used"} for j in done])
    outs = [atomic_save_npy(out_feat, feats), atomic_save_npy(out_cov, cov),
            atomic_write_json(paths.dem_dir("features_100m") / "normalization.json", norm),
            atomic_write_csv(paths.dem_dir("manifests") / "window_log.csv", win_df)]
    stats = {"windows": int(len(plan)), "covered_cells": int((cov > 0).sum()),
             "covered_share_of_footprint": float((cov[valid] > 0).mean()),
             "mean_coverage_covered": float(cov[cov > 0].mean()) if (cov > 0).any() else 0.0,
             "seconds_this_session": round(time.time() - t0, 1)}
    return write_manifest(man, stage="dem", fingerprint=stage_fp, outputs=outs, root=paths.root,
                          extra={"channels": list(DEM_CHANNELS), "stats": stats, "tiles_fingerprint": tiles_fp})


def _progress(n: int, total: int, t0: float):
    if n % 25 == 0 or n == total:
        el = time.time() - t0
        log.info("dem windows %d/%d  (%.1f min this session)", n, total, el / 60)
