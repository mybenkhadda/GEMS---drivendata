"""Section 06: memory-bounded 1 m DEM morphology, streamed tile by tile and aggregated to the 100 m grid.

Per tile: read ``block_px`` x ``block_px`` work-resolution blocks (1 m pixels block-averaged to ``work_res_m``)
with a ``halo_px`` margin (boundless reads, nodata kept as NaN) -> features -> crop the halo -> aggregate onto the
native 100 m window with GDAL resampling: ``sum`` of (value x valid) and of valid (exact means across block / tile
/ datum boundaries), ``max`` and ``q3`` (upper quartile; a cell straddling two blocks keeps the larger value).
High quantiles, maxima, orientation concentration and linearity keep the sub-cell structure that a mean of
elevation would erase. Temporary arrays are released after every block; nothing holds a mosaic at 1 m.

Base features come from ``gems_bench.dem.morphology.block_features`` (slope, curvatures, microrelief, slope break,
structure-tensor strike / coherence, scarp mask / skeleton / length, persistence). M7 adds (all label-free):

    hillshade_md / hillshade_range   multidirectional hillshade (4 azimuths) and its directional range
    tpi_<s>m                         |topographic position| = |z - G_s(z)| at each scale
    tri                              terrain ruggedness (mean |dz| to the 8 neighbours)
    openness_pos / openness_neg      positive / negative openness (8 directions, radius ``openness_radius_m``)
    openness_diff / svf / rrim       (pos - neg) / 2 (red-relief differential), sky-view factor, slope x |diff|
    scarp_response                   max over scales of the scale-normalised curvature along the gradient, slope-gated
    dirdisc_max / dirdisc_aniso      max / anisotropy of directional second derivatives (8 directions)
    ridge_persist / valley_persist   number of scales with a strong ridge / valley Hessian response
    edge_consistency                 coherence-weighted agreement of the structure-tensor strike across scales
    relief_<s>m                      local relief (range) at the two coarser scales
    coherence_mid                    structure-tensor coherence at the middle scale (surface linearity)
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from ..dem import morphology as M

DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def extra_stats(mc: dict) -> dict[str, tuple[str, ...]]:
    sc = list(mc["scales_m"])
    st = {"hillshade_md": ("mean",), "hillshade_range": ("mean", "q3"), "tri": ("mean", "q3"),
          "openness_pos": ("mean",), "openness_neg": ("mean",), "openness_diff": ("mean", "q3"), "svf": ("mean",),
          "rrim": ("mean", "q3"), "scarp_response": ("mean", "q3", "max"), "dirdisc_max": ("mean", "q3", "max"),
          "dirdisc_aniso": ("mean",), "ridge_persist": ("mean",), "valley_persist": ("mean",),
          "edge_consistency": ("mean",), "coherence_mid": ("mean",)}
    for s in sc:
        st[f"tpi_{s:g}m"] = ("mean", "q3")
    for s in sc[1:]:
        st[f"relief_{s:g}m"] = ("mean", "q3")
    return st


def band_names(mc: dict) -> list[str]:
    return list(M.BANDS) + [f"{k}_{s}" for k, ss in extra_stats(mc).items() for s in ss]


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """a[r + dr, c + dc] with edge replication (the halo keeps edge effects outside the kept core)."""
    p = max(abs(dr), abs(dc))
    if p == 0:
        return a
    b = np.pad(a, p, mode="edge")
    h, w = a.shape
    return b[p + dr:p + dr + h, p + dc:p + dc + w]


def block_average(z: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return z
    h, w = z.shape
    H, W = int(np.ceil(h / f)) * f, int(np.ceil(w / f)) * f
    zp = np.full((H, W), np.nan, np.float32)
    zp[:h, :w] = z
    r = zp.reshape(H // f, f, W // f, f)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(r).sum((1, 3))
        s = np.nansum(r, (1, 3))
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype(np.float32)


def extra_features(z: np.ndarray, res: float, mc: dict) -> dict[str, np.ndarray]:
    valid = np.isfinite(z)
    zf = np.where(valid, z, np.nanmean(z) if valid.any() else 0.0).astype(np.float32)
    s = (lambda m: max(float(m) / res, 0.7))
    out = {}
    gy, gx = np.gradient(zf, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    alt = np.radians(40.0)
    hs = np.stack([np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(np.radians(a) - aspect),
                           0, 1) for a in (315.0, 45.0, 135.0, 225.0)])
    out["hillshade_md"], out["hillshade_range"] = hs.mean(0), hs.max(0) - hs.min(0)
    del hs
    sc = list(mc["scales_m"])
    for sm in sc:
        out[f"tpi_{sm:g}m"] = np.abs(zf - ndimage.gaussian_filter(zf, s(sm), mode="nearest"))
    out["tri"] = np.mean([np.abs(zf - _shift(zf, dr, dc)) for dr, dc in DIRS8], axis=0)
    R = max(int(round(float(mc["openness_radius_m"]) / res)), 1)
    steps = np.unique(np.round(np.geomspace(1, R, 8)).astype(int))
    pos, neg, svf = np.zeros_like(zf), np.zeros_like(zf), np.zeros_like(zf)
    for dr, dc in DIRS8:
        dn = np.hypot(dr, dc) * res
        pmax, pmin = np.full(zf.shape, -np.pi / 2, np.float32), np.full(zf.shape, np.pi / 2, np.float32)
        for k in steps:
            ang = np.arctan((_shift(zf, k * dr, k * dc) - zf) / (k * dn))
            pmax, pmin = np.maximum(pmax, ang), np.minimum(pmin, ang)
        pos += 90.0 - np.degrees(pmax)
        neg += 90.0 + np.degrees(pmin)
        svf += 1.0 - np.sin(np.maximum(pmax, 0.0))
    n = float(len(DIRS8))
    out["openness_pos"], out["openness_neg"], out["svf"] = pos / n, neg / n, svf / n
    out["openness_diff"] = (out["openness_pos"] - out["openness_neg"]) / 2.0
    out["rrim"] = (np.degrees(slope) / 90.0) * np.abs(out["openness_diff"])
    del pos, neg, svf
    sr = np.zeros_like(zf)
    thr = np.radians(float(mc["scarp_slope_deg"]) / 2.0)
    ridge_p, valley_p = np.zeros_like(zf), np.zeros_like(zf)
    strikes, cohs = [], []
    for i, sm in enumerate(sc):
        ss = s(sm)
        zx, zy = ndimage.gaussian_filter(zf, ss, order=(0, 1)), -ndimage.gaussian_filter(zf, ss, order=(1, 0))
        zxx, zyy = ndimage.gaussian_filter(zf, ss, order=(0, 2)), ndimage.gaussian_filter(zf, ss, order=(2, 0))
        zxy = -ndimage.gaussian_filter(zf, ss, order=(1, 1))
        g2 = zx ** 2 + zy ** 2 + 1e-12
        d2n = (zxx * zx ** 2 + 2 * zxy * zx * zy + zyy * zy ** 2) / g2
        sr = np.maximum(sr, np.abs(d2n) * ss ** 2 * (slope >= thr))
        if i == 0:
            d2 = np.stack([np.abs(zxx * np.cos(t) ** 2 + 2 * zxy * np.cos(t) * np.sin(t) + zyy * np.sin(t) ** 2)
                           for t in np.radians(np.arange(0, 180, 22.5))]) * ss ** 2
            out["dirdisc_max"] = d2.max(0)
            out["dirdisc_aniso"] = (d2.max(0) - d2.min(0)) / (d2.max(0) + d2.min(0) + 1e-12)
            del d2
        half = 0.5 * (zxx + zyy)
        root = np.sqrt((0.5 * (zxx - zyy)) ** 2 + zxy ** 2)
        l1, l2 = (half - root) * ss ** 2, (half + root) * ss ** 2
        t = np.nanquantile(np.abs(np.concatenate([l1[valid], l2[valid]])), 0.9) if valid.any() else np.inf
        ridge_p += (l1 < -t)
        valley_p += (l2 > t)
        st, coh, _ = M._structure(zf, ss, res)
        strikes.append(st)
        cohs.append(coh)
        if i == len(sc) // 2:
            out["coherence_mid"] = coh
        del zx, zy, zxx, zyy, zxy, g2, d2n, half, root, l1, l2
    out["scarp_response"], out["ridge_persist"], out["valley_persist"] = sr, ridge_p, valley_p
    num = sum(c * np.cos(np.radians(2 * (st - strikes[0]))) for st, c in zip(strikes, cohs))
    out["edge_consistency"] = num / np.maximum(sum(cohs), 1e-12)
    del strikes, cohs
    for sm in sc[1:]:
        w = int(round(float(sm) / res)) | 1
        out[f"relief_{sm:g}m"] = ndimage.maximum_filter(zf, w) - ndimage.minimum_filter(zf, w)
    for k in out:
        out[k] = np.where(valid, out[k], np.nan).astype(np.float32)
    return out


def _accumulate_ext(acc: dict, feats: dict, valid: np.ndarray, stats: dict, src_T, src_crs, dst_T, dst_crs, shape):
    from rasterio.warp import Resampling, reproject

    rs = {"max": Resampling.max, "q3": Resampling.q3}
    for k, a in feats.items():
        for st in stats.get(k, ()):
            name = f"{k}_{st}"
            if st == "mean":
                dst = np.zeros(shape, np.float64)
                reproject(np.where(valid, np.nan_to_num(a), 0).astype(np.float32), dst, src_transform=src_T,
                          src_crs=src_crs, dst_transform=dst_T, dst_crs=dst_crs, resampling=Resampling.sum)
                acc[name] = acc.get(name, 0) + dst
            else:
                dst = np.full(shape, np.nan, np.float32)
                reproject(np.where(valid, a, np.nan).astype(np.float32), dst, src_transform=src_T, src_crs=src_crs,
                          dst_transform=dst_T, dst_crs=dst_crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=rs[st])
                acc[name] = np.fmax(acc.get(name, np.full(shape, np.nan, np.float32)), dst)


def process_tile(path: Path, native_profile: dict, mc: dict, project_id: int = 0, vectors: bool = False) -> tuple[dict, dict]:
    """(100 m bands on the native window covering the tile incl. ``window``, timing / size statistics)."""
    import rasterio
    from rasterio.transform import Affine, rowcol
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform

    t0 = time.perf_counter()
    stats = extra_stats(mc)
    base_cfg = M.MorphConfig(scarp_slope_deg=float(mc["scarp_slope_deg"]), scarp_coherence=float(mc["scarp_coherence"]))
    with rasterio.open(path) as src:
        res1 = float(abs(src.res[0]))
        f = max(int(round(float(mc["work_res_m"]) / res1)), 1)
        res = res1 * f
        b = transform_bounds(src.crs, native_profile["crs"], *src.bounds)
        T = native_profile["transform"]
        r0, c0 = rowcol(T, b[0], b[3])
        r1, c1 = rowcol(T, b[2], b[1])
        r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
        r1, c1 = min(r1 + 2, native_profile["height"]), min(c1 + 2, native_profile["width"])
        shape = (r1 - r0, c1 - c0)
        dst_T = Affine(T.a, T.b, T.c + c0 * T.a, T.d, T.e, T.f + r0 * T.e)
        acc_b, acc_e = {}, {}
        B, h = int(mc["block_px"]) * f, int(mc["halo_px"]) * f
        block_s, vec = [], []
        nod = src.nodata if src.nodata is not None else np.nan
        for y in range(0, src.height, B):
            for x in range(0, src.width, B):
                tb = time.perf_counter()
                wy, wx = min(B, src.height - y), min(B, src.width - x)
                z = src.read(1, window=Window(x - h, y - h, wx + 2 * h, wy + 2 * h), boundless=True, fill_value=nod,
                             masked=True).astype(np.float32).filled(np.nan)
                if not np.isfinite(z).any():
                    continue
                zw = block_average(z, f)
                del z
                hw = h // f
                cy, cx = int(np.ceil(wy / f)), int(np.ceil(wx / f))
                fb = M.block_features(zw, res, base_cfg)
                fe = extra_features(zw, res, mc)
                fb = {k: a[hw:hw + cy, hw:hw + cx] for k, a in fb.items()}
                fe = {k: a[hw:hw + cy, hw:hw + cx] for k, a in fe.items()}
                sT = win_transform(Window(x, y, wx, wy), src.transform) * Affine.scale(f)
                M._accumulate(acc_b, fb, sT, src.crs, dst_T, native_profile["crs"], shape)
                _accumulate_ext(acc_e, fe, fb["valid"] > 0, stats, sT, src.crs, dst_T, native_profile["crs"], shape)
                if vectors and "scarp_skel" in fb:
                    vec += _scarp_vectors(np.nan_to_num(fb["scarp_skel"]) > 0, sT, res, float(mc.get("min_scarp_vector_m", 60.0)))
                del zw, fb, fe
                gc.collect()
                block_s.append(time.perf_counter() - tb)
        out = M.finalize(acc_b, res, project_id, cell_m=abs(T.a))
        n = acc_b.get("_count")
        nn = np.where(n > 0, n, np.nan) if n is not None else None
        for name, a in acc_e.items():
            if name.endswith("_mean"):
                out[name] = (a / nn).astype(np.float32) if nn is not None else None
            else:
                out[name] = np.where(n > 0, a, np.nan).astype(np.float32) if n is not None else a
        out["window"] = (r0, r1, c0, c1)
        info = {"seconds": round(time.perf_counter() - t0, 1), "n_blocks": len(block_s),
                "block_seconds_mean": float(np.mean(block_s)) if block_s else np.nan,
                "raw_bytes": int(Path(path).stat().st_size), "work_res_m": res, "src_crs": str(src.crs),
                "covered_cells": int(np.nansum(out.get("coverage", np.zeros(1)) >= 0.9))}
    if vectors:
        out["_vectors"] = vec
    return out, info


def _scarp_vectors(skel: np.ndarray, T, res: float, min_m: float) -> list:
    """Traced 1 m-scale scarp polylines (tile CRS) longer than ``min_m``."""
    from ..p3.lineaments import split_branches, trace

    lab, _, _ = split_branches(skel, max(int(min_m / (res * 1.12)), 2))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        loc = np.argwhere(lab[sl] == i) + np.array([sl[0].start, sl[1].start])
        p = trace(loc)
        xs, ys = T * (p[:, 1] + 0.5, p[:, 0] + 0.5)
        out.append(np.c_[xs, ys])
    return out


def save_tile(out: dict, native_profile: dict, path: Path, meta: dict, names: list[str]) -> Path:
    import rasterio
    from rasterio.transform import Affine

    from ..artifacts.atomic import atomic_save, atomic_write_json

    r0, r1, c0, c1 = out["window"]
    T = native_profile["transform"]
    prof = {"driver": "GTiff", "height": r1 - r0, "width": c1 - c0, "count": len(names), "dtype": "float32",
            "crs": native_profile["crs"], "transform": Affine(T.a, T.b, T.c + c0 * T.a, T.d, T.e, T.f + r0 * T.e),
            "nodata": np.nan, "compress": "deflate", "predictor": 3}

    def write(t):
        with rasterio.open(t, "w", **prof) as dst:
            for i, b in enumerate(names, 1):
                a = out.get(b)
                dst.write(np.full((r1 - r0, c1 - c0), np.nan, np.float32) if a is None else a.astype(np.float32), i)
                dst.set_band_description(i, b)

    atomic_save(path, write)
    atomic_write_json(Path(path).with_suffix(".json"), json.loads(json.dumps({**meta, "window": out["window"]},
                                                                             default=str)))
    return path


MAX_LIKE = ("_max", "_q3")


def mosaic(tile_paths: list[Path], native_profile: dict, out_path: Path, names: list[str]) -> Path:
    """Band-by-band merge of the tile windows onto the full grid (one full-grid band in memory at a time):
    coverage-weighted means, fmax for maxima / upper quartiles, best-coverage source project."""
    import rasterio

    from ..artifacts.atomic import atomic_save

    H, W = native_profile["height"], native_profile["width"]
    T = native_profile["transform"]
    wins = []
    for p in tile_paths:
        with rasterio.open(p) as src:
            r0 = int(round((src.transform.f - T.f) / T.e))
            c0 = int(round((src.transform.c - T.c) / T.a))
            d = list(src.descriptions)
            cov = np.nan_to_num(src.read(d.index("coverage") + 1))
        wins.append((p, r0, c0, d, cov))
    prof = {"driver": "GTiff", "height": H, "width": W, "count": len(names), "dtype": "float32",
            "crs": native_profile["crs"], "transform": T, "nodata": np.nan, "compress": "deflate", "predictor": 3,
            "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}

    def write(t):
        with rasterio.open(t, "w", **prof) as dst:
            for i, b in enumerate(names, 1):
                num = np.zeros((H, W), np.float32)
                cv = np.zeros((H, W), np.float32)
                mx = np.full((H, W), np.nan, np.float32)
                best = np.zeros((H, W), np.float32)
                for p, r0, c0, d, cov in wins:
                    if b not in d:
                        continue
                    with rasterio.open(p) as src:
                        a = src.read(d.index(b) + 1)
                    sl = (slice(r0, r0 + a.shape[0]), slice(c0, c0 + a.shape[1]))
                    if b == "coverage":
                        cv[sl] += cov
                    elif b == "n_tiles":
                        num[sl] += (cov > 0)
                    elif b == "source_project":
                        better = cov > best[sl]
                        mx[sl] = np.where(better, a, mx[sl])
                        best[sl] = np.maximum(best[sl], cov)
                    elif b.endswith(MAX_LIKE) or b in ("slope_max", "scarp_component_max_m"):
                        mx[sl] = np.fmax(mx[sl], a)
                    else:
                        num[sl] += np.nan_to_num(a) * cov
                        cv[sl] += cov
                if b == "coverage":
                    o = np.clip(cv, 0, 1)
                elif b == "n_tiles":
                    o = num
                elif b == "source_project" or b.endswith(MAX_LIKE) or b in ("slope_max", "scarp_component_max_m"):
                    o = mx
                else:
                    o = np.where(cv > 0, num / np.maximum(cv, 1e-9), np.nan)
                dst.write(o.astype(np.float32), i)
                dst.set_band_description(i, b)
                del num, cv, mx, best

    return atomic_save(out_path, write)


class DemMosaic:
    """Lazy band reader of the M7 DEM mosaic (one full-grid band at a time; small cache)."""

    def __init__(self, path: Path, max_cached: int = 4):
        import rasterio

        self.path = Path(path)
        with rasterio.open(self.path) as src:
            self.names = list(src.descriptions)
        self._cache: dict = {}
        self.max_cached = max_cached

    def has(self, name: str) -> bool:
        return name in self.names

    def band(self, name: str) -> np.ndarray:
        import rasterio

        if name not in self._cache:
            if len(self._cache) >= self.max_cached:
                self._cache.pop(next(iter(self._cache)))
            with rasterio.open(self.path) as src:
                self._cache[name] = src.read(self.names.index(name) + 1).astype(np.float32)
        return self._cache[name]

    def coverage(self, min_cov: float) -> np.ndarray:
        return np.nan_to_num(self.band("coverage")) >= float(min_cov)


def mosaic_path(m7) -> Path:
    return m7.ws.root / "dem" / "features" / "m7_dem_features_100m.tif"


def run(m7, pilot: pd.DataFrame) -> dict:
    from ..pipeline.m6 import fingerprint
    from .dem_pilot import tile_path

    d = m7.cfg["dem"]
    mc = d["morphology"]
    names = band_names(mc)
    prof = m7.ctx.data.profile
    projects = {p: i for i, p in enumerate(sorted(pilot.project_dir.dropna().astype(str).unique()), start=1)}
    tdir = m7.ws.root / "dem" / "features" / "tiles"
    fp = fingerprint("m7-dem-morph", mc, names)
    rows, done, vec_rows = [], [], []
    for r in pilot.to_dict("records"):
        src = tile_path(m7, r)
        dst = tdir / f"{r['tile_id']}.tif"
        meta_p = dst.with_suffix(".json")
        if dst.exists() and meta_p.exists() and json.loads(meta_p.read_text(encoding="utf-8")).get("fingerprint") == fp \
                and not m7.cfg.get("overwrite_existing"):
            done.append(dst)
            rows.append({"tile_id": r["tile_id"], "status": "reused", **json.loads(meta_p.read_text(encoding="utf-8"))
                         .get("timing", {})})
            continue
        if not src.exists():
            m7.stage_in([src.relative_to(m7.local).as_posix()])
        if not src.exists():
            rows.append({"tile_id": r["tile_id"], "status": "raw tile missing (download not approved / not done)"})
            continue
        out, info = process_tile(src, prof, mc, projects.get(str(r["project_dir"]), 0),
                                 vectors=bool(d.get("save_1m_scarp_vectors")))
        vecs = out.pop("_vectors", [])
        save_tile(out, prof, dst, {"tile_id": r["tile_id"], "project": r["project_dir"], "source": src.name,
                                   "reason": r.get("reason"), "fingerprint": fp, "config": mc, "timing": info}, names)
        info["output_bytes"] = int(dst.stat().st_size)
        rows.append({"tile_id": r["tile_id"], "status": "processed", **info})
        done.append(dst)
        if vecs:
            vec_rows.append((r["tile_id"], info["src_crs"], vecs))
        m7.push([dst.relative_to(m7.local).as_posix(), meta_p.relative_to(m7.local).as_posix()])
        del out
        gc.collect()
        print(f"[m7] s06 {r['tile_id']}: {info['seconds']:.0f}s")
    timing = pd.DataFrame(rows)
    mos = mosaic(done, prof, mosaic_path(m7), names) if done else None
    est = estimator(m7, timing)
    return {"timing": timing, "mosaic": mos, "estimate": est, "vectors": vec_rows, "bands": names}


def estimator(m7, timing: pd.DataFrame) -> dict:
    """Measured per-tile cost (this session) extrapolated to every intersecting tile of the inventory."""
    inv = m7.ws.load_table("inventory", "dem/inventory")
    proc = timing[timing.status == "processed"] if "status" in timing else timing.iloc[:0]
    if not len(proc):
        return {"status": "no tile processed in this session: no measured cost yet"}
    if len(inv) and "tile_id" in inv:
        sel = inv[inv["in_study_area"].fillna(False).astype(bool)] if "in_study_area" in inv else inv
        n_all = int(sel.tile_id.nunique())
    else:
        n_all = np.nan
    sec = float(proc.seconds.mean())
    raw = float(proc.raw_bytes.mean())
    outb = float(proc.get("output_bytes", pd.Series([0.0])).mean())
    return {"tiles_measured": int(len(proc)), "seconds_per_tile_mean": sec,
            "seconds_per_tile_p90": float(proc.seconds.quantile(0.9)), "raw_gib_per_tile": raw / 2**30,
            "n_intersecting_tiles": n_all, "projected_hours_all": sec * n_all / 3600.0 if np.isfinite(n_all) else np.nan,
            "projected_raw_gib_all": raw * n_all / 2**30 if np.isfinite(n_all) else np.nan,
            "projected_output_gib_all": outb * n_all / 2**30 if np.isfinite(n_all) else np.nan,
            "note": "single-session measurement; CPU type, Drive I/O and tile content vary"}


def save_vectors(m7, vec_rows: list) -> Path | None:
    """1 m scarp polylines (one GeoPackage layer per tile, in the tile's own CRS)."""
    if not vec_rows:
        return None
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        print("[m7] geopandas missing: 1 m scarp vectors skipped")
        return None
    p = m7.ws.root / "dem" / "candidates" / "scarps_1m.gpkg"
    for tile, crs, vecs in vec_rows:
        vecs = [v for v in vecs if len(v) > 1]
        if not vecs:
            continue
        g = gpd.GeoDataFrame({"tile_id": [tile] * len(vecs),
                              "length_m": [float(np.hypot(*np.diff(v, axis=0).T).sum()) for v in vecs]},
                             geometry=[LineString(v) for v in vecs], crs=crs)
        g.to_file(p, layer=f"scarps_{tile}", driver="GPKG")
    m7.ws._track(p)
    return p
