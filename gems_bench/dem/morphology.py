"""Experiment F3: fine-scale fault morphology at 1 m, aggregated exactly to the native 100 m grid.

Processing never holds a mosaic: each tile is read in ``block_px`` windows with a ``halo_px`` margin (filters
up to ~32 m), features are computed at 1 m, the halo is cropped, and the block is aggregated onto the native
grid window by GDAL ``sum`` resampling of (value x valid, valid) - so 100 m cells straddling blocks, tiles or the
NAD83 / WGS84 datum difference get exact averages - and ``max`` resampling for extreme-value bands.

1 m features (all label-free)
------------------------------
slope (deg); profile / plan curvature (Gaussian derivatives, σ = 2 m); microrelief = z - G_10m(z);
slope break = |second derivative along the gradient| (σ = 3 m); structure-tensor strike and coherence (σ = 3 m);
scarp mask = slope >= ``scarp_slope_deg`` & coherence >= ``scarp_coherence`` & local relief in
[``scarp_relief_m``]; scarp skeleton and connected scarp length; persistence = number of scales (2, 8, 32 m)
with coherence >= 0.6.

100 m bands written (``BANDS``): coverage, slope_mean, slope_max, microrelief_rms, curv_profile_absmean,
curv_plan_absmean, slope_break_mean, scarp_density, scarp_length_m, scarp_component_max_m, orient_c2, orient_s2,
orient_resultant, orient_entropy, persistence_mean, source_project, n_tiles.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

MEAN_BANDS = ("slope_mean", "microrelief_ms", "curv_profile_absmean", "curv_plan_absmean", "slope_break_mean",
              "scarp_density", "scarp_skel", "orient_c2w", "orient_s2w", "orient_w", "persistence_mean")
MAX_BANDS = ("slope_max", "scarp_component_max_m")
N_ORIENT_BINS = 8
BANDS = ("coverage", "slope_mean", "slope_max", "microrelief_rms", "curv_profile_absmean", "curv_plan_absmean",
         "slope_break_mean", "scarp_density", "scarp_length_m", "scarp_component_max_m", "orient_c2", "orient_s2",
         "orient_resultant", "orient_entropy", "persistence_mean", "source_project", "n_tiles")


@dataclass(frozen=True)
class MorphConfig:
    block_px: int = 2000
    halo_px: int = 100
    curv_sigma_m: float = 2.0
    relief_sigma_m: float = 10.0
    break_sigma_m: float = 3.0
    orient_sigma_m: float = 3.0
    persistence_sigmas_m: tuple = (2.0, 8.0, 32.0)
    scarp_slope_deg: float = 15.0
    scarp_coherence: float = 0.6
    scarp_relief_m: tuple = (0.5, 20.0)
    relief_window_m: float = 15.0
    min_scarp_len_m: float = 20.0


def _deriv(z, s, order):
    return ndimage.gaussian_filter(z, s, order=order, mode="nearest")


def _structure(z, sigma_px, res):
    gr, gc = _deriv(z, sigma_px, (1, 0)) / res, _deriv(z, sigma_px, (0, 1)) / res
    ge, gn = gc, -gr
    si = 3 * sigma_px
    jee, jnn, jen = (ndimage.gaussian_filter(a, si, mode="nearest") for a in (ge * ge, gn * gn, ge * gn))
    phi = 0.5 * np.arctan2(2 * jen, jee - jnn)
    strike = (90.0 - np.degrees(phi) + 90.0) % 180
    energy = jee + jnn
    coh = np.sqrt((jee - jnn) ** 2 + 4 * jen ** 2) / np.maximum(energy, 1e-12)
    return strike, coh, energy


def block_features(z: np.ndarray, res: float, cfg: MorphConfig) -> dict[str, np.ndarray]:
    """1 m feature rasters for one (halo-padded) block; ``z`` float32 with NaN nodata."""
    valid = np.isfinite(z)
    zf = np.where(valid, z, np.nanmean(z) if valid.any() else 0.0).astype(np.float32)
    s = lambda m: max(m / res, 0.7)
    zx, zy = _deriv(zf, s(1.0), (0, 1)) / res, -_deriv(zf, s(1.0), (1, 0)) / res
    slope = np.degrees(np.arctan(np.hypot(zx, zy)))
    sc = s(cfg.curv_sigma_m)
    fx, fy = _deriv(zf, sc, (0, 1)) / res, -_deriv(zf, sc, (1, 0)) / res
    fxx, fyy = _deriv(zf, sc, (0, 2)) / res ** 2, _deriv(zf, sc, (2, 0)) / res ** 2
    fxy = -_deriv(zf, sc, (1, 1)) / res ** 2
    p2 = fx ** 2 + fy ** 2 + 1e-12
    curv_prof = -(fxx * fx ** 2 + 2 * fxy * fx * fy + fyy * fy ** 2) / (p2 * (1 + p2) ** 1.5)
    curv_plan = -(fxx * fy ** 2 - 2 * fxy * fx * fy + fyy * fx ** 2) / (p2 ** 1.5)
    relief = zf - ndimage.gaussian_filter(zf, s(cfg.relief_sigma_m), mode="nearest")
    sb = s(cfg.break_sigma_m)
    bx, by = _deriv(zf, sb, (0, 1)), -_deriv(zf, sb, (1, 0))
    bxx, byy, bxy = _deriv(zf, sb, (0, 2)), _deriv(zf, sb, (2, 0)), -_deriv(zf, sb, (1, 1))
    g2 = bx ** 2 + by ** 2 + 1e-12
    slope_break = np.abs((bxx * bx ** 2 + 2 * bxy * bx * by + byy * by ** 2) / g2) / res ** 2
    strike, coh, energy = _structure(zf, s(cfg.orient_sigma_m), res)
    pers = np.zeros_like(zf)
    for sg in cfg.persistence_sigmas_m:
        pers += (_structure(zf, s(sg), res)[1] >= 0.6)
    w = int(round(cfg.relief_window_m / res)) | 1
    loc_range = ndimage.maximum_filter(zf, w) - ndimage.minimum_filter(zf, w)
    lo, hi = cfg.scarp_relief_m
    scarp = valid & (slope >= cfg.scarp_slope_deg) & (coh >= cfg.scarp_coherence) & (loc_range >= lo) & (loc_range <= hi)
    from skimage.morphology import skeletonize

    skel = skeletonize(scarp)
    lab, n = ndimage.label(skel, structure=np.ones((3, 3), bool))
    comp_m = np.zeros_like(zf)
    if n:
        length = np.bincount(lab.ravel()) * res * 1.12
        length[0] = 0
        keep = length >= cfg.min_scarp_len_m
        skel &= keep[lab]
        comp_m = np.where(skel, length[lab], 0).astype(np.float32)
    ow = (coh * np.sqrt(energy)).astype(np.float32)
    t2 = np.radians(2 * strike)
    out = {"valid": valid.astype(np.float32), "slope_mean": slope, "slope_max": slope, "microrelief_ms": relief ** 2,
           "curv_profile_absmean": np.abs(curv_prof), "curv_plan_absmean": np.abs(curv_plan),
           "slope_break_mean": slope_break, "scarp_density": scarp.astype(np.float32),
           "scarp_skel": skel.astype(np.float32), "scarp_component_max_m": comp_m,
           "orient_c2w": np.cos(t2) * ow, "orient_s2w": np.sin(t2) * ow, "orient_w": ow, "persistence_mean": pers}
    b = np.minimum((strike / (180.0 / N_ORIENT_BINS)).astype(int), N_ORIENT_BINS - 1)
    for i in range(N_ORIENT_BINS):
        out[f"obin{i}"] = np.where(b == i, ow, 0).astype(np.float32)
    for k in out:
        out[k] = np.where(valid, out[k], np.nan).astype(np.float32) if k != "valid" else out[k]
    return out


def _accumulate(acc: dict, feats: dict, src_transform, src_crs, dst_transform, dst_crs, shape) -> None:
    from rasterio.warp import Resampling, reproject

    v = feats["valid"]
    for k, a in feats.items():
        if k == "valid":
            continue
        if k in MAX_BANDS:
            dst = np.full(shape, np.nan, np.float32)
            reproject(np.where(v > 0, a, np.nan).astype(np.float32), dst, src_transform=src_transform, src_crs=src_crs,
                      dst_transform=dst_transform, dst_crs=dst_crs, src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.max)
            acc[k] = np.fmax(acc.get(k, np.full(shape, np.nan, np.float32)), dst)
            continue
        dst = np.zeros(shape, np.float64)
        reproject(np.nan_to_num(a).astype(np.float32), dst, src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.sum)
        acc[k] = acc.get(k, 0) + dst
    cnt = np.zeros(shape, np.float64)
    reproject(v.astype(np.float32), cnt, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform,
              dst_crs=dst_crs, resampling=Resampling.sum)
    acc["_count"] = acc.get("_count", 0) + cnt


def process_tile(path: Path, native_profile: dict, cfg: MorphConfig = MorphConfig(), project_id: int = 0) -> dict:
    """Aggregated 100 m bands on the native-grid window covering the tile: {"window": (r0, r1, c0, c1), bands...}."""
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform

    with rasterio.open(path) as src:
        res = float(abs(src.res[0]))
        b = transform_bounds(src.crs, native_profile["crs"], *src.bounds)
        T = native_profile["transform"]
        r0, c0 = rowcol(T, b[0], b[3])
        r1, c1 = rowcol(T, b[2], b[1])
        r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
        r1, c1 = min(r1 + 2, native_profile["height"]), min(c1 + 2, native_profile["width"])
        shape = (r1 - r0, c1 - c0)
        dst_T = rasterio.transform.Affine(T.a, T.b, T.c + c0 * T.a, T.d, T.e, T.f + r0 * T.e)
        acc: dict = {}
        B, h = cfg.block_px, cfg.halo_px
        for y in range(0, src.height, B):
            for x in range(0, src.width, B):
                wy, wx = min(B, src.height - y), min(B, src.width - x)
                win = Window(x - h, y - h, wx + 2 * h, wy + 2 * h)
                z = src.read(1, window=win, boundless=True, fill_value=src.nodata if src.nodata is not None else np.nan,
                             masked=True).astype(np.float32).filled(np.nan)
                if not np.isfinite(z).any():
                    continue
                f = block_features(z, res, cfg)
                f = {k: a[h:h + wy, h:h + wx] for k, a in f.items()}
                _accumulate(acc, f, win_transform(Window(x, y, wx, wy), src.transform), src.crs, dst_T,
                            native_profile["crs"], shape)
    out = finalize(acc, res, project_id, cell_m=abs(T.a))
    out["window"] = (r0, r1, c0, c1)
    return out


def finalize(acc: dict, res: float, project_id: int, cell_m: float = 100.0) -> dict[str, np.ndarray]:
    n = acc.get("_count")
    if n is None:
        return {}
    nn = np.where(n > 0, n, np.nan)
    o = {k: (acc[k] / nn).astype(np.float32) for k in MEAN_BANDS if k in acc}
    obins = np.stack([acc[f"obin{i}"] for i in range(N_ORIENT_BINS)])
    p = obins / np.maximum(obins.sum(0, keepdims=True), 1e-12)
    ent = -(p * np.log(np.maximum(p, 1e-12))).sum(0) / np.log(N_ORIENT_BINS)
    w = o.pop("orient_w")
    c2, s2 = o.pop("orient_c2w") / np.maximum(w, 1e-12), o.pop("orient_s2w") / np.maximum(w, 1e-12)
    full_cell = (cell_m / res) ** 2                       # 1 m pixels in one fully covered native cell
    out = {"coverage": (n / max(full_cell, 1.0)).astype(np.float32),
           "slope_mean": o["slope_mean"], "slope_max": acc.get("slope_max"),
           "microrelief_rms": np.sqrt(o["microrelief_ms"]).astype(np.float32),
           "curv_profile_absmean": o["curv_profile_absmean"], "curv_plan_absmean": o["curv_plan_absmean"],
           "slope_break_mean": o["slope_break_mean"], "scarp_density": o["scarp_density"],
           "scarp_length_m": (o["scarp_skel"] * n * res * 1.12).astype(np.float32),
           "scarp_component_max_m": acc.get("scarp_component_max_m"),
           "orient_c2": (c2 / np.maximum(np.hypot(c2, s2), 1e-12)).astype(np.float32),
           "orient_s2": (s2 / np.maximum(np.hypot(c2, s2), 1e-12)).astype(np.float32),
           "orient_resultant": np.hypot(c2, s2).astype(np.float32), "orient_entropy": ent.astype(np.float32),
           "persistence_mean": o["persistence_mean"],
           "source_project": np.where(n > 0, float(project_id), np.nan).astype(np.float32),
           "n_tiles": (n > 0).astype(np.float32)}
    for k, v in out.items():
        if v is not None and k not in ("coverage", "n_tiles"):
            out[k] = np.where(n > 0, v, np.nan).astype(np.float32)
    return out


def save_tile(out: dict, native_profile: dict, path: Path, meta: dict) -> Path:
    """Per-tile 100 m bands as a GeoTIFF window of the native grid (+ JSON with provenance)."""
    import rasterio
    from rasterio.transform import Affine

    from ..artifacts.atomic import atomic_save, atomic_write_json

    r0, r1, c0, c1 = out["window"]
    T = native_profile["transform"]
    prof = {"driver": "GTiff", "height": r1 - r0, "width": c1 - c0, "count": len(BANDS), "dtype": "float32",
            "crs": native_profile["crs"], "transform": Affine(T.a, T.b, T.c + c0 * T.a, T.d, T.e, T.f + r0 * T.e),
            "nodata": np.nan, "compress": "deflate", "predictor": 3}

    def write(t):
        with rasterio.open(t, "w", **prof) as dst:
            for i, b in enumerate(BANDS, 1):
                a = out.get(b)
                dst.write(np.full((r1 - r0, c1 - c0), np.nan, np.float32) if a is None else a.astype(np.float32), i)
                dst.set_band_description(i, b)

    atomic_save(path, write)
    atomic_write_json(Path(path).with_suffix(".json"), {**meta, "window": out["window"]})
    return path


def mosaic(tile_paths: list[Path], native_profile: dict, out_path: Path) -> Path:
    """Coverage-weighted merge of per-tile windows onto the full native grid (max for *_max bands)."""
    import rasterio

    from ..artifacts.atomic import atomic_save

    H, W = native_profile["height"], native_profile["width"]
    num = {b: np.zeros((H, W), np.float64) for b in BANDS}
    mx = {b: np.full((H, W), np.nan, np.float32) for b in ("slope_max", "scarp_component_max_m")}
    cov = np.zeros((H, W), np.float64)
    best_cov = np.zeros((H, W), np.float32)
    src_proj = np.full((H, W), np.nan, np.float32)
    ntiles = np.zeros((H, W), np.float32)
    T = native_profile["transform"]
    for p in tile_paths:
        with rasterio.open(p) as src:
            r0 = int(round((src.transform.f - T.f) / T.e))
            c0 = int(round((src.transform.c - T.c) / T.a))
            a = {d: src.read(i + 1) for i, d in enumerate(src.descriptions)}
        sl = (slice(r0, r0 + a["coverage"].shape[0]), slice(c0, c0 + a["coverage"].shape[1]))
        cv = np.nan_to_num(a["coverage"])
        for b in BANDS:
            if b in mx or b in ("coverage", "source_project", "n_tiles"):
                continue
            num[b][sl] += np.nan_to_num(a[b]) * cv
        for b in mx:
            mx[b][sl] = np.fmax(mx[b][sl], a[b])
        better = cv > best_cov[sl]
        src_proj[sl] = np.where(better, a["source_project"], src_proj[sl])
        best_cov[sl] = np.maximum(best_cov[sl], cv)
        cov[sl] += cv
        ntiles[sl] += (cv > 0)
    out = {}
    for b in BANDS:
        if b == "coverage":
            out[b] = np.clip(cov, 0, 1).astype(np.float32)
        elif b == "source_project":
            out[b] = src_proj
        elif b == "n_tiles":
            out[b] = ntiles
        elif b in mx:
            out[b] = mx[b]
        else:
            out[b] = np.where(cov > 0, num[b] / np.maximum(cov, 1e-9), np.nan).astype(np.float32)
    prof = {"driver": "GTiff", "height": H, "width": W, "count": len(BANDS), "dtype": "float32",
            "crs": native_profile["crs"], "transform": T, "nodata": np.nan, "compress": "deflate", "predictor": 3,
            "tiled": True, "blockxsize": 256, "blockysize": 256}

    def write(t):
        with rasterio.open(t, "w", **prof) as dst:
            for i, b in enumerate(BANDS, 1):
                dst.write(out[b], i)
                dst.set_band_description(i, b)

    return atomic_save(out_path, write)


def config_record(cfg: MorphConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg)))


def tile_table(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        j = Path(p).with_suffix(".json")
        rows.append({"tile": Path(p).stem, **(json.loads(j.read_text(encoding="utf-8")) if j.exists() else {})})
    return pd.DataFrame(rows)
