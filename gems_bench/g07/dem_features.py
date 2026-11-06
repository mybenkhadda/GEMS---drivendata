"""Native-resolution geomorphometry of a DEM window (label-free; metres, degrees, 1/m).

Input: a padded work-resolution elevation window ``z`` (metres, NaN = no data) and its pixel size ``res`` (m).
Smoothing is NORMALISED convolution (``g06.feature_physics.masked_smooth``): gaps never become zeros, so tile / gap
edges lose support instead of creating artificial scarps. Derivatives are central differences in metres.

Feature ids ``dem_<op>[_s<scale m>]`` and their 100 m aggregation (``KINDS``):

    slope_s          deg            mean;  slope_max_s  max
    hs<az>           [0, 1]         mean   (Lambertian hillshade, sun altitude 45 deg, azimuth az)
    hs_range         [0, 1]         mean   (max - min over azimuths: direction-dependent relief = lineaments)
    curvprof_s / curvplan_s / curvmean_s   1/m   mean of |.|
    relief_w         m              mean + relief_max_w max (max - min elevation in a w-m window)
    tpi_s            m              mean of |z - G_s(z)| (topographic position)
    rough_s          m              mean (std of the detrended surface z - G_s(z) in a 3 s window)
    openpos / openneg  deg          mean (positive / negative openness over ``openness_radius_m``)
    ridge_s / valley_s m            mean + *_max (scale-normalised Hessian line strength)
    aniso_s          [0, 1]         mean (structure-tensor coherence) + orientation (doubled strike angle, weighted)
Terrain-structure features (scarps, lineaments, truncation, drainage) come from ``terrain_structure`` and are merged
by ``dem_multiscale_features``. Scales finer than 1.5 x the work resolution are dropped (recorded).
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..g06.feature_physics import masked_smooth
from .dem_io import MAX, MEAN, ORIENT


def _sig(scale_m: float, res: float) -> float:
    return max(float(scale_m) / float(res), 0.5)


def usable_scales(scales_m, res: float) -> list[float]:
    return [float(s) for s in scales_m if float(s) >= 1.5 * float(res)]


def smooth_derivs(z: np.ndarray, res: float, scale_m: float) -> dict:
    """Smoothed surface and its first / second derivatives (per metre; x = east, y = north)."""
    S, _ = masked_smooth(z, _sig(scale_m, res))
    gr, gc = np.gradient(S, res)                     # rows point south
    grr, grc = np.gradient(gr, res)
    _, gcc = np.gradient(gc, res)
    zx, zy = gc, -gr
    zxx, zyy, zxy = gcc, grr, -grc
    return {"S": S, "zx": zx, "zy": zy, "zxx": zxx, "zyy": zyy, "zxy": zxy, "res": res, "scale": scale_m}


def dem_slope(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(slope deg, aspect rad of the downslope direction from east, CCW)."""
    g = np.hypot(d["zx"], d["zy"])
    return np.degrees(np.arctan(g)), np.arctan2(-d["zy"], -d["zx"])


def dem_hillshade(d: dict, azimuths=(315.0, 45.0, 135.0, 225.0), altitude: float = 45.0) -> dict:
    """Lambertian hillshade in [0, 1] per sun azimuth (deg clockwise from north) + the across-azimuth range."""
    slope = np.arctan(np.hypot(d["zx"], d["zy"]))
    aspect = np.arctan2(d["zy"], -d["zx"])
    alt = np.radians(altitude)
    out = {}
    for az in azimuths:
        zen = np.pi / 2 - alt
        a = np.radians(360.0 - az + 90.0)
        hs = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(a - aspect)
        out[f"hs{int(az)}"] = np.clip(hs, 0, 1)
    st = np.stack(list(out.values()))
    out["hs_range"] = st.max(0) - st.min(0)
    return out


def dem_curvature(d: dict) -> dict:
    """Profile, plan and mean curvature (1/m; Evans-Young conventions on the smoothed surface)."""
    p, q, r, t, s = d["zx"], d["zy"], d["zxx"], d["zyy"], d["zxy"]
    p2 = p * p + q * q + 1e-12
    prof = -(r * p * p + 2 * s * p * q + t * q * q) / (p2 * (1 + p2) ** 1.5)
    plan = -(r * q * q - 2 * s * p * q + t * p * p) / (p2 ** 1.5)
    mean = -((1 + q * q) * r - 2 * p * q * s + (1 + p * p) * t) / (2 * (1 + p2) ** 1.5)
    return {"curvprof": prof, "curvplan": plan, "curvmean": mean}


def dem_local_relief(z: np.ndarray, res: float, window_m: float) -> np.ndarray:
    """max - min elevation over a square window (NaN-aware: missing pixels never act as extremes)."""
    w = max(int(round(window_m / res)) | 1, 3)
    m = np.isfinite(z)
    mx = ndimage.maximum_filter(np.where(m, z, -np.inf), w, mode="constant", cval=-np.inf)
    mn = ndimage.minimum_filter(np.where(m, z, np.inf), w, mode="constant", cval=np.inf)
    r = mx - mn
    return np.where(m & np.isfinite(r), r, np.nan)


def dem_terrain_position(z: np.ndarray, res: float, scale_m: float) -> np.ndarray:
    """Topographic position index z - G_scale(z) (m)."""
    S, _ = masked_smooth(z, _sig(scale_m, res))
    return np.where(np.isfinite(z), z - S, np.nan)


def dem_roughness(z: np.ndarray, res: float, scale_m: float) -> np.ndarray:
    """Std of the detrended surface (z - G_s(z)) over a Gaussian window of 3 s (m)."""
    t = dem_terrain_position(z, res, scale_m)
    m1, _ = masked_smooth(t, _sig(3 * scale_m, res))
    m2, _ = masked_smooth(t * t, _sig(3 * scale_m, res))
    return np.where(np.isfinite(z), np.sqrt(np.maximum(m2 - m1 * m1, 0)), np.nan)


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """a[r + dr, c + dc] with NaN beyond the array."""
    out = np.full(a.shape, np.nan, a.dtype)
    h, w = a.shape
    rs, re = max(-dr, 0), min(h - dr, h)
    cs, ce = max(-dc, 0), min(w - dc, w)
    if rs < re and cs < ce:
        out[rs:re, cs:ce] = a[rs + dr:re + dr, cs + dc:ce + dc]
    return out


def dem_openness(z: np.ndarray, res: float, radius_m: float, n_dirs: int = 8, n_steps: int = 6) -> dict:
    """Positive / negative openness (deg) from elevation angles along ``n_dirs`` rays (geometric distances)."""
    dists = np.unique(np.round(np.geomspace(res, radius_m, n_steps) / res).astype(int))
    dists = dists[dists >= 1]
    pos, neg = [], []
    z = np.asarray(z, np.float64)
    for k in range(n_dirs):
        ang = 2 * np.pi * k / n_dirs
        dr, dc = -np.sin(ang), np.cos(ang)
        mx = np.full(z.shape, -np.inf)
        mn = np.full(z.shape, np.inf)
        for d in dists:
            zz = _shift(z, int(round(dr * d)), int(round(dc * d)))
            e = np.arctan((zz - z) / (d * res))
            mx = np.fmax(mx, np.where(np.isfinite(e), e, -np.inf))
            mn = np.fmin(mn, np.where(np.isfinite(e), e, np.inf))
        pos.append(np.where(np.isfinite(mx), 90.0 - np.degrees(mx), np.nan))
        neg.append(np.where(np.isfinite(mn), 90.0 + np.degrees(mn), np.nan))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        op, on = np.nanmean(np.stack(pos), 0), np.nanmean(np.stack(neg), 0)
    valid = np.isfinite(z)
    return {"openpos": np.where(valid, op, np.nan), "openneg": np.where(valid, on, np.nan)}


def ridge_valley(d: dict) -> dict:
    """Scale-normalised ridge / valley strength (m): max(0, -l2) s^2, max(0, l1) s^2 of the Hessian."""
    tr = d["zxx"] + d["zyy"]
    disc = np.sqrt(np.maximum(0.25 * (d["zxx"] - d["zyy"]) ** 2 + d["zxy"] ** 2, 0))
    l1, l2 = 0.5 * tr + disc, 0.5 * tr - disc
    s2 = float(d["scale"]) ** 2
    return {"ridge": np.maximum(0, -l2) * s2, "valley": np.maximum(0, l1) * s2}


def anisotropy(d: dict, rho_m: float) -> dict:
    """Structure-tensor coherence and doubled STRIKE angle (rad) with its weight (coherence x gradient energy)."""
    res = d["res"]
    ge, gn = d["zx"], d["zy"]
    jee, _ = masked_smooth(ge * ge, _sig(rho_m, res))
    jen, _ = masked_smooth(ge * gn, _sig(rho_m, res))
    jnn, _ = masked_smooth(gn * gn, _sig(rho_m, res))
    tr = jee + jnn
    disc = np.sqrt(np.maximum((jee - jnn) ** 2 + 4 * jen * jen, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(tr > 0, (disc / tr) ** 2, 0.0)
    theta_grad = 0.5 * np.arctan2(2 * jen, jee - jnn)
    strike2 = 2 * (theta_grad + np.pi / 2)
    return {"aniso": coh, "orient": strike2, "orient__w": coh * np.sqrt(np.maximum(tr, 0)), "theta_grad": theta_grad}


# =============================================================================== bank
def tier_scales(cfg: dict, tier: str) -> list:
    return list(cfg["scales_m"] if tier == "fine" else cfg.get("coarse_scales_m", []))


def feature_kinds(cfg: dict, res: float, tier: str = "fine") -> dict:
    """{feature id: aggregation kind} produced by ``dem_multiscale_features`` for this tier / resolution. The
    ``fine`` tier (native / work resolution) also carries hillshades, local relief and openness; the ``coarse`` tier
    (``coarse_res_m``) carries the 50-500 m scales only."""
    from . import terrain_structure as TS

    sc = usable_scales(tier_scales(cfg, tier), res)
    k = {}
    for s in sc:
        t = int(round(s))
        k.update({f"dem_slope_s{t}": MEAN, f"dem_slope_max_s{t}": MAX, f"dem_curvprof_s{t}": MEAN,
                  f"dem_curvplan_s{t}": MEAN, f"dem_curvmean_s{t}": MEAN, f"dem_tpi_s{t}": MEAN,
                  f"dem_rough_s{t}": MEAN, f"dem_ridge_s{t}": MEAN, f"dem_valley_s{t}": MEAN,
                  f"dem_ridge_max_s{t}": MAX, f"dem_valley_max_s{t}": MAX, f"dem_aniso_s{t}": MEAN,
                  f"dem_orient_s{t}": ORIENT})
    if tier == "fine":
        for az in cfg.get("hillshade_azimuths", [315, 45, 135, 225]):
            k[f"dem_hs{int(az)}"] = MEAN
        k["dem_hs_range"] = MEAN
        for w in cfg.get("relief_windows_m", [30, 150]):
            k[f"dem_relief_w{int(w)}"] = MEAN
            k[f"dem_relief_max_w{int(w)}"] = MAX
        if cfg.get("openness_radius_m"):
            k.update({"dem_openpos": MEAN, "dem_openneg": MEAN})
    k.update(TS.structure_kinds(cfg, res, sc, fine=tier == "fine"))
    return k


def dem_multiscale_features(z: np.ndarray, res: float, cfg: dict, tier: str = "fine") -> dict:
    """Every feature of ``feature_kinds(cfg, res, tier)`` for one padded window (NaN where the DEM is missing)."""
    from . import terrain_structure as TS

    valid = np.isfinite(z)
    out, ds = {}, {}
    sc = usable_scales(tier_scales(cfg, tier), res)
    for s in sc:
        t = int(round(s))
        d = smooth_derivs(z, res, s)
        ds[s] = d
        slope, _ = dem_slope(d)
        out[f"dem_slope_s{t}"] = slope
        out[f"dem_slope_max_s{t}"] = slope
        cv = dem_curvature(d)
        for k, v in cv.items():
            out[f"dem_{k}_s{t}"] = np.abs(v)
        out[f"dem_tpi_s{t}"] = np.abs(dem_terrain_position(z, res, s))
        out[f"dem_rough_s{t}"] = dem_roughness(z, res, s)
        rv = ridge_valley(d)
        out[f"dem_ridge_s{t}"], out[f"dem_valley_s{t}"] = rv["ridge"], rv["valley"]
        out[f"dem_ridge_max_s{t}"], out[f"dem_valley_max_s{t}"] = rv["ridge"], rv["valley"]
        an = anisotropy(d, float(cfg.get("integration_factor", 2.0)) * s)
        d["theta_grad"], d["aniso"] = an["theta_grad"], an["aniso"]
        out[f"dem_aniso_s{t}"] = an["aniso"]
        out[f"dem_orient_s{t}"], out[f"dem_orient_s{t}__w"] = an["orient"], an["orient__w"]
    if tier == "fine":
        d0 = ds[sc[0]] if sc else smooth_derivs(z, res, max(2 * res, 1.0))
        out.update({f"dem_{k}": v for k, v in dem_hillshade(d0, cfg.get("hillshade_azimuths", [315, 45, 135, 225]),
                                                             float(cfg.get("sun_altitude_deg", 45))).items()})
        for w in cfg.get("relief_windows_m", [30, 150]):
            r = dem_local_relief(z, res, float(w))
            out[f"dem_relief_w{int(w)}"], out[f"dem_relief_max_w{int(w)}"] = r, r
        if cfg.get("openness_radius_m"):
            out.update({f"dem_{k}": v for k, v in dem_openness(z, res, float(cfg["openness_radius_m"]),
                                                               int(cfg.get("openness_dirs", 8))).items()})
    out.update(TS.structure_bank(z, res, ds, cfg, fine=tier == "fine"))
    for k in list(out):
        if not k.endswith("__w"):
            out[k] = np.where(valid, np.asarray(out[k], np.float32), np.nan)
    return apply_edge_guard(out, valid, res, cfg, sc)


def apply_edge_guard(feats: dict, valid: np.ndarray, res: float, cfg: dict, scales: list) -> dict:
    """NaN where a feature's operator reaches missing data. Normalised convolution keeps a value there, but one-sided
    support bends the smoothed surface (slope error ~11 % at 2 sigma from the edge, ~0.05 % at 4), so slope / curvature /
    scarp / lineament responses would run along every data edge, at the 10 km tile spacing: an artificial lineament.
    Guard = ``edge_guard_sigma`` (4 = the Gaussian truncation) x the feature's smoothing scale, x 3 for structure-tensor
    products (integration over 2 s) and x 4 for roughness (3 s window), half a relief window, the openness radius, plus
    2 px of derivative reach. Coverage still counts the DEM pixels themselves; every guard fits inside the halo."""
    k_sigma = float(cfg.get("edge_guard_sigma", 4.0))
    if k_sigma <= 0 or valid.all():
        return feats
    dist = ndimage.distance_transform_edt(valid)
    smin = min(scales) if scales else 2 * res
    smax = max(scales) if scales else 2 * res
    for k in list(feats):
        if k.endswith("__w"):
            continue
        m = re.search(r"_([sw])(\d+)$", k)
        if m and m.group(1) == "s":
            mult = 4.0 if k.startswith("dem_rough") else (3.0 if k.startswith(("dem_aniso", "dem_orient")) else 1.0)
            px = k_sigma * mult * float(m.group(2)) / res
        elif m:
            px = float(m.group(2)) / (2 * res)
        elif k.startswith("dem_hs"):
            px = k_sigma * smin / res                     # hillshades use the smallest scale
        elif k.startswith("dem_open"):
            px = float(cfg.get("openness_radius_m", 0)) / res
        else:
            px = k_sigma * smax / res                     # tier-wide persistence / drainage
        feats[k] = np.where(dist > np.ceil(px) + 2, feats[k], np.nan)
    return feats


# =============================================================================== processing drivers
def aggregated_names(kinds: dict) -> list[str]:
    """Analysis-grid band names produced by aggregating ``kinds`` (orientation -> cos2 / sin2 / resultant)."""
    out = []
    for k, v in kinds.items():
        out += [f"{k}_cos2", f"{k}_sin2", f"{k}_resultant"] if v == ORIENT else [k]
    return out


def process_window_stream(ds, grid, cfg: dict, tier: str, factor: int, vscale: float, acc, halo: int,
                          windows=None, clip_bounds=None, progress=None, neighbours=None) -> int:
    """Stream work windows of an open DEM dataset through the feature bank into an ``Accumulator``. Returns the
    number of windows with data. ``clip_bounds`` (west, south, east, north, dataset CRS) restricts the pixels that
    are aggregated (the halo outside still feeds the filters); ``neighbours`` fill the halo beyond this raster."""
    from .dem_io import core_transform, iter_dem_tiles, read_dem_window

    res = float(abs(ds.res[0])) * factor
    tile = int(cfg.get("tile_px", 1024))
    n = 0
    wins = windows if windows is not None else list(iter_dem_tiles(ds, tile, halo, factor))
    for i, w in enumerate(wins):
        z = read_dem_window(ds, w, factor, vscale, neighbours=neighbours)
        rs, cs = w.core
        valid = np.isfinite(z[rs, cs])
        if not valid.any():                               # data only in the halo: nothing to aggregate here
            continue
        f = dem_multiscale_features(z, res, cfg, tier)
        core = {k: v[rs, cs] for k, v in f.items()}
        T = core_transform(ds, w, factor)
        if clip_bounds is not None:
            rr, cc = np.indices(valid.shape)
            x, y = T.c + (cc + 0.5) * T.a, T.f + (rr + 0.5) * T.e
            west, south, east, north = clip_bounds
            valid &= (x >= west) & (x < east) & (y > south) & (y <= north)
        acc.add(core, valid, T, ds.crs)
        n += 1
        if progress and i % 10 == 0:
            progress(f"    window {i + 1}/{len(wins)}")
    return n


def process_fine_source(path, src, cfg: dict, grid, out_path, factor: int, meta: dict, progress=None, neighbours=None):
    """Fine tier of one DEM source -> analysis-window GeoTIFF (+ json); returns the path or None (no data).
    ``neighbours`` = DEMSource list whose aligned pixels fill the halo across this tile's edges."""
    import rasterio

    from .dem_io import Accumulator, analysis_window, source_vscale, write_window

    opened = []
    try:
        with rasterio.open(path) as ds:
            res = float(abs(ds.res[0])) * factor
            kinds = feature_kinds(cfg, res, "fine")
            win = analysis_window(tuple(ds.bounds), ds.crs, grid)
            if win is None:
                return None
            for s in neighbours or []:
                opened.append((rasterio.open(s.path), source_vscale(s)))
            acc = Accumulator(grid, win, kinds)
            halo = max(int(cfg.get("halo_px", 128)), required_halo_px(cfg, res, "fine"))
            n = process_window_stream(ds, grid, cfg, "fine", factor, source_vscale(src), acc, halo, progress=progress,
                                      neighbours=opened)
            if not n:
                return None
            out = acc.finalize(res)                   # counts are WORK pixels (res = native x factor)
    finally:
        for d, _ in opened:
            d.close()
    names = ["dem_coverage"] + aggregated_names(kinds)
    return write_window(out_path, out, names, grid, win, {**meta, "tier": "fine", "work_res_m": res, "halo_px": halo,
                                                          "native_px": acc.native_px,
                                                          "neighbours": [s.tile_id for s in neighbours or []]})


def coarse_part(src, coarse_res_m: float, parts_dir) -> Path:
    """NaN-aware block mean of one source at ``coarse_res_m`` in metres (cached: pilot and full builds share parts)."""
    import rasterio

    from .dem_io import _read_clean, block_mean, source_vscale
    from .persistence import atomic_save

    p = Path(parts_dir) / f"{src.tile_id}_{Path(src.path).stat().st_size}.tif"
    if p.exists():
        return p
    with rasterio.open(src.path) as ds:
        f = max(int(round(coarse_res_m / abs(ds.res[0]))), 1)
        z = block_mean(_read_clean(ds, None) * source_vscale(src), f)
        T = ds.transform
        prof = {"driver": "GTiff", "height": z.shape[0], "width": z.shape[1], "count": 1, "dtype": "float32",
                "crs": ds.crs, "nodata": np.nan, "compress": "deflate", "predictor": 3,
                "transform": rasterio.transform.Affine(T.a * f, T.b, T.c, T.d, T.e * f, T.f)}

    def w(t):
        with rasterio.open(t, "w", **prof) as d:
            d.write(z.astype(np.float32), 1)

    return atomic_save(p, w)


def merge_coarse_parts(parts: list, coarse_res_m: float, out_path, crs: str) -> Path:
    """Union of the parts on one grid (first part wins in overlaps), through a disk-backed array: the merged raster is
    written block by block and never held in RAM. Parts of tiles in one project share one pixel phase."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    from .persistence import atomic_save

    boxes = []
    for p in parts:
        with rasterio.open(p) as d:
            boxes.append(tuple(d.bounds))
    west, north = min(b[0] for b in boxes), max(b[3] for b in boxes)
    east, south = max(b[2] for b in boxes), min(b[1] for b in boxes)
    res = float(coarse_res_m)
    W, H = int(np.ceil((east - west) / res - 1e-9)), int(np.ceil((north - south) / res - 1e-9))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_arr = out_path.with_name(out_path.stem + ".merge.npy")
    mm = np.lib.format.open_memmap(tmp_arr, mode="w+", dtype=np.float32, shape=(H, W))
    try:
        mm[:] = np.nan
        for p in parts:
            with rasterio.open(p) as d:
                a, T = d.read(1), d.transform
            c0, r0 = int(round((T.c - west) / res)), int(round((north - T.f) / res))
            h, w = min(a.shape[0], H - r0), min(a.shape[1], W - c0)
            cur = mm[r0:r0 + h, c0:c0 + w]
            hole = np.isnan(cur)
            cur[hole] = a[:h, :w][hole]
        prof = {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "float32", "crs": crs,
                "transform": Affine(res, 0.0, west, 0.0, -res, north), "nodata": np.nan, "compress": "deflate",
                "predictor": 3, "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}

        def writer(t):
            with rasterio.open(t, "w", **prof) as d:
                for r in range(0, H, 2048):
                    n = min(2048, H - r)
                    d.write(np.asarray(mm[r:r + n], np.float32), 1, window=Window(0, r, W, n))
                d.update_tags(units="metre", G07="coarse DEM (NaN-aware block mean)")

        atomic_save(out_path, writer)
    finally:
        mm = None                                 # noqa: F841 - closes the memmap before its file goes
        tmp_arr.unlink(missing_ok=True)
    return out_path


def build_coarse_dem(sources, coarse_res_m: float, out_path, crs: str, parts_dir=None, progress=None, on_part=None
                     ) -> Path:
    """One coarse (e.g. 10 m) DEM of every usable source of one CRS. Parts are cached per tile in ``parts_dir`` and
    reported through ``on_part`` (persisted to Drive), so a disconnect never re-reads a finished tile."""
    out_path = Path(out_path)
    parts_dir = Path(parts_dir) if parts_dir else out_path.parent / "parts"
    parts = []
    for i, s in enumerate(sources):
        existed = (parts_dir / f"{s.tile_id}_{Path(s.path).stat().st_size}.tif").exists()
        p = coarse_part(s, coarse_res_m, parts_dir)
        parts.append(p)
        if not existed:
            if on_part:
                on_part(p)
            if progress:
                progress(f"  coarse part {i + 1}/{len(sources)} {s.tile_id}")
    return merge_coarse_parts(parts, coarse_res_m, out_path, crs)


def process_coarse(coarse_path, cfg: dict, grid, out_dir, meta: dict, progress=None, on_product=None, deadline=None
                   ) -> tuple[list, int]:
    """Coarse tier over the coarse DEM: one analysis-window product per work window (seamless halos). Returns
    (products, windows deferred past ``deadline``, a ``time.time()`` value). Windows are large (``coarse_tile_px``)
    because the halo of the 300 m operators would otherwise dominate the work."""
    import rasterio

    from .dem_io import Accumulator, analysis_window, core_transform, iter_dem_tiles, write_window

    out, deferred = [], 0
    with rasterio.open(coarse_path) as ds:
        res = float(abs(ds.res[0]))
        kinds = feature_kinds(cfg, res, "coarse")
        names = ["dem_coverage"] + aggregated_names(kinds)
        halo = max(int(cfg.get("halo_px", 128)), required_halo_px(cfg, res, "coarse"))
        for w in iter_dem_tiles(ds, int(cfg.get("coarse_tile_px", 2048)), halo, 1):
            p = Path(out_dir) / f"coarse_{w.tile_id}.tif"
            if p.exists() and p.with_suffix(".json").exists():
                out.append(p)
                continue
            if deadline is not None and time.time() > deadline:
                deferred += 1
                continue
            T = core_transform(ds, w, 1)
            h, wd = w.core_shape
            win = analysis_window((T.c, T.f + h * T.e, T.c + wd * T.a, T.f), ds.crs, grid)
            if win is None:
                continue
            acc = Accumulator(grid, win, kinds)
            if process_window_stream(ds, grid, cfg, "coarse", 1, 1.0, acc, halo, windows=[w]):
                out.append(write_window(p, acc.finalize(res), names, grid, win,
                                        {**meta, "tier": "coarse", "work_res_m": res, "halo_px": halo}))
                if on_product:
                    on_product(p)
            if progress:
                progress(f"  coarse window {w.tile_id}")
    return out, deferred


def stage_tile_local(path, cache_dir) -> tuple[Path, bool]:
    """(path to read, is a temporary copy): a Drive tile is copied to local disk once for its window reads."""
    src, dst = Path(path), Path(cache_dir) / Path(path).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return dst, True


def required_halo_px(cfg: dict, res: float, tier: str = "fine") -> int:
    """Support (work pixels) of the largest operator of a tier: Gaussian 4 sigma + tensor integration, roughness
    window, openness rays, relief window, structure sampling."""
    from . import terrain_structure as TS

    sc = usable_scales(tier_scales(cfg, tier), res) or [2 * res]
    smax = max(sc)
    g = int(np.ceil(4 * smax / res)) + 2
    rho = int(np.ceil(4 * float(cfg.get("integration_factor", 2.0)) * smax / res))
    rough = g + int(np.ceil(4 * 3 * smax / res))         # TPI smoothing, then the 3 s roughness window
    opn = int(np.ceil(float(cfg.get("openness_radius_m", 0)) / res)) if tier == "fine" else 0
    rel = int(np.ceil(max(cfg.get("relief_windows_m", [0]) or [0]) / res / 2)) if tier == "fine" else 0
    return max(g + rho, rough, opn, rel, TS.required_extra_px(cfg, res, sc) + g) + 2
