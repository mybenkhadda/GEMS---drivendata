"""Windowed multi-tile DEM reading and exact aggregation to the 100 m master grid.

``DEMReader.read_dem_window`` reads ONLY the tiles whose footprint intersects the requested (halo-expanded)
bounds, warps each through a ``WarpedVRT`` onto one output grid (bilinear, nodata preserved), merges them by the
catalog priority (first valid value wins), records which tile filled each pixel, and flags seams where adjacent
pixels from different tiles differ by more than ``dem.seam_tolerance_m`` relative to the local slope.
Vertical units are converted to metres with the catalog ``vertical_scale``; tiles with unknown vertical units are
used only if ``allow_unknown_vertical`` is True (flagged in the window's quality flags).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DEMWindowError(RuntimeError):
    pass


@dataclass
class DEMWindow:
    elevation: np.ndarray            # float32 metres, NaN = nodata
    transform: tuple
    crs: str
    valid: np.ndarray
    source_index: np.ndarray         # int16 index into source_tiles (-1 = none)
    source_tiles: list
    resolution: float
    seam_mask: np.ndarray
    quality_flags: list = field(default_factory=list)
    halo_px: int = 0

    @property
    def core(self) -> tuple[slice, slice]:
        h = self.halo_px
        return slice(h, self.elevation.shape[0] - h or None), slice(h, self.elevation.shape[1] - h or None)


class DEMReader:
    def __init__(self, catalog, cfg: dict, allow_unknown_vertical: bool = False):
        from .dem_catalog import DEMIndex

        self.cfg = cfg
        self.index = DEMIndex(catalog)
        self.allow_unknown = allow_unknown_vertical
        self.cache: OrderedDict = OrderedDict()
        self.max_items = int(cfg["dem"]["window_cache_items"])
        self.native_res = float(np.nanmin(catalog.resolution_m)) if len(catalog) else np.nan

    def tiles_for(self, bounds, crs) -> pd.DataFrame:
        from rasterio.warp import transform_bounds
        from shapely.geometry import box

        b = bounds
        if crs is not None and self.index.cat.crs is not None and str(crs) != str(self.index.cat.crs):
            b = transform_bounds(crs, self.index.cat.crs, *bounds, densify_pts=21)
        t = self.index.query(box(*b))
        if not self.allow_unknown and len(t):
            t = t[~t.quality_flags.fillna("").str.contains("UNKNOWN_VERTICAL_UNIT|CONFLICTING_VERTICAL_UNIT")]
        return t

    def read_dem_window(self, bounds, target_crs, resolution: float | None = None, halo_m: float = 0.0,
                        resampling: str = "bilinear", snap: bool = True) -> DEMWindow:
        """Elevation on a north-up grid covering ``bounds`` (+ ``halo_m`` on every side) at ``resolution`` m.

        With ``snap`` the origin is snapped to a multiple of ``resolution`` so windows from adjacent requests nest;
        ``snap=False`` keeps the exact bounds (used when they are already aligned to the master grid).
        """
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin
        from rasterio.vrt import WarpedVRT

        res = float(resolution or self.native_res)
        if not np.isfinite(res) or res <= 0:
            raise DEMWindowError("no usable DEM tiles / resolution")
        x0, y0, x1, y1 = bounds
        x0, y0, x1, y1 = x0 - halo_m, y0 - halo_m, x1 + halo_m, y1 + halo_m
        if snap:
            x0, y1 = np.floor(x0 / res) * res, np.ceil(y1 / res) * res
        w, h = int(np.ceil((x1 - x0) / res - 1e-9)), int(np.ceil((y1 - y0) / res - 1e-9))
        if w * h / 1e6 > float(self.cfg["dem"]["max_window_mpix"]):
            raise DEMWindowError(f"window {w}x{h} at {res} m exceeds dem.max_window_mpix - use a coarser resolution")
        key = (round(x0, 3), round(y1, 3), w, h, res, str(target_crs), resampling)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        T = from_origin(x0, y1, res, res)
        z = np.full((h, w), np.nan, np.float32)
        src_idx = np.full((h, w), -1, np.int16)
        tiles = self.tiles_for((x0, y1 - h * res, x0 + w * res, y1), target_crs)
        flags, used = [], []
        rs = getattr(Resampling, resampling)
        for r in tiles.itertuples():
            with rasterio.open(r.file_path) as ds:
                nd = ds.nodata
                with WarpedVRT(ds, crs=target_crs, transform=T, width=w, height=h, resampling=rs,
                               src_nodata=nd, nodata=np.nan, dtype="float32") as vrt:
                    a = vrt.read(1).astype(np.float32)
            if nd is not None and np.isfinite(nd):
                a[a == nd] = np.nan
            a[np.abs(a) > 1e20] = np.nan
            a *= np.float32(r.vertical_scale)
            fill = np.isnan(z) & np.isfinite(a)
            if fill.any():
                k = len(used)
                used.append(r.tile_id)
                z[fill] = a[fill]
                src_idx[fill] = k
                if "UNKNOWN_VERTICAL_UNIT" in str(r.quality_flags):
                    flags.append(f"{r.tile_id}: unknown vertical unit (assumed metres)")
        seam = detect_seams(z, src_idx, res, float(self.cfg["dem"]["seam_tolerance_m"]))
        if seam.any():
            flags.append(f"{int(seam.sum())} seam pixels")
        win = DEMWindow(z, tuple(T)[:6], str(target_crs), np.isfinite(z), src_idx, used, res, seam, flags,
                        int(round(halo_m / res)))
        self.cache[key] = win
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return win


def detect_seams(z: np.ndarray, src: np.ndarray, res: float, tol_m: float) -> np.ndarray:
    """Pixels on a boundary between two source tiles whose elevation step exceeds ``tol_m`` plus the step the
    local slope explains (|dz| > tol + 2 * median |dz| of the surrounding same-source neighbours)."""
    seam = np.zeros(z.shape, bool)
    for axis in (0, 1):
        a, b = (z[1:], z[:-1]) if axis == 0 else (z[:, 1:], z[:, :-1])
        sa, sb = (src[1:], src[:-1]) if axis == 0 else (src[:, 1:], src[:, :-1])
        dz = np.abs(a - b)
        same = (sa == sb) & (sa >= 0)
        typical = np.nanmedian(dz[same]) if same.any() else 0.0
        boundary = (sa != sb) & (sa >= 0) & (sb >= 0) & (dz > tol_m + 2 * np.nan_to_num(typical))
        if axis == 0:
            seam[1:] |= boundary
            seam[:-1] |= boundary
        else:
            seam[:, 1:] |= boundary
            seam[:, :-1] |= boundary
    return seam


# ------------------------------------------------------------------------------------- local DEM mosaic
def build_dem_mosaic(reader: DEMReader, grid, covered: np.ndarray, res_m: float, out_path, chunk_cells: int = 64,
                     log=print):
    """One-time local GeoTIFF mosaic of the usable DEM at ``res_m`` aligned to the master grid (factor 100/res).

    Used by the E4 patch dataset so training never re-reads 1 m tiles from Drive. Only chunks with DEM coverage are
    read; the rest stays nodata. Written tiled + compressed, atomically, to local scratch.
    """
    import os

    import rasterio
    from rasterio.transform import Affine

    f = int(round(grid.res[0] / res_m))
    if abs(f * res_m - grid.res[0]) > 1e-6:
        raise DEMWindowError(f"mosaic resolution {res_m} must divide the {grid.res[0]} m master cell")
    a, _, c0, _, e, f0 = grid.transform[:6]
    H, W = grid.shape
    T = Affine(a / f, 0, c0, 0, e / f, f0)
    p = __import__("pathlib").Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".partial" + p.suffix)
    prof = {"driver": "GTiff", "height": H * f, "width": W * f, "count": 1, "dtype": "float32", "crs": grid.crs,
            "transform": T, "nodata": np.nan, "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "deflate",
            "predictor": 3, "BIGTIFF": "YES"}
    n = 0
    with rasterio.open(tmp, "w", **prof) as ds:
        for r0 in range(0, H, chunk_cells):
            for q0 in range(0, W, chunk_cells):
                r1, q1 = min(r0 + chunk_cells, H), min(q0 + chunk_cells, W)
                if not covered[r0:r1, q0:q1].any():
                    continue
                b = (c0 + a * q0, f0 + e * r1, c0 + a * q1, f0 + e * r0)
                w = reader.read_dem_window(b, grid.crs, resolution=res_m, resampling="average", snap=False)
                z = w.elevation[: (r1 - r0) * f, : (q1 - q0) * f]
                ds.write(z[None], window=rasterio.windows.Window(q0 * f, r0 * f, z.shape[1], z.shape[0]))
                n += 1
    os.replace(tmp, p)
    if log:
        log(f"[dem-mosaic] {n} chunks written at {res_m} m -> {p}")
    return p


class MosaicDEMReader:
    """``read_dem_window`` served from a local mosaic (same interface as :class:`DEMReader` for its resolution)."""

    def __init__(self, path, source_name: str = "local_mosaic"):
        import rasterio

        self.path = str(path)
        self.ds = rasterio.open(self.path)
        self.res = float(self.ds.res[0])
        self.name = source_name

    def read_dem_window(self, bounds, target_crs, resolution=None, halo_m: float = 0.0, resampling="average",
                        snap: bool = True) -> DEMWindow:
        from rasterio.windows import from_bounds

        x0, y0, x1, y1 = bounds[0] - halo_m, bounds[1] - halo_m, bounds[2] + halo_m, bounds[3] + halo_m
        win = from_bounds(x0, y0, x1, y1, self.ds.transform).round_offsets().round_lengths()
        z = self.ds.read(1, window=win, boundless=True, fill_value=np.nan, masked=False).astype(np.float32)
        t = self.ds.window_transform(win)
        v = np.isfinite(z)
        return DEMWindow(z, tuple(t)[:6], str(target_crs), v, np.where(v, 0, -1).astype(np.int16), [self.name],
                         self.res, np.zeros(z.shape, bool), [], int(round(halo_m / self.res)))


# ------------------------------------------------------------------------------------- 100 m aggregation
BASE_STATS = ("elev_mean", "elev_median", "elev_min", "elev_max", "elev_std", "elev_p05", "elev_p95", "local_relief",
              "slope_mean", "slope_p95", "aspect_c_mean", "aspect_s_mean", "aspect_resultant", "roughness_mean",
              "profile_curv_mean", "profile_curv_std", "plan_curv_mean", "tri_mean", "tpi_mean", "entropy",
              "valid_fraction", "seam_fraction")


def _blocks(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape
    return a[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, f * f)


def block_stats(z: np.ndarray, valid: np.ndarray, layers: dict, f: int, min_support: float = 0.25) -> dict:
    """Exact per-cell statistics of ``f x f`` pixel blocks (``z`` and ``layers`` already on a nested grid)."""
    import warnings

    zb = _blocks(np.where(valid, z, np.nan), f)
    vf = _blocks(valid.astype(np.float32), f).mean(-1)
    ok = vf >= min_support
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(zb, [5, 50, 95], axis=-1)
        out.update(elev_mean=np.nanmean(zb, -1), elev_median=q[1], elev_min=np.nanmin(zb, -1), elev_max=np.nanmax(zb, -1),
                   elev_std=np.nanstd(zb, -1), elev_p05=q[0], elev_p95=q[2])
        out["local_relief"] = out["elev_max"] - out["elev_min"]
        sb = _blocks(layers["slope"], f)
        out["slope_mean"], out["slope_p95"] = np.nanmean(sb, -1), np.nanpercentile(sb, 95, axis=-1)
        ac, as_ = np.nanmean(_blocks(layers["aspect_c"], f), -1), np.nanmean(_blocks(layers["aspect_s"], f), -1)
        out["aspect_c_mean"], out["aspect_s_mean"], out["aspect_resultant"] = ac, as_, np.hypot(ac, as_)
        out["roughness_mean"] = np.nanmean(_blocks(layers["roughness"], f), -1)
        pc = _blocks(layers["profile_curvature"], f)
        out["profile_curv_mean"], out["profile_curv_std"] = np.nanmean(pc, -1), np.nanstd(pc, -1)
        out["plan_curv_mean"] = np.nanmean(_blocks(layers["plan_curvature"], f), -1)
        out["tri_mean"] = np.nanmean(_blocks(layers["tri"], f), -1)
        out["tpi_mean"] = np.nanmean(_blocks(layers["tpi"], f), -1)
        # elevation entropy: Shannon entropy (bits) of a 16-bin histogram within the cell
        lo, hi = out["elev_min"][..., None], out["elev_max"][..., None]
        bins = np.clip(((zb - lo) / np.maximum(hi - lo, 1e-6) * 15.999).astype(np.float32), 0, 15)
        ent = np.zeros(vf.shape, np.float32)
        for k in range(16):
            pk = np.nanmean(np.where(np.isfinite(zb), (np.floor(bins) == k).astype(np.float32), np.nan), -1)
            ent -= np.where(pk > 0, pk * np.log2(np.maximum(pk, 1e-12)), 0)
        out["entropy"] = ent
    out["valid_fraction"] = vf
    if "seam" in layers:
        out["seam_fraction"] = _blocks(layers["seam"].astype(np.float32), f).mean(-1)
    for k in out:
        if k not in ("valid_fraction", "seam_fraction"):
            out[k] = np.where(ok, out[k], np.nan).astype(np.float32)
    return out


def dem_stat_names(cfg: dict) -> list[str]:
    """Names (without the ``dem100_`` prefix) of every DEM 100 m statistic the configuration produces."""
    ctx_scales = [float(s) for s in cfg["dem"]["context_scales_m"]]
    return list(BASE_STATS) + [f"{k}_ctx{int(s)}m" for s in ctx_scales for k in ("relief", "roughness", "tpi", "slope")]


def stats_factor(grid, res: float) -> int:
    cell = grid.res[0]
    f = int(round(cell / res))
    if f < 1 or abs(f * res - cell) > 1e-6:
        raise DEMWindowError(f"DEM statistics resolution {res} m does not divide the {cell} m master cell")
    return f


def halo_cells_for(cfg: dict, grid) -> int:
    cell = grid.res[0]
    return int(np.ceil(max([float(s) for s in cfg["dem"]["context_scales_m"]] + [cell]) * 2 / cell))


def dem_chunk_stats(reader, grid, r0: int, r1: int, q0: int, q1: int, cfg: dict, res: float) -> dict | None:
    """DEM statistics of the master cells [r0:r1, q0:q1] (dict name -> float32 (r1-r0, q1-q0)), or None when the
    window holds no valid DEM pixel. The window carries a halo of the largest context scale, so the values of a chunk
    do not depend on how the grid is cut into chunks."""
    import warnings

    from .features.terrain import compute_terrain_features, curvatures, derivatives_m, slope_aspect

    from gems_bench.g06.feature_physics import masked_smooth

    f = stats_factor(grid, res)
    ctx_scales = [float(s) for s in cfg["dem"]["context_scales_m"]]
    halo = halo_cells_for(cfg, grid)
    H, W = grid.shape
    a, _, c0, _, e, f0 = grid.transform[:6]
    hr0, hq0 = max(r0 - halo, 0), max(q0 - halo, 0)
    hr1, hq1 = min(r1 + halo, H), min(q1 + halo, W)
    bounds = (c0 + a * hq0, f0 + e * hr1, c0 + a * hq1, f0 + e * hr0)
    win = reader.read_dem_window(bounds, grid.crs, resolution=res, resampling="average", snap=False)
    try:
        reader._last_tiles = list(win.source_tiles)          # provenance for the chunk sidecar
    except AttributeError:
        pass
    z, v = win.elevation, win.valid
    if not v.any():
        return None
    S, _ = masked_smooth(z, 0.5, 0.5)
    d = derivatives_m(S, res)
    slope, aspect = slope_aspect(d)
    cv = curvatures(d)
    S3, _ = masked_smooth(z, 2.5, 0.5)
    layers = {"slope": np.where(v, slope, np.nan), "aspect_c": np.where(v, np.cos(np.radians(aspect)), np.nan),
              "aspect_s": np.where(v, np.sin(np.radians(aspect)), np.nan),
              "roughness": np.where(v, np.abs(z - S3), np.nan), "tri": np.where(v, np.abs(z - S), np.nan),
              "tpi": np.where(v, z - S3, np.nan), "profile_curvature": np.where(v, cv["profile_curvature"], np.nan),
              "plan_curvature": np.where(v, cv["plan_curvature"], np.nan), "seam": win.seam_mask}
    nh, nw = (hr1 - hr0) * f, (hq1 - hq0) * f
    z, v = z[:nh, :nw], v[:nh, :nw]
    layers = {k: x[:nh, :nw] for k, x in layers.items()}
    st = block_stats(z, v, layers, f)
    ctx = compute_terrain_features(z, v, res, ctx_scales, cfg, families=("first_order",)).arrays
    for s in ctx_scales:
        k = f"{int(s)}m"
        for nm, src in (("relief", f"relief_{k}"), ("roughness", f"roughness_{k}"), ("tpi", f"tpi_{k}"),
                        ("slope", f"slope_{k}")):
            if src in ctx:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    st[f"{nm}_ctx{int(s)}m"] = np.nanmean(_blocks(ctx[src], f), -1).astype(np.float32)
    ir0, iq0 = r0 - hr0, q0 - hq0
    out = {}
    for n in dem_stat_names(cfg):
        blk = np.full((r1 - r0, q1 - q0), np.nan, np.float32)
        if n in st:
            src = st[n][ir0:ir0 + (r1 - r0), iq0:iq0 + (q1 - q0)]
            blk[: src.shape[0], : src.shape[1]] = src
        out[n] = blk
    return out


def aggregate_dem_to_master(reader: DEMReader, grid, covered: np.ndarray, cfg: dict, store=None, progress=print,
                            chunk_cells: int | None = None) -> dict:
    """Per-100 m-cell DEM statistics + context statistics at ``dem.context_scales_m`` (in-memory, full grid).

    Work resolution ``dem.stats_resolution_m``; windows nest exactly in master cells (f = 100 / res pixels per cell
    side). Chunks of ``chunk_cells`` cells with a halo of the largest context scale; only chunks containing covered
    cells are read. Results are full-grid float32 arrays (NaN outside DEM coverage); with ``store`` they are written
    to the feature store. The resumable, chunk-persistent version is :mod:`gems_bench.g11.dem_engine`.
    """
    res = float(cfg["dem"]["stats_resolution_m"])
    stats_factor(grid, res)
    C = int(chunk_cells or cfg["dem"]["aggregate_chunk_cells"])
    H, W = grid.shape
    names = dem_stat_names(cfg)
    full = {n: np.full((H, W), np.nan, np.float32) for n in names}
    n_done = 0
    for r0 in range(0, H, C):
        for q0 in range(0, W, C):
            r1, q1 = min(r0 + C, H), min(q0 + C, W)
            if not covered[r0:r1, q0:q1].any():
                continue
            st = dem_chunk_stats(reader, grid, r0, r1, q0, q1, cfg, res)
            if st is None:
                continue
            for n in names:
                full[n][r0:r1, q0:q1] = st[n]
            n_done += 1
            if progress and n_done % 10 == 0:
                progress(f"[dem-aggregate] {n_done} chunks")
    for n in names:
        full[n][~covered] = np.nan
        if store is not None:
            store.put(f"dem100_{n}", full[n], family="dem_100m", scale_m=100.0, source="dem_1m aggregation")
    return full
