"""High-resolution DEM access: catalog, tiled native-resolution reads with halo, exact aggregation to 100 m.

* Catalog = the USGS TNM URL list (``gems_bench.dem.inventory.load_inventory``: tile id, project, year, nominal
  footprint) merged with the DEM GeoTIFFs actually present under the local / Drive roots, whose HEADERS give CRS,
  resolution, bounds, nodata and the vertical unit. A catalog-only tile is never treated as available data.
* Reads: a work-grid window (core + halo) is read from the native raster (``boundless`` semantics through clipping +
  NaN padding), nodata -> NaN, vertical unit -> metres, optionally block-averaged by an integer ``factor`` (NaN-aware;
  a work pixel needs >= 50 % valid native pixels). Halos come from the same source; with ``build_vrt`` (GDAL VRT of
  the tiles of one CRS / resolution) windows cross tile boundaries seamlessly; without it a tile edge is a data edge
  (NaN halo - features there lose support instead of inventing values).
* Aggregation: features computed at the work resolution are aggregated onto the 100 m analysis grid with
  ``rasterio.warp.reproject`` - ``sum`` of (value x valid) and of valid, i.e. exact area-weighted means even for
  cells straddling blocks, tiles or the NAD83 / WGS84 datum difference; ``max`` for extreme-value bands; doubled-angle
  weighted means for orientations. Per-cell DEM coverage is always written. The raw DEM is never downsampled to 100 m
  before the features are computed.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ContractError, DEMSource, GridSpec, validate_dem_source, vertical_scale

MEAN, MAX, SHARE, ORIENT = "mean", "max", "share", "orientation"


# =============================================================================== catalog
def read_header(path) -> dict:
    """CRS / resolution / bounds / dtype / nodata / vertical unit of one local DEM (header only)."""
    import rasterio

    with rasterio.open(path) as ds:
        units = (ds.units[0] if getattr(ds, "units", None) else "") or ""
        tags = {**ds.tags(), **ds.tags(1)}
        unit = units or tags.get("UNITTYPE") or tags.get("units") or tags.get("VERTICAL_UNITS") or ""
        if not unit and ds.crs is not None:
            try:
                unit = ds.crs.linear_units or ""             # horizontal unit; recorded only as a hint below
                unit = f"hint:{unit}"
            except Exception:
                unit = ""
        return {"path": str(path), "crs": ds.crs.to_string() if ds.crs else None, "res_m": float(abs(ds.res[0])),
                "res_y_m": float(abs(ds.res[1])), "width": ds.width, "height": ds.height, "dtype": ds.dtypes[0],
                "nodata": ds.nodata, "west": ds.bounds.left, "south": ds.bounds.bottom, "east": ds.bounds.right,
                "north": ds.bounds.top, "vertical_unit_raw": unit}


def local_dem_files(roots, subdirs) -> list[Path]:
    out = []
    for r in roots:
        for s in subdirs:
            out += [Path(p) for p in glob.glob(str(Path(r) / s / "**" / "*.tif"), recursive=True)]
    seen, uniq = set(), []
    for p in out:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return sorted(uniq)


def open_dem_catalog(cfg: dict, roots) -> tuple[pd.DataFrame, list[DEMSource]]:
    """(catalog table, DEMSource list). ``cfg`` = the ``dem`` section."""
    from ..dem.inventory import load_inventory, parse_url

    cat = cfg.get("catalog") or {}
    inv = pd.DataFrame()
    for r in roots:
        for c in cat.get("url_lists", []):
            p = Path(r) / c
            if p.exists():
                inv = load_inventory(p)
                inv["catalog_file"] = str(p)
                break
        if len(inv):
            break
    files = local_dem_files(roots, cat.get("local_dirs", ["data/raw/dem_1m"]))
    override = cfg.get("vertical_unit_override")
    sources, rows = [], []
    by_file = {Path(str(f)).name: f for f in files}
    names = set(inv.file) if len(inv) and "file" in inv else set()
    for name in sorted(names | set(by_file)):
        meta = parse_url(name)
        r = inv[inv.file == name].iloc[0].to_dict() if name in names else {}
        path = by_file.get(name)
        h = {}
        if path is not None:
            try:
                h = read_header(path)
            except Exception as e:
                h = {"error": f"{type(e).__name__}: {e}"}
        vu = h.get("vertical_unit_raw", "")
        if (not vu or vu.startswith("hint:")) and override:
            vu = str(override)
            vu_note = "ASSUMED via dem.vertical_unit_override (header has no vertical unit)"
        else:
            vu_note = "header"
        src = DEMSource(meta.get("tile_id") or Path(name).stem, r.get("url"), str(path) if path else None,
                        str(meta.get("project") or r.get("project") or ""), meta.get("year"), h.get("crs"),
                        h.get("res_m"), (h["west"], h["south"], h["east"], h["north"]) if "west" in h else
                        ((meta["west"], meta["south"], meta["east"], meta["north"]) if meta.get("parse_ok") else None),
                        vu or "unknown", h.get("nodata"), h.get("dtype"),
                        "LOCAL" if path is not None and "error" not in h else ("LOCAL_INVALID" if path else
                                                                                "CATALOG_ONLY"))
        if path is not None:
            src.problems = tuple(validate_dem_source(src, float(cfg.get("target_resolution_m", 1.0)),
                                                     float(cfg.get("max_accepted_res_m", 3.0))) +
                                 ([h["error"]] if "error" in h else []))
            if src.problems and src.status == "LOCAL" and any("resolution" in x and "coarser" in x or "CRS" in x
                                                              for x in src.problems):
                src.status = "LOCAL_INVALID"
        sources.append(src)
        rows.append({**src.to_row(), "vertical_unit_source": vu_note if path else "",
                     "bounds_source": "header" if "west" in h else "tile name (nominal)",
                     "size_bytes": r.get("size_bytes"), "in_catalog": name in names, "zone": meta.get("zone")})
    return pd.DataFrame(rows), sources


def build_vrt(paths, out_path) -> Path | None:
    """GDAL VRT of tiles sharing CRS / resolution (seamless halos across tile edges); None if GDAL python is
    unavailable (per-tile processing then)."""
    try:
        from osgeo import gdal
    except ImportError:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vrt = gdal.BuildVRT(str(out_path), [str(p) for p in paths], resampleAlg="nearest")
    if vrt is None:
        return None
    vrt.FlushCache()
    vrt = None
    return out_path


def group_sources(sources: list[DEMSource]) -> dict:
    """{(crs, res): [paths]} of usable local sources (one VRT / processing group each)."""
    g: dict = {}
    for s in sources:
        if s.status == "LOCAL" and s.path:
            g.setdefault((s.crs, round(float(s.res_m), 6)), []).append(s.path)
    return g


# =============================================================================== windows
def work_shape(ds, factor: int) -> tuple[int, int]:
    return (int(np.ceil(ds.height / factor)), int(np.ceil(ds.width / factor)))


def iter_dem_tiles(ds, tile_px: int, halo_px: int, factor: int = 1):
    """Work-grid windows (``g06.raster_io.TileWindow``) covering a DEM dataset."""
    from ..g06.raster_io import iter_windows

    return iter_windows(work_shape(ds, factor), tile_px, halo_px)


def work_transform(ds, factor: int):
    from rasterio.transform import Affine

    T = ds.transform
    return Affine(T.a * factor, T.b, T.c, T.d, T.e * factor, T.f)


def block_mean(a: np.ndarray, f: int, min_valid: float = 0.5) -> np.ndarray:
    """NaN-aware f x f block average (array dims must be multiples of f)."""
    if f <= 1:
        return a
    h, w = a.shape[0] // f, a.shape[1] // f
    b = a[:h * f, :w * f].reshape(h, f, w, f)
    m = np.isfinite(b)
    s = np.where(m, b, 0.0).sum((1, 3))
    n = m.sum((1, 3))
    return np.where(n >= min_valid * f * f, s / np.maximum(n, 1), np.nan).astype(np.float32)


def read_dem_window(ds, win, factor: int = 1, vscale: float = 1.0, nodata=None) -> np.ndarray:
    """Padded work-resolution window (core + halo) in metres, NaN for nodata and beyond the raster."""
    from rasterio.windows import Window

    f = int(factor)
    r0n, r1n = (win.row0 - win.halo) * f, (win.row1 + win.halo) * f
    c0n, c1n = (win.col0 - win.halo) * f, (win.col1 + win.halo) * f
    rr0, rr1, cc0, cc1 = max(r0n, 0), min(r1n, ds.height), max(c0n, 0), min(c1n, ds.width)
    out = np.full((r1n - r0n, c1n - c0n), np.nan, np.float32)
    if rr1 > rr0 and cc1 > cc0:
        a = ds.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0), masked=True)
        a = np.ma.filled(a.astype(np.float32), np.nan)
        nd = nodata if nodata is not None else ds.nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
        a[~np.isfinite(a) | (np.abs(a) > 1e20)] = np.nan
        out[rr0 - r0n:rr1 - r0n, cc0 - c0n:cc1 - c0n] = a * float(vscale)
    return block_mean(out, f)


def core_transform(ds, win, factor: int):
    """Affine of the CORE of a work window."""
    from rasterio.transform import Affine

    T = work_transform(ds, factor)
    return Affine(T.a, T.b, T.c + win.col0 * T.a, T.d, T.e, T.f + win.row0 * T.e)


# =============================================================================== aggregation
def analysis_window(bounds, src_crs, grid: GridSpec, pad: int = 1) -> tuple[int, int, int, int] | None:
    """Analysis-grid window (r0, r1, c0, c1) covering ``bounds`` (west, south, east, north in ``src_crs``)."""
    from rasterio.transform import rowcol
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds(src_crs, grid.crs, *bounds)
    T = grid.affine()
    r0, c0 = rowcol(T, w, n)
    r1, c1 = rowcol(T, e, s)
    r0, c0 = max(min(r0, r1) - pad, 0), max(min(c0, c1) - pad, 0)
    r1, c1 = min(max(r0, r1) + pad + 1, grid.height), min(max(c0, c1) + pad + 1, grid.width)
    if r1 <= r0 or c1 <= c0:
        return None
    return int(r0), int(r1), int(c0), int(c1)


class Accumulator:
    """Area-weighted aggregation of work-resolution feature blocks onto one analysis-grid window."""

    def __init__(self, grid: GridSpec, window: tuple, kinds: dict):
        from rasterio.transform import Affine

        self.grid, self.window, self.kinds = grid, window, kinds
        r0, r1, c0, c1 = window
        T = grid.affine()
        self.T = Affine(T.a, T.b, T.c + c0 * T.a, T.d, T.e, T.f + r0 * T.e)
        self.shape = (r1 - r0, c1 - c0)
        self.sum: dict = {}
        self.max: dict = {}
        self.count = np.zeros(self.shape, np.float64)
        self.native_px = 0

    def add(self, feats: dict, valid: np.ndarray, src_T, src_crs) -> None:
        from rasterio.warp import Resampling, reproject

        v = np.asarray(valid, np.float32)
        if not v.any():
            return
        self.native_px += int(v.sum())
        cnt = np.zeros(self.shape, np.float64)
        reproject(v, cnt, src_transform=src_T, src_crs=src_crs, dst_transform=self.T, dst_crs=self.grid.crs,
                  resampling=Resampling.sum)
        self.count += cnt
        for k, a in feats.items():
            kind = self.kinds.get(k, MEAN)
            a = np.asarray(a, np.float32)
            if kind == MAX:
                dst = np.full(self.shape, np.nan, np.float32)
                reproject(np.where(v > 0, a, np.nan).astype(np.float32), dst, src_transform=src_T, src_crs=src_crs,
                          dst_transform=self.T, dst_crs=self.grid.crs, src_nodata=np.nan, dst_nodata=np.nan,
                          resampling=Resampling.max)
                self.max[k] = np.fmax(self.max.get(k, np.full(self.shape, np.nan, np.float32)), dst)
                continue
            if kind == ORIENT:                      # a = doubled angle (radians), weight in feats[k + "__w"]
                w = np.asarray(feats.get(k + "__w", np.ones_like(a)), np.float32)
                for part, val in (("c", np.cos(a) * w), ("s", np.sin(a) * w), ("w", w)):
                    self._sum(f"{k}__{part}", np.where(v > 0, val, 0.0), src_T, src_crs)
                continue
            if k.endswith("__w"):
                continue
            self._sum(k, np.where((v > 0) & np.isfinite(a), a, 0.0), src_T, src_crs)
            self._sum(k + "__n", ((v > 0) & np.isfinite(a)).astype(np.float32), src_T, src_crs)

    def _sum(self, k, a, src_T, src_crs):
        from rasterio.warp import Resampling, reproject

        dst = np.zeros(self.shape, np.float64)
        reproject(np.asarray(a, np.float32), dst, src_transform=src_T, src_crs=src_crs, dst_transform=self.T,
                  dst_crs=self.grid.crs, resampling=Resampling.sum)
        self.sum[k] = self.sum.get(k, 0) + dst

    def finalize(self, native_res_m: float) -> dict:
        """{band: analysis-window array} + ``dem_coverage`` (share of the 100 m cell with valid DEM)."""
        out = {}
        cell = (self.grid.pixel_m / float(native_res_m)) ** 2
        out["dem_coverage"] = np.clip(self.count / max(cell, 1.0), 0, 1).astype(np.float32)
        has = self.count > 0
        for k, kind in self.kinds.items():
            if k.endswith("__w"):
                continue
            if kind == MAX:
                out[k] = self.max.get(k, np.full(self.shape, np.nan, np.float32))
            elif kind == ORIENT:
                c, s, w = (self.sum.get(f"{k}__{p}") for p in ("c", "s", "w"))
                if c is None:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    cc, ss = c / np.maximum(w, 1e-12), s / np.maximum(w, 1e-12)
                    r = np.hypot(cc, ss)
                    out[f"{k}_cos2"] = np.where(has, cc / np.maximum(r, 1e-12), np.nan).astype(np.float32)
                    out[f"{k}_sin2"] = np.where(has, ss / np.maximum(r, 1e-12), np.nan).astype(np.float32)
                    out[f"{k}_resultant"] = np.where(has, r, np.nan).astype(np.float32)
            else:
                s, n = self.sum.get(k), self.sum.get(k + "__n")
                if s is None:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[k] = np.where(n > 0, s / np.maximum(n, 1e-12), np.nan).astype(np.float32)
        return out


def aggregate_dem_feature(acc: Accumulator, feats: dict, valid: np.ndarray, src_T, src_crs) -> None:
    acc.add(feats, valid, src_T, src_crs)


def align_dem_to_analysis_grid(feats: dict, valid: np.ndarray, src_T, src_crs, grid: GridSpec, kinds: dict,
                               native_res_m: float) -> tuple[tuple, dict] | None:
    """One block of work-resolution features -> (analysis window, aggregated arrays)."""
    h, w = np.shape(valid)
    west, north = src_T.c, src_T.f
    east, south = west + w * src_T.a, north + h * src_T.e
    win = analysis_window((west, south, east, north), src_crs, grid)
    if win is None:
        return None
    acc = Accumulator(grid, win, kinds)
    acc.add(feats, valid, src_T, src_crs)
    return win, acc.finalize(native_res_m)


def validate_dem_tile(src: DEMSource, cfg: dict) -> list[str]:
    return validate_dem_source(src, float(cfg.get("target_resolution_m", 1.0)), float(cfg.get("max_accepted_res_m", 3.0)))


def source_vscale(src: DEMSource) -> float:
    try:
        return vertical_scale(src.vertical_unit)
    except ContractError:
        raise


# =============================================================================== per-tile products + mosaic
def write_window(path, arrays: dict, names: list[str], grid: GridSpec, window: tuple, meta: dict) -> Path:
    import rasterio
    from rasterio.transform import Affine

    from .persistence import atomic_save, write_json

    r0, r1, c0, c1 = window
    T = grid.affine()
    prof = {"driver": "GTiff", "height": r1 - r0, "width": c1 - c0, "count": len(names), "dtype": "float32",
            "crs": grid.crs, "transform": Affine(T.a, T.b, T.c + c0 * T.a, T.d, T.e, T.f + r0 * T.e),
            "nodata": np.nan, "compress": "deflate", "predictor": 3}

    def w(t):
        with rasterio.open(t, "w", **prof) as ds:
            for i, n in enumerate(names, 1):
                a = arrays.get(n)
                ds.write(np.full((r1 - r0, c1 - c0), np.nan, np.float32) if a is None else np.asarray(a, np.float32), i)
                ds.set_band_description(i, n)

    atomic_save(Path(path), w)
    write_json(Path(path).with_suffix(".json"), {**meta, "window": list(window), "names": names})
    return Path(path)


def mosaic_windows(paths, grid: GridSpec, names: list[str], out_path, max_bands=()) -> Path:
    """Coverage-weighted merge of per-tile analysis windows onto the full grid, one band at a time."""
    import rasterio

    H, W = grid.shape
    metas = []
    for p in paths:
        with rasterio.open(p) as ds:
            T = ds.transform
            r0 = int(round((T.f - grid.transform[5]) / grid.transform[4]))
            c0 = int(round((T.c - grid.transform[2]) / grid.transform[0]))
            metas.append((p, r0, c0, ds.height, ds.width, list(ds.descriptions)))
    cov_full = np.zeros((H, W), np.float64)
    for p, r0, c0, h, w, desc in metas:
        with rasterio.open(p) as ds:
            cov_full[r0:r0 + h, c0:c0 + w] += np.nan_to_num(ds.read(desc.index("dem_coverage") + 1))
    from ..g06.raster_io import geotiff_profile

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    try:
        with rasterio.open(tmp, "w", **geotiff_profile(grid, len(names))) as dst:      # one full band in memory
            for i, n in enumerate(names, 1):
                if n == "dem_coverage":
                    band = np.clip(cov_full, 0, 1)
                else:
                    band = np.zeros((H, W), np.float64) if n not in max_bands else np.full((H, W), np.nan)
                    for p, r0, c0, h, w, desc in metas:
                        if n not in desc:
                            continue
                        with rasterio.open(p) as ds:
                            a = ds.read(desc.index(n) + 1).astype(np.float64)
                            cv = np.nan_to_num(ds.read(desc.index("dem_coverage") + 1))
                        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
                        if n in max_bands:
                            band[sl] = np.fmax(band[sl], a)
                        else:
                            band[sl] += np.nan_to_num(a) * cv
                    if n not in max_bands:
                        band = np.where(cov_full > 0, band / np.maximum(cov_full, 1e-12), np.nan)
                dst.write(band.astype(np.float32), i)
                dst.set_band_description(i, n)
            dst.update_tags(G07="dem_features_100m")
        import os

        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


def coverage_from_bounds(sources: list[DEMSource], grid: GridSpec, local_only: bool = True) -> np.ndarray:
    """Approximate per-cell coverage (0 / 1) of DEM sources from their bounds (catalog or header)."""
    out = np.zeros(grid.shape, bool)
    for s in sources:
        if s.bounds is None or (local_only and s.status != "LOCAL"):
            continue
        crs = s.crs or "EPSG:26911"
        win = analysis_window(s.bounds, crs, grid, pad=0)
        if win is not None:
            r0, r1, c0, c1 = win
            out[r0:r1, c0:c1] = True
    return out


def json_safe(d: dict) -> dict:
    return json.loads(json.dumps(d, default=str))


# =============================================================================== stores (FeatureStore interface)
class DemFeatureStore:
    """Read-only view of ``dem_features_100m.tif`` with the ``g06.feature_store.FeatureStore`` interface used by the
    G06 sampling / prediction code (names, schema_hash, grid, footprint, load, sample, by_family, usable)."""

    def __init__(self, path, min_coverage: float = 0.5, fingerprint: str = ""):
        import rasterio

        from ..g06.persistence import stable_hash

        self.path_ = Path(path)
        with rasterio.open(self.path_) as ds:
            self.names = list(ds.descriptions)
            self.grid = GridSpec.from_dataset(ds)
        self.min_coverage = float(min_coverage)
        self.schema_hash = stable_hash("dem", self.names, fingerprint or str(self.path_.stat().st_size))
        self._idx = {n: i + 1 for i, n in enumerate(self.names)}
        self.registry = pd.DataFrame({"feature_id": self.names, "family": ["qc" if n == "dem_coverage" else "dem"
                                                                           for n in self.names],
                                      "permitted_tracks": "A,B", "usable": [n != "dem_coverage" for n in self.names]})

    def path(self, group=None) -> Path:
        return self.path_

    def coverage(self, window=None) -> np.ndarray:
        return self.load(["dem_coverage"], window)[0]

    def footprint(self) -> np.ndarray:
        return np.nan_to_num(self.coverage()) >= self.min_coverage

    def usable(self, names=None) -> list[str]:
        return [n for n in (names or self.names) if n != "dem_coverage"]

    def by_family(self, families, track="A", usable_only=True) -> list[str]:
        return self.usable() if "dem" in families else []

    def load(self, names, window=None) -> np.ndarray:
        from ..g06.raster_io import read_window_bands

        h, w = self.grid.shape
        return read_window_bands(self.path_, [self._idx[n] for n in names], window or (0, h, 0, w))

    def sample(self, names, rows, cols) -> np.ndarray:
        from ..g06.raster_io import read_points

        return read_points(self.path_, [self._idx[n] for n in names], rows, cols)
