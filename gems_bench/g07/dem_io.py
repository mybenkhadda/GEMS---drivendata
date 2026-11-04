"""High-resolution DEM access: catalog, tiled native-resolution reads with halo, exact aggregation to 100 m.

* Catalog = the USGS TNM URL list (``gems_bench.dem.inventory.load_inventory``: tile id, project, year, nominal
  footprint) merged with the DEM GeoTIFFs actually present under the local / Drive roots, whose HEADERS give CRS,
  resolution, bounds, nodata and the vertical unit. A catalog-only tile is never treated as available data.
* Reads: a work-grid window (core + halo) is read from the native raster (``boundless`` semantics through clipping +
  NaN padding), nodata -> NaN, vertical unit -> metres, optionally block-averaged by an integer ``factor`` (NaN-aware;
  a work pixel needs >= 50 % valid native pixels). Missing halo pixels are filled from grid-aligned neighbouring
  tiles (``neighbours``), so a window crosses tile boundaries seamlessly; a real data edge stays NaN and the features
  within the edge guard of it become NaN (``dem_features``) instead of inventing a lineament along the tile edge.
* Aggregation: features computed at the work resolution are aggregated onto the 100 m analysis grid with
  ``rasterio.warp.reproject`` - ``sum`` of (value x valid) and of valid, i.e. exact area-weighted means even for
  cells straddling blocks, tiles or the NAD83 / WGS84 datum difference; ``max`` for extreme-value bands; doubled-angle
  weighted means for orientations. Per-cell DEM coverage is always written. The raw DEM is never downsampled to 100 m
  before the features are computed.
"""
from __future__ import annotations

import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (CATALOG_ONLY, LOCAL, LOCAL_INVALID, LOCAL_OUTSIDE, UNKNOWN_VERTICAL_UNIT, ContractError,
                        DEMSource, GridSpec, dem_source_issues, validate_dem_source, vertical_scale)

MEAN, MAX, SHARE, ORIENT = "mean", "max", "share", "orientation"


# =============================================================================== catalog
def vertical_unit_from_wkt(wkt: str | None) -> str:
    """Unit of the VERTICAL component of a compound CRS WKT ('' when there is none). The horizontal unit is never
    returned: it says nothing about the elevation values."""
    m = re.search(r'(?:VERT_CS|VERTCRS|VERTICALCRS)\[.*?(?:LENGTHUNIT|UNIT)\["([^"]+)"', str(wkt or ""), re.S)
    return m.group(1) if m else ""


def vertical_unit_from_sidecar(path) -> str:
    """``<altunits>`` of the FGDC XML that USGS publishes next to a tile ('' when absent)."""
    p = Path(path).with_suffix(".xml")
    if not p.exists():
        return ""
    try:
        m = re.search(r"<altunits>\s*([^<]+?)\s*</altunits>", p.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return ""
    return m.group(1) if m else ""


def _vertical_unit(ds, path) -> tuple[str, str]:
    """(unit, where it was stated) from explicit vertical metadata only; ('', '') when nothing states it."""
    band = (ds.units[0] if getattr(ds, "units", None) else "") or ""
    if band:
        return band, "band_units"
    tags = {**ds.tags(), **ds.tags(1)}
    tag = tags.get("UNITTYPE") or tags.get("units") or tags.get("VERTICAL_UNITS") or tags.get("VERTICAL_UNIT") or ""
    if tag:
        return tag, "tag"
    wkt = vertical_unit_from_wkt(ds.crs.to_wkt()) if ds.crs else ""
    if wkt:
        return wkt, "crs_vertical"
    xml = vertical_unit_from_sidecar(path)
    return (xml, "sidecar_xml") if xml else ("", "")


def read_header(path) -> dict:
    """CRS / resolution / bounds / dtype / nodata / explicit vertical unit / validity flags of one local DEM."""
    import rasterio

    with rasterio.open(path) as ds:
        unit, source = _vertical_unit(ds, path)
        T = ds.transform
        finite = all(np.isfinite(x) for x in (T.a, T.b, T.c, T.d, T.e, T.f))
        crs_ok = False
        if ds.crs is not None:
            try:
                crs_ok = bool(ds.crs.is_projected) and abs(float(ds.crs.linear_units_factor[1]) - 1.0) < 1e-9
            except Exception:
                crs_ok = False
        return {"path": str(path), "crs": ds.crs.to_string() if ds.crs else None, "res_m": float(abs(ds.res[0])),
                "res_y_m": float(abs(ds.res[1])), "width": ds.width, "height": ds.height, "dtype": ds.dtypes[0],
                "nodata": ds.nodata, "west": ds.bounds.left, "south": ds.bounds.bottom, "east": ds.bounds.right,
                "north": ds.bounds.top, "vertical_unit_raw": unit, "vertical_unit_source": source,
                "crs_projected_metric": crs_ok,
                "transform_valid": bool(finite and T.a > 0 and T.e < 0 and abs(T.b) < 1e-9 and abs(T.d) < 1e-9
                                        and ds.width > 0 and ds.height > 0)}


def local_dem_files(roots, subdirs) -> list[Path]:
    out = []
    for r in roots:
        for s in subdirs:
            for pat in ("*.tif", "*.tiff"):
                out += [Path(p) for p in glob.glob(str(Path(r) / s / "**" / pat), recursive=True)]
    seen, uniq = set(), []
    for p in out:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return sorted(uniq)


INVENTORY_COLUMNS = ("tile_id", "url", "path", "project", "year", "crs", "res_m", "bounds", "vertical_unit", "nodata",
                     "dtype", "status", "license", "problems", "exclusion_codes", "warnings", "vertical_unit_source",
                     "bounds_source", "size_bytes", "in_catalog", "zone")


def _spatial_issue(bounds, crs, grid: GridSpec | None, footprint) -> tuple[str, str] | None:
    """(code, message) when a source with a valid header does not reach the analysis grid / data footprint."""
    if grid is None or bounds is None:
        return None
    try:
        win = analysis_window(bounds, crs, grid, pad=0)
    except Exception as e:
        return "CRS_UNUSABLE", f"cannot transform the bounds to the analysis CRS: {type(e).__name__}: {e}"
    if win is None:
        return "NO_GRID_INTERSECTION", "the tile lies outside the analysis grid"
    if footprint is not None and not np.asarray(footprint, bool)[win[0]:win[1], win[2]:win[3]].any():
        return "NO_FOOTPRINT_INTERSECTION", "the tile covers no pixel of the GeoDAWN data footprint"
    return None


def open_dem_catalog(cfg: dict, roots, grid: GridSpec | None = None, footprint=None
                     ) -> tuple[pd.DataFrame, list[DEMSource]]:
    """(catalog table, DEMSource list) with one strict status per source (``cfg`` = the ``dem`` section):

    CATALOG_ONLY              listed in the URL list, no local file (never treated as data)
    LOCAL_INVALID             local file excluded: unreadable, CRS / transform / resolution / vertical-unit problem
    LOCAL_OUTSIDE_STUDY_AREA  valid header, but no intersection with the grid / footprint (when these are given)
    LOCAL                     valid and usable: every later stage may rely on it

    Every excluded file carries ``exclusion_codes`` + ``problems``. The vertical unit comes from explicit metadata
    only; ``dem.vertical_unit_override`` applies only where that metadata is silent and is recorded as an assumption.
    """
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
    names = {f for f in inv.file if isinstance(f, str)} if len(inv) and "file" in inv else set()
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
        vu, vu_source = h.get("vertical_unit_raw", ""), h.get("vertical_unit_source", "")
        if not vu and override:
            vu, vu_source = str(override), "override"
        src = DEMSource(meta.get("tile_id") or Path(name).stem, r.get("url"), str(path) if path else None,
                        str(meta.get("project") or r.get("project") or ""), meta.get("year"), h.get("crs"),
                        h.get("res_m"), (h["west"], h["south"], h["east"], h["north"]) if "west" in h else
                        ((meta["west"], meta["south"], meta["east"], meta["north"]) if meta.get("parse_ok") else None),
                        vu or "unknown", h.get("nodata"), h.get("dtype"), CATALOG_ONLY)
        src.vertical_unit_source = vu_source
        if path is not None:
            blocking, src.warnings = dem_source_issues(src, float(cfg.get("target_resolution_m", 1.0)),
                                                       float(cfg.get("max_accepted_res_m", 3.0)), h)
            src.warnings = tuple(src.warnings)
            src.status = LOCAL_INVALID if blocking else LOCAL
            if not blocking:
                sp = _spatial_issue(src.bounds, src.crs, grid, footprint)
                if sp is not None:
                    blocking = [sp]
                    src.status = LOCAL_INVALID if sp[0] == "CRS_UNUSABLE" else LOCAL_OUTSIDE
            src.exclusion_codes = tuple(c for c, _ in blocking)
            src.problems = tuple(m for _, m in blocking)
        sources.append(src)
        rows.append({**src.to_row(), "bounds_source": "header" if "west" in h else "tile name (nominal)",
                     "size_bytes": r.get("size_bytes"), "in_catalog": name in names, "zone": meta.get("zone")})
    return (pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(INVENTORY_COLUMNS))), sources


def source_validation_summary(T: pd.DataFrame, cfg: dict) -> dict:
    """What ``dem_source_validation.json`` records: counts per status / exclusion code and the exact remedy."""
    local = T[T.status != CATALOG_ONLY] if len(T) else T
    codes: dict = {}
    for s in (local.exclusion_codes if len(local) else []):
        for c in str(s).split(";"):
            if c and c != "nan":
                codes[c] = codes.get(c, 0) + 1
    remedy = []
    if codes.get(UNKNOWN_VERTICAL_UNIT):
        remedy.append("Headers of these tiles state no vertical unit. Check the USGS metadata of one tile, then set "
                      "OVERRIDES['dem']['vertical_unit_override'] (e.g. 'metre' for NAVD88 metres). This is a recorded "
                      "assumption, never inferred from the horizontal CRS.")
    if codes.get("RESOLUTION_TOO_COARSE"):
        remedy.append("Coarser files are never used as a high-resolution DEM; remove them from data/raw/dem_1m if "
                      "they are derived products.")
    n_local = int((T.status == LOCAL).sum()) if len(T) else 0
    if not n_local:
        remedy.append("No usable DEM: DEM-dependent stages are BLOCKED_INPUT. Expected location: "
                      f"{(cfg.get('catalog') or {}).get('local_dirs')} under the Drive root.")
    return {"n_sources": int(len(T)), "n_files_present": int(len(local)), "n_usable_local": n_local,
            "by_status": {k: int(v) for k, v in T.status.value_counts().items()} if len(T) else {},
            "exclusion_codes": codes, "vertical_unit_override": cfg.get("vertical_unit_override"),
            "vertical_unit_sources": {str(k): int(v) for k, v in local.vertical_unit_source.value_counts().items()}
            if len(local) else {}, "remedy": remedy}


def neighbour_sources(src: DEMSource, sources: list[DEMSource], pad_m: float) -> list[DEMSource]:
    """Usable sources of the same CRS / resolution within ``pad_m`` of ``src``: they feed its halo across tile edges."""
    w, s, e, n = src.bounds
    out = []
    for o in sources:
        if o.tile_id == src.tile_id or o.status != LOCAL or o.crs != src.crs or o.bounds is None:
            continue
        if abs(float(o.res_m) - float(src.res_m)) > 1e-6:
            continue
        ow, os_, oe, on = o.bounds
        if ow < e + pad_m and oe > w - pad_m and os_ < n + pad_m and on > s - pad_m:
            out.append(o)
    return out


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


def _read_clean(ds, window, nodata=None) -> np.ndarray:
    a = ds.read(1, window=window, masked=True)
    a = np.ma.filled(a.astype(np.float32), np.nan)
    nd = nodata if nodata is not None else ds.nodata
    if nd is not None and np.isfinite(nd):
        a[a == nd] = np.nan
    a[~np.isfinite(a) | (np.abs(a) > 1e20)] = np.nan
    return a


def _fill_from_neighbour(out: np.ndarray, x0: float, y0: float, res: float, nds, vscale: float) -> bool:
    """Paste a neighbouring tile into the NaN pixels of ``out`` (native pixels, upper-left corner x0, y0). Only exactly
    aligned grids of the same resolution are used: a resampled neighbour would itself create a seam. False = skipped."""
    from rasterio.windows import Window

    T = nds.transform
    if abs(abs(T.a) - res) > 1e-6:
        return False
    fc, fr = (x0 - T.c) / T.a, (y0 - T.f) / T.e
    c0, r0 = int(round(fc)), int(round(fr))
    if abs(fc - c0) > 0.01 or abs(fr - r0) > 0.01:
        return False
    h, w = out.shape
    a0, a1, b0, b1 = max(r0, 0), min(r0 + h, nds.height), max(c0, 0), min(c0 + w, nds.width)
    if a1 <= a0 or b1 <= b0:
        return True
    sub = out[a0 - r0:a1 - r0, b0 - c0:b1 - c0]            # a view of ``out``
    missing = np.isnan(sub)
    if missing.any():
        a = _read_clean(nds, Window(b0, a0, b1 - b0, a1 - a0))
        sub[missing] = (a * float(vscale))[missing]
    return True


def read_dem_window(ds, win, factor: int = 1, vscale: float = 1.0, nodata=None, neighbours=None) -> np.ndarray:
    """Padded work-resolution window (core + halo) in metres, NaN for nodata and beyond the available data.
    ``neighbours`` = [(open dataset, vertical scale)] of aligned tiles that fill the halo beyond this raster."""
    from rasterio.windows import Window

    f = int(factor)
    r0n, r1n = (win.row0 - win.halo) * f, (win.row1 + win.halo) * f
    c0n, c1n = (win.col0 - win.halo) * f, (win.col1 + win.halo) * f
    rr0, rr1, cc0, cc1 = max(r0n, 0), min(r1n, ds.height), max(c0n, 0), min(c1n, ds.width)
    out = np.full((r1n - r0n, c1n - c0n), np.nan, np.float32)
    if rr1 > rr0 and cc1 > cc0:
        a = _read_clean(ds, Window(cc0, rr0, cc1 - cc0, rr1 - rr0), nodata)
        out[rr0 - r0n:rr1 - r0n, cc0 - c0n:cc1 - c0n] = a * float(vscale)
    if neighbours and np.isnan(out).any():
        T = ds.transform
        for nds, nv in neighbours:
            _fill_from_neighbour(out, T.c + c0n * T.a, T.f + r0n * T.e, abs(T.a), nds, nv)
    return block_mean(out, f)


def core_transform(ds, win, factor: int):
    """Affine of the CORE of a work window."""
    from rasterio.transform import Affine

    T = work_transform(ds, factor)
    return Affine(T.a, T.b, T.c + win.col0 * T.a, T.d, T.e, T.f + win.row0 * T.e)


# =============================================================================== aggregation
def analysis_window(bounds, src_crs, grid: GridSpec, pad: int = 1) -> tuple[int, int, int, int] | None:
    """Half-open analysis-grid window (r0, r1, c0, c1) of every cell touched by ``bounds`` (west, south, east, north in
    ``src_crs``), padded by ``pad`` cells and clipped to the grid; None when the bounds do not reach the grid.

    The raw index range is computed first, the emptiness test uses it (padding never turns an outside tile into a
    window), and padding / clipping are applied independently to each side."""
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds(src_crs, grid.crs, *bounds, densify_pts=21)
    if not all(math.isfinite(x) for x in (w, s, e, n)) or e <= w or n <= s:
        return None
    T = grid.affine()
    rows = sorted(((n - T.f) / T.e, (s - T.f) / T.e))
    cols = sorted(((w - T.c) / T.a, (e - T.c) / T.a))
    raw_r0, raw_r1 = math.floor(rows[0] + 1e-9), math.ceil(rows[1] - 1e-9)
    raw_c0, raw_c1 = math.floor(cols[0] + 1e-9), math.ceil(cols[1] - 1e-9)
    if raw_r1 <= 0 or raw_r0 >= grid.height or raw_c1 <= 0 or raw_c0 >= grid.width:
        return None
    r0, r1 = max(raw_r0 - pad, 0), min(raw_r1 + pad, grid.height)
    c0, c1 = max(raw_c0 - pad, 0), min(raw_c1 + pad, grid.width)
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
        n_valid = int(np.count_nonzero(v > 0))
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
                m = (v > 0) & np.isfinite(a) & np.isfinite(w)          # edge-guarded pixels carry no direction
                for part, val in (("c", np.cos(a) * w), ("s", np.sin(a) * w), ("w", w)):
                    self._sum(f"{k}__{part}", np.where(m, val, 0.0), src_T, src_crs)
                continue
            if k.endswith("__w"):
                continue
            m = (v > 0) & np.isfinite(a)
            self._sum(k, np.where(m, a, 0.0), src_T, src_crs)
            if int(np.count_nonzero(m)) == n_valid:                    # same support as the coverage count
                self.sum[k + "__n"] = self.sum.get(k + "__n", 0) + cnt
            else:
                self._sum(k + "__n", m.astype(np.float32), src_T, src_crs)

    def _sum(self, k, a, src_T, src_crs):
        from rasterio.warp import Resampling, reproject

        dst = np.zeros(self.shape, np.float64)
        reproject(np.asarray(a, np.float32), dst, src_transform=src_T, src_crs=src_crs, dst_transform=self.T,
                  dst_crs=self.grid.crs, resampling=Resampling.sum)
        self.sum[k] = self.sum.get(k, 0) + dst

    def finalize(self, block_res_m: float) -> dict:
        """{band: analysis-window array} + ``dem_coverage`` (share of the 100 m cell with valid DEM).
        ``block_res_m`` = pixel size of the blocks given to ``add`` (the WORK resolution, not the native one: counts
        are work pixels, so a 1 m tile averaged to 2 m has 2500 work pixels per fully covered 100 m cell)."""
        out = {}
        cell = (self.grid.pixel_m / float(block_res_m)) ** 2
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
                    ok = has & (w > 0)
                    out[f"{k}_cos2"] = np.where(ok, cc / np.maximum(r, 1e-12), np.nan).astype(np.float32)
                    out[f"{k}_sin2"] = np.where(ok, ss / np.maximum(r, 1e-12), np.nan).astype(np.float32)
                    out[f"{k}_resultant"] = np.where(ok, r, np.nan).astype(np.float32)
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
                               block_res_m: float) -> tuple[tuple, dict] | None:
    """One block of work-resolution features (pixel size ``block_res_m``) -> (analysis window, aggregated arrays)."""
    h, w = np.shape(valid)
    west, north = src_T.c, src_T.f
    east, south = west + w * src_T.a, north + h * src_T.e
    win = analysis_window((west, south, east, north), src_crs, grid)
    if win is None:
        return None
    acc = Accumulator(grid, win, kinds)
    acc.add(feats, valid, src_T, src_crs)
    return win, acc.finalize(block_res_m)


def coverage_100m_from_dem(path, grid: GridSpec, block_px: int = 4096) -> np.ndarray:
    """Share of every 100 m cell with valid elevation in one DEM (streamed in blocks; used for the merged coarse DEM)."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    out = np.zeros(grid.shape, np.float32)
    with rasterio.open(path) as ds:
        res = float(abs(ds.res[0]))
        T = ds.transform
        for r in range(0, ds.height, block_px):
            for c in range(0, ds.width, block_px):
                h, w = min(block_px, ds.height - r), min(block_px, ds.width - c)
                valid = np.isfinite(_read_clean(ds, Window(c, r, w, h)))
                if not valid.any():
                    continue
                bT = Affine(T.a, T.b, T.c + c * T.a, T.d, T.e, T.f + r * T.e)
                got = align_dem_to_analysis_grid({}, valid, bT, ds.crs, grid, {}, res)
                if got is not None:
                    (r0, r1, c0, c1), arrs = got
                    out[r0:r1, c0:c1] += arrs["dem_coverage"]
    return np.clip(out, 0, 1)


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


def mosaic_windows(paths, grid: GridSpec, names: list[str], out_path, max_bands=(), coverage_paths=None,
                   extra_bands: dict | None = None) -> Path:
    """Merge per-tile analysis windows onto the full grid, one band at a time. Every band is weighted by the coverage
    of the pixels that really carry it (a tile whose band is NaN in a cell does not dilute that cell, so a tile edge
    cannot lower the values along itself). ``dem_coverage`` is summed over ``coverage_paths`` only (default all): the
    fine and coarse tiers describe the same ground and must not add up. ``extra_bands`` = {name: full-grid array}."""
    import rasterio

    H, W = grid.shape
    metas = []
    for p in paths:
        with rasterio.open(p) as ds:
            T = ds.transform
            r0 = int(round((T.f - grid.transform[5]) / grid.transform[4]))
            c0 = int(round((T.c - grid.transform[2]) / grid.transform[0]))
            metas.append((p, r0, c0, ds.height, ds.width, list(ds.descriptions)))
    cov_set = None if coverage_paths is None else {str(p) for p in coverage_paths}
    cov_full = np.zeros((H, W), np.float64)
    for p, r0, c0, h, w, desc in metas:
        if cov_set is not None and str(p) not in cov_set:
            continue
        with rasterio.open(p) as ds:
            cov_full[r0:r0 + h, c0:c0 + w] += np.nan_to_num(ds.read(desc.index("dem_coverage") + 1))
    from ..g06.raster_io import geotiff_profile

    extra = dict(extra_bands or {})
    bands = list(names) + [n for n in extra if n not in names]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    try:
        with rasterio.open(tmp, "w", **geotiff_profile(grid, len(bands))) as dst:      # one full band in memory
            for i, n in enumerate(bands, 1):
                if n in extra:
                    band = np.asarray(extra[n], np.float64)
                elif n == "dem_coverage":
                    band = np.clip(cov_full, 0, 1)
                else:
                    band = np.zeros((H, W), np.float64) if n not in max_bands else np.full((H, W), np.nan)
                    wgt = np.zeros((H, W), np.float64)
                    for p, r0, c0, h, w, desc in metas:
                        if n not in desc:
                            continue
                        with rasterio.open(p) as ds:
                            a = ds.read(desc.index(n) + 1).astype(np.float64)
                            cv = np.nan_to_num(ds.read(desc.index("dem_coverage") + 1)) if n not in max_bands else None
                        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
                        if n in max_bands:
                            band[sl] = np.fmax(band[sl], a)
                        else:
                            ok = np.isfinite(a)
                            band[sl] += np.where(ok, a, 0.0) * cv
                            wgt[sl] += ok * cv
                    if n not in max_bands:
                        band = np.where(wgt > 0, band / np.maximum(wgt, 1e-12), np.nan)
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
        if s.bounds is None or (local_only and s.status != LOCAL):
            continue
        crs = s.crs or "EPSG:26911"
        win = analysis_window(s.bounds, crs, grid, pad=0)
        if win is not None:
            r0, r1, c0, c1 = win
            out[r0:r1, c0:c1] = True
    return out


def seam_audit(store, sources: list[DEMSource], grid: GridSpec, ratio_flag: float = 1.5, max_features: int = 60
               ) -> pd.DataFrame:
    """Per 100 m feature: mean |value| in covered cells crossed by a source boundary vs other covered cells. A ratio far
    from 1 flags a tile-edge artefact (e.g. an elevation offset between two projects). Report only: label-free, no
    feature is altered."""
    edge = np.zeros(grid.shape, bool)
    for s in sources:
        win = analysis_window(s.bounds, s.crs, grid, pad=0) if s.status == LOCAL and s.bounds else None
        if win is None:
            continue
        r0, r1, c0, c1 = win
        edge[r0:min(r0 + 2, r1), c0:c1] = edge[max(r1 - 2, r0):r1, c0:c1] = True
        edge[r0:r1, c0:min(c0 + 2, c1)] = edge[r0:r1, max(c1 - 2, c0):c1] = True
    cov = store.footprint()
    rows = []
    for n in store.usable()[:max_features]:
        a = np.abs(store.load([n])[0])
        e, i = a[edge & cov], a[~edge & cov]
        e, i = e[np.isfinite(e)], i[np.isfinite(i)]
        if not e.size or not i.size:
            continue
        me, mi = float(e.mean()), float(i.mean())
        ratio = me / mi if mi > 0 else np.nan
        rows.append({"feature": n, "n_edge_cells": int(e.size), "mean_abs_edge": me, "mean_abs_inside": mi,
                     "ratio": ratio, "flag": bool(np.isfinite(ratio) and (ratio > ratio_flag or ratio < 1 / ratio_flag))})
    return pd.DataFrame(rows, columns=["feature", "n_edge_cells", "mean_abs_edge", "mean_abs_inside", "ratio", "flag"])


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
        qc = [n.startswith("dem_coverage") for n in self.names]
        self.registry = pd.DataFrame({"feature_id": self.names, "family": ["qc" if q else "dem" for q in qc],
                                      "permitted_tracks": "A,B", "usable": [not q for q in qc]})

    def path(self, group=None) -> Path:
        return self.path_

    def coverage(self, window=None) -> np.ndarray:
        return self.load(["dem_coverage"], window)[0]

    def coverage_for_patches(self) -> np.ndarray:
        """Coverage of the merged coarse DEM the surface patches are read from (fine-tier coverage if it is absent)."""
        return self.load(["dem_coverage_coarse" if "dem_coverage_coarse" in self.names else "dem_coverage"])[0]

    def footprint(self) -> np.ndarray:
        return np.nan_to_num(self.coverage()) >= self.min_coverage

    def usable(self, names=None) -> list[str]:
        return [n for n in (names or self.names) if not n.startswith("dem_coverage")]

    def by_family(self, families, track="A", usable_only=True) -> list[str]:
        return self.usable() if "dem" in families else []

    def load(self, names, window=None) -> np.ndarray:
        from ..g06.raster_io import read_window_bands

        h, w = self.grid.shape
        return read_window_bands(self.path_, [self._idx[n] for n in names], window or (0, h, 0, w))

    def sample(self, names, rows, cols) -> np.ndarray:
        from ..g06.raster_io import read_points

        return read_points(self.path_, [self._idx[n] for n in names], rows, cols)
