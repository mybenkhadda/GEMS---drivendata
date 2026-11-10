"""DEM tile catalog: header metadata, footprints in the master CRS, spatial index and 100 m coverage.

Overlap priority (``dem.priority``) is deterministic and documented: higher valid fraction, newer acquisition year,
finer resolution, then tile id. Vertical units come from the CRS WKT or the XML sidecar (``g07.dem_io``); a tile
with an unknown vertical unit is flagged, never silently assumed metric.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"(19|20)\d{2}")
FEET = {"foot", "us_survey_foot", "ft", "feet", "us survey foot"}


class DEMCatalogError(RuntimeError):
    pass


def _footprint(ds, dst_crs):
    from rasterio.warp import transform_geom
    from shapely.geometry import box, mapping, shape

    b = box(*ds.bounds)
    if ds.crs is None:
        return None
    g = transform_geom(ds.crs, dst_crs, mapping(b), precision=-1)
    return shape(g)


def dem_files(dem_roots) -> tuple[list[Path], list[dict]]:
    """GeoTIFFs below one or several roots, de-duplicated by file name (first root wins; duplicates reported)."""
    roots = [dem_roots] if isinstance(dem_roots, (str, Path)) else list(dem_roots)
    seen, files, dups = {}, [], []
    for r in roots:
        if r is None or not Path(r).is_dir():
            continue
        for p in sorted(Path(r).rglob("*")):
            if p.suffix.lower() not in (".tif", ".tiff") or ".partial" in p.name:
                continue
            if p.name in seen:
                dups.append({"tile": p.name, "kept": str(seen[p.name]), "duplicate": str(p)})
                continue
            seen[p.name] = p
            files.append(p)
    return files, dups


def build_dem_catalog(dem_root, master_crs, cfg: dict, sample_px: int = 250_000, units=None):
    """GeoDataFrame of every DEM GeoTIFF below ``dem_root`` (one path or a list; footprint in ``master_crs``).

    Columns: tile_id, file_path, crs, epsg, bounds, resolution_m, width, height, valid_fraction, year, source_id,
    vertical_unit, vertical_unit_source, vertical_scale, nodata, quality_flags, priority_rank, geometry.
    ``units`` (``dem_units.audit_vertical_units`` table) supplies the vertical-unit class; otherwise it is computed.
    Raises ``DEMCatalogError`` if any tile lacks a CRS (its location cannot be established).
    """
    import geopandas as gpd
    import rasterio

    from .dem_units import CONFLICT, UNKNOWN, VERIFIED_FT, audit_vertical_units

    lo, hi = cfg["dem"]["native_res_range_m"]
    files, dups = dem_files(dem_root)
    if units is None:
        units = audit_vertical_units(files)
    umap = {r.path: r for r in units.itertuples()}
    rows, no_crs = [], []
    for p in files:
        try:
            ds = rasterio.open(p)
        except Exception as e:                                         # noqa: BLE001
            rows.append({"tile_id": p.stem, "file_path": str(p), "quality_flags": f"UNREADABLE:{type(e).__name__}",
                         "geometry": None})
            continue
        with ds:
            if ds.crs is None:
                no_crs.append(str(p))
                continue
            if sample_px:
                f = max(1, int(np.ceil(np.sqrt(ds.width * ds.height / sample_px))))
                a = ds.read(1, out_shape=(max(1, ds.height // f), max(1, ds.width // f)), masked=True)
                vf = float((~np.ma.getmaskarray(a) & np.isfinite(a.filled(np.nan))).mean())
            else:
                vf = np.nan
            u = umap.get(str(p))
            ucls = u[3] if u is not None else UNKNOWN                     # itertuples: (Index, tile_id, path, class, ...)
            unit = {"VERIFIED_METRES": "metre", VERIFIED_FT: "foot"}.get(ucls, ucls.lower())
            usrc = u.statements if u is not None else ""
            res = float(max(ds.res))
            flags = []
            if not lo <= res <= hi:
                flags.append("RES_OUT_OF_RANGE")
            if ucls == UNKNOWN:
                flags.append("UNKNOWN_VERTICAL_UNIT")
            if ucls == CONFLICT:
                flags.append("CONFLICTING_VERTICAL_UNIT")
            if np.isfinite(vf) and vf < 0.05:
                flags.append("LOW_VALID")
            ym = YEAR_RE.findall(p.stem)
            year = int(YEAR_RE.search(p.stem).group(0)) if ym else None
            src = re.sub(r"_?(x\d+y\d+|\d{4}).*$", "", p.stem) or p.parent.name
            rows.append({"tile_id": p.stem, "file_path": str(p), "crs": ds.crs.to_wkt(), "epsg": ds.crs.to_epsg(),
                         "bounds": tuple(ds.bounds), "resolution_m": res, "width": ds.width, "height": ds.height,
                         "valid_fraction": vf, "year": year, "source_id": src, "vertical_unit": unit,
                         "vertical_unit_source": usrc,
                         "vertical_scale": float(u.vertical_scale) if u is not None and u.vertical_scale is not None
                         and np.isfinite(u.vertical_scale) else 1.0,
                         "nodata": ds.nodata, "quality_flags": ";".join(flags),
                         "geometry": _footprint(ds, master_crs)})
    if no_crs:
        raise DEMCatalogError(f"{len(no_crs)} DEM tile(s) without CRS - location cannot be established: {no_crs[:5]}")
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs=master_crs)
    if len(g):
        g["year_k"] = g["year"].fillna(-1)
        g["vf_k"] = g["valid_fraction"].fillna(0)
        g = g.sort_values(["vf_k", "year_k", "resolution_m", "tile_id"],
                          ascending=[False, False, True, True]).reset_index(drop=True)
        g["priority_rank"] = np.arange(len(g))
        g = g.drop(columns=["year_k", "vf_k"])
    g.attrs["duplicates"] = dups
    return g


class DEMIndex:
    """STRtree over usable tile footprints (master CRS)."""

    def __init__(self, catalog):
        from shapely.strtree import STRtree

        usable = catalog[catalog.geometry.notna() & ~catalog.quality_flags.fillna("").str.contains("UNREADABLE")]
        self.cat = usable.reset_index(drop=True)
        self.tree = STRtree(list(self.cat.geometry)) if len(self.cat) else None

    def query(self, geom) -> pd.DataFrame:
        if self.tree is None:
            return self.cat.iloc[0:0]
        idx = self.tree.query(geom, predicate="intersects")
        return self.cat.iloc[np.sort(idx)].sort_values("priority_rank")


def coverage_100m(catalog, grid, mode: str = "footprint") -> dict:
    """Coverage rasters on the master grid.

    ``count``: number of usable tile footprints covering each cell (overlap); ``covered``: count > 0;
    ``source``: priority_rank of the preferred tile (-1 = none).
    """
    from rasterio.features import rasterize

    h, w = grid.shape
    count = np.zeros((h, w), np.int16)
    source = np.full((h, w), -1, np.int32)
    usable = catalog[catalog.geometry.notna()] if len(catalog) else catalog
    for r in usable.sort_values("priority_rank", ascending=False).itertuples():
        m = rasterize([(r.geometry, 1)], out_shape=(h, w), transform=grid.affine, fill=0, dtype="uint8").astype(bool)
        count += m
        source[m] = r.priority_rank
    return {"count": count, "covered": count > 0, "source": source}


def save_catalog(catalog, out_dir, grid, ctx=None) -> list[Path]:
    import tempfile

    from .registry import atomic_write_bytes, atomic_write_parquet

    out_dir = Path(out_dir)
    tab = pd.DataFrame(catalog.drop(columns="geometry"))
    tab["bounds"] = tab["bounds"].map(lambda b: list(b) if isinstance(b, tuple) else b) if "bounds" in tab else None
    tab["footprint_wkt"] = catalog.geometry.map(lambda g: g.wkt if g is not None else None)
    outs = [atomic_write_parquet(out_dir / "dem_catalog.parquet", tab)]
    if len(catalog):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "dem_coverage.gpkg"
            catalog[["tile_id", "source_id", "year", "resolution_m", "valid_fraction", "priority_rank", "quality_flags",
                     "geometry"]].to_file(tmp, driver="GPKG", layer="dem_tiles")
            outs.append(atomic_write_bytes(out_dir / "dem_coverage.gpkg", tmp.read_bytes()))
    cov = coverage_100m(catalog, grid)
    from .features import write_geotiff

    outs.append(write_geotiff(out_dir / "dem_coverage_100m.tif", np.stack([cov["count"].astype(np.float32),
                                                                            cov["source"].astype(np.float32)]),
                              grid, ["tile_overlap_count", "preferred_tile_priority_rank"]))
    return outs
