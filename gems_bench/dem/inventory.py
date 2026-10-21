"""Experiment F1: inventory of the USGS TNM 1 m DEM download list (no downloads, no raster reads by default).

Input: the JSON of GeoTIFF download URLs (placed at ``data/raw/dem_1m/tnm_urls.json``). Accepted shapes:
* TNM Access API response: ``{"items": [{"downloadURL", "title", "boundingBox", "sizeInBytes", ...}, ...]}``
* a list of URL strings, a list of objects with ``downloadURL`` / ``url``, or a ``{name: url}`` mapping.

Per URL: schema check, tile id, UTM zone, 10 km tile index, project id (e.g. ``NV_WesternNV_2021_D21``),
nominal footprint from the tile name, duplicates (same tile in several projects = overlap), intersection
with the native-grid data footprint, size estimate. Tile names follow ``USGS_1M_<zone>_x<E/10km>y<N/10km>_<project>``;
the name is taken to give the tile's upper-left corner (``corner="upper_left"``). ``inspect_headers`` reads real
GeoTIFF headers over HTTP (``/vsicurl/``, a few KB each, run by the user) to confirm CRS, resolution and bounds;
``verify_corners`` flags any disagreement between names and headers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

TILE_RE = re.compile(r"USGS_(?:1M|one_meter)_(?P<zone>\d{1,2})_x(?P<x>\d+)y(?P<y>\d+)_(?P<project>[A-Za-z0-9_\-]+?)(?:_\d{8})?\.tif",
                     re.IGNORECASE)
TILE_M = 10_000.0
TYPICAL_TILE_BYTES = 350e6           # 10 000 x 10 000 float32 with LZW, typical of TNM 1 m tiles


def _records(obj) -> list[dict]:
    if isinstance(obj, dict) and "items" in obj:
        return [dict(x) for x in obj["items"]]
    if isinstance(obj, dict):
        return [{"title": k, "downloadURL": v} for k, v in obj.items()]
    if isinstance(obj, list):
        return [{"downloadURL": x} if isinstance(x, str) else dict(x) for x in obj]
    raise ValueError("unsupported JSON structure for the DEM URL list")


def parse_url(url: str) -> dict:
    name = Path(urlparse(url).path).name
    m = TILE_RE.search(name)
    out = {"url": url, "file": name, "parse_ok": bool(m), "scheme": urlparse(url).scheme}
    if m:
        zone, x, y = int(m["zone"]), int(m["x"]), int(m["y"])
        proj_dir = re.search(r"/Projects/([^/]+)/", url)
        out.update(zone=zone, tile=f"x{x}y{y}", tile_id=f"{zone}_x{x}y{y}", project=m["project"],
                   project_dir=proj_dir.group(1) if proj_dir else m["project"],
                   west=x * TILE_M, north=y * TILE_M, east=x * TILE_M + TILE_M, south=y * TILE_M - TILE_M,
                   year=_year(m["project"]))
    return out


def _year(project: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", project)
    return int(m.group(0)) if m else None


def load_inventory(path: Path) -> pd.DataFrame:
    """One row per URL with parsed tile metadata, schema flags and TNM fields when present."""
    recs = _records(json.loads(Path(path).read_text(encoding="utf-8")))
    rows = []
    for r in recs:
        url = r.get("downloadURL") or r.get("url") or r.get("download_url")
        if not url:
            rows.append({"parse_ok": False, "error": "no URL field", "raw": json.dumps(r)[:200]})
            continue
        row = parse_url(url)
        row.update(size_bytes=r.get("sizeInBytes"), publication_date=r.get("publicationDate"),
                   last_updated=r.get("lastUpdated"), title=r.get("title"), source_id=r.get("sourceId"),
                   meta_url=r.get("metaUrl"), format=r.get("format"))
        bb = r.get("boundingBox") or {}
        row.update({f"bbox_{k}": bb.get(k) for k in ("minX", "minY", "maxX", "maxY")})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["is_tif"] = df.get("file", pd.Series(dtype=str)).fillna("").str.lower().str.endswith(".tif")
    df["duplicate_url"] = df.duplicated("url", keep="first") if "url" in df else False
    if "tile_id" in df:
        df["n_projects_for_tile"] = df.groupby("tile_id").project.transform("nunique")
    return df


def native_footprint_utm(ctx, zone: int = 11) -> tuple[np.ndarray, object]:
    """(valid mask, transform) of the native grid in its own CRS (EPSG:32611 = WGS84 / UTM 11N)."""
    return ctx.data.valid, ctx.data.grid.transform


def study_area_intersection(inv: pd.DataFrame, ctx, min_valid_frac: float = 0.05) -> pd.DataFrame:
    """Share of each nominal tile footprint inside the native-grid data footprint (zone-11 tiles only).

    NAD83 / UTM 11N tiles vs the WGS84 / UTM 11N native grid differ by ~1-2 m: negligible at 100 m.
    """
    from rasterio.transform import rowcol

    d = ctx.data
    inv = inv.copy()
    fr, nvalid, nlab = [], [], []
    for r in inv.itertuples():
        if not getattr(r, "parse_ok", False) or getattr(r, "zone", None) != 11:
            fr.append(np.nan), nvalid.append(0), nlab.append(0)
            continue
        r0, c0 = rowcol(d.grid.transform, r.west, r.north)
        r1, c1 = rowcol(d.grid.transform, r.east, r.south)
        r0, r1 = max(min(r0, r1), 0), min(max(r0, r1), d.shape[0])
        c0, c1 = max(min(c0, c1), 0), min(max(c0, c1), d.shape[1])
        if r1 <= r0 or c1 <= c0:
            fr.append(0.0), nvalid.append(0), nlab.append(0)
            continue
        v = d.valid[r0:r1, c0:c1]
        fr.append(float(v.mean())), nvalid.append(int(v.sum())), nlab.append(int(d.labels[r0:r1, c0:c1].sum()))
    inv["valid_frac"], inv["valid_px"], inv["label_px"] = fr, nvalid, nlab
    inv["in_study_area"] = inv.valid_frac.fillna(0) >= min_valid_frac
    return inv


def storage_estimate(inv: pd.DataFrame) -> dict:
    sel = inv[inv.get("in_study_area", True)]
    known = sel.size_bytes.dropna().astype(float) if "size_bytes" in sel else pd.Series(dtype=float)
    est = known.sum() + (len(sel) - len(known)) * TYPICAL_TILE_BYTES
    return {"n_tiles": int(len(sel)), "n_unique_tiles": int(sel.tile_id.nunique()) if "tile_id" in sel else None,
            "bytes_known": float(known.sum()), "gib_estimate": float(est / 2**30), "n_without_size": int(len(sel) - len(known))}


def choose_per_tile(inv: pd.DataFrame, policy: str = "latest") -> pd.DataFrame:
    """One URL per tile where projects overlap: ``latest`` (year, then publication date) or ``first``."""
    sel = inv[inv.parse_ok.fillna(False)].copy()
    if policy == "latest":
        sel = sel.sort_values(["tile_id", "year", "publication_date"], ascending=[True, False, False], na_position="last")
    return sel.drop_duplicates("tile_id", keep="first")


def inspect_headers(urls: list[str], max_n: int = 50) -> pd.DataFrame:
    """CRS / resolution / bounds / dtype / nodata from remote GeoTIFF headers (HTTP range requests only)."""
    import rasterio

    rows = []
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    with rasterio.Env(**env):
        for u in urls[:max_n]:
            try:
                with rasterio.open("/vsicurl/" + u) as src:
                    rows.append({"url": u, "crs": str(src.crs), "res_x": src.res[0], "res_y": src.res[1],
                                 "width": src.width, "height": src.height, "dtype": src.dtypes[0], "nodata": src.nodata,
                                 "west_h": src.bounds.left, "south_h": src.bounds.bottom, "east_h": src.bounds.right,
                                 "north_h": src.bounds.top, "ok": True})
            except Exception as e:
                rows.append({"url": u, "ok": False, "error": str(e)[:200]})
    return pd.DataFrame(rows)


def verify_corners(inv: pd.DataFrame, headers: pd.DataFrame, tol_m: float = 50.0) -> pd.DataFrame:
    m = inv.merge(headers[headers.ok], on="url", how="inner")
    m["corner_ok"] = (np.abs(m.west - m.west_h) <= tol_m) & (np.abs(m.north - m.north_h) <= tol_m)
    return m[["url", "tile_id", "west", "north", "west_h", "north_h", "crs", "res_x", "corner_ok"]]
