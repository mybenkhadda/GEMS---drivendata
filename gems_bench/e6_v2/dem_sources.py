"""USGS 1 m DEM tile discovery, header checks and the vertical-unit rule.

Vertical unit, first rule that answers wins:
1. band ``units`` of the GeoTIFF;
2. a vertical CRS in the GeoTIFF CRS WKT (``VERT_CS`` / ``VERTCRS`` with its ``UNIT``);
3. the USGS FGDC metadata XML next to the tile (``<stem>.xml``, ``<name>.xml``): ``<altunits>``;
4. ``dem.vertical_unit_override`` from the config (set only after reading the USGS metadata yourself);
otherwise the tile is ``unknown`` and the preflight blocks.

Overlap order (deterministic): newer project year first, then project name, then file name.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import is_temporary

UNIT_FACTORS = {"metre": 1.0, "foot": 0.3048, "us_survey_foot": 1200.0 / 3937.0}
_ALIASES = {
    "m": "metre", "meter": "metre", "meters": "metre", "metre": "metre", "metres": "metre",
    "ft": "foot", "foot": "foot", "feet": "foot", "international foot": "foot", "intl foot": "foot",
    "us survey foot": "us_survey_foot", "us survey feet": "us_survey_foot", "us_survey_foot": "us_survey_foot",
    "foot_us": "us_survey_foot", "us foot": "us_survey_foot", "survey foot": "us_survey_foot",
}


def normalise_unit(text: str | None) -> str | None:
    if not text:
        return None
    t = re.sub(r"\s+", " ", str(text).strip().lower().replace('"', ""))
    if t in _ALIASES:
        return _ALIASES[t]
    if "survey" in t and ("foot" in t or "feet" in t):
        return "us_survey_foot"
    if "foot" in t or "feet" in t:
        return "foot"
    if t.startswith("met"):
        return "metre"
    return None


def _wkt_vertical_unit(wkt: str) -> str | None:
    for tag in ("VERT_CS", "VERTCRS", "VERTICALCRS"):
        i = wkt.find(tag)
        if i >= 0:
            m = re.search(r'(?:LENGTHUNIT|UNIT)\["([^"]+)"', wkt[i:])
            if m:
                return normalise_unit(m.group(1))
    return None


def _xml_vertical_unit(tif: Path) -> tuple[str | None, str | None]:
    for cand in (tif.with_suffix(".xml"), tif.with_name(tif.name + ".xml")):
        if cand.exists() and not is_temporary(cand):
            try:
                txt = cand.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            m = re.search(r"<altunits>\s*([^<]+?)\s*</altunits>", txt, re.I)
            if m:
                return normalise_unit(m.group(1)), cand.name
            return None, cand.name
    return None, None


def project_of(path: Path) -> tuple[str, int]:
    stem = path.stem
    m = re.search(r"x\d+y\d+[_-](.+)$", stem, re.I)
    proj = m.group(1) if m else path.parent.name
    years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", proj)
    return proj, int(years[-1]) if years else 0


def discover_tiles(drive_root: Path, cfg: dict) -> tuple[list[Path], list[dict]]:
    """All DEM GeoTIFFs under ``raw.dem_dirs``; temporary files and forbidden locations are rejected."""
    forbidden = [Path(drive_root) / f for f in cfg["raw"]["forbidden_prefixes"]]
    found, rejected = [], []
    for d in cfg["raw"]["dem_dirs"]:
        base = Path(drive_root) / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in (".tif", ".tiff") or not p.is_file():
                continue
            if is_temporary(p):
                rejected.append({"path": str(p), "reason": "temporary file"})
            elif any(_under(p, f) for f in forbidden):
                rejected.append({"path": str(p), "reason": "forbidden location"})
            else:
                found.append(p)
    return found, rejected


def _under(p: Path, parent: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def read_tile_header(path: Path, cfg: dict, target_crs_wkt: str) -> dict:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    rec = {"path": str(path), "name": path.name, "bytes": path.stat().st_size}
    proj, year = project_of(path)
    rec.update(project=proj, year=year)
    try:
        with rasterio.open(path) as src:
            rec.update(crs_epsg=src.crs.to_epsg() if src.crs else None, crs_wkt=src.crs.to_wkt() if src.crs else "",
                       res_x=float(abs(src.transform.a)), res_y=float(abs(src.transform.e)), width=src.width,
                       height=src.height, count=src.count, dtype=src.dtypes[0],
                       nodata=src.nodata if src.nodata is not None else np.nan)
            b = src.bounds
            rec.update(native_left=b.left, native_bottom=b.bottom, native_right=b.right, native_top=b.top)
            if src.crs:
                l, bo, r, t = transform_bounds(src.crs, CRS.from_wkt(target_crs_wkt), *b, densify_pts=21)
                rec.update(left=l, bottom=bo, right=r, top=t)
            band_unit = normalise_unit((src.units or [None])[0]) if src.units else None
            wkt_unit = _wkt_vertical_unit(rec["crs_wkt"])
    except Exception as exc:  # unreadable header -> recorded, blocks in the preflight
        rec.update(header_error=f"{type(exc).__name__}: {exc}")
        return rec
    xml_unit, xml_name = _xml_vertical_unit(path)
    override = normalise_unit(cfg["dem"].get("vertical_unit_override"))
    for src_rule, unit in (("band_units", band_unit), ("crs_wkt", wkt_unit), ("fgdc_xml", xml_unit),
                           ("config_override", override)):
        if unit:
            rec.update(vertical_unit=unit, vertical_unit_source=src_rule)
            break
    else:
        rec.update(vertical_unit="unknown", vertical_unit_source="none")
    rec["metadata_xml"] = xml_name or ""
    return rec


def validate_tiles(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = cfg["dem"]
    status = []
    for r in df.itertuples():
        if isinstance(getattr(r, "header_error", None), str) and r.header_error:
            status.append("rejected:header_error")
        elif not getattr(r, "crs_wkt", ""):
            status.append("rejected:no_crs")
        elif not (d["native_res_min_m"] <= r.res_x <= d["native_res_max_m"]
                  and d["native_res_min_m"] <= r.res_y <= d["native_res_max_m"]):
            status.append(f"rejected:resolution_{r.res_x:.3g}m")
        elif r.count < 1:
            status.append("rejected:no_band")
        elif r.vertical_unit == "unknown":
            status.append("blocked:vertical_unit_unknown")
        else:
            status.append("ok")
    out = df.copy()
    out["validation_status"] = status
    return out


def overlap_order(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic overlap priority: rank 0 wins wherever tiles overlap."""
    ok = df[df.validation_status == "ok"].copy()
    ok = ok.sort_values(["year", "project", "name"], ascending=[False, True, True]).reset_index(drop=True)
    ok["priority"] = np.arange(len(ok))
    ok["project_id"] = pd.factorize(ok["project"])[0]
    return ok
