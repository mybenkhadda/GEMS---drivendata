"""DEM vertical-unit audit from AUTHORITATIVE metadata only (never inferred from elevation magnitudes).

Statements collected per tile:
  band_units     GDAL band unit (``ds.units``)
  tag            UNITTYPE / VERTICAL_UNIT(S) / units tags (dataset + band)
  vertical_crs   unit of the VERTICAL component of the compound CRS. USGS 3DEP GeoTIFFs store NAVD88 in the
                 GeoKeys; GDAL reports it only with ``GTIFF_REPORT_COMPD_CS=YES`` (set here).
  vertical_epsg  EPSG of the vertical datum when it is a metre datum by definition (NAVD88 height 5703, ...)
  xml_altunits   ``<altunits>`` of the FGDC XML sidecar
  catalog        vertical-unit columns of a downloaded acquisition catalog CSV (if such a column exists)
Classification: VERIFIED_METRES / VERIFIED_FEET (all statements agree) / CONFLICTING / UNKNOWN (no statement).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

METRE = {"metre", "meter", "metres", "meters", "m", "metre (m)"}
FOOT = {"foot", "feet", "ft", "us survey foot", "us_survey_foot", "foot_us", "us foot", "international foot",
        "foot (international)", "ftus"}
METRE_VERTICAL_EPSG = {5703: "NAVD88 height", 5773: "EGM96 height", 3855: "EGM2008 height"}
FEET_VERTICAL_EPSG = {6360: "NAVD88 height (ftUS)", 8228: "NAVD88 height (ft)"}
US_SURVEY_FOOT_M, INTERNATIONAL_FOOT_M = 1200.0 / 3937.0, 0.3048
VERIFIED_M, VERIFIED_FT, UNKNOWN, CONFLICT = "VERIFIED_METRES", "VERIFIED_FEET", "UNKNOWN", "CONFLICTING"


def normalise(unit: str | None) -> str | None:
    if not unit:
        return None
    u = str(unit).strip().strip('"').lower().replace("_", " ")
    if u in METRE or u.startswith("metre") or u.startswith("meter"):
        return "metre"
    if u in FOOT or "foot" in u or "feet" in u:
        return "foot"
    return None


def _vertical_from_wkt(wkt: str) -> tuple[str | None, int | None]:
    m = re.search(r'(?:VERT_CS|VERTCRS|VERTICALCRS)\[(.*)', wkt or "", re.S)
    if not m:
        return None, None
    body = m.group(1)
    u = re.search(r'(?:LENGTHUNIT|UNIT)\["([^"]+)"', body)
    e = re.findall(r'(?:ID|AUTHORITY)\["EPSG",\s*"?(\d+)"?\]', body)
    return (u.group(1) if u else None), (int(e[-1]) if e else None)


def tile_statements(path) -> list[tuple[str, str]]:
    import rasterio

    st = []
    with rasterio.Env(GTIFF_REPORT_COMPD_CS="YES"):
        with rasterio.open(path) as ds:
            for u in (ds.units or ()):
                if u:
                    st.append(("band_units", u))
            tags = {**ds.tags(), **ds.tags(1)}
            for k in ("UNITTYPE", "VERTICAL_UNIT", "VERTICAL_UNITS", "units", "UNITS"):
                if tags.get(k):
                    st.append(("tag", tags[k]))
            if ds.crs is not None:
                u, epsg = _vertical_from_wkt(ds.crs.to_wkt())
                if u:
                    st.append(("vertical_crs", u))
                if epsg in FEET_VERTICAL_EPSG:
                    st.append(("vertical_epsg", "foot"))
                elif epsg in METRE_VERTICAL_EPSG:
                    st.append(("vertical_epsg", "metre"))
    xml = Path(path).with_suffix(".xml")
    if xml.is_file():
        m = re.search(r"<altunits>\s*([^<]+?)\s*</altunits>", xml.read_text(encoding="utf-8", errors="ignore"))
        if m:
            st.append(("xml_altunits", m.group(1)))
    return st


def classify(statements: list[tuple[str, str]]) -> tuple[str, str]:
    norm = {(src, normalise(v)) for src, v in statements}
    units = {u for _, u in norm if u}
    srcs = ";".join(sorted({f"{s}={v}" for s, v in statements}))
    if not units:
        return UNKNOWN, srcs
    if len(units) > 1:
        return CONFLICT, srcs
    return (VERIFIED_M if units == {"metre"} else VERIFIED_FT), srcs


def catalog_units(catalog_csvs) -> dict:
    """{tile filename stem: unit statement} from acquisition CSVs having a vertical-unit-like column (else empty)."""
    out = {}
    for p in catalog_csvs:
        p = Path(p)
        if not p.is_file():
            continue
        try:
            t = pd.read_csv(p, low_memory=False)
        except Exception:                                          # noqa: BLE001 - unreadable catalog is skipped
            continue
        ucol = next((c for c in t.columns if re.search(r"vert.*unit|alt.*unit|z.?unit|elev.*unit", c, re.I)), None)
        ncol = next((c for c in t.columns if re.search(r"file|name|title|tile", c, re.I)), None)
        if ucol and ncol:
            for n, u in zip(t[ncol].astype(str), t[ucol].astype(str)):
                out[Path(n).stem] = u
    return out


def audit_vertical_units(paths, catalog_csvs=()) -> pd.DataFrame:
    cat = catalog_units(catalog_csvs)
    rows = []
    for p in paths:
        p = Path(p)
        try:
            st = tile_statements(p)
        except Exception as e:                                      # noqa: BLE001 - recorded per tile
            rows.append({"tile_id": p.stem, "path": str(p), "class": UNKNOWN, "statements": f"UNREADABLE {e}",
                         "vertical_scale": None})
            continue
        if p.stem in cat:
            st.append(("catalog", cat[p.stem]))
        cls, srcs = classify(st)
        if cls == VERIFIED_M:
            scale = 1.0
        elif cls == VERIFIED_FT:                     # US survey foot unless a statement says international
            scale = INTERNATIONAL_FOOT_M if "international" in srcs.lower() else US_SURVEY_FOOT_M
        else:
            scale = None
        rows.append({"tile_id": p.stem, "path": str(p), "class": cls, "statements": srcs, "vertical_scale": scale})
    return pd.DataFrame(rows, columns=["tile_id", "path", "class", "statements", "vertical_scale"])
