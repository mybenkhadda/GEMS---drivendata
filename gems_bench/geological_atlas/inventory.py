"""Raw-source discovery (originals) and inventories of the snapshot.

Discovery rules (``discover_sources``) - only these ORIGINAL raw locations under ``drive_root`` are sources:
* GeoDAWN raster, fault labels, template: the configured file names in ``raw.geodawn_dirs`` (first hit in that
  order is the primary; further hits are recorded as ``duplicate_candidate`` and compared by size);
* DEM tiles: ``raw.dem_dir/**/*.tif|*.tiff``; DEM metadata: ``raw.dem_dir/**/*.xml``;
* DEM catalogs: ``raw.catalog_files``; provenance / licence documents: ``raw.provenance_globs``.
Never sources: ``outputs/``, ``experiments/``, feature stores, other atlas / E6 namespaces, ``.partial`` files.
An unreadable source is ``INACCESSIBLE`` (with its error class), never ``MISSING``; a Drive disconnect stops the scan.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import is_partial
from .reliability import classify, raise_infrastructure

TILE_RE = re.compile(r"USGS_(?:1M|one_meter)_(?P<zone>\d{1,2})_x(?P<x>\d+)y(?P<y>\d+)_(?P<project>[A-Za-z0-9_\-]+?)"
                     r"(?:_\d{8})?\.tiff?$", re.IGNORECASE)
FORBIDDEN_PARTS = ("outputs", "experiments", "feature_store", "processed", "model_registry", "e6_v2_from_raw",
                   "GEMS_GEOLOGICAL_DATA_ATLAS")


def _stat(p: Path) -> tuple[int | None, int | None, str]:
    try:
        st = os.stat(p)
        return int(st.st_size), int(st.st_mtime_ns), "OK"
    except Exception as exc:
        raise_infrastructure(exc, f"stat {p}")
        return None, None, f"INACCESSIBLE:{classify(exc)}"


def _allowed(p: Path, drive_root: Path) -> bool:
    try:
        parts = p.relative_to(drive_root).parts
    except ValueError:
        return False
    return not any(any(f.lower() in part.lower() for f in FORBIDDEN_PARTS) for part in parts) and not is_partial(p)


def _walk(base: Path, suffixes: tuple[str, ...]) -> list[Path]:
    out = []
    try:
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                if f.lower().endswith(suffixes):
                    out.append(Path(dirpath) / f)
    except Exception as exc:
        raise_infrastructure(exc, f"listing {base}")
        raise
    return sorted(out)


def discover_sources(cfg, drive_root: Path) -> pd.DataFrame:
    r = cfg["raw"]
    rows = []

    def add(role, p, dest, **kw):
        size, mtime, status = _stat(p)
        rows.append({"role": role, "source_path": str(p), "rel_source": p.relative_to(drive_root).as_posix(),
                     "snapshot_rel": dest, "size_bytes": size, "mtime_ns": mtime, "access_status": status, **kw})

    for role, key, sub in (("geodawn_features", "geodawn_file", "geodawn"), ("fault_labels", "faults_file", "faults"),
                           ("official_template", "template_file", "official_template")):
        hits = []
        for d in r["geodawn_dirs"]:
            p = drive_root / d / r[key]
            try:
                if p.is_file() and _allowed(p, drive_root):
                    hits.append(p)
            except Exception as exc:
                raise_infrastructure(exc, f"probing {p}")
        if not hits:
            rows.append({"role": role, "source_path": None, "rel_source": None, "snapshot_rel": f"00_raw_snapshot/{sub}/"
                         f"{r[key]}", "size_bytes": None, "mtime_ns": None, "access_status": "MISSING"})
            continue
        add(role, hits[0], f"00_raw_snapshot/{sub}/{r[key]}")
        for h in hits[1:]:
            add(f"{role}:duplicate_candidate", h, None)
    dem_base = drive_root / r["dem_dir"]
    if dem_base.exists():
        for p in _walk(dem_base, (".tif", ".tiff")):
            if _allowed(p, drive_root):
                add("dem_tile", p, f"00_raw_snapshot/dem_1m/{p.relative_to(dem_base).as_posix()}")
        for p in _walk(dem_base, (".xml",)):
            if _allowed(p, drive_root) and not p.name.endswith(".aux.xml"):
                add("dem_metadata", p, f"00_raw_snapshot/dem_1m/{p.relative_to(dem_base).as_posix()}")
    for c in r["catalog_files"]:
        p = drive_root / c
        if p.is_file():
            add("dem_catalog", p, f"00_raw_snapshot/source_metadata/{c.replace('/', '__')}")
    seen = {row["source_path"] for row in rows}
    for g in r["provenance_globs"]:
        for p in sorted(drive_root.glob(g)):
            if p.is_file() and str(p) not in seen and _allowed(p, drive_root) and p.suffix.lower() != ".tif":
                add("provenance_document", p, f"00_raw_snapshot/source_metadata/{p.relative_to(drive_root).as_posix().replace('/', '__')}")
                seen.add(str(p))
    df = pd.DataFrame(rows, dtype=object)       # object first: no int -> float64 coercion of mtime_ns
    df.insert(0, "source_id", [f"src{i:05d}" for i in range(len(df))])
    # nanosecond mtimes exceed float64 precision: keep them as nullable integers
    df["size_bytes"] = pd.array([None if v is None or v != v else int(v) for v in df.size_bytes], dtype="Int64")
    df["mtime_ns"] = pd.array([None if v is None or v != v else int(v) for v in df.mtime_ns], dtype="Int64")
    df["mtime_utc"] = [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(m) / 1e9)) if pd.notna(m) else None
                       for m in df.mtime_ns]
    return df


# ---------------------------------------------------------------------------
# GeoDAWN band registry (from the real raster metadata)
# ---------------------------------------------------------------------------
KNOWN_MODALITY = {
    "rtp": ("magnetic", "nT", "reduced-to-pole magnetic anomaly"),
    "mag_anom": ("magnetic", "nT", "magnetic anomaly (total field, not reduced to pole)"),
    "tmi": ("magnetic", "nT", "total magnetic intensity"), "tc": ("magnetic", "unknown", "magnetic product 'tc'"),
    "tmi_hg": ("magnetic", "nT/m", "horizontal gradient of TMI (delivered)"),
    "tmi_vg": ("magnetic", "nT/m", "vertical gradient of TMI (delivered)"),
    "iso_grav_anom": ("gravity", "mGal", "isostatic gravity anomaly"),
    "iso_grav_anom_hg": ("gravity", "mGal/km", "horizontal gradient of the isostatic anomaly (delivered)"),
    "iso_grav_anom_vg": ("gravity", "mGal/km", "vertical gradient of the isostatic anomaly (delivered)"),
    "iso_grav_anom_slope": ("gravity", "unknown", "slope of the isostatic anomaly surface (delivered)"),
    "det_elev": ("terrain", "m", "elevation"), "det_elev_slope": ("terrain", "degree", "terrain slope (delivered)"),
    "depth_to_base_surf": ("basement", "m", "depth to basement surface"),
    "cond_surf": ("conductivity", "unknown", "surface conductivity"),
    "geod_2ndinv": ("strain", "unknown", "geodetic strain-rate second invariant"),
    "geod_shearrate": ("strain", "unknown", "geodetic shear strain rate"),
    "geod_dilaterate": ("strain", "unknown", "geodetic dilatation rate"),
    "deq_n100a15": ("seismicity", "count", "earthquake density (deep)"),
    "ieq_n100a15": ("seismicity", "count", "earthquake density"),
}


def band_names(src) -> list[str]:
    out = []
    for i in range(1, src.count + 1):
        tags = src.tags(i)
        desc = src.descriptions[i - 1] or ""
        out.append(tags.get("band_name") or desc.split(" - ")[0].strip() or f"band_{i}")
    return out


def band_registry(path: Path, cfg, sample_step: int = 4) -> pd.DataFrame:
    """One row per band of the actual raster: identifiers, metadata, units, statistics, valid fraction."""
    import rasterio

    nod_abs = float(cfg["raw"]["nodata_abs"])
    rows = []
    with rasterio.open(path) as src:
        names = band_names(src)
        for i, n in enumerate(names, start=1):
            tags = src.tags(i)
            a = src.read(i, out_shape=(src.height // sample_step, src.width // sample_step)).astype(np.float64)
            bad = ~np.isfinite(a) | (np.abs(a) >= nod_abs)
            if src.nodata is not None and np.isfinite(src.nodata):
                bad |= a == src.nodata
            v = a[~bad]
            known = KNOWN_MODALITY.get(n, ("unknown", "unknown", ""))
            unit_meta = tags.get("units") or tags.get("unit") or (src.units[i - 1] if src.units else "") or ""
            q = np.percentile(v, [1, 50, 99]) if v.size else [np.nan] * 3
            rows.append({"band_index": i, "band_name": n, "description": src.descriptions[i - 1] or "",
                         "tags": ";".join(f"{k}={v}" for k, v in tags.items()), "modality": known[0],
                         "unit": unit_meta or known[1],
                         "unit_source": "raster metadata" if unit_meta else ("documented default" if known[1] != "unknown"
                                                                              else "unknown"),
                         "meaning": known[2], "dtype": src.dtypes[i - 1], "nodata": src.nodata,
                         "valid_fraction_sampled": float(v.size / a.size), "min": float(v.min()) if v.size else np.nan,
                         "p01": float(q[0]), "p50": float(q[1]), "p99": float(q[2]),
                         "max": float(v.max()) if v.size else np.nan, "mean": float(v.mean()) if v.size else np.nan,
                         "std": float(v.std()) if v.size else np.nan, "stats_sampling": f"every {sample_step}th pixel"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DEM inventory (on the snapshot copies)
# ---------------------------------------------------------------------------
UNIT_FACTORS = {"metre": 1.0, "foot": 0.3048, "us_survey_foot": 1200.0 / 3937.0}


def normalise_unit(t) -> str | None:
    if not t:
        return None
    s = re.sub(r"\s+", " ", str(t).strip().lower().replace('"', ""))
    if "survey" in s and ("foot" in s or "feet" in s):
        return "us_survey_foot"
    if s in ("ft", "foot", "feet", "international foot") or "foot" in s or "feet" in s:
        return "foot"
    if s in ("m", "meter", "meters", "metre", "metres") or s.startswith("met"):
        return "metre"
    return None


def vertical_unit(src, tif: Path, override) -> tuple[str, str]:
    cands = [("band_units", (src.units or [None])[0] if src.units else None)]
    tags = src.tags(1) | src.tags()
    cands += [("tags", tags.get(k)) for k in ("UNITTYPE", "units", "VERTICAL_UNITS", "VERTICAL_UNIT")]
    wkt = src.crs.to_wkt() if src.crs else ""
    for tag in ("VERT_CS", "VERTCRS", "VERTICALCRS"):
        i = wkt.find(tag)
        if i >= 0:
            m = re.search(r'(?:LENGTHUNIT|UNIT)\["([^"]+)"', wkt[i:])
            cands.append(("vertical_crs", m.group(1) if m else None))
    for x in (tif.with_suffix(".xml"), tif.with_name(tif.name + ".xml")):
        if x.exists():
            m = re.search(r"<altunits>\s*([^<]+?)\s*</altunits>", x.read_text(encoding="utf-8", errors="ignore"), re.I)
            cands.append(("fgdc_xml", m.group(1) if m else None))
    for src_name, val in cands:
        u = normalise_unit(val)
        if u:
            return u, src_name
    if override:
        u = normalise_unit(override)
        if u:
            return u, "config_override"
    return "unknown", "none"


def has_metadata(tif: Path) -> bool:
    return tif.with_suffix(".xml").exists() or tif.with_name(tif.name + ".xml").exists()


def project_of(path: Path) -> tuple[str, str | None, int]:
    m = TILE_RE.search(path.name)
    proj = m.group("project") if m else path.parent.name
    tile = f"{m['zone']}_x{m['x']}y{m['y']}" if m else None
    years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", proj)
    return proj, tile, int(years[-1]) if years else 0


def dem_inventory(snapshot_dem: Path, grid, footprint: np.ndarray, cfg, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Header-level classification of every snapshot DEM tile (+ catalog-only rows)."""
    import rasterio
    from rasterio.warp import transform_bounds

    from .geospatial import window_of_bounds

    dp = cfg["dem"]
    rows = []
    for p in _walk(snapshot_dem, (".tif", ".tiff")):
        if is_partial(p):
            continue
        proj, tile, year = project_of(p)
        rec = {"file": p.name, "snapshot_path": str(p), "project": proj, "tile_id": tile, "year": year,
               "has_metadata": has_metadata(p), "bytes": p.stat().st_size}
        try:
            with rasterio.open(p) as s:
                rec.update(crs_epsg=s.crs.to_epsg() if s.crs else None, crs_wkt=s.crs.to_wkt() if s.crs else "",
                           res_x=abs(s.transform.a), res_y=abs(s.transform.e), width=s.width, height=s.height,
                           nodata=s.nodata, dtype=s.dtypes[0], projected=bool(s.crs and s.crs.is_projected))
                unit, usrc = vertical_unit(s, p, dp["vertical_unit_override"])
                rec.update(vertical_unit=unit, vertical_unit_source=usrc)
                # real data check: a decimated read must contain finite elevations
                small = s.read(1, out_shape=(max(s.height // 50, 1), max(s.width // 50, 1)), masked=True)
                rec["sample_valid_fraction"] = float(np.ma.count(small) / small.size)
                if s.crs:
                    b = transform_bounds(s.crs, grid.crs, *s.bounds, densify_pts=21)
                    rec.update(left=b[0], bottom=b[1], right=b[2], top=b[3])
        except Exception as exc:
            raise_infrastructure(exc, f"reading DEM {p.name}")
            rec.update(status="UNREADABLE", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
            rows.append(rec)
            continue
        if not rec["projected"]:
            rec.update(status="UNREADABLE", reason="CRS missing or not projected")
        elif not (dp["native_res_min_m"] <= rec["res_x"] <= dp["native_res_max_m"]):
            rec.update(status="UNREADABLE", reason=f"resolution {rec['res_x']:.3g} m outside accepted range")
        elif rec["sample_valid_fraction"] <= 0:
            rec.update(status="UNREADABLE", reason="no finite elevation in a decimated read")
        elif rec["vertical_unit"] == "unknown":
            rec.update(status="UNKNOWN_VERTICAL_UNIT", reason="no unit in band units, tags, vertical CRS or FGDC XML")
        else:
            w = window_of_bounds((rec["left"], rec["bottom"], rec["right"], rec["top"]), grid)
            if w is None or not footprint[w[0]:w[1], w[2]:w[3]].any():
                rec.update(status="OUTSIDE_STUDY_AREA", reason="no overlap with the GeoDAWN data footprint")
            else:
                rec.update(status="VALID", reason="")
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df) and "tile_id" in df:
        ok = df.status == "VALID"
        order = df[ok].sort_values(["tile_id", "year", "project"], ascending=[True, False, True])
        keep = set(order.drop_duplicates("tile_id", keep="first").index) | set(df[ok & df.tile_id.isna()].index)
        dup = ok & ~df.index.isin(keep)
        df.loc[dup, "status"] = "DUPLICATE_OVERLAP"
        df.loc[dup, "reason"] = "another project covers the same 10 km tile (newest year kept)"
    if catalog is not None and len(catalog):
        local = set(df.file) if len(df) else set()
        cat = catalog[~catalog.file.isin(local)].copy()
        cat["status"] = "CATALOG_ONLY"
        cat["reason"] = "listed in the TNM catalog, not downloaded"
        df = pd.concat([df, cat], ignore_index=True)
    return df


def load_catalog(snapshot_meta: Path) -> pd.DataFrame | None:
    """Parse the snapshot copies of the TNM catalog JSON (URL lists or TNM Access API responses)."""
    import json

    rows = []
    for p in sorted(snapshot_meta.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        recs = obj["items"] if isinstance(obj, dict) and "items" in obj else (
            [{"title": k, "downloadURL": v} for k, v in obj.items()] if isinstance(obj, dict) else
            [{"downloadURL": x} if isinstance(x, str) else dict(x) for x in obj] if isinstance(obj, list) else [])
        for r in recs:
            url = r.get("downloadURL") or r.get("url")
            if not url:
                continue
            name = url.rsplit("/", 1)[-1]
            m = TILE_RE.search(name)
            if not m:
                continue
            pdir = re.search(r"/Projects/([^/]+)/", url)
            x, y, z = int(m["x"]), int(m["y"]), int(m["zone"])
            rows.append({"file": name, "url": url, "project": m["project"], "project_dir": pdir.group(1) if pdir
                         else m["project"], "tile_id": f"{z}_x{x}y{y}", "zone": z, "west": x * 1e4, "north": y * 1e4,
                         "east": x * 1e4 + 1e4, "south": y * 1e4 - 1e4, "size_bytes": r.get("sizeInBytes"),
                         "catalog_file": p.name})
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates("url")
