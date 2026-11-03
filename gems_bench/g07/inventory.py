"""What exists, what is missing, and whether it may be used.

* DEM: catalog (TNM URL list) vs local files (headers), resolution / CRS / vertical unit / nodata checks, coverage
  measured on the ELIGIBLE fault-discovery region of the 100 m grid (footprint & template & not a known fault) and
  on the V1 evaluation zones - never as a bounding-box overlap.
* G06 feature store: the configured run directory (or the newest one with a COMPLETE ``features`` manifest), its
  schema hash and group files.
* W2: P6 held-out unit maps (``stack/maps/<task>_<zone>.npy``), the full-region score (``stack/maps/
  full_score.tif``) and the stacker gate (``stack/candidate.json``).
* Licences / eligibility: a table of every source; the DEM is public domain (USGS 3DEP) but its use as external data
  in the competition must be confirmed by the user (``dem.competition_eligibility_verified``) before any submission
  that depends on it.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..g06.inventory import EngineUnavailable, attach_engine  # noqa: F401  (re-exported for the pipeline)
from .contracts import GridSpec


def discover_dem_sources(dem_cfg: dict, roots) -> tuple[pd.DataFrame, list]:
    """Catalog + local DEM files with header checks (``dem_io.open_dem_catalog``) and existing DEM products."""
    from .dem_io import open_dem_catalog

    T, sources = open_dem_catalog(dem_cfg, roots)
    prods = []
    for r in roots:
        for rel in (dem_cfg.get("existing_products") or []):
            for p in glob.glob(str(Path(r) / rel)):
                prods.append({"product": rel, "path": p, "note": "earlier 100 m DEM aggregate (inventoried; G07 "
                                                                "recomputes its own definitions)"})
    T.attrs["existing_products"] = prods
    return T, sources


def inspect_dem_coverage(sources: list, grid: GridSpec, footprint: np.ndarray, template_valid: np.ndarray,
                         labels: np.ndarray, eval_zones: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """(per-region coverage table, masks). Coverage = share of the ELIGIBLE region covered by LOCAL valid DEM
    footprints (header bounds) and by the catalog (nominal footprints)."""
    from .dem_io import coverage_from_bounds

    elig = np.asarray(footprint, bool) & np.asarray(template_valid, bool) & ~np.asarray(labels, bool)
    local = coverage_from_bounds(sources, grid, local_only=True)
    catalog = coverage_from_bounds(sources, grid, local_only=False)
    rows = []
    regions = {"eligible": elig, **{f"V1_eval_zone_{d}": np.asarray(m, bool) & elig for d, m in (eval_zones or {})
                                    .items()}}
    for name, m in regions.items():
        n = int(m.sum())
        rows.append({"region": name, "px": n, "km2": n * 0.01,
                     "local_dem_share": float((local & m).sum() / max(n, 1)),
                     "catalog_dem_share": float((catalog & m).sum() / max(n, 1)),
                     "local_dem_km2": float((local & m).sum() * 0.01)})
    return pd.DataFrame(rows), {"local": local, "catalog": catalog, "eligible": elig}


def coverage_geojson(sources: list) -> dict:
    """Tile footprints (WGS84 polygons) with status / resolution / project attributes."""
    from rasterio.warp import transform_bounds

    feats = []
    for s in sources:
        if s.bounds is None:
            continue
        try:
            w, so, e, n = transform_bounds(s.crs or "EPSG:26911", "EPSG:4326", *s.bounds)
        except Exception:
            continue
        feats.append({"type": "Feature", "properties": {"tile_id": s.tile_id, "status": s.status, "res_m": s.res_m,
                                                        "project": s.project, "year": s.year,
                                                        "problems": ";".join(s.problems)},
                      "geometry": {"type": "Polygon", "coordinates": [[[w, so], [e, so], [e, n], [w, n], [w, so]]]}})
    return {"type": "FeatureCollection", "features": feats}


def inspect_existing_g06_store(g06_cfg: dict, roots) -> dict:
    """Locate the G06 run directory whose feature store is COMPLETE (explicit ``run_dir`` wins)."""
    cand = []
    if g06_cfg.get("run_dir"):
        cand = [Path(g06_cfg["run_dir"])]
    else:
        for r in roots:
            cand += [Path(p).parent.parent.parent for p in
                     glob.glob(str(Path(r) / g06_cfg.get("search_root", "outputs/g06_geological_feature_store") /
                                   "*" / "manifests" / "stages" / "features.json"))]
    rows = []
    for d in cand:
        m = d / "manifests" / "stages" / "features.json"
        s = d / "features" / "feature_schema.json"
        man = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
        sch = json.loads(s.read_text(encoding="utf-8")) if s.exists() else {}
        rows.append({"run_dir": str(d), "status": man.get("status"), "schema_hash": sch.get("schema_hash"),
                     "n_features": len(sch.get("feature_ids", [])), "written": man.get("written"),
                     "groups": sorted(sch.get("groups", {}))})
    ok = [r for r in rows if r["status"] in ("COMPLETE", "REUSED_VALID_CACHE") and r["schema_hash"]]
    best = sorted(ok, key=lambda r: str(r["written"]))[-1] if ok else None
    return {"candidates": rows, "selected": best,
            "status": "FOUND" if best else "MISSING",
            "detail": "" if best else "no COMPLETE G06 feature store: run the G06 notebook first-run configuration "
                                      "(or set CONFIG['g06_run_dir'])"}


def inspect_w2_artifacts(w2_cfg: dict, roots) -> dict:
    """Held-out W2 maps, full-region score and stacker gate of the P6 workspace."""
    out = {"maps": [], "full_score": None, "candidate": None}
    for r in roots:
        base = Path(r) / w2_cfg.get("p6_dir", "outputs/p6_fast_track") / "stack"
        if not base.exists():
            continue
        out["maps"] = sorted(p.name for p in (base / "maps").glob("*.npy"))
        fs = base / "maps" / "full_score.tif"
        out["full_score"] = str(fs) if fs.exists() else None
        c = base / "candidate.json"
        if c.exists():
            cand = json.loads(c.read_text(encoding="utf-8"))
            out["candidate"] = {k: cand.get(k) for k in ("gate", "features", "disc_dti", "geo_dti")}
        out["base"] = str(base)
        break
    out["status"] = "FOUND" if out["maps"] else "MISSING"
    out["full_region_status"] = "FOUND" if out["full_score"] else "MISSING (P6 stack.full_region not run: final " \
                                                                  "construction and export are BLOCKED_INPUT)"
    return out


def source_licenses(cfg: dict, dem_used: bool) -> pd.DataFrame:
    rows = [{"source": "GeoDAWN numerical raster, existing faults, template", "provider": "DrivenData GEMS",
             "license": "competition data (competition rules)", "usable_for_submission": True, "verified_by": "rules"}]
    dem = cfg.get("dem") or {}
    rows.append({"source": "USGS 3DEP 1 m DEM (TNM)", "provider": "USGS", "license": "public domain (USGS)",
                 "usable_for_submission": bool(dem.get("competition_eligibility_verified", False)),
                 "verified_by": "user flag dem.competition_eligibility_verified" if dem.get(
                     "competition_eligibility_verified") else "NOT VERIFIED - research use only until confirmed",
                 "used_in_this_run": dem_used})
    return pd.DataFrame(rows)


def build_dependency_report(items: dict) -> tuple[pd.DataFrame, dict]:
    """``items``: {name: {"mandatory": bool, "status": str, "detail": str, "blocks": [stages]}}."""
    rows = [{"dependency": k, **v, "blocks": ",".join(v.get("blocks", []))} for k, v in items.items()]
    T = pd.DataFrame(rows)
    ok = T.status.isin(["OK", "FOUND"])
    miss = {"mandatory_missing": list(T.loc[T.mandatory & ~ok, "dependency"]),
            "optional_missing": list(T.loc[~T.mandatory & ~ok, "dependency"]),
            "blocked_stages": sorted({s for r in rows if r["status"] not in ("OK", "FOUND") for s in
                                      str(r["blocks"]).split(",") if s})}
    return T, miss
