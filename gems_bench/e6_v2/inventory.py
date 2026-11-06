"""Raw-data inventory: every raw input the pipeline may read, with checksum and header facts.

Writes ``manifests/raw_data_inventory.json`` and ``.csv`` (columns: source_id, type, path, size, checksum,
checksum_kind, crs, resolution, shape, bands, nodata, bounds, vertical_unit, validation_status) and the raw
fingerprint that keys every downstream cache. Checksums of unchanged files (same path, size and mtime) are
reused from ``manifests/inventory_hash_cache.json`` so a new session does not re-hash gigabytes.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .cache import atomic_write_csv, atomic_write_json, is_temporary, quick_sha256, read_json, sha256_file
from .config import stable_hash
from .contracts import check_band_contract
from .dem_sources import discover_tiles, read_tile_header, validate_tiles
from .paths import E6V2Paths

COLUMNS = ["source_id", "type", "path", "size", "checksum", "checksum_kind", "crs", "resolution", "shape", "bands",
           "nodata", "bounds", "vertical_unit", "validation_status"]
RAW_KEYS = {"features": "features_file", "faults": "faults_file", "template": "template_file"}


def _forbidden(path: Path, drive_root: Path, cfg: dict) -> bool:
    for f in cfg["raw"]["forbidden_prefixes"]:
        try:
            path.resolve().relative_to((drive_root / f).resolve())
            return True
        except ValueError:
            continue
    return False


def locate_raw(paths: E6V2Paths, cfg: dict) -> dict:
    """{key: {"path": Path | None, "candidates": [...], "issue": str | None}} for the three GeoDAWN files."""
    out = {}
    for key, ck in RAW_KEYS.items():
        name = cfg["raw"][ck]
        cands = []
        for sub in cfg["raw"]["search_dirs"]:
            p = paths.drive_root / sub / name
            if p.is_file() and not is_temporary(p) and not _forbidden(p, paths.drive_root, cfg):
                cands.append(p)
        issue = None
        if not cands:
            issue = f"{name} not found in {[str(paths.drive_root / s) for s in cfg['raw']['search_dirs']]}"
        elif len({c.stat().st_size for c in cands}) > 1:
            issue = f"{name} found in several places with different sizes: {[str(c) for c in cands]}"
        out[key] = {"path": cands[0] if cands else None, "candidates": [str(c) for c in cands], "issue": issue}
    return out


class HashCache:
    def __init__(self, path: Path):
        self.path = path
        self.data = read_json(path, {}) or {}

    def get(self, p: Path, kind: str) -> str:
        st = p.stat()
        key = f"{p}|{kind}"
        rec = self.data.get(key)
        if rec and rec["size"] == st.st_size and rec["mtime"] == int(st.st_mtime):
            return rec["checksum"]
        cs = sha256_file(p) if kind == "sha256" else quick_sha256(p)
        self.data[key] = {"size": st.st_size, "mtime": int(st.st_mtime), "checksum": cs}
        return cs

    def save(self):
        atomic_write_json(self.path, self.data)


def _raster_row(source_id: str, typ: str, p: Path, hc: HashCache) -> dict:
    import rasterio

    from .raw_data import band_names

    row = {"source_id": source_id, "type": typ, "path": str(p), "size": p.stat().st_size,
           "checksum": hc.get(p, "sha256"), "checksum_kind": "sha256", "vertical_unit": ""}
    try:
        with rasterio.open(p) as src:
            row.update(crs=f"EPSG:{src.crs.to_epsg()}" if src.crs and src.crs.to_epsg() else (src.crs.to_wkt() if src.crs else ""),
                       resolution=f"{abs(src.transform.a):g}x{abs(src.transform.e):g}",
                       shape=f"{src.count}x{src.height}x{src.width}", bands="|".join(band_names(src)),
                       nodata=src.nodata, bounds="|".join(f"{v:.3f}" for v in src.bounds), validation_status="ok")
    except Exception as exc:
        row.update(validation_status=f"rejected:{type(exc).__name__}: {exc}")
    return row


def build_inventory(paths: E6V2Paths, cfg: dict, with_dem: bool = True) -> dict:
    """Locate, open and hash every raw input; write the inventory; return {rows, fingerprint, issues}."""
    paths.manifests.mkdir(parents=True, exist_ok=True)
    hc = HashCache(paths.manifests / "inventory_hash_cache.json")
    loc = locate_raw(paths, cfg)
    rows, issues = [], []
    for key, info in loc.items():
        if info["issue"]:
            issues.append(info["issue"])
        if info["path"] is not None:
            row = _raster_row(f"geodawn_{key}", f"geodawn_{key}", info["path"], hc)
            rows.append(row)
            if not row["validation_status"].startswith("ok"):
                issues.append(f"{key}: {row['validation_status']}")
    feat = next((r for r in rows if r["source_id"] == "geodawn_features"), None)
    if feat and feat["validation_status"] == "ok":
        issues += check_band_contract(cfg, feat["bands"].split("|"))
    tiles = pd.DataFrame()
    if with_dem:
        found, rejected = discover_tiles(paths.drive_root, cfg)
        target_wkt = _target_wkt(cfg)
        hdr = [read_tile_header(p, cfg, target_wkt) for p in found]
        tiles = validate_tiles(pd.DataFrame(hdr), cfg) if hdr else pd.DataFrame()
        for r in rejected:
            rows.append({"source_id": f"dem_rejected_{len(rows)}", "type": "dem_1m", "path": r["path"],
                         "validation_status": f"rejected:{r['reason']}"})
        for i, r in enumerate(tiles.to_dict("records")):
            p = Path(r["path"])
            rows.append({"source_id": f"dem_{i:05d}_{p.stem}", "type": "dem_1m", "path": str(p), "size": r["bytes"],
                         "checksum": hc.get(p, "quick") if r["validation_status"] != "rejected:header_error" else "",
                         "checksum_kind": "sha256(size,first MiB,last MiB)",
                         "crs": f"EPSG:{r.get('crs_epsg')}" if r.get("crs_epsg") else "",
                         "resolution": f"{r.get('res_x', float('nan')):g}x{r.get('res_y', float('nan')):g}",
                         "shape": f"{r.get('count', '')}x{r.get('height', '')}x{r.get('width', '')}", "bands": "",
                         "nodata": r.get("nodata"), "bounds": "|".join(
                             f"{r.get(k, float('nan')):.3f}" for k in ("left", "bottom", "right", "top")),
                         "vertical_unit": f"{r.get('vertical_unit', '')} ({r.get('vertical_unit_source', '')})",
                         "validation_status": r["validation_status"]})
            if r.get("metadata_xml"):
                xp = p.with_name(r["metadata_xml"])
                rows.append({"source_id": f"demmeta_{i:05d}_{xp.stem}", "type": "dem_1m_metadata", "path": str(xp),
                             "size": xp.stat().st_size, "checksum": hc.get(xp, "sha256"), "checksum_kind": "sha256",
                             "validation_status": "ok"})
        if tiles.empty:
            issues.append(f"no DEM GeoTIFF under {[str(paths.drive_root / d) for d in cfg['raw']['dem_dirs']]}")
    hc.save()
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    fp = stable_hash(sorted((r["source_id"], r.get("checksum") or "", r.get("validation_status") or "") for r in rows))
    atomic_write_csv(paths.manifests / "raw_data_inventory.csv", df)
    atomic_write_json(paths.manifests / "raw_data_inventory.json",
                      {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "raw_fingerprint": fp,
                       "drive_root": str(paths.drive_root), "issues": issues,
                       "dem_tiles_ok": int((tiles.validation_status == "ok").sum()) if not tiles.empty else 0,
                       "rows": df.where(pd.notna(df), None).to_dict("records")})
    if not tiles.empty:
        atomic_write_csv(paths.dem_dir("source_inventory") / "dem_sources.csv", tiles)
    return {"rows": df, "fingerprint": fp, "issues": issues, "located": loc, "dem_tiles": tiles}


def _target_wkt(cfg: dict) -> str:
    from rasterio.crs import CRS

    return CRS.from_epsg(int(cfg["grid_expected"]["epsg"])).to_wkt()


def load_inventory(paths: E6V2Paths) -> dict:
    inv = read_json(paths.manifests / "raw_data_inventory.json")
    if inv is None:
        from .cache import MissingArtifact

        raise MissingArtifact("manifests/raw_data_inventory.json missing - run section 02 (raw data inventory)")
    return inv


def raw_path(inv: dict, source_id: str) -> Path:
    for r in inv["rows"]:
        if r["source_id"] == source_id and str(r.get("validation_status", "")).startswith("ok"):
            return Path(r["path"])
    from .cache import MissingArtifact

    raise MissingArtifact(f"{source_id} not in the raw-data inventory (or rejected) - see section 02")
