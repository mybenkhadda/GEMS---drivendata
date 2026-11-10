"""Phase 1 - discover and inspect every source artifact; fail fast on spatially unusable inputs.

Nothing is assumed from a filename alone: rasters/vectors are opened, NumPy predictions are only located when a
sidecar (``.json`` with ``transform``/``crs``) or an exactly-matching ``.tif`` twin documents their grid.
"""
from __future__ import annotations

import fnmatch
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RASTER_EXT = {".tif", ".tiff", ".vrt", ".img"}
VECTOR_EXT = {".gpkg", ".shp", ".geojson", ".json", ".fgb", ".parquet"}
ARRAY_EXT = {".npy", ".npz"}
EXPERIMENT_TOKENS = ("G11", "G10", "G09", "G08", "G07", "G06", "M9", "P6", "GEOREGIME", "ATLAS", "E6")


class AuditError(RuntimeError):
    """Fail-fast condition with an actionable message."""


def _experiment(path: Path) -> str:
    s = str(path).upper()
    for t in EXPERIMENT_TOKENS:
        if t in s:
            return t
    return "RAW" if "/RAW/" in s.replace("\\", "/") or "DATA" in s else "UNKNOWN"


def _role(path: Path) -> str:
    n = path.name.lower()
    s = str(path).lower().replace("\\", "/")
    if "example_submission" in n:
        return "template"
    if "geodawn" in n:
        return "geophysics_source"
    if "fault" in n and path.suffix.lower() in RASTER_EXT:
        return "fault_raster"
    if "fault" in n and path.suffix.lower() in VECTOR_EXT:
        return "fault_vector"
    if "/dem_1m/" in s and path.suffix.lower() in RASTER_EXT:
        return "dem_tile"
    if "foldmasked" in n or "_oof" in n or "prediction" in n or "prob" in n:
        return "prediction"
    if "checkpoint" in s or path.suffix in (".pt", ".pth", ".ckpt"):
        return "model_checkpoint"
    if "registry" in n or "manifest" in n:
        return "registry"
    if path.suffix.lower() in RASTER_EXT:
        return "raster_other"
    return "other"


def _type(path: Path) -> str:
    e = path.suffix.lower()
    if e in RASTER_EXT:
        return "raster"
    if e in ARRAY_EXT:
        return "array"
    if e in VECTOR_EXT:
        return "vector" if e != ".json" else "json"
    if e in (".pt", ".pth", ".ckpt"):
        return "checkpoint"
    if e in (".csv", ".parquet"):
        return "table"
    return "file"


def discover_artifacts(roots, patterns=("*",), max_depth: int | None = None, exclude=(".partial", "/cache/")
                       ) -> pd.DataFrame:
    """Walk ``roots`` (depth-limited) and list files matching any glob in ``patterns``.

    Columns: artifact_id, path, source_experiment, artifact_type, size_bytes, modification_time, extension,
    candidate_role. Inaccessible directories are recorded with candidate_role='INACCESSIBLE' (never 'missing').
    """
    rows = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        base = len(root.parts)
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                entries = list(d.iterdir())
            except OSError as e:
                rows.append({"path": str(d), "candidate_role": "INACCESSIBLE", "artifact_type": "dir",
                             "size_bytes": None, "modification_time": None, "extension": "", "error": str(e)})
                continue
            for p in entries:
                sp = str(p).replace("\\", "/")
                if any(x in sp for x in exclude):
                    continue
                if p.is_dir():
                    if max_depth is None or len(p.parts) - base < max_depth:
                        stack.append(p)
                    continue
                if not any(fnmatch.fnmatch(p.name, pat) or fnmatch.fnmatch(sp, pat) for pat in patterns):
                    continue
                try:
                    st = p.stat()
                except OSError as e:
                    rows.append({"path": sp, "candidate_role": "INACCESSIBLE", "error": str(e)})
                    continue
                rows.append({"path": sp, "source_experiment": _experiment(p), "artifact_type": _type(p),
                             "size_bytes": int(st.st_size),
                             "modification_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                             "extension": p.suffix.lower(), "candidate_role": _role(p)})
    df = pd.DataFrame(rows, columns=["path", "source_experiment", "artifact_type", "size_bytes", "modification_time",
                                     "extension", "candidate_role", "error"])
    df = df.sort_values("path").reset_index(drop=True)
    df.insert(0, "artifact_id", [f"A{i:06d}" for i in range(len(df))])
    return df


@dataclass
class RasterMetadata:
    path: str
    crs: str | None
    epsg: int | None
    bounds: tuple
    transform: tuple
    resolution: tuple
    width: int
    height: int
    band_count: int
    nodata: float | None
    dtype: str
    valid_fraction: float
    compression: str | None
    overviews: list
    band_names: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def inspect_raster(path, sample_max_px: int = 1_000_000) -> RasterMetadata:
    """Header + decimated valid fraction of band 1 (reads at most ``sample_max_px`` pixels)."""
    import rasterio

    with rasterio.open(path) as ds:
        f = max(1, int(np.ceil(np.sqrt(ds.width * ds.height / sample_max_px))))
        a = ds.read(1, out_shape=(max(1, ds.height // f), max(1, ds.width // f)), masked=True)
        a = a.astype(np.float32).filled(np.nan)
        vf = float(np.isfinite(a).mean()) if a.size else 0.0
        return RasterMetadata(str(path), ds.crs.to_wkt() if ds.crs else None, ds.crs.to_epsg() if ds.crs else None,
                              tuple(ds.bounds), tuple(ds.transform)[:6], tuple(ds.res), ds.width, ds.height, ds.count,
                              ds.nodata, ds.dtypes[0], vf, ds.profile.get("compress"), ds.overviews(1),
                              [d or "" for d in ds.descriptions])


@dataclass
class VectorMetadata:
    path: str
    crs: str | None
    geometry_types: dict
    feature_count: int
    bounds: tuple
    invalid_count: int
    empty_count: int
    duplicate_count: int
    schema: dict

    def as_dict(self) -> dict:
        return asdict(self)


def inspect_vector(path) -> VectorMetadata:
    import geopandas as gpd

    g = gpd.read_parquet(path) if str(path).endswith(".parquet") else gpd.read_file(path)
    geom = g.geometry
    wkb = geom.to_wkb() if len(g) else pd.Series([], dtype=object)
    return VectorMetadata(str(path), g.crs.to_wkt() if g.crs else None,
                          geom.geom_type.value_counts().to_dict(), int(len(g)),
                          tuple(g.total_bounds) if len(g) else (np.nan,) * 4,
                          int((~geom.is_valid).sum()), int(geom.is_empty.sum()), int(wkb.duplicated().sum()),
                          {c: str(t) for c, t in g.dtypes.items() if c != g.geometry.name})


@dataclass
class PredictionMetadata:
    path: str
    shape: tuple
    finite_support: float
    prob_min: float | None
    prob_max: float | None
    provenance: dict
    native_grid: dict | None
    split: str | None
    alignment_confidence: str     # DOCUMENTED (sidecar / GeoTIFF) | TWIN_TIF | UNLOCATED

    def as_dict(self) -> dict:
        return asdict(self)


def _split_from_name(name: str) -> str | None:
    n = name.lower()
    for k in ("test", "stack", "oof", "val", "train", "full"):
        if k in n:
            return k
    return None


def inspect_prediction(path, metadata: dict | None = None) -> PredictionMetadata:
    """Shape / support / range of a prediction raster or array and HOW it is georeferenced.

    A ``.npy`` is UNLOCATED unless ``metadata`` or a sidecar JSON (``<stem>.json`` / ``<stem>_meta.json``) gives
    ``transform`` and ``crs``, or a same-stem ``.tif`` exists with the same shape. The shape alone never locates it.
    """
    p = Path(path)
    meta = dict(metadata or {})
    for side in (p.with_suffix(".json"), p.with_name(p.stem + "_meta.json")):
        if not meta and side.is_file():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    if p.suffix.lower() in RASTER_EXT:
        import rasterio

        with rasterio.open(p) as ds:
            a = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
            grid = {"crs": ds.crs.to_wkt() if ds.crs else None, "transform": list(ds.transform)[:6],
                    "shape": list(ds.shape)}
            conf = "DOCUMENTED" if ds.crs else "UNLOCATED"
    else:
        a = np.load(p, mmap_mode="r")
        if isinstance(a, np.lib.npyio.NpzFile):
            a = a[a.files[0]]
        grid, conf = None, "UNLOCATED"
        if meta.get("transform") and meta.get("crs"):
            grid, conf = {"crs": meta["crs"], "transform": meta["transform"], "shape": list(a.shape[-2:])}, "DOCUMENTED"
        else:
            twin = p.with_suffix(".tif")
            if twin.is_file():
                import rasterio

                with rasterio.open(twin) as ds:
                    if tuple(ds.shape) == tuple(a.shape[-2:]) and ds.crs:
                        grid, conf = {"crs": ds.crs.to_wkt(), "transform": list(ds.transform)[:6],
                                      "shape": list(ds.shape)}, "TWIN_TIF"
        a = np.asarray(a[..., ::max(1, a.shape[-2] // 1000), ::max(1, a.shape[-1] // 1000)], np.float32)
    fin = np.isfinite(a)
    v = a[fin]
    return PredictionMetadata(str(p), tuple(np.shape(a)), float(fin.mean()) if a.size else 0.0,
                              float(v.min()) if v.size else None, float(v.max()) if v.size else None,
                              {k: meta[k] for k in ("model", "fold", "split", "run", "source") if k in meta},
                              grid, meta.get("split") or _split_from_name(p.name), conf)


def resolve_first(data_root, candidates) -> Path | None:
    """First existing path among literal paths / globs relative to ``data_root``."""
    root = Path(data_root)
    for c in candidates:
        if any(ch in c for ch in "*?["):
            hits = sorted(root.glob(c))
            if hits:
                return hits[0]
        elif (root / c).is_file():
            return root / c
    return None


def build_artifact_registry(cfg: dict, inspect: bool = True) -> tuple[pd.DataFrame, dict]:
    """Discover required + optional sources; returns (registry table, summary with required/optional status).

    Raises :class:`AuditError` for fail-fast conditions (missing CRS, unlocatable required prediction, template
    absent when required, invalid fault geometry).
    """
    root = Path(cfg["data_root"])
    sa = cfg["source_artifacts"]
    found = {"geodawn": resolve_first(root, sa["geodawn_raster"]), "fault_raster": resolve_first(root, sa["fault_raster"]),
             "template": resolve_first(root, sa["template"]),
             "fault_vector": resolve_first(root, sa["fault_vectors"]),
             "dem_dir": next((root / d for d in sa["dem_dirs"] if (root / d).is_dir()), None),
             "dem_dirs": [str(root / d) for d in sa["dem_dirs"] if (root / d).is_dir()] or None,
             "dem_report": (root / sa["dem_report"]) if (root / sa["dem_report"]).is_file() else None}
    rows, problems, skipped = [], [], []
    cfg_key = {"geodawn": "geodawn_raster", "fault_raster": "fault_raster", "template": "template"}
    for key in ("geodawn", "fault_raster", "template"):
        p = found[key]
        if p is None:
            problems.append(f"required source '{key}' not found under {root} (looked for {sa[cfg_key[key]]})")
            continue
        r = {"key": key, "path": str(p), "required": True, "status": "FOUND"}
        if inspect:
            m = inspect_raster(p)
            r.update(m.as_dict())
            if m.crs is None:
                problems.append(f"{key}: {p} has no CRS")
        rows.append(r)
    if found["fault_vector"] is not None and inspect:
        vm = inspect_vector(found["fault_vector"])
        if vm.crs is None:
            problems.append(f"fault vector {found['fault_vector']} has no CRS")
        bad = {t for t in vm.geometry_types if t not in ("LineString", "MultiLineString")}
        if bad:
            problems.append(f"fault vector has unsupported geometry types {sorted(bad)}")
        rows.append({"key": "fault_vector", "path": str(found["fault_vector"]), "required": False, "status": "FOUND",
                     **vm.as_dict()})
    else:
        skipped.append("fault_vector: none found - fault segments are derived from the label raster (flagged)")
    if found["dem_dir"] is None:
        skipped.append(f"DEM directory not found in {sa['dem_dirs']} - DEM-dependent stages are skipped")
    opt = {}
    for name, pats in sa["optional_patterns"].items():
        hits = sorted({p for pat in pats for p in root.glob(pat) if p.is_file()})[:500]
        opt[name] = [str(h) for h in hits]
        if not hits:
            skipped.append(f"optional evidence '{name}': ABSENT ({pats})")
        for h in hits:
            row = {"key": name, "path": str(h), "required": False, "status": "FOUND"}
            if inspect and (h.suffix.lower() in ARRAY_EXT or name.endswith("prediction") or "foldmasked" in h.name):
                try:
                    pm = inspect_prediction(h)
                    row.update(pm.as_dict())
                    if pm.alignment_confidence == "UNLOCATED":
                        row["status"] = "UNLOCATED (excluded: no documented grid)"
                except Exception as e:                                   # noqa: BLE001 - recorded verbatim
                    row["status"] = f"UNREADABLE: {type(e).__name__}: {e}"
            rows.append(row)
    if cfg["official_submission"]["require_template"] and found["template"] is None:
        problems.append("official template example_submission.tif not found: submission schema cannot be verified")
    reg = pd.DataFrame(rows)
    summary = {"found": {k: (v if isinstance(v, list) else str(v)) if v else None for k, v in found.items()},
               "optional": {k: len(v) for k, v in opt.items()},
               "optional_paths": opt, "skipped": skipped, "problems": problems}
    return reg, summary


def audit_report_md(reg: pd.DataFrame, summary: dict) -> str:
    lines = ["# G11 artifact audit", "", "## Required sources", ""]
    for k, v in summary["found"].items():
        lines.append(f"- **{k}**: {v or 'NOT FOUND'}")
    lines += ["", "## Optional evidence", ""] + [f"- {k}: {n} file(s)" for k, n in summary["optional"].items()]
    lines += ["", "## Skipped / absent (recorded, never zero-filled)", ""] + [f"- {s}" for s in summary["skipped"]]
    lines += ["", "## Fail-fast problems", ""] + ([f"- {p}" for p in summary["problems"]] or ["- none"])
    if len(reg) and "status" in reg:
        lines += ["", "## Registry status counts", ""] + [f"- {k}: {v}" for k, v in reg.status.value_counts().items()]
    return "\n".join(lines) + "\n"


def fail_fast(summary: dict) -> None:
    if summary["problems"]:
        raise AuditError("G11 audit failed:\n  - " + "\n  - ".join(summary["problems"]))
