"""Content-level artifact validation: a directory existing is never evidence that a stage finished.

Every check opens the artifact (GeoTIFF header, Parquet schema, JSON keys, .npy header) and returns a
``Check`` instead of raising, so the pipeline can tell *missing* (nothing there), *partial* (some of a set),
*invalid* (present but unreadable / wrong grid / empty) and *ok* apart, and report the exact file.

Composite validators cover the artifacts the master pipeline reuses or regenerates:

* raw competition files (``check_geotiff`` / ``check_json``)
* geographic folds and discovery splits (``validate_folds_dir`` / ``validate_discovery_dir``)
* the minimal Phase-1 EDA set (``validate_eda_minimal``) and native segment rasters (``validate_native_segments``)
* feature caches (``check_npy``) and legacy run folders (``validate_legacy_run``)

``fingerprint`` gives a cheap (size + mtime) or exact (sha256) identity for manifests.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

OK, MISSING, PARTIAL, INVALID, INCOMPATIBLE = "ok", "missing", "partial", "invalid", "incompatible"


@dataclass
class Check:
    path: str
    status: str                     # ok | missing | invalid | incompatible
    reason: str = ""
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    """Result of validating one artifact set (all checks + an overall status)."""

    name: str
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.checks:
            return MISSING
        st = [c.status for c in self.checks]
        if all(s == OK for s in st):
            return OK
        if any(s == INCOMPATIBLE for s in st):
            return INCOMPATIBLE
        if any(s == INVALID for s in st):
            return INVALID
        if all(s == MISSING for s in st):
            return MISSING
        return PARTIAL

    @property
    def ok(self) -> bool:
        return self.status == OK

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        return c

    def summary(self) -> str:
        bad = self.failed()
        if not bad:
            return f"{self.name}: ok ({len(self.checks)} checks)"
        head = "; ".join(f"{Path(c.path).name}: {c.status}{' - ' + c.reason if c.reason else ''}" for c in bad[:4])
        more = f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""
        return f"{self.name}: {self.status} - {head}{more}"

    def table(self):
        import pandas as pd

        return pd.DataFrame([{"set": self.name, **c.as_dict()} for c in self.checks])


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------
def fingerprint(path: Path, sha: bool = False) -> dict | None:
    """{"bytes", "mtime_ns"[, "sha256"]} of a file; None when absent. Directories: aggregated over files."""
    p = Path(path)
    if not p.exists():
        return None
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file() and not f.name.endswith(".partial"))
        return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
    st = p.stat()
    out = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if sha:
        from .atomic import sha256_file

        out["sha256"] = sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# single-file checks
# ---------------------------------------------------------------------------
def same_grid(a: dict, b: dict, tol: float = 1e-6) -> bool:
    """Two rasterio profiles (or dicts with crs / transform / height / width) describe the same grid."""
    if int(a["height"]) != int(b["height"]) or int(a["width"]) != int(b["width"]):
        return False
    if str(a.get("crs")) != str(b.get("crs")):
        return False
    ta, tb = tuple(a["transform"])[:6], tuple(b["transform"])[:6]
    return all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in zip(ta, tb))


def grid_of(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        return {"crs": src.crs, "transform": src.transform, "height": src.height, "width": src.width,
                "count": src.count, "dtype": src.dtypes[0], "descriptions": list(src.descriptions)}


def check_geotiff(path: Path, *, grid: dict | None = None, count: int | None = None, min_count: int | None = None,
                  bands: Iterable[str] | None = None, min_bytes: int = 1, expected_bytes: int | None = None) -> Check:
    """Opens the GeoTIFF (fails on HTML error pages / truncation), then optional grid / band / size checks."""
    p = Path(path)
    if not p.exists():
        return Check(str(p), MISSING, "file not found")
    size = p.stat().st_size
    if size < min_bytes:
        return Check(str(p), INVALID, f"{size} bytes")
    try:
        g = grid_of(p)
    except Exception as e:                              # rasterio raises several error types
        return Check(str(p), INVALID, f"not a readable GeoTIFF: {type(e).__name__}: {str(e)[:120]}")
    info = {"bytes": size, "count": g["count"], "height": g["height"], "width": g["width"], "crs": str(g["crs"])}
    if count is not None and g["count"] != count:
        return Check(str(p), INVALID, f"{g['count']} bands, expected {count}", info)
    if min_count is not None and g["count"] < min_count:
        return Check(str(p), INVALID, f"{g['count']} bands, expected >= {min_count}", info)
    if bands:
        missing = [b for b in bands if b not in g["descriptions"]]
        if missing:
            return Check(str(p), INVALID, f"band descriptions missing: {missing}", info)
    if grid is not None and not same_grid(g, grid):
        return Check(str(p), INCOMPATIBLE, "grid (CRS / transform / shape) differs from the native grid", info)
    if expected_bytes and size != expected_bytes:
        info["size_note"] = f"{size} bytes, known release {expected_bytes} (new data release?)"
    return Check(str(p), OK, "", info)


def check_json(path: Path, *, required_keys: Iterable[str] = (), non_empty: bool = True) -> Check:
    p = Path(path)
    if not p.exists():
        return Check(str(p), MISSING, "file not found")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return Check(str(p), INVALID, f"invalid JSON: {str(e)[:120]}")
    if non_empty and not obj:
        return Check(str(p), INVALID, "empty JSON document")
    if isinstance(obj, dict):
        miss = [k for k in required_keys if k not in obj]
        if miss:
            return Check(str(p), INVALID, f"keys missing: {miss}")
    elif required_keys:
        return Check(str(p), INVALID, "expected a JSON object")
    n = len(obj) if isinstance(obj, (dict, list)) else 1
    return Check(str(p), OK, "", {"entries": n})


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def check_parquet(path: Path, *, columns: Iterable[str] = (), min_rows: int = 1) -> Check:
    """Schema + row count from the Parquet footer (no full read)."""
    p = Path(path)
    if not p.exists():
        return Check(str(p), MISSING, "file not found")
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(p).metadata
        names = list(meta.schema.names)
        rows = meta.num_rows
    except Exception as e:
        return Check(str(p), INVALID, f"unreadable Parquet: {str(e)[:120]}")
    miss = [c for c in columns if c not in names]
    if miss:
        return Check(str(p), INVALID, f"columns missing: {miss}", {"rows": rows})
    if rows < min_rows:
        return Check(str(p), INVALID, f"{rows} rows (< {min_rows})", {"rows": rows})
    return Check(str(p), OK, "", {"rows": rows, "columns": len(names)})


def check_csv(path: Path, *, columns: Iterable[str] = (), min_rows: int = 1) -> Check:
    p = Path(path)
    if not p.exists():
        return Check(str(p), MISSING, "file not found")
    try:
        import pandas as pd

        df = pd.read_csv(p)
    except Exception as e:
        return Check(str(p), INVALID, f"unreadable CSV: {str(e)[:120]}")
    miss = [c for c in columns if c not in df.columns]
    if miss:
        return Check(str(p), INVALID, f"columns missing: {miss}", {"rows": len(df)})
    if len(df) < min_rows:
        return Check(str(p), INVALID, f"{len(df)} rows (< {min_rows})", {"rows": len(df)})
    return Check(str(p), OK, "", {"rows": len(df)})


def check_npy(path: Path, *, shape: tuple | None = None, dtype: str | None = "float32") -> Check:
    """.npy header only (memory-mapped open): shape / dtype of a cached feature."""
    p = Path(path)
    if not p.exists():
        return Check(str(p), MISSING, "file not found")
    try:
        import numpy as np

        a = np.load(p, mmap_mode="r")
        sh, dt = tuple(a.shape), str(a.dtype)
        del a
    except Exception as e:
        return Check(str(p), INVALID, f"unreadable .npy: {str(e)[:120]}")
    if shape is not None and sh != tuple(shape):
        return Check(str(p), INCOMPATIBLE, f"shape {sh}, native grid {tuple(shape)}")
    if dtype and dt != dtype:
        return Check(str(p), INVALID, f"dtype {dt}, expected {dtype}")
    return Check(str(p), OK, "", {"shape": list(sh)})


def check_vector(path: Path, *, min_features: int = 1) -> Check:
    """GeoPackage / vector file: feature count through pyogrio or geopandas (size check if neither exists)."""
    p = Path(path)
    if not p.exists():
        return Check(str(p), MISSING, "file not found")
    try:
        import pyogrio

        n = int(pyogrio.read_info(p)["features"])
    except ImportError:
        try:
            import geopandas as gpd

            n = len(gpd.read_file(p))
        except ImportError:
            ok = p.stat().st_size > 0
            return Check(str(p), OK if ok else INVALID, "" if ok else "empty file", {"note": "no vector reader installed"})
        except Exception as e:
            return Check(str(p), INVALID, f"unreadable vector file: {str(e)[:120]}")
    except Exception as e:
        return Check(str(p), INVALID, f"unreadable vector file: {str(e)[:120]}")
    if n < min_features:
        return Check(str(p), INVALID, f"{n} features (< {min_features})")
    return Check(str(p), OK, "", {"features": n})


# ---------------------------------------------------------------------------
# composite validators
# ---------------------------------------------------------------------------
FOLD_FILES = ("fold_map.tif", "blocks.parquet", "fold_params.json")
DISCOVERY_FILES = ("discovery_map.tif", "fault_systems.csv", "discovery_params.json")


def _params_match(saved: dict, expected: dict) -> list[str]:
    out = []
    for k, v in expected.items():
        if k not in saved:
            out.append(f"{k}: not recorded")
            continue
        s = saved[k]
        if isinstance(v, float) or isinstance(s, float):
            try:
                if abs(float(s) - float(v)) > 1e-9:
                    out.append(f"{k}: saved {s} != {v}")
            except (TypeError, ValueError):
                out.append(f"{k}: saved {s} != {v}")
        elif s != v:
            out.append(f"{k}: saved {s} != {v}")
    return out


def validate_folds_dir(directory: Path, grid: dict | None = None, expected: dict | None = None) -> Report:
    """GeoFolds folder: fold_map.tif (1 + K bands, native grid), blocks.parquet, fold_params.json (params match)."""
    d = Path(directory)
    rep = Report(f"folds:{d.name}")
    k = (expected or {}).get("n_folds")
    rep.add(check_geotiff(d / "fold_map.tif", grid=grid, count=(1 + int(k)) if k else None, min_count=2))
    rep.add(check_parquet(d / "blocks.parquet", min_rows=1))
    c = rep.add(check_json(d / "fold_params.json", required_keys=("n_folds",) if expected else ()))
    if c.ok and expected:
        diff = _params_match(read_json(d / "fold_params.json", {}), expected)
        if diff:
            c.status, c.reason = INCOMPATIBLE, "; ".join(diff)
    return rep


def validate_discovery_dir(directory: Path, grid: dict | None = None, expected: dict | None = None) -> Report:
    d = Path(directory)
    rep = Report(f"discovery:{d.name}")
    n = (expected or {}).get("n_splits")
    rep.add(check_geotiff(d / "discovery_map.tif", grid=grid, count=(1 + int(n)) if n else None, min_count=2))
    rep.add(check_csv(d / "fault_systems.csv", columns=("system_id", "trace_km"), min_rows=1))
    c = rep.add(check_json(d / "discovery_params.json", required_keys=("n_splits",) if expected else ()))
    if c.ok and expected:
        diff = _params_match(read_json(d / "discovery_params.json", {}), expected)
        if diff:
            c.status, c.reason = INCOMPATIBLE, "; ".join(diff)
    return rep


# Minimal Phase-1 artifact set consumed downstream (segments.py, features.FeatureStore._eda, p3.context).
# Installed / validated as groups: members of one group must come from the same EDA run (seg_id consistency).
EDA_GROUPS = {
    "segments": {
        "datasets/known_fault_segments.parquet": ("parquet", ("seg_id", "length_km", "azimuth", "sinuosity")),
        "datasets/known_fault_segments.gpkg": ("vector", ()),
        "datasets/known_fault_nodes.parquet": ("parquet", ()),
        "datasets/segment_archetypes.parquet": ("parquet", ("seg_id", "primary_archetype")),
    },
    "lineaments": {
        "features/lineament_maps.tif": ("tif", ("fused_score", "fused_fused", "fused_support", "response_terrain")),
    },
    "crossmodal": {
        "features/crossmodal_orientation.tif": ("tif", ("edge_agreement_count",)),
    },
}
EDA_MANIFEST_NAMES = {
    "datasets/known_fault_segments.parquet": "known_fault_segments",
    "datasets/known_fault_segments.gpkg": "known_fault_segments_gpkg",
    "datasets/known_fault_nodes.parquet": "known_fault_nodes",
    "datasets/segment_archetypes.parquet": "segment_archetypes",
    "features/lineament_maps.tif": "lineament_maps",
    "features/crossmodal_orientation.tif": "crossmodal_orientation",
}


def check_eda_file(path: Path, spec: tuple) -> Check:
    kind, cols = spec
    if kind == "parquet":
        return check_parquet(path, columns=cols, min_rows=1)
    if kind == "vector":
        return check_vector(path, min_features=1)
    return check_geotiff(path, bands=cols)


def validate_eda_group(eda_dir: Path, group: str) -> Report:
    rep = Report(f"eda:{group}")
    for rel, spec in EDA_GROUPS[group].items():
        rep.add(check_eda_file(Path(eda_dir) / rel, spec))
    return rep


def validate_eda_minimal(eda_dir: Path) -> Report:
    """Every group + manifest.json listing each artifact (manifest is written by the EDA / the rebuild)."""
    rep = Report("eda_minimal")
    for g in EDA_GROUPS:
        for c in validate_eda_group(eda_dir, g).checks:
            rep.add(c)
    man = Path(eda_dir) / "manifest.json"
    c = rep.add(check_json(man))
    if c.ok:
        entries = read_json(man, {})
        miss = [n for n in EDA_MANIFEST_NAMES.values() if n not in entries]
        if miss:
            c.status, c.reason = INVALID, f"manifest lacks entries {miss}"
    return rep


def validate_native_segments(bench_out: Path, grid: dict | None = None) -> Report:
    d = Path(bench_out) / "datasets"
    rep = Report("native_segments")
    rep.add(check_geotiff(d / "native_segment_ids.tif", grid=grid, count=1, bands=("seg_id",)))
    rep.add(check_parquet(d / "segment_attributes.parquet", columns=("seg_id", "length_km", "primary_archetype"),
                          min_rows=1))
    return rep


def validate_legacy_run(run_dir: Path, *, grid: dict | None = None, require_oof: bool = True,
                        require_models: bool = False) -> Report:
    """A finished persist.ExperimentRun folder: run.json status done, config.json, oof.tif (native grid)."""
    d = Path(run_dir)
    rep = Report(f"run:{d.name}")
    c = rep.add(check_json(d / "run.json"))
    if c.ok and (read_json(d / "run.json", {}) or {}).get("status") != "done":
        c.status, c.reason = INVALID, "run.json status is not 'done' (interrupted run)"
    rep.add(check_json(d / "config.json"))
    if require_oof:
        rep.add(check_geotiff(d / "oof.tif", grid=grid))
    if require_models:
        m = d / "models"
        has = m.is_dir() and any(f.is_file() for f in m.iterdir())
        rep.add(Check(str(m), OK if has else MISSING, "" if has else "no model files (checkpoint absent)"))
    return rep


def validate_discovery_run(run_dir: Path, grid: dict | None = None) -> Report:
    d = Path(run_dir)
    rep = Report(f"disc:{d.name}")
    rep.add(check_geotiff(d / "pred.tif", grid=grid))
    return rep


def merge(name: str, reports: Iterable[Report]) -> Report:
    out = Report(name)
    for r in reports:
        out.checks.extend(r.checks)
        out.notes.extend(r.notes)
    return out
