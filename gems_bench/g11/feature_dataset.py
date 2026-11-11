"""Finalized, immutable, versioned feature datasets: ``03_features/datasets/<version>/``.

A dataset version is a manifest over verified pool layers (no 30 GB copy is made). It separates

* **raw source data**   - referenced by hash only (``source_provenance.json``);
* **feature data**      - predictor layers, grouped into a geophysical cube and a DEM 100 m summary cube
                          (``feature_schema.parquet``; each row = channel name, pool layer id, file, sha256, unit,
                          scale, processing resolution, availability);
* **label data**        - fault targets (``label_schema.parquet``) and the fault geometry registry - never predictors;
* split assignments and training datasets are NOT part of a feature dataset (they live in ``05_splits/<version>``
  and ``06_models/.../training_data``), so changing a split, a loss or a hyper-parameter never touches it.

Also stored: the geophysical valid mask, the DEM availability mask (0 = no DEM tile, 1 = DEM present but excluded
- unknown / conflicting vertical units, 2 = usable), per-layer availability, the native DEM feature-window index,
the processing configuration and the completion manifest ``COMPLETE.json`` (written only after validation passed).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_versions import grid_signature, spec_diff
from .features import FeatureStore
from .registry import atomic_write_json, atomic_write_npy, atomic_write_parquet, read_json, sha256_file

DATASET_FILES = ("dataset_manifest.json", "feature_schema.parquet", "label_schema.parquet", "valid_mask.npy",
                 "dem_availability.npy", "feature_availability.parquet", "native_window_index.parquet",
                 "fault_geometry_registry.parquet", "processing_config.json", "source_provenance.json")
CUBES = {"geophysical": "geophysical", "dem": "dem_100m_summary"}


class FeatureDatasetError(RuntimeError):
    pass


class FeatureDatasetIncomplete(FeatureDatasetError):
    def __init__(self, msg, missing=None):
        super().__init__(msg)
        self.missing = list(missing or [])


@dataclass
class DatasetManifest:
    version: str
    path: Path
    spec: dict
    dem_statistics_resolution_m: float | None
    n_predictors: int
    n_labels: int
    complete: bool
    created_utc: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    passed: bool
    checks: pd.DataFrame
    level: str

    @property
    def problems(self) -> list[str]:
        c = self.checks
        return [f"{r.check}: {r.detail}" for r in c[~c.passed].itertuples()] if len(c) else []

    def as_dict(self) -> dict:
        return {"passed": self.passed, "level": self.level, "checks": self.checks.to_dict("records")}


def dataset_dir(features_root, version: str) -> Path:
    return Path(features_root) / "datasets" / version


class FeatureDataset(FeatureStore):
    """Read-only :class:`FeatureStore` view of a finalized version (same ``get`` / ``predictors`` / ``sample`` /
    ``localize`` interface), with predictors and labels from the dataset schema."""

    def __init__(self, features_root, version: str, grid=None):
        self.root = Path(features_root)
        self.dir = dataset_dir(features_root, version)
        self.version = version
        self.grid = grid
        self.manifest = read_json(self.dir / "dataset_manifest.json") or {}
        f = pd.read_parquet(self.dir / "feature_schema.parquet")
        lab = pd.read_parquet(self.dir / "label_schema.parquet")
        reg = pd.concat([f.assign(label_derived=False), lab.assign(label_derived=True)], ignore_index=True)
        reg["fingerprint"] = reg["spec_hash"]
        self.reg = reg
        self.reg_path = self.dir / "feature_schema.parquet"
        self.complete = (self.dir / "COMPLETE.json").is_file()

    def put(self, *a, **k):
        raise FeatureDatasetError(f"feature dataset {self.version} is immutable (write new layers to the pool and "
                                  "finalize a new version)")

    def flush(self):
        return self.reg_path

    @property
    def spec(self) -> dict:
        return self.manifest.get("spec", {})

    @property
    def dem_resolution_m(self):
        return self.manifest.get("dem_statistics_resolution_m")

    def cube(self, name: str) -> list[str]:
        return list(self.manifest.get("cubes", {}).get(name, []))

    def valid_mask(self) -> np.ndarray:
        return np.load(self.dir / "valid_mask.npy").astype(bool)

    def dem_availability(self) -> np.ndarray:
        return np.load(self.dir / "dem_availability.npy")


class PoolView(FeatureStore):
    """Read-only view over the pool layers currently verified for the requested spec (before finalisation).

    ``rows`` - expected-layer table restricted to reusable layers, with pool ``relpath``; canonical channel names.
    """

    def __init__(self, features_root, rows: pd.DataFrame, grid=None):
        self.root = Path(features_root)
        self.grid = grid
        reg = rows.copy()
        reg["fingerprint"] = reg.get("spec_hash")
        reg["label_derived"] = reg["dep_class"].eq("label")
        self.reg = reg.reset_index(drop=True)
        self.reg_path = self.root / "layer_manifest.parquet"

    def put(self, *a, **k):
        raise FeatureDatasetError("PoolView is read-only - commit layers through persist.commit_layer")

    def flush(self):
        return self.reg_path


# ------------------------------------------------------------------------------------------ finalize
def finalize_feature_dataset(features_root, version: str, grid, pool, expected: pd.DataFrame, spec: dict, *,
                             dem_availability: np.ndarray | None, native_dir=None, fault_registry=None,
                             fault_segments=None, processing_config: dict | None = None, provenance: dict | None = None,
                             code_commit: str | None = None, log=print) -> DatasetManifest:
    """Create ``datasets/<version>`` from the pool layers listed in ``expected`` (all must be verified for ``spec``).

    Idempotent for the same spec; refuses to overwrite a version finalized with a different spec.
    """
    d = dataset_dir(features_root, version)
    done = read_json(d / "COMPLETE.json")
    if done:
        diff = spec_diff(done.get("spec", {}), spec)
        if diff:
            raise FeatureDatasetError(f"feature dataset {version} already exists with another specification "
                                      f"({diff}); choose a new feature_version - versions are never overwritten")
        log(f"[features] dataset {version} already finalized ({done.get('completed_utc')}) - reused")
        return _manifest_of(d, done)
    missing, rows = [], []
    for r in expected.itertuples():
        spec_h = spec[r.dep_class]
        ok, why = pool.reusable(r.layer_id, spec_h)
        if not ok:
            missing.append(f"{r.layer_id} ({why})")
            continue
        pr = pool.row(r.layer_id)
        rows.append({"name": r.name, "layer_id": r.layer_id, "family": r.family, "dep_class": r.dep_class,
                     "cube": CUBES.get(r.dep_class, "labels"), "relpath": pr["relpath"], "size": int(pr["size"]),
                     "sha256": pr["sha256"], "spec_hash": spec_h, "unit": pr.get("unit") or getattr(r, "unit", ""),
                     "scale_m": pr.get("scale_m") if pr.get("scale_m") is not None else getattr(r, "scale_m", None),
                     "processing_resolution_m": pr.get("processing_resolution_m"), "source": pr.get("source"),
                     "finite_fraction": pr.get("finite_fraction"), "dtype": "float32", "nodata": "NaN"})
    if missing:
        raise FeatureDatasetIncomplete(f"feature dataset {version}: {len(missing)} required layer(s) not verified - "
                                       f"first: {missing[:5]}", missing)
    t = pd.DataFrame(rows)
    feats, labs = t[t.dep_class != "label"].reset_index(drop=True), t[t.dep_class == "label"].reset_index(drop=True)
    d.mkdir(parents=True, exist_ok=True)
    valid = grid.valid.astype(np.uint8)
    da = np.zeros(grid.shape, np.uint8) if dem_availability is None else np.asarray(dem_availability, np.uint8)
    avail = feats[["name", "family", "dep_class", "finite_fraction"]].copy()
    usable = da == 2
    avail["fraction_of_dem_usable_area"] = np.where(avail.dep_class == "dem",
                                                   avail.finite_fraction * valid.sum() / max(int(usable.sum()), 1),
                                                   np.nan)
    nat = _native_index(native_dir, features_root)
    fr = pd.read_parquet(fault_registry) if fault_registry and Path(fault_registry).is_file() else pd.DataFrame()
    outs = {"feature_schema.parquet": atomic_write_parquet(d / "feature_schema.parquet", feats),
            "label_schema.parquet": atomic_write_parquet(d / "label_schema.parquet", labs),
            "valid_mask.npy": atomic_write_npy(d / "valid_mask.npy", valid),
            "dem_availability.npy": atomic_write_npy(d / "dem_availability.npy", da),
            "feature_availability.parquet": atomic_write_parquet(d / "feature_availability.parquet", avail),
            "native_window_index.parquet": atomic_write_parquet(d / "native_window_index.parquet", nat),
            "fault_geometry_registry.parquet": atomic_write_parquet(d / "fault_geometry_registry.parquet", fr),
            "processing_config.json": atomic_write_json(d / "processing_config.json", processing_config or {}),
            "source_provenance.json": atomic_write_json(d / "source_provenance.json", {
                **(provenance or {}), "fault_segments": str(fault_segments) if fault_segments else None,
                "fault_segments_sha256": sha256_file(fault_segments) if fault_segments and Path(fault_segments).is_file()
                else None})}
    dem_res = sorted({float(x) for x in feats.processing_resolution_m.dropna()})
    man = {"version": version, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "code_commit": code_commit, "spec": spec, "grid": grid_signature(grid),
           "dem_statistics_resolution_m": dem_res[0] if dem_res else None,
           "n_predictors": int(len(feats)), "n_labels": int(len(labs)),
           "cubes": {"geophysical": feats[feats.dep_class == "geophysical"].name.tolist(),
                     "dem_100m_summary": feats[feats.dep_class == "dem"].name.tolist()},
           "labels": labs.name.tolist(), "dem_availability_codes": {"0": "no DEM tile", "1": "DEM present but excluded "
                                                                    "(vertical unit unknown / conflicting)", "2": "usable"},
           "dem_area_km2": {"usable": float(usable.sum() * grid.res[0] * grid.res[1] / 1e6),
                            "excluded": float((da == 1).sum() * grid.res[0] * grid.res[1] / 1e6)},
           "files": {k: {"sha256": sha256_file(p), "size": Path(p).stat().st_size} for k, p in outs.items()},
           "immutable": True, "contains_split_assignments": False,
           "note": "layers are referenced in the pool (03_features/<relpath>), not copied"}
    if len(dem_res) > 1:
        raise FeatureDatasetError(f"{version}: DEM layers at several resolutions {dem_res} - never mixed in a version")
    atomic_write_json(d / "dataset_manifest.json", man)
    rep = validate_feature_dataset(features_root, version, grid, level="QUICK", require_complete=False)
    atomic_write_json(d / "validation_report.json", rep.as_dict())
    if not rep.passed:
        raise FeatureDatasetError(f"feature dataset {version} failed validation - COMPLETE.json NOT written: "
                                  f"{rep.problems[:6]}")
    atomic_write_json(d / "COMPLETE.json", {"version": version, "spec": spec, "completed_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_predictors": man["n_predictors"], "n_labels": man["n_labels"],
        "dem_statistics_resolution_m": man["dem_statistics_resolution_m"], "validation": "QUICK passed",
        "code_commit": code_commit})
    log(f"[features] dataset {version} finalized: {man['n_predictors']} predictors, {man['n_labels']} labels, "
        f"DEM {man['dem_statistics_resolution_m']} m")
    return _manifest_of(d, read_json(d / "COMPLETE.json"))


def _manifest_of(d: Path, done: dict) -> DatasetManifest:
    m = read_json(d / "dataset_manifest.json") or {}
    return DatasetManifest(m.get("version", d.name), d, m.get("spec", {}), m.get("dem_statistics_resolution_m"),
                           int(m.get("n_predictors", 0)), int(m.get("n_labels", 0)), bool(done),
                           m.get("created_utc", ""), {"completed_utc": (done or {}).get("completed_utc")})


def _native_index(native_dir, features_root) -> pd.DataFrame:
    rows = []
    if native_dir and Path(native_dir).is_dir():
        import rasterio

        for p in sorted(Path(native_dir).glob("*.tif")):
            if ".partial" in p.name:
                continue
            with rasterio.open(p) as ds:
                rows.append({"relpath": str(p.relative_to(features_root)).replace("\\", "/"), "crs": str(ds.crs),
                             "res_m": float(ds.res[0]), "width": ds.width, "height": ds.height,
                             "left": ds.bounds.left, "bottom": ds.bounds.bottom, "right": ds.bounds.right,
                             "top": ds.bounds.top, "bands": "|".join(d or "" for d in ds.descriptions),
                             "size": p.stat().st_size})
    return pd.DataFrame(rows, columns=["relpath", "crs", "res_m", "width", "height", "left", "bottom", "right", "top",
                                       "bands", "size"])


# ------------------------------------------------------------------------------------------ validate / load
def validate_feature_dataset(features_root, version: str, grid, level: str = "QUICK", sample: int = 8,
                             require_complete: bool = True, seed: int = 0) -> ValidationReport:
    """Integrity of a dataset version. QUICK: every file present with its recorded size, every layer header
    (shape / dtype), sha256 of ``sample`` random layers, schema, grid, masks, DEM availability. FULL: sha256 of all."""
    from .recovery import inspect_npy

    d = dataset_dir(features_root, version)
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)[:2000]})

    man = read_json(d / "dataset_manifest.json")
    add("manifest_present", man is not None, d / "dataset_manifest.json")
    if man is None:
        return ValidationReport(False, pd.DataFrame(checks), level)
    if require_complete:
        add("complete_marker", (d / "COMPLETE.json").is_file(), "COMPLETE.json")
    miss = [f for f in DATASET_FILES if not (d / f).is_file()]
    add("dataset_files", not miss, miss or "all present")
    bad_files = [k for k, v in man.get("files", {}).items()
                 if (d / k).is_file() and (d / k).stat().st_size != v["size"]]
    add("dataset_file_sizes", not bad_files, bad_files or "ok")
    f = pd.read_parquet(d / "feature_schema.parquet") if (d / "feature_schema.parquet").is_file() else pd.DataFrame()
    lab = pd.read_parquet(d / "label_schema.parquet") if (d / "label_schema.parquet").is_file() else pd.DataFrame()
    t = pd.concat([f, lab], ignore_index=True)
    root = Path(features_root)
    absent, size_bad, hdr_bad = [], [], []
    for r in t.itertuples():
        p = root / r.relpath
        if not p.is_file():
            absent.append(r.layer_id)
            continue
        if p.stat().st_size != int(r.size):
            size_bad.append(r.layer_id)
            continue
        info = inspect_npy(p, grid.shape)
        if not info["header_ok"]:
            hdr_bad.append(f"{r.layer_id}: {info.get('problems') or info.get('error')}")
    add("layers_present", not absent, absent[:20] or f"{len(t)} layer file(s)")
    add("layer_sizes", not size_bad, size_bad[:20] or "ok")
    add("layer_headers", not hdr_bad, hdr_bad[:10] or "shape / dtype ok")
    pick = t if level == "FULL" else t.sample(min(sample, len(t)), random_state=seed) if len(t) else t
    sha_bad = [r.layer_id for r in pick.itertuples() if (root / r.relpath).is_file() and
               sha256_file(root / r.relpath) != r.sha256]
    add("layer_sha256", not sha_bad, sha_bad or f"{len(pick)} layer(s) re-hashed")
    add("unique_channel_names", t.name.is_unique if len(t) else True, t.name[t.name.duplicated()].tolist()[:10])
    add("no_label_derived_predictor", not (len(f) and (f.dep_class == "label").any()) and not (
        len(f) and f.name.str.startswith("Y_").any()), "labels are kept in label_schema.parquet")
    res = sorted({float(x) for x in f.processing_resolution_m.dropna()}) if len(f) else []
    add("single_dem_resolution", len(res) <= 1 and (not res or res[0] == man.get("dem_statistics_resolution_m")),
        res)
    add("grid_alignment", man.get("grid") == grid_signature(grid), f"{man.get('grid')} vs {grid_signature(grid)}")
    if (d / "valid_mask.npy").is_file():
        vm = np.load(d / "valid_mask.npy")
        add("valid_mask", vm.shape == tuple(grid.shape) and (grid.valid is None or np.array_equal(vm.astype(bool),
                                                                                                   grid.valid)),
            f"shape {vm.shape}")
    if (d / "dem_availability.npy").is_file():
        da = np.load(d / "dem_availability.npy")
        add("dem_availability_mask", da.shape == tuple(grid.shape) and set(np.unique(da)) <= {0, 1, 2},
            f"codes {sorted(np.unique(da).tolist())}")
        probe = f[f.name == "dem100_valid_fraction"] if len(f) else f
        if len(probe):
            a = np.load(root / probe.relpath.iloc[0], mmap_mode="r")
            outside = int((np.isfinite(a) & (da != 2)).sum())
            add("dem_missing_coverage_explicit", outside == 0,
                f"{outside} DEM-feature pixel(s) outside the usable DEM area" if outside else "NaN outside usable DEM")
    return ValidationReport(all(c["passed"] for c in checks), pd.DataFrame(checks), level)


def load_feature_dataset(features_root, version: str, grid=None, expected_spec: dict | None = None,
                         verify: str | None = "QUICK") -> FeatureDataset:
    """Open a finalized version; raises :class:`FeatureDatasetError` if it is incomplete, of another spec, or fails
    validation (``verify`` = QUICK / FULL / None)."""
    d = dataset_dir(features_root, version)
    done = read_json(d / "COMPLETE.json")
    if done is None:
        raise FeatureDatasetIncomplete(f"feature dataset {version} is not finalized (no {d / 'COMPLETE.json'})")
    if expected_spec is not None:
        diff = spec_diff(done.get("spec", {}), expected_spec)
        if diff:
            raise FeatureDatasetError(f"feature dataset {version} was built with another specification: {diff}")
    if verify and grid is not None:
        rep = validate_feature_dataset(features_root, version, grid, level=verify)
        if not rep.passed:
            raise FeatureDatasetError(f"feature dataset {version} failed {verify} validation: {rep.problems[:6]}")
    return FeatureDataset(features_root, version, grid)
