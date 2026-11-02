"""Strict schemas and reusable data contracts.

Every raster, feature, split, dataset and model that crosses a module boundary is described by one of these
dataclasses, and every crossing is checked by a validator that raises instead of coercing: a feature stack with a
different channel order, a raster on another grid, a dataset whose feature columns carry truth, or a prediction made
with a schema the model was not fitted on all fail loudly.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np


class ContractError(ValueError):
    """A raster / array / table violates its declared contract."""


class SchemaError(ContractError):
    """Feature order, names or schema hash differ from what a consumer was built for."""


class LeakageError(RuntimeError):
    """A truth-derived quantity would reach a model input, or a held-out label would reach training."""


class ProvenanceError(RuntimeError):
    """An upstream prediction / feature does not carry the provenance required for this validation unit."""


# tokens that may never appear in a model-input column name (targets, truth, split identity)
FORBIDDEN_FEATURE_TOKENS = ("y_", "truth", "target", "label_conf", "withheld", "system_id", "recovered",
                            "near_share", "is_pos", "class", "cls", "incl_prob", "weight", "inner_fold", "fit_id")
_FORBIDDEN_RE = re.compile(r"(^y_|truth|target|label_conf|withheld|system_id|recovered|near_share|is_pos|"
                           r"(^|_)class($|_)|(^|_)cls($|_)|incl_prob|(^|_)weight($|_)|inner_fold|fit_id)")


# =============================================================================== dataclasses
@dataclass(frozen=True)
class GridSpec:
    height: int
    width: int
    crs: str                    # "EPSG:32611"
    transform: tuple            # affine (a, b, c, d, e, f)
    pixel_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @classmethod
    def from_dataset(cls, ds) -> "GridSpec":
        t = ds.transform
        crs = ds.crs.to_string() if ds.crs is not None else ""
        return cls(int(ds.height), int(ds.width), crs, tuple(float(x) for x in (t.a, t.b, t.c, t.d, t.e, t.f)),
                   float(abs(t.a)))

    def affine(self):
        from rasterio.transform import Affine

        return Affine(*self.transform)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BandSpec:
    index: int                  # 1-based band index in the source file
    identifier: str             # exact identifier (description before " - ")
    description: str
    modality: str
    unit: str = ""
    transform: str = "identity"
    scale: float = 1.0


@dataclass
class FeatureSpec:
    feature_id: str
    group: str                  # raster group (file) the channel lives in
    family: str                 # model family: raw | physics | texture | structure | context | qc
    description: str
    unit: str
    source: str                 # band identifier(s) or upstream artifact
    transform: str              # mathematical transformation
    scale_m: float | None = None
    expected_range: tuple | None = None
    missing: str = "nan_outside_support"
    label_free: bool = True
    permitted_tracks: tuple = ("A", "B")
    fit_scope: str = "none"     # none | full_footprint_unsupervised (label-free statistic over the footprint)
    version: str = "g06.fs.1"
    usable: bool = True         # False for constant / all-missing channels (kept, never fed to a model)
    params: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        d["expected_range"] = json.dumps(list(self.expected_range)) if self.expected_range else ""
        d["permitted_tracks"] = ",".join(self.permitted_tracks)
        d["params"] = json.dumps(self.params, sort_keys=True, default=str)
        return d

    @classmethod
    def from_row(cls, r: dict) -> "FeatureSpec":
        er = r.get("expected_range")
        sc = r.get("scale_m")
        return cls(str(r["feature_id"]), str(r["group"]), str(r["family"]), str(r.get("description", "")),
                   str(r.get("unit", "")), str(r.get("source", "")), str(r.get("transform", "")),
                   None if sc is None or (isinstance(sc, float) and np.isnan(sc)) else float(sc),
                   tuple(json.loads(er)) if isinstance(er, str) and er else None, str(r.get("missing", "")),
                   bool(r.get("label_free", True)), tuple(str(r.get("permitted_tracks", "A,B")).split(",")),
                   str(r.get("fit_scope", "none")), str(r.get("version", "")), bool(r.get("usable", True)),
                   json.loads(r["params"]) if isinstance(r.get("params"), str) and r.get("params") else {})


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    directory: str              # relative to features/
    filename: str
    feature_ids: tuple
    mode: str = "tiled"         # tiled | global

    @property
    def rel_path(self) -> str:
        return f"features/{self.directory}/{self.filename}"


@dataclass(frozen=True)
class RasterArtifact:
    path: str
    grid: GridSpec
    band_names: tuple
    dtype: str = "float32"
    nodata: float | None = None
    role: str = "feature"       # feature | prediction | research | submission
    sha256: str | None = None


@dataclass(frozen=True)
class SplitSpec:
    """One training visibility: which labels are visible and where training rows may come from."""
    fit_id: str                 # "V1:0", "V3:2", "V1:0|1" (outer 0, inner 1), "FULL"
    scenario: str
    outer_zone: int | None
    inner_zone: int | None
    visible_rule: str
    region_rule: str
    visible_hash: str
    region_hash: str
    predicts: tuple = ()        # unit keys this fit is allowed to predict


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str                   # pixel | patch | segment | connection
    schema_version: str
    feature_columns: tuple
    target_columns: tuple
    meta_columns: tuple
    n_rows: int
    fingerprint: str
    feature_schema_hash: str
    path: str


@dataclass(frozen=True)
class ModelSpec:
    experiment_id: str
    fit_id: str
    track: str                  # A (geology only) | B (W2-aware)
    model_kind: str
    feature_columns: tuple
    feature_schema_hash: str
    params: dict
    strictness: str             # STRICT_NESTED | STRICT_LABEL_FREE | SCREENING_ONLY
    fingerprint: str
    n_train: int = 0
    n_positive: int = 0


@dataclass
class ExperimentResult:
    experiment_id: str
    scenario: str
    rep: str
    frac: float
    dti: float
    tp_w: float
    fp_w: float
    fn_w: float
    worst_zone_dti: float
    worst_zone: int
    n_units: int
    strictness: str
    status: str = "EVALUATED"


# =============================================================================== validators
def validate_shape(a: np.ndarray, shape: tuple, what: str = "array") -> None:
    if tuple(np.shape(a)[-2:]) != tuple(shape):
        raise ContractError(f"{what}: shape {np.shape(a)} does not end with the grid shape {tuple(shape)}")


def validate_crs(crs: str, expected_epsg: int, what: str = "raster") -> None:
    c = str(crs).upper()
    if not (c == f"EPSG:{int(expected_epsg)}" or c.endswith(f":{int(expected_epsg)}")):
        raise ContractError(f"{what}: CRS {crs!r} is not EPSG:{expected_epsg}")


def validate_transform(transform, expected, atol: float = 1e-6, what: str = "raster") -> None:
    a, b = np.asarray(tuple(transform)[:6], float), np.asarray(tuple(expected)[:6], float)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0):
        raise ContractError(f"{what}: transform {tuple(a)} differs from the grid transform {tuple(b)}")


def validate_grid(grid: GridSpec, expected: GridSpec, epsg: int | None = None, what: str = "raster") -> None:
    if grid.shape != expected.shape:
        raise ContractError(f"{what}: grid {grid.shape} != expected {expected.shape}")
    validate_transform(grid.transform, expected.transform, what=what)
    if epsg is not None:
        validate_crs(grid.crs, epsg, what)


def validate_feature_order(expected, actual, what: str = "feature stack") -> None:
    e, a = list(expected), list(actual)
    if e != a:
        miss = [x for x in e if x not in a]
        extra = [x for x in a if x not in e]
        first = next((i for i, (x, y) in enumerate(zip(e, a)) if x != y), min(len(e), len(a)))
        raise SchemaError(f"{what}: channel order differs from the fitted schema (first mismatch at position "
                          f"{first}; missing {miss[:5]}, unexpected {extra[:5]}, n {len(a)} vs {len(e)})")


def validate_band_identity(resolved: dict, expected: dict) -> None:
    """``resolved`` {identifier: index}; ``expected`` {identifier: index} recorded when the store was built."""
    bad = {k: (resolved.get(k), v) for k, v in expected.items() if resolved.get(k) != v}
    if bad:
        raise ContractError(f"band identity changed since the feature store was built: {bad}")


def validate_dtype(a: np.ndarray, dtype, what: str = "array") -> None:
    if np.asarray(a).dtype != np.dtype(dtype):
        raise ContractError(f"{what}: dtype {np.asarray(a).dtype} != {np.dtype(dtype)}")


def finite_coverage(a: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Share of finite values (inside ``valid`` when given)."""
    a = np.asarray(a)
    m = np.isfinite(a)
    if valid is not None:
        v = np.asarray(valid, bool)
        return float(m[v].mean()) if v.any() else float("nan")
    return float(m.mean()) if m.size else float("nan")


def validate_finite_coverage(a: np.ndarray, valid: np.ndarray, min_share: float, what: str) -> float:
    c = finite_coverage(a, valid)
    if not np.isfinite(c) or c < min_share:
        raise ContractError(f"{what}: finite coverage {c:.4f} < {min_share}")
    return c


def assert_label_free_columns(columns, what: str = "feature matrix") -> None:
    """No truth / target / split-identity column may be a model input."""
    bad = [c for c in columns if _FORBIDDEN_RE.search(str(c).lower())]
    if bad:
        raise LeakageError(f"{what}: truth- or split-derived columns among the model inputs: {bad[:8]}")


def validate_target_separation(feature_columns, target_columns) -> None:
    """Feature and target column sets are disjoint and the features are label-free by name."""
    both = sorted(set(feature_columns) & set(target_columns))
    if both:
        raise LeakageError(f"columns used both as feature and target: {both}")
    assert_label_free_columns(feature_columns)


def validate_fold_provenance(prov: dict, required: dict, what: str = "unit") -> None:
    """Each required key must carry exactly the required provenance tag (e.g. W2 map of this unit only)."""
    bad = {k: (prov.get(k), v) for k, v in required.items() if prov.get(k) != v}
    if bad:
        raise ProvenanceError(f"{what}: provenance mismatch {bad}")


def validate_model_input(model_columns, X_columns, model_schema_hash: str | None = None,
                         store_schema_hash: str | None = None, what: str = "model input") -> None:
    validate_feature_order(model_columns, X_columns, what)
    if model_schema_hash and store_schema_hash and model_schema_hash != store_schema_hash:
        raise SchemaError(f"{what}: model was fitted on feature schema {model_schema_hash}, the store is "
                          f"{store_schema_hash} (rebuild the dataset / refit the model)")


def validate_probability(p: np.ndarray, what: str = "prediction") -> None:
    v = np.asarray(p, np.float64)
    v = v[np.isfinite(v)]
    if v.size and (v.min() < -1e-6 or v.max() > 1 + 1e-6):
        raise ContractError(f"{what}: values outside [0, 1] (min {v.min():.4g}, max {v.max():.4g})")
