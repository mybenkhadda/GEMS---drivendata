"""G07 schemas. Grid / feature / leakage validators are reused from G06 (``g06.contracts``); this module adds the
DEM, candidate, model and prediction contracts and the native-vs-analysis-grid checks.

Two grids never mix silently:
* NATIVE / WORK grid of a DEM source (1-2 m, its own CRS, usually NAD83 / UTM 11N = EPSG:26911): features are
  computed here;
* ANALYSIS grid = the GeoDAWN 100 m grid (EPSG:32611, 3730 x 3292): every model input / output lives here, reached
  only through ``dem_io.align_dem_to_analysis_grid`` (explicit area-weighted aggregation, coverage recorded).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ..g06 import contracts as _C6

# shared G06 contracts, re-exported under the G07 namespace
ContractError, GridSpec, LeakageError, ProvenanceError, SchemaError = (_C6.ContractError, _C6.GridSpec,
                                                                       _C6.LeakageError, _C6.ProvenanceError,
                                                                       _C6.SchemaError)
assert_label_free_columns, validate_feature_order = _C6.assert_label_free_columns, _C6.validate_feature_order
validate_grid, validate_probability = _C6.validate_grid, _C6.validate_probability

METRE_UNITS = ("m", "metre", "meter", "metres", "meters")
FOOT_UNITS = ("ft", "foot", "feet", "us survey foot", "us_survey_foot", "ftus", "foot_us", "us_ft")
LOCAL, CATALOG_ONLY, LOCAL_INVALID, LOCAL_OUTSIDE = "LOCAL", "CATALOG_ONLY", "LOCAL_INVALID", "LOCAL_OUTSIDE_STUDY_AREA"
UNKNOWN_VERTICAL_UNIT = "UNKNOWN_VERTICAL_UNIT"
STRICTNESS = ("STRICT_NESTED", "STRICT_ISOLATED", "STRICT_LABEL_FREE", "SCREENING_ONLY", "BLOCKED_UPSTREAM",
              "REFERENCE")


@dataclass
class DEMSource:
    """One DEM file (local or catalog-only)."""
    tile_id: str
    url: str | None
    path: str | None
    project: str
    year: int | None
    crs: str | None
    res_m: float | None
    bounds: tuple | None                   # (west, south, east, north) in ``crs``
    vertical_unit: str = "unknown"
    nodata: float | None = None
    dtype: str | None = None
    status: str = "CATALOG_ONLY"           # CATALOG_ONLY | LOCAL | LOCAL_INVALID | LOCAL_OUTSIDE_STUDY_AREA
    license: str = "USGS 3DEP - public domain (verify competition eligibility)"
    problems: tuple = ()                   # blocking problems: why the source is excluded
    exclusion_codes: tuple = ()            # machine-readable form of ``problems`` (EXCLUSION_* codes)
    warnings: tuple = ()                   # non-blocking remarks (recorded, never a reason to exclude)
    vertical_unit_source: str = ""         # band_units | tag | crs_vertical | sidecar_xml | override | ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["bounds"] = json.dumps(list(self.bounds)) if self.bounds else ""
        d["problems"] = ";".join(self.problems)
        d["exclusion_codes"] = ";".join(self.exclusion_codes)
        d["warnings"] = ";".join(self.warnings)
        return d


@dataclass(frozen=True)
class DEMTile:
    """A processing window of a DEM source: core window + halo on the work grid."""
    source_id: str
    row0: int
    row1: int
    col0: int
    col1: int
    halo: int
    work_res_m: float
    factor: int                            # native pixels per work pixel (block average)


@dataclass
class DEMFeatureSpec:
    feature_id: str
    description: str
    unit: str
    scale_m: float | None
    operator: str
    aggregation: str                       # mean | max | share | orientation
    native_res_m: float
    family: str = "dem"
    label_free: bool = True
    missing: str = "nan_outside_dem_coverage"

    def to_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingVisibility:
    fit_id: str
    scenario: str
    visible_hash: str
    region_hash: str
    predicts: tuple
    label_scope: str = "visible faults of the fit only"


@dataclass
class CandidateGeometry:
    label: int
    n_px: int
    length_km: float
    row: float
    col: float
    theta: float                           # axis angle (rad, from +column toward +row)
    linearity: float
    straightness: float


@dataclass
class FaultCandidate:
    candidate_id: str
    unit: str
    geometry: CandidateGeometry
    sources: tuple                         # pools supporting the candidate (provenance)
    scores: dict = field(default_factory=dict)
    relevance: float | None = None         # evaluation side (training units only)


@dataclass(frozen=True)
class ModelArtifact:
    experiment_id: str
    fit_id: str
    kind: str
    path: str
    feature_columns: tuple
    schema_hash: str
    fingerprint: str
    strictness: str
    upstream: tuple = ()                   # (input name, provenance tag)


@dataclass(frozen=True)
class PredictionArtifact:
    experiment_id: str
    unit: str
    path: str
    fit_id: str
    grid: str                              # "analysis_100m" | "native"
    fingerprint: str
    strictness: str


@dataclass
class EvaluationResult:
    model: str
    scenario: str
    policy: str
    frac_requested: float
    frac_actual: float
    dti: float
    tp_w: float
    fp_w: float
    fn_w: float
    worst_zone_dti: float
    n_units: int
    strictness: str


# =============================================================================== validators
def vertical_scale(unit: str) -> float:
    """Multiplier to metres for a vertical unit; raises on unknown units (never guessed silently)."""
    u = str(unit or "").strip().lower()
    if u in METRE_UNITS:
        return 1.0
    if u in FOOT_UNITS:
        return 0.3048006096 if "survey" in u or u == "ftus" else 0.3048
    raise ContractError(f"unknown DEM vertical unit {unit!r}: set dem.vertical_unit_override after checking the "
                        "source metadata")


def dem_source_issues(src: DEMSource, target_res_m: float, max_res_m: float, header: dict | None = None
                      ) -> tuple[list[tuple[str, str]], list[str]]:
    """(blocking [(code, message)], non-blocking warnings) of one DEM source against the G07 contract.

    ``header`` = ``dem_io.read_header`` output: it adds the checks that need the open dataset (projected CRS in
    metres, north-up geotransform, square pixels). The CRS may be any projected metric CRS (compound 3DEP CRSs have no
    single EPSG code): the aggregation reprojects to the analysis grid exactly. The vertical unit is never inferred
    from the horizontal CRS unit."""
    h = header or {}
    blocking, warn = [], []
    if h.get("error"):
        blocking.append(("UNREADABLE", f"cannot read the GeoTIFF: {h['error']}"))
    if src.res_m is None:
        blocking.append(("RESOLUTION_UNKNOWN", "resolution unknown (header not read)"))
    elif src.res_m > max_res_m + 1e-9:
        blocking.append(("RESOLUTION_TOO_COARSE", f"resolution {src.res_m} m coarser than the accepted {max_res_m} m"))
    elif h and abs(float(h.get("res_y_m", src.res_m)) - float(src.res_m)) > 1e-6:
        blocking.append(("NON_SQUARE_PIXELS", f"pixel {src.res_m} x {h.get('res_y_m')} m is not square"))
    if src.crs is None:
        blocking.append(("CRS_MISSING", "the GeoTIFF has no CRS"))
    elif h and not h.get("crs_projected_metric", False):
        blocking.append(("CRS_NOT_PROJECTED_METRIC", f"CRS {str(src.crs)[:80]} is not a projected CRS in metres"))
    if h and not h.get("transform_valid", False):
        blocking.append(("INVALID_TRANSFORM", "geotransform is not north-up with finite, non-zero pixel sizes"))
    try:
        vertical_scale(src.vertical_unit)
    except ContractError as e:
        blocking.append((UNKNOWN_VERTICAL_UNIT, str(e)))
    if src.nodata is None:
        warn.append("nodata undeclared (non-finite values only are treated as missing)")
    if src.res_m is not None and target_res_m and src.res_m > target_res_m * 1.5:
        warn.append(f"resolution {src.res_m} m differs from the {target_res_m} m target (recorded, not substituted)")
    return blocking, warn


def validate_dem_source(src: DEMSource, target_res_m: float, max_res_m: float, header: dict | None = None
                        ) -> list[str]:
    """Blocking problems of one DEM source (empty = usable); see ``dem_source_issues``."""
    return [m for _, m in dem_source_issues(src, target_res_m, max_res_m, header)[0]]


def assert_analysis_grid(a: np.ndarray, grid: GridSpec, what: str) -> None:
    if tuple(np.shape(a)[-2:]) != grid.shape:
        raise ContractError(f"{what}: {np.shape(a)} is not on the 100 m analysis grid {grid.shape}")


def assert_not_100m_substitute(src: DEMSource, analysis_res_m: float = 100.0) -> None:
    """A 100 m elevation channel must never be labelled a high-resolution DEM."""
    if src.res_m is not None and src.res_m >= analysis_res_m / 2:
        raise ContractError(f"{src.tile_id}: {src.res_m} m is not a high-resolution DEM")
