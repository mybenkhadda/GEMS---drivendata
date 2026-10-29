"""Central artifact registry of the exhaustive EDA.

Single source of the stage output CONTRACT: ``ExhaustiveEDA.OUTPUTS`` (path -> required columns / NPZ keys). This
module adds, without duplicating it:
* RAW            mandatory raw inputs (paths relative to the Drive root) and their metadata contract,
* CONSUMERS      which later stages read each artifact (from the actual readers in parts_* / run / synthesis),
* REQUIREMENT    required / conditional (frozen engine) / optional (DEM),
* RULES          domain rules on top of columns (row masks, as in products.py),
* GRID_ARRAYS    NPZ arrays that must have the raster grid shape,
* RELOAD         how a later session reloads the artifact.
``validate(ws, rel, grid_shape)`` is used by the stage cache check (``ExhaustiveEDA.check_outputs``) and the
preflight, so reuse and preflight apply exactly the same rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ARTIFACT_SCHEMA = "gde-artifacts.1"

# --------------------------------------------------------------------------- raw inputs (Drive-root relative)
RAW = {
    "features": {"path": "data/raw/geodawn/gems-geodawn-numerical-features.tif", "count": 19, "dtype": "float32",
                 "required": True},
    "faults": {"path": "data/raw/faults/existing_faults.tif", "count": 1, "dtype": "int8", "required": True},
    "submission": {"path": "data/raw/example_submission.tif", "count": 1, "dtype": "float32", "required": True},
    "dem_urls": {"path": "data/raw/dem_1m/tnm_urls.json", "required": False},
}
GRID = {"epsg": 32611, "res_m": 100.0, "height": 3730, "width": 3292, "fault_px": 60988}

# --------------------------------------------------------------------------- consumers (verified readers)
STATIONS = ("e05_signatures", "e06_scales", "e07_crossmodal", "e08_label_bias", "e09_regional", "e11_forensics")
CONSUMERS = {
    "audit/band_metadata.csv": ("e02_landscape",),
    "rasters/spatial_domains/domains.npz": ("e03_geometry", "e04_profiles", "e07_crossmodal", "e08_label_bias",
                                            "e09_regional", "e11_forensics", "e12_candidates"),
    "datasets/physical_regions.parquet": ("e02_landscape", "e03_geometry"),
    "datasets/matched_controls.parquet": STATIONS,
    "datasets/fault_profiles.parquet": STATIONS,
    "datasets/station_structure.parquet": STATIONS,
    "datasets/fault_profile_cube.npz": STATIONS,
    "tables/matched_control_tests.csv": ("e09_regional", "e15_synthesis"),
    "tables/scale_localization.csv": ("e06_scales", "e15_synthesis"),
    "tables/scale_regional.csv": ("e09_regional",),
    "datasets/station_scale_values.parquet": ("e06_scales",),
    "rasters/evidence_maps/crests_1km.npz": ("e07_crossmodal", "e08_label_bias", "e11_forensics", "e12_candidates"),
    "datasets/structural_lineaments.parquet": ("e08_label_bias", "e11_forensics", "e12_candidates"),
    "rasters/evidence_maps/lineaments.npz": ("e08_label_bias", "e11_forensics", "e12_candidates"),
    "validation/scenarios.csv": ("e11_forensics",),
    "atlas/cache/honest_maps.npz": ("e11_forensics", "e12_candidates"),
    "atlas/atlas_index.csv": ("review_widget", "e15_synthesis"),
    "atlas/reviewed_annotations.csv": ("review_widget", "e15_synthesis"),
}
CONDITIONAL_STAGES = {"01_attach": "frozen engine (factory)", "03_scenarios": "frozen engine",
                      "e11_forensics": "frozen engine", "e13_dem": "dem.enabled + cached DEM"}
OPTIONAL_STAGES = {"e13_dem": "dem.enabled"}

# --------------------------------------------------------------------------- domain rules (row masks)


def _geometry_rules():
    from .products import BAND_RULES, GEOMETRY_RULES

    return BAND_RULES, GEOMETRY_RULES


RULES = {
    "datasets/matched_controls.parquet": {
        "unique_sample_id": lambda d: ~d.sample_id.duplicated(keep=False),
        "has_fault_and_control": lambda d: {"fault", "control"} <= set(d.kind),
        "non_negative_rc": lambda d: (d.r >= 0) & (d.c >= 0)},
    "datasets/fault_profiles.parquet": {
        "abs_step_non_negative": lambda d: (d.abs_step >= 0) | d.abs_step.isna()},
    "datasets/station_scale_values.parquet": {"positive_scale": lambda d: d.scale_km > 0},
    "datasets/structural_lineaments.parquet": {
        "unique_lineament_id": lambda d: ~d.lineament_id.duplicated(keep=False),
        "axial_azimuth": lambda d: d.azimuth.between(0, 180, inclusive="left") | d.azimuth.isna(),
        "known_status": lambda d: d.status.isin(["mapped", "near_mapped", "unmapped"])},
    "tables/scale_localization.csv": {"positive_scale": lambda d: d.scale_km > 0},
    "atlas/atlas_index.csv": {"unique_candidate_id": lambda d: ~d.candidate_id.duplicated(keep=False)},
}
GRID_ARRAYS = {"rasters/spatial_domains/domains.npz": ("domain",),
               "rasters/evidence_maps/lineaments.npz": ("labels", "family_count"),
               "atlas/cache/honest_maps.npz": ("w2", "b0", "b1", "zone_id")}
# consumers that read an artifact only when its producer is valid in the same run (not a DEPENDS edge)
CONDITIONAL_CONSUMERS = {("atlas/cache/honest_maps.npz", "e12_candidates")}
NON_STAGE_CONSUMERS = ("review_widget", "e15_synthesis")
RELOAD = {".csv": "E.ws.load_table(rel) / pandas.read_csv", ".parquet": "E.table(rel) / pandas.read_parquet",
          ".npz": "numpy.load(E.ws.path(rel))", ".json": "E.ws.read_json(rel)", ".md": "text", ".txt": "text"}


def rules_for(rel: str) -> dict:
    if rel == "tables/band_characterization.csv":
        return _geometry_rules()[0]
    if rel == "tables/fault_geometry.csv":
        return _geometry_rules()[1]
    return RULES.get(rel, {})


@dataclass
class Artifact:
    rel: str
    producer: str
    required: tuple
    requirement: str                       # required | conditional | optional
    consumers: tuple = ()
    rules: tuple = ()
    grid_arrays: tuple = ()
    kind: str = field(default="")

    def as_row(self) -> dict:
        return {"artifact": self.rel, "producer_stage": self.producer, "requirement": self.requirement,
                "kind": self.kind, "required_fields": ";".join(self.required or ()),
                "consumers": ";".join(self.consumers), "rules": ";".join(self.rules),
                "grid_shape_arrays": ";".join(self.grid_arrays), "reload": RELOAD.get(self.kind, ""),
                "schema": ARTIFACT_SCHEMA}


def registry(cls=None) -> list[Artifact]:
    """Every contracted artifact of ``ExhaustiveEDA.OUTPUTS`` with its producer, consumers and rules."""
    if cls is None:
        from .run import ExhaustiveEDA as cls
    out = []
    for stage, specs in cls.OUTPUTS.items():
        req = "optional" if stage in OPTIONAL_STAGES else "conditional" if stage in CONDITIONAL_STAGES else "required"
        for rel, cols in specs:
            out.append(Artifact(rel, stage, tuple(cols or ()), req, CONSUMERS.get(rel, ()), tuple(rules_for(rel)),
                                GRID_ARRAYS.get(rel, ()), "." + rel.rsplit(".", 1)[-1]))
    return out


def contract_table(cls=None) -> pd.DataFrame:
    return pd.DataFrame([a.as_row() for a in registry(cls)])


def undeclared_consumption(cls=None) -> list[str]:
    """Consumed artifacts that no stage contracts (must be empty)."""
    declared = {a.rel for a in registry(cls)}
    return sorted(set(CONSUMERS) - declared)


def validate(ws, rel: str, grid_shape=None, full: bool = True) -> list[str]:
    """Problems of one artifact beyond the base column check: domain rules (full read) and NPZ grid shapes.
    Empty list = valid. Never raises."""
    from .products import rule_violations

    p = ws.path(rel)
    if not p.exists():
        return [f"{rel}: missing"]
    probs = []
    try:
        if rel.endswith(".npz"):
            if grid_shape is not None and rel in GRID_ARRAYS:
                with np.load(p, allow_pickle=False) as z:
                    for k in GRID_ARRAYS[rel]:
                        if k in z.files and tuple(z[k].shape) != tuple(grid_shape):
                            probs.append(f"{rel}: array {k} shape {z[k].shape} != grid {tuple(grid_shape)}")
            if rel == "datasets/fault_profile_cube.npz":
                with np.load(p, allow_pickle=False) as z:
                    if "cube" in z.files and (z["cube"].ndim != 3 or z["cube"].shape[1] != len(z["signals"])
                                              or z["cube"].shape[2] != len(z["offsets"])):
                        probs.append(f"{rel}: cube {z['cube'].shape} inconsistent with signals / offsets")
            if rel == "rasters/evidence_maps/crests_1km.npz" and grid_shape is not None:
                with np.load(p, allow_pickle=False) as z:
                    if "shape" in z.files and tuple(int(x) for x in z["shape"]) != tuple(grid_shape):
                        probs.append(f"{rel}: shape {tuple(z['shape'])} != grid {tuple(grid_shape)}")
                    if len([k for k in z.files if k != "shape"]) == 0:
                        probs.append(f"{rel}: no family crest array")
            return probs
        r = rules_for(rel)
        if not r or not full:
            return probs
        T = pd.read_parquet(p) if rel.endswith(".parquet") else pd.read_csv(p)
        if not len(T):
            return probs                           # typed-empty: column check is the base check's job
        for k, (n, s) in rule_violations(T, r).items():
            probs.append(f"{rel}: rule {k} violated by {n} rows, e.g. {str(s[:2])[:300]}")
    except Exception as e:                         # unreadable / unexpected layout: a problem, not a crash
        probs.append(f"{rel}: validation error {type(e).__name__}: {e}")
    return probs
