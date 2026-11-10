"""Official submission adapter.

Schema source = the official template ``example_submission.tif`` (DrivenData problem description: single-band
float32 GeoTIFF, EPSG:32611, 100 m, template bounds, values in [0, 1], nodata outside the footprint). No vector
columns or geometry requirements are invented: a candidate set is rasterised to this schema.

Evaluator: only an INSTALLED official evaluator is used (``official_submission.evaluator_command``). None is
shipped with the competition data in this repository, so by default :func:`run_official_evaluator` raises
``OfficialEvaluatorUnavailable``. ``gems_bench.metric`` (repository re-implementation of DTI) is exposed separately
as :func:`proxy_dti` and always labelled "offline proxy".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class OfficialSchemaError(RuntimeError):
    pass


class OfficialEvaluatorUnavailable(RuntimeError):
    pass


@dataclass
class OfficialSchema:
    path: str
    crs: str
    transform: tuple
    shape: tuple
    dtype: str
    count: int
    footprint: np.ndarray

    def as_dict(self) -> dict:
        return {"path": self.path, "crs": self.crs, "transform": list(self.transform), "shape": list(self.shape),
                "dtype": self.dtype, "count": self.count, "footprint_px": int(self.footprint.sum()),
                "values": "[0, 1] float32, nodata / masked outside the footprint"}


def load_official_schema(template_path) -> OfficialSchema:
    import rasterio

    p = Path(template_path)
    if not p.is_file():
        raise OfficialSchemaError(f"official template not found: {p} - submission schema cannot be verified")
    with rasterio.open(p) as ds:
        if ds.crs is None or ds.count != 1:
            raise OfficialSchemaError(f"{p}: template must be single-band with a CRS")
        fp = ds.read(1, masked=True)
        foot = ~np.ma.getmaskarray(fp) & np.isfinite(fp.filled(np.nan).astype(np.float32))
        return OfficialSchema(str(p), ds.crs.to_wkt(), tuple(ds.transform)[:6], tuple(ds.shape), ds.dtypes[0], ds.count, foot)


def validate_submission(path, schema: OfficialSchema) -> dict:
    """Reopen a written file and check it against the template. Returns {check: bool, ..., 'passed': bool}."""
    import rasterio
    from rasterio.crs import CRS

    with rasterio.open(path) as ds:
        v = ds.read(1)
        mask = ds.dataset_mask() > 0
        fin = np.isfinite(v)
        checks = {"single_band": ds.count == 1, "float32": ds.dtypes[0] == "float32",
                  "crs": CRS.from_user_input(ds.crs) == CRS.from_user_input(schema.crs),
                  "transform": np.allclose(tuple(ds.transform)[:6], schema.transform),
                  "shape": tuple(ds.shape) == tuple(schema.shape),
                  "finite_in_footprint": bool(fin[schema.footprint].all()),
                  "range_0_1_in_footprint": bool(((v >= 0) & (v <= 1))[schema.footprint & fin].all()),
                  "no_values_outside_footprint": bool(not (mask & ~schema.footprint).any())}
        n_distinct = int(np.unique(v[schema.footprint & fin][:: 97]).size)
    checks["passed"] = all(checks.values())
    checks["distinct_values_sampled"] = n_distinct          # informational (not an official requirement)
    return checks


def export_submission(prob: np.ndarray, schema: OfficialSchema, out_path, manifest: dict) -> dict:
    """Write with the repository's fixed V1 writer (finite [0,1] everywhere + internal validity mask), validate,
    and save ``<name>.json`` (experiment manifest) and ``<name>.sha256``. Refuses a failed validation."""
    import rasterio

    from gems_bench.georegime.run import _write_submission_masked

    from ..registry import atomic_write_json, atomic_write_text, sha256_file

    if prob.shape != tuple(schema.shape):
        raise OfficialSchemaError(f"prediction shape {prob.shape} != template {schema.shape}")
    with rasterio.open(schema.path) as t:
        profile = t.profile.copy()
    out = Path(out_path)
    _write_submission_masked(out, prob, profile, schema.footprint)
    checks = validate_submission(out, schema)
    if not checks["passed"]:
        bad = out.with_name(out.stem + ".FAILED_VALIDATION.tif")
        out.replace(bad)
        raise OfficialSchemaError(f"submission failed validation {checks}; moved to {bad}")
    sha = sha256_file(out)
    atomic_write_text(out.with_suffix(".sha256"), f"{sha}  {out.name}\n")
    atomic_write_json(out.with_suffix(".json"), {**manifest, "sha256": sha, "validation": checks,
                                                 "schema": schema.as_dict(), "status": "READY_FOR_MANUAL_UPLOAD",
                                                 "uploaded": False})
    return {"path": str(out), "sha256": sha, "validation": checks}


def run_official_evaluator(submission_path, cfg: dict, truth_path=None) -> dict:
    cmd = cfg["official_submission"].get("evaluator_command")
    if not cmd:
        raise OfficialEvaluatorUnavailable("no installed official evaluator configured "
                                           "(official_submission.evaluator_command is null); the official DTI is "
                                           "only available from the DrivenData leaderboard")
    exe = cmd.split()[0]
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise OfficialEvaluatorUnavailable(f"configured evaluator '{exe}' is not installed")
    args = cmd.split() + [str(submission_path)] + ([str(truth_path)] if truth_path else [])
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"official evaluator failed: {r.stderr[-2000:]}")
    try:
        return {"official": True, **json.loads(r.stdout)}
    except json.JSONDecodeError:
        return {"official": True, "stdout": r.stdout[-4000:]}


def proxy_dti(prob: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, exclude: np.ndarray | None = None) -> dict:
    """Repository DTI re-implementation (``gems_bench.metric``), pixel-exact exclusion convention. OFFLINE PROXY."""
    from gems_bench import metric as M

    c = M.dti_components(prob, truth, eval_mask=eval_mask, exclude_mask=exclude,
                         exclude_supports_tp=M.OFFICIAL_EXCLUDE_SUPPORTS_TP)
    return {**c.as_dict(), "label": "offline DTI proxy (not the official leaderboard DTI)"}
