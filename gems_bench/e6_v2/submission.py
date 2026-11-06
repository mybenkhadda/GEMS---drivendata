"""Submission export and validation against the official template.

Finalise rule (both maps): NaN where the template is NaN; 0 inside the template but outside the data footprint;
float32 values in [0, 1] elsewhere; written with the template's CRS / transform / shape.

continuous          the fold-mean E6 probability.
area_budget_binary  1 on the top ``n = round(f x template-valid footprint px)`` probabilities, 0 elsewhere; ties
                    broken by raster order (stable sort); f from ``export.area_fraction_rule``:
                    training_label_density -> mapped-fault share of the footprint x density_factor (clipped),
                    fixed -> area_fraction_fixed. Known (mapped) fault pixels are excluded from the ranking unless
                    ``export.known_faults_in_budget`` (they are masked in the hidden test).
``export.final`` selects which of the two becomes ``submission_e6_final.tif``. The choice is made in the config
before the run; neither the policy nor f is tuned on validation DTI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .cache import atomic_write_geotiff, atomic_write_json, atomic_write_text, sha256_file


def area_fraction(cfg: dict, label_density: float) -> float:
    e = cfg["export"]
    if e["area_fraction_rule"] == "fixed":
        return float(e["area_fraction_fixed"])
    lo, hi = e["area_fraction_clip"]
    return float(np.clip(label_density * float(e["density_factor"]), lo, hi))


def budget_select(prob: np.ndarray, eligible: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(prob.shape, bool)
    idx = np.flatnonzero(eligible.ravel() & np.isfinite(prob.ravel()))
    if n <= 0 or not len(idx):
        return out
    order = np.argsort(-prob.ravel()[idx], kind="stable")
    out.ravel()[idx[order[:min(n, len(idx))]]] = True
    return out


def finalise(values: np.ndarray, template_valid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(template_valid.shape, np.nan, np.float32)
    out[template_valid] = 0.0
    inside = template_valid & valid & np.isfinite(values)
    out[inside] = np.clip(values[inside], 0.0, 1.0).astype(np.float32)
    return out


def validate(path: Path, template_path: Path, expected: np.ndarray, policy: str) -> dict:
    import rasterio

    checks = {}
    with rasterio.open(template_path) as t, rasterio.open(path) as s:
        checks["count_1"] = s.count == 1
        checks["dtype_float32"] = s.dtypes[0] == "float32"
        checks["shape_equal"] = (s.height, s.width) == (t.height, t.width)
        checks["crs_equal"] = s.crs == t.crs
        checks["transform_equal"] = bool(np.allclose(tuple(s.transform), tuple(t.transform), atol=1e-6))
        a = s.read(1)
        tv = t.read(1)
        tvalid = np.isfinite(tv) & ((tv != t.nodata) if t.nodata is not None and np.isfinite(t.nodata) else True)
    checks["nan_pattern_equals_template"] = bool(np.array_equal(~np.isfinite(a), ~tvalid))
    fin = a[np.isfinite(a)]
    checks["range_0_1"] = bool(fin.size and fin.min() >= 0.0 and fin.max() <= 1.0)
    if policy == "area_budget_binary":
        checks["binary_values"] = bool(np.isin(fin, (0.0, 1.0)).all())
    checks["readback_equal"] = bool(np.array_equal(a, expected, equal_nan=True))
    return {"policy": policy, "checks": checks, "ok": all(checks.values()), "n_finite": int(fin.size),
            "n_positive": int((fin > 0).sum()), "mean": float(fin.mean()) if fin.size else float("nan"),
            "sha256": sha256_file(path), "bytes": Path(path).stat().st_size}


def export(out_dir: Path, template_path: Path, template_profile: dict, prob: np.ndarray, template_valid: np.ndarray,
           valid: np.ndarray, labels: np.ndarray, cfg: dict, provenance: dict) -> dict:
    e = cfg["export"]
    label_density = float((labels & valid).sum()) / max(float(valid.sum()), 1.0)
    f = area_fraction(cfg, label_density)
    foot = template_valid & valid
    n = int(round(f * foot.sum()))
    elig = foot if e["known_faults_in_budget"] else foot & ~labels
    sel = budget_select(prob, elig, n)
    maps = {"continuous": finalise(prob, template_valid, valid),
            "area_budget_binary": finalise(sel.astype(np.float32), template_valid, valid)}
    prof = {"crs": template_profile["crs"], "transform": template_profile["transform"]}
    files = {"continuous": out_dir / "e6_continuous.tif", "area_budget_binary": out_dir / "e6_area_budget_binary.tif"}
    report = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "final_policy": e["final"],
              "area_fraction": f, "area_fraction_rule": e["area_fraction_rule"], "label_density": label_density,
              "selected_px": int(sel.sum()), "budget_px": n, "footprint_px": int(foot.sum()),
              "known_faults_in_budget": bool(e["known_faults_in_budget"]), "provenance": provenance, "files": {}}
    for k, p in files.items():
        atomic_write_geotiff(p, maps[k], prof, [f"e6_{k}"])
        report["files"][k] = validate(p, template_path, maps[k], k)
    final = out_dir / "submission_e6_final.tif"
    atomic_write_geotiff(final, maps[e["final"]], prof, ["e6_fault_score"])
    report["files"]["final"] = validate(final, template_path, maps[e["final"]], e["final"])
    report["ok"] = all(v["ok"] for v in report["files"].values())
    atomic_write_json(out_dir / "submission_validation.json", report)
    atomic_write_text(out_dir / "submission_sha256.txt",
                      "".join(f"{v['sha256']}  {Path(files.get(k, final)).name}\n" for k, v in report["files"].items()))
    return report


def read_validation(out_dir: Path) -> dict:
    p = Path(out_dir) / "submission_validation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
