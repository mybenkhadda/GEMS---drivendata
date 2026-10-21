"""Full-region inference and submission export (never uploads anything).

Format (DrivenData problem description): single-band float32 GeoTIFF, EPSG:32611, 100 m,
same bounds as the training data, values in [0, 1], null / NaN outside the bounds.
Here: profile copied from ``example_submission.tif``; NaN outside the valid footprint.

Inference mode ``fold_average``: every geographic-fold model predicts the whole region and
the maps are averaged (as the organizer reference averages its CV iterations); then the
output transform chosen on all OOF folds is applied.

Known-fault policy: the test labels are faults *absent* from the USGS map, and the
organizers do not say whether known faults are excluded from scoring. Two variants are
therefore exported per model and only leaderboard feedback can arbitrate:
``model``             the model probability;
``model_max_known``   max(model, mapped-fault mask).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from .data import NativeData
from .ensemble import apply_transform_file, read_map
from .features import FeatureStore
from .persist import ExperimentRun


def gbm_full_map(run: ExperimentRun, data: NativeData, store: FeatureStore) -> np.ndarray:
    import lightgbm as lgb

    from .gbm import predict_mask

    names = run.config["features"]
    maps = []
    for f in sorted(run.models_dir.glob("fold*.txt")):
        model = lgb.Booster(model_file=str(f))
        maps.append(predict_mask(model, store, names, data.valid))
        print(f"  predicted full grid with {f.name}")
    return np.nanmean(np.stack(maps), 0)


def cnn_full_map(run: ExperimentRun, data: NativeData, store: FeatureStore) -> np.ndarray:
    from .cnn import device_of, load_fold_models, predict_mask, prepare_inputs

    device = device_of()
    models = load_fold_models(run, device)
    stack, _, _ = prepare_inputs(models[0][0], data, store)
    maps = [predict_mask(m, stack, data.valid, spec, device) for spec, m in models]
    return np.nanmean(np.stack(maps), 0)


def load_transform(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "output_transform.json").read_text(encoding="utf-8"))


def finalise(prob: np.ndarray, data: NativeData, transform: dict | None, known_policy: str = "model") -> np.ndarray:
    p = prob if transform is None else apply_transform_file(prob, transform)
    p = np.clip(np.nan_to_num(p, nan=0.0), 0, 1).astype(np.float32)
    if known_policy == "model_max_known":
        p = np.maximum(p, data.labels.astype(np.float32))
    p[~data.valid] = np.nan
    return p


def write_submission(p: np.ndarray, data: NativeData, path: Path) -> Path:
    prof = data.template_profile.copy()
    prof.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3,
                tiled=True, blockxsize=256, blockysize=256)
    if not data.template_matches():
        raise ValueError("example_submission.tif grid differs from the feature grid; resample before export")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(p.astype(np.float32), 1)
        dst.set_band_description(1, "fault_probability")
    return path


def check_submission(path: Path, data: NativeData) -> dict:
    """Sanity checks against the template; raises AssertionError on a hard failure."""
    t = data.template_profile
    with rasterio.open(path) as src:
        a = src.read(1)
        res = {
            "file": str(path), "count": src.count, "dtype": src.dtypes[0], "crs": str(src.crs),
            "same_crs": src.crs == t["crs"], "same_transform": src.transform == t["transform"],
            "same_shape": (src.height, src.width) == (t["height"], t["width"]),
            "res": src.res,
        }
    fin = np.isfinite(a)
    res.update(min=float(np.nanmin(a)), max=float(np.nanmax(a)), finite_frac=float(fin.mean()),
               finite_inside_valid=float(fin[data.valid].mean()), finite_outside_valid=int(fin[~data.valid].sum()),
               mean_inside=float(np.nanmean(a[data.valid])), frac_ge_0_5=float((a[data.valid] >= 0.5).mean()))
    assert res["count"] == 1 and res["dtype"] == "float32", res
    assert res["same_crs"] and res["same_transform"] and res["same_shape"], res
    assert 0.0 <= res["min"] and res["max"] <= 1.0, res
    assert res["finite_inside_valid"] == 1.0 and res["finite_outside_valid"] == 0, res
    return res


def export_candidate(name: str, prob: np.ndarray, data: NativeData, transform: dict | None, sub_dir: Path,
                     policies=("model", "model_max_known"), meta: dict | None = None) -> list[dict]:
    out = []
    for pol in policies:
        path = sub_dir / f"{name}__{pol}.tif"
        write_submission(finalise(prob, data, transform, pol), data, path)
        chk = check_submission(path, data)
        info = {"candidate": name, "policy": pol, "path": str(path), "transform": transform, **(meta or {}), **chk}
        (sub_dir / f"{name}__{pol}.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        out.append(info)
    return out


def ensemble_full_map(member_maps: dict[str, np.ndarray], cfg: dict, data: NativeData) -> np.ndarray:
    """Recombine *full-region* member maps with an ensemble config written by ensemble.build_ensembles."""
    from .ensemble import rank_normalise

    names = cfg["members"]
    if cfg["kind"] == "mean":
        return np.nanmean(np.stack([member_maps[k] for k in names]), 0)
    if cfg["kind"] == "rank_mean":
        return np.nanmean(np.stack([rank_normalise(member_maps[k], data.valid) for k in names]), 0)
    w = cfg["weights"]
    return np.sum([w[k] * np.nan_to_num(member_maps[k]) for k in names], 0)


def save_full_map(p: np.ndarray, data: NativeData, path: Path) -> Path:
    from gems_eda.io import write_geotiff

    return write_geotiff(path, {"probability_full_region": p.astype(np.float32)}, data.grid)


__all__ = ["gbm_full_map", "cnn_full_map", "finalise", "write_submission", "check_submission",
           "export_candidate", "ensemble_full_map", "save_full_map", "load_transform", "read_map"]
