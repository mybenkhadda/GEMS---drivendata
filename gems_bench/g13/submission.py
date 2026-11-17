"""Official-format GeoTIFF export and validation (sections 31-34).

Geometry comes from ``example_submission.tif``. Values: float32 in [0, 1] where the template is finite (0 where
the template is finite but no data exists), NaN where the template is NaN. Known-fault pixels: ``zero`` (default)
writes 0 on the pixel-exact mapped faults, which is neutral under the organizers' stated pixel-exact mask; no
buffer is ever applied by default. Nothing is uploaded: the leaderboard step is manual.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_write_json, sha256_file

COMPETITION_URL = "https://www.drivendata.org/competitions/306/competition-doe-gems/"
HISTORY_COLUMNS = [
    "date_utc",
    "submission_file",
    "sha256",
    "candidate",
    "model",
    "representation",
    "public_score",
    "offline_geo_proxy",
    "offline_disc_proxy",
    "offline_test_proxy",
    "pred_mass",
    "positive_px",
    "notes",
]


def finalise(q: np.ndarray, data, known_policy: str = "zero") -> np.ndarray:
    out = np.clip(np.nan_to_num(np.asarray(q, np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    if known_policy == "zero":
        out[data.labels] = 0.0
    out[~data.template_valid] = np.nan
    return out


def write(q: np.ndarray, data, path: Path, *, known_policy: str = "zero", tags: dict | None = None) -> Path:
    import rasterio

    a = finalise(q, data, known_policy)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(data.template_path) as t:
        prof = t.profile.copy()
    prof.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    tmp = path.with_name(path.stem + ".partial.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(a, 1)
        dst.set_band_description(1, "fault_confidence")
        if tags:
            dst.update_tags(**{k: str(v)[:1000] for k, v in tags.items()})
    tmp.replace(path)
    return path


def validate(path: Path, data, known_policy: str = "zero") -> dict:
    """Hard checks against the template (``checks_failed`` empty = valid) plus descriptive facts."""
    import rasterio

    with rasterio.open(data.template_path) as t:
        tprof = {"crs": t.crs, "transform": t.transform, "shape": (t.height, t.width)}
    with rasterio.open(path) as src:
        a = src.read(1)
        res = {
            "file": Path(path).name,
            "path": str(path),
            "count": src.count,
            "dtype": src.dtypes[0],
            "epsg": src.crs.to_epsg() if src.crs else None,
            "res_m": tuple(src.res),
            "same_crs": src.crs == tprof["crs"],
            "same_transform": src.transform == tprof["transform"],
            "same_shape": (src.height, src.width) == tprof["shape"],
            "bounds": tuple(src.bounds),
        }
    tv = data.template_valid
    fin = np.isfinite(a)
    inside = a[tv]
    res.update(
        valid_px=int(tv.sum()),
        finite_inside=int(np.isfinite(inside).sum()),
        nan_inside=int((~np.isfinite(inside)).sum()),
        finite_outside=int(fin[~tv].sum()),
        min=float(np.nanmin(inside)) if inside.size else np.nan,
        max=float(np.nanmax(inside)) if inside.size else np.nan,
        mean=float(np.nanmean(inside)) if inside.size else np.nan,
        std=float(np.nanstd(inside)) if inside.size else 0.0,
        pred_mass=float(np.nansum(inside)),
        positive_px=int((np.nan_to_num(inside) > 0).sum()),
        positive_frac=float((np.nan_to_num(inside) > 0).mean()) if inside.size else 0.0,
        n_unique=int(len(np.unique(inside[np.isfinite(inside)][:2_000_000]))),
        positive_on_known_px=int(((np.nan_to_num(a) > 0) & data.labels).sum()),
        sha256=sha256_file(path),
        size_mb=Path(path).stat().st_size / 2**20,
    )
    fails = []
    if res["count"] != 1 or res["dtype"] != "float32":
        fails.append("must be one float32 band")
    if res["epsg"] != 32611:
        fails.append(f"EPSG {res['epsg']} != 32611")
    if not (res["same_crs"] and res["same_transform"] and res["same_shape"]):
        fails.append("grid differs from example_submission.tif")
    if abs(res["res_m"][0] - 100) > 1e-6 or abs(res["res_m"][1] - 100) > 1e-6:
        fails.append("resolution != 100 m")
    if res["nan_inside"]:
        fails.append("NaN inside the template coverage")
    if res["finite_outside"]:
        fails.append("finite values outside the template coverage")
    if np.isfinite(res["min"]) and (res["min"] < 0 or res["max"] > 1):
        fails.append("values outside [0, 1]")
    if res["positive_px"] == 0 or res["std"] == 0:
        fails.append("degenerate (constant or empty) prediction")
    if known_policy == "zero" and res["positive_on_known_px"]:
        fails.append("positive values on known-fault pixels under the 'zero' policy")
    res["checks_failed"] = fails
    res["valid"] = not fails
    return res


def export_candidate(
    name: str, q: np.ndarray, data, ws, meta: dict, known_policy: str = "zero", extra_names: tuple = ()
) -> dict:
    path = ws.path("submissions", f"{name}.tif")
    write(
        q,
        data,
        path,
        known_policy=known_policy,
        tags={"candidate": name, "model": meta.get("model", ""), "representation": meta.get("representation", "")},
    )
    rep = validate(path, data, known_policy)
    copies = []
    for extra in extra_names:
        p2 = ws.path("submissions", extra)
        p2.write_bytes(path.read_bytes())
        copies.append(str(p2))
    record = {
        **meta,
        "candidate": name,
        "validation": rep,
        "copies": copies,
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "known_policy": known_policy,
        "upload": f"manual: {COMPETITION_URL} -> Submissions -> upload {path.name}",
    }
    atomic_write_json(ws.path("submissions", f"{name}.json"), record)
    return record


def history_path(ws) -> Path:
    return ws.path("submissions", "submission_history.csv")


def log_leaderboard(ws, submission_file: str, public_score: float, notes: str = "") -> pd.DataFrame:
    """Append a REAL platform score (typed in by the user after uploading). Never called automatically."""
    p = history_path(ws)
    h = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    meta_p = ws.root / "submissions" / (Path(submission_file).stem + ".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    v = meta.get("validation", {})
    row = {
        "date_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        "submission_file": Path(submission_file).name,
        "sha256": v.get("sha256", ""),
        "candidate": meta.get("candidate", ""),
        "model": meta.get("model", ""),
        "representation": meta.get("representation", ""),
        "public_score": float(public_score),
        "offline_geo_proxy": meta.get("geo_proxy_dti"),
        "offline_disc_proxy": meta.get("disc_proxy_dti"),
        "offline_test_proxy": meta.get("test_proxy_dti"),
        "pred_mass": v.get("pred_mass"),
        "positive_px": v.get("positive_px"),
        "notes": notes,
    }
    h = pd.concat([h, pd.DataFrame([row])], ignore_index=True)
    h.to_csv(p, index=False)
    return h


def leaderboard_feedback(ws) -> pd.DataFrame:
    """Offline proxy vs public score for every logged submission (mismatch analysis)."""
    p = history_path(ws)
    if not p.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    h = pd.read_csv(p)
    for c in ("offline_geo_proxy", "offline_disc_proxy", "offline_test_proxy"):
        if c in h:
            h[f"gap_{c}"] = h.public_score - h[c]
    return h
