"""Milestone B: full-region prediction of the selected B0 / B1 combination and a validated competition GeoTIFF.

    raw:        predictions/<member>__full_region_raw.tif     (fold-average sigmoid of each member; cached)
                predictions/<candidate>__score.tif            (fused rank score, before any transform)
    submission: submissions/<candidate>.tif                   (transformed; never overwrites the raw maps)
                submissions/<candidate>.json                  (manifest: members, checkpoints, transforms, checks)

Full-region rule = the development rule applied to one zone (the whole footprint): per-zone rank of each member,
fusion weight w, representation, area fraction of the footprint, known-fault ring (known = every mapped fault),
optional known-fault fill. Nothing is uploaded.

Known-fault handling (docs/GEMS_OFFICIAL_METRIC_AUDIT.md): masked pixels do not count towards penalty terms, so
``known_fill`` cannot lower the score under either reading; it can only help if masked predictions support TP,
which the organizers' wording suggests they do not. Default off; one probe submission can settle it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..artifacts.atomic import atomic_save, atomic_write_json
from ..evaluation.comparative import file_fp
from . import representation as R


def member_record(inv_row) -> dict:
    """Provenance of one member from the M6 inventory row (read-only)."""
    from ..inference.full_raster import cv_checkpoints

    run = Path(str(inv_row["run_dir"]))
    ck = cv_checkpoints(run)
    g = (lambda k: "" if inv_row.get(k) is None or (isinstance(inv_row.get(k), float) and np.isnan(inv_row.get(k)))
         else str(inv_row.get(k)))
    return {"model": str(inv_row["model"]), "stem": g("stem"), "experiment_id": g("experiment_id"), "run_dir": str(run),
            "status": g("status"), "config_hash": g("config_hash"), "git_commit": g("git_commit"),
            "feature_version": g("feature_version"), "protocol_hash": g("protocol_hash"), "oof": g("oof"),
            "checkpoints": [{"file": f.name, "fp": file_fp(f)} for f in ck]}


def _fp(member: dict, source: str) -> str:
    from ..pipeline.m6 import fingerprint

    return fingerprint(source, member["model"], member["checkpoints"] if source == "fold_average" else member["oof"])


def member_full_map(ctx, ws, member: dict, source: str = "fold_average", device=None) -> tuple[np.ndarray, dict]:
    """Raw full-region map of one member, cached under predictions/ (reused while its fingerprint matches)."""
    from ..ensemble import read_map

    path = ws.root / "predictions" / f"{member['model']}__full_region_raw.tif"
    meta_p = path.with_suffix(".json")
    fp = _fp(member, source)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    if path.exists() and meta.get("fingerprint") == fp and not ws.cfg.get("overwrite_existing"):
        return read_map(path), {**meta, "reused": True}
    t0 = time.perf_counter()
    if source == "fold_average":
        from ..inference.full_raster import cnn5_full_region

        if not member["checkpoints"]:
            raise FileNotFoundError(f"{member['model']}: no CV checkpoints in {member['run_dir']}/checkpoints; "
                                    "stage them from Drive or set submission.source='oof_stitched'")
        p = cnn5_full_region(ctx, member["run_dir"], device)
    else:
        p = np.array(ctx.oof(member["model"]), np.float32)
    p = np.where(ctx.data.valid, p, np.nan).astype(np.float32)
    from gems_eda.io import write_geotiff

    atomic_save(path, lambda t: write_geotiff(t, {"probability_full_region": p}, ctx.data.grid))
    meta = {"model": member["model"], "source": source, "fingerprint": fp, "seconds": round(time.perf_counter() - t0, 1),
            "finite_frac_valid": float(np.isfinite(p[ctx.data.valid]).mean()), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    atomic_write_json(meta_p, meta)
    return p, {**meta, "reused": False}


BUDGET_DOMAINS = ("eligible", "evaluation_zone")


def ring_zone(known: np.ndarray, ring_px: int) -> np.ndarray:
    """Pixels within ``ring_px`` of a known fault, the known pixels themselves excluded (an experimental
    suppression region; NOT an organizer exclusion: the official mask is the known pixels only)."""
    from scipy import ndimage

    from ..postprocess.topology import _disk

    known = np.asarray(known, bool)
    if ring_px <= 0 or not known.any():
        return np.zeros(known.shape, bool)
    return ndimage.binary_dilation(known, structure=_disk(int(ring_px))) & ~known


def eligible_mask(data, ring_px: int = 0, domain: str = "eligible") -> np.ndarray:
    """Budget domain of a full-region submission.

    ``eligible``         valid & template finite & not a known fault (masked pixel-exactly by the scorer) & not in the
                         optional suppression ring: every selected pixel can change the score.
    ``evaluation_zone``  valid & template finite (historical rule: known pixels compete for the budget)."""
    if domain not in BUDGET_DOMAINS:
        raise KeyError(f"budget domain {domain!r}; known {BUDGET_DOMAINS}")
    m = np.asarray(data.valid, bool) & np.asarray(data.template_valid, bool)
    if domain == "evaluation_zone":
        return m
    return m & ~np.asarray(data.labels, bool) & ~ring_zone(data.labels, int(ring_px or 0))


def area_ledger(q: np.ndarray, data, ring_px: int = 0, step: str = "") -> dict:
    """Positive (q > 0) area of one prediction state, split by pixel class, as counts and fractions.

    Fractions are reported both as a share of the valid footprint and as a share of the eligible domain, so a
    budget leaking onto masked pixels (known faults, ring, template nodata, outside the footprint) is visible."""
    a = np.nan_to_num(np.asarray(q, np.float32), nan=0.0) > 0
    valid, tv, known = (np.asarray(x, bool) for x in (data.valid, data.template_valid, data.labels))
    ring = ring_zone(known, int(ring_px or 0))
    elig = valid & tv & ~known & ~ring
    n_valid, n_elig = max(int(valid.sum()), 1), max(int(elig.sum()), 1)
    return {"step": step, "positive_px": int(a.sum()), "frac_of_valid": float((a & valid).sum() / n_valid),
            "frac_of_eligible": float((a & elig).sum() / n_elig), "eligible_px": int(elig.sum()),
            "on_eligible_px": int((a & elig).sum()), "on_known_px": int((a & known).sum()),
            "on_ring_px": int((a & ring).sum()), "on_template_nodata_px": int((a & ~tv).sum()),
            "outside_valid_px": int((a & ~valid).sum())}


def transform(maps: list[np.ndarray], data, params: dict, rep: dict, *, known_fill: bool = False,
              sigma_px: float = 1.0, domain: str = "eligible") -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(fused score, transformed prediction, area ledger) on the full footprint with the development rule.

    The area fraction counts pixels of the budget domain (``eligible_mask``); the ledger records the positive
    area after every step (selection, ring, fill) so the effect of each masking operation is explicit."""
    valid = data.valid
    ring = int(params.get("ring", 0) or 0)
    s = R.fuse(R.rank01(maps[0], valid), R.rank01(maps[1], valid), float(params["w"]))
    elig = eligible_mask(data, ring, domain)
    q = R.apply(R.with_ring(rep, ring), s, valid, float(params["frac"]), known=data.labels, sigma_px=sigma_px,
                eligible=elig)
    ledger = [area_ledger(q, data, ring, f"selected+post (domain={domain})")]
    if known_fill:
        q = R.op_known_fill(q, {}, {"known": data.labels})
        ledger.append(area_ledger(q, data, ring, "known_fill"))
    return np.where(valid, s, np.nan).astype(np.float32), q, ledger


def finalise(q: np.ndarray, data, outside_valid: str = "template") -> np.ndarray:
    """float32 in [0, 1]; NaN where the official template is NaN (outside the competition bounds), 0 where the
    template is finite but the data footprint is not (``'template'``), or NaN outside the data footprint too
    (``'nan'``). The template decides nodata: a finite value is never written where the template is NaN, even
    inside the data footprint (the earlier rule ``~valid & ~template_valid`` let such pixels through)."""
    out = np.clip(np.nan_to_num(np.asarray(q, np.float32), nan=0.0), 0.0, 1.0).astype(np.float32)
    tv = np.asarray(data.template_valid, bool)
    if outside_valid == "template":
        out[~tv] = np.nan
        out[tv & ~data.valid] = 0.0
    else:
        out[~data.valid | ~tv] = np.nan
    return out


def validate(path: Path, data, outside_valid: str = "template") -> dict:
    """Hard checks against the template (raises AssertionError listing every failure) + descriptive facts."""
    import rasterio

    t = data.template_profile
    with rasterio.open(path) as src:
        a = src.read(1)
        res = {"file": Path(path).name, "count": src.count, "dtype": src.dtypes[0], "crs": str(src.crs),
               "epsg": src.crs.to_epsg() if src.crs else None, "res": tuple(src.res), "nodata": src.nodata,
               "shape": (src.height, src.width), "same_crs": src.crs == t["crs"],
               "same_transform": src.transform == t["transform"],
               "same_shape": (src.height, src.width) == (t["height"], t["width"])}
    fin = np.isfinite(a)
    v = a[fin]
    tv, valid = np.asarray(data.template_valid, bool), np.asarray(data.valid, bool)
    scored = valid & tv                                       # inside the bounds AND inside the data footprint
    res.update(min=float(v.min()) if v.size else np.nan, max=float(v.max()) if v.size else np.nan,
               has_inf=bool(np.isinf(a).any()), finite_inside_valid=float(fin[valid].mean()),
               finite_inside_scored=float(fin[scored].mean()) if scored.any() else np.nan,
               valid_but_template_nan_px=int((valid & ~tv).sum()),
               nan_where_template_nan=bool((~fin[~tv]).all()) if (~tv).any() else True,
               finite_where_template_finite=float(fin[tv].mean()) if tv.any() else np.nan,
               positive_frac_valid=float((a[valid] > 0).mean()), mean_valid=float(np.nanmean(a[valid])),
               positive_px_on_known=int(((np.nan_to_num(a) > 0) & np.asarray(data.labels, bool)).sum()),
               unique_values_sample=int(len(np.unique(v[:1_000_000]))) if v.size else 0)
    fails = []
    if res["count"] != 1 or res["dtype"] != "float32":
        fails.append("single float32 band")
    if res["epsg"] != 32611:
        fails.append(f"EPSG {res['epsg']} != 32611")
    if not (res["same_crs"] and res["same_transform"] and res["same_shape"]):
        fails.append("grid differs from example_submission.tif")
    if not (abs(res["res"][0] - 100.0) < 1e-6 and abs(res["res"][1] - 100.0) < 1e-6):
        fails.append(f"resolution {res['res']} != 100 m")
    if v.size and (res["min"] < 0 or res["max"] > 1):
        fails.append("values outside [0, 1]")
    if res["has_inf"]:
        fails.append("infinite values")
    if scored.any() and res["finite_inside_scored"] < 1.0:
        fails.append("NaN inside the data footprint (where the template is finite)")
    if not res["nan_where_template_nan"]:
        fails.append("finite values where the template is NaN")
    if outside_valid == "template" and res["finite_where_template_finite"] < 1.0:
        fails.append("NaN where the template is finite")
    res["checks_failed"] = fails
    assert not fails, f"submission {path} fails: {fails}"
    return res


def export(ws, name: str, score: np.ndarray, q: np.ndarray, data, manifest: dict, outside_valid: str = "template",
           ledger: list[dict] | None = None, ring_px: int = 0) -> dict:
    from gems_eda.io import write_geotiff

    from ..submission import write_submission

    raw_p = ws.root / "predictions" / f"{name}__score.tif"
    atomic_save(raw_p, lambda t: write_geotiff(t, {"fused_rank_score": np.asarray(score, np.float32)}, data.grid))
    ws._track(raw_p)
    arr = finalise(q, data, outside_valid)
    led = list(ledger or []) + [area_ledger(arr, data, ring_px, f"finalised (outside_valid={outside_valid})")]
    sub_p = ws.root / "submissions" / f"{name}.tif"
    atomic_save(sub_p, lambda t: write_submission(arr, data, Path(t)))
    ws._track(sub_p)
    checks = validate(sub_p, data, outside_valid)
    info = {**manifest, "candidate": name, "submission": sub_p.name, "raw_score": raw_p.name,
            "outside_valid": outside_valid, "area_ledger": led, "validation": checks,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ws.save_json(json.loads(json.dumps(info, default=str)), f"submissions/{name}.json")
    return info


def candidate_name(params: dict, known_fill: bool, domain: str = "evaluation_zone") -> str:
    """File stem; the eligible budget domain gets its own suffix so files of the historical rule are never
    overwritten by the corrected one."""
    s = f"gems3_{params['rep']}_w{float(params['w']):g}_f{float(params['frac']):g}_r{int(params.get('ring', 0) or 0)}"
    return s + ("_elig" if domain == "eligible" else "") + ("_knownfill" if known_fill else "")


def resolve_parameters(selection: dict, sub_cfg: dict, primary_task: str) -> dict:
    """Submission parameters from manifests/selected_representation.json (written by the metric stage)."""
    ch = selection["choice"][primary_task]["params_all"]
    params = {"rep": ch["rep"], "w": float(ch["w"]), "frac": float(ch["frac"]), "ring": int(ch.get("ring", 0) or 0)}
    if sub_cfg["representation"] == "binary":
        params["rep"] = "binary"
    src = sub_cfg["area_frac_source"]
    if src == "geo":
        lookup = selection.get("geo_frac") or {}
        key = f"{params['rep']}|{params['w']:g}"
        f = lookup.get(key, lookup.get(f"binary|{params['w']:g}"))
        if f is None:
            raise KeyError(f"no geo area fraction for {key}; run the metric stage with 'geo' in tasks")
        params["frac"], params["frac_source"] = float(f), f"geo all-zone choice ({key})"
    elif isinstance(src, (int, float)):
        params["frac"], params["frac_source"] = float(src), "explicit"
    else:
        params["frac_source"] = f"{primary_task} all-zone choice (share of its evaluation zones)"
    if sub_cfg["known_ring_px"] != "selected":
        params["ring"] = int(sub_cfg["known_ring_px"])
    return params
