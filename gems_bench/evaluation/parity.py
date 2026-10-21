"""Experiment A2: metric parity and convention audit.

Three independent checks
------------------------
1. ``reference_dti``: a second implementation of the distance-weighted Tversky index written from the
   problem statement with a KD-tree (no shared code with ``metric.dti_components``), compared on synthetic
   cases that probe each convention: kernel boundary (k(300 m) = 0), offsets of 1-3 px, clipping, NaN,
   evaluation mask, known-fault exclusion, empty truth, soft probabilities.
2. ``official_adapter``: if the organisers' scorer is available locally (e.g. a script from the DrivenData
   benchmark repository - never downloaded by this package), it is imported from a user-given path and
   compared on the same cases and on real OOF maps. Without it, parity with the leaderboard is UNVERIFIED.
3. ``convention_audit``: label rasterisation, nodata, template grid and evaluation-region facts read from the
   data, plus the list of conventions that only the official scorer / leaderboard can settle.

A local DTI is never called "leaderboard-equivalent" unless check 2 passed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..metric import DTI_ALPHA, DTI_BETA, DTI_EPS, DTI_RADIUS_M, dti_components


def reference_dti(pred, truth, *, eval_mask=None, exclude_mask=None, radius_m: float = DTI_RADIUS_M,
                  pixel_m=(100.0, 100.0), alpha=DTI_ALPHA, beta=DTI_BETA, eps=DTI_EPS) -> dict:
    """Problem-statement DTI by explicit neighbour search (slow; for rasters up to ~10^5 px)."""
    from scipy.spatial import cKDTree

    p = np.where(np.isfinite(pred), pred, 0.0).astype(np.float64)
    p = np.clip(p, 0, 1)
    ev = np.ones(p.shape, bool) if eval_mask is None else np.asarray(eval_mask, bool)
    ex = np.zeros(p.shape, bool) if exclude_mask is None else np.asarray(exclude_mask, bool)
    p[~ev] = 0.0
    G = np.asarray(truth, bool) & ev & ~ex
    rr, cc = np.indices(p.shape)
    xy = np.stack([rr.ravel() * pixel_m[0], cc.ravel() * pixel_m[1]], 1)
    pos = p.ravel() > 0
    g_xy = xy[G.ravel()]
    n = len(g_xy)
    tp = 0.0
    if n and pos.any():
        tree = cKDTree(xy[pos])
        pv = p.ravel()[pos]
        for i, nb in enumerate(tree.query_ball_point(g_xy, r=radius_m)):
            if nb:
                d = np.hypot(*(xy[pos][nb] - g_xy[i]).T)
                tp += float(np.max(pv[nb] * np.clip(1 - d / radius_m, 0, None)))
    fp_region = (ev & ~ex).ravel() & pos
    if n:
        dist, _ = cKDTree(g_xy).query(xy[fp_region])
        kd = np.clip(1 - dist / radius_m, 0, None)
    else:
        kd = np.zeros(int(fp_region.sum()))
    fp = float((p.ravel()[fp_region] * (1 - kd)).sum())
    fn = n - tp
    return {"dti": tp / (tp + alpha * fp + beta * fn + eps), "tp_w": tp, "fp_w": fp, "fn_w": fn, "n_truth": n}


def _line(h=40, w=40, r=20, c0=5, c1=35):
    t = np.zeros((h, w), bool)
    t[r, c0:c1] = True
    return t


def parity_cases(seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    t = _line()
    cases = [{"case": "perfect", "pred": t.astype(float), "truth": t}]
    for off in (1, 2, 3, 4):
        p = np.zeros_like(t, float)
        p[20 + off, 5:35] = 1.0
        cases.append({"case": f"offset_{off}px", "pred": p, "truth": t})
    cases.append({"case": "all_zero", "pred": np.zeros_like(t, float), "truth": t})
    cases.append({"case": "all_one", "pred": np.ones_like(t, float), "truth": t})
    nanp = t.astype(float)
    nanp[:, :10] = np.nan
    cases.append({"case": "nan_as_zero", "pred": nanp, "truth": t})
    cases.append({"case": "clip_out_of_range", "pred": t * 1.7 - 0.2, "truth": t})
    soft = np.clip(rng.normal(0.2, 0.2, t.shape) + 0.6 * t, 0, 1)
    cases.append({"case": "soft_random", "pred": soft, "truth": t})
    ev = np.ones_like(t)
    ev[:, 25:] = False
    cases.append({"case": "eval_mask", "pred": soft, "truth": t, "eval_mask": ev})
    ex = np.zeros_like(t)
    ex[18:23, 5:15] = True
    cases.append({"case": "exclude_mask", "pred": soft, "truth": t, "exclude_mask": ex})
    cases.append({"case": "empty_truth", "pred": soft, "truth": np.zeros_like(t)})
    return cases


def official_adapter(path: Path, func: str = "score") -> Callable:
    """Import ``func`` from a local copy of the official scorer. Expected call: ``func(pred, truth) -> float``
    (arrays) - wrap it here if the organisers' signature differs (e.g. file paths)."""
    spec = importlib.util.spec_from_file_location("gems_official_scorer", Path(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func)


def run_parity(official: Callable | None = None, tol: float = 1e-9) -> pd.DataFrame:
    rows = []
    for c in parity_cases():
        kw = {k: c[k] for k in ("eval_mask", "exclude_mask") if k in c}
        loc = dti_components(np.asarray(c["pred"], np.float64), c["truth"], **kw).dti
        ref = reference_dti(c["pred"], c["truth"], **kw)["dti"]
        r = {"case": c["case"], "local": loc, "reference": ref, "abs_diff_ref": abs(loc - ref),
             "pass_reference": abs(loc - ref) <= tol}
        if official is not None and not kw:
            off = float(official(np.asarray(c["pred"], np.float32), c["truth"].astype(np.uint8)))
            r.update(official=off, abs_diff_official=abs(loc - off), pass_official=abs(loc - off) <= 1e-6)
        rows.append(r)
    return pd.DataFrame(rows)


def convention_audit(ctx) -> pd.DataFrame:
    """Facts read from the data + the conventions only the official scorer / leaderboard can confirm."""
    d = ctx.data
    t_valid, valid = d.template_valid, d.valid
    rows = [
        ("label_rule", "value >= 1 is a fault (organizer reference solution)", str(d.label_info.get("values_preview")), "verified-local"),
        ("labels_outside_valid", "label pixels outside the data footprint", int((d.labels & ~valid).sum()), "verified-local"),
        ("template_same_grid", "example_submission grid == feature grid", bool(d.template_matches()), "verified-local"),
        ("template_valid_vs_data_valid", "IoU of template finite mask and data footprint",
         float((t_valid & valid).sum() / max((t_valid | valid).sum(), 1)), "verified-local"),
        ("pixel_m", "pixel size (m)", f"{d.grid.pixel_km_x * 1000:.1f} x {d.grid.pixel_km_y * 1000:.1f}", "verified-local"),
        ("alpha_beta", "Tversky alpha / beta", f"{DTI_ALPHA} / {DTI_BETA}", "problem statement"),
        ("radius", "distance support R (k(d) = max(0, 1 - d/R); k(R) = 0)", f"{DTI_RADIUS_M} m", "problem statement"),
        ("eps", "denominator epsilon", DTI_EPS, "UNSPECIFIED by organisers"),
        ("nan_policy", "NaN / nodata predictions", "treated as 0 locally", "UNVERIFIED (official scorer)"),
        ("clip_policy", "predictions outside [0, 1]", "clipped locally; submit within [0, 1]", "UNVERIFIED"),
        ("eval_region", "pixels scored on the hidden test", "local: data footprint", "UNVERIFIED (template mask? full raster?)"),
        ("known_faults", "are released faults truth, excluded, or FP on the hidden test?", "both policies reported", "UNKNOWN: leaderboard probes"),
        ("hidden_truth_density", "hidden-test fault density vs released map", "unknown", "UNKNOWN"),
    ]
    return pd.DataFrame(rows, columns=["item", "description", "value", "status"])


def submission_audit(path: Path, data) -> dict:
    """``submission.check_submission`` plus metadata the leaderboard depends on (nodata, tiling, band count)."""
    import rasterio

    from ..submission import check_submission

    res = check_submission(path, data)
    with rasterio.open(path) as src:
        res.update(nodata=src.nodata, tiled=src.profile.get("tiled"), compress=src.profile.get("compress"),
                   band_description=src.descriptions[0], bounds=tuple(src.bounds))
    res["template_nodata"] = data.template_profile.get("nodata")
    return res
