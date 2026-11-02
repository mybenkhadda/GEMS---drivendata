"""Full-footprint research prediction (never an automatic submission).

* ``predict_full_raster``: a FULL-fit pixel model (all mapped faults visible) predicted tile by tile over the whole
  footprint, features read in the model's exact column order (schema hash checked), probability GeoTIFF on the
  source grid with tags ROLE=RESEARCH_NOT_A_SUBMISSION, model / schema provenance.
* ``verify_inference_parity``: the training-time feature rows (pool parquet) and the store re-read at the same
  pixels must be identical, and the written raster must equal the model applied to those rows.
* ``export_submission_candidate`` is reachable only through ``submission_gate``: an evaluated PROMOTE_CANDIDATE with
  strict validation, ``RUN['EXPORT_SUBMISSION']`` and a passed parity check. Even then the file is
  ``READY_FOR_MANUAL_UPLOAD``; nothing is uploaded.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_ROLE = "RESEARCH_NOT_A_SUBMISSION"


def write_research_geotiff(path, arrays: dict, grid, tags: dict) -> Path:
    from .raster_io import write_feature_raster

    return write_feature_raster(path, arrays, grid, {"ROLE": RESEARCH_ROLE, **tags})


def predict_full_raster(model, store, out_path, tile_px: int = 512, tags: dict | None = None, progress=print) -> dict:
    """Probability raster of a pixel model over the footprint (window-wise; never the full feature stack)."""
    from .models_pixel import predict_pixel_raster
    from .persistence import sha256_file
    from .raster_io import GroupWriter, iter_windows

    fp = store.footprint()
    grid = store.grid
    t0 = time.perf_counter()
    gw = GroupWriter(out_path, ["probability"], grid, {"ROLE": RESEARCH_ROLE, "MODEL_KIND": model.kind,
                                                       "SCHEMA_HASH": store.schema_hash, **(tags or {})})
    s1 = n = 0.0
    try:
        wins = list(iter_windows(grid.shape, tile_px, 0))
        for i, w in enumerate(wins):
            m = fp[w.grid_slice]
            p = predict_pixel_raster(model, store, (w.row0, w.row1, w.col0, w.col1), m, tile_px) if m.any() else \
                np.full(w.core_shape, np.nan, np.float32)
            gw.write(w, p[None])
            fin = np.isfinite(p)
            s1 += float(p[fin].sum())
            n += float(fin.sum())
            if progress and i % 10 == 0:
                progress(f"  full inference tile {i + 1}/{len(wins)} ({time.perf_counter() - t0:.0f} s)")
        gw.close()
    except Exception:
        gw.abort()
        raise
    return {"path": str(out_path), "sha256": sha256_file(out_path), "n_px": int(n), "mean_probability": s1 / max(n, 1),
            "seconds": round(time.perf_counter() - t0, 1), "role": RESEARCH_ROLE}


def read_full(path) -> np.ndarray:
    from .raster_io import read_band_full

    return read_band_full(path, 1)


def ensemble_full_predictions(paths: dict, weights: dict, footprint: np.ndarray) -> np.ndarray:
    """Weighted mean of footprint ranks of research rasters (rank average; no fitting)."""
    from ..gems3.representation import rank01

    tot = sum(float(weights.get(k, 0)) for k in paths)
    acc = np.zeros(footprint.shape, np.float32)
    for k, p in paths.items():
        wk = float(weights.get(k, 0))
        if wk:
            acc += wk * rank01(read_full(p), footprint)
    return np.where(footprint, acc / max(tot, 1e-12), np.nan).astype(np.float32)


def predict_full_segments(store, prob_full: np.ndarray, labels: np.ndarray, valid: np.ndarray, seg_est, cols: list,
                          seg_cfg: dict, known_cfg: dict, lam: float = 0.5) -> tuple[np.ndarray, pd.DataFrame,
                                                                                   np.ndarray]:
    """Full-region candidates (label-free pools: G06 prediction + physics), segment features with every mapped fault
    as known context (full-region deployment), scored by a segment model fitted on all units; returns
    (labels, scores, segment-assisted map preserving pixel variation)."""
    from ..gems3.representation import rank01, top_mask
    from .datasets import generate_candidates, segment_features
    from .feature_context import permitted_fault_context
    from .models_segment import _predict, segment_to_raster

    fp = np.asarray(valid, bool) & np.isfinite(prob_full)
    pc = seg_cfg.get("pools") or {}
    pools = {}
    if "g06" in pc:
        pools["g06"] = top_mask(rank01(np.nan_to_num(prob_full), fp), fp, float(pc["g06"]["frac"]))
    if "physics" in pc and pc["physics"]["feature"] in store.names:
        a = store.load([pc["physics"]["feature"]])[0]
        pools["physics"] = top_mask(rank01(np.nan_to_num(a, nan=-np.inf), fp), fp, float(pc["physics"]["frac"]))
    lab = generate_candidates(pools, seg_cfg)
    kctx = permitted_fault_context(labels, valid, known_cfg)
    h, w = lab.shape
    T = segment_features(lab, store, (0, h, 0, w), seg_cfg, kctx, {"g06": rank01(np.nan_to_num(prob_full), fp)})
    if not len(T):
        return lab, pd.DataFrame(), np.where(fp, prob_full, np.nan)
    for c in cols:
        if c not in T:
            T[c] = np.nan
    T["score"] = _predict(seg_est, T[cols].to_numpy(np.float32))
    smap = segment_to_raster(lab, T[["label", "score"]], prob_full, fp, lam)
    return lab, T, smap


def verify_inference_parity(model, store, raster_path, pool_X: pd.DataFrame, pool_meta: pd.DataFrame, n: int = 2000,
                            seed: int = 0, atol: float = 1e-5) -> dict:
    """(1) store re-read == training feature rows; (2) raster == model(training feature rows) at the same pixels."""
    from .raster_io import read_points

    rng = np.random.default_rng(seed)
    M = pool_meta.sample(min(n, len(pool_meta)), random_state=int(rng.integers(1 << 31)))
    rows, cols = M.row.to_numpy(), M.col.to_numpy()
    Xs = store.sample(model.feature_columns, rows, cols)
    Xp = pool_X.set_index("row_id").loc[M.row_id, model.feature_columns].to_numpy(np.float32)
    same_nan = np.array_equal(np.isnan(Xs), np.isnan(Xp))
    fd = float(np.nanmax(np.abs(Xs - Xp))) if np.isfinite(Xs).any() else 0.0
    pr = model.predict_proba(Xp)
    rv = read_points(raster_path, [1], rows, cols)[:, 0]
    ok = np.isfinite(rv)
    pdiff = float(np.max(np.abs(rv[ok] - pr[ok]))) if ok.any() else float("nan")
    status = "PASS" if same_nan and fd <= atol and ok.any() and pdiff <= 1e-4 else "FAIL"
    return {"status": status, "n_points": int(len(M)), "feature_max_abs_diff": fd, "feature_nan_pattern_equal":
            bool(same_nan), "prediction_max_abs_diff": pdiff, "n_points_in_raster": int(ok.sum())}


def submission_gate(decision: pd.DataFrame, model: str, run_flags: dict, parity: dict | None) -> tuple[bool, list]:
    why = []
    if not run_flags.get("EXPORT_SUBMISSION"):
        why.append("RUN['EXPORT_SUBMISSION'] is False")
    row = decision[decision.model == model] if decision is not None and len(decision) else pd.DataFrame()
    if not len(row):
        why.append(f"no evaluated promotion decision for {model}")
    elif row.iloc[0].status != "PROMOTE_CANDIDATE":
        why.append(f"{model} status {row.iloc[0].status} (only PROMOTE_CANDIDATE may be exported)")
    if not parity or parity.get("status") != "PASS":
        why.append("inference parity not verified (PASS required)")
    return (not why), why


def export_submission_candidate(score_full: np.ndarray, template_profile: dict, template_valid: np.ndarray,
                                footprint: np.ndarray, labels: np.ndarray, rep: dict, frac: float, out_path,
                                tags: dict) -> dict:
    """Binary candidate on the template grid at the fixed policy; READY_FOR_MANUAL_UPLOAD (never uploaded)."""
    import rasterio

    from ..gems3.representation import apply
    from .persistence import atomic_write, sha256_file

    ev = np.asarray(template_valid, bool) & np.asarray(footprint, bool)
    elig = ev & ~np.asarray(labels, bool)
    q = apply(rep, np.nan_to_num(score_full, nan=-1.0), ev, float(frac), known=np.asarray(labels, bool),
              eligible=elig)
    prof = {**template_profile, "dtype": "float32", "count": 1, "nodata": None, "compress": "deflate"}

    def w(t):
        with rasterio.open(t, "w", **prof) as ds:
            ds.write(q.astype(np.float32), 1)
            ds.update_tags(ROLE="VALIDATED_SUBMISSION_CANDIDATE", STATUS="READY_FOR_MANUAL_UPLOAD",
                           **{k: str(v) for k, v in tags.items()})

    atomic_write(Path(out_path), w)
    return {"path": str(out_path), "sha256": sha256_file(out_path), "positive_px": int((q > 0).sum()),
            "area_frac_eligible": float((q > 0)[elig].mean()) if elig.any() else 0.0,
            "status": "READY_FOR_MANUAL_UPLOAD", "uploaded": False}
