"""Full-footprint reconstruction, parity and the gated submission writer (nothing is uploaded).

* Surface / subsurface FULL-fit models are predicted tile by tile over their footprint
  (``g06.inference.predict_full_raster``: exact feature order, schema hash, research tags).
* The W2 base is the frozen P6 full-region stacker score (``stack/maps/full_score.tif``), ranked inside the footprint
  - the same quantity as the held-out unit maps used in validation; it is never refitted here.
* Final construction = the validated policy on the full grid: exact-k on the competition budget domain
  (``gems3.submission.eligible_mask``: footprint & template & not a known fault), W2-preserving portfolio of
  candidates (or the plain W2 exact-k reference), optional confidence corridor; values in [0, 1].
* Parity: (1) every model's store re-read == its training rows and the written raster == the model on those rows,
  (2) the final selection has exactly k eligible pixels and nothing on known faults, (3) the written file read back
  equals the finalised array, (4) W2 full-region score vs held-out unit maps (rank agreement - a diagnostic: the two
  differ by construction, full fit vs held-out fit).
* ``write_submission_candidate`` = ``gems3.submission.finalise`` + ``gems_bench.submission.write_submission`` +
  ``gems3.submission.validate``; refused unless every gate passed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = "RESEARCH_NOT_A_SUBMISSION"


def predict_surface_full_region(model, store, out_path, tile_px: int = 512, tags=None, progress=print) -> dict:
    from ..g06.inference import predict_full_raster

    return predict_full_raster(model, store, out_path, tile_px, {"SYSTEM": "A_surface", **(tags or {})}, progress)


def predict_subsurface_full_region(model, store, out_path, tile_px: int = 512, tags=None, progress=print) -> dict:
    from ..g06.inference import predict_full_raster

    return predict_full_raster(model, store, out_path, tile_px, {"SYSTEM": "B_subsurface", **(tags or {})}, progress)


def read_w2_full(path, valid: np.ndarray) -> np.ndarray:
    from ..g06.raster_io import read_band_full
    from ..gems3.representation import rank01

    a = read_band_full(path, 1)
    return np.where(valid, rank01(np.nan_to_num(a, nan=0.0), np.asarray(valid, bool)), np.nan).astype(np.float32)


def predict_candidate_inventory_full(store, scores: dict, eligible: np.ndarray, labels: np.ndarray,
                                     valid: np.ndarray, cand_cfg: dict, known_cfg: dict, est, cols: list,
                                     seed: int, calibrator=None) -> tuple[np.ndarray, pd.DataFrame]:
    """Full-region pools -> merged candidates -> features (every mapped fault is known context) -> scores."""
    from ..g06.datasets import _means, segment_features
    from ..g06.feature_context import permitted_fault_context
    from .candidate_generation import deduplicate_candidates, merge_candidate_inventory
    from .models_candidates import score_candidate_inventory
    from .representation import exact_k_mask

    H, W = np.shape(eligible)
    pools = {}
    for k, s in scores.items():
        pc = (cand_cfg.get("pools") or {}).get(k)
        if pc and s is not None:
            pools[k], _ = exact_k_mask(s, eligible & np.isfinite(s), float(pc["frac"]), width_full=W, seed=seed)
    lab = merge_candidate_inventory(pools, cand_cfg)
    kctx = permitted_fault_context(labels, valid, known_cfg)
    from ..gems3.representation import rank01

    ev = {f"{k}_rank": rank01(np.nan_to_num(s, nan=0.0), np.asarray(valid, bool)) for k, s in scores.items()
          if s is not None and k in ("w2", "cnn", "surface", "subsurface")}
    seg_cfg = {**cand_cfg, "geology_features": [f for f in cand_cfg.get("geology_features", []) if f in store.names]}
    T = segment_features(lab, store, (0, H, 0, W), seg_cfg, kctx, ev)
    if len(T):
        n = int(lab.max())
        rr, cc = np.nonzero(lab)
        agree = np.zeros(n)
        for k, m in pools.items():
            T[f"on_{k}"], _ = _means(lab[rr, cc], m[rr, cc].astype(np.float64), n)
            agree += np.nan_to_num(T[f"on_{k}"].to_numpy()) >= 0.5
        T["seg_n_experts"] = agree
        T["score"] = score_candidate_inventory(est, T, cols)
        T["p"] = calibrator(T.score.to_numpy()) if calibrator is not None else T.score.rank(pct=True)
        T = T.merge(deduplicate_candidates(lab, pools)[["label", "dominant_source", "n_sources"]], on="label")
    return lab, T


def construct_final_prediction(w2: np.ndarray, eligible: np.ndarray, frac: float, policy: str, lab=None,
                               utility: pd.DataFrame | None = None, base_share: float = 0.85, seed: int = 2026
                               ) -> tuple[np.ndarray, dict]:
    """``policy``: ``w2_exact`` (G07-A0) or ``portfolio`` (G07-D2). Returns (float32 q in {0, 1}, report)."""
    from .representation import candidate_portfolio_selection, exact_k_mask

    W = np.shape(eligible)[1]
    if policy == "w2_exact" or lab is None or utility is None:
        m, rep = exact_k_mask(w2, eligible, frac, width_full=W, seed=seed)
    else:
        m, rep = candidate_portfolio_selection(w2, eligible, frac, lab, utility, base_share, width_full=W, seed=seed)
    return m.astype(np.float32), rep


def verify_export_parity(q: np.ndarray, eligible: np.ndarray, frac: float, labels: np.ndarray, written=None,
                         data=None, model_parity: dict | None = None, w2_agreement: dict | None = None) -> dict:
    from .representation import budget_k

    k = budget_k(frac, int(np.asarray(eligible, bool).sum()))
    sel = np.asarray(q) > 0
    checks = {"exact_budget": int((sel & eligible).sum()) == k and not (sel & ~np.asarray(eligible, bool)).any(),
              "nothing_on_known_faults": not (sel & np.asarray(labels, bool)).any(),
              "values_in_unit_interval": bool(np.nanmin(q) >= 0 and np.nanmax(q) <= 1)}
    if model_parity:
        checks["model_feature_parity"] = all(v.get("status") == "PASS" for v in model_parity.values())
    if written is not None and data is not None:
        from ..g06.raster_io import read_band_full
        from ..gems3.submission import finalise

        back = read_band_full(written, 1)
        ref = finalise(q, data)
        checks["file_readback_equal"] = bool(np.array_equal(np.isnan(back), np.isnan(ref)) and
                                             np.allclose(back[np.isfinite(back)], ref[np.isfinite(ref)]))
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "k": k,
            "selected_eligible_px": int((sel & eligible).sum()), "w2_full_vs_heldout": w2_agreement or {}}


def write_submission_candidate(q: np.ndarray, data, out_path, manifest: dict, gates: dict) -> dict:
    """Validated GeoTIFF on the template grid; refused when any gate is not True. Never uploaded."""
    from ..artifacts.atomic import atomic_save
    from ..gems3.submission import finalise, validate
    from ..submission import write_submission
    from .persistence import sha256_file, write_json

    failed = [k for k, v in gates.items() if v is not True]
    if failed:
        return {"status": "REFUSED", "failed_gates": failed, "uploaded": False}
    arr = finalise(q, data)
    out_path = Path(out_path)
    if out_path.exists():
        out_path = out_path.with_name(out_path.stem + "__" + time.strftime("%Y%m%dT%H%M%S") + ".tif")
    atomic_save(out_path, lambda t: write_submission(arr, data, Path(t)))
    chk = validate(out_path, data)
    res = {"status": "READY_FOR_MANUAL_UPLOAD", "path": str(out_path), "sha256": sha256_file(out_path),
           "validation": chk, "uploaded": False, **manifest}
    write_json(out_path.with_suffix(".json"), res)
    return res
