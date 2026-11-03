"""Training / validation isolation for every G07 model.

* Outer folds = the G06 training visibilities (``g06.splits``): ``V1:d`` (withheld systems of split d hidden),
  ``V3:k`` (fold k + buffer removed), nested ``V1:j|k`` / ``V3:j|k`` (both outer and inner truth hidden), ``FULL``.
  A fit predicts only the units in its ``predicts`` list. Inner folds = spatial blocks of the training region.
* Pixel models (surface / subsurface) are visibility-isolated: ``STRICT_ISOLATED``.
* Candidate ranker: held-out unit (scenario s, zone z) <- candidates (real + synthetic) of the OTHER zones of s, rows
  near z's evaluation zone dropped. Inputs coming from supervised upstream models must be NESTED for training units
  (prediction of fit ``s:z|k``); W2 / CNN evidence is never nested -> ``SCREENING_ONLY``. With
  ``ranker.require_strict`` a missing nested input BLOCKS the model (``BLOCKED_UPSTREAM``) instead of degrading it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

SUPERVISED_NON_NESTED = {"w2", "cnn"}


def build_outer_folds(n_splits: int, n_folds: int, scenarios, nested: bool, nested_scenarios=("V1", "V3")) -> list:
    from ..g06.splits import fit_ids

    return fit_ids(n_splits, n_folds, scenarios, nested, nested_scenarios)


def build_inner_folds(region, visible, block_km: float, n_inner: int, seed: int) -> np.ndarray:
    from ..g06.splits import make_inner_folds

    return make_inner_folds(region, visible, block_km, n_inner, seed)


def validate_upstream_honesty(inputs: dict, nested_complete: bool) -> tuple[str, str]:
    """(strictness, reason) from {input name: provenance tag}."""
    used = {k for k, v in inputs.items() if v}
    if used & SUPERVISED_NON_NESTED:
        return "SCREENING_ONLY", f"inputs {sorted(used & SUPERVISED_NON_NESTED)} were trained with other units' " \
                                 "truth (W2 / CNN are not refitted per outer fold)"
    if not nested_complete:
        return "SCREENING_ONLY", "a supervised input lacked its nested prediction for a training unit"
    return "STRICT_NESTED", "label-free or nested inputs only"


# =============================================================================== pixel models (surface / subsurface)
def crossfit_pixel_experiment(exp_id: str, kind: str, cols: list[str], store, pool_X: pd.DataFrame,
                              pool_meta: pd.DataFrame, labels_dir, fit_list: list[str], units: dict, out_root,
                              params: dict, seed: int, fingerprint: str, weight_col: str = "weight",
                              target: str = "classification", region_mask=None, progress=print) -> list[dict]:
    """Fit ``exp_id`` on every fit and predict its units (``g06.models_pixel`` objects and paths).

    ``target``: ``classification`` (target column, ``weight_col``) or ``proximity`` (regression of the proximity
    target). ``region_mask`` (full grid) restricts training rows AND predictions (e.g. DEM coverage)."""
    from ..g06.contracts import ModelSpec
    from ..g06.datasets import unit_file
    from ..g06.models_pixel import PixelModel, fit_pixel_model, importance, oof_path, predict_pixel_raster
    from ..g06.persistence import atomic_write, write_json
    from ..g06.splits import parse_fit_id
    from ..g06.supervision import validate_label_visibility
    from .contracts import ModelArtifact
    from .models_subsurface import train_fault_distance_model

    validate_label_visibility("pixel", cols)
    out_root = Path(out_root)
    rows = []
    for fid in fit_list:
        t0 = time.perf_counter()
        mdir = out_root / "models" / exp_id / fingerprint[:10] / unit_file(fid)
        lp = Path(labels_dir) / f"{unit_file(fid)}.parquet"
        if not lp.exists():
            rows.append({"experiment": exp_id, "fit_id": fid, "status": "BLOCKED_INPUT", "detail": f"{lp.name} absent"})
            continue
        reused = (mdir / "model.joblib").exists()
        if reused:
            model = PixelModel.load(mdir)
        else:
            L = pd.read_parquet(lp)
            L = L[L[weight_col] > 0]
            J = L.merge(pool_meta[["row_id", "row", "col"]], on="row_id", how="left") \
                .merge(pool_X[["row_id"] + cols], on="row_id", how="left")
            if region_mask is not None:
                J = J[np.asarray(region_mask)[J.row.to_numpy(), J.col.to_numpy()]]
            X = J[cols].to_numpy(np.float32)
            w = J[weight_col].to_numpy(np.float32)
            inner, ib = J.inner_fold.to_numpy(np.int8), J.inner_buffer.to_numpy(bool)
            if target == "proximity":
                est, imp, info = train_fault_distance_model(X, J.proximity.to_numpy(np.float32), w, params, seed,
                                                            inner, ib)
            else:
                y = J.target.to_numpy(np.int8)
                if np.unique(y).size < 2:
                    rows.append({"experiment": exp_id, "fit_id": fid, "status": "FAILED", "detail": "single class"})
                    continue
                est, imp, info = fit_pixel_model(kind, X, y, w, params, seed, inner, ib)
            spec = ModelSpec(exp_id, fid, "A", kind, tuple(cols), store.schema_hash, dict(params),
                             "FULL_FIT_RESEARCH" if fid == "FULL" else "STRICT_ISOLATED", fingerprint, int(len(X)),
                             int((J.cls == 1).sum()))
            model = PixelModel(kind if target != "proximity" else "lightgbm_proximity", est, cols, store.schema_hash,
                               imp, spec, {**info, "n_train": int(len(X)), "target": target, "weight": weight_col,
                                           "seconds": round(time.perf_counter() - t0, 1)})
            model.save(mdir)
            importance(model).to_csv(mdir / "feature_importance.csv", index=False)
        preds = []
        if fid != "FULL":
            sc, j, k = parse_fit_id(fid)
            z = j if k is None else k
            targets = [f"V1:{z}"] if sc == "V1" else [x for x in (f"V3:{z}", f"V2:{z}") if x in units]
            for key in targets:
                vu = units.get(key)
                if vu is None:
                    continue
                pf = oof_path(out_root, exp_id, fid, key, fingerprint[:10])
                if pf.exists() and reused:
                    preds.append(str(pf.relative_to(out_root)))
                    continue
                win = (vu.sl[0].start, vu.sl[0].stop, vu.sl[1].start, vu.sl[1].stop)
                mask = vu.unit.ev & store.footprint()[vu.sl]
                if region_mask is not None:
                    mask &= np.asarray(region_mask)[vu.sl]
                p = predict_pixel_raster(model, store, win, mask)
                pf.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(pf, lambda t, p=p: np.save(t, p))
                art = ModelArtifact(exp_id, fid, model.kind, str(mdir.relative_to(out_root)), tuple(cols),
                                    store.schema_hash, fingerprint[:10], "STRICT_ISOLATED")
                write_json(pf.with_suffix(".json"), {"artifact": art.__dict__, "unit": key, "window": list(win),
                                                     "nested_outer": j if k is not None else None})
                preds.append(str(pf.relative_to(out_root)))
        rows.append({"experiment": exp_id, "fit_id": fid, "status": "REUSED" if reused else "COMPLETE",
                     "fingerprint": fingerprint[:10], "kind": model.kind, "n_features": len(cols),
                     "n_train": model.info.get("n_train"), "best_iteration": model.info.get("best_iteration"),
                     "target": target, "predictions": ";".join(preds), "model_dir": str(mdir.relative_to(out_root)),
                     "seconds": round(time.perf_counter() - t0, 1)})
        if progress:
            progress(f"  {exp_id} {fid}: {rows[-1]['status']} ({rows[-1]['seconds']} s)")
    return rows


def crossfit_surface_model(*a, **k):
    return crossfit_pixel_experiment(*a, **k)


def crossfit_subsurface_model(*a, **k):
    return crossfit_pixel_experiment(*a, **k)


# =============================================================================== candidate ranker
def crossfit_candidate_ranker(exp_id: str, D: pd.DataFrame, cols_fn, kind: str, params: dict, zone_masks: dict,
                              seed: int, unit_columns=None, inputs: dict | None = None, require_strict: bool = False,
                              progress=print) -> tuple[pd.DataFrame, dict]:
    """Held-out candidate scores of every unit. ``D`` = candidates (real + ``synthetic``) with ``target``,
    ``label_conf``, ``system_id``, ``seg_length_km``; ``cols_fn(frame) -> feature columns``;
    ``unit_columns(unit, outer)`` -> nested override columns (None = missing)."""
    from ..g06.models_segment import _override, _predict, train_segment_classifier, train_segment_ranker

    rows, info = [], {"units": {}, "nested_complete": True}
    real = D[~D.synthetic.astype(bool)] if "synthetic" in D else D
    for key in sorted(real.unit.unique()):
        sc, z = key.split(":")
        z = int(z)
        test = real[real.unit == key].copy()
        parts = []
        for k in sorted(u for u in D.unit.unique() if u.split(":")[0] == sc and int(u.split(":")[1]) != z):
            Tk = D[D.unit == k].copy()
            if unit_columns is not None:
                ov = unit_columns(k, z)
                if ov is None:
                    info["nested_complete"] = False
                    if require_strict:
                        info["units"][key] = {"status": "BLOCKED_UPSTREAM", "detail": f"nested inputs of {k}"}
                        parts = []
                        break
                else:
                    Tk = _override(Tk, ov)
            parts.append(Tk)
        if not parts:
            info["units"].setdefault(key, {"status": "FAILED", "detail": "no training unit"})
            continue
        if unit_columns is not None:
            ov = unit_columns(key, None)
            if ov is not None:
                test = _override(test, ov)
        TR = pd.concat(parts, ignore_index=True)
        inside = zone_masks[key][TR.r_full.to_numpy().astype(int), TR.c_full.to_numpy().astype(int)]
        TR = TR[~inside & TR.target.notna()]
        y = TR.target.to_numpy(int)
        if np.unique(y).size < 2:
            info["units"][key] = {"status": "FAILED", "detail": "single class"}
            continue
        cols = cols_fn(pd.concat([TR, test], ignore_index=True))
        w = system_balanced_weights(TR, y)
        X = TR[cols].to_numpy(np.float32)
        est = train_segment_ranker(X, y, TR.unit.astype("category").cat.codes.to_numpy(), params, seed) \
            if kind == "ranker" else train_segment_classifier(kind, X, y, w, params, seed)
        s = _predict(est, test[cols].to_numpy(np.float32))
        rows.append(pd.DataFrame({"unit": key, "label": test.label.to_numpy(), "score": s.astype(np.float32),
                                  "experiment": exp_id}))
        info["units"][key] = {"status": "COMPLETE", "n_train": int(len(TR)), "n_pos": int(y.sum()),
                              "n_synthetic": int(TR.get("synthetic", pd.Series(False, index=TR.index)).sum()),
                              "n_test": int(len(test)), "n_features": len(cols),
                              "importance": dict(zip(cols, np.asarray(getattr(est, "feature_importances_",
                                                                              np.zeros(len(cols))), float).tolist()))}
        if progress:
            progress(f"  {exp_id} {key}: {len(TR)} training candidates ({int(y.sum())} relevant)")
    st, why = validate_upstream_honesty(inputs or {}, info["nested_complete"])
    if require_strict and not info["nested_complete"]:
        st, why = "BLOCKED_UPSTREAM", "require_strict and a nested input is missing"
    info["strictness"], info["strictness_reason"] = st, why
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["unit", "label", "score",
                                                                               "experiment"])), info


def system_balanced_weights(T: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Positives: 1 / (relevant candidates of the same truth system) x confidence; negatives: length x confidence;
    both classes then balanced to equal totals (one long system cannot dominate)."""
    conf = np.maximum(T.label_conf.to_numpy(float), 0.1)
    w = np.maximum(T.seg_length_km.to_numpy(float), 0.05) * conf
    pos = y == 1
    if pos.any() and "system_id" in T:
        sid = T.system_id.to_numpy()[pos]
        n = pd.Series(sid).map(pd.Series(sid).value_counts()).to_numpy().astype(float)
        w[pos] = conf[pos] / n
    half = 0.5 * w.sum()                                       # totals fixed BEFORE rescaling: classes end equal
    for v in (0, 1):
        m = y == v
        if m.any():
            w[m] *= half / max(w[m].sum(), 1e-12)
    return w.astype(np.float32)
