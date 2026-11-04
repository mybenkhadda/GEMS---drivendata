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
                              target: str = "classification", region_mask=None, progress=print,
                              strictness: str = "STRICT_ISOLATED", on_fit=None) -> list[dict]:
    """Fit ``exp_id`` on every fit and predict its units (``g06.models_pixel`` objects and paths). A failing fit gets
    a FAILED row and the others continue; ``validate_model_registry`` then decides what the experiment counts as.

    ``target``: ``classification`` (target column, ``weight_col``) or ``proximity`` (regression of the proximity
    target). ``region_mask`` (full grid) restricts training rows AND predictions (e.g. DEM coverage). ``strictness`` is
    recorded per fit (SCREENING_ONLY for exploratory DEM training); ``on_fit`` persists finished fits."""
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

    def one_fit(fid: str, t0: float) -> dict:
        mdir = out_root / "models" / exp_id / fingerprint[:10] / unit_file(fid)
        lp = Path(labels_dir) / f"{unit_file(fid)}.parquet"
        if not lp.exists():
            return {"experiment": exp_id, "fit_id": fid, "status": "BLOCKED_INPUT", "detail": f"{lp.name} absent"}
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
                    return {"experiment": exp_id, "fit_id": fid, "status": "FAILED", "detail": "single class"}
                est, imp, info = fit_pixel_model(kind, X, y, w, params, seed, inner, ib)
            spec = ModelSpec(exp_id, fid, "A", kind, tuple(cols), store.schema_hash, dict(params),
                             "FULL_FIT_RESEARCH" if fid == "FULL" else strictness, fingerprint, int(len(X)),
                             int((J.cls == 1).sum()))
            model = PixelModel(kind if target != "proximity" else "lightgbm_proximity", est, cols, store.schema_hash,
                               imp, spec, {**info, "n_train": int(len(X)), "target": target, "weight": weight_col,
                                           "seconds": round(time.perf_counter() - t0, 1)})
            model.save(mdir)
            importance(model).to_csv(mdir / "feature_importance.csv", index=False)
        preds = []
        if fid != "FULL":
            sc, j, k = parse_fit_id(fid)
            for key in fit_prediction_units(fid, units):
                vu = units[key]
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
                                    store.schema_hash, fingerprint[:10], strictness)
                write_json(pf.with_suffix(".json"), {"artifact": art.__dict__, "unit": key, "window": list(win),
                                                     "nested_outer": j if k is not None else None})
                preds.append(str(pf.relative_to(out_root)))
        return {"experiment": exp_id, "fit_id": fid, "status": "REUSED" if reused else "COMPLETE",
                "fingerprint": fingerprint[:10], "kind": model.kind, "device": "cpu", "strictness": strictness,
                "n_features": len(cols), "n_train": model.info.get("n_train"),
                "best_iteration": model.info.get("best_iteration"), "target": target, "predictions": ";".join(preds),
                "model_dir": str(mdir.relative_to(out_root)), "seconds": round(time.perf_counter() - t0, 1)}

    rows = []
    for fid in fit_list:
        t0 = time.perf_counter()
        try:
            rows.append(one_fit(fid, t0))
        except Exception as e:                       # MemoryError included: the registry records it, the next fit runs
            rows.append({"experiment": exp_id, "fit_id": fid, "status": "FAILED",
                         "detail": f"{type(e).__name__}: {e}"[:500], "seconds": round(time.perf_counter() - t0, 1)})
        if progress:
            progress(f"  {exp_id} {fid}: {rows[-1]['status']} ({rows[-1].get('seconds', 0)} s) "
                     f"{rows[-1].get('detail', '')}")
        if on_fit:
            on_fit()
    return rows


def crossfit_surface_model(*a, **k):
    return crossfit_pixel_experiment(*a, **k)


def crossfit_subsurface_model(*a, **k):
    return crossfit_pixel_experiment(*a, **k)


# =============================================================================== registry completeness
def fit_prediction_units(fit_id: str, units: dict) -> list[str]:
    """Units a fit must predict: ``V1:d`` -> V1:d, ``V3:k`` -> V3:k + V2:k, nested -> the inner unit, FULL -> none."""
    from ..g06.splits import parse_fit_id

    sc, j, k = parse_fit_id(fit_id)
    if sc == "FULL":
        return []
    z = j if k is None else k
    keys = [f"V1:{z}"] if sc == "V1" else [f"V3:{z}", f"V2:{z}"]
    return [x for x in keys if x in units]


def check_prediction(path, shape: tuple, ev: np.ndarray | None = None, probability: bool = True) -> str | None:
    """Why a held-out prediction array is unusable (None = valid): readable, unit shape, finite values, [0, 1]."""
    p = Path(path)
    if not p.exists():
        return f"{p.name} missing"
    try:
        a = np.load(p, mmap_mode="r")
    except (OSError, ValueError) as e:
        return f"{p.name} unreadable ({type(e).__name__})"
    if tuple(a.shape) != tuple(shape):
        return f"{p.name} shape {tuple(a.shape)} != unit {tuple(shape)}"
    v = np.asarray(a)
    inside = v[np.asarray(ev, bool)] if ev is not None else v
    if np.isinf(v).any():
        return f"{p.name} contains infinite values"
    if not np.isfinite(inside).any():
        return f"{p.name} has no finite value inside the evaluation zone"
    fin = v[np.isfinite(v)]
    if probability and (fin.min() < -1e-6 or fin.max() > 1 + 1e-6):
        return f"{p.name} leaves [0, 1]: [{fin.min():.4g}, {fin.max():.4g}]"
    return None


VALIDATION_COLUMNS = ["experiment", "fit_id", "registry_status", "valid", "problem", "fingerprint", "strictness", "device",
                      "n_predictions_expected", "n_predictions_valid"]


def validate_model_registry(root, registry: pd.DataFrame, expected: dict, units: dict, load_model=None,
                            schema_hash: str | None = None) -> pd.DataFrame:
    """One row per EXPECTED (experiment, fit): the registry says it finished, its model loads and matches the feature
    schema, and every prediction it owes exists with the unit's shape and valid values. ``load_model(model_dir)``
    returns an object with an optional ``schema_hash`` (default: ``g06.models_pixel.PixelModel.load``)."""
    from ..g06.models_pixel import PixelModel, oof_path

    root = Path(root)
    load_model = load_model or PixelModel.load
    rows = []
    reg = {(r.get("experiment"), r.get("fit_id")): r for r in registry.to_dict("records")} if len(registry) else {}
    for exp, fits in expected.items():
        for fid in fits:
            r = reg.get((exp, fid))
            row = dict.fromkeys(VALIDATION_COLUMNS)
            row.update(experiment=exp, fit_id=fid, valid=False, n_predictions_valid=0,
                       n_predictions_expected=len(fit_prediction_units(fid, units)))
            if r is None:
                row.update(registry_status="MISSING", problem="no registry row for this expected fit")
                rows.append(row)
                continue
            row.update(registry_status=r.get("status"), fingerprint=r.get("fingerprint"), device=r.get("device"),
                       strictness=r.get("strictness"))
            if r.get("status") not in ("COMPLETE", "REUSED"):
                row["problem"] = str(r.get("detail") or f"status {r.get('status')}")
                rows.append(row)
                continue
            problems = []
            try:
                m = load_model(root / str(r.get("model_dir")))
                sh = getattr(m, "schema_hash", None)
                if schema_hash is not None and sh is not None and sh != schema_hash:
                    problems.append(f"model schema {sh} != store schema {schema_hash}")
            except Exception as e:
                problems.append(f"model unreadable: {type(e).__name__}: {e}")
            for key in fit_prediction_units(fid, units):
                vu = units[key]
                why = check_prediction(oof_path(root, exp, fid, key, str(r.get("fingerprint"))), vu.unit.ev.shape,
                                       vu.unit.ev)
                if why:
                    problems.append(why)
                else:
                    row["n_predictions_valid"] += 1
            if not str(r.get("strictness") or "") or str(r.get("strictness")) == "nan":
                problems.append("strictness not recorded")
            row.update(valid=not problems, problem="; ".join(problems))
            rows.append(row)
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def summarize_experiments(validation: pd.DataFrame) -> pd.DataFrame:
    """Per experiment: COMPLETE (every expected fit valid), PARTIAL (some), BLOCKED (none, all blocked), FAILED (none)."""
    rows = []
    for exp, g in validation.groupby("experiment", sort=False):
        n, ok = len(g), int(g.valid.sum())
        blocked = bool(g.registry_status.isin(["BLOCKED_INPUT", "MISSING"]).all())
        status = "COMPLETE" if ok == n else ("PARTIAL" if ok else ("BLOCKED" if blocked else "FAILED"))
        rows.append({"experiment": exp, "status": status, "n_expected_fits": n, "n_valid_fits": ok,
                     "invalid_fits": ";".join(g.loc[~g.valid.astype(bool), "fit_id"])})
    return pd.DataFrame(rows, columns=["experiment", "status", "n_expected_fits", "n_valid_fits", "invalid_fits"])


# =============================================================================== candidate ranker
def crossfit_candidate_ranker(exp_id: str, D: pd.DataFrame, cols_fn, kind: str, params: dict, zone_masks: dict,
                              seed: int, unit_columns=None, inputs: dict | None = None, require_strict: bool = False,
                              progress=print) -> tuple[pd.DataFrame, dict]:
    """Held-out candidate scores of every unit. ``D`` = candidates (real + ``synthetic``) with ``target``,
    ``label_conf``, ``system_id``, ``seg_length_km``; ``cols_fn(frame) -> feature columns``;
    ``unit_columns(unit, outer)`` -> nested override columns (None = missing)."""
    from ..g06.models_segment import _override, _predict, train_segment_classifier, train_segment_ranker

    rows, info = [], {"units": {}, "nested_complete": True, "synthetic_isolation": "NONE"}
    syn_all = D.synthetic.astype(bool) if "synthetic" in D else pd.Series(False, index=D.index)
    if syn_all.any():
        if {"outer_fold", "training_visibility"} <= set(D.columns):
            info["synthetic_isolation"] = "PROVEN"
        else:                                        # geometry of unknown origin never trains anything
            D = D[~syn_all]
            info["synthetic_isolation"] = "UNPROVEN"
    real = D[~D.synthetic.astype(bool)] if "synthetic" in D else D
    for key in sorted(real.unit.unique()):
        sc, z = key.split(":")
        z = int(z)
        test = real[real.unit == key].copy()
        parts = []
        for k in sorted(u for u in D.unit.unique() if u.split(":")[0] == sc and int(u.split(":")[1]) != z):
            Tk = D[D.unit == k].copy()
            if info["synthetic_isolation"] == "PROVEN":      # only the rows generated for THIS outer fold
                Tk = Tk[~Tk.synthetic.astype(bool) | (Tk.outer_fold == z)].copy()
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
        assert_synthetic_isolation(TR, key)
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
    if info["synthetic_isolation"] == "UNPROVEN" and st != "BLOCKED_UPSTREAM":
        st, why = "SCREENING_ONLY", "synthetic candidates carried no fit provenance: they were dropped"
    info["strictness"], info["strictness_reason"] = st, why
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["unit", "label", "score",
                                                                               "experiment"])), info


def assert_synthetic_isolation(train: pd.DataFrame, heldout_key: str) -> None:
    """Every synthetic training row of held-out unit s:z was generated for the fit that hides z, from another unit."""
    from ..g06.contracts import LeakageError

    if "synthetic" not in train or "outer_fold" not in train:
        return
    S = train[train.synthetic.astype(bool)]
    if not len(S):
        return
    sc, z = heldout_key.split(":")
    fit = f"{'V1' if sc == 'V1' else 'V3'}:{z}"
    if (S.unit == heldout_key).any():
        raise LeakageError(f"{heldout_key}: synthetic candidates of the held-out unit reached its training set")
    wrong = S[(S.outer_fold != int(z)) | (S.training_visibility != fit)]
    if len(wrong):
        raise LeakageError(f"{heldout_key}: {len(wrong)} synthetic candidates were generated for another fit: "
                           f"{sorted(set(wrong.training_visibility))[:3]}")


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
