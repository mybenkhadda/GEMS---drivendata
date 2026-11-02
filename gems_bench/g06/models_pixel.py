"""Classical pixel classifiers (Track A: geology only) - Random Forest, LightGBM, XGBoost.

Each experiment = (feature families, model kind). It is cross-fitted over training visibilities (``splits``): the
model of fit ``V1:d`` sees only the labels visible to split d and predicts only unit ``V1:d``; ``V3:k`` predicts
``V3:k`` and ``V2:k``; nested fits ``V1:j|k`` / ``V3:j|k`` produce the inner predictions that strict meta / segment
models of outer j train on. Fixed, compact, regularised hyper-parameters from ``benchmarks.yaml`` (no search).
Early stopping uses the inner spatial folds of the TRAINING region only (never the outer unit), then the model is
refitted on every training row with the selected number of trees.

Saved per fit (``models/<kind>/<experiment>/<fp10>/<fit>/``): ``model.joblib`` (estimator + imputation medians +
feature order + schema hash), ``model.json`` (spec, training summary, importance). Predictions per unit:
``predictions/oof/<experiment>/<fit>__<unit>.npy`` (crop, float32, NaN outside the evaluation zone) + ``.json``.
Missing values: LightGBM / XGBoost handle NaN natively; Random Forest gets training-row medians (stored with the
model, applied identically at inference: train / inference parity).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ModelSpec, validate_model_input, validate_probability
from .supervision import validate_label_visibility


class PixelModel:
    """Fitted estimator + its exact input contract."""

    def __init__(self, kind: str, est, feature_columns: list[str], schema_hash: str, impute: np.ndarray | None,
                 spec: ModelSpec | None = None, info: dict | None = None):
        self.kind, self.est = kind, est
        self.feature_columns = list(feature_columns)
        self.schema_hash = schema_hash
        self.impute = impute
        self.spec, self.info = spec, info or {}

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        if self.impute is not None:
            bad = ~np.isfinite(X)
            if bad.any():
                X = np.where(bad, self.impute[None, :], X)
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.feature_columns):
            raise ValueError(f"{X.shape[1]} columns != {len(self.feature_columns)} fitted features")
        if not len(X):
            return np.zeros(0, np.float32)
        p = self.est.predict_proba(self._prep(X))[:, 1].astype(np.float32)
        return p

    def save(self, d) -> Path:
        import joblib

        from .persistence import atomic_write, write_json

        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write(d / "model.joblib", lambda t: joblib.dump({"kind": self.kind, "est": self.est,
                                                                "features": self.feature_columns,
                                                                "schema_hash": self.schema_hash,
                                                                "impute": self.impute}, t))
        write_json(d / "model.json", {"kind": self.kind, "features": self.feature_columns,
                                      "schema_hash": self.schema_hash, "spec": asdict(self.spec) if self.spec else {},
                                      "info": self.info})
        return d

    @classmethod
    def load(cls, d) -> "PixelModel":
        import joblib

        z = joblib.load(Path(d) / "model.joblib")
        meta = json.loads((Path(d) / "model.json").read_text(encoding="utf-8"))
        spec = ModelSpec(**meta["spec"]) if meta.get("spec") else None
        return cls(z["kind"], z["est"], z["features"], z["schema_hash"], z["impute"], spec, meta.get("info"))


# =============================================================================== features
def feature_columns_for(store, families, track: str = "A", exclude=()) -> list[str]:
    """Usable store channels of ``families`` permitted for ``track`` (store order; QC copies never included)."""
    cols = [c for c in store.by_family(list(families), track) if not c.startswith("phys_") and c not in exclude]
    validate_label_visibility("pixel", cols)
    return cols


# =============================================================================== estimators
def train_random_forest(X, y, w, params: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    p = dict(params)
    est = RandomForestClassifier(n_estimators=int(p.get("n_estimators", 200)), max_depth=p.get("max_depth", 14),
                                 min_samples_leaf=int(p.get("min_samples_leaf", 20)),
                                 max_features=p.get("max_features", "sqrt"), max_samples=p.get("max_samples"),
                                 n_jobs=int(p.get("n_jobs", -1)), random_state=int(seed), class_weight=None)
    est.fit(X, y, sample_weight=w)
    return est, {}


def _lgbm(params: dict, seed: int, n_estimators: int | None = None):
    import lightgbm as lgb

    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    if n_estimators is not None:
        p["n_estimators"] = int(n_estimators)
    return lgb.LGBMClassifier(**p, random_state=int(seed), verbose=-1)


def train_lightgbm(X, y, w, params: dict, seed: int, X_val=None, y_val=None, w_val=None):
    """Early stopping on the inner validation rows (when given), then a refit on all rows with the best size."""
    import lightgbm as lgb

    info = {}
    es = int(params.get("early_stopping_rounds", 0) or 0)
    if es and X_val is not None and len(X_val) and np.unique(y_val).size == 2:
        m = _lgbm(params, seed)
        m.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
              eval_metric="average_precision", callbacks=[lgb.early_stopping(es, verbose=False)])
        best = int(m.best_iteration_ or params.get("n_estimators", 500))
        info = {"best_iteration": best, "inner_val_rows": int(len(X_val))}
        Xa, ya, wa = np.concatenate([X, X_val]), np.concatenate([y, y_val]), np.concatenate([w, w_val])
        est = _lgbm(params, seed, max(best, 10))
        est.fit(Xa, ya, sample_weight=wa)
        return est, info
    est = _lgbm(params, seed)
    est.fit(X, y, sample_weight=w)
    return est, info


def train_xgboost(X, y, w, params: dict, seed: int, X_val=None, y_val=None, w_val=None):
    import xgboost as xgb

    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    es = int(params.get("early_stopping_rounds", 0) or 0)
    info = {}
    if es and X_val is not None and len(X_val) and np.unique(y_val).size == 2:
        m = xgb.XGBClassifier(**p, random_state=int(seed), early_stopping_rounds=es, eval_metric="aucpr")
        m.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=False)
        best = int(getattr(m, "best_iteration", p.get("n_estimators", 400)) or p.get("n_estimators", 400)) + 1
        info = {"best_iteration": best, "inner_val_rows": int(len(X_val))}
        est = xgb.XGBClassifier(**{**p, "n_estimators": max(best, 10)}, random_state=int(seed))
        est.fit(np.concatenate([X, X_val]), np.concatenate([y, y_val]),
                sample_weight=np.concatenate([w, w_val]), verbose=False)
        return est, info
    est = xgb.XGBClassifier(**p, random_state=int(seed))
    est.fit(X, y, sample_weight=w, verbose=False)
    return est, info


TRAINERS = {"random_forest": train_random_forest, "lightgbm": train_lightgbm, "xgboost": train_xgboost}


def fit_pixel_model(kind: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, params: dict, seed: int,
                    inner: np.ndarray | None = None, inner_buffer: np.ndarray | None = None,
                    val_fold: int = 0) -> tuple[object, np.ndarray | None, dict]:
    """(estimator, imputation medians or None, info). ``inner``: inner-fold id per row (early stopping)."""
    impute = None
    if kind == "random_forest":
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        impute = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        X = np.where(np.isfinite(X), X, impute[None, :])
        est, info = train_random_forest(X, y, w, params, seed)
        return est, impute, info
    if inner is not None and kind in ("lightgbm", "xgboost") and (inner == val_fold).any():
        ib = inner_buffer if inner_buffer is not None else np.zeros(len(inner), bool)
        va = (inner == val_fold) & ~ib
        tr = (inner != val_fold) & ~ib
        est, info = TRAINERS[kind](X[tr], y[tr], w[tr], params, seed, X[va], y[va], w[va])
        rest = ~(tr | va)
        info["buffer_rows_excluded_from_early_stopping"] = int(rest.sum())
        return est, None, info
    est, info = TRAINERS[kind](X, y, w, params, seed)
    return est, None, info


def importance(model: PixelModel) -> pd.DataFrame:
    est = model.est
    if hasattr(est, "booster_"):
        g = est.booster_.feature_importance(importance_type="gain")
    elif hasattr(est, "feature_importances_"):
        g = np.asarray(est.feature_importances_, float)
    else:
        g = np.zeros(len(model.feature_columns))
    T = pd.DataFrame({"feature": model.feature_columns, "importance": g})
    T["share"] = T.importance / max(float(T.importance.sum()), 1e-12)
    return T.sort_values("importance", ascending=False).reset_index(drop=True)


# =============================================================================== prediction
def predict_pixel_raster(model: PixelModel, store, window: tuple, mask: np.ndarray, tile_px: int = 512,
                         chunk: int = 1_000_000) -> np.ndarray:
    """Probabilities over ``window`` (r0, r1, c0, c1) inside ``mask`` (crop bool); NaN elsewhere. Features are read
    tile by tile in the model's exact column order (schema hash checked)."""
    validate_model_input(model.feature_columns, [c for c in model.feature_columns if c in store.names],
                         model.schema_hash, store.schema_hash, f"{model.kind} prediction")
    r0, r1, c0, c1 = (int(x) for x in window)
    out = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
    for a in range(r0, r1, tile_px):
        for b in range(c0, c1, tile_px):
            ra, rb, ca, cb = a, min(a + tile_px, r1), b, min(b + tile_px, c1)
            m = np.asarray(mask, bool)[ra - r0:rb - r0, ca - c0:cb - c0]
            if not m.any():
                continue
            A = store.load(model.feature_columns, (ra, rb, ca, cb))
            X = A[:, m].T
            p = np.concatenate([model.predict_proba(X[i:i + chunk]) for i in range(0, len(X), chunk)])
            blk = out[ra - r0:rb - r0, ca - c0:cb - c0]
            blk[m] = p
    validate_probability(out, f"{model.kind} raster")
    return out


# =============================================================================== cross-fitting
def training_matrix(pool_X: pd.DataFrame, labels: pd.DataFrame, cols: list[str]) -> tuple:
    """(X, y, w, inner, inner_buffer, row_ids) for one fit; rows with weight 0 dropped."""
    L = labels[labels.weight > 0]
    J = L.merge(pool_X[["row_id"] + cols], on="row_id", how="left", validate="one_to_one")
    validate_label_visibility("pixel", cols, ["target", "cls", "weight"])
    X = J[cols].to_numpy(np.float32)
    return (X, J.target.to_numpy(np.int8), J.weight.to_numpy(np.float32), J.inner_fold.to_numpy(np.int8),
            J.inner_buffer.to_numpy(bool), J.row_id.to_numpy())


def crossfit_pixel_model(exp_id: str, exp_cfg: dict, model_params: dict, store, ds_root, fit_list: list[str],
                         units: dict, out_root, seed: int, fingerprint: str, predict: bool = True,
                         progress=print) -> list[dict]:
    """Fit ``exp_id`` on every fit of ``fit_list`` and predict the units each fit may predict.
    Returns one registry row per fit (status, sizes, model dir, prediction files)."""
    from .datasets import load_dataset, unit_file

    kind = exp_cfg["model"]
    cols = feature_columns_for(store, exp_cfg["families"], "A", tuple(exp_cfg.get("exclude", ())))
    if not cols:
        raise ValueError(f"{exp_id}: no usable feature for families {exp_cfg['families']}")
    pool = load_dataset(ds_root, "pixel", columns=cols, expected_schema_hash=store.schema_hash)
    out_root = Path(out_root)
    rows = []
    for fid in fit_list:
        t0 = time.perf_counter()
        mdir = out_root / "models" / kind / exp_id / fingerprint[:10] / unit_file(fid)
        lab_path = Path(ds_root) / "datasets" / "pixels" / "labels" / f"{unit_file(fid)}.parquet"
        if not lab_path.exists():
            rows.append({"experiment": exp_id, "fit_id": fid, "status": "BLOCKED_MISSING_INPUT",
                         "detail": f"labels {lab_path.name} absent"})
            continue
        if (mdir / "model.joblib").exists():
            model = PixelModel.load(mdir)
            reused = True
        else:
            labels = pd.read_parquet(lab_path)
            X, y, w, inner, ib, _ = training_matrix(pool["X"], labels, cols)
            if np.unique(y).size < 2:
                rows.append({"experiment": exp_id, "fit_id": fid, "status": "FAILED",
                             "detail": "a single class in the training rows"})
                continue
            est, impute, info = fit_pixel_model(kind, X, y, w, model_params, seed, inner, ib)
            spec = ModelSpec(exp_id, fid, "A", kind, tuple(cols), store.schema_hash, dict(model_params),
                             "FULL_FIT_RESEARCH" if fid == "FULL" else "STRICT_ISOLATED", fingerprint,
                             int(len(y)), int(y.sum()))
            model = PixelModel(kind, est, cols, store.schema_hash, impute, spec,
                               {**info, "n_train": int(len(y)), "n_pos": int(y.sum()),
                                "seconds": round(time.perf_counter() - t0, 1)})
            model.save(mdir)
            importance(model).to_csv(mdir / "feature_importance.csv", index=False)
            reused = False
        preds = []
        if predict and fid != "FULL":
            from .splits import parse_fit_id

            sc, j, k = parse_fit_id(fid)
            targets = [f"V1:{j if k is None else k}"] if sc == "V1" else \
                [x for x in (f"V3:{j if k is None else k}", f"V2:{j if k is None else k}") if x in units]
            for key in targets:
                vu = units.get(key)
                if vu is None:
                    continue
                pf = oof_path(out_root, exp_id, fid, key, fingerprint[:10])
                pdir = pf.parent
                if pf.exists() and reused:
                    preds.append(str(pf.relative_to(out_root)))
                    continue
                win = (vu.sl[0].start, vu.sl[0].stop, vu.sl[1].start, vu.sl[1].stop)
                mask = vu.unit.ev & store.footprint()[vu.sl]
                p = predict_pixel_raster(model, store, win, mask)
                pdir.mkdir(parents=True, exist_ok=True)
                from .persistence import atomic_write, write_json

                atomic_write(pf, lambda t, p=p: np.save(t, p))
                write_json(pf.with_suffix(".json"), {"experiment": exp_id, "fit_id": fid, "unit": key,
                                                     "window": list(win), "schema_hash": store.schema_hash,
                                                     "model_dir": str(mdir.relative_to(out_root)),
                                                     "fingerprint": fingerprint, "nested_outer": j if k is not None
                                                     else None, "provenance": "visibility-isolated fit"})
                preds.append(str(pf.relative_to(out_root)))
        rows.append({"experiment": exp_id, "fit_id": fid, "status": "REUSED" if reused else "COMPLETE",
                     "fingerprint": fingerprint[:10], "kind": kind, "families": ",".join(exp_cfg["families"]),
                     "model_dir": str(mdir.relative_to(out_root)), "n_features": len(cols),
                     "n_train": model.info.get("n_train"), "n_pos": model.info.get("n_pos"),
                     "best_iteration": model.info.get("best_iteration"), "predictions": ";".join(preds),
                     "seconds": round(time.perf_counter() - t0, 1)})
        progress and progress(f"  {exp_id} {fid}: {rows[-1]['status']} ({rows[-1]['seconds']} s)")
    return rows


def oof_path(out_root, exp_id: str, fit_id: str, unit_key: str, fp10: str) -> Path:
    """``predictions/oof/<experiment>/<fp10>/[nested/]<fit>__<unit>.npy`` (fingerprint-keyed: never overwritten)."""
    from .datasets import unit_file
    from .splits import parse_fit_id

    _, _, k = parse_fit_id(fit_id)
    d = Path(out_root) / "predictions" / "oof" / exp_id / fp10
    if k is not None:
        d = d / "nested"
    return d / f"{unit_file(fit_id)}__{unit_file(unit_key)}.npy"


def heldout_fit_for(unit_key: str) -> str:
    """Primary fit that is allowed to predict a unit: V1:d -> V1:d, V3:k / V2:k -> V3:k."""
    sc, z = unit_key.split(":")
    return f"V1:{z}" if sc == "V1" else f"V3:{z}"


def load_oof(out_root, exp_id: str, unit_key: str, fp10: str, outer: int | None = None) -> np.ndarray | None:
    """Held-out prediction of ``unit_key`` (primary fit), or the nested inner prediction for outer ``outer``."""
    sc, z = unit_key.split(":")
    base = "V1" if sc == "V1" else "V3"
    fid = heldout_fit_for(unit_key) if outer is None else f"{base}:{outer}|{z}"
    p = oof_path(out_root, exp_id, fid, unit_key, fp10)
    return np.load(p) if p.exists() else None


def current_fingerprints(registry: pd.DataFrame) -> dict:
    """{experiment: fp10} of the most recent COMPLETE / REUSED fits in a pixel registry."""
    if registry is None or not len(registry) or "fingerprint" not in registry:
        return {}
    R = registry[registry.status.isin(["COMPLETE", "REUSED"])]
    return {e: str(g.fingerprint.iloc[-1]) for e, g in R.groupby("experiment", sort=False)}
