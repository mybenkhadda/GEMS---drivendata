"""LightGBM pixel baselines with geographic CV, OOF maps, importance and SHAP.

Per fold k:
  1. sample training pixels from ``inner_train(k)`` and early-stopping pixels from
     ``inner_val(k)`` (both PU-aware, see sampling.py);
  2. train with early stopping on the inner fold only;
  3. optionally refit on all of ``train(k)`` with the chosen number of rounds;
  4. predict every pixel of ``val(k)`` -> stitched OOF map (each valid pixel is predicted
     exactly once, by the model that never saw its fold or buffer).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from gems_eda.io import write_geotiff

from .config import BenchConfig
from .data import NativeData
from .features import FeatureStore
from .folds import GeoFolds
from .metric import cross_fitted_transform, dti_components, full_diagnostics, transform_component_table
from .persist import ExperimentRun
from .sampling import sample_pixels, sample_summary


def _pu_map(store: FeatureStore):
    for name in ("eda:fused_score", "terrain:ridge@1km"):
        if store.has(name):
            return store.get(name)
    return None


def lgbm_safe_names(names) -> list[str]:
    """LightGBM rejects JSON special characters in feature names (':' '@' '.'): map to unique [A-Za-z0-9_]."""
    import re

    out: list[str] = []
    for n in names:
        base = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", str(n))).strip("_") or "feature"
        cand, i = base, 2
        while cand in out:
            cand, i = f"{base}_{i}", i + 1
        out.append(cand)
    return out


def train_lgbm(X, y, w, Xv, yv, wv, params: dict, rounds: int, early_stop: int | None, names):
    import lightgbm as lgb

    names = lgbm_safe_names(names)       # store.sample() keeps the original names; only LightGBM sees these
    dtr = lgb.Dataset(X, y, weight=w, feature_name=names, free_raw_data=True)
    hist: dict = {}
    callbacks = [lgb.record_evaluation(hist)]
    valid_sets, valid_names = [dtr], ["train"]
    if Xv is not None:
        dv = lgb.Dataset(Xv, yv, weight=wv, reference=dtr)
        valid_sets, valid_names = [dtr, dv], ["train", "inner_val"]
        if early_stop:
            callbacks.append(lgb.early_stopping(early_stop, verbose=False))
    p = {**params, "metric": ["binary_logloss", "auc"]}
    model = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=valid_sets, valid_names=valid_names,
                      callbacks=callbacks)
    rows = []
    for split, m in hist.items():
        for metric, vals in m.items():
            rows += [{"split": split, "metric": metric, "iteration": i + 1, "value": v} for i, v in enumerate(vals)]
    return model, pd.DataFrame(rows)


def predict_mask(model, store: FeatureStore, names: list[str], mask: np.ndarray, chunk_rows: int = 128) -> np.ndarray:
    """Predict every True pixel of ``mask``; NaN elsewhere."""
    out = np.full(mask.shape, np.nan, np.float32)
    rows = np.flatnonzero(mask.any(1))
    if not len(rows):
        return out
    for r0 in range(int(rows[0]), int(rows[-1]) + 1, chunk_rows):
        r1 = min(r0 + chunk_rows, mask.shape[0])
        m = mask[r0:r1]
        if not m.any():
            continue
        slab = store.rows(names, r0, r1)                     # (F, rows, W)
        X = slab[:, m].T
        out[r0:r1][m] = model.predict(X, num_threads=0).astype(np.float32)
    return out


def shap_summary(model, store: FeatureStore, names, mask, labels, rng, n: int) -> pd.DataFrame:
    """Mean |SHAP| (LightGBM pred_contrib) on a bounded, half-positive validation sample."""
    pos, neg = np.flatnonzero(mask & labels), np.flatnonzero(mask & ~labels)
    if not len(pos) or not len(neg):
        return pd.DataFrame()
    idx = np.r_[rng.choice(pos, min(len(pos), n // 2), replace=False), rng.choice(neg, min(len(neg), n // 2), replace=False)]
    X = store.sample(names, idx)
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    y = labels.reshape(-1)[idx]
    return pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(contrib).mean(0),
                         "mean_shap_pos": contrib[y].mean(0), "mean_shap_neg": contrib[~y].mean(0)})


def run_gbm_cv(exp_id: str, group: str, names: list[str], data: NativeData, store: FeatureStore,
               folds: GeoFolds, cfg: BenchConfig, *, refit_full: bool = True, overwrite: bool = False,
               sample_scale: float = 1.0) -> ExperimentRun:
    run = ExperimentRun(data.paths, exp_id, "gbm", {
        "model": "lightgbm", "group": group, "features": names, "seed": cfg.seed,
        "params": cfg.lgbm_params, "rounds": cfg.lgbm_rounds, "early_stop": cfg.lgbm_early_stop,
        "refit_full": refit_full, "folds": data.paths.rel(folds.directory) if folds.directory else None,
        "sampling": {k: getattr(cfg, k) for k in ("n_pos", "n_neg_uniform", "n_neg_ring", "ignore_px",
                                                  "ring_km", "pu_weight", "pu_quantile")},
        "sample_scale": sample_scale,
    })
    if run.done and not overwrite:
        print(f"[{exp_id}] already done -> {run.dir}")
        return run
    run.start()
    rng = np.random.default_rng(cfg.seed)
    labels, grid = data.labels, data.grid
    pu = _pu_map(store)
    oof = np.full(data.shape, np.nan, np.float32)
    hist_all, imp_all, shap_all, met_rows, samp_rows = [], [], [], [], []
    n = lambda v: int(v * sample_scale)
    params = {**cfg.lgbm_params, "seed": cfg.seed}

    for k in range(folds.n_folds):
        t0 = time.perf_counter()
        train_area = folds.train(k)
        visible = labels & train_area
        kw = dict(ignore_px=cfg.ignore_px, ring_km=cfg.ring_km, pu_map=pu, pu_quantile=cfg.pu_quantile,
                  pu_weight=cfg.pu_weight)
        s_tr = sample_pixels(folds.inner_train(k), visible, grid, rng, n_pos=n(cfg.n_pos),
                             n_neg_uniform=n(cfg.n_neg_uniform), n_neg_ring=n(cfg.n_neg_ring), **kw)
        s_iv = sample_pixels(folds.inner_val(k), visible, grid, rng, n_pos=n(cfg.n_pos // 4),
                             n_neg_uniform=n(cfg.n_neg_uniform // 4), n_neg_ring=n(cfg.n_neg_ring // 4), **kw)
        samp_rows += [sample_summary(s_tr, k, "inner_train"), sample_summary(s_iv, k, "inner_val")]
        X, Xv = store.sample(names, s_tr.idx.to_numpy()), store.sample(names, s_iv.idx.to_numpy())
        with run.timer(f"fold{k}_train"):
            model, hist = train_lgbm(X, s_tr.y.to_numpy(), s_tr.w.to_numpy(), Xv, s_iv.y.to_numpy(),
                                     s_iv.w.to_numpy(), params, cfg.lgbm_rounds, cfg.lgbm_early_stop, names)
        best_it = int(model.best_iteration or model.current_iteration())
        hist["fold"], hist["stage"] = k, "inner"
        hist_all.append(hist)
        if refit_full:
            s_full = sample_pixels(train_area, visible, grid, rng, n_pos=n(cfg.n_pos),
                                   n_neg_uniform=n(cfg.n_neg_uniform), n_neg_ring=n(cfg.n_neg_ring), **kw)
            samp_rows.append(sample_summary(s_full, k, "train_full"))
            Xf = store.sample(names, s_full.idx.to_numpy())
            with run.timer(f"fold{k}_refit"):
                model, h2 = train_lgbm(Xf, s_full.y.to_numpy(), s_full.w.to_numpy(), None, None, None,
                                       params, best_it, None, names)
            h2["fold"], h2["stage"] = k, "refit"
            hist_all.append(h2)
        model.save_model(str(run.models_dir / f"fold{k}.txt"))
        val = folds.val(k)
        with run.timer(f"fold{k}_predict"):
            oof[val] = predict_mask(model, store, names, val)[val]
        imp_all.append(pd.DataFrame({"feature": names, "fold": k,
                                     "gain": model.feature_importance("gain"),
                                     "split": model.feature_importance("split")}))
        sh = shap_summary(model, store, names, val & data.valid, labels, rng, cfg.shap_rows)
        if len(sh):
            sh["fold"] = k
            shap_all.append(sh)
        diag = full_diagnostics(oof, labels, val & data.valid, rng, cfg.recall_tolerances_m, cfg.report_threshold)
        met_rows.append({"fold": k, "best_iteration": best_it, "seconds": round(time.perf_counter() - t0, 1), **diag})
        print(f"[{exp_id}] fold {k}: DTI={diag['dti']:.4f}  AUC={diag.get('roc_auc', np.nan):.3f}  it={best_it}")

    write_geotiff(run.oof_path, {"oof_probability": oof}, grid)
    run.register("oof", run.oof_path, "raster")
    return _finish_cv(run, oof, data, folds, cfg, hist_all, imp_all, shap_all, met_rows, samp_rows, rng)


def _finish_cv(run, oof, data, folds, cfg, hist_all, imp_all, shap_all, met_rows, samp_rows, rng,
               transform_grid: dict | None = None) -> ExperimentRun:
    """Shared tail for GBM / CNN CV runs: tables, pooled metrics, cross-fitted output transform."""
    if hist_all:
        run.save_table(pd.concat(hist_all, ignore_index=True), "history")
    if imp_all:
        run.save_table(pd.concat(imp_all, ignore_index=True), "importance")
    if shap_all:
        run.save_table(pd.concat(shap_all, ignore_index=True), "shap_summary")
    if samp_rows:
        run.save_table(pd.concat(samp_rows, ignore_index=True), "sampling_summary")
    met = pd.DataFrame(met_rows)
    pooled = full_diagnostics(oof, data.labels, data.valid, rng, cfg.recall_tolerances_m, cfg.report_threshold)
    tab = transform_component_table(oof, data.labels, folds.fold_map, data.valid, grid=transform_grid)
    run.save_table(tab, "transform_grid")
    per_fold, final_tf = cross_fitted_transform(tab)
    run.save_table(per_fold, "transform_crossfit")
    (run.dir / "output_transform.json").write_text(json.dumps(final_tf, indent=2, default=float), encoding="utf-8")
    run.register("output_transform", run.dir / "output_transform.json", "config")
    met = pd.concat([met, pd.DataFrame([{"fold": "pooled", **pooled}])], ignore_index=True)
    run.save_table(met, "metrics_fold")
    folds_only = met[met.fold != "pooled"]
    summary = {
        "dti_pooled": pooled["dti"],
        "dti_pooled_crossfit_tf": final_tf["dti_crossfit_pooled"],
        "dti_fold_mean": float(folds_only.dti.mean()), "dti_fold_std": float(folds_only.dti.std(ddof=0)),
        "dti_fold_min": float(folds_only.dti.min()),
        "roc_auc_pooled": pooled.get("roc_auc"), "ap_pooled": pooled.get("avg_precision"),
        "recall300_pooled": pooled.get("recall@300m"),
        "n_features": len(run.config.get("features", run.config.get("channels", []))),
        "train_seconds": float(sum(v for k, v in run.state["timings_s"].items() if "train" in k or "refit" in k)),
        "group": run.config.get("group", run.config.get("input_config")),
    }
    run.finish(summary)
    print(f"[{run.exp_id}] pooled DTI raw={summary['dti_pooled']:.4f}  cross-fitted transform="
          f"{summary['dti_pooled_crossfit_tf']:.4f}  folds {summary['dti_fold_mean']:.4f}±{summary['dti_fold_std']:.4f}")
    return run


def fit_full_gbm(exp_id: str, names: list[str], data: NativeData, store: FeatureStore, cfg: BenchConfig,
                 train_area: np.ndarray, rounds: int, *, ignore_mask: np.ndarray | None = None,
                 predict_area: np.ndarray | None = None, overwrite: bool = False, extra_config: dict | None = None):
    """Single model on ``train_area`` with fixed rounds (pseudo-discovery / final refit). Writes ``pred.tif``."""
    run = ExperimentRun(data.paths, exp_id, "gbm", {"model": "lightgbm", "features": names, "seed": cfg.seed,
                                                    "params": cfg.lgbm_params, "rounds": rounds,
                                                    **(extra_config or {})})
    pred_path = run.dir / "pred.tif"
    if run.state.get("status") == "done" and pred_path.exists() and not overwrite:
        return run
    run.start()
    rng = np.random.default_rng(cfg.seed)
    visible = data.labels & train_area & (~ignore_mask if ignore_mask is not None else True)
    s = sample_pixels(train_area, visible, data.grid, rng, n_pos=cfg.n_pos, n_neg_uniform=cfg.n_neg_uniform,
                      n_neg_ring=cfg.n_neg_ring, ignore_px=cfg.ignore_px, ring_km=cfg.ring_km,
                      ignore_mask=ignore_mask, pu_map=_pu_map(store), pu_quantile=cfg.pu_quantile,
                      pu_weight=cfg.pu_weight)
    run.save_table(sample_summary(s, "full", "train"), "sampling_summary")
    X = store.sample(names, s.idx.to_numpy())
    with run.timer("train"):
        model, hist = train_lgbm(X, s.y.to_numpy(), s.w.to_numpy(), None, None, None,
                                 {**cfg.lgbm_params, "seed": cfg.seed}, rounds, None, names)
    run.save_table(hist, "history")
    model.save_model(str(run.models_dir / "model.txt"))
    area = data.valid if predict_area is None else predict_area
    with run.timer("predict"):
        pred = predict_mask(model, store, names, area)
    write_geotiff(pred_path, {"probability": pred}, data.grid)
    run.register("pred", pred_path, "raster")
    run.save_table(pd.DataFrame({"feature": names, "gain": model.feature_importance("gain"),
                                 "split": model.feature_importance("split")}), "importance")
    run.state.update(status="done")
    run._flush()
    return run


def run_map_baseline(exp_id: str, score: np.ndarray, data: NativeData, folds: GeoFolds, cfg: BenchConfig,
                     description: str, overwrite: bool = False) -> ExperimentRun:
    """Zero-training baseline: a label-free score map, rank-normalised to [0, 1].

    Only the output transform is fitted (cross-fitted on folds), so this is the floor that
    every learned model must beat under the same protocol.
    """
    from .ensemble import rank_normalise

    run = ExperimentRun(data.paths, exp_id, "baseline", {"model": "unsupervised_map", "description": description,
                                                         "seed": cfg.seed, "features": []})
    if run.done and not overwrite:
        return run
    run.start()
    rng = np.random.default_rng(cfg.seed)
    oof = rank_normalise(np.asarray(score, np.float32), data.valid)
    write_geotiff(run.oof_path, {"oof_probability": oof}, data.grid)
    run.register("oof", run.oof_path, "raster")
    met = []
    for k in range(folds.n_folds):
        val = folds.val(k) & data.valid
        met.append({"fold": k, **full_diagnostics(oof, data.labels, val, rng, cfg.recall_tolerances_m,
                                                  cfg.report_threshold)})
    return _finish_cv(run, oof, data, folds, cfg, [], [], [], met, [], rng, transform_grid=RANK_TRANSFORM_GRID)


RANK_TRANSFORM_GRID = {"gamma": (1.0, 4.0, 16.0), "scale": (0.5, 1.0, 2.0), "floor": (0.8, 0.9, 0.95, 0.98, 0.99)}


def median_best_iteration(run: ExperimentRun) -> int:
    m = run.table("metrics_fold")
    return int(pd.to_numeric(m.best_iteration, errors="coerce").median())


def dti_of_map(pred: np.ndarray, data: NativeData) -> float:
    return dti_components(pred, data.labels, eval_mask=data.valid).dti
