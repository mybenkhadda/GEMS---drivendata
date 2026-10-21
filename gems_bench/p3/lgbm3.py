"""LightGBM early-stopping diagnostics (3C) and G3 modality ablation (3D).

Same folds, sampler, seed and scoring for every run; results go to outputs/phase3/experiments/.
Scores: raw pooled DTI, ROC-AUC, and the *cross-fitted binary-area* DTI (curves.py) — the
scale-free number that does not depend on an output-transform grid.

Diagnosis already visible in the Phase-2 histories (``history_diagnostics``): inner-validation
AUC is ~0.67 at the first tree and never improves, while train AUC climbs to ~0.95 — the extra
trees learn location-specific structure that does not transfer to another region. Early stopping
is then *correct*; the question is whether stronger regularisation, different negatives or a
different stopping metric produce rankings that transfer better.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from gems_eda.io import write_geotiff

from ..gbm import lgbm_safe_names, predict_mask
from ..metric import full_diagnostics
from ..persist import ExperimentRun
from ..sampling import sample_pixels, sample_summary
from .context import P3Context
from .curves import crossfit_select, zone_curves

COMPLEXITY = {
    "C1": {"num_leaves": 63, "min_child_samples": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
           "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255},                       # Phase-2 setting
    "C2": {"num_leaves": 15, "min_child_samples": 2000, "feature_fraction": 0.5, "bagging_fraction": 0.5,
           "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63, "min_gain_to_split": 0.0},  # conservative
}
SAMPLING = {
    "S1": {"pos": 1.0, "ring": 1.0, "uniform": 1.0, "note": "Phase-2: positives + 0.5-2.5 km ring + uniform"},
    "S2": {"pos": 0.2, "ring": 0.0, "uniform": 0.4, "note": "thinned positives, uniform negatives only (no ring)"},
}
LGBM_GRID = {
    "D0_p2_replica":      {"lr": 0.05, "complexity": "C1", "sampling": "S1", "es": "any"},
    "D1_es_auc":          {"lr": 0.05, "complexity": "C1", "sampling": "S1", "es": "auc"},
    "D2_conservative":    {"lr": 0.05, "complexity": "C2", "sampling": "S1", "es": "auc"},
    "D3_conservative_lr": {"lr": 0.01, "complexity": "C2", "sampling": "S1", "es": "auc"},
    "D4_complex_lr":      {"lr": 0.01, "complexity": "C1", "sampling": "S1", "es": "auc"},
    "D5_conservative_S2": {"lr": 0.05, "complexity": "C2", "sampling": "S2", "es": "auc"},
    "D6_cons_lr_S2":      {"lr": 0.01, "complexity": "C2", "sampling": "S2", "es": "auc"},
    "F1_fixed100_C2":     {"lr": 0.05, "complexity": "C2", "sampling": "S1", "es": None, "rounds": 100},
    "F2_fixed400_C2":     {"lr": 0.05, "complexity": "C2", "sampling": "S1", "es": None, "rounds": 400},
    "F3_fixed400_C1":     {"lr": 0.05, "complexity": "C1", "sampling": "S1", "es": None, "rounds": 400},
}


# ---------------------------------------------------------------------------
# 1. diagnostics of the existing Phase-2 histories (no training)
# ---------------------------------------------------------------------------
def history_diagnostics(ctx: P3Context, names=("G1", "G2", "G3", "G4")) -> pd.DataFrame:
    rows = []
    for n in names:
        run = ctx.phase2_run(n)
        if not (run.dir / "history.csv").exists():
            continue
        h = pd.read_csv(run.dir / "history.csv")
        samp = pd.read_csv(run.dir / "sampling_summary.csv") if (run.dir / "sampling_summary.csv").exists() else None
        for f, g in h[h.stage == "inner"].groupby("fold"):
            s = lambda split, metric: g[(g.split == split) & (g.metric == metric)].set_index("iteration").value
            vl, va, ta = s("inner_val", "binary_logloss"), s("inner_val", "auc"), s("train", "auc")
            it_ll, it_auc = int(vl.idxmin()), int(va.idxmax())
            row = {"model": n, "fold": int(f), "iterations_recorded": int(vl.index.max()),
                   "best_iter_logloss": it_ll, "best_iter_auc": it_auc,
                   "inner_auc_iter1": float(va.iloc[0]), "inner_auc_best": float(va.max()),
                   "inner_auc_gain_over_iter1": float(va.max() - va.iloc[0]),
                   "train_auc_at_best_ll": float(ta.loc[it_ll]), "train_auc_last": float(ta.iloc[-1]),
                   "generalisation_gap_at_best": float(ta.loc[it_ll] - va.loc[it_ll]),
                   "inner_ll_iter1": float(vl.iloc[0]), "inner_ll_min": float(vl.min()), "inner_ll_last": float(vl.iloc[-1]),
                   "refit_rounds": int(h[(h.stage == "refit") & (h.fold == f) & (h.metric == "auc")].iteration.max())
                   if (h.stage == "refit").any() else np.nan}
            if samp is not None:
                sf = samp[(samp.fold.astype(str) == str(f)) & (samp.split == "inner_train")]
                pos, neg = sf[sf.kind == "pos"].n.sum(), sf[sf.kind != "pos"].n.sum()
                row.update(n_pos=int(pos), n_neg=int(neg), ring_share_of_neg=float(sf[sf.kind == "ring"].n.sum() / max(neg, 1)),
                           pu_downweighted=int(sf.pu.sum()))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. one configurable CV run
# ---------------------------------------------------------------------------
def variant_params(v: dict, seed: int, base: dict) -> dict:
    p = {**base, **COMPLEXITY[v["complexity"]], "learning_rate": v["lr"], "seed": seed, "verbose": -1,
         "objective": "binary", "deterministic": True, "force_row_wise": True}
    return p


def train3(X, y, w, Xv, yv, wv, params: dict, rounds: int, es: str | None, es_rounds: int, names):
    import lightgbm as lgb

    names = lgbm_safe_names(names)
    dtr = lgb.Dataset(X, y, weight=w, feature_name=names, free_raw_data=True)
    hist: dict = {}
    cbs = [lgb.record_evaluation(hist)]
    sets, snames = [dtr], ["train"]
    metrics = ["auc", "binary_logloss"] if es == "auc" else ["binary_logloss", "auc"]
    if Xv is not None:
        sets, snames = [dtr, lgb.Dataset(Xv, yv, weight=wv, reference=dtr)], ["train", "inner_val"]
        if es:
            cbs.append(lgb.early_stopping(es_rounds, first_metric_only=(es != "any"), verbose=False))
    model = lgb.train({**params, "metric": metrics}, dtr, num_boost_round=rounds, valid_sets=sets,
                      valid_names=snames, callbacks=cbs)
    rows = [{"split": s, "metric": m, "iteration": i + 1, "value": v}
            for s, md in hist.items() for m, vals in md.items() for i, v in enumerate(vals)]
    return model, pd.DataFrame(rows)


def _sample(ctx, area, visible, rng, v: dict, ignore_mask=None):
    from ..gbm import _pu_map

    cfg, sc = ctx.cfg, SAMPLING[v["sampling"]]
    return sample_pixels(area, visible, ctx.data.grid, rng, n_pos=int(cfg.n_pos * sc["pos"]),
                         n_neg_uniform=int(cfg.n_neg_uniform * sc["uniform"]), n_neg_ring=int(cfg.n_neg_ring * sc["ring"]),
                         ignore_px=cfg.ignore_px, ring_km=cfg.ring_km, ignore_mask=ignore_mask, pu_map=_pu_map(ctx.store),
                         pu_quantile=cfg.pu_quantile, pu_weight=cfg.pu_weight)


def run_variant_cv(ctx: P3Context, exp_id: str, names: list[str], v: dict, *, names_per_fold: dict | None = None,
                   overwrite: bool = False, tag: str = "") -> ExperimentRun:
    """Geographic CV of one LightGBM variant. ``names_per_fold`` overrides the feature list per fold (A5)."""
    cfg = ctx.cfg
    params = variant_params(v, cfg.seed, cfg.lgbm_params)
    rounds = int(v.get("rounds", cfg.lgbm_rounds))
    es_rounds = max(100, int(round(5.0 / v["lr"])))
    run = ExperimentRun(ctx.paths, exp_id, "gbm", {
        "model": "lightgbm", "variant": v, "params": params, "rounds": rounds, "es_rounds": es_rounds, "tag": tag,
        "features": names, "features_per_fold": names_per_fold, "seed": cfg.seed,
        "folds": ctx.bench.rel(ctx.folds.directory), "sampling": SAMPLING[v["sampling"]]})
    if run.done and not overwrite:
        return run
    run.start()
    rng = np.random.default_rng(cfg.seed)
    lab, data = ctx.data.labels, ctx.data
    oof = np.full(data.shape, np.nan, np.float32)
    hist_all, imp_all, met, samp = [], [], [], []
    for k in ctx.fold_ids:
        t0 = time.perf_counter()
        nm = names_per_fold[k] if names_per_fold else names
        visible = lab & ctx.folds.train(k)
        s_tr = _sample(ctx, ctx.folds.inner_train(k), visible, rng, v)
        s_iv = _sample(ctx, ctx.folds.inner_val(k), visible, rng, v)
        samp += [sample_summary(s_tr, k, "inner_train"), sample_summary(s_iv, k, "inner_val")]
        X, Xv = ctx.store.sample(nm, s_tr.idx.to_numpy()), ctx.store.sample(nm, s_iv.idx.to_numpy())
        with run.timer(f"fold{k}_train"):
            model, h = train3(X, s_tr.y.to_numpy(), s_tr.w.to_numpy(), Xv, s_iv.y.to_numpy(), s_iv.w.to_numpy(),
                              params, rounds, v.get("es"), es_rounds, nm)
        best_it = int(model.best_iteration or model.current_iteration()) if v.get("es") else rounds
        h["fold"], h["stage"] = k, "inner"
        hist_all.append(h)
        iv = h[h.split == "inner_val"]
        bi = {}                                    # where each inner-val metric peaks, whatever stopped training
        for m, g in iv.groupby("metric"):
            vals = g.set_index("iteration").value
            bi[f"best_iter_{m}"] = int(vals.idxmin() if m == "binary_logloss" else vals.idxmax())
            bi[f"inner_{m}_best"] = float(vals.min() if m == "binary_logloss" else vals.max())
            bi[f"inner_{m}_iter1"] = float(vals.iloc[0])
        s_full = _sample(ctx, ctx.folds.train(k), visible, rng, v)
        samp.append(sample_summary(s_full, k, "train_full"))
        with run.timer(f"fold{k}_refit"):
            model, h2 = train3(ctx.store.sample(nm, s_full.idx.to_numpy()), s_full.y.to_numpy(), s_full.w.to_numpy(),
                               None, None, None, params, best_it, None, es_rounds, nm)
        h2["fold"], h2["stage"] = k, "refit"
        hist_all.append(h2)
        model.save_model(str(run.models_dir / f"fold{k}.txt"))
        val = ctx.folds.val(k)
        with run.timer(f"fold{k}_predict"):
            oof[val] = predict_mask(model, ctx.store, nm, val)[val]
        imp_all.append(pd.DataFrame({"feature": nm, "fold": k, "gain": model.feature_importance("gain"),
                                     "split": model.feature_importance("split")}))
        diag = full_diagnostics(oof, lab, val & data.valid, rng, cfg.recall_tolerances_m, cfg.report_threshold)
        met.append({"fold": k, "best_iteration": best_it, **bi, "seconds": round(time.perf_counter() - t0, 1), **diag})
        print(f"[{exp_id}] fold {k}: it={best_it} raw DTI={diag['dti']:.4f} AUC={diag.get('roc_auc', np.nan):.3f}")
    write_geotiff(run.oof_path, {"oof_probability": oof}, data.grid)
    run.register("oof", run.oof_path, "raster")
    return finish_variant(ctx, run, oof, hist_all, imp_all, met, samp)


def prediction_quantiles(p: np.ndarray, zones: np.ndarray, valid: np.ndarray, labels: np.ndarray,
                         qs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)) -> pd.DataFrame:
    rows = []
    for z in np.unique(zones[valid & (zones >= 0)]):
        for part, m in (("all", valid & (zones == z)), ("on_label", valid & (zones == z) & labels)):
            v = p[m]
            v = v[np.isfinite(v)]
            if len(v):
                rows.append({"zone": int(z), "part": part, "mean": float(v.mean()),
                             **{f"q{q:g}": float(x) for q, x in zip(qs, np.quantile(v, qs))}})
    return pd.DataFrame(rows)


def finish_variant(ctx: P3Context, run: ExperimentRun, oof, hist_all, imp_all, met, samp) -> ExperimentRun:
    """Shared tail for GBM and CNN Phase-3 runs (empty lists are skipped)."""
    data = ctx.data
    for parts, name in ((hist_all, "history"), (imp_all, "importance"), (samp, "sampling_summary")):
        if parts:
            run.save_table(pd.concat(parts, ignore_index=True), name)
    run.save_table(prediction_quantiles(oof, ctx.folds.fold_map, data.valid, data.labels), "prediction_quantiles")
    curves = zone_curves(oof, data.labels, ctx.folds.fold_map, data.valid)
    run.save_table(curves, "area_curves")
    per, s = crossfit_select(curves, ["target_frac"])
    run.save_table(per, "area_crossfit")
    m = pd.DataFrame(met)
    pooled = full_diagnostics(oof, data.labels, data.valid, np.random.default_rng(ctx.cfg.seed),
                              ctx.cfg.recall_tolerances_m, ctx.cfg.report_threshold)
    m = pd.concat([m, pd.DataFrame([{"fold": "pooled", **pooled}])], ignore_index=True)
    run.save_table(m, "metrics_fold")
    folds_only = m[m.fold != "pooled"]
    summary = {"dti_pooled_raw": pooled["dti"], "dti_crossfit_area": s["dti_crossfit"],
               "dti_area_insample": s["dti_insample"], "area_fold_min": s["fold_min"],
               "frac_all_folds": s["params_all"]["target_frac"], "roc_auc_pooled": pooled.get("roc_auc"),
               "best_iter_median": float(pd.to_numeric(folds_only.best_iteration).median()),
               "best_iters": folds_only.best_iteration.tolist(),
               "train_seconds": float(sum(v for k, v in run.state["timings_s"].items() if "train" in k or "refit" in k)),
               "peak_rss_mb": max(run.state.get("rss_mb", {}).values(), default=None)}
    run.finish(summary)
    print(f"[{run.exp_id}] cross-fitted area DTI={s['dti_crossfit']:.4f}  raw={pooled['dti']:.4f}")
    return run


def fit_full_variant(ctx: P3Context, exp_id: str, names: list[str], v: dict, rounds: int, *,
                     ignore_mask: np.ndarray, predict_area: np.ndarray, overwrite: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Pseudo-discovery model: whole map, withheld systems + ring weight 0 (never negatives). Returns (pred, gain)."""
    from ..ensemble import read_map

    run = ExperimentRun(ctx.paths, exp_id, "gbm", {"variant": v, "features": names, "rounds": rounds, "seed": ctx.cfg.seed})
    pred_path = run.dir / "pred.tif"
    if run.state.get("status") == "done" and pred_path.exists() and not overwrite:
        return read_map(pred_path), run.table("importance")
    run.start()
    rng = np.random.default_rng(ctx.cfg.seed)
    visible = ctx.data.labels & ~ignore_mask
    s = _sample(ctx, ctx.data.valid, visible, rng, v, ignore_mask=ignore_mask)
    run.save_table(sample_summary(s, "full", "train"), "sampling_summary")
    params = variant_params(v, ctx.cfg.seed, ctx.cfg.lgbm_params)
    with run.timer("train"):
        model, h = train3(ctx.store.sample(names, s.idx.to_numpy()), s.y.to_numpy(), s.w.to_numpy(), None, None, None,
                          params, rounds, None, 100, names)
    run.save_table(h, "history")
    model.save_model(str(run.models_dir / "model.txt"))
    with run.timer("predict"):
        pred = predict_mask(model, ctx.store, names, predict_area)
    write_geotiff(pred_path, {"probability": pred}, ctx.data.grid)
    run.register("pred", pred_path, "raster")
    imp = pd.DataFrame({"feature": names, "gain": model.feature_importance("gain")})
    run.save_table(imp, "importance")
    run.state.update(status="done")
    run._flush()
    return pred, imp


# ---------------------------------------------------------------------------
# 3. modality ablation
# ---------------------------------------------------------------------------
ABLATIONS = {
    "A0_full": (),
    "A1_no_terrain": ("terrain",),
    "A2_no_magnetic": ("magnetic",),
    "A3_no_gravity_basement": ("gravity", "basement"),
    "A4_no_seismicity_strain": ("seismicity", "strain"),
    "A5_top_features": "top",
}


def ablation_features(ctx: P3Context, base_names: list[str], drop) -> list[str]:
    return [n for n in base_names if ctx.store.specs[n].modality not in drop]


def top_features_per_fold(run: ExperimentRun, k_top: int = 15) -> dict[int, list[str]]:
    """Top-k features by gain of each fold model (trained without that fold: label-safe for the fold)."""
    imp = run.table("importance")
    return {int(f): g.sort_values("gain", ascending=False).feature.head(k_top).tolist() for f, g in imp.groupby("fold")}


def top_features_from_gain(gain: pd.DataFrame, k_top: int = 15) -> list[str]:
    return gain.sort_values("gain", ascending=False).feature.head(k_top).tolist()


def jaccard_top(a: np.ndarray, b: np.ndarray, valid: np.ndarray, zones: np.ndarray, frac: float = 0.05) -> float:
    from .budget import binary_mask

    ma, mb = binary_mask(a, valid, zones, frac), binary_mask(b, valid, zones, frac)
    return float((ma & mb).sum() / max((ma | mb).sum(), 1))
