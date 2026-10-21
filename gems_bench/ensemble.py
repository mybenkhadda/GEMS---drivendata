"""Model complementarity and OOF-only ensembles.

All maps compared here are stitched OOF maps on the same geographic folds, so any
combination of them is itself a valid OOF prediction. Ensemble weights and the output
transform are cross-fitted across folds (chosen on folds != k, scored on fold k).
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import rasterio

from gems_eda.io import write_geotiff

from .data import NativeData
from .metric import (DTI_ALPHA, DTI_BETA, DTI_EPS, MaskedScorer, apply_output_transform,
                     cross_fitted_transform, transform_component_table)
from .persist import ExperimentRun


def read_map(path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.ma.filled(src.read(1, masked=True).astype(np.float32), np.nan)


def rank_normalise(p: np.ndarray, valid: np.ndarray, method: str = "ordinal") -> np.ndarray:
    """Global rank of ``p`` over ``valid`` scaled to [0, 1].

    ``ordinal`` (Phase-2 default, kept for reproducibility) breaks ties by raster order, i.e. by
    position — tied blocks are cut north-to-south by any threshold. ``average`` gives ties one rank.
    """
    out = np.full(p.shape, np.nan, np.float32)
    v = np.nan_to_num(p[valid], nan=0.0)
    if method == "ordinal":
        r = np.empty(len(v), np.float32)
        r[np.argsort(v, kind="stable")] = np.arange(len(v), dtype=np.float32)
    else:
        from scipy.stats import rankdata

        r = (rankdata(v, method=method) - 1).astype(np.float32)
    out[valid] = r / max(len(v) - 1, 1)
    return out


def pairwise_agreement(maps: dict[str, np.ndarray], valid: np.ndarray, labels: np.ndarray,
                       rng: np.random.Generator, n: int = 500_000, top_fracs=(0.01, 0.03, 0.10)) -> pd.DataFrame:
    """Pearson / Spearman on a pixel sample; Jaccard of each model's top-q% pixels (scale-free)."""
    from scipy.stats import spearmanr

    idx = rng.choice(np.flatnonzero(valid), min(n, int(valid.sum())), replace=False)
    names = list(maps)
    vals = {k: np.nan_to_num(maps[k].reshape(-1)[idx]) for k in names}
    tops = {}
    for k in names:
        v = np.nan_to_num(maps[k][valid])
        tops[k] = {q: v >= np.quantile(v, 1 - q) for q in top_fracs}
    lab = labels[valid]
    rows = []
    for a, b in itertools.combinations(names, 2):
        row = {"model_a": a, "model_b": b, "pearson": float(np.corrcoef(vals[a], vals[b])[0, 1]),
               "spearman": float(spearmanr(vals[a], vals[b]).correlation)}
        for q in top_fracs:
            ta, tb = tops[a][q], tops[b][q]
            row[f"jaccard_top{q:g}"] = float((ta & tb).sum() / max((ta | tb).sum(), 1))
            row[f"label_hit_only_a_top{q:g}"] = float((ta & ~tb & lab).sum() / max(lab.sum(), 1))
            row[f"label_hit_only_b_top{q:g}"] = float((~ta & tb & lab).sum() / max(lab.sum(), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def _simplex(n: int, step: float):
    k = int(round(1 / step))
    for c in itertools.product(range(k + 1), repeat=n):
        if sum(c) == k:
            yield np.array(c, float) / k


def weighted_ensemble_crossfit(maps: dict[str, np.ndarray], data: NativeData, fold_map: np.ndarray,
                               step: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """Weights on the simplex, chosen per fold on the other folds (pooled DTI after transform-free scaling)."""
    names = list(maps)
    folds = [int(f) for f in np.unique(fold_map[data.valid & (fold_map >= 0)])]
    scorers = {f: MaskedScorer(data.labels, data.valid & (fold_map == f)) for f in folds}
    stack = np.stack([np.nan_to_num(maps[k]) for k in names])
    rows = []
    for w in _simplex(len(names), step):
        e = np.tensordot(w, stack, axes=1)
        for f in folds:
            c = scorers[f].components(e)
            rows.append({"w": json.dumps(dict(zip(names, w.round(3).tolist()))), "fold": f,
                         "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn})
    tab = pd.DataFrame(rows)

    def dti(g):
        return g.tp_w.sum() / (g.tp_w.sum() + DTI_ALPHA * g.fp_w.sum() + DTI_BETA * g.fn_w.sum() + DTI_EPS)

    choice = []
    for f in folds:
        best = tab[tab.fold != f].groupby("w").apply(dti).idxmax()
        held = tab[(tab.fold == f) & (tab.w == best)]
        choice.append({"fold": f, "w": best, "dti_heldout": float(dti(held)),
                       "tp_w": held.tp_w.sum(), "fp_w": held.fp_w.sum(), "fn_w": held.fn_w.sum()})
    choice = pd.DataFrame(choice)
    final_w = json.loads(tab.groupby("w").apply(dti).idxmax())
    return choice, {"weights_all_folds": final_w, "dti_crossfit_pooled": float(dti(choice))}


def build_ensembles(runs: dict[str, ExperimentRun], data: NativeData, fold_map: np.ndarray, out_dir,
                    step: float = 0.1) -> pd.DataFrame:
    """Equal mean, rank-normalised mean and cross-fitted weighted mean of OOF maps; each gets a cross-fitted
    output transform; rasters + configs saved under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = {k: read_map(r.oof_path) for k, r in runs.items()}
    names = list(maps)
    cands = {
        "mean": np.nanmean(np.stack([maps[k] for k in names]), 0),
        "rank_mean": np.nanmean(np.stack([rank_normalise(maps[k], data.valid) for k in names]), 0),
    }
    choice, wfinal = weighted_ensemble_crossfit(maps, data, fold_map, step)
    choice.to_csv(out_dir / "weights_crossfit.csv", index=False)
    w = wfinal["weights_all_folds"]
    cands["weighted"] = np.sum([w[k] * np.nan_to_num(maps[k]) for k in names], 0)
    rows = []
    for name, e in cands.items():
        e = np.where(data.valid, e, np.nan).astype(np.float32)
        from .gbm import RANK_TRANSFORM_GRID

        tab = transform_component_table(e, data.labels, fold_map, data.valid,
                                        grid=RANK_TRANSFORM_GRID if name == "rank_mean" else None)
        per_fold, tf = cross_fitted_transform(tab)
        path = out_dir / f"ens_{name}.tif"
        write_geotiff(path, {"oof_probability": e}, data.grid)
        cfg = {"members": names, "kind": name, "weights": w if name == "weighted" else None,
               "output_transform": tf, "member_oof": {k: str(r.oof_path) for k, r in runs.items()},
               "note": "weighted: cross-fitted weights on OOF only; transform cross-fitted on folds"}
        (out_dir / f"ens_{name}.json").write_text(json.dumps(cfg, indent=2, default=float), encoding="utf-8")
        per_fold.to_csv(out_dir / f"ens_{name}_transform_crossfit.csv", index=False)
        rows.append({"ensemble": name, "dti_crossfit_tf": tf["dti_crossfit_pooled"],
                     "fold_min": per_fold.dti_heldout.min(), "fold_max": per_fold.dti_heldout.max(),
                     "path": str(path)})
    rows.append({"ensemble": "weighted(weights only, no transform)", "dti_crossfit_tf": wfinal["dti_crossfit_pooled"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ensemble_summary.csv", index=False)
    return df


def apply_transform_file(p: np.ndarray, tf: dict) -> np.ndarray:
    return apply_output_transform(p, tf["gamma"], tf["scale"], tf["floor"])
