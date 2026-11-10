"""Gradient-boosted pixel baseline (E0-E2) with PU-style weighting.

Unlabeled pixels are not confirmed negatives: they enter with weight ``pu_unlabeled_weight`` (< 1). Early stopping
uses the VALIDATION split only. XGBoost is preferred; LightGBM is the fallback when xgboost is not installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PixelModel:
    backend: str
    booster: object
    feature_names: list
    params: dict
    history: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "xgboost":
            import xgboost as xgb

            return self.booster.predict(xgb.DMatrix(X, feature_names=self.feature_names),
                                        iteration_range=(0, self.booster.best_iteration + 1)
                                        if getattr(self.booster, "best_iteration", None) is not None else (0, 0))
        return self.booster.predict(X, num_iteration=self.booster.best_iteration or None)

    def importance(self) -> pd.DataFrame:
        if self.backend == "xgboost":
            imp = self.booster.get_score(importance_type="gain")
        else:
            imp = dict(zip(self.feature_names, self.booster.feature_importance("gain")))
        return pd.DataFrame({"feature": list(imp), "gain": list(imp.values())}).sort_values("gain", ascending=False)

    def save(self, out_dir) -> list[Path]:
        from ..registry import atomic_write_json

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / ("model.json" if self.backend == "xgboost" else "model.txt")
        tmp = p.with_name(p.stem + ".partial" + p.suffix)          # keep the extension (xgboost picks the format)
        self.booster.save_model(str(tmp))
        tmp.replace(p)
        return [p, atomic_write_json(d / "feature_schema.json", {"features": self.feature_names, "backend": self.backend,
                                                                 "params": self.params}),
                atomic_write_json(d / "training_history.json", self.history)]


def train_pixel_model(train, val, params: dict, seed: int = 2026, rounds: int | None = None) -> PixelModel:
    """``train`` / ``val``: GeoPixelDataset. Returns the model with the validation-selected iteration."""
    p = dict(params)
    n = int(rounds or p.pop("n_estimators", 600))
    p.pop("pu_unlabeled_weight", None)
    try:
        import xgboost as xgb

        dtr = xgb.DMatrix(train.X, label=train.y, weight=train.weight, feature_names=train.feature_names)
        dva = xgb.DMatrix(val.X, label=val.y, weight=val.weight, feature_names=train.feature_names)
        hp = {"objective": "binary:logistic", "eval_metric": ["aucpr", "logloss"], "seed": seed,
              "max_depth": p.get("max_depth", 6), "eta": p.get("learning_rate", 0.05),
              "subsample": p.get("subsample", 0.8), "colsample_bytree": p.get("colsample_bytree", 0.8),
              "tree_method": p.get("tree_method", "hist")}
        hist: dict = {}
        bst = xgb.train(hp, dtr, n, evals=[(dtr, "train"), (dva, "validation")], early_stopping_rounds=50,
                        evals_result=hist, verbose_eval=False)
        return PixelModel("xgboost", bst, train.feature_names, hp, {k: {m: list(v) for m, v in d.items()}
                                                                   for k, d in hist.items()})
    except ImportError:
        import lightgbm as lgb

        hp = {"objective": "binary", "metric": ["average_precision", "binary_logloss"], "seed": seed, "verbose": -1,
              "learning_rate": p.get("learning_rate", 0.05), "num_leaves": 2 ** p.get("max_depth", 6) - 1,
              "bagging_fraction": p.get("subsample", 0.8), "bagging_freq": 1,
              "feature_fraction": p.get("colsample_bytree", 0.8)}
        dtr = lgb.Dataset(train.X, train.y, weight=train.weight, feature_name=train.feature_names)
        dva = lgb.Dataset(val.X, val.y, weight=val.weight, reference=dtr)
        rec: dict = {}
        bst = lgb.train(hp, dtr, n, valid_sets=[dtr, dva], valid_names=["train", "validation"],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.record_evaluation(rec)])
        return PixelModel("lightgbm", bst, train.feature_names, hp, json.loads(json.dumps(rec, default=float)))


def predict_full_grid(model: PixelModel, store, valid: np.ndarray, chunk_rows: int = 256) -> np.ndarray:
    """Predict every valid pixel in row chunks (memory-bounded); NaN outside ``valid``."""
    H, W = valid.shape
    out = np.full((H, W), np.nan, np.float32)
    mms = [store.get(n) for n in model.feature_names]
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        m = valid[r0:r1]
        if not m.any():
            continue
        X = np.stack([np.asarray(a[r0:r1], np.float32)[m] for a in mms], 1)
        out[r0:r1][m] = model.predict(X).astype(np.float32)
    return out
