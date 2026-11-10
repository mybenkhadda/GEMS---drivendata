"""Segment-level ranker (Model D): binary classification or learning-to-rank over candidate segments.

Groups for ranking = geographic group ids (candidates are ranked within their region). Training uses the train
split, early stopping the validation split; the test split is never touched here. LightGBM preferred,
XGBoost fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OBJECTIVES = ("binary", "lambdarank", "rank_xendcg")


@dataclass
class SegmentRanker:
    backend: str
    objective: str
    model: object
    feature_names: list
    history: dict = field(default_factory=dict)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].to_numpy(np.float32)
        if self.backend == "lightgbm":
            return self.model.predict(X, num_iteration=self.model.best_iteration or None)
        import xgboost as xgb

        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def save(self, out_dir) -> list[Path]:
        from ..registry import atomic_write_json

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / ("ranker.txt" if self.backend == "lightgbm" else "ranker.json")
        self.model.save_model(str(p))
        return [p, atomic_write_json(d / "feature_schema.json", {"features": self.feature_names,
                                                                 "objective": self.objective, "backend": self.backend}),
                atomic_write_json(d / "training_history.json", self.history)]


def _groups(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    d = df.sort_values("group", kind="stable")
    return d, d.groupby("group", sort=False).size().to_numpy()


def train_ranker(train: pd.DataFrame, val: pd.DataFrame, features: list, objective: str = "binary",
                 params: dict | None = None, seed: int = 2026) -> SegmentRanker:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective {objective} not in {OBJECTIVES}")
    p = dict(params or {})
    try:
        import lightgbm as lgb

        hp = {"objective": objective, "learning_rate": p.get("learning_rate", 0.03), "num_leaves": p.get("num_leaves", 31),
              "seed": seed, "verbose": -1, "min_data_in_leaf": 5,
              "metric": "average_precision" if objective == "binary" else "ndcg"}
        if objective != "binary":
            tr, gtr = _groups(train)
            va, gva = _groups(val)
            dtr = lgb.Dataset(tr[features].to_numpy(np.float32), tr.label.to_numpy(), group=gtr)
            dva = lgb.Dataset(va[features].to_numpy(np.float32), va.label.to_numpy(), group=gva, reference=dtr)
        else:
            dtr = lgb.Dataset(train[features].to_numpy(np.float32), train.label.to_numpy())
            dva = lgb.Dataset(val[features].to_numpy(np.float32), val.label.to_numpy(), reference=dtr)
        rec: dict = {}
        m = lgb.train(hp, dtr, int(p.get("n_estimators", 500)), valid_sets=[dva], valid_names=["validation"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.record_evaluation(rec)])
        return SegmentRanker("lightgbm", objective, m, list(features),
                             {k: {mm: [float(x) for x in v] for mm, v in d.items()} for k, d in rec.items()})
    except ImportError:
        import xgboost as xgb

        obj = {"binary": "binary:logistic", "lambdarank": "rank:pairwise", "rank_xendcg": "rank:ndcg"}[objective]
        tr, gtr = _groups(train)
        va, gva = _groups(val)
        dtr = xgb.DMatrix(tr[features].to_numpy(np.float32), label=tr.label, feature_names=features)
        dva = xgb.DMatrix(va[features].to_numpy(np.float32), label=va.label, feature_names=features)
        if objective != "binary":
            dtr.set_group(gtr)
            dva.set_group(gva)
        m = xgb.train({"objective": obj, "eta": p.get("learning_rate", 0.03), "seed": seed}, dtr,
                      int(p.get("n_estimators", 500)), evals=[(dva, "validation")], early_stopping_rounds=50,
                      verbose_eval=False)
        return SegmentRanker("xgboost", objective, m, list(features))
