"""Experiment matrix E0-E7 (Part 11). Each entry fixes the inputs, model, target and objective BEFORE any test result.

``feature_selector(store)`` returns the predictor names for an experiment (label-derived layers are never
returned - ``FeatureStore.predictors`` excludes them).
"""
from __future__ import annotations

EXPERIMENTS = {
    "E0": {"geophysics": "raw 100 m bands", "dem": "no", "model": "xgboost", "objective": "baseline",
           "families": ["raw"], "kind": "pixel"},
    "E1": {"geophysics": "G11 multiscale geophysics (+ G06 store when present)", "dem": "no", "model": "xgboost",
           "objective": "physical indices", "families": ["raw", "geo_magnetic", "geo_gravity", "geo_elevation",
                                                         "geo_cross", "g06"], "kind": "pixel"},
    "E2": {"geophysics": "as E1", "dem": "100 m summaries of the 1 m DEM", "model": "xgboost",
           "objective": "coarse DEM benefit", "families": ["raw", "geo_magnetic", "geo_gravity", "geo_elevation",
                                                           "geo_cross", "g06", "dem_100m", "multimodal"],
           "kind": "pixel", "requires_dem": True},
    "E3": {"geophysics": "top-k E1 channels", "dem": "no", "model": "segformer", "objective": "deep baseline",
           "kind": "patch", "requires_torch": True},
    "E4": {"geophysics": "top-k E1 channels", "dem": "native DEM patches", "model": "gated_fusion",
           "objective": "native DEM benefit", "kind": "multires", "requires_torch": True, "requires_dem": True},
    "E5": {"geophysics": "ensemble of E0-E4 OOF-valid maps", "dem": "no", "model": "fusion",
           "objective": "model complementarity", "kind": "fusion"},
    "E6": {"geophysics": "best pixel map", "dem": "segment cross-fault profiles", "model": "lightgbm ranker",
           "objective": "candidate ranking", "kind": "segment"},
    "E7": {"geophysics": "best pixel map", "dem": "segment DEM", "model": "topology + ranker",
           "objective": "full system", "kind": "segment", "topology": True},
}


def feature_selector(store, exp: str) -> list[str]:
    fam = EXPERIMENTS[exp].get("families")
    return store.predictors(fam) if fam else store.predictors()


def matrix_table():
    import pandas as pd

    return pd.DataFrame([{"experiment": k, **{c: v for c, v in d.items() if c != "families"}}
                         for k, d in EXPERIMENTS.items()])
