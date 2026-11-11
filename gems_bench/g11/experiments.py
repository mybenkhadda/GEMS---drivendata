"""Experiment matrix (fixed BEFORE any test result). Numbering follows the recovery-first brief:

========  ===================================================  ============  ===========================
id        inputs                                               model         kind
========  ===================================================  ============  ===========================
B0        raw 100 m GeoDAWN bands only (reference baseline)    xgboost       pixel (optional)
E0        100 m geophysics: raw + multiscale + G06 indices     xgboost       pixel
E1        E0 + validated DEM 100 m summaries + geo x DEM       xgboost       pixel (requires DEM)
E2        top-k E0 channels                                    segformer     patch  (+ E2__<loss>, E2__finetune)
E3        top-k E0 channels + DEM patches + modality masks     gated fusion  multires (+ E3__<mode>)
E4        candidate segments: geophysics, DEM morphology,      lightgbm      segment ranker (INFER stage)
          topology
E5        pixel ensemble of the G11 maps (fit stack, select    fusion        fusion
          validation)
========  ===================================================  ============  ===========================

``feature_selector(store, exp)`` returns predictor names; label-derived layers are never returned.
"""
from __future__ import annotations

GEO_FAMILIES = ["raw", "geo_magnetic", "geo_gravity", "geo_elevation", "geo_cross", "g06"]

EXPERIMENTS = {
    "B0": {"geophysics": "raw 100 m bands", "dem": "no", "model": "xgboost", "objective": "reference baseline",
           "families": ["raw"], "kind": "pixel"},
    "E0": {"geophysics": "100 m geophysics + G06 indices", "dem": "no", "model": "xgboost",
           "objective": "geophysical baseline", "families": GEO_FAMILIES, "kind": "pixel"},
    "E1": {"geophysics": "as E0", "dem": "validated DEM 100 m summaries + geo x DEM", "model": "xgboost",
           "objective": "DEM benefit", "families": GEO_FAMILIES + ["dem_100m", "multimodal"], "kind": "pixel",
           "requires_dem": True},
    "E2": {"geophysics": "top-k E0 channels", "dem": "no", "model": "segformer", "objective": "deep segmentation",
           "kind": "patch", "requires_torch": True},
    "E3": {"geophysics": "top-k E0 channels", "dem": "DEM patches + modality masks", "model": "gated_fusion",
           "objective": "multiresolution fusion", "kind": "multires", "requires_torch": True, "requires_dem": True},
    "E4": {"geophysics": "selected pixel map + geophysical support", "dem": "candidate DEM morphology",
           "model": "lightgbm / xgboost ranker", "objective": "candidate ranking with topology", "kind": "segment"},
    "E5": {"geophysics": "ensemble of the G11 pixel maps", "dem": "-", "model": "fusion",
           "objective": "model complementarity", "kind": "fusion"},
}
PIXEL_DEFAULT = ["E0", "E1"]
DEEP_PATCH, DEEP_MULTIRES = "E2", "E3"
CHANNEL_SOURCE = "E0"              # deep-model channels = top-k features of this experiment's importance


def feature_selector(store, exp: str) -> list[str]:
    fam = EXPERIMENTS[exp.split("__")[0]].get("families")
    return store.predictors(fam) if fam else store.predictors()


def matrix_table():
    import pandas as pd

    return pd.DataFrame([{"experiment": k, **{c: v for c, v in d.items() if c != "families"}}
                         for k, d in EXPERIMENTS.items()])
