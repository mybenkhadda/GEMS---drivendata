"""System C - candidate (fault-hypothesis) relevance and utility.

Features per candidate (prefix -> family): ``seg_`` geometry / expert count, ``on_<pool>`` provenance (W2 / CNN pools
= model evidence), ``geo_`` geology and DEM means / steps (``geo_mean_dem_*`` = surface evidence), ``ori_``
orientation agreement, ``kn_`` permitted known-fault context, ``ev_`` honest model evidence (``ev_w2*`` / ``ev_cnn*``
= W2-aware; ``ev_surface*`` / ``ev_subsurface*`` = geology models), ``unc_`` uncertainty.

Targets: candidate relevance = near-truth share of TRAINING units (POSITIVE >= 0.5, NEGATIVE < 0.1, AMBIGUOUS
excluded), augmented with synthetic perturbations of training truth (graded positives / near misses) and random
far-from-truth segments (negatives). Models: LightGBM / Random Forest / CatBoost classifiers or a LightGBM
lambdarank (groups = units); weights system-balanced (``crossfit.system_balanced_weights``).

Utility (label-free at inference): expected marginal value of promoting a candidate on top of the exact-k W2 base:
    u = n_new x (p - kappa (1 - p)),  n_new = eligible candidate pixels farther than ``tol_px`` from the base
with p the calibrated relevance; kappa (> 0) expresses the FP cost relative to a TP (DTI: 0.2 FP vs 0.8 FN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import LeakageError, assert_label_free_columns

W2_AWARE = ("ev_w2", "ev_cnn", "on_w2", "on_cnn", "unc_w2")


def column_family(col: str) -> str:
    if col.startswith(W2_AWARE):
        return "model_evidence"
    if col.startswith(("ev_surface", "ev_subsurface", "ev_g06", "on_surface", "on_subsurface")):
        return "geology_model"
    for pre, fam in (("seg_", "geometry"), ("on_", "geometry"), ("geo_", "geology"), ("ori_", "orientation"),
                     ("kn_", "context_known"), ("ev_", "model_evidence"), ("unc_", "uncertainty")):
        if col.startswith(pre):
            return fam
    return "meta"


def candidate_feature_columns(T: pd.DataFrame, w2_aware: bool, families=None) -> list[str]:
    allow = set(families or ["geometry", "geology", "orientation", "context_known", "uncertainty", "geology_model"])
    if w2_aware:
        allow.add("model_evidence")
    cols = [c for c in T.columns if column_family(c) in allow and pd.api.types.is_numeric_dtype(T[c])]
    assert_label_free_columns(cols, "candidate features")
    if not w2_aware and any(column_family(c) == "model_evidence" for c in cols):
        raise LeakageError("geology-only candidate model received W2 / CNN columns")
    return cols


def train_candidate_classifier(kind: str, X, y, w, params: dict, seed: int):
    from ..g06.models_segment import train_segment_classifier

    return train_segment_classifier(kind, X, y, w, params, seed)


def train_candidate_ranker(X, y, groups, params: dict, seed: int):
    from ..g06.models_segment import train_segment_ranker

    return train_segment_ranker(X, y, groups, params, seed)


def estimate_candidate_utility(lab: np.ndarray, scores: pd.DataFrame, base: np.ndarray, eligible: np.ndarray,
                               kappa: float = 0.25, tol_px: float = 2.0, prob_col: str = "p") -> pd.DataFrame:
    """Utility of every candidate given the exact-k W2 base mask (label-free)."""
    from scipy import ndimage

    lab = np.asarray(lab, np.int32)
    n = int(lab.max())
    if not n or not len(scores):
        return pd.DataFrame(columns=["label", "n_new", "p", "utility"])
    near = ndimage.distance_transform_edt(~np.asarray(base, bool)) <= tol_px if np.asarray(base).any() else \
        np.zeros(lab.shape, bool)
    newm = (lab > 0) & np.asarray(eligible, bool) & ~near
    n_new = np.bincount(lab[newm], minlength=n + 1)[1:]
    S = pd.DataFrame({"label": np.arange(1, n + 1), "n_new": n_new})
    S = S.merge(scores[["label", prob_col]].rename(columns={prob_col: "p"}), on="label", how="left")
    S["p"] = S.p.fillna(0.0).clip(0, 1)
    S["utility"] = S.n_new * (S.p - float(kappa) * (1 - S.p))
    return S


def score_candidate_inventory(est, T: pd.DataFrame, cols: list[str]) -> np.ndarray:
    from ..g06.models_segment import _predict

    for c in cols:
        if c not in T:
            T[c] = np.nan
    return _predict(est, T[cols].to_numpy(np.float32))
