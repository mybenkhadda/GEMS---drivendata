"""System B - geology-only subsurface models on the REUSED G06 feature store (no W2 input).

Experiments (same visibilities, same pool, same features = the G06-P4 families raw + physics + structure):
    G07-B0  G06-P4 reference: G06 labels / weights (reproduces the G06 design under exact-k evaluation)
    G07-B1  improved supervision: system-balanced positives, distance-graded PU weights, stronger hard negatives
    G07-B2  distance-aware: LightGBM regression of the proximity target exp(-d / lambda) (localisation)
The question is whether label design, not model size, limited geology-only ML.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ProximityRegressor:
    """LightGBM regressor exposing ``predict_proba`` (clipped proximity in [0, 1]) so the G06 ``PixelModel`` /
    ``predict_pixel_raster`` code serves it unchanged."""

    def __init__(self, est):
        self.est = est

    def predict_proba(self, X):
        p = np.clip(self.est.predict(X), 0.0, 1.0)
        return np.stack([1 - p, p], 1)

    @property
    def booster_(self):
        return self.est.booster_


def load_g06_feature_store(run_dir, local_root=None, drive_root=None, stage_local: bool = True):
    """Open the G06 store; with ``stage_local`` the features directory is first mirrored from Drive to local disk
    (window reads over the Drive FUSE mount are slow). Nothing is recomputed."""
    from ..g06.feature_store import open_feature_store

    run_dir = Path(run_dir)
    if stage_local and drive_root and local_root and str(run_dir).startswith(str(drive_root)):
        from ..artifacts.sync import DriveSync

        rel = run_dir.relative_to(drive_root)
        DriveSync(Path(local_root), Path(drive_root)).stage_in([f"{rel.as_posix()}/features"])
        run_dir = Path(local_root) / rel
    return open_feature_store(run_dir)


def select_subsurface_features(store, families) -> list[str]:
    from ..g06.models_pixel import feature_columns_for

    return feature_columns_for(store, list(families), "A")


def train_subsurface_classifier(X, y, w, params: dict, seed: int, inner=None, inner_buffer=None):
    from ..g06.models_pixel import fit_pixel_model

    return fit_pixel_model("lightgbm", X, y, w, params, seed, inner, inner_buffer)


def train_fault_distance_model(X, target, w, params: dict, seed: int, inner=None, inner_buffer=None):
    """(ProximityRegressor, None, info); early stopping on the inner fold (L2), refit on all rows."""
    import lightgbm as lgb

    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    es = int(params.get("early_stopping_rounds", 0) or 0)
    info = {}
    if es and inner is not None and (inner == 0).any():
        ib = inner_buffer if inner_buffer is not None else np.zeros(len(inner), bool)
        va, tr = (inner == 0) & ~ib, (inner != 0) & ~ib
        m = lgb.LGBMRegressor(**p, random_state=int(seed), verbose=-1)
        m.fit(X[tr], target[tr], sample_weight=w[tr], eval_set=[(X[va], target[va])], eval_sample_weight=[w[va]],
              callbacks=[lgb.early_stopping(es, verbose=False)])
        p["n_estimators"] = max(int(m.best_iteration_ or p.get("n_estimators", 500)), 10)
        info["best_iteration"] = p["n_estimators"]
    est = lgb.LGBMRegressor(**p, random_state=int(seed), verbose=-1)
    est.fit(X, target, sample_weight=w)
    return ProximityRegressor(est), None, info


def predict_subsurface_raster(model, store, window, mask, tile_px: int = 512) -> np.ndarray:
    from ..g06.models_pixel import predict_pixel_raster

    return predict_pixel_raster(model, store, window, mask, tile_px)
