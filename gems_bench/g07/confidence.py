"""Confidence and uncertainty (never raw probabilities read as geological certainty).

* ``calibrate_candidate_confidence``: isotonic map from held-out candidate scores to relevance, fitted on the OTHER
  units of the scenario (their held-out scores were produced by models that never saw them) and applied to the
  held-out unit - honest with respect to that unit.
* ``estimate_localization_uncertainty``: per candidate, the half-maximum half-width (px) of the score profile across
  the candidate (sampled along its normal) - a broad response = uncertain centreline.
* ``estimate_model_disagreement``: per-pixel standard deviation of expert ranks.
* ``build_confidence_corridor``: continuous map = calibrated confidence on the candidate pixels with a halo whose
  width follows the localisation uncertainty (``representation.soft_corridor``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def calibrate_candidate_confidence(train_scores: np.ndarray, train_targets: np.ndarray, test_scores: np.ndarray,
                                   min_rows: int = 30) -> tuple[np.ndarray, dict]:
    """(calibrated test probabilities, info). Falls back to rank scaling when too few training rows."""
    from sklearn.isotonic import IsotonicRegression

    s, t = np.asarray(train_scores, float), np.asarray(train_targets, float)
    ok = np.isfinite(s) & np.isfinite(t)
    if ok.sum() < min_rows or np.unique(t[ok]).size < 2:
        r = pd.Series(test_scores).rank(pct=True).to_numpy()
        return r, {"method": "rank_fallback", "n_train": int(ok.sum())}
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s[ok], t[ok])
    return iso.predict(np.asarray(test_scores, float)), {"method": "isotonic", "n_train": int(ok.sum()),
                                                         "base_rate": float(t[ok].mean())}


def estimate_localization_uncertainty(score: np.ndarray, lab: np.ndarray, max_px: int = 4) -> pd.DataFrame:
    """Per label: mean across-candidate half-width (px) at half the centre value (``max_px`` when never reached)."""
    from ..p6.segment_recovery import label_geometry

    lab = np.asarray(lab, np.int32)
    n = int(lab.max())
    if not n:
        return pd.DataFrame(columns=["label", "halfwidth_px"])
    G = label_geometry(lab)
    th = G.theta.to_numpy()
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    nr, nc = np.cos(th[lb - 1]), -np.sin(th[lb - 1])
    s = np.nan_to_num(np.asarray(score, np.float64), nan=0.0)
    centre = s[rr, cc]
    hw = np.full(rr.size, float(max_px))
    for d in range(1, max_px + 1):
        a = ndimage.map_coordinates(s, [rr + d * nr, cc + d * nc], order=1, mode="nearest")
        b = ndimage.map_coordinates(s, [rr - d * nr, cc - d * nc], order=1, mode="nearest")
        drop = (0.5 * (a + b) <= 0.5 * centre) & (hw == max_px)
        hw[drop] = d
    m = pd.Series(hw).groupby(lb).mean()
    return pd.DataFrame({"label": m.index.to_numpy(), "halfwidth_px": m.to_numpy()})


def estimate_model_disagreement(rank_maps: dict, ev: np.ndarray) -> np.ndarray:
    from ..gems3.representation import rank01

    if len(rank_maps) < 2:
        return np.zeros(np.shape(ev), np.float32)
    st = np.stack([rank01(np.nan_to_num(m, nan=0.0), ev) for m in rank_maps.values()])
    return np.where(ev, st.std(0), np.nan).astype(np.float32)


def build_confidence_corridor(lab: np.ndarray, selected_labels, conf: pd.DataFrame, width: pd.DataFrame,
                              ev: np.ndarray, halo_value: float = 0.5, v_min: float = 0.4) -> np.ndarray:
    """Soft map of the selected candidates: calibrated confidence on their pixels, halo scale = median half-width."""
    from .representation import soft_corridor

    lab = np.asarray(lab, np.int32)
    core = np.isin(lab, np.asarray(list(selected_labels), np.int32)) & (lab > 0)
    c = np.zeros(lab.shape, np.float32)
    if len(conf):
        lut = np.zeros(int(lab.max()) + 1, np.float32)
        lut[conf.label.to_numpy().astype(int)] = conf.p.to_numpy(np.float32)
        c = lut[lab]
    hw = float(np.nanmedian(width.halfwidth_px)) if len(width) else 1.0
    return soft_corridor(core, c, max(hw, 0.5), halo_value, v_min, ev)
