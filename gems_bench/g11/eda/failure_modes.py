"""Failure-mode taxonomy: DESCRIPTIVE scores in [0, 1] per candidate / region (never definitive labels).

Each score combines available label-free evidence; a score is NaN when its evidence is missing (e.g. no DEM).
Mode -> evidence:
  agreement            geophysical edge rank AND terrain scarp rank high
  geophysics_only      geo high, terrain low (terrain observed)
  terrain_only         terrain high, geo low
  weak_both            both low
  fragmented           many short collinear candidates nearby (local density x 1 / length)
  crossing_branching   branch_count > 0
  road_like            very high linearity + very low sinuosity + low relief contrast (anthropogenic straightness)
  drainage_confusion   valley strength high + low scarp strength
  seam_artifact        candidate within one cell of a DEM seam
  strong_gradient      high slope without cross-fault step
  missing_dem          no DEM coverage
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _r(x):
    return np.clip(np.asarray(x, float), 0, 1)


def failure_scores(seg: pd.DataFrame) -> pd.DataFrame:
    g = seg.get("geo_edge_rank_mean", pd.Series(np.nan, index=seg.index))
    t = seg.get("terrain_scarp_rank_mean", pd.Series(np.nan, index=seg.index))
    out = pd.DataFrame(index=seg.index)
    out["agreement"] = np.sqrt(_r(g) * _r(t))
    out["geophysics_only"] = _r(g) * (1 - _r(t))
    out["terrain_only"] = _r(t) * (1 - _r(g))
    out["weak_both"] = (1 - _r(g)) * (1 - _r(t))
    dens = seg.get("local_density_km_per_km2", pd.Series(0, index=seg.index)).fillna(0)
    L = seg.get("length_m", pd.Series(np.nan, index=seg.index))
    out["fragmented"] = _r(dens / (dens.quantile(0.95) + 1e-9)) * _r(1000.0 / L.clip(lower=1))
    out["crossing_branching"] = (seg.get("branch_count", pd.Series(0, index=seg.index)).fillna(0) > 0).astype(float)
    lin, sin = seg.get("linearity", pd.Series(np.nan, index=seg.index)), seg.get("sinuosity",
                                                                                 pd.Series(np.nan, index=seg.index))
    step = seg.get("elevation_step_m", pd.Series(np.nan, index=seg.index)).abs()
    out["road_like"] = _r((lin - 0.98) / 0.02) * _r((1.02 - sin) / 0.02) * _r(1 - step / 2.0)
    out["drainage_confusion"] = _r(seg.get("valley_rank_mean", pd.Series(np.nan, index=seg.index))) * \
        _r(1 - _r(seg.get("scarp_strength", pd.Series(np.nan, index=seg.index)) / 5.0))
    out["seam_artifact"] = _r(seg.get("dem_seam_mean", pd.Series(np.nan, index=seg.index)) * 5)
    out["strong_gradient"] = _r(seg.get("slope_rank_mean", pd.Series(np.nan, index=seg.index))) * _r(1 - step / 5.0)
    pv = seg.get("profile_valid_fraction", pd.Series(np.nan, index=seg.index))
    out["missing_dem"] = 1 - _r(pv.fillna(0))
    scores = out.astype(float)
    any_ok = scores.notna().any(axis=1)
    out["dominant_mode"] = np.where(any_ok, scores.fillna(-1.0).idxmax(axis=1), "insufficient_evidence")
    return out
