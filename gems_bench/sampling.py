"""PU-aware pixel sampling for tabular models.

Sample kinds (all drawn only from an ``allowed`` training mask):

``pos``      mapped-fault pixels (value >= 1).
``ring``     matched near-fault negatives at ``ring_km`` from visible labels — the
             "hard but reliable" negatives that teach line *localisation*.
``uniform``  background negatives anywhere outside the ignore band.

Rules:

* Pixels 1..``ignore_px`` from a visible label are never negatives (thin-label
  localisation uncertainty; the metric already tolerates 300 m).
* Pixels in ``ignore_mask`` (pseudo-discovery withheld systems + ring) are never sampled.
* Negatives on strong *unsupervised* lineament evidence (``pu_map`` >= quantile) far from
  visible labels are kept but down-weighted to ``pu_weight`` — unlabelled is not negative.
* Distances are computed only from ``visible_labels`` (labels the fold is allowed to see),
  so validation-area faults never influence which training pixels become negatives.
* Positive and negative total weights are balanced; probability scale is fixed later by the
  OOF output transform (metric.cross_fitted_transform).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

KIND_CODES = {"pos": 1, "ring": 2, "uniform": 3}


def label_distance_km(visible_labels: np.ndarray, grid) -> np.ndarray:
    return ndimage.distance_transform_edt(~visible_labels, sampling=(grid.pixel_km_y, grid.pixel_km_x)).astype(np.float32)


def _draw(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    return np.sort(idx)


def sample_pixels(allowed: np.ndarray, visible_labels: np.ndarray, grid, rng: np.random.Generator, *,
                  n_pos: int, n_neg_uniform: int, n_neg_ring: int, ignore_px: int = 2,
                  ring_km: tuple[float, float] = (0.5, 2.5), ignore_mask: np.ndarray | None = None,
                  pu_map: np.ndarray | None = None, pu_quantile: float = 0.97, pu_weight: float = 0.3,
                  pu_min_km: float = 2.5, dist_km: np.ndarray | None = None) -> pd.DataFrame:
    allowed = allowed.copy()
    if ignore_mask is not None:
        allowed &= ~ignore_mask
    d = label_distance_km(visible_labels, grid) if dist_km is None else dist_km
    band_km = ignore_px * grid.pixel_km * 1.0001
    pos_m = allowed & visible_labels
    neg_ok = allowed & ~visible_labels & (d > band_km)
    ring_m = neg_ok & (d >= ring_km[0]) & (d <= ring_km[1])
    uni_m = neg_ok & ~ring_m

    parts = []
    for kind, m, n in (("pos", pos_m, n_pos), ("ring", ring_m, n_neg_ring), ("uniform", uni_m, n_neg_uniform)):
        idx = _draw(m, n, rng)
        parts.append(pd.DataFrame({"idx": idx, "y": int(kind == "pos"), "kind": KIND_CODES[kind]}))
    df = pd.concat(parts, ignore_index=True)
    df["w"] = 1.0
    df["dist_km"] = d.reshape(-1)[df.idx.to_numpy()]
    if pu_map is not None:
        v = np.asarray(pu_map).reshape(-1)
        thr = np.nanquantile(v[np.flatnonzero(allowed)[:: max(1, allowed.sum() // 1_000_000)]], pu_quantile)
        s = v[df.idx.to_numpy()]
        pu = (df.y == 0) & (np.nan_to_num(s, nan=-np.inf) >= thr) & (df.dist_km >= pu_min_km)
        df.loc[pu, "w"] = pu_weight
        df["pu_candidate"] = pu
    else:
        df["pu_candidate"] = False
    wp, wn = df.w[df.y == 1].sum(), df.w[df.y == 0].sum()
    if wp > 0 and wn > 0:
        df.loc[df.y == 1, "w"] *= wn / wp
    return df


def sample_summary(df: pd.DataFrame, fold, split: str) -> pd.DataFrame:
    inv = {v: k for k, v in KIND_CODES.items()}
    g = df.groupby("kind").agg(n=("idx", "size"), weight=("w", "sum"), pu=("pu_candidate", "sum")).reset_index()
    g["kind"] = g.kind.map(inv)
    g["fold"], g["split"] = fold, split
    return g
