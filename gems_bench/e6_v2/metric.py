"""Distance-weighted Tversky index (DTI), canonical definition, re-implemented for this pipeline.

    k(d)  = max(1 - d / R, 0),  R = 300 m
    TP_w  = sum_{g in G} max_x p(x) k(d(x, g))
    FP_w  = sum_x p(x) [1 - k(dist(x, G))]
    FN_w  = |G| - TP_w
    DTI   = TP_w / (TP_w + 0.2 FP_w + 0.8 FN_w + 1e-6)

Conventions: NaN predictions are 0; values are clipped to [0, 1]; pixels outside ``eval_mask`` contribute
nothing (zeroed before the local max); ``exclude_mask`` pixels (known faults in the hidden test, retained faults in
discovery) are removed from G and from the FP sum and are zeroed before the local max, so they have no effect at
all (the organiser statement "it should not matter whether these known faults are included"). Pooling sums
components over units / folds before forming the index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

ALPHA, BETA, RADIUS_M, EPS = 0.2, 0.8, 300.0, 1e-6


@dataclass(frozen=True)
class Components:
    tp: float
    fp: float
    fn: float
    n_truth: int
    pred_mass: float

    @property
    def dti(self) -> float:
        return float(self.tp / (self.tp + ALPHA * self.fp + BETA * self.fn + EPS))

    def __add__(self, o: "Components") -> "Components":
        return Components(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn, self.n_truth + o.n_truth,
                          self.pred_mass + o.pred_mass)

    def row(self) -> dict:
        return {"dti": self.dti, "tp_w": self.tp, "fp_w": self.fp, "fn_w": self.fn, "n_truth": self.n_truth,
                "pred_mass": self.pred_mass, "recall_w": self.tp / max(self.n_truth, 1),
                "precision_w": self.tp / max(self.tp + self.fp, 1e-12)}


def pooled(cs: list[Components]) -> Components:
    out = Components(0.0, 0.0, 0.0, 0, 0.0)
    for c in cs:
        out = out + c
    return out


def offsets(radius_m: float, px: float) -> list[tuple[int, int, float]]:
    r = int(np.floor(radius_m / px))
    out = []
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            k = 1.0 - np.hypot(dr * px, dc * px) / radius_m
            if k > 0:
                out.append((dr, dc, float(k)))
    return out


def local_weighted_max(p: np.ndarray, radius_m: float, px: float) -> np.ndarray:
    offs = offsets(radius_m, px)
    r = max(max(abs(a), abs(b)) for a, b, _ in offs)
    h, w = p.shape
    P = np.pad(p, r)
    M = np.zeros_like(p, dtype=np.float64)
    for dr, dc, k in offs:
        np.maximum(M, k * P[r + dr:r + dr + h, r + dc:r + dc + w], out=M)
    return M


def _bbox(mask: np.ndarray, pad: int):
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return slice(0, 0), slice(0, 0)
    return (slice(max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, mask.shape[0])),
            slice(max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, mask.shape[1])))


def components(pred: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, exclude_mask: np.ndarray | None = None,
               px: float = 100.0, radius_m: float = RADIUS_M) -> Components:
    sl = _bbox(eval_mask, int(np.ceil(radius_m / px)) + 1)
    ev = eval_mask[sl]
    ex = np.zeros_like(ev) if exclude_mask is None else exclude_mask[sl]
    p = np.asarray(pred[sl], np.float64)
    p = np.clip(np.where(np.isfinite(p), p, 0.0), 0.0, 1.0)
    p[~ev | ex] = 0.0
    G = truth[sl] & ev & ~ex
    n = int(G.sum())
    region = ev & ~ex
    mass = float(p[ev].sum())
    if n == 0:
        return Components(0.0, float(p[region].sum()), 0.0, 0, mass)
    tp = float(local_weighted_max(p, radius_m, px)[G].sum())
    k = np.clip(1.0 - ndimage.distance_transform_edt(~G) * px / radius_m, 0.0, None)
    fp = float((p * (1.0 - k))[region].sum())
    return Components(tp, fp, float(n - tp), n, mass)


def support(pred: np.ndarray, eval_mask: np.ndarray, exclude_mask=None, px: float = 100.0,
            radius_m: float = RADIUS_M) -> np.ndarray:
    """Per-pixel TP support M(g) on the full grid (for segment-level recall)."""
    p = np.clip(np.where(np.isfinite(pred), pred, 0.0), 0.0, 1.0).astype(np.float64)
    p[~eval_mask] = 0.0
    if exclude_mask is not None:
        p[exclude_mask] = 0.0
    return local_weighted_max(p, radius_m, px)
