"""Overlap blending with a separable Hann window floored at ``floor`` (tile centres dominate; seams vanish)."""
from __future__ import annotations

import numpy as np


def window_2d(h: int, w: int, floor: float = 0.05) -> np.ndarray:
    return np.maximum(np.outer(np.hanning(h), np.hanning(w)), floor).astype(np.float32)


class BlendAccumulator:
    def __init__(self, shape, channels: int = 1, floor: float = 0.05):
        self.acc = np.zeros((channels, *shape), np.float32)
        self.wsum = np.zeros(shape, np.float32)
        self.floor = floor

    def add(self, r0: int, c0: int, pred: np.ndarray, valid: np.ndarray | None = None) -> None:
        pred = pred if pred.ndim == 3 else pred[None]
        h, w = pred.shape[-2:]
        wt = window_2d(h, w, self.floor)
        if valid is not None:
            wt = wt * valid
        self.acc[:, r0:r0 + h, c0:c0 + w] += pred * wt
        self.wsum[r0:r0 + h, c0:c0 + w] += wt

    def result(self, mask: np.ndarray | None = None) -> np.ndarray:
        ok = self.wsum > 0
        out = np.where(ok, self.acc / np.where(ok, self.wsum, 1), np.nan).astype(np.float32)
        if mask is not None:
            out[:, ~mask] = np.nan
        return out[0] if out.shape[0] == 1 else out


def predict_tiled(predict_fn, x: np.ndarray, valid: np.ndarray, tile: int, overlap: int, batch: int = 8) -> np.ndarray:
    """``predict_fn(batch (B, C, t, t)) -> (B, t, t)`` probabilities; ``x`` (C, H, W) with NaN filled by caller."""
    from .tiling import tiles

    H, W = valid.shape
    acc = BlendAccumulator((H, W))
    wins = tiles((H, W), tile, overlap, valid)
    for i in range(0, len(wins), batch):
        ws = wins[i:i + batch]
        xb = np.stack([x[:, r0:r1, c0:c1] for r0, r1, c0, c1 in ws])
        pb = predict_fn(xb)
        for (r0, r1, c0, c1), p in zip(ws, pb):
            acc.add(r0, c0, p, valid[r0:r1, c0:c1])
    return acc.result(valid)
