"""Physically valid augmentation: the dihedral group of the square grid, nothing else.

Every input channel is either a rotation/flip-invariant scalar (field magnitudes, THD, tilt, analytic signal,
curvatures, statistics ...) or an axial doubled-angle pair (cos 2t, sin 2t). For an axial pair
* a 90-degree rotation maps t -> t + 90 deg:   (c, s) -> (-c, -s);
* a flip of either axis maps t -> -t (mod 180): (c, s) -> (c, -s).
Intensity jitter, scaling, elastic warps and noise are deliberately absent (they would change physical units).
"""
from __future__ import annotations

import numpy as np


def draw(rng: np.random.Generator, cfg: dict) -> tuple[int, bool]:
    a = cfg["training"]["augmentation"]
    k = int(rng.integers(4)) if a["rot90"] else 0
    flip = bool(rng.random() < 0.5) if a["flips"] else False
    return k, flip


def apply(x: np.ndarray, k: int, flip: bool, axial_pairs=()) -> np.ndarray:
    """``x``: (C, H, W). Spatial transform first, then the axial sign rule."""
    if k:
        x = np.rot90(x, k, axes=(1, 2))
    if flip:
        x = x[:, :, ::-1]
    if axial_pairs and (k % 2 or flip):
        x = np.array(x, copy=True)
        for ci, si in axial_pairs:
            if k % 2:
                x[ci] = -x[ci]
                x[si] = -x[si]
            if flip:
                x[si] = -x[si]
    return np.ascontiguousarray(x)
