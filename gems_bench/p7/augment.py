"""Geologically controlled augmentation (pure NumPy; applied per patch on the CPU before batching).

Geometric operations act on the WHOLE patch at once (every input channel, every target), so modalities can never
be misaligned and channels are never rotated independently. Axial orientation targets are doubled-angle vectors
(cos 2t, sin 2t) of the strike t (geology.orientation conventions; the reference axis is grid-aligned), so:

    flip left-right or up-down:   t -> -t (mod 180)        =>  (c, s) -> ( c, -s)
    rot90 (either sense):         t -> t -/+ 90            =>  (c, s) -> (-c, -s)
    rot180:                       t -> t                   =>  unchanged

Presets
    conservative  rot180 only (M9 parity: keeps the regional strike families), mild channel noise and intensity
                  scaling; the default.
    structural    any rot90 + flips, stronger noise, intensity scaling and modality dropout. Rot90 / flips change
                  the strike distribution relative to the regional stress field (e.g. N-S Basin-and-Range trends), so
                  this preset is compared only after the baseline architecture is established.
Crop variation = the random patch origin of the sampler (positive-centred draws included); patches are never
resampled or rescaled (resampling changes gradient magnitudes of potential-field derivatives).
"""
from __future__ import annotations

import numpy as np

ORIENT = ("ocos", "osin")
PRESETS = {
    "none": {"ops": [], "noise": 0.0, "scale": 0.0, "modality_dropout": 0.0},
    "conservative": {"ops": ["rot180"], "noise": 0.02, "scale": 0.05, "modality_dropout": 0.0},
    "structural": {"ops": ["rot90", "flip_lr", "flip_ud"], "noise": 0.05, "scale": 0.10, "modality_dropout": 0.10},
}


def transform_orientation(c: np.ndarray, s: np.ndarray, op: str) -> tuple[np.ndarray, np.ndarray]:
    """Doubled-angle components after the VALUE part of a geometric op (the spatial part is ``apply_spatial``)."""
    if op in ("flip_lr", "flip_ud"):
        return c, -s
    if op in ("rot90", "rot270"):
        return -c, -s
    if op in ("rot180", "identity"):
        return c, s
    raise KeyError(op)


def apply_spatial(a: np.ndarray, op: str) -> np.ndarray:
    """Spatial part on the last two axes (works for (H, W) and (C, H, W))."""
    if op == "identity":
        return a
    if op == "rot90":
        return np.rot90(a, 1, axes=(-2, -1))
    if op == "rot180":
        return np.rot90(a, 2, axes=(-2, -1))
    if op == "rot270":
        return np.rot90(a, 3, axes=(-2, -1))
    if op == "flip_lr":
        return a[..., :, ::-1]
    if op == "flip_ud":
        return a[..., ::-1, :]
    raise KeyError(op)


def apply_geometric(x: np.ndarray, targets: dict, op: str) -> tuple[np.ndarray, dict]:
    """(inputs, targets) after ``op``; orientation components get the value transform too."""
    if op == "identity":
        return x, targets
    xo = np.ascontiguousarray(apply_spatial(x, op))
    t = {k: np.ascontiguousarray(apply_spatial(v, op)) for k, v in targets.items()}
    if ORIENT[0] in t and ORIENT[1] in t:
        t[ORIENT[0]], t[ORIENT[1]] = transform_orientation(t[ORIENT[0]], t[ORIENT[1]], op)
    return xo, t


def sample_ops(rng: np.random.Generator, preset: dict) -> list[str]:
    """Ops drawn for one patch: rot180 w.p. 0.5; rot90 picks one of rot90/180/270/identity; flips w.p. 0.5 each."""
    ops = []
    allowed = preset["ops"]
    if "rot90" in allowed:
        ops.append(["identity", "rot90", "rot180", "rot270"][int(rng.integers(4))])
    elif "rot180" in allowed and rng.random() < 0.5:
        ops.append("rot180")
    for f in ("flip_lr", "flip_ud"):
        if f in allowed and rng.random() < 0.5:
            ops.append(f)
    return [o for o in ops if o != "identity"]


def photometric(x: np.ndarray, valid: np.ndarray, groups: dict[str, list[int]], preset: dict,
                rng: np.random.Generator, protect: list[int] | None = None) -> tuple[np.ndarray, list[str]]:
    """Channel noise + per-channel intensity scaling inside the footprint, and modality dropout (one whole group
    zeroed, never every group). ``protect`` channels (e.g. the footprint) are left untouched. Spatially uniform per
    channel: no misalignment is introduced."""
    x = np.array(x, np.float32, copy=True)
    applied = []
    keep = set(protect or [])
    idx = [i for i in range(x.shape[0]) if i not in keep]
    vm = np.asarray(valid, bool)
    if preset["scale"] > 0 and idx:
        f = np.exp(rng.uniform(-preset["scale"], preset["scale"], len(idx))).astype(np.float32)
        x[idx] *= f[:, None, None]
        applied.append("scale")
    if preset["noise"] > 0 and idx:
        x[idx] += (rng.standard_normal((len(idx),) + x.shape[1:]).astype(np.float32) * preset["noise"]) * vm[None]
        applied.append("noise")
    names = [g for g, ix in groups.items() if [i for i in ix if i not in keep]]
    if preset["modality_dropout"] > 0 and len(names) > 1 and rng.random() < preset["modality_dropout"]:
        g = names[int(rng.integers(len(names)))]
        x[[i for i in groups[g] if i not in keep]] = 0.0
        applied.append(f"drop_{g}")
    return x, applied
