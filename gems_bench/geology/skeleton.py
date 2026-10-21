"""Fault-trace topology on the native grid: skeleton, endpoints, branch points, component lengths.

Used to (1) build orientation targets only where the local trace direction is defined (not at branch
points / intersections, not on very short pieces), (2) score endpoint / intersection / continuity recall,
(3) describe candidate lineaments. All functions take boolean rasters; none reads model predictions.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
_S8 = np.ones((3, 3), bool)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize as _sk

    return _sk(np.asarray(mask, bool))


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    return ndimage.convolve(skel.astype(np.uint8), _K8, mode="constant") * skel


def endpoints(skel: np.ndarray) -> np.ndarray:
    return skel & (neighbour_count(skel) == 1)


def branch_points(skel: np.ndarray) -> np.ndarray:
    return skel & (neighbour_count(skel) >= 3)


def component_length_km(skel: np.ndarray, pixel_km: float) -> np.ndarray:
    """Per-pixel length (km) of the 8-connected skeleton component the pixel belongs to (0 off-skeleton).

    Length = pixel count x pixel size x 1.12 (mean step of a digital curve with mixed axial / diagonal moves);
    adequate for strata (<1, 1-2, 2-5 km ...), not for surveying.
    """
    lab, n = ndimage.label(skel, structure=_S8)
    if n == 0:
        return np.zeros(skel.shape, np.float32)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (counts[lab] * pixel_km * 1.12).astype(np.float32)


def near(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, bool)
    return ndimage.distance_transform_edt(~mask) <= radius_px


def orientation_targets(visible: np.ndarray, grid, *, kern: np.ndarray | None = None, scale_km: float = 0.3,
                        min_coherence: float = 0.7, junction_px: float = 3.0, min_length_km: float = 1.0,
                        exclude_endpoints_px: float = 0.0) -> dict[str, np.ndarray]:
    """(cos 2θ, sin 2θ, valid) training targets from VISIBLE labels.

    θ = structure-tensor strike of the visible label raster (Phase-4 convention). Supervision is restricted
    to pixels with ``kern > 0`` (within the 300 m metric kernel of a visible trace) and removed where the
    direction is ambiguous: low coherence, within ``junction_px`` of a branch point / intersection, on
    skeleton components shorter than ``min_length_km`` (propagated to their kernel neighbourhood), and
    optionally near endpoints.
    """
    from .orientation import axial_vector, strike_field

    vis = np.asarray(visible, bool)
    st, coh = strike_field(vis.astype(np.float32), grid, scale_km)
    sk = skeletonize(vis)
    bad = near(branch_points(sk), junction_px)
    if exclude_endpoints_px > 0:
        bad |= near(endpoints(sk), exclude_endpoints_px)
    length = component_length_km(sk, grid.pixel_km)
    short_sk = sk & (length < min_length_km)
    if short_sk.any():                                  # spread "short" to the kernel neighbourhood of the piece
        _, idx = ndimage.distance_transform_edt(~sk, return_indices=True)
        bad |= short_sk[idx[0], idx[1]]
    support = (kern > 0) if kern is not None else near(vis, 3)
    ok = support & ~bad & (np.nan_to_num(coh) >= min_coherence)
    c2, s2 = axial_vector(np.nan_to_num(st))
    return {"ocos": np.where(ok, c2, 0).astype(np.float32), "osin": np.where(ok, s2, 0).astype(np.float32),
            "ovalid": ok.astype(np.float32)}


def trace_strata(labels: np.ndarray, grid) -> dict[str, np.ndarray]:
    """Boolean label-pixel strata for geometry metrics: endpoint / intersection / interior, length bins."""
    sk = skeletonize(labels)
    _, idx = ndimage.distance_transform_edt(~sk, return_indices=True)
    length = component_length_km(sk, grid.pixel_km)[idx[0], idx[1]] * labels
    ep, bp = near(endpoints(sk), 2), near(branch_points(sk), 2)
    return {
        "endpoint": labels & ep, "intersection": labels & bp, "interior": labels & ~ep & ~bp,
        "len_lt1": labels & (length < 1.0), "len_1_2": labels & (length >= 1.0) & (length < 2.0),
        "len_2_5": labels & (length >= 2.0) & (length < 5.0), "len_ge5": labels & (length >= 5.0),
    }
