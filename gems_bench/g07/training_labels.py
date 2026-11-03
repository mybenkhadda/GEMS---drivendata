"""Improved fault supervision (training-visible faults only).

Builds on ``g06.supervision`` (POSITIVE / IGNORE / UNLABELED / LIKELY_BACKGROUND from VISIBLE labels):

* ``visible_fault_labels``      visible label mask of a training visibility (hidden faults never appear)
* ``fault_distance_targets``    proximity target exp(-d / lambda) in [0, 1] and the capped distance (km)
* ``fault_orientation_targets`` doubled-angle strike of the visible traces where the orientation is stable
* ``positive_unlabeled_targets`` PU classes + confidence-aware weights (system-balanced positives)
* ``make_hard_negative_pool``   label-free high-evidence pixels far from every VISIBLE fault
* ``build_fault_system_groups`` system id of every visible fault pixel + coarse spatial groups
* ``perturbed_candidates``      controlled geometric perturbations (lateral shift, rotation, truncation) of TRAINING
                                truth traces -> many graded positive / near-miss candidates for the candidate ranker.
                                Only the truth of a TRAINING unit is perturbed; a held-out unit's truth is never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..g06 import supervision as _S6

IGNORE, LIKELY_BACKGROUND, POSITIVE, UNLABELED = _S6.IGNORE, _S6.LIKELY_BACKGROUND, _S6.POSITIVE, _S6.UNLABELED
build_orientation_targets, build_pixel_supervision = _S6.build_orientation_targets, _S6.build_pixel_supervision
class_confidence, visible_distance_km = _S6.class_confidence, _S6.visible_distance_km


def visible_fault_labels(vis) -> np.ndarray:
    """The visible labels of a ``g06.splits.Visibility`` (asserts hidden and visible are disjoint)."""
    from ..g06.contracts import LeakageError

    if (vis.visible & vis.hidden).any():
        raise LeakageError(f"{vis.fit_id}: visible and hidden labels overlap")
    return np.asarray(vis.visible, bool)


def fault_distance_targets(visible: np.ndarray, lambda_km: float = 0.3, cap_km: float = 5.0) -> dict:
    d = visible_distance_km(visible)
    return {"dist_km": np.minimum(d, cap_km).astype(np.float32),
            "proximity": np.exp(-np.minimum(d, cap_km) / float(lambda_km)).astype(np.float32)}


def fault_orientation_targets(visible: np.ndarray, sigma_px: float = 3.0, min_coherence: float = 0.5) -> dict:
    return build_orientation_targets(visible, sigma_px, min_coherence)


def positive_unlabeled_targets(visible: np.ndarray, region: np.ndarray, valid: np.ndarray, cfg: dict) -> dict:
    sup = build_pixel_supervision(visible, region, valid, cfg)
    sup["confidence"] = class_confidence(sup["classes"], cfg)
    return sup


def make_hard_negative_pool(evidence: np.ndarray, visible: np.ndarray, region: np.ndarray, min_km: float,
                            quantile: float, rng, n: int) -> np.ndarray:
    """Flat indices of label-free high-evidence pixels inside ``region`` at >= ``min_km`` from VISIBLE faults."""
    e = np.asarray(evidence, np.float64)
    ok = np.asarray(region, bool) & np.isfinite(e)
    if not ok.any():
        return np.zeros(0, np.int64)
    far = visible_distance_km(visible) >= float(min_km)
    thr = np.quantile(e[ok], float(quantile))
    cand = np.flatnonzero((ok & far & (e >= thr)).ravel())
    return np.sort(rng.choice(cand, min(int(n), cand.size), replace=False)) if cand.size else cand


def build_fault_system_groups(visible: np.ndarray, system_map: np.ndarray, block_km: float = 20.0) -> dict:
    from ..g06.splits import block_ids

    sm = np.where(np.asarray(visible, bool), np.asarray(system_map), -1).astype(np.int32)
    sizes = pd.Series(sm[sm >= 0]).value_counts()
    return {"system_map": sm, "system_px": sizes, "blocks": block_ids(np.shape(visible), block_km)}


def _skeleton_traces(truth: np.ndarray, sysmap: np.ndarray, min_px: int = 5) -> list:
    """Per truth system: ordered skeleton pixel coordinates (row, col)."""
    from ..postprocess.topology import skeleton

    sk = skeleton(np.asarray(truth, bool))
    out = []
    for sid in np.unique(np.asarray(sysmap)[sk & (np.asarray(sysmap) >= 0)]):
        rr, cc = np.nonzero(sk & (np.asarray(sysmap) == sid))
        if rr.size >= min_px:
            out.append((int(sid), rr, cc))
    return out


def perturbed_candidates(truth: np.ndarray, sysmap: np.ndarray, rng, n_per_system: int = 4, max_shift_px: float = 4.0,
                         max_rot_deg: float = 15.0, keep_range=(0.4, 1.0), max_len_px: int = 60) -> tuple[np.ndarray,
                                                                                                      pd.DataFrame]:
    """Synthetic candidate label raster from controlled perturbations of a TRAINING unit's truth traces.

    Each candidate = a contiguous piece (``keep_range`` share, <= ``max_len_px``) of one system skeleton, rotated by
    a random angle around its centroid and shifted laterally; relevance is measured afterwards like any candidate
    (near-truth share). Returns (int32 labels, table: label, system_id, shift_px, rot_deg, keep)."""
    h, w = np.shape(truth)
    lab = np.zeros((h, w), np.int32)
    rows, nxt = [], 1
    for sid, rr, cc in _skeleton_traces(truth, sysmap):
        v = np.array([0.0, 1.0])
        if rr.size > 2:
            _, vec = np.linalg.eigh(np.cov(np.stack([rr - rr.mean(), cc - cc.mean()])))
            v = vec[:, -1]                                            # principal axis (row, col)
        order = np.argsort((rr - rr.mean()) * v[0] + (cc - cc.mean()) * v[1])
        rr, cc = rr[order], cc[order]
        nr, nc = -v[1], v[0]                                          # lateral (normal) direction
        for _ in range(int(n_per_system)):
            keep = rng.uniform(*keep_range)
            n = max(min(int(round(keep * rr.size)), max_len_px), 3)
            s0 = int(rng.integers(0, max(rr.size - n, 0) + 1))
            pr, pc = rr[s0:s0 + n].astype(float), cc[s0:s0 + n].astype(float)
            mr, mc = pr.mean(), pc.mean()
            a = np.radians(rng.uniform(-max_rot_deg, max_rot_deg))
            dr, dc = pr - mr, pc - mc
            r2, c2 = mr + dr * np.cos(a) - dc * np.sin(a), mc + dr * np.sin(a) + dc * np.cos(a)
            sh = rng.uniform(-max_shift_px, max_shift_px)
            r2, c2 = r2 + sh * nr, c2 + sh * nc
            ri, ci = np.round(r2).astype(int), np.round(c2).astype(int)
            ok = (ri >= 0) & (ri < h) & (ci >= 0) & (ci < w)
            if ok.sum() < 3:
                continue
            free = lab[ri[ok], ci[ok]] == 0
            if free.sum() < 3:
                continue
            lab[ri[ok][free], ci[ok][free]] = nxt
            rows.append({"label": nxt, "system_id": sid, "shift_px": float(sh), "rot_deg": float(np.degrees(a)),
                         "keep": float(keep), "synthetic": True})
            nxt += 1
    return lab, pd.DataFrame(rows, columns=["label", "system_id", "shift_px", "rot_deg", "keep", "synthetic"])


def random_negative_candidates(eligible: np.ndarray, truth_like: np.ndarray, rng, n: int, length_px: int = 20,
                               min_dist_px: float = 10.0) -> np.ndarray:
    """Straight random segments far (>= ``min_dist_px``) from ``truth_like`` (training truth + visible faults):
    explicit negatives for the candidate ranker. Label-free placement, training units only."""
    h, w = np.shape(eligible)
    far = ndimage.distance_transform_edt(~np.asarray(truth_like, bool)) >= min_dist_px
    ok = np.asarray(eligible, bool) & far
    idx = np.flatnonzero(ok.ravel())
    lab = np.zeros((h, w), np.int32)
    if not idx.size:
        return lab
    nxt = 1
    for i in rng.choice(idx, min(int(n), idx.size), replace=False):
        r0, c0 = divmod(int(i), w)
        a = rng.uniform(0, np.pi)
        t = np.arange(length_px) - length_px / 2
        rr, cc = np.round(r0 + t * np.sin(a)).astype(int), np.round(c0 + t * np.cos(a)).astype(int)
        m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc = rr[m], cc[m]
        m2 = ok[rr, cc] & (lab[rr, cc] == 0)
        if m2.sum() >= 5:
            lab[rr[m2], cc[m2]] = nxt
            nxt += 1
    return lab
