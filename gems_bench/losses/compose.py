"""Assemble training rasters and the Phase-5 objective from one ``training.spec.P5Spec``.

Training rasters (fp16, built from VISIBLE labels only, per fold / discovery split)
-----------------------------------------------------------------------------------
``target`` ``kern`` ``weight``    as Phase 3/4 (cnn3.build_targets; PU ignore band at weight 0)
``weight`` x B1 weights            on unlabeled pixels only, when ``spec.pu == "bg_weight"``
``dist``                           proximity (``aux_dist`` > 0)
``ocos`` ``osin`` ``ovalid``       orientation targets with junction / short-piece exclusion (``aux_orient`` > 0)

Objective
---------
base     ``dti`` (B0 reference: cnn3.loss_fn, i.e. the DTI surrogate + small BCE) or ``nnpu`` (B2)
+ aux_dist * proximity + aux_orient * orientation (cosine | mse)
+ w_centerline * soft-clDice + w_thin * thinness + w_continuity * along-strike continuity
+ consistency * rampup * mean-teacher term (added by the trainer, which owns the teacher)
"""
from __future__ import annotations

import numpy as np

from . import geometry, pu

AUX_ORDER5 = ("target", "kern", "weight", "dist", "ocos", "osin", "ovalid")


def build_targets5(visible: np.ndarray, grid, spec, weight_mask: np.ndarray, *, bg_weights: np.ndarray | None = None,
                   ) -> tuple[np.ndarray, list[str]]:
    from scipy import ndimage

    from ..geology.skeleton import orientation_targets
    from ..p3 import cnn3

    target, kern, band = cnn3.build_targets(visible, grid, spec.ignore_px)
    w = (weight_mask & ~band).astype(np.float32)
    if bg_weights is not None:
        w = np.where(visible, w, w * bg_weights)                 # positives keep weight 1
    layers = {"target": target, "kern": kern, "weight": w.astype(np.float16)}
    if spec.aux_dist > 0:
        d = ndimage.distance_transform_edt(~visible, sampling=(grid.pixel_km_y, grid.pixel_km_x))
        layers["dist"] = (1 - np.clip(d / spec.dist_clip_km, 0, 1)).astype(np.float16)
    if spec.aux_orient > 0:
        o = orientation_targets(visible, grid, kern=kern, min_coherence=spec.orient_min_coherence,
                                junction_px=spec.orient_junction_px, min_length_km=spec.orient_min_length_km)
        layers.update({k: v.astype(np.float16) for k, v in o.items()})
    names = [n for n in AUX_ORDER5 if n in layers]
    return np.stack([layers[n] for n in names]), names


def loss5(out: dict, a, names: list[str], spec, *, epoch: int = 0, return_parts: bool = False):
    """Total Phase-5 loss for one batch; ``parts`` are detached device tensors (no per-step host sync)."""
    from ..p3 import cnn3

    i = {n: k for k, n in enumerate(names)}

    def sl(n):
        return a[:, i[n]:i[n] + 1]

    y, k, w = sl("target"), sl("kern"), sl("weight")
    logits = out["seg"].float()
    parts = {}
    if spec.pu == "nnpu":
        loss, pp = pu.nnpu_dti_surrogate(logits, y, k, w, spec.pu_prior, spec.tversky_alpha, spec.tversky_beta,
                                         spec.bce_weight, return_parts=True)
        parts.update({f"pu_{n}": v for n, v in pp.items()})
    else:
        loss = cnn3.loss_fn(logits, y, k, w, spec)
    parts["base"] = loss.detach()
    if spec.aux_dist > 0 and "dist" in out:
        l_d = geometry.proximity_loss(out["dist"], sl("dist"), w)
        loss = loss + spec.aux_dist * l_d
        parts["dist"] = l_d.detach()
    if spec.aux_orient > 0 and "orient" in out:
        l_o = geometry.orientation_loss(out["orient"], sl("ocos"), sl("osin"), sl("ovalid") * w, spec.orient_loss)
        loss = loss + spec.aux_orient * l_o
        parts["orient"] = l_o.detach()
    need_p = spec.w_centerline > 0 or spec.w_thin > 0 or spec.w_continuity > 0
    if need_p:
        import torch

        p = torch.sigmoid(logits)
        if spec.w_centerline > 0:
            l_c = geometry.cl_dice(p, y, k, w)
            loss = loss + spec.w_centerline * l_c
            parts["cldice"] = l_c.detach()
        if spec.w_thin > 0:
            l_t = geometry.thinness(p, w)
            loss = loss + spec.w_thin * l_t
            parts["thin"] = l_t.detach()
        if spec.w_continuity > 0 and "orient" in out:
            l_k = geometry.continuity(p, out["orient"], w, spec.continuity_step_px)
            loss = loss + spec.w_continuity * l_k
            parts["continuity"] = l_k.detach()
    parts["unlabeled_mean_p"] = pu.unlabeled_mean(logits, y, w)
    return (loss, parts) if return_parts else loss
