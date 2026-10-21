"""Geometry-aware supervision for thin, elongated, oriented fault traces (Phase-5 Experiment C).

Heads (decoder output, training-only; the submission stays the single ``seg`` raster):
``seg``     fault-centreline confidence (logits)
``dist``    proximity field 1 - min(d / d_clip, 1) (sigmoid), as Phase-4 ``aux_dist``
``orient``  axial strike as (cos 2θ, sin 2θ) (tanh), supervised only where the direction is defined
            (``geology.skeleton.orientation_targets``: no intersections, no very short pieces, coherent)

Losses (each weighted from the spec; 0 disables)
-------------------------------------------------
``orientation_loss``   "cosine": 1 - <o, t> / |o| (axial: only the doubled angle enters, so θ and θ+180°
                       agree); "mse": the Phase-4 objective, kept for parity.
``cl_dice``            soft-clDice (Shit et al., CVPR 2021) with the metric's 300 m tolerance: Tsens = label
                       centreline covered by the dilated prediction, Tprec = predicted soft skeleton within
                       the tolerance volume. NOTE: Tprec treats unmapped faults as errors (negative-label
                       bias, see losses.pu) - use with the PU weights or keep its weight small.
``thinness``           mass of p off its own soft skeleton / mass of p. Label-free: favours line-like
                       outputs without assuming anything about unmapped areas.
``continuity``         |p(x) - p(x +- δ t)| along the *predicted* strike t (detached), weighted by p:
                       continuity along strike only, over δ ~ 1-2 px, so it cannot bridge separate systems
                       (unlike isotropic smoothness / CRF-style terms).
"""
from __future__ import annotations


def _soft_erode(x):
    import torch
    import torch.nn.functional as F

    p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate(x):
    import torch.nn.functional as F

    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters: int = 5):
    """Differentiable morphological skeleton (soft-clDice recipe) of a probability map in [0, 1]."""
    import torch.nn.functional as F

    img1 = _soft_dilate(_soft_erode(x))
    skel = F.relu(x - img1)
    for _ in range(iters):
        x = _soft_erode(x)
        img1 = _soft_dilate(_soft_erode(x))
        delta = F.relu(x - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def orientation_loss(orient_logits, ocos, osin, valid, mode: str = "cosine"):
    import torch

    o = torch.tanh(orient_logits.float())
    t = torch.cat([ocos, osin], 1)
    if mode == "mse":
        per = ((o - t) ** 2).sum(1, keepdim=True)
    else:
        per = 1 - (o * t).sum(1, keepdim=True) / (o.norm(dim=1, keepdim=True) + 1e-6)
    return (per * valid).sum() / valid.sum().clamp_min(1.0)


def proximity_loss(dist_logits, dist_target, weight):
    import torch

    return (torch.abs(torch.sigmoid(dist_logits.float()) - dist_target) * weight).sum() / weight.sum().clamp_min(1.0)


def cl_dice(p, truth, kern, weight, iters: int = 5, tol_px: int = 3):
    import torch.nn.functional as F

    sk_p = soft_skeleton(p, iters)
    vol_l = (kern > 0).float()
    dil_p = F.max_pool2d(p, 2 * tol_px + 1, 1, tol_px)
    tw = truth * weight
    tprec = (sk_p * vol_l * weight).sum() / (sk_p * weight).sum().clamp_min(1e-6)
    tsens = (tw * dil_p).sum() / tw.sum().clamp_min(1e-6)
    return 1 - 2 * tprec * tsens / (tprec + tsens).clamp_min(1e-6)


def thinness(p, weight, iters: int = 5):
    import torch.nn.functional as F

    sk = soft_skeleton(p, iters)
    return (F.relu(p - _soft_dilate(sk)) * weight).sum() / (p * weight).sum().clamp_min(1e-6)


def continuity(p, orient_logits, weight, step_px: float = 1.0):
    """Along-strike consistency using the predicted (detached) strike field."""
    import torch
    import torch.nn.functional as F

    B, _, H, W = p.shape
    o = torch.tanh(orient_logits.detach().float())
    th = 0.5 * torch.atan2(o[:, 1], o[:, 0])                      # strike (rad, clockwise from north)
    dr, dc = -torch.cos(th) * step_px, torch.sin(th) * step_px     # along-strike step (rows, cols)
    yy, xx = torch.meshgrid(torch.arange(H, device=p.device, dtype=p.dtype),
                            torch.arange(W, device=p.device, dtype=p.dtype), indexing="ij")
    loss = 0.0
    wd = p.detach() * weight
    for sgn in (1.0, -1.0):
        gy = (yy[None] + sgn * dr) / max(H - 1, 1) * 2 - 1
        gx = (xx[None] + sgn * dc) / max(W - 1, 1) * 2 - 1
        q = F.grid_sample(p, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)
        loss = loss + (torch.abs(p - q) * wd).sum() / wd.sum().clamp_min(1e-6)
    return loss / 2
