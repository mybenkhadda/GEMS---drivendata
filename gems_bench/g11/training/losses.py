"""Masked segmentation losses (torch). Every loss takes ``valid`` (1 = scored pixel) so embargo / other-split /
nodata pixels never contribute. Shapes: logits, target, valid = (B, 1, H, W)."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _mean(x, valid):
    return (x * valid).sum() / valid.sum().clamp_min(1.0)


def weighted_bce_loss(logits, target, valid, pos_weight: float = 5.0):
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    return _mean(l, valid)


def focal_loss(logits, target, valid, gamma: float = 2.0, alpha: float = 0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    return _mean(a * (1 - pt) ** gamma * ce, valid)


def soft_dice_loss(logits, target, valid, eps: float = 1.0):
    p = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (p * t).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + t.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def boundary_loss(logits, target, valid, distance_m, sigma_m: float = 300.0):
    """BCE re-weighted by exp(-d / sigma) where d = distance to the nearest fault (training-label derived TARGET,
    never an input): concentrates the loss on the fault neighbourhood without hard buffers."""
    w = torch.exp(-distance_m.clamp_min(0) / sigma_m)
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _mean(l * (0.25 + w), valid)


def positive_unlabeled_loss(logits, target, valid, prior: float = 0.02):
    """Non-negative PU risk (Kiryo et al. 2017): unlabeled pixels are not negatives."""
    pos = (target > 0.5).float() * valid
    unl = (target <= 0.5).float() * valid
    sp = F.softplus(-logits)                        # loss for predicting positive
    sn = F.softplus(logits)                         # loss for predicting negative
    npos, nunl = pos.sum().clamp_min(1), unl.sum().clamp_min(1)
    rp = (sp * pos).sum() / npos
    rn = (sn * unl).sum() / nunl - prior * (sn * pos).sum() / npos
    return prior * rp + torch.clamp(rn, min=0)


def build_loss(name: str, pos_weight: float = 5.0):
    """Objective factory for the controlled ablation (training.loss)."""
    if name == "weighted_bce":
        return lambda o, b: weighted_bce_loss(o["logits"], b["y"], b["valid"], pos_weight)
    if name == "bce_dice":
        return lambda o, b: weighted_bce_loss(o["logits"], b["y"], b["valid"], pos_weight) + \
            soft_dice_loss(o["logits"], b["y"], b["valid"])
    if name == "focal":
        return lambda o, b: focal_loss(o["logits"], b["y"], b["valid"])
    if name == "bce_boundary":
        return lambda o, b: weighted_bce_loss(o["logits"], b["y"], b["valid"], pos_weight) + \
            boundary_loss(o["logits"], b["y"], b["valid"], b["distance"])
    raise ValueError(f"unknown loss {name}")


def masked_multitask_loss(out: dict, batch: dict, main, weights: dict | None = None):
    """main(out, batch) + optional auxiliary orientation (axial cosine, on fault pixels) and distance (log-L1)."""
    w = {"orientation": 0.1, "distance": 0.05, **(weights or {})}
    total = main(out, batch)
    parts = {"main": float(total.detach())}
    if "orientation" in out and "orientation" in batch:
        m = batch["valid"] * (batch["y"] > 0.5).float()
        cos = (out["orientation"] * batch["orientation"]).sum(1, keepdim=True)
        lo = _mean(1 - cos, m)
        total = total + w["orientation"] * lo
        parts["orientation"] = float(lo.detach())
    if "distance" in out and "distance" in batch:
        ld = _mean(torch.abs(torch.log1p(out["distance"]) - torch.log1p(batch["distance"] / 100.0)), batch["valid"])
        total = total + w["distance"] * ld
        parts["distance"] = float(ld.detach())
    return total, parts
