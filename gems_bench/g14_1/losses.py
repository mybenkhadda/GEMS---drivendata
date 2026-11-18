"""Selectable objectives (Phase E2) and per-term gradient diagnostics (Phase E1).

Every objective returns named TERMS and their WEIGHTS (never a pre-summed scalar), so diagnostics can report raw
magnitudes, weighted contributions, gradient norms per term and gradient conflicts (cosine between term gradients
with respect to the logits).

Targets (dict of (B, H, W) float tensors, see ``training.CorrectedDataset``):

``positive``      supervised fault pixels (target systems / training labels) inside the loss mask
``neg_weight``    weight of each pixel as a negative (0 = unlabelled / ignored)
``loss_mask``     pixels that may carry loss (context faults and other geography excluded)
``soft``          DTI kernel target max(1 - d / R, 0), R = 300 m in pixels (resolution-aware)
``reliable_neg``  negatives far from every visible fault

Distinguish (CORRECTED_EVALUATION.md): ``dti_surrogate`` here is a TRAINING SURROGATE on patches; it is not the
patch diagnostic, the region DTI nor the leaderboard DTI.

UNTESTED: written without execution.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..g14.losses import dti_surrogate as _g14_dti_surrogate
from ..g14.losses import soft_cldice

EPS = 1e-6
OBJECTIVES = ("bce", "weighted_bce", "bce_dice", "bce_tversky", "focal", "bce_focal_tversky", "distance_aware",
              "dti_surrogate", "cldice", "background_sparsity", "g14_original", "corrected_composite", "nnpu")


def _f(x):
    return x.float()


def _w(tg):
    return _f(tg["loss_mask"]) * (_f(tg["positive"]) + (1 - _f(tg["positive"])) * _f(tg["neg_weight"]))


def bce(logit, tg, pos_weight: float = 1.0):
    """Masked BCE; an optional per-pixel ``pos_w`` (M6 residual emphasis) multiplies the positive weight."""
    y = _f(tg["positive"])
    pw = pos_weight * (_f(tg["pos_w"]) if "pos_w" in tg else 1.0)
    w = _f(tg["loss_mask"]) * (y * pw + (1 - y) * _f(tg["neg_weight"]))
    l = F.binary_cross_entropy_with_logits(_f(logit), y, reduction="none")
    return (l * w).sum() / w.sum().clamp(min=1.0)


def soft_dice(prob, tg):
    w = _w(tg)
    y = _f(tg["positive"])
    inter = (prob * y * w).sum()
    return 1 - (2 * inter + EPS) / ((prob * w).sum() + (y * w).sum() + EPS)


def tversky_index(prob, tg, alpha=0.2, beta=0.8):
    w = _w(tg)
    y = _f(tg["positive"])
    tp, fp, fn = (prob * y * w).sum(), (prob * (1 - y) * w).sum(), ((1 - prob) * y * w).sum()
    return (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)


def focal(logit, tg, gamma: float = 2.0, alpha: float = 0.75):
    y = _f(tg["positive"])
    p = torch.sigmoid(_f(logit))
    ce = F.binary_cross_entropy_with_logits(_f(logit), y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    w = _w(tg)
    return (a * (1 - pt) ** gamma * ce * w).sum() / w.sum().clamp(min=1.0)


def distance_aware(logit, tg):
    """BCE against the resolution-aware DTI kernel target (credit decays linearly to 0 at 300 m)."""
    s = _f(tg["soft"])
    w = _f(tg["loss_mask"]) * torch.where(s > 0, torch.ones_like(s), _f(tg["neg_weight"]))
    l = F.binary_cross_entropy_with_logits(_f(logit), s, reduction="none")
    return (l * w).sum() / w.sum().clamp(min=1.0)


def background_sparsity(prob, tg):
    r = _f(tg["reliable_neg"])
    return (prob * r).sum() / r.sum().clamp(min=1.0)


def nnpu(logit, tg, prior: float, beta: float = 0.0, gamma: float = 1.0):
    """Non-negative PU risk (Kiryo et al., NeurIPS 2017) with the sigmoid loss: positives = supervised targets,
    unlabelled = pixels with neg_weight > 0. ``prior``: class prior of faults among unlabelled pixels."""
    m = _f(tg["loss_mask"])
    y = _f(tg["positive"]) * m
    u = (_f(tg["neg_weight"]) > 0).float() * m * (1 - y)
    z = _f(logit)
    lp, ln = torch.sigmoid(-z), torch.sigmoid(z)                 # loss for label +1 / -1
    n_p, n_u = y.sum().clamp(min=1.0), u.sum().clamp(min=1.0)
    rp = (lp * y).sum() / n_p
    rn = (ln * u).sum() / n_u - prior * (ln * y).sum() / n_p
    return prior * rp + (torch.clamp(rn, min=-beta) if gamma else rn)


# =============================================================================== registry
def objective(name: str, logit, tg, *, radius_px: float, pos_weight: float = 8.0, cldice_iters: int = 5,
              pu_prior: float = 0.01) -> tuple[dict, dict]:
    """(terms, weights) of one objective; total = sum(weights[k] * terms[k])."""
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; known {OBJECTIVES}")
    prob = torch.sigmoid(_f(logit))
    t, w = {}, {}

    def add(k, v, wt):
        t[k], w[k] = v, wt

    if name == "bce":
        add("bce", bce(logit, tg, 1.0), 1.0)
    elif name == "weighted_bce":
        add("bce_w", bce(logit, tg, pos_weight), 1.0)
    elif name == "bce_dice":
        add("bce_w", bce(logit, tg, pos_weight), 1.0)
        add("dice", soft_dice(prob, tg), 1.0)
    elif name == "bce_tversky":
        add("bce_w", bce(logit, tg, pos_weight), 1.0)
        add("tversky", 1 - tversky_index(prob, tg), 1.0)
    elif name == "focal":
        add("focal", focal(logit, tg), 1.0)
    elif name == "bce_focal_tversky":
        add("bce_w", bce(logit, tg, pos_weight), 1.0)
        add("focal_tversky", (1 - tversky_index(prob, tg)).clamp(min=0) ** 0.75, 1.0)
    elif name == "distance_aware":
        add("bce_w", bce(logit, tg, pos_weight), 0.5)
        add("distance", distance_aware(logit, tg), 0.5)
    elif name == "dti_surrogate":
        add("bce_w", bce(logit, tg, pos_weight), 0.5)
        add("dti_surrogate", _g14_dti_surrogate(prob, _f(tg["positive"]), _f(tg["loss_mask"]), radius_px), 1.0)
    elif name == "cldice":
        add("bce_w", bce(logit, tg, pos_weight), 1.0)
        v, ok = soft_cldice(prob, _f(tg["positive"]), _f(tg["loss_mask"]), cldice_iters)
        if ok:
            add("cldice", v, 0.5)
    elif name == "background_sparsity":
        add("bce_w", bce(logit, tg, pos_weight), 1.0)
        add("sparsity", background_sparsity(prob, tg), 1.0)
    elif name == "g14_original":                                  # G14 LossCfg default weights, same terms
        add("bce_w", bce(logit, tg, pos_weight), 0.35)
        add("tversky", 1 - tversky_index(prob, tg), 0.25)
        add("distance", distance_aware(logit, tg), 0.20)
        v, ok = soft_cldice(prob, _f(tg["positive"]), _f(tg["loss_mask"]), cldice_iters)
        if ok:
            add("cldice", v, 0.15)
        add("sparsity", background_sparsity(prob, tg), 0.05)
    elif name == "corrected_composite":
        # starting point only: replace by the objective that wins the discovery-validated ablation (M4)
        add("bce_w", bce(logit, tg, pos_weight), 1.0)
        add("tversky", 1 - tversky_index(prob, tg), 0.5)
        add("dti_surrogate", _g14_dti_surrogate(prob, _f(tg["positive"]), _f(tg["loss_mask"]), radius_px), 0.25)
    elif name == "nnpu":
        add("nnpu", nnpu(logit, tg, pu_prior), 1.0)
    return t, w


def total(terms: dict, weights: dict):
    out = None
    for k, v in terms.items():
        out = weights[k] * v if out is None else out + weights[k] * v
    return out


# =============================================================================== diagnostics
def diagnostics(terms: dict, weights: dict, logit, tg) -> dict:
    """Raw / weighted magnitudes, gradient norm of each weighted term w.r.t. the logits, pairwise cosine conflicts,
    supervision masses. Call on a batch whose ``logit`` requires grad (graph retained)."""
    out = {"positive_mass": float(_f(tg["positive"]).sum()), "negative_mass": float(_f(tg["neg_weight"]).sum()),
           "loss_px": float(_f(tg["loss_mask"]).sum()), "empty_positive_batch": bool(_f(tg["positive"]).sum() < 1),
           "reliable_negative_px": float(_f(tg["reliable_neg"]).sum())}
    grads = {}
    for k, v in terms.items():
        out[f"raw_{k}"] = float(v.detach())
        out[f"weighted_{k}"] = float(weights[k] * v.detach())
        if logit.requires_grad and v.requires_grad:
            g = torch.autograd.grad(weights[k] * v, logit, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                grads[k] = g.detach().flatten().float()
                out[f"gradnorm_{k}"] = float(grads[k].norm())
    ks = list(grads)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = grads[ks[i]], grads[ks[j]]
            out[f"cos_{ks[i]}__{ks[j]}"] = float((a @ b) / (a.norm() * b.norm() + 1e-12))
    return out


def kernel_radius_px(pixel_m: float, radius_m: float = 300.0) -> float:
    r = radius_m / pixel_m
    if not math.isfinite(r) or r <= 0:
        raise ValueError(f"bad pixel size {pixel_m}")
    return r
