"""Training losses.

nnPU (Kiryo et al. 2017) with the logistic loss, per batch, over weighted pixels:
    R+ = mean_{labelled} softplus(-z)        R+- = mean_{labelled} softplus(z)
    Ru- = mean_{unlabelled, w>0} softplus(z)
    neg = Ru- - pi R+-
    loss = pi R+ + neg            if neg >= 0
           -neg                   otherwise (gradient step that pushes the negative risk back to >= 0)
pi = the unit's prior (visible-fault share x inflation, from training pixels only).

Soft Tversky: t = max(1 - d / 300 m, 0) from visible faults, alpha 0.2 (FP), beta 0.8 (FN), over w > 0.
Proximity: Smooth L1 on min(d, 300) / 300 (within 1 km of visible faults). Orientation: 1 - cos of the angle
between doubled-angle vectors, within 300 m where the mapped trace is coherent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nnpu(logit: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor, prior: float) -> tuple[torch.Tensor, dict]:
    z = logit.float()
    w = weight.float()
    p = (pos > 0.5).float() * w
    u = (pos <= 0.5).float() * w
    n_p, n_u = p.sum(), u.sum()
    if n_p < 1:
        r_u = (F.softplus(z) * u).sum() / n_u.clamp_min(1.0)
        return r_u, {"nnpu_pos": 0.0, "nnpu_neg": float(r_u), "nnpu_corrected": 0.0}
    r_pp = (F.softplus(-z) * p).sum() / n_p
    r_pn = (F.softplus(z) * p).sum() / n_p
    r_un = (F.softplus(z) * u).sum() / n_u.clamp_min(1.0)
    neg = r_un - prior * r_pn
    corrected = bool(neg.detach() < 0)
    loss = -neg if corrected else prior * r_pp + neg
    return loss, {"nnpu_pos": float(prior * r_pp), "nnpu_neg": float(neg), "nnpu_corrected": float(corrected)}


def soft_tversky(logit, soft, weight, alpha: float, beta: float, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logit.float())
    t = soft.float()
    w = weight.float()
    tp = (w * p * t).sum()
    fp = (w * p * (1 - t)).sum()
    fn = (w * (1 - p) * t).sum()
    return 1.0 - tp / (tp + alpha * fp + beta * fn + eps)


def proximity(pred, target, w) -> torch.Tensor:
    w = w.float()
    if w.sum() < 1:
        return pred.sum() * 0.0
    return (F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=0.1) * w).sum() / w.sum()


def orientation(pred, target, w) -> torch.Tensor:
    w = w.float()
    if w.sum() < 1:
        return pred.sum() * 0.0
    cos = F.cosine_similarity(pred.float(), target.float(), dim=1, eps=1e-6).unsqueeze(1)
    return ((1.0 - cos) * w).sum() / w.sum()


def total(out: dict, b: dict, prior: float, cfg: dict) -> tuple[torch.Tensor, dict]:
    L = cfg["training"]["loss"]
    l_pu, info = nnpu(out["logit"], b["pos"], b["weight"], prior)
    l_tv = soft_tversky(out["logit"], b["soft"], b["weight"], L["tversky_alpha"], L["tversky_beta"])
    l_px = proximity(out["proximity"], b["prox"], b["prox_w"])
    l_or = orientation(out["orientation"], b["orient"], b["orient_w"])
    loss = L["nnpu"] * l_pu + L["tversky"] * l_tv + L["proximity"] * l_px + L["orientation"] * l_or
    info.update(loss=float(loss), nnpu=float(l_pu), tversky=float(l_tv), proximity=float(l_px),
                orientation=float(l_or))
    return loss, info
