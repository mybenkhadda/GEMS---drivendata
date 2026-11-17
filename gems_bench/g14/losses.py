"""Composite loss framework (Phase 6).

Supervision masks (all (B, H, W), float in [0, 1], produced by ``multiscale.MultiScaleDataset``):

``loss_mask``     pixels of the unit's training region with valid data (everything else contributes nothing)
``trace``         visible mapped-fault pixels (positives)
``neg_weight``    pixels that may act as negatives: loss_mask, outside the ignore ring (<= 300 m of a visible
                  fault) and outside the discovery ignore zone. Unknown / unlabeled-near pixels get weight 0.
``reliable_neg``  negatives beyond ``reliable_negative_px`` (1 km): only these feed the background-sparsity term
``soft``          DTI kernel target max(1 - d / 3 px, 0)
``dist_px``       distance to the nearest visible fault (pixels)

All reductions are explicit (sum of weighted terms / sum of weights, clamped denominators) and computed in float32
for mixed-precision safety. Patches without positives are handled: Tversky and clDice are aggregated over the whole
batch (a positive-free patch only adds false-positive mass), and clDice is skipped when the batch has no positive.

The DTI surrogate mirrors the official construction (``gems_bench.metric.dti_components``: TP = local weighted max
of the prediction over each truth pixel, FP = p * (1 - kernel), FN = n - TP) on a differentiable soft prediction.
It is a TRAINING SURROGATE, not the competition metric: evaluation always uses the official scorer.

UNTESTED: written without execution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def _f(x):
    return x.float()


# =============================================================================== core terms
def masked_bce(logit, target, pos_mask, neg_weight, loss_mask, pos_weight: float):
    """Weighted BCE: positives (visible traces inside the loss mask) x pos_weight, negatives x neg_weight."""
    logit, target = _f(logit), _f(target)
    w = loss_mask * (pos_mask * pos_weight + (1 - pos_mask) * neg_weight)
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (l * w).sum() / w.sum().clamp(min=1.0)


def soft_tversky(prob, target, weight, alpha: float, beta: float):
    """1 - TP / (TP + alpha FP + beta FN) over the batch (DTI uses alpha 0.2, beta 0.8)."""
    p, t, w = _f(prob), _f(target), _f(weight)
    tp = (p * t * w).sum()
    fp = (p * (1 - t) * w).sum()
    fn = ((1 - p) * t * w).sum()
    return 1 - (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)


def distance_centerline(logit, soft, loss_mask, neg_weight):
    """BCE against the DTI kernel target: full credit on the trace, partial within 300 m, so a prediction one pixel
    off the mapped trace is penalised less than one far away (matches the metric's spatial tolerance)."""
    logit, soft = _f(logit), _f(soft)
    w = loss_mask * torch.where(soft > 0, torch.ones_like(soft), neg_weight)
    l = F.binary_cross_entropy_with_logits(logit, soft, reduction="none")
    return (l * w).sum() / w.sum().clamp(min=1.0)


def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), (1, 1), (1, 0)), -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1)))


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters: int):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021): iterative soft morphological opening residuals.
    Cost O(iters) pooling ops: cheap at 128 px."""
    x = _f(x)
    skel = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(prob, target, weight, iters: int):
    """1 - clDice. Returns (loss, valid flag); invalid (0, False) when the batch has no positive in the mask."""
    p, t, w = _f(prob) * _f(weight), _f(target) * _f(weight)
    if t.sum() < 1:
        return p.new_zeros(()), False
    sp, st = soft_skeleton(p, iters), soft_skeleton(t, iters)
    tprec = ((sp * t).sum() + EPS) / (sp.sum() + EPS)
    tsens = ((st * p).sum() + EPS) / (st.sum() + EPS)
    return 1 - 2 * tprec * tsens / (tprec + tsens), True


def background_sparsity(prob, reliable_neg):
    """Mean probability mass on reliable negatives (> 1 km from any visible fault): the DTI false-positive term."""
    p, r = _f(prob), _f(reliable_neg)
    return (p * r).sum() / r.sum().clamp(min=1.0)


def orientation_consistency(pred_cs, cos2, sin2, weight):
    """1 - cos(2 dtheta) on pixels near visible faults (axial: 0 and 180 deg identical)."""
    c, s = _f(pred_cs[:, 0]), _f(pred_cs[:, 1])
    w = _f(weight)
    return ((1 - (c * _f(cos2) + s * _f(sin2))) * w).sum() / w.sum().clamp(min=1.0)


def connectivity_loss(logits, endpoint, junction, loss_mask):
    l = (F.binary_cross_entropy_with_logits(_f(logits[:, 0]), _f(endpoint), reduction="none", pos_weight=
                                            torch.tensor(10.0, device=logits.device))
         + F.binary_cross_entropy_with_logits(_f(logits[:, 1]), _f(junction), reduction="none", pos_weight=
                                              torch.tensor(10.0, device=logits.device)))
    m = _f(loss_mask)
    return (l * m).sum() / m.sum().clamp(min=1.0)


def dti_surrogate(prob, trace, loss_mask, radius_px: float = 3.0, alpha: float = 0.2, beta: float = 0.8):
    """1 - soft DTI over the batch, mirroring the official construction on a soft prediction.

    TP = sum over truth pixels of max_offsets k(offset) * p(pixel + offset), k = 1 - |offset| / radius (offsets with
    k > 0); FP = sum p * (1 - k_truth) with k_truth = max(1 - d / radius, 0) from the nearest truth pixel;
    FN = n_truth - TP. Unlike the official scorer this is computed per patch window (no cross-patch neighbours)."""
    p = _f(prob) * _f(loss_mask)
    t = _f(trace) * _f(loss_mask)
    r = int(radius_px)
    pad = F.pad(p.unsqueeze(1), (r, r, r, r)).squeeze(1)
    H, W = p.shape[-2:]
    M = torch.zeros_like(p)
    kd = torch.zeros_like(t)
    tpad = F.pad(t.unsqueeze(1), (r, r, r, r)).squeeze(1)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = 1.0 - (dy * dy + dx * dx) ** 0.5 / radius_px
            if k <= 0:
                continue
            sh = pad[..., r + dy:r + dy + H, r + dx:r + dx + W]
            M = torch.maximum(M, k * sh)
            kd = torch.maximum(kd, k * tpad[..., r + dy:r + dy + H, r + dx:r + dx + W])
    tp = (M * t).sum()
    fp = (p * (1 - kd)).sum()
    fn = t.sum() - tp
    return 1 - tp / (tp + alpha * fp + beta * fn + EPS)


# =============================================================================== composite
class CompositeLoss:
    """Weighted sum of the enabled terms; returns (total, per-term float dict for logging)."""

    def __init__(self, cfg):
        self.cfg = cfg.loss
        self.heads = cfg.model.heads
        self.unc = cfg.model.uncertainty

    def __call__(self, out: dict, tg: dict):
        L = self.cfg
        mask = _f(tg["loss_mask"])
        trace = _f(tg["trace"]) * mask
        negw = _f(tg["neg_weight"])
        logit = out["fault"][:, 0]
        if self.unc == "heteroscedastic" and "logvar" in out:
            # Kendall & Gal (2017): average the BCE over logits sampled with the predicted aleatoric variance
            std = torch.exp(0.5 * _f(out["logvar"][:, 0]))
            samples = [_f(logit) + std * torch.randn_like(std) for _ in range(4)]
        else:
            samples = [logit]
        prob = torch.sigmoid(_f(logit))
        terms = {}
        w = L.weights
        if w.get("masked_bce", 0):
            terms["masked_bce"] = sum(masked_bce(s, trace, trace, negw, mask, L.pos_weight) for s in samples) / len(samples)
        if w.get("soft_tversky", 0):
            terms["soft_tversky"] = soft_tversky(prob, trace, mask * torch.maximum(trace, negw), L.tversky_alpha,
                                                 L.tversky_beta)
        if w.get("distance_centerline", 0):
            terms["distance_centerline"] = distance_centerline(logit, tg["soft"], mask, negw)
        if w.get("soft_cldice", 0):
            v, ok = soft_cldice(prob, trace, mask, L.cldice_iters)
            if ok:
                terms["soft_cldice"] = v
        if w.get("background_sparsity", 0):
            terms["background_sparsity"] = background_sparsity(prob, tg["reliable_neg"])
        total = sum(w[k] * v for k, v in terms.items())
        a = L.aux_weights
        if a.get("orientation", 0) and "orientation" in out:
            terms["orientation"] = orientation_consistency(out["orientation"], tg["cos2"], tg["sin2"], tg["orient_w"])
            total = total + a["orientation"] * terms["orientation"]
        if a.get("connectivity", 0) and "connectivity" in out:
            terms["connectivity"] = connectivity_loss(out["connectivity"], tg["endpoint"], tg["junction"], mask)
            total = total + a["connectivity"] * terms["connectivity"]
        if a.get("dti_surrogate", 0):
            terms["dti_surrogate"] = dti_surrogate(prob, trace, mask, L.kernel_radius_px, L.tversky_alpha, L.tversky_beta)
            total = total + a["dti_surrogate"] * terms["dti_surrogate"]
        if a.get("deep_supervision", 0) and out.get("aux"):
            ds = sum(masked_bce(x[:, 0], trace, trace, negw, mask, L.pos_weight) for x in out["aux"]) / len(out["aux"])
            terms["deep_supervision"] = ds
            total = total + a["deep_supervision"] * ds
        logs = {k: float(v.detach()) for k, v in terms.items()}
        logs["total"] = float(total.detach())
        return total, logs
