"""Modular P7 losses:  L = lf Lfault + lc Lcenter + lo Lorient + ld Ldistance + lt Ltopology + le Lendpoint.

Every term is a weighted mean over pixels with positive loss weight (``w``: 1 on visible positives, the PU
background weight on unlabeled pixels, 0 on ignored / withheld / invalid pixels). Empty targets are safe:
* ``masked_mean`` divides by max(sum w, eps) and returns an exact 0 (still attached to the graph) for zero weight;
* Dice terms are computed over the whole micro-batch and are skipped (0) when the batch has no positive pixel;
* the orientation term uses only pixels with a defined direction (``ovalid``) and is 0 when there is none.
The topology term (soft-clDice on the fault probability) measures skeleton overlap inside the weighted support; it
rewards thin, centred responses but never adds a connectivity reward, so unrelated short faults are not pulled into
one structure. Weights come from the configuration only (``loss.weights``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    s = w.sum()
    if float(s) <= 0:
        return (x * 0.0).sum()
    return (x * w).sum() / s.clamp_min(EPS)


def bce(logit, y, w, pos_weight: float = 1.0):
    pw = torch.as_tensor(float(pos_weight), device=logit.device, dtype=torch.float32)
    l = F.binary_cross_entropy_with_logits(logit.float(), y.float(), reduction="none", pos_weight=pw)
    return masked_mean(l, w)


def focal(logit, y, w, gamma: float = 1.5, pos_weight: float = 1.0):
    logit, y = logit.float(), y.float()
    l = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = y * p + (1 - y) * (1 - p)
    a = y * float(pos_weight) + (1 - y)
    return masked_mean(a * (1 - pt).clamp_min(0) ** float(gamma) * l, w)


def soft_dice(p, y, w):
    """1 - 2 sum(p y w) / (sum(p w) + sum(y w)) over the batch; 0 when no weighted positive exists."""
    yw = (y.float() * w).sum()
    if float(yw) <= 0:
        return (p * 0.0).sum()
    inter = (p.float() * y.float() * w).sum()
    return 1.0 - (2.0 * inter + EPS) / ((p.float() * w).sum() + yw + EPS)


def axial_orientation(pred2, c, s, valid):
    """1 - cos(2 dt) between the predicted doubled-angle direction and the target: axial (t and t + 180 deg agree),
    in [0, 2]. ``pred2`` (B, 2, H, W) raw; normalised to a unit vector."""
    v = pred2.float()
    n = torch.sqrt((v ** 2).sum(1, keepdim=True) + EPS)
    u = v / n
    cos2d = u[:, :1] * c.float() + u[:, 1:2] * s.float()
    return masked_mean(1.0 - cos2d, valid.float())


def distance_loss(logit, target, w):
    return masked_mean(F.smooth_l1_loss(torch.sigmoid(logit.float()), target.float(), reduction="none", beta=0.1), w)


def _soft_erode(x):
    return -F.max_pool2d(-x, (3, 3), 1, 1)


def _soft_dilate(x):
    return F.max_pool2d(x, (3, 3), 1, 1)


def soft_skeleton(x, iters: int = 5):
    """Differentiable morphological skeleton (Shit et al., clDice)."""
    s = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(int(iters)):
        x = _soft_erode(x)
        d = F.relu(x - _soft_dilate(_soft_erode(x)))
        s = s + F.relu(d - s * d)
    return s


def soft_cldice(p, y, w, iters: int = 5):
    """1 - clDice inside the weighted support; 0 when the batch has no weighted positive."""
    y, p = y.float() * (w > 0), p.float() * (w > 0)
    if float(y.sum()) <= 0:
        return (p * 0.0).sum()
    sp, sy = soft_skeleton(p, iters), soft_skeleton(y, iters)
    tprec = ((sp * y).sum() + EPS) / (sp.sum() + EPS)
    tsens = ((sy * p).sum() + EPS) / (sy.sum() + EPS)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + EPS)


def compute(out: dict, b: dict, lc: dict) -> tuple[torch.Tensor, dict]:
    """Total loss + float parts. ``out``: head logits; ``b``: batch tensors (x, w, seg, kern, center, ocos, osin,
    ovalid, dist, ep, jn, topo_valid, valid)."""
    W = lc["weights"]
    w = b["w"]
    parts = {}
    total = out["seg"].new_zeros((), dtype=torch.float32)
    fc = lc["fault"]
    yf = torch.maximum(b["seg"], b["kern"]) if fc.get("kernel_target") else b["seg"]
    if fc["kind"] == "focal":
        lf = focal(out["seg"], (yf > 0.5).float() if not fc.get("kernel_target") else yf, w, float(fc["gamma"]),
                   float(fc["pos_weight"]))
    else:
        lf = bce(out["seg"], yf, w, float(fc["pos_weight"]))
    pf = torch.sigmoid(out["seg"].float())
    if float(fc.get("dice", 0)) > 0:
        lf = lf + float(fc["dice"]) * soft_dice(pf, b["seg"], w)
    parts["fault"] = lf
    total = total + float(W.get("fault", 1.0)) * lf
    if "center" in out and float(W.get("center", 0)) > 0:
        cc = lc["center"]
        l = bce(out["center"], b["center"], w, float(cc["pos_weight"]))
        if float(cc.get("dice", 0)) > 0:
            l = l + float(cc["dice"]) * soft_dice(torch.sigmoid(out["center"].float()), b["center"], w)
        parts["center"] = l
        total = total + float(W["center"]) * l
    if "orient" in out and float(W.get("orient", 0)) > 0:
        l = axial_orientation(out["orient"], b["ocos"], b["osin"], b["ovalid"] * (w > 0))
        parts["orient"] = l
        total = total + float(W["orient"]) * l
    if "dist" in out and float(W.get("distance", 0)) > 0:
        l = distance_loss(out["dist"], b["dist"], (w > 0).float() * b["valid"])
        parts["distance"] = l
        total = total + float(W["distance"]) * l
    if float(W.get("topology", 0)) > 0:
        l = soft_cldice(pf, b["seg"], w, int(lc["topology"].get("iters", 5)))
        parts["topology"] = l
        total = total + float(W["topology"]) * l
    if float(W.get("endpoint", 0)) > 0:
        g = float(lc["endpoint"].get("gamma", 2.0))
        tv = b["topo_valid"] * (w > 0)
        l = 0.0
        for k in ("ep", "jn"):
            if k in out:
                l = l + focal(out[k], b[k], tv, g, 1.0)
        parts["endpoint"] = l if torch.is_tensor(l) else total.new_zeros(())
        total = total + float(W["endpoint"]) * parts["endpoint"]
    unl = (b["seg"] < 0.5) & (b["w"] > 0)
    parts["unlabeled_mean_p"] = masked_mean(pf.detach(), unl.float())
    parts["total"] = total
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}
