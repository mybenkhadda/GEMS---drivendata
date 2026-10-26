"""Section 07: the structured, staged training objective of GeoFaultNet.

    L1  positive supervision      focal BCE; mapped faults positive; unlabeled background is a WEAK negative (weight
                                  ``pu.unlabeled_weight`` x label-free background weight), never a confirmed one;
                                  pixels shown as map context carry ``context_pixel_weight``.
    L2  distance-aware coverage   soft Tversky with the competition's alpha = 0.2 (FP) / beta = 0.8 (FN) and a
                                  300 m kernel: TP support = max_r k(r) maxpool_r(p) at truth pixels, FP = p (1 - kern).
                                  A differentiable SURROGATE, not the official metric (the exact scorer evaluates).
    L3  skeleton connectivity     skeleton recall (recall of the label skeleton by p within 1 px) or soft-clDice.
    L4  orientation               doubled-angle cosine loss where orientation targets are reliable.
    L5  endpoint / junction       focal BCE on sparse Gaussian heatmaps inside the topology-valid mask.
    L6  reconstruction            BCE inside the neighbourhood of geometry REMOVED by the corruption operators,
                                  removed pixels weighted ``reconstruction_weight``.
    L7  consistency (optional)    EMA-teacher consistency on shifted views (losses.consistency), in train.py.

Curriculum (epoch fraction e):  A (e < stage_a_end) L1 + L6;  B (e < stage_b_end) + L4 + L5;  C + the enabled
``stage_c`` terms (L2, L3). Loss weights are never tuned jointly: Stage C terms enter only through the stage_c variant
after it beats core under the same protocol.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..losses.geometry import cl_dice, orientation_loss

KERNEL_PX = ((0, 1.0), (1, 2.0 / 3.0), (2, 1.0 / 3.0))          # k(d) = 1 - d / 300 m at 0, 100, 200 m


def stage_of(epoch: int, n_epochs: int, lc: dict) -> str:
    e = epoch / max(n_epochs, 1)
    cur = lc["curriculum"]
    return "A" if e < float(cur["stage_a_end"]) else "B" if e < float(cur["stage_b_end"]) else "C"


def active_terms(stage: str, lc: dict) -> dict:
    w = {k: float(v) for k, v in lc["weights"].items()}
    on = {"l1": True, "l6": True, "l4": stage in ("B", "C"), "l5": stage in ("B", "C"),
          "l2": stage == "C" and bool(lc["stage_c"].get("l2")), "l3": stage == "C" and bool(lc["stage_c"].get("l3"))}
    return {k: (w[k] if on[k] else 0.0) for k in ("l1", "l2", "l3", "l4", "l5", "l6")}


def _focal(logit, y, w, gamma: float, pos_weight: float):
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * y + (1 - p) * (1 - y)
    a = w * (1.0 + (pos_weight - 1.0) * y)
    return ((1 - pt).clamp_min(0) ** gamma * bce * a).sum() / a.sum().clamp_min(1.0)


def support(p):
    """TP support of every pixel: max_r k(r) max_{|o| <= r} p(x + o) (square neighbourhoods, 300 m kernel)."""
    s = p * KERNEL_PX[0][1]
    for r, k in KERNEL_PX[1:]:
        s = torch.maximum(s, k * F.max_pool2d(p, 2 * r + 1, 1, r))
    return s


def tversky_surrogate(p, y, kern, w, alpha: float, beta: float):
    sup = support(p)
    tw = y * (w > 0).float()
    tp = (sup * tw).sum()
    fn = ((1 - sup) * tw).sum()
    fp = (p * (1 - kern) * w).sum()
    return 1 - tp / (tp + alpha * fp + beta * fn + 1e-6)


def skeleton_recall(p, skel, w):
    sw = skel * (w > 0).float()
    return 1 - (F.max_pool2d(p, 3, 1, 1) * sw).sum() / sw.sum().clamp_min(1.0)


def compute(out: dict, b: dict, lc: dict, cc: dict, stage: str) -> tuple:
    """(total loss, {part: value}) for one batch (all tensors on the same device, float32)."""
    t = active_terms(stage, lc)
    seg = out["seg"].float()
    y, w = b["seg"], b["w"]
    parts = {}
    total = seg.new_zeros(())
    l1 = _focal(seg, y, w, float(lc["focal_gamma"]), float(lc["pos_weight"]))
    parts["l1"] = l1
    total = total + t["l1"] * l1
    p = torch.sigmoid(seg)
    if t["l2"] > 0:
        parts["l2"] = tversky_surrogate(p, y, b["kern"], w, float(lc["tversky"]["alpha"]), float(lc["tversky"]["beta"]))
        total = total + t["l2"] * parts["l2"]
    if t["l3"] > 0:
        parts["l3"] = cl_dice(p, y, b["kern"], w) if lc["l3_mode"] == "cldice" else skeleton_recall(p, b["skel"], w)
        total = total + t["l3"] * parts["l3"]
    if t["l4"] > 0:
        ov = b["ovalid"] * (w > 0).float()
        parts["l4"] = orientation_loss(out["orient"], b["ocos"], b["osin"], ov, mode="cosine")
        total = total + t["l4"] * parts["l4"]
    if t["l5"] > 0:
        tv = b["topo_valid"] * (w > 0).float()
        l5 = 0.0
        for h, tgt in (("ep", b["ep"]), ("jn", b["jn"])):
            lg = out[h].float()
            bce = F.binary_cross_entropy_with_logits(lg, tgt, reduction="none")
            q = torch.sigmoid(lg)
            a = tv * (1.0 + 9.0 * tgt)
            l5 = l5 + ((q - tgt).abs() ** 1.5 * bce * a).sum() / a.sum().clamp_min(1.0)
        parts["l5"] = l5 / 2
        total = total + t["l5"] * parts["l5"]
    rz = b["rec_zone"]
    if t["l6"] > 0 and float(rz.sum()) > 0:
        a = rz * (1.0 + (float(cc["reconstruction_weight"]) - 1.0) * b["rec"])
        bce = F.binary_cross_entropy_with_logits(seg, y, reduction="none")
        parts["l6"] = (bce * a).sum() / a.sum().clamp_min(1.0)
        total = total + t["l6"] * parts["l6"]
    with torch.no_grad():
        unl = (y < 0.5) & (w > 0)
        parts["unlabeled_mean_p"] = p[unl].mean() if bool(unl.any()) else p.new_zeros(())
        rem = b["rec"] > 0.5
        parts["removed_recall_p"] = p[rem].mean() if bool(rem.any()) else p.new_full((), float("nan"))
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}
