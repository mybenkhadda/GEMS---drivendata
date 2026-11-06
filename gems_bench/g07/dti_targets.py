"""Optional, DTI-inspired learning targets and losses for FUTURE supervised research.

Not called by FAST_COARSE. Never uses withheld labels. Inputs must be constructed
from training-visible fault labels and a training-validity mask only.
No surrogate is guaranteed to reproduce official 300m DTI.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def visible_fault_soft_target(visible_labels, training_mask, *, pixel_m=100.0,
                              radius_m=300.0) -> tuple[np.ndarray, np.ndarray]:
    """Finite-support, triangular soft label computed from TRAINING-VISIBLE faults.

    Outside training_mask targets are not supervised; they must be masked out
    in the loss. A validation or withheld label must never be provided here.
    """
    allowed = np.asarray(training_mask, dtype=bool)
    fault = np.asarray(visible_labels, dtype=bool) & allowed
    if not fault.any():
        target = np.zeros(allowed.shape, np.float32)
    else:
        d = distance_transform_edt(~fault, sampling=float(pixel_m))
        target = np.clip(1.0 - d / float(radius_m), 0.0, 1.0).astype(np.float32)
    target[~allowed] = 0
    return target, allowed


def dti_inspired_loss(logits, target, train_mask, *,
                      bce_weight=1.0, tversky_weight=0.5,
                      focal_weight=0.0, focal_gamma=1.33,
                      alpha=0.2, beta=0.8, eps=1e-6):
    """Masked BCE + soft Tversky (+ optional focal Tversky).

    torch is imported only when this optional function is called.
    alpha=.2 beta=.8 mirror the *relative* competition penalties; this is
    not the canonical spatial/distance-weighted DTI scorer.
    The 300m distance support is represented by visible_fault_soft_target.
    """
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    t = torch.as_tensor(target, dtype=p.dtype, device=p.device)
    mask = torch.as_tensor(train_mask, dtype=p.dtype, device=p.device)
    if p.shape != t.shape or p.shape != mask.shape:
        raise ValueError("logits, target and training mask shapes must match")
    if mask.sum().item() < 1:
        raise ValueError("No training-visible pixels in supervised loss")

    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * mask).sum() / mask.sum()
    tp = (p * t * mask).sum()
    fp = (p * (1-t) * mask).sum()
    fn = ((1-p) * t * mask).sum()
    tversky = 1 - (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    focal = tversky.pow(focal_gamma)
    return bce_weight*bce + tversky_weight*tversky + focal_weight*focal
