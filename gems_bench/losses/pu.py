"""Positive-unlabeled supervision for incomplete fault maps (Phase-5 Experiment B).

Mapped fault pixels are positives P. Every other training pixel is *unlabeled* U = (true negatives N) +
(unmapped faults H). The Phase-3/4 objective (B0, ``cnn3.dti_surrogate``) treats U as N: it pays a false-
positive cost for every confident prediction on an unmapped fault, which is the negative-label bias this
experiment tests.

B1  ``background_weights``: label-free downweighting of U where independent structural evidence is strong.
B2  ``nnpu_dti_surrogate``: the Tversky surrogate with a PU-corrected FP term plus a non-negative PU BCE.

Assumptions (explicit, and partly violated - see docs/VALIDATION_PROTOCOL.md section B)
-----------------------------------------------------------------------------------------
* Prior ``pi`` = P(fault | unlabeled pixel). Not identifiable from the data; it is a fixed hyper-parameter
  swept as an ablation (default grid 0.5x, 1x, 2x of 0.01 = roughly "the map is half complete").
* SCAR ("selected completely at random"): hidden positives look like mapped ones, so E_H[f] ~ E_P[f]. For
  faults this is *not* true: mapped faults are preferentially long, terrain-expressed, accessible ones
  (missing-not-at-random). E_P[p] then overestimates E_H[p], the correction overestimates the hidden mass
  and over-forgives FP. The non-negative clamp, the prior sweep and matched-area evaluation are the
  safeguards; PU here is a principled reweighting, not a guarantee.
* Sums, not means, enter the Tversky surrogate: sum_U f = sum_N f + sum_H f ~ sum_N f + pi |U| E_P[f], so
  the corrected FP mass is ``max(0, FP_U - pi * |U|_w * mean_P(p))`` with the same (1 - k) weighting.

Never derive B1 weights from a model trained on the evaluated labels (self-confirmation); only label-free
maps (``LABEL_FREE_MAPS``) or input features are accepted.
"""
from __future__ import annotations

import numpy as np

LABEL_FREE_MAPS = ("B0_lineament", "B0_terrain")


# ---------------------------------------------------------------------------
# B1: label-free background weights (numpy, built once per fold)
# ---------------------------------------------------------------------------
def background_weights(evidence: np.ndarray, valid: np.ndarray, *, lam: float = 0.5, q0: float = 0.8,
                       w_min: float = 0.2, seed: int = 2026) -> np.ndarray:
    """w_U = max(w_min, 1 - lam * clip((rank(E) - q0) / (1 - q0), 0, 1)); 1 where evidence is weak.

    ``evidence`` must be label-free (an input feature or a LABEL_FREE_MAPS map). Positives are not touched
    here: ``compose.build_targets5`` multiplies this map into the weight only on unlabeled pixels.
    """
    from ..rs_features import prank

    r = prank(np.where(valid, evidence, np.nan), valid, seed=seed)
    s = np.clip((np.nan_to_num(r) - q0) / max(1 - q0, 1e-6), 0, 1)
    return np.maximum(w_min, 1 - lam * s).astype(np.float32)


def resolve_evidence(ctx, source: str) -> np.ndarray:
    """``map:<name>`` (label-free baseline map) or ``feature:<store name>``; anything else is refused."""
    kind, _, name = source.partition(":")
    if kind == "map":
        if name not in LABEL_FREE_MAPS:
            raise ValueError(f"B1 evidence map {name!r} is not label-free (allowed: {LABEL_FREE_MAPS}); "
                             "weights derived from a supervised model would confirm its own errors")
        return ctx.oof(name)
    if kind == "feature":
        if name not in ctx.store.specs:
            raise KeyError(f"unknown feature {name}")
        if not ctx.store.has(name):
            ctx.store.build([name], verbose=False)
        return np.asarray(ctx.store.get(name), np.float32)
    raise ValueError(f"evidence source must be 'map:<name>' or 'feature:<name>', got {source!r}")


# ---------------------------------------------------------------------------
# B2: PU-corrected DTI surrogate (torch)
# ---------------------------------------------------------------------------
def nnpu_dti_surrogate(logits, truth, kern, weight, prior: float, alpha: float = 0.2, beta: float = 0.8,
                       bce_weight: float = 0.1, pixel_m=(100.0, 100.0), return_parts: bool = False):
    """Tversky surrogate (as cnn3.dti_surrogate) with FP_N = max(0, FP_U - pi |U|_w mean_P p) and nnPU BCE.

    ``truth`` 1 on visible positives, ``weight`` 0 on ignore bands / withheld systems / outside the zone.
    """
    import torch
    import torch.nn.functional as F

    from ..p3.cnn3 import _offsets

    p = torch.sigmoid(logits)
    offs, r = _offsets(tuple(pixel_m))
    H, W = p.shape[-2:]
    P = F.pad(p, (r, r, r, r))
    M = None
    for dr, dc, k in offs:
        t = k * P[..., r + dr:r + dr + H, r + dc:r + dc + W]
        M = t if M is None else torch.maximum(M, t)
    pos = truth * weight
    unl = (1 - truth) * weight
    n = pos.sum()
    tp = (M * pos).sum()
    mean_p_pos = (p * pos).sum() / n.clamp_min(1.0)
    u_mass = ((1 - kern) * unl).sum()
    fp_u = (p * (1 - kern) * unl).sum()
    fp = torch.clamp_min(fp_u - prior * u_mass * mean_p_pos, 0.0)
    loss = 1 - (tp + 1) / (tp + alpha * fp + beta * (n - tp) + 1)
    parts = {"fp_u": fp_u.detach(), "fp_corrected": fp.detach(), "tp": tp.detach()}
    if bce_weight:
        sp_neg, sp_pos = F.softplus(logits), F.softplus(-logits)      # loss(f, -1), loss(f, +1)
        nu = unl.sum().clamp_min(1.0)
        r_pos = (sp_pos * pos).sum() / n.clamp_min(1.0)
        r_neg = ((sp_neg * unl).sum() / nu - prior * (sp_neg * pos).sum() / n.clamp_min(1.0)) / (1 - prior)
        bce = r_pos + torch.clamp_min(r_neg, 0.0)
        loss = loss + bce_weight * bce
        parts["neg_risk"] = r_neg.detach()
    return (loss, parts) if return_parts else loss


def unlabeled_mean(logits, truth, weight):
    """Mean predicted probability on weighted unlabeled pixels (collapse monitor; detached)."""
    import torch

    with torch.no_grad():
        unl = (1 - truth) * weight
        return (torch.sigmoid(logits.float()) * unl).sum() / unl.sum().clamp_min(1.0)
