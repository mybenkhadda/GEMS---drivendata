"""Known-fault shortcut audit (Phase D).

Static mechanism (docs/g14_1/STATIC_CODE_AUDIT.md, S01): in every G14 training unit the permitted known-fault map
contains every training positive, so the ``known:*`` input channels encode the target. This module measures it.

CPU-only, no model (R02 / R07):

* ``target_context_overlap``   share of training / inner-validation positives that are also context pixels
* ``copy_baseline``            region DTI of the trivial predictor "context raster itself" on the inner-validation
                               region: the inner score attainable by copying the input
* ``known_only_separability``  AUC of -distance-to-context for the unit's training targets (training pixels) versus
                               its held-out truth (held-out pixels)

With an existing checkpoint (R06, inference only): ``counterfactuals`` re-runs the trained G14 unit model with its
known-context channels intervened:

``original``        as trained
``disabled``        known-context availability mask = 0 (the modality encoder zeroes the modality)
``neutral``         known channels set to the unit's training median (normalised value 0) with availability 1
``shifted``         context map translated by ``shift_px`` (zero fill)
``subset``          a random ``keep`` share of context components kept
``target_removed``  inner-validation faults removed from the context (the honest inner discovery test)

These perturbations create distribution shift; a drop under perturbation is evidence of dependence, not by itself
proof of leakage. ``target_removed`` is the decisive comparison: it is the only intervention that matches what a
discovery target looks like at test time.

UNTESTED: written without execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

INTERVENTIONS = ("original", "disabled", "neutral", "shifted", "subset", "target_removed")


# =============================================================================== CPU-only
def target_context_overlap(unit) -> dict:
    """How much of the supervision is literally present in the input context."""
    tr = unit.labels & unit.train
    iv = unit.labels & unit.inner_val
    ctx = unit.permitted_known
    near = ndimage.binary_dilation(ctx, iterations=1) if ctx.any() else ctx
    return {"unit": unit.key, "train_pos_px": int(tr.sum()), "inner_pos_px": int(iv.sum()),
            "context_px": int(ctx.sum()),
            "train_pos_in_context": float((tr & ctx).sum() / max(tr.sum(), 1)),
            "train_pos_within_1px_context": float((tr & near).sum() / max(tr.sum(), 1)),
            "inner_pos_in_context": float((iv & ctx).sum() / max(iv.sum(), 1)),
            "heldout_truth_in_context": int((unit.heldout & unit.permitted_known & ~unit.train).sum()),
            "status": "SHORTCUT" if (tr & ctx).any() else "DISJOINT"}


def copy_baseline(unit, data, pixel_m=(100.0, 100.0)) -> pd.DataFrame:
    """Region DTI of predictors that only copy the context, on the unit's inner-validation region (truth = its
    faults) and held-out region (truth = held-out faults). If the copy reaches the G14 inner score, the inner metric
    measures copying."""
    from .metric_audit import region_components

    ctx = unit.permitted_known
    d = ndimage.distance_transform_edt(~ctx) if ctx.any() else np.full(ctx.shape, np.inf)
    preds = {"copy_binary": ctx.astype(np.float32), "copy_kernel": np.clip(1 - d / 3.0, 0, 1).astype(np.float32)}
    rows = []
    for region_name, region, truth in (("inner_val", unit.inner_val & data.valid, data.labels & unit.inner_val),
                                       ("heldout", unit.heldout & data.valid, data.labels & unit.heldout)):
        if not region.any():
            continue
        for k, p in preds.items():
            c = region_components(p, truth, region, None, pixel_m)
            rows.append({"unit": unit.key, "region": region_name, "predictor": k, "dti": c.dti, "tp_w": c.tp,
                         "fp_w": c.fp, "fn_w": c.fn, "n_truth": c.n_truth})
    return pd.DataFrame(rows)


def known_only_separability(unit, data, max_n: int = 1_000_000, seed: int = 0) -> dict:
    """AUC of a context-distance score for (a) training targets on training pixels, (b) held-out truth on held-out
    pixels. (a) >> (b) means the context explains the training labels far better than the test labels."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    ctx = unit.permitted_known
    d = ndimage.distance_transform_edt(~ctx) if ctx.any() else np.full(ctx.shape, 1e6)
    out = {"unit": unit.key}
    for name, region, truth in (("train", unit.train & data.valid, data.labels & unit.train),
                                ("heldout", unit.heldout & data.valid, data.labels & unit.heldout)):
        y, s = truth[region], -d[region]
        if y.size > max_n:
            i = rng.choice(y.size, max_n, replace=False)
            y, s = y[i], s[i]
        out[f"auc_{name}"] = float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else np.nan
    return out


# =============================================================================== checkpoint counterfactuals
def set_context(inputs, permitted: np.ndarray, data) -> None:
    """Recompute the known-context pyramid of a ``g14.multiscale.UnitInputs`` from another permitted map (same
    construction as ``UnitInputs.__init__``)."""
    from ..g14.contract import block_fraction, block_nanmean, known_context

    if not inputs.known_names:
        raise ValueError("this model has no known-context channels")
    kc = known_context(permitted, data.valid)
    for f in inputs.factors:
        inputs.known[f] = np.stack([block_nanmean(kc[n], f) for n in inputs.known_names]).astype(np.float32)
    inputs.known_mask = {f: block_fraction(data.valid, f) for f in inputs.factors}


def intervene(inputs, how: str, unit, data, shift_px: int = 20, keep: float = 0.5, seed: int = 0) -> np.ndarray:
    """Apply one intervention in place; returns the context map now in effect (for dependence curves)."""
    base = unit.permitted_known
    C = len(inputs.channels)
    if how == "original":
        set_context(inputs, base, data)
        return base
    if how == "disabled":
        set_context(inputs, base, data)
        inputs.known_mask = {f: np.zeros_like(m) for f, m in inputs.known_mask.items()}
        return np.zeros_like(base)
    if how == "neutral":
        set_context(inputs, base, data)
        med = inputs.norm.med[C:C + len(inputs.known_names)]
        for f in inputs.factors:
            inputs.known[f] = np.broadcast_to(med[:, None, None], inputs.known[f].shape).astype(np.float32).copy()
        return np.zeros_like(base)
    if how == "shifted":
        sh = np.zeros_like(base)
        H, W = base.shape
        s = shift_px
        sh[s:, s:] = base[:H - s, :W - s]
        sh &= data.valid
        set_context(inputs, sh, data)
        return sh
    if how == "subset":
        lab, n = ndimage.label(base, structure=np.ones((3, 3), bool))
        keep_ids = np.random.default_rng(seed).random(n + 1) < keep
        keep_ids[0] = False
        sub = keep_ids[lab]
        set_context(inputs, sub, data)
        return sub
    if how == "target_removed":
        sub = base & ~(data.labels & unit.inner_val)
        set_context(inputs, sub, data)
        return sub
    raise KeyError(how)


def counterfactuals(g14_cfg, stack, data, unit, exp_dir, interventions=INTERVENTIONS, pixel_m=(100.0, 100.0),
                    transform=None, log=print, save_dir=None) -> pd.DataFrame:
    """Inference-only interventions on one trained G14 unit. Scores the inner-validation region (truth = its faults)
    and the held-out region (truth = held-out faults) with official region components."""
    from ..g14.inference import predict_full
    from ..g14.train import load_unit_model
    from .metric_audit import region_components
    from .probability import distance_curve

    model, inputs, meta = load_unit_model(g14_cfg, stack, data, unit, exp_dir)
    rows, curves = [], []
    regions = (("inner_val", unit.inner_val & data.valid, data.labels & unit.inner_val),
               ("heldout", unit.heldout & data.valid, data.labels & unit.heldout))
    for how in interventions:
        ctx = intervene(inputs, how, unit, data)
        for rname, region, truth in regions:
            if not region.any():
                continue
            r = predict_full(model, inputs, g14_cfg, region, mc_samples=0, tta=False, log=log)
            p = np.nan_to_num(r["prob"])
            # discovery-style scoring of the inner region excludes the faults that are still in the context
            excl = (ctx & truth) if rname == "inner_val" else None
            for scoring, ex in (("all_truth", None), ("context_excluded", excl)):
                if scoring == "context_excluded" and (ex is None or not ex.any()):
                    continue
                c = region_components(p, truth, region, ex, pixel_m)
                rows.append({"unit": unit.key, "intervention": how, "region": rname, "scoring": scoring,
                             "dti": c.dti, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "n_truth": c.n_truth,
                             "pred_mass": c.pred_mass, "mean_p": float(p[region].mean())})
            if transform is not None:
                dc = distance_curve(np.where(region, p, np.nan), region, unit.permitted_known, transform)
                dc.insert(0, "region", rname)
                dc.insert(0, "intervention", how)
                dc.insert(0, "unit", unit.key)
                curves.append(dc)
            if save_dir is not None:
                from pathlib import Path

                Path(save_dir).mkdir(parents=True, exist_ok=True)
                np.save(Path(save_dir) / f"{unit.key}__{how}__{rname}.npy", r["prob"].astype(np.float16))
    set_context(inputs, unit.permitted_known, data)
    out = pd.DataFrame(rows)
    out.attrs["curves"] = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    out.attrs["manifest_channels"] = meta.get("channels", [])
    return out


def g14_config_for(source_experiment: str, overrides: dict | None = None):
    """Rebuild the G14 config whose semantic hash names ``source_experiment`` (``<experiment>__<hash>``).

    Raises with an actionable message when the hash differs (the original notebook OVERRIDES must be repeated:
    only semantic keys matter; paths, device and flags do not)."""
    from ..g14.config import G14Config
    from ..g14.experiments import REGISTRY, resolve

    name, want = source_experiment.rsplit("__", 1)
    base = G14Config.from_dict({"run_mode": "EVALUATE", "inference_enabled": True, "experiment": name,
                                **(overrides or {})})
    cfg = resolve(name, base) if name in REGISTRY else base
    got = cfg.semantic_hash()
    if got != want:
        raise ValueError(f"G14 config hash {got} != {want} of {source_experiment}: pass the semantic OVERRIDES used "
                         "for the original run (data / model / loss / train / split / eval sections) via "
                         "CONFIG['g14_overrides']")
    return cfg
