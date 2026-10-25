"""Static GPU-memory and GPU-time budgeting (no allocation, no benchmark: estimates from shapes and history).

Memory model for one Phase-5 CNN training process on the native grid (H x W = 3730 x 3292 by default):

    input stack     C x H x W x 2 B (fp16)      29 channels -> ~0.66 GiB; one float32 layer = ~47 MiB
    aux targets     A x H x W x 4 B             target / kern / weight (+ dist, orient) layers, float32
    context stack   C x (H/f) x (W/f) x 2 B     only when the model has a context branch
    model + optim   P x 4 B x 4                 weights, gradients, AdamW first / second moments (fp32)
    activations     batch x per-sample          per-sample GiB as in training.gpu.autoconfig (~0.4 on P4_B_gated)
    inference       infer_batch x (tile activations, no gradients) ~ 1/3 of training activations
    allocator slack 15 %

``placement_overrides`` keeps the stack (and aux layers) on the GPU only when everything fits in
``max_vram_frac`` of the card; otherwise the stack stays on the host and patches move per batch (slower, same
results: ``data_on_gpu`` is excluded from the spec hash). Every number is an *estimate* - the measured peaks
are in each run's ``metrics/gpu_memory_profile.csv``.

GPU time: ``estimate_unit_minutes`` uses the recorded mean epoch time of comparable runs (training_history
``epoch_s``) when available, else the ~40 s / 150-step T4 epoch seen in the first advanced run.
"""
from __future__ import annotations

from dataclasses import dataclass

GRID_PX = 3730 * 3292
T4_EPOCH_S_PER_STEP = 40.0 / 150                 # first advanced run: ~39-42 s per 150-step epoch on a T4


@dataclass(frozen=True)
class MemoryEstimate:
    stack_gib: float
    aux_gib: float
    context_gib: float
    model_optim_gib: float
    activations_gib: float
    inference_gib: float
    total_gpu_resident_gib: float
    total_host_resident_gib: float

    def as_dict(self) -> dict:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


def estimate_memory(n_channels: int, *, n_params: int = 9_000_000, batch_size: int = 16, patch_px: int = 256,
                    infer_batch: int = 16, n_aux: int = 3, context: bool = False, context_factor: int = 4,
                    grid_px: int = GRID_PX) -> MemoryEstimate:
    g = 2 ** 30
    stack = n_channels * grid_px * 2 / g
    aux = n_aux * grid_px * 4 / g
    ctx = (n_channels * grid_px / context_factor ** 2 * 2 / g) if context else 0.0
    mo = n_params * 4 * 4 / g
    per_sample = (patch_px / 256) ** 2 * (0.35 + 0.004 * n_channels)
    act = batch_size * per_sample
    inf = infer_batch * per_sample / 3
    slack = 1.15
    on_gpu = (stack + aux + ctx + mo + max(act, inf)) * slack
    on_host = (mo + max(act, inf)) * slack
    return MemoryEstimate(stack, aux, ctx, mo, act, inf, on_gpu, on_host)


def spec_memory(spec, n_params: int | None = None) -> MemoryEstimate:
    n_aux = 3 + (1 if getattr(spec, "aux_dist", 0) > 0 else 0) + (2 if getattr(spec, "aux_orient", 0) > 0 else 0)
    return estimate_memory(len(spec.channels), n_params=n_params or 9_000_000, batch_size=int(spec.batch_size),
                           patch_px=int(spec.patch_px), infer_batch=int(getattr(spec, "infer_batch", 16)),
                           n_aux=n_aux, context=bool(getattr(spec, "context", False)),
                           context_factor=int(getattr(spec, "context_factor", 4)))


def placement_overrides(spec, gpu: dict, *, vram_gib: float | None = None, max_vram_frac: float = 0.85) -> dict:
    """``{"data_on_gpu": False}`` when the GPU-resident estimate does not fit, else ``{}`` (autoconfig stands)."""
    if not gpu.get("amp", False) and not gpu.get("data_on_gpu", False):
        return {}                                          # CPU runtime: autoconfig already chose the host
    if vram_gib is None:
        from ..training.gpu import detect

        vram_gib = float(detect().get("vram_gb") or 0.0)
    if vram_gib <= 0:
        return {}
    bs = int(gpu.get("batch_size", spec.batch_size))
    est = estimate_memory(len(spec.channels), batch_size=bs, patch_px=int(spec.patch_px),
                          infer_batch=int(gpu.get("infer_batch", 16)), context=bool(getattr(spec, "context", False)))
    if gpu.get("data_on_gpu", True) and est.total_gpu_resident_gib > max_vram_frac * vram_gib:
        print(f"[budget] estimated {est.total_gpu_resident_gib:.1f} GiB GPU-resident > {max_vram_frac:.0%} of "
              f"{vram_gib:.1f} GiB -> input stack stays on the host (patches moved per batch)")
        return {"data_on_gpu": False}
    return {}


def epoch_seconds(history=None, steps_per_epoch: int = 150, history_steps: int = 150) -> float:
    """Median recorded epoch time (training_history ``epoch_s``, runs of ``history_steps`` steps) scaled to
    ``steps_per_epoch``; the T4 figure of the first advanced run otherwise."""
    try:
        if history is not None and len(history) and "epoch_s" in history:
            per_step = float(history.epoch_s.median()) / history_steps
            if per_step == per_step and per_step > 0:
                return per_step * steps_per_epoch
    except (AttributeError, TypeError, ValueError):
        pass
    return T4_EPOCH_S_PER_STEP * steps_per_epoch


def estimate_unit_minutes(epochs: int, steps_per_epoch: int, history=None, *, inner_eval_s: float = 6.0,
                          predict_min: float = 1.5) -> float:
    """One training unit (fold or split): epochs x epoch time + per-epoch inner eval + one inference pass."""
    return (epochs * (epoch_seconds(history, steps_per_epoch) + inner_eval_s)) / 60.0 + predict_min


def profile_cost(fid, base_epochs: int = 40, base_steps: int = 150, n_folds: int = 3, n_splits: int = 3,
                 history=None, typical_cv_epochs: int = 17, typical_disc_epochs: int = 11) -> dict:
    """Upper-bound and typical GPU minutes of one experiment at fidelity ``fid`` (estimates, not measurements).

    ``typical_*_epochs`` default to what B0 actually ran (17 CV epochs with patience, 11 fixed discovery epochs)."""
    folds = len(fid.fold_ids(range(n_folds)))
    splits = len(fid.split_ids(n_splits))
    ep = base_epochs if fid.max_epochs is None else min(base_epochs, fid.max_epochs)
    st = base_steps if fid.max_train_steps_per_epoch is None else min(base_steps, fid.max_train_steps_per_epoch)
    cv_typ, disc_typ = min(typical_cv_epochs, ep), min(typical_disc_epochs, ep)
    unit = lambda e: estimate_unit_minutes(e, st, history)
    return {"profile": fid.profile, "units": folds + splits, "epochs_cap": ep, "steps_per_epoch": st,
            "gpu_min_typical": round(folds * unit(cv_typ) + splits * unit(disc_typ), 1),
            "gpu_min_upper": round((folds + splits) * unit(ep), 1)}
