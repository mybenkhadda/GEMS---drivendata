"""Self-supervised geophysical representation learning (Phase 4, P2).

SSL01  masked geophysical autoencoder: random 16 px blocks (ratio 0.5) are zeroed in every channel and whole
       modalities are dropped with probability ``channel_drop_p``; the network reconstructs the original
       normalised channels inside the masked region, plus finite-difference gradients (``grad_weight``).
SSL02  cross-modal reconstruction (``mask_mode="modality"``): one of terrain / magnetic / gravity is withheld
       per sample and reconstructed from the others with a smooth-L1 loss (``robust=True``), which tolerates
       genuinely unpredictable cross-modal variation better than MSE.
SSL03  contrastive learning: deliberately NOT implemented. Deferred until SSL01 shows a fine-tuning gain;
       if added, negatives must exclude geographically adjacent patches and augmentations must keep strike
       semantics (180 deg rotation only; no flips, no channel-wise colour jitter of potential fields).

Unlabelled pixels are never treated as fault-free: no label enters pretraining at all.

Loss balance: per-channel weights are the inverse variance of the normalised field (and of its gradients for
the gradient term) over the pretraining region, so smooth large-amplitude fields cannot dominate.

Evaluation protocols (reported separately, never mixed)
    strict        one pretraining per geographic fold on ``folds.train(k)`` only: validation and buffer
                  features never enter pretraining (inductive). Fine-tuning fold k loads encoder fold k.
    transductive  one pretraining on the whole valid footprint. Legitimate for the leaderboard (the hidden
                  test faults lie inside the same raster, whose features are all given) but optimistic
                  for local geographic validation.
Pseudo-discovery fine-tuning uses the transductive encoder: discovery models already see the withheld
systems as *inputs* (Phase-3 protocol), so this adds no new information about withheld labels.

Fine-tuning = ``phase4_training.run_cnn4_cv`` with ``P4Spec.pretrained`` set; the random-init control is the
same architecture/channels/seed without ``pretrained`` (configs/phase4/experiments.yaml: ssl.arch).
A low reconstruction error is never a promotion argument (configs/phase4/experiments.yaml: promotion).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .cnn import PatchSampler, device_of
from .p3 import cnn3
from .persist import ExperimentRun


@dataclass
class SSLSpec:
    name: str
    channels: list
    enc_width: int = 32
    depth: int = 5
    patch_px: int = 256
    batch_size: int = 16
    epochs: int = 30
    steps_per_epoch: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mask_mode: str = "spatial"            # spatial | modality | mixed
    mask_ratio: float = 0.5
    block_px: int = 16
    channel_drop_p: float = 0.15
    grad_weight: float = 0.5
    robust: bool = False
    target_modalities: tuple = ("terrain", "magnetic", "gravity")
    min_valid_frac: float = 0.3
    eval_patches: int = 64
    eval_every: int = 2
    seed: int = 2026
    norm: str = "robust"
    amp: bool = True
    amp_dtype: str = "auto"
    channels_last: bool = True
    cudnn_benchmark: bool = True


def ssl_spec_for(p4spec, name: str, cfg: dict, seed: int) -> SSLSpec:
    """SSL spec whose encoder matches a P4 fine-tuning spec (channels, width, depth, patch)."""
    keys = {f for f in SSLSpec.__dataclass_fields__} - {"name", "channels", "enc_width", "patch_px", "seed"}
    return SSLSpec(name, list(p4spec.channels), enc_width=p4spec.enc_width, patch_px=p4spec.patch_px, seed=seed,
                   **{k: v for k, v in cfg.items() if k in keys})


def modality_index(ctx, channels: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, c in enumerate(channels):
        out.setdefault(ctx.store.specs[c].modality, []).append(i)
    return out


def channel_weights(stack: np.ndarray, region: np.ndarray, n: int = 200_000, seed: int = 2026,
                    clip=(0.1, 10.0)) -> tuple[np.ndarray, np.ndarray]:
    """Inverse variance of each normalised channel and of its column differences over the region (mean 1)."""
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(region[:, :-1] & region[:, 1:])
    sel = rng.choice(len(rr), min(n, len(rr)), replace=False)
    rr, cc = rr[sel], cc[sel]
    var_f, var_g = [], []
    for c in range(stack.shape[0]):
        a = stack[c]
        v0 = a[rr, cc].astype(np.float32)
        g = a[rr, cc + 1].astype(np.float32) - v0
        var_f.append(float(np.var(v0)))
        var_g.append(float(np.var(g)))

    def norm(v):
        w = 1.0 / np.clip(np.asarray(v), 1e-6, None)
        w = np.clip(w / np.median(w), *clip)
        return (w / w.mean()).astype(np.float32)
    return norm(var_f), norm(var_g)


def make_masks(B: int, C: int, s: int, spec: SSLSpec, mods: dict[str, list[int]], gen, device):
    """(spatial (B, s, s) bool, channel (B, C) bool) masks for one batch."""
    import torch
    import torch.nn.functional as F

    spatial = torch.zeros((B, s, s), dtype=torch.bool, device=device)
    chan = torch.zeros((B, C), dtype=torch.bool, device=device)
    if spec.mask_mode in ("spatial", "mixed"):
        g = -(-s // spec.block_px)
        m = (torch.rand((B, 1, g, g), generator=gen, device=device) < spec.mask_ratio).float()
        spatial = F.interpolate(m, scale_factor=spec.block_px, mode="nearest")[:, 0, :s, :s] > 0.5
    names = [m for m in mods]
    if spec.mask_mode == "spatial":
        drop = torch.rand((B, len(names)), generator=gen, device=device) < spec.channel_drop_p
        for j, m in enumerate(names):
            chan[:, mods[m]] |= drop[:, j:j + 1]
    else:                                    # modality / mixed: exactly one target modality withheld
        tgt = [m for m in spec.target_modalities if m in mods]
        pick = torch.randint(len(tgt), (B,), generator=gen, device=device)
        for j, m in enumerate(tgt):
            sel = pick == j
            if spec.mask_mode == "mixed":
                sel &= torch.rand((B,), generator=gen, device=device) < spec.channel_drop_p
            chan[:, mods[m]] |= sel[:, None]
    return spatial, chan


def ssl_loss(pred, x, region, chan_w, grad_w, spec: SSLSpec):
    """Weighted masked reconstruction (+ gradient term). ``region`` (B, C, s, s) bool = scored pixels."""
    import torch
    import torch.nn.functional as F

    R = region.float()
    e = F.smooth_l1_loss(pred, x, reduction="none", beta=1.0) if spec.robust else (pred - x) ** 2
    n_c = R.sum((0, 2, 3))
    present = n_c > 0
    per_c = (e * R).sum((0, 2, 3)) / n_c.clamp_min(1)
    rec = (chan_w * per_c)[present].sum() / chan_w[present].sum().clamp_min(1e-6)
    out = {"rec": rec}
    loss = rec
    if spec.grad_weight > 0:
        g = 0.0
        for dim in (-1, -2):
            dp, dx = torch.diff(pred, dim=dim), torch.diff(x, dim=dim)
            Rg = (region.narrow(dim, 1, region.shape[dim] - 1) & region.narrow(dim, 0, region.shape[dim] - 1)).float()
            ng = Rg.sum((0, 2, 3))
            per_g = (torch.abs(dp - dx) * Rg).sum((0, 2, 3)) / ng.clamp_min(1)
            g = g + (grad_w * per_g)[ng > 0].sum() / grad_w[ng > 0].sum().clamp_min(1e-6)
        out["grad"] = g / 2
        loss = loss + spec.grad_weight * out["grad"]
    out["loss"] = loss
    return out


def _batch(gstack, gvalid, sampler: PatchSampler, spec: SSLSpec, rng, device, coords=None):
    import torch

    s, xs, vs = spec.patch_px, [], []
    todo = list(coords) if coords is not None else None
    while len(xs) < (len(todo) if todo is not None else spec.batch_size):
        if todo is not None:
            r0, c0 = todo[len(xs)]
        else:
            rc = sampler.draw(False, 1000)
            if rc is None:
                raise RuntimeError("no admissible SSL patch in the pretraining region")
            r0, c0 = rc
        x, v = gstack[:, r0:r0 + s, c0:c0 + s], gvalid[r0:r0 + s, c0:c0 + s]
        if todo is None and rng.random() < 0.5:
            x, v = x.flip((-2, -1)), v.flip((-2, -1))
        xs.append(x), vs.append(v)
    return torch.stack(xs).to(device).float(), torch.stack(vs).to(device)


def pretrain(ctx, exp_id: str, spec: SSLSpec, region: np.ndarray, protocol: str, *, overwrite: bool = False,
             stack=None) -> ExperimentRun:
    """Masked-autoencoder pretraining on patches fully inside ``region``; saves models/encoder.pt (best eval loss)."""
    import torch

    from .geological_encoders import MaskedAutoencoder, count_params
    from .phase4_training import to_tensor_stack

    run = ExperimentRun(ctx.paths, exp_id, "ssl", {**asdict(spec), "protocol": protocol, "region_px": int(region.sum()),
                                                   "folds": ctx.bench.rel(ctx.folds.directory)})
    enc_path = run.models_dir / "encoder.pt"
    if run.state.get("status") == "done" and enc_path.exists() and not overwrite:
        print(f"[{exp_id}] already done -> {enc_path}")
        return run
    run.start()
    device = device_of()
    cnn3._seed_all(spec.seed, spec.cudnn_benchmark and device.type == "cuda")
    rng = np.random.default_rng(spec.seed)
    if stack is None:
        stack, _ = cnn3.prepare_stack(spec, ctx)
    valid = ctx.data.valid
    reg = region & valid
    chan_w_np, grad_w_np = channel_weights(np.asarray(stack) if not torch.is_tensor(stack) else stack.cpu().numpy(), reg)
    pd.DataFrame({"channel": spec.channels, "w_field": chan_w_np, "w_grad": grad_w_np}).to_csv(
        run.dir / "channel_weights.csv", index=False)
    gstack = to_tensor_stack(stack, device)
    gvalid = torch.from_numpy(valid).to(gstack.device)
    chan_w, grad_w = torch.from_numpy(chan_w_np).to(device), torch.from_numpy(grad_w_np).to(device)
    mods = modality_index(ctx, spec.channels)
    sampler = PatchSampler(reg, valid, np.zeros_like(valid), spec.patch_px, spec.min_valid_frac, rng)
    ev_rng = np.random.default_rng(spec.seed + 1)
    ev_sampler = PatchSampler(reg, valid, np.zeros_like(valid), spec.patch_px, spec.min_valid_frac, ev_rng)
    ev_coords = [ev_sampler.draw(False, 1000) for _ in range(spec.eval_patches)]
    ev_coords = [c for c in ev_coords if c is not None]
    model = MaskedAutoencoder(len(spec.channels), spec.enc_width, spec.depth).to(device)
    if spec.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec.lr, total_steps=spec.epochs * spec.steps_per_epoch)
    amp = cnn3._amp_dtype(spec, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    gen = torch.Generator(device=device).manual_seed(spec.seed)
    hist, best, best_ep = [], np.inf, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step_loss(x, v):
        B, C, s, _ = x.shape
        spatial, chan = make_masks(B, C, s, spec, mods, gen, device)
        hide = spatial[:, None] | chan[:, :, None, None]
        xin = x.masked_fill(hide, 0.0)
        with torch.autocast(device_type=device.type, dtype=amp or torch.float16, enabled=amp is not None):
            pred = model(cnn3._to_model_layout(xin, spec, device))
        return ssl_loss(pred.float(), x, hide & v[:, None], chan_w, grad_w, spec), pred, hide & v[:, None]

    for ep in range(spec.epochs):
        model.train()
        t0 = time.perf_counter()
        acc = {"loss": 0.0, "rec": 0.0, "grad": 0.0}
        for _ in range(spec.steps_per_epoch):
            x, v = _batch(gstack, gvalid, sampler, spec, rng, device)
            opt.zero_grad(set_to_none=True)
            out, _, _ = step_loss(x, v)
            scaler.scale(out["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k in acc:
                if k in out:
                    acc[k] += float(out[k].detach())
        row = {"exp_id": exp_id, "protocol": protocol, "epoch": ep + 1, "epoch_s": round(time.perf_counter() - t0, 1),
               **{f"train_{k}": v / spec.steps_per_epoch for k, v in acc.items()}}
        if ((ep + 1) % max(spec.eval_every, 1) == 0 or ep + 1 == spec.epochs) and ev_coords:
            model.eval()
            gen_state = gen.get_state()
            gen.manual_seed(spec.seed + 7)                      # identical eval masks every time
            sse, sst, ev_loss = np.zeros(len(spec.channels)), np.zeros(len(spec.channels)), 0.0
            with torch.inference_mode():
                for i in range(0, len(ev_coords), spec.batch_size):
                    x, v = _batch(gstack, gvalid, sampler, spec, rng, device, coords=ev_coords[i:i + spec.batch_size])
                    out, pred, R = step_loss(x, v)
                    ev_loss += float(out["loss"]) * len(x)
                    Rf = R.float()
                    sse += (((pred.float() - x) ** 2) * Rf).sum((0, 2, 3)).cpu().numpy()
                    mu = (x * Rf).sum((0, 2, 3)) / Rf.sum((0, 2, 3)).clamp_min(1)
                    sst += (((x - mu[None, :, None, None]) ** 2) * Rf).sum((0, 2, 3)).cpu().numpy()
            gen.set_state(gen_state)
            row["eval_loss"] = ev_loss / len(ev_coords)
            for m, idx in mods.items():
                row[f"eval_r2_{m}"] = float(1 - sse[idx].sum() / max(sst[idx].sum(), 1e-9))
            if row["eval_loss"] < best:
                best, best_ep = row["eval_loss"], ep + 1
                torch.save({"encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                            "channels": list(spec.channels), "exp_id": exp_id, "epochs": best_ep, "protocol": protocol,
                            "spec": asdict(spec)}, enc_path)
        if device.type == "cuda":
            row["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        hist.append(row)
        print(f"[{exp_id}] ep {ep + 1:3d} loss {row['train_loss']:.4f} eval {row.get('eval_loss', float('nan')):.4f}"
              f" ({row['epoch_s']:.0f}s)")
    torch.save({"state_dict": model.state_dict(), "channels": list(spec.channels), "spec": asdict(spec)},
               run.models_dir / "mae_last.pt")
    run.save_table(pd.DataFrame(hist), "history")
    run.finish({"best_eval_loss": best, "best_epoch": best_ep, "n_params": count_params(model), "protocol": protocol,
                "train_seconds": float(sum(h["epoch_s"] for h in hist))})
    del gstack
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run


def encoder_path_template(ctx, key: str, protocol: str) -> str:
    """Path for P4Spec.pretrained ('{fold}' is filled by run_cnn4_cv for the strict protocol)."""
    if protocol == "strict":
        return str(ctx.paths.experiments / f"ssl_{key}_strict_fold{{fold}}" / "models" / "encoder.pt")
    return str(ctx.paths.experiments / f"ssl_{key}_transductive" / "models" / "encoder.pt")


def run_protocol(ctx, key: str, spec: SSLSpec, protocol: str, *, overwrite: bool = False) -> list[ExperimentRun]:
    """strict: one pretraining per fold on folds.train(k); transductive: one on the whole valid footprint."""
    stack, _ = cnn3.prepare_stack(spec, ctx)
    runs = []
    if protocol == "strict":
        for k in ctx.fold_ids:
            runs.append(pretrain(ctx, f"ssl_{key}_strict_fold{k}", spec, ctx.folds.train(k) & ctx.data.valid, protocol,
                                 overwrite=overwrite, stack=stack))
    elif protocol == "transductive":
        runs.append(pretrain(ctx, f"ssl_{key}_transductive", spec, ctx.data.valid, protocol, overwrite=overwrite, stack=stack))
    else:
        raise KeyError(protocol)
    return runs


def history_table(ctx, exp_ids: list[str]) -> pd.DataFrame:
    parts = []
    for e in exp_ids:
        p = ctx.paths.experiments / e / "history.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fig_ssl_history(hist: pd.DataFrame):
    import matplotlib.pyplot as plt

    if not len(hist):
        return None
    r2 = [c for c in hist if c.startswith("eval_r2_")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for e, g in hist.groupby("exp_id"):
        axs[0].plot(g.epoch, g.train_loss, linewidth=1.1, label=e)
        if "eval_loss" in g:
            gg = g.dropna(subset=["eval_loss"])
            axs[0].plot(gg.epoch, gg.eval_loss, "--", linewidth=0.9)
        for c in r2:
            gg = g.dropna(subset=[c])
            axs[1].plot(gg.epoch, gg[c], linewidth=1.0, label=f"{e}:{c[8:]}")
    axs[0].set_title("SSL loss (solid train, dashed held-out patches)", fontsize=10, loc="left")
    axs[1].set_title("masked-region R² per modality (diagnostic only)", fontsize=10, loc="left")
    for ax in axs:
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlabel("epoch", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
    return fig
