"""Phase-4 training: LightGBM RS ablations, the missing Phase-3 CNN controls, geology-aware encoders.

LightGBM (RS01-RS05, drop-group ablations)
    ``lgbm3.run_variant_cv`` / ``lgbm3.fit_full_variant`` unchanged: same folds, sampler, seed, variant
    (D2_conservative) and scoring as the Phase-3 reference run ``p3_lgbm_D2_conservative_G3_s2026``.

CNN controls (DL00-DL02)
    ``cnn3.run_cnn3_cv`` / ``cnn3.run_cnn3_discovery`` unchanged; the spec is re-loaded from the Phase-3 run's
    config.json and only the declared field changes (loss or channel list). Discovery epochs follow the
    Phase-3 rule max(min_epochs, median best epoch of the CV run).

Geology-aware encoders (P4_*)
    ``train4`` = ``cnn3.train3`` generalised to dict-output models (auxiliary heads), an optional low-resolution
    context input and SSL-initialised encoders. Unchanged from train3: PatchSampler (patches fully inside the
    fold's inner-training zone), RNG call order, DTI-surrogate loss, PU ignore band, inner-val binary-area
    DTI checkpointing, patience, OneCycle, bf16 autocast, channels_last.

Strict geography: training patches never touch validation or buffer pixels. The context branch reads 4x the
patch extent; with ``context_strict`` its training-time context is built from the stack with everything
outside the fold's training zone zeroed (inputs, not labels; at OOF inference the full stack is used).

GPU memory (RTX 3070 Laptop, 8 GB): the normalised stack lives on the GPU in fp16 (~25 MB per channel);
batch 16 x 256 px with bf16; ``grad_accum`` raises the effective batch without memory; inference batches
``infer_batch`` tiles; ``torch.cuda.empty_cache()`` between folds. If the stack does not fit it stays in
host memory and only patches are moved.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import pandas as pd
from scipy import ndimage

from gems_eda.io import write_geotiff

from . import rs_features
from .cnn import PatchSampler, _window, bbox, device_of
from .persist import ExperimentRun
from .p3 import cnn3, lgbm3
from .p3.curves import BinaryCurve
from .phase4_context import P4Context, phase2_features


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def run_summary(ctx: P4Context, exp_id: str) -> dict:
    return json.loads((ctx.run_dir(exp_id) / "run.json").read_text(encoding="utf-8")).get("summary", {})


def discovery_epochs(ctx: P4Context, exp_id: str, min_epochs: int = 5) -> int:
    """Phase-3 rule: fixed epoch / round budget = median best iteration of the geographic CV run."""
    it = [x for x in run_summary(ctx, exp_id).get("best_iters", []) if x == x]
    return max(min_epochs, int(np.median(it))) if it else min_epochs


def available_channels(ctx: P4Context, names: list[str], build_legacy: bool = True) -> list[str]:
    """Drop channels that are neither cached nor buildable; build missing Phase-2 legacy features on request."""
    legacy_missing = [n for n in names if n in ctx.store.specs and n not in ctx.store._rs_set and not ctx.store.has(n)]
    if build_legacy and legacy_missing:
        ctx.store.build(legacy_missing, verbose=False)
    out = [n for n in names if n in ctx.store.specs and ctx.store.has(n)]
    gone = [n for n in names if n not in out]
    if gone:
        print(f"[channels] not available (build RS features first or fix the config): {gone}")
    return out


def modality_filter(ctx: P4Context, names: list[str], keep=None, drop=None) -> list[str]:
    mods = {n: ctx.store.specs[n].modality for n in names}
    if keep:
        names = [n for n in names if mods[n] in set(keep)]
    if drop:
        names = [n for n in names if mods[n] not in set(drop)]
    return names


# ---------------------------------------------------------------------------
# LightGBM RS experiments
# ---------------------------------------------------------------------------
def rs_selected(ctx: P4Context, group: str, redundant: set[str] | None = None) -> list[str]:
    return rs_features.rs_group_names(ctx.rs_specs, group, ctx.store.available, exclude=redundant)


def rs_experiment_features(ctx: P4Context, exp_def: dict, redundant: set[str] | None = None) -> list[str]:
    names = modality_filter(ctx, phase2_features(ctx, exp_def["base"]), drop=exp_def.get("drop_modalities"))
    for g in exp_def.get("add_groups", []):
        names += [n for n in rs_selected(ctx, g, redundant) if n not in names]
    return names


def lgbm_exp_id(ctx: P4Context, key: str) -> str:
    return f"p4_{key}_s{ctx.cfg.seed}"


def run_rs_cv(ctx: P4Context, key: str, names: list[str], variant: str, overwrite: bool = False) -> ExperimentRun:
    exp = lgbm_exp_id(ctx, key)
    run = lgbm3.run_variant_cv(ctx, exp, names, lgbm3.LGBM_GRID[variant], overwrite=overwrite, tag=key)
    ctx.cfg4.setdefault("reuse", {}).setdefault("oof", {})[key] = exp
    return run


def run_rs_discovery(ctx: P4Context, key: str, names: list[str], variant: str, overwrite: bool = False) -> int:
    v = lgbm3.LGBM_GRID[variant]
    rounds = max(1, int(np.median(run_summary(ctx, lgbm_exp_id(ctx, key))["best_iters"])))
    for d in range(ctx.splits.n_splits):
        lgbm3.fit_full_variant(ctx, f"disc{d}_{key}", names, v, rounds, ignore_mask=ctx.splits.ignore(d),
                               predict_area=ctx.splits.eval_zone(d) & ctx.data.valid, overwrite=overwrite)
    return rounds


def rs_families(ctx: P4Context, names: list[str]) -> dict[str, list[str]]:
    fam: dict[str, list[str]] = {}
    for n in names:
        if n in ctx.store._rs_set:
            fam.setdefault(rs_features.family_of(ctx.store, n), []).append(n)
    return fam


def drop_group_runs(ctx: P4Context, base_key: str, base_names: list[str], variant: str,
                    overwrite: bool = False) -> dict[str, tuple[str, ExperimentRun]]:
    """One CV run per RS family removed from ``base_names`` (incremental value of each family)."""
    out = {}
    for fam, members in rs_families(ctx, base_names).items():
        key = f"{base_key}_drop_{fam.replace(':', '-')}"
        out[key] = (fam, run_rs_cv(ctx, key, [n for n in base_names if n not in members], variant, overwrite))
    return out


# ---------------------------------------------------------------------------
# missing Phase-3 CNN controls (DL00-DL02): cnn3 unchanged
# ---------------------------------------------------------------------------
def spec_from_config(cfg: dict, cls=cnn3.CNN3Spec):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg.items() if k in names})


def phase3_cnn_spec(ctx: P4Context, key: str, **override) -> cnn3.CNN3Spec:
    """Exact Phase-3 spec (from its config.json) with only ``override`` changed."""
    cfg = json.loads((ctx.run_dir(ctx.oof_alias.get(key, key)) / "config.json").read_text(encoding="utf-8"))
    return replace(spec_from_config(cfg), **override)


def cnn_exp_id(ctx: P4Context, key: str, patch: int) -> str:
    return f"p4_{key}_p{patch}_s{ctx.cfg.seed}"


def run_cnn_control(ctx: P4Context, key: str, spec, overwrite: bool = False) -> ExperimentRun:
    exp = cnn_exp_id(ctx, key, spec.patch_px)
    run = cnn3.run_cnn3_cv(ctx, exp, spec, input_config=key, overwrite=overwrite)
    ctx.cfg4.setdefault("reuse", {}).setdefault("oof", {})[key] = exp
    return run


def run_cnn_control_discovery(ctx: P4Context, key: str, spec, cv_exp_id: str, overwrite: bool = False) -> int:
    import torch

    epochs = discovery_epochs(ctx, cv_exp_id, spec.min_epochs)
    stack = cnn3.device_stack(cnn3.prepare_stack(spec, ctx)[0], spec, device_of())
    for d in range(ctx.splits.n_splits):
        cnn3.run_cnn3_discovery(ctx, f"disc{d}_{key}", spec, d, epochs, overwrite=overwrite, stack=stack)
    del stack
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return epochs


# ---------------------------------------------------------------------------
# geology-aware encoders (P4_*)
# ---------------------------------------------------------------------------
@dataclass
class P4Spec(cnn3.CNN3Spec):
    arch: str = "single"                 # single | dual
    enc_width: int = 32
    dec_width: int = 32
    groups: dict = field(default_factory=dict)   # {"terrain": [...], "subsurface": [...]} (dual)
    fusion: str = "concat"               # concat | gated
    context: bool = False
    context_factor: int = 4
    context_strict: bool = True
    aux_dist: float = 0.0                # weight of the proximity head (0 = no head)
    aux_orient: float = 0.0              # weight of the cos/sin(2 theta) head
    dist_clip_km: float = 2.0
    orient_min_coherence: float = 0.7
    pretrained: str = ""                 # SSL encoder checkpoint; "{fold}" is replaced by the fold id
    encoder_lr_mult: float = 1.0
    freeze_encoder_epochs: int = 0
    grad_accum: int = 1


P4_FIELDS = {f.name for f in fields(P4Spec)}


def make_p4_spec(name: str, channels: list[str], common: dict, arch_cfg: dict, gpu: dict, seed: int, **extra) -> P4Spec:
    kw = {**common, **{k: v for k, v in arch_cfg.items() if k in P4_FIELDS}, **gpu, **extra}
    kw = {k: v for k, v in kw.items() if k in P4_FIELDS and k not in ("name", "channels")}
    return P4Spec(name, list(channels), **{"encoder": "builtin", "norm": "robust", "seed": seed, **kw})


def resolve_arch_specs(cfg4: dict, groups: dict[str, list[str]], seed: int,
                       channels_override: dict[str, list[str]] | None = None) -> tuple[dict[str, P4Spec], pd.DataFrame]:
    """P4Spec per enabled architecture; resolves 'matched:<name>' (parameter-matched width) and 'same:<name>'."""
    from .geological_encoders import build_geo_model, count_params, matched_single_width

    arch = cfg4["architectures"]
    common, gpu = arch.get("common", {}), cfg4["cnn_controls"]["gpu"]
    union = [c for g in groups.values() for c in g]
    todo = {k: dict(v) for k, v in arch.items() if k.startswith("P4_") and v.get("enabled", True)}
    specs: dict[str, P4Spec] = {}
    rows = []
    for _ in range(len(todo) + 1):
        for name, v in list(todo.items()):
            w = v.get("enc_width", 32)
            chans = (channels_override or {}).get(name, union)
            if isinstance(w, str):
                how, ref = w.split(":", 1)
                if ref not in specs:
                    continue
                if how == "same":
                    w = specs[ref].enc_width
                else:
                    target = count_params(build_geo_model(specs[ref]))
                    w, _ = matched_single_width(target, len(chans), context=bool(v.get("context", False)))
            v = {**v, "enc_width": int(w), "dec_width": int(v.get("dec_width", w))}
            sp = make_p4_spec(name, chans, common, v, gpu, seed, groups=groups if v.get("arch") == "dual" else {})
            specs[name] = sp
            rows.append({"model": name, "arch": sp.arch, "fusion": sp.fusion if sp.arch == "dual" else "",
                         "enc_width": sp.enc_width, "dec_width": sp.dec_width if sp.arch == "dual" else sp.enc_width,
                         "context": sp.context, "aux_dist": sp.aux_dist, "aux_orient": sp.aux_orient,
                         "n_channels": len(sp.channels), "n_params": count_params(build_geo_model(sp)),
                         "note": v.get("note", "")})
            del todo[name]
    if todo:
        raise ValueError(f"unresolved architecture references: {list(todo)}")
    return specs, pd.DataFrame(rows)


AUX_ORDER = ("target", "kern", "weight", "dist", "ocos", "osin", "ovalid")


def build_aux(visible: np.ndarray, grid, spec: P4Spec, weight_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """(A, H, W) fp16 training rasters from VISIBLE labels only: target, kernel, weight (+ proximity, orientation)."""
    target, kern, band = cnn3.build_targets(visible, grid, spec.ignore_px)
    layers = {"target": target, "kern": kern, "weight": (weight_mask & ~band).astype(np.float16)}
    if spec.aux_dist > 0:
        d = ndimage.distance_transform_edt(~visible, sampling=(grid.pixel_km_y, grid.pixel_km_x))
        layers["dist"] = (1 - np.clip(d / spec.dist_clip_km, 0, 1)).astype(np.float16)
    if spec.aux_orient > 0:
        from gems_eda.enhance import Enhancer

        st, coh, _ = Enhancer(grid).structure_tensor(visible.astype(np.float32), 0.3)
        th = np.radians(np.nan_to_num(st))
        ok = (kern > 0) & (np.nan_to_num(coh) >= spec.orient_min_coherence)   # no supervision at intersections
        layers["ocos"] = np.where(ok, np.cos(2 * th), 0).astype(np.float16)
        layers["osin"] = np.where(ok, np.sin(2 * th), 0).astype(np.float16)
        layers["ovalid"] = ok.astype(np.float16)
    names = [n for n in AUX_ORDER if n in layers]
    return np.stack([layers[n] for n in names]), names


def to_tensor_stack(stack, device):
    """fp16 (C, H, W) tensor on ``device``; host tensor if the GPU copy does not fit."""
    import torch

    if torch.is_tensor(stack):
        return stack
    t = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float16))
    if device.type != "cuda":
        return t
    try:
        return t.to(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"stack ({t.numel() * 2 / 2**30:.2f} GiB) does not fit on the GPU -> host-resident, patches moved per batch")
        return t


def pooled_context(gstack, factor: int, mask: np.ndarray | None = None):
    """(C, H//f, W//f) average-pooled stack; ``mask`` zeroes inputs outside the training zone (strict context)."""
    import torch
    import torch.nn.functional as F

    C, H, W = gstack.shape
    out = torch.empty((C, H // factor, W // factor), dtype=gstack.dtype, device=gstack.device)
    m = None if mask is None else torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).to(gstack.device)
    for c in range(C):
        x = gstack[c].float()
        if m is not None:
            x = x * m
        out[c] = F.avg_pool2d(x[None, None], factor)[0, 0].to(gstack.dtype)
    return out


def context_crop(gctx, r0: int, c0: int, s: int, f: int):
    """Context crop (C, s, s) of the pooled stack centred on the patch (r0, c0, s): covers f x the patch extent."""
    C, h, w = gctx.shape
    cr, cc = (r0 + s // 2) // f, (c0 + s // 2) // f
    y0, x0 = cr - s // 2, cc - s // 2
    out = gctx.new_zeros((C, s, s))
    ya, yb, xa, xb = max(y0, 0), min(y0 + s, h), max(x0, 0), min(x0 + s, w)
    if ya < yb and xa < xb:
        out[:, ya - y0:yb - y0, xa - x0:xb - x0] = gctx[:, ya:yb, xa:xb]
    return out


def make_batch4(gstack, gaux, sampler: PatchSampler, spec: P4Spec, rng, device, gctx=None):
    """cnn3.make_batch_gpu (same RNG call order -> same patches) + aux layers + optional context crops."""
    import torch

    s, xs, auxs, cs = spec.patch_px, [], [], []
    while len(xs) < spec.batch_size:
        rc = sampler.draw(rng.random() < spec.pos_patch_frac) or sampler.draw(False, 1000)
        if rc is None:
            raise RuntimeError("no admissible patch: training zone too fragmented for this patch size")
        r0, c0 = rc
        x, a = gstack[:, r0:r0 + s, c0:c0 + s], gaux[:, r0:r0 + s, c0:c0 + s]
        cx = context_crop(gctx, r0, c0, s, spec.context_factor) if gctx is not None else None
        if spec.rot180 and rng.random() < 0.5:           # 180 deg rotation keeps strikes (axial targets unchanged)
            x, a = x.flip((-2, -1)), a.flip((-2, -1))
            cx = cx.flip((-2, -1)) if cx is not None else None
        xs.append(x), auxs.append(a), cs.append(cx)
    x = torch.stack(xs).to(device, non_blocking=True).float()
    a = torch.stack(auxs).to(device, non_blocking=True).float()
    c = torch.stack(cs).to(device, non_blocking=True).float() if gctx is not None else None
    return x, a, c


def _forward(model, x, c, spec: P4Spec, device, amp):
    import torch

    with torch.autocast(device_type=device.type, dtype=amp or torch.float16, enabled=amp is not None):
        return model(cnn3._to_model_layout(x, spec, device),
                     None if c is None else cnn3._to_model_layout(c, spec, device))


def loss4(out: dict, a, names: list[str], spec: P4Spec):
    import torch

    i = {n: k for k, n in enumerate(names)}
    sl = lambda n: a[:, i[n]:i[n] + 1]
    w = sl("weight")
    loss = cnn3.loss_fn(out["seg"].float(), sl("target"), sl("kern"), w, spec)
    if spec.aux_dist > 0 and "dist" in out:
        l_d = (torch.abs(torch.sigmoid(out["dist"].float()) - sl("dist")) * w).sum() / w.sum().clamp_min(1)
        loss = loss + spec.aux_dist * l_d
    if spec.aux_orient > 0 and "orient" in out:
        o = torch.tanh(out["orient"].float())
        t = torch.cat([sl("ocos"), sl("osin")], 1)
        v = sl("ovalid") * w
        l_o = (((o - t) ** 2).sum(1, keepdim=True) * v).sum() / v.sum().clamp_min(1)
        loss = loss + spec.aux_orient * l_o
    return loss


def predict_tiled4(model, gstack, box, spec: P4Spec, device, gctx=None) -> np.ndarray:
    """cnn3.predict_tiled_gpu for dict-output models, optional context crops, host- or device-resident stacks."""
    import torch

    patch, stride = spec.patch_px, int(spec.patch_px * spec.infer_stride_frac)
    r0, r1, c0, c1 = box
    C, H, W = gstack.shape
    m = patch // 4
    R0, R1, C0, C1 = r0 - m, r1 + m, c0 - m, c1 + m
    R1, C1 = max(R1, R0 + patch), max(C1, C0 + patch)
    ys = list(range(R0, max(R1 - patch, R0) + 1, stride))
    xs = list(range(C0, max(C1 - patch, C0) + 1, stride))
    if ys[-1] + patch < R1:
        ys.append(R1 - patch)
    if xs[-1] + patch < C1:
        xs.append(C1 - patch)
    acc = torch.zeros((R1 - R0, C1 - C0), dtype=torch.float32, device=device)
    wsum = torch.zeros_like(acc)
    win = torch.from_numpy(_window(patch)).to(device)
    tiles = [(y, x) for y in ys for x in xs]
    amp = cnn3._amp_dtype(spec, device)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(tiles), spec.infer_batch):
            chunk = tiles[i:i + spec.infer_batch]
            xb = torch.zeros((len(chunk), C, patch, patch), dtype=gstack.dtype, device=gstack.device)
            cb = torch.stack([context_crop(gctx, y, x, patch, spec.context_factor) for y, x in chunk]) if gctx is not None else None
            for j, (y, x) in enumerate(chunk):
                ya, yb, xa, xb_ = max(y, 0), min(y + patch, H), max(x, 0), min(x + patch, W)
                if ya < yb and xa < xb_:
                    xb[j, :, ya - y:yb - y, xa - x:xb_ - x] = gstack[:, ya:yb, xa:xb_]
            xb = xb.to(device).float()
            cb = cb.to(device).float() if cb is not None else None
            p = torch.sigmoid(_forward(model, xb, cb, spec, device, amp)["seg"].float())[:, 0]
            for j, (y, x) in enumerate(chunk):
                acc[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += p[j] * win
                wsum[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += win
    out = acc / wsum.clamp_min(1e-6)
    return out[m:m + (r1 - r0), m:m + (c1 - c0)].cpu().numpy()


def predict_mask4(model, gstack, mask: np.ndarray, spec: P4Spec, device, gctx=None) -> np.ndarray:
    out = np.full(mask.shape, np.nan, np.float32)
    if not mask.any():
        return out
    r0, r1, c0, c1 = bbox(mask)
    p = predict_tiled4(model, gstack, (r0, r1, c0, c1), spec, device, gctx)
    sub = out[r0:r1, c0:c1]
    sub[mask[r0:r1, c0:c1]] = p[mask[r0:r1, c0:c1]]
    return out


def train4(spec: P4Spec, gstack, gaux, aux_names: list[str], allowed, positives, valid, device, rng, *,
           gctx_train=None, gctx_eval=None, inner_mask=None, inner_truth=None, fixed_epochs: int | None = None,
           log_prefix: str = "", ckpt_path=None, pretrained_path=None):
    import torch

    from .geological_encoders import build_geo_model, count_params, load_encoder_weights
    from .metric import dti_components

    cnn3._seed_all(spec.seed, spec.cudnn_benchmark and device.type == "cuda")
    model = build_geo_model(spec)
    init = load_encoder_weights(model, pretrained_path, spec.channels) if pretrained_path else None
    model = model.to(device)
    if spec.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    enc_ids = {id(p) for p in model.encoder_parameters()}
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": enc, "lr": spec.lr * spec.encoder_lr_mult}, {"params": rest, "lr": spec.lr}]
    opt = torch.optim.AdamW(groups, lr=spec.lr, weight_decay=spec.weight_decay)
    epochs = fixed_epochs or spec.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=epochs * spec.steps_per_epoch)
    amp = cnn3._amp_dtype(spec, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    sampler = PatchSampler(allowed, valid, positives, spec.patch_px, spec.min_valid_frac, rng)
    accum = max(int(spec.grad_accum), 1)
    best, best_ep, best_state, stale, hist = -1.0, 0, None, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        frozen = ep < spec.freeze_encoder_epochs
        for p in enc:
            p.requires_grad_(not frozen)
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        for _ in range(spec.steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            for _a in range(accum):
                x, a, c = make_batch4(gstack, gaux, sampler, spec, rng, device, gctx_train)
                out = _forward(model, x, c, spec, device, amp)
                loss = loss4(out, a, aux_names, spec) / accum
                scaler.scale(loss).backward()
                loss_sum += loss.detach()
            scaler.step(opt)
            scaler.update()
            sched.step()
        row = {"epoch": ep + 1, "train_loss": float(loss_sum) / spec.steps_per_epoch, "lr": float(sched.get_last_lr()[-1]),
               "encoder_frozen": frozen, "epoch_s": round(time.perf_counter() - t0, 1)}
        evaluate = (ep + 1) % max(spec.eval_every, 1) == 0 or ep + 1 == epochs
        if evaluate and inner_mask is not None and inner_mask.any():
            t1 = time.perf_counter()
            pv = predict_mask4(model, gstack, inner_mask, spec, device, gctx_eval)
            b = BinaryCurve(pv, inner_truth, inner_mask).best(np.geomspace(0.003, 0.5, 30))
            row.update(inner_dti_binary_best=b["dti"], inner_best_frac=b["target_frac"],
                       inner_dti_raw=dti_components(pv, inner_truth, eval_mask=inner_mask).dti,
                       inner_eval_s=round(time.perf_counter() - t1, 1))
            score = row["inner_dti_binary_best"] if spec.select_by == "binary_area" else row["inner_dti_raw"]
            if score > best:
                best, best_ep, stale = score, ep + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        if device.type == "cuda":
            row["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        hist.append(row)
        print(f"{log_prefix} ep {ep + 1:3d} loss {row['train_loss']:.4f} inner {row.get('inner_dti_binary_best', float('nan')):.4f}"
              f"  ({row['epoch_s']:.0f}s train, {row.get('inner_eval_s', 0):.0f}s eval)")
        if inner_mask is not None and ep + 1 >= spec.min_epochs and stale >= spec.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = len(hist)
    if ckpt_path is not None:
        torch.save({"state_dict": model.state_dict(), "spec": asdict(spec), "best_inner": best, "best_epoch": best_ep,
                    "n_params": count_params(model), "init": init}, ckpt_path)
    return model, pd.DataFrame(hist), best_ep


def run_cnn4_cv(ctx: P4Context, exp_id: str, spec: P4Spec, *, overwrite: bool = False, stack=None) -> ExperimentRun:
    """Geographic CV of a P4 model (OOF map + the Phase-3 finish: area curves, cross-fit, metrics, registry)."""
    import torch

    from .geological_encoders import build_geo_model, count_params
    from .metric import full_diagnostics

    run = ExperimentRun(ctx.paths, exp_id, "cnn", {**asdict(spec), "n_params": count_params(build_geo_model(spec)),
                                                   "folds": ctx.bench.rel(ctx.folds.directory)})
    if run.done and not overwrite:
        print(f"[{exp_id}] already done -> {run.dir}")
        return run
    run.start()
    device = device_of()
    rng = np.random.default_rng(spec.seed)
    if stack is None:
        with run.timer("load_stack"):
            stack, stats = cnn3.prepare_stack(spec, ctx)
        (run.models_dir / "normalisation.json").write_text(json.dumps({"channels": spec.channels, **stats}), encoding="utf-8")
    gstack = to_tensor_stack(stack, device)
    gctx_full = pooled_context(gstack, spec.context_factor) if spec.context else None
    print(f"[{exp_id}] device={device} | stack on {gstack.device} | amp={cnn3._amp_dtype(spec, device)} | "
          f"params={run.config['n_params']:,}")
    data = ctx.data
    oof = np.full(data.shape, np.nan, np.float32)
    hist_all, met = [], []
    for k in ctx.fold_ids:
        allowed = ctx.folds.inner_train(k)
        visible = data.labels & ctx.folds.train(k)
        aux, names = build_aux(visible, data.grid, spec, allowed & data.valid)
        gaux = torch.from_numpy(aux).to(gstack.device)
        gctx_train = gctx_full
        if spec.context and spec.context_strict:
            gctx_train = pooled_context(gstack, spec.context_factor, ctx.folds.train(k) & data.valid)
        pre = spec.pretrained.format(fold=k) if spec.pretrained else None
        with run.timer(f"fold{k}_train"):
            model, hist, best_ep = train4(spec, gstack, gaux, names, allowed, visible, data.valid, device, rng,
                                          gctx_train=gctx_train, gctx_eval=gctx_train,
                                          inner_mask=ctx.folds.inner_val(k) & data.valid, inner_truth=visible,
                                          log_prefix=f"[{exp_id} f{k}]", ckpt_path=run.models_dir / f"fold{k}.pt",
                                          pretrained_path=pre)
        hist["fold"] = k
        hist_all.append(hist)
        val = ctx.folds.val(k)
        with run.timer(f"fold{k}_predict"):
            oof[val] = predict_mask4(model, gstack, val, spec, device, gctx_full)[val]
        diag = full_diagnostics(oof, data.labels, val & data.valid, rng, ctx.cfg.recall_tolerances_m, ctx.cfg.report_threshold)
        met.append({"fold": k, "best_iteration": best_ep, "epochs_run": len(hist), "pretrained": pre or "",
                    "gpu_peak_mb": float(hist.gpu_peak_mb.max()) if "gpu_peak_mb" in hist else np.nan, **diag})
        del model, gaux
        if spec.context and spec.context_strict:
            del gctx_train
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_geotiff(run.oof_path, {"oof_probability": oof}, data.grid)
    run.register("oof", run.oof_path, "raster")
    out = lgbm3.finish_variant(ctx, run, oof, hist_all, [], met, [])
    ctx.cfg4.setdefault("reuse", {}).setdefault("oof", {})[spec.name] = exp_id
    return out


def run_cnn4_discovery(ctx: P4Context, exp_id: str, spec: P4Spec, d: int, epochs: int, *, overwrite: bool = False,
                       stack=None, pretrained_path: str | None = None) -> np.ndarray:
    """Whole-map P4 model for discovery split d (withheld systems + 1 km ring weight 0, fixed epochs)."""
    import torch

    from .ensemble import read_map

    run = ExperimentRun(ctx.paths, exp_id, "cnn", {**asdict(spec), "discovery_split": d, "epochs_fixed": epochs,
                                                   "pretrained_used": pretrained_path or ""})
    pred_path = run.dir / "pred.tif"
    if run.state.get("status") == "done" and pred_path.exists() and not overwrite:
        return read_map(pred_path)
    run.start()
    device = device_of()
    rng = np.random.default_rng(spec.seed + d)
    if stack is None:
        stack, _ = cnn3.prepare_stack(spec, ctx)
    gstack = to_tensor_stack(stack, device)
    gctx = pooled_context(gstack, spec.context_factor) if spec.context else None
    data, ign = ctx.data, ctx.splits.ignore(d)
    visible = data.labels & ~ign
    aux, names = build_aux(visible, data.grid, spec, data.valid & ~ign)
    gaux = torch.from_numpy(aux).to(gstack.device)
    with run.timer("train"):
        model, hist, _ = train4(spec, gstack, gaux, names, data.valid, visible, data.valid, device, rng, gctx_train=gctx,
                                fixed_epochs=epochs, log_prefix=f"[{exp_id}]", ckpt_path=run.models_dir / "model.pt",
                                pretrained_path=pretrained_path)
    run.save_table(hist, "history")
    zone = ctx.splits.eval_zone(d) & data.valid
    with run.timer("predict"):
        pred = predict_mask4(model, gstack, zone, spec, device, gctx)
    write_geotiff(pred_path, {"probability": pred}, data.grid)
    run.register("pred", pred_path, "raster")
    run.state.update(status="done")
    run._flush()
    del model, gaux
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred


def run_cnn4_discovery_all(ctx: P4Context, key: str, spec: P4Spec, cv_exp_id: str, *, overwrite: bool = False,
                           pretrained_path: str | None = None) -> int:
    epochs = discovery_epochs(ctx, cv_exp_id, spec.min_epochs)
    stack, _ = cnn3.prepare_stack(spec, ctx)
    gstack = to_tensor_stack(stack, device_of())
    del stack
    for d in range(ctx.splits.n_splits):
        run_cnn4_discovery(ctx, f"disc{d}_{key}", spec, d, epochs, overwrite=overwrite, stack=gstack,
                           pretrained_path=pretrained_path)
    return epochs


def load_cnn4_fold_models(run_dir, device=None) -> list:
    import torch

    from .geological_encoders import build_geo_model

    device = device or device_of()
    out = []
    for ck in sorted((run_dir / "models").glob("fold*.pt")):
        obj = torch.load(ck, map_location=device, weights_only=False)
        spec = spec_from_config(obj["spec"], P4Spec)
        m = build_geo_model(spec).to(device)
        m.load_state_dict(obj["state_dict"])
        m.eval()
        out.append((spec, m))
    return out


def gate_window(model, gstack, r0: int, c0: int, spec: P4Spec, device, gctx=None) -> np.ndarray | None:
    """Level-0 terrain-branch gate weight (0..1) on one patch of a gated dual encoder: terrain vs subsurface reliance."""
    import torch

    s = spec.patch_px
    x = gstack[:, r0:r0 + s, c0:c0 + s][None].to(device).float()
    c = context_crop(gctx, r0, c0, s, spec.context_factor)[None].to(device).float() if gctx is not None else None
    model.eval()
    with torch.inference_mode():
        _forward(model, x, c, spec, device, cnn3._amp_dtype(spec, device))
    g = model.gates()
    return None if not g or g[0] is None else g[0][0, 0].float().cpu().numpy()
