"""Corrected training (Phase M) on the G14 data layer.

One ``fit_corrected`` call trains one ``TrainSpec`` on one unit. Differences from ``g14.train.fit_unit``:

* supervision families: ``original`` (G14 control, shortcut kept: M0), ``geophysics_only`` (no known-fault input:
  M1), ``conditional`` (per-epoch context / target system partition, targets hidden from every input: M2);
* separated train / inner targets with a label-free boundary embargo (M3);
* any objective of ``losses.OBJECTIVES`` (M4) and any negative policy of ``discovery.supervision_masks`` (M5);
* channel subsets by source and modality (``ChannelView``: Phase K ablations) without rebuilding the stack;
* augmentation policies (Phase L), noise restricted to available pixels;
* inner validation reported four ways (enriched-patch diagnostic, uniform tiles, deterministic tiles, full-region)
  with checkpoint selection on the FULL-REGION inner DTI by default (ratio of summed official components);
* per-term loss / gradient diagnostics (E1).

The data layer (stack, normaliser, multiscale windows, inference) is G14's, so G14 and G14.1 models are comparable.

UNTESTED: written without execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..g14.multiscale import LEVEL_NAMES, collate, lattice_centres, transform_channels, transform_scalars

FEATURE_SETS = {
    "all": (("raw", "g13", "g14", "known"), None),
    "known_only": (("known",), None),
    "geophysics": (("raw", "g13", "g14"), None),
    "magnetic": (("raw", "g13", "g14"), ("magnetic",)),
    "gravity": (("raw", "g13", "g14"), ("gravity",)),
    "terrain": (("raw", "g13", "g14"), ("terrain",)),
    "conductivity": (("raw", "g13", "g14"), ("conductivity",)),
    "geodesy": (("raw", "g13", "g14"), ("geodesy",)),
    "structural": (("raw", "g13", "g14"), ("structural",)),
    "raw_only": (("raw",), None),
    "engineered_only": (("g13", "g14"), None),
    "raw_engineered": (("raw", "g13", "g14"), None),
}
AUGMENTATIONS = {
    #                      d4     rot180  noise  modality sensor known_drop_allowed
    "none": dict(d4=False, rot180=False, noise=0.0, modality=0.0, sensor=0.0),
    "noise_only": dict(d4=False, rot180=False, noise=0.02, modality=0.0, sensor=0.0),
    "sensor_dropout_only": dict(d4=False, rot180=False, noise=0.0, modality=0.0, sensor=0.10),
    "modality_dropout_only": dict(d4=False, rot180=False, noise=0.0, modality=0.15, sensor=0.0),
    "rotation_safe": dict(d4=False, rot180=True, noise=0.0, modality=0.0, sensor=0.0),
    "conservative": dict(d4=False, rot180=True, noise=0.02, modality=0.10, sensor=0.0),
    "original": dict(d4=True, rot180=False, noise=0.02, modality=0.15, sensor=0.05),
}
SUPERVISION = ("original", "geophysics_only", "conditional")


@dataclass
class TrainSpec:
    name: str
    supervision: str = "geophysics_only"
    feature_set: str = "geophysics"
    objective: str = "weighted_bce"
    negative_policy: str = "distance_masked"
    neg_weight: float = 1.0
    augmentation: str = "conservative"
    architecture: str = "g14_local"
    separate_targets: bool = True
    boundary_m: float = 300.0
    ignore_ring_m: float = 300.0
    reliable_negative_m: float = 1000.0
    known_dropout: float = 0.0                 # probability of dropping the whole known-context modality (M2)
    positive_center_frac: float = 0.5
    pos_weight: float = 8.0
    pu_prior: float = 0.01
    deep_supervision: float = 0.3
    hard_negative_frac: float = 0.0            # share of windows taken from model-selected hard negatives (M5)
    residual_champion: bool = False            # M6 (needs a nested champion map; see champion_weights)
    residual_lambda: float = 1.0
    notes: str = ""

    def validate(self) -> None:
        from .architectures import ARCHS
        from .losses import OBJECTIVES

        err = []
        if self.supervision not in SUPERVISION:
            err.append(f"supervision {self.supervision!r} not in {SUPERVISION}")
        if self.feature_set not in FEATURE_SETS:
            err.append(f"feature_set {self.feature_set!r} not in {sorted(FEATURE_SETS)}")
        if self.objective not in OBJECTIVES:
            err.append(f"objective {self.objective!r} not in {OBJECTIVES}")
        if self.augmentation not in AUGMENTATIONS:
            err.append(f"augmentation {self.augmentation!r} not in {sorted(AUGMENTATIONS)}")
        if self.architecture not in ARCHS:
            err.append(f"architecture {self.architecture!r} not in {ARCHS}")
        uses_known = "known" in FEATURE_SETS.get(self.feature_set, ((), None))[0]
        if self.supervision == "geophysics_only" and uses_known:
            err.append("geophysics_only supervision cannot use a feature set with known-context channels")
        if self.supervision == "conditional" and not uses_known:
            err.append("conditional supervision needs known-context channels (feature_set 'all' or 'known_only')")
        if self.negative_policy not in ("all_unlabeled", "distance_masked", "reduced_weight", "pu", "hard_negative"):
            err.append(f"unknown negative_policy {self.negative_policy!r}")
        if self.negative_policy == "pu" and self.objective != "nnpu":
            err.append("negative_policy 'pu' requires objective 'nnpu'")
        if err:
            raise ValueError(f"invalid TrainSpec {self.name}:\n  " + "\n  ".join(err))

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


# =============================================================================== channel subsets
class ChannelView:
    """A channel / modality subset of a ``g14.multiscale.UnitInputs`` with the same interface (window, kinds,
    modality_index, modalities, channel_names, n_channels) so G14 models, datasets and inference work unchanged."""

    def __init__(self, inputs, sources: tuple, modalities: tuple | None = None):
        self.inp = inputs
        chans = inputs.channels
        C = len(chans)
        keep = [i for i, c in enumerate(chans) if c.source in sources and (modalities is None or c.modality in modalities)]
        if "known" in sources and (modalities is None or "known_context" in modalities):
            if not inputs.known_names:
                raise ValueError("feature set needs known-context channels but the inputs have none "
                                 "(g14 cfg data.known_context=False or 'known_context' not in data.modalities)")
            keep += [C + j for j in range(len(inputs.known_names))]
        if not keep:
            raise ValueError(f"no channel matches sources={sources} modalities={modalities}")
        self.idx = keep
        full_mi = inputs.modality_index()
        sel = set(keep)
        self.modalities = [m for m in inputs.modalities if any(i in sel for i in full_mi.get(m, []))]
        self.mod_rows = [inputs.modalities.index(m) for m in self.modalities]
        pos = {i: k for k, i in enumerate(keep)}
        self._mi = {m: [pos[i] for i in full_mi[m] if i in pos] for m in self.modalities}
        self.kinds = [inputs.kinds[i] for i in keep]
        names = inputs.channel_names()
        self._names = [names[i] for i in keep]

    # interface used by models / datasets / inference
    def window(self, level, top, left, size):
        x, m = self.inp.window(level, top, left, size)
        return x[self.idx], m[self.mod_rows]

    def channel_names(self):
        return list(self._names)

    def modality_index(self):
        return {k: list(v) for k, v in self._mi.items()}

    @property
    def n_channels(self):
        return len(self.idx)

    @property
    def data(self):
        return self.inp.data

    @property
    def unit(self):
        return self.inp.unit

    @property
    def norm(self):
        return self.inp.norm

    @property
    def factors(self):
        return self.inp.factors

    @property
    def uses_known(self) -> bool:
        return "known_context" in self.modalities


# =============================================================================== dataset
class CorrectedDataset:
    """Aligned three-view samples with explicit supervision masks (``discovery.supervision_masks``)."""

    def __init__(self, view: ChannelView, sup: dict, soft: np.ndarray, windows: np.ndarray, cfg, spec: TrainSpec,
                 seed: int, augment: bool):
        d = cfg.data
        self.view, self.sup, self.soft, self.cfg, self.spec = view, sup, soft, cfg, spec
        self.L = d.local.size_px
        self.views = [(1, d.local.size_px), (int(round(d.regional.resolution_m / d.local.resolution_m)), d.regional.size_px),
                      (int(round(d.global_.resolution_m / d.local.resolution_m)), d.global_.size_px)]
        self.win = np.asarray(windows, int)
        self.seed, self.augment = seed, augment
        self.aug = AUGMENTATIONS[spec.augmentation]

    def __len__(self):
        return len(self.win)

    def _crop(self, a, t, l):
        H, W = a.shape
        out = np.zeros((self.L, self.L), np.float32)
        r0, c0, r1, c1 = max(t, 0), max(l, 0), min(t + self.L, H), min(l + self.L, W)
        if r1 > r0 and c1 > c0:
            out[r0 - t:r1 - t, c0 - l:c1 - l] = a[r0:r1, c0:c1]
        return np.nan_to_num(out)

    def __getitem__(self, i):
        t, l = int(self.win[i, 0]), int(self.win[i, 1])
        rng = np.random.default_rng((self.seed, i))
        cy, cx = t + self.L // 2, l + self.L // 2
        item = {}
        for name, (f, S) in zip(LEVEL_NAMES, self.views):
            x, m = self.view.window(f, cy // f - S // 2, cx // f - S // 2, S)
            item[name], item[f"{name}_mask"] = x, m
        keys = ("positive", "neg_weight", "loss_mask", "reliable_neg") + (("pos_w",) if "pos_w" in self.sup else ())
        tg = {k: self._crop(self.sup[k], t, l) for k in keys}
        tg["soft"] = self._crop(self.soft, t, l)
        if self.augment:
            a = self.aug
            k, flip = 0, False
            if a["d4"]:
                k, flip = int(rng.integers(4)), bool(rng.integers(2))
            elif a["rot180"]:
                k = 2 * int(rng.integers(2))
            if k or flip:
                for name in LEVEL_NAMES:
                    item[name] = transform_channels(item[name], self.view.kinds, k, flip)
                    item[f"{name}_mask"] = transform_scalars(item[f"{name}_mask"], k, flip)
                for kk in tg:
                    tg[kk] = transform_scalars(tg[kk][None], k, flip)[0]
            self._radiometric(item, rng)
        item["targets"] = tg
        item["meta"] = {"top": t, "left": l}
        return item

    def _radiometric(self, item, rng):
        a, spec = self.aug, self.spec
        mi = self.view.modality_index()
        mods = self.view.modalities
        scal = np.array([k == "scalar" for k in self.view.kinds])
        if a["noise"] > 0:
            for name in LEVEL_NAMES:
                x, m = item[name], item[f"{name}_mask"]
                for j, mod in enumerate(mods):
                    ch = [c for c in mi[mod] if scal[c]]
                    if ch:                                   # noise only where the modality is available
                        x[ch] += (rng.normal(0, a["noise"], x[ch].shape) * (m[j] > 0)).astype(np.float32)
        geo_mods = [m for m in mods if m != "known_context"]
        drop = []
        if geo_mods and rng.random() < a["modality"]:
            drop.append(geo_mods[int(rng.integers(len(geo_mods)))])
        if "known_context" in mods and rng.random() < spec.known_dropout:
            drop.append("known_context")
        for mod in drop:
            j = mods.index(mod)
            for name in LEVEL_NAMES:
                item[name][mi[mod]] = 0.0
                item[f"{name}_mask"][j] = 0.0
        if geo_mods and rng.random() < a["sensor"]:
            mod = geo_mods[int(rng.integers(len(geo_mods)))]
            j = mods.index(mod)
            h, w = int(rng.integers(8, self.L // 2)), int(rng.integers(8, self.L // 2))
            y0, x0 = int(rng.integers(-self.L // 2, self.L // 2 - h)), int(rng.integers(-self.L // 2, self.L // 2 - w))
            for name, (f, S) in zip(LEVEL_NAMES, self.views):
                cy = cx = S // 2
                ys, xs = cy + y0 // f, cx + x0 // f
                ye, xe = cy + (y0 + h) // f + 1, cx + (x0 + w) // f + 1
                item[name][mi[mod], max(ys, 0):ye, max(xs, 0):xe] = 0.0
                item[f"{name}_mask"][j, max(ys, 0):ye, max(xs, 0):xe] = 0.0


def soft_target(target: np.ndarray, radius_px: float) -> np.ndarray:
    from .discovery import dist_px

    return np.clip(1 - dist_px(target) / radius_px, 0, 1).astype(np.float32)


def sample_windows(region, target, L, step, min_frac, n, pos_frac, rng, hard: np.ndarray | None = None,
                   hard_frac: float = 0.0) -> np.ndarray:
    allw, posw = lattice_centres(region, target, L, step, min_frac)
    if not len(allw):
        raise ValueError("no admissible window: lower min_train_frac or check the region")
    n_pos = int(round(n * pos_frac)) if len(posw) else 0
    n_hard = int(round(n * hard_frac)) if hard is not None and len(hard) else 0
    parts = [posw[rng.integers(0, len(posw), n_pos)] if n_pos else np.empty((0, 2), int)]
    if n_hard:
        parts.append(hard[rng.integers(0, len(hard), n_hard)])
    parts.append(allw[rng.integers(0, len(allw), n - n_pos - n_hard)])
    return np.vstack(parts)


def champion_weights(target: np.ndarray, champion_rank: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """M6 residual emphasis (true labels stay the targets): positives the champion ranks low and negatives it ranks
    high are up-weighted. ``champion_rank`` MUST come from a champion that never saw the unit's held-out labels."""
    r = np.clip(np.nan_to_num(champion_rank), 0, 1)
    return (1 + lam * (1 - r)).astype(np.float32), (1 + lam * r).astype(np.float32)


# =============================================================================== training
def _infer_cfg(arch_cfg):
    return arch_cfg.updated({"run_mode": "TRAIN", "inference_enabled": True})


def inner_metrics(model, view, infer_cfg, unit, data, eval_ctx, pixel_m, tile_px, enriched_dl=None,
                  loss_fn=None, device=None, amp_dtype=None, log=print) -> dict:
    """Four inner-validation readings. Region DTI = ratio of summed official components over the inner region."""
    from ..g14.inference import predict_full
    from .metric_audit import ratio_of_sums_vs_mean_of_ratios, region_components, tiled_components

    out = {}
    region = unit.inner_val & data.valid
    truth = data.labels & unit.inner_val
    if region.any():
        r = predict_full(model, view, infer_cfg, region, mc_samples=0, tta=False, log=lambda *a: None)
        p = np.nan_to_num(r["prob"])
        ex = eval_ctx & region if eval_ctx is not None and (eval_ctx & region).any() else None
        c = region_components(p, truth, region, ex, (pixel_m, pixel_m))
        out.update(inner_full_region_dti=c.dti, inner_tp=c.tp, inner_fp=c.fp, inner_fn=c.fn, inner_n_truth=c.n_truth,
                   inner_mass=c.pred_mass)
        tb = tiled_components(p, truth, region, ex, (pixel_m, pixel_m), tile_px)
        if len(tb):
            rr = ratio_of_sums_vs_mean_of_ratios(tb)
            out.update(inner_tiled_dti=rr["ratio_of_sums"], inner_tiled_mean_of_ratios=rr["mean_of_ratios"],
                       inner_tiles=rr["n"])
    if enriched_dl is not None:
        import torch

        from ..g14.losses import dti_surrogate

        model.eval()
        s, n, ls = 0.0, 0, 0.0
        with torch.no_grad():
            for b in enriched_dl:
                bb = {k: v.to(device) for k, v in b.items() if k not in ("targets", "meta")}
                tg = {k: v.to(device) for k, v in b["targets"].items()}
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    o = model(bb)
                prob = torch.sigmoid(o["fault"][:, 0].float())
                s += float(1 - dti_surrogate(prob, tg["positive"], tg["loss_mask"], 300.0 / pixel_m))
                if loss_fn is not None:
                    ls += float(loss_fn(o, tg)[0])
                n += 1
        out.update(enriched_surrogate=s / max(n, 1), inner_loss=ls / max(n, 1) if loss_fn is not None else np.nan)
    return out


def fit_corrected(spec: TrainSpec, g14cfg, stack, data, unit, out_dir, tcfg, *, pixel_m: float = 100.0,
                  device_pref: str = "auto", amp: bool = True, num_workers: int = 2, link_m: float = 1000.0,
                  angle_deg: float = 30.0, champion_rank: np.ndarray | None = None, max_minutes: float = 60.0,
                  log=print, resume: bool = True) -> dict:
    """Train one spec on one unit; returns the manifest (also written to ``out_dir/manifests``)."""
    import torch
    from torch.utils.data import DataLoader

    from ..g14.multiscale import UnitInputs
    from ..g14.train import Checkpoint, amp_setup, lr_lambda, resolve_device, seed_everything, unit_seed
    from . import architectures as A
    from . import discovery as D
    from .leakage import set_context
    from .losses import diagnostics, objective, total

    spec.validate()
    if spec.residual_champion and champion_rank is None:
        raise ValueError(f"{spec.name}: residual_champion needs a nested champion rank map for unit {unit.key}")
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    seed = unit_seed(2026, f"{spec.name}:{unit.key}")
    seed_everything(seed, True)
    device = resolve_device(device_pref)
    arch_cfg = A.g14_cfg_for(spec.architecture, g14cfg)
    icfg = _infer_cfg(arch_cfg)
    R = 300.0 / pixel_m
    L = arch_cfg.data.local.size_px
    step = int(round(arch_cfg.data.global_.resolution_m / arch_cfg.data.local.resolution_m))

    # ---------------------------------------------------------- supervision family
    cu = None
    if spec.supervision == "original":
        u_in, eval_ctx, loss_ctx = unit, unit.permitted_known, np.zeros_like(unit.labels)
    elif spec.supervision == "geophysics_only":
        u_in = D.geophysics_only(unit)
        eval_ctx, loss_ctx = None, np.zeros_like(unit.labels)
    else:
        cu = D.ConditionalDiscoveryUnit.build(unit, link_m, angle_deg, pixel_m)
        u_in = D.with_context(unit, cu.eval_context())
        eval_ctx, loss_ctx = cu.eval_context(), None
    inputs = UnitInputs(stack, data, u_in, arch_cfg, seed)
    sources, mods = FEATURE_SETS[spec.feature_set]
    view = ChannelView(inputs, sources, mods)
    region = D.embargoed_train(unit, spec.boundary_m, pixel_m) if spec.separate_targets else unit.train
    region = region & data.valid
    if not spec.separate_targets:
        base_target = unit.labels & (unit.train | unit.inner_val)      # G14 combined raster (S03 kept for controls)
    else:
        base_target = unit.labels & unit.train

    model = A.build(spec.architecture, arch_cfg, view).to(device)
    n_params = A.count(model)
    schema = hashlib.sha256(json.dumps(view.channel_names()).encode()).hexdigest()[:12]
    ck = Checkpoint(out_dir / "checkpoints", f"{spec.name}__{unit.key}", spec.hash() + arch_cfg.semantic_hash(), schema)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=0.05)
    steps_per_epoch = max(1, tcfg.samples_per_epoch // (tcfg.batch_size * tcfg.accumulation))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg.epochs * steps_per_epoch, 0.05))
    amp_dtype, scaler = amp_setup(device, amp)

    def loss_fn(o, tg):
        terms, w = objective(spec.objective, o["fault"][:, 0], tg, radius_px=R, pos_weight=spec.pos_weight,
                             pu_prior=spec.pu_prior)
        tot = total(terms, w)
        if spec.deep_supervision and o.get("aux"):
            from .losses import bce

            tot = tot + spec.deep_supervision * sum(bce(a[:, 0], tg, spec.pos_weight) for a in o["aux"]) / len(o["aux"])
        return tot, terms, w

    start, best, best_epoch, bad, history, diag_log, draws = 0, -math.inf, -1, 0, [], [], []
    if resume:
        st = ck.load("latest", map_location=device)
        if st is not None:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optimizer"])
            sched.load_state_dict(st["scheduler"])
            start, best, best_epoch, bad, history = st["epoch"] + 1, st["best"], st["best_epoch"], st["bad"], st["history"]
            log(f"[g14.1] {spec.name}/{unit.key}: resumed at epoch {start}")
    hard = None
    for epoch in range(start, tcfg.epochs):
        if (time.perf_counter() - t_start) / 60 > max_minutes:
            log(f"[g14.1] {spec.name}/{unit.key}: time budget reached at epoch {epoch}")
            break
        rng = np.random.default_rng(seed + 31 * epoch)
        if cu is not None:
            ctx, tgt, rec = cu.draw(seed + epoch if tcfg.target_resample_every_epoch else seed)
            set_context(inputs, ctx, data)
            target, lctx = tgt & region, ctx
            draws.append({"epoch": epoch, **rec})
        else:
            target, lctx = base_target & region, loss_ctx
        combined = not spec.separate_targets and cu is None
        # combined (G14) construction: distances / ignore rings / soft targets from train | inner labels (S03)
        dist_src = base_target if combined else target
        sup = D.supervision_masks(region, data.valid, lctx, dist_src, unit.ignore, pixel_m, spec.ignore_ring_m,
                                  spec.reliable_negative_m, spec.negative_policy, spec.neg_weight)
        if spec.residual_champion:
            wp, wn = champion_weights(target, champion_rank, spec.residual_lambda)
            sup["neg_weight"] = sup["neg_weight"] * wn
            sup["pos_w"] = wp
        soft = soft_target(dist_src, R) * sup["loss_mask"]
        win = sample_windows(region & ~lctx, target, L, step, arch_cfg.data.min_train_frac, tcfg.samples_per_epoch,
                             spec.positive_center_frac, rng, hard, spec.hard_negative_frac)
        ds = CorrectedDataset(view, sup, soft, win, arch_cfg, spec, seed + epoch, augment=True)
        dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate,
                        drop_last=True, generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        sums, n, t0 = {}, 0, time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(dl):
            bb = {k: v.to(device, non_blocking=True) for k, v in b.items() if k not in ("targets", "meta")}
            tg = {k: v.to(device, non_blocking=True) for k, v in b["targets"].items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                o = model(bb)
            tot, terms, w = loss_fn(o, tg)
            if it % 50 == 0:
                diag_log.append({"epoch": epoch, "iter": it, **diagnostics(terms, w, o["fault"], tg)})
            loss = tot / tcfg.accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.name}/{unit.key}: non-finite loss at epoch {epoch} iter {it}")
            (scaler.scale(loss) if scaler is not None else loss).backward()
            if (it + 1) % tcfg.accumulation == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            sums["total"] = sums.get("total", 0.0) + float(tot.detach())
            n += 1
        rec = {"epoch": epoch, **{f"train_{k}": v / max(n, 1) for k, v in sums.items()},
               "samples_per_s": n * tcfg.batch_size / max(time.perf_counter() - t0, 1e-6), "lr": opt.param_groups[0]["lr"]}
        if spec.negative_policy == "hard_negative" and spec.hard_negative_frac > 0:
            hard = mine_hard_negatives(model, view, icfg, region & ~lctx, target, L, step, arch_cfg, device, rng)
        if epoch % tcfg.inner_eval_every == 0 or epoch == tcfg.epochs - 1:
            if cu is not None or (eval_ctx is not None and view.uses_known):
                set_context(inputs, eval_ctx, data)
            en = _enriched_loader(view, unit, data, eval_ctx, arch_cfg, spec, R, pixel_m, L, step, seed, tcfg)
            rec.update(inner_metrics(model, view, icfg, unit, data, eval_ctx, pixel_m, L, en,
                                     lambda o, tg: loss_fn(o, tg)[:1], device, amp_dtype))
            score = {"inner_full_region_dti": rec.get("inner_full_region_dti"), "inner_tiled_dti": rec.get("inner_tiled_dti"),
                     "inner_loss": -rec.get("inner_loss", np.inf), "enriched_surrogate": rec.get("enriched_surrogate")
                     }[tcfg.selection_metric]
            score = -math.inf if score is None or not np.isfinite(score) else float(score)
            if score > best:
                best, best_epoch, bad = score, epoch, 0
                ck.save("best", {"model": model.state_dict(), "epoch": epoch, "score": score})
            else:
                bad += 1
        history.append(rec)
        ck.save("latest", {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                           "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad": bad, "history": history})
        log(f"[g14.1] {spec.name}/{unit.key} e{epoch}: loss {rec.get('train_total', float('nan')):.4f} "
            f"inner {rec.get('inner_full_region_dti', float('nan')):.4f} (best e{best_epoch})")
        if bad >= tcfg.early_stopping_patience:
            break
    if best_epoch < 0:                    # no finite selection score (e.g. empty inner region): keep the last model
        ck.save("best", {"model": model.state_dict(), "epoch": len(history) - 1, "score": float("nan"),
                         "note": "no finite inner selection score; last epoch kept"})
    meta = {"spec": asdict(spec), "spec_hash": spec.hash(), "unit": unit.key, "task": unit.task, "zone": unit.zone,
            "architecture": spec.architecture, "arch_cfg_hash": arch_cfg.semantic_hash(), "schema": schema,
            "channels": view.channel_names(), "view_index": view.idx, "modalities": view.modalities,
            "normalizer": inputs.norm.to_json(), "parameters": n_params, "best_epoch": best_epoch, "best_score": best,
            "selection_metric": tcfg.selection_metric, "history": history, "draws": draws,
            "loss_diagnostics": diag_log[-200:], "seconds": time.perf_counter() - t_start,
            "pixel_m": pixel_m, "kernel_radius_px": R, "status": "UNTESTED_CODE_PATH"}
    p = out_dir / "manifests" / f"{spec.name}__{unit.key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=1, default=str))
    return meta


def _enriched_loader(view, unit, data, eval_ctx, cfg, spec, R, pixel_m, L, step, seed, tcfg):
    """G14-style enriched inner patches (positive_frac 0.5), kept ONLY as the 'patch diagnostic'."""
    from torch.utils.data import DataLoader

    from . import discovery as D

    region = unit.inner_val & data.valid
    truth = data.labels & unit.inner_val
    if not truth.any():
        return None
    ctx = np.zeros_like(truth) if eval_ctx is None else (eval_ctx & region)
    sup = D.supervision_masks(region, data.valid, ctx, truth, unit.ignore, pixel_m, spec.ignore_ring_m,
                              spec.reliable_negative_m, "distance_masked")
    try:
        win = sample_windows(region, truth, L, step, 0.3, 256, 0.5, np.random.default_rng(seed + 999))
    except ValueError:
        return None
    ds = CorrectedDataset(view, sup, soft_target(truth, R) * sup["loss_mask"], win, cfg, spec, seed + 999, augment=False)
    return DataLoader(ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)


def mine_hard_negatives(model, view, icfg, region, target, L, step, cfg, device, rng, n_pool: int = 512,
                        keep: float = 0.25) -> np.ndarray:
    """Windows far from every target whose mean predicted probability is highest (model-selected hard negatives)."""
    import torch

    from .discovery import dist_px

    allw, _ = lattice_centres(region & (dist_px(target) > 10), np.zeros_like(target), L, step, 0.6)
    if not len(allw):
        return np.empty((0, 2), int)
    pool = allw[rng.integers(0, len(allw), min(n_pool, len(allw)))]
    spec = TrainSpec("probe", supervision="original", feature_set="all", augmentation="none")
    ds = CorrectedDataset(view, {k: np.zeros(target.shape, np.float32) for k in
                                 ("positive", "neg_weight", "loss_mask", "reliable_neg")},
                          np.zeros(target.shape, np.float32), pool, cfg, spec, 0, augment=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), 16):
            b = collate([ds[j] for j in range(i, min(i + 16, len(ds)))])
            bb = {k: v.to(device) for k, v in b.items() if k not in ("targets", "meta")}
            scores.append(torch.sigmoid(model(bb)["fault"][:, 0].float()).mean((1, 2)).cpu().numpy())
    s = np.concatenate(scores)
    return pool[np.argsort(-s)[:max(1, int(keep * len(s)))]]


# =============================================================================== prediction
def load_trained(spec: TrainSpec, g14cfg, stack, data, unit, out_dir, context: np.ndarray | None, tag: str = "best",
                 pixel_m: float = 100.0):
    """(model, view, infer_cfg) for a trained (spec, unit), with the given context map for its known channels."""
    import torch

    from ..g14.contract import UnitNormalizer
    from ..g14.multiscale import UnitInputs
    from ..g14.train import Checkpoint
    from . import architectures as A
    from .discovery import with_context
    from .leakage import set_context

    meta = json.loads((Path(out_dir) / "manifests" / f"{spec.name}__{unit.key}.json").read_text())
    arch_cfg = A.g14_cfg_for(spec.architecture, g14cfg)
    u = with_context(unit, context if context is not None else np.zeros_like(unit.labels))
    inputs = UnitInputs(stack, data, u, arch_cfg, 0)
    nj = meta["normalizer"]
    inputs.norm = UnitNormalizer(np.asarray(nj["median"], np.float32), np.asarray(nj["iqr"], np.float32),
                                 nj["kinds"], nj["fitted_on"])
    sources, mods = FEATURE_SETS[spec.feature_set]
    view = ChannelView(inputs, sources, mods)
    if view.channel_names() != meta["channels"]:
        raise RuntimeError(f"{spec.name}/{unit.key}: channel layout differs from the trained model")
    if context is not None and view.uses_known:
        set_context(inputs, context, data)
    model = A.build(spec.architecture, arch_cfg, view)
    ck = Checkpoint(Path(out_dir) / "checkpoints", f"{spec.name}__{unit.key}", spec.hash() + arch_cfg.semantic_hash(),
                    meta["schema"])
    st = ck.load(tag)
    if st is None:
        raise FileNotFoundError(ck.path(tag))
    model.load_state_dict(st["model"])
    return model, view, _infer_cfg(arch_cfg)


def heldout_context(spec: TrainSpec, unit) -> np.ndarray | None:
    """Context at held-out prediction time: none for geophysics-only models; otherwise the unit's permitted known
    map (faults visible to the unit; never its held-out truth: asserted by g14 ``assert_no_leakage``)."""
    return None if spec.supervision == "geophysics_only" else unit.permitted_known


def predict_heldout(spec, g14cfg, stack, data, unit, out_dir, pixel_m=100.0, log=print) -> np.ndarray:
    from ..g14.inference import predict_full

    model, view, icfg = load_trained(spec, g14cfg, stack, data, unit, out_dir, heldout_context(spec, unit),
                                     pixel_m=pixel_m)
    r = predict_full(model, view, icfg, unit.heldout & data.valid, mc_samples=0, tta=False, log=log)
    p = Path(out_dir) / "maps" / f"{spec.name}__{unit.key}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, r["prob"])
    return r["prob"]
