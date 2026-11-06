"""Training of one unit (geo0..2, disc0..2) through the E6 stages.

A  reference: dual-encoder ReferenceNet trained from scratch (terrain + field), early stopping on inner-val DTI
B  freeze:    the best reference is written once as the immutable ``reference.pt``
C  init:      E6Net initialised from the reference; parity check (DEM gate closed -> identical logits)
D  DEM:       only the DEM encoder and the DEM paths of the gates train; everything shared is frozen
E  joint:     controlled fine-tune of all E6Net weights (shared weights at lr x finetune_lr_mult) with 30 % DEM
              modality dropout; the frozen reference keeps every uncovered pixel exact
(F evaluation and G full-region inference are run by ``evaluator`` / ``pipeline``.)

Losses in D / E are restricted to DEM-covered pixels: elsewhere the output IS the frozen reference (no gradient).
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from . import losses, metric
from .cache import atomic_write_csv, atomic_write_json, read_json
from .checkpoint import CheckpointManager, rng_state, set_rng_state
from .config import stable_hash
from .datasets import E6Data, PatchSampler, UnitTargets
from .inference import bbox_of, predict, reference_fn, system_fn
from .model import E6Net, E6System, ReferenceNet, model_spec
from .paths import E6V2Paths
from .spatial_splits import unit_masks

log = logging.getLogger("e6v2.trainer")


class BudgetExhausted(RuntimeError):
    pass


def amp_dtype(cfg: dict, device: torch.device):
    mode = cfg["training"]["amp"]
    if device.type != "cuda" or mode in (False, "off", "none"):
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def to_device(b: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in b.items()}


def _e6_state(system_state: dict) -> dict:
    """E6Net weights from an E6System state dict (the reference part is the immutable file, not this one)."""
    return {k[3:]: v for k, v in system_state.items() if k.startswith("e6.")}


class UnitTrainer:
    def __init__(self, unit: str, cfg: dict, paths: E6V2Paths, data: E6Data, splits: dict, base: dict,
                 identity: dict, device: torch.device):
        self.unit, self.cfg, self.paths, self.data, self.device = unit, cfg, paths, data, device
        self.masks = unit_masks(unit, splits, base["labels"] & base["valid"], base["valid"])
        self.tg = UnitTargets(data, self.masks, cfg)
        seed = int(cfg["training"]["seed"]) + int(stable_hash(unit, 6), 16) % 10000
        self.seed = seed
        self.sampler = PatchSampler(data, self.tg, cfg, seed)
        self.spec = model_spec(cfg, data.schema)
        kind = "geo" if unit.startswith("geo") else "discovery"
        self.identity = {**identity, "unit": unit, "split_identity": splits["identity"][kind],
                         "split_inner": splits["identity"].get("discovery_inner") if kind == "discovery" else None,
                         "model_spec": stable_hash(self.spec), "prior": round(self.tg.prior, 8)}
        self.ckpt = CheckpointManager(paths.unit_dir(unit), self.identity)
        self.amp = amp_dtype(cfg, device)
        torch.manual_seed(seed)
        atomic_write_json(self.ckpt.dir / "unit_info.json", {"unit": unit, "identity": self.identity,
                                                             "targets": self.tg.stats, "sampler": self.sampler.stats})

    # ------------------------------------------------------------------
    def inner_val_dti(self, model_fn) -> float:
        iv = self.masks["inner_val"]
        if not iv.any():
            return float("nan")
        r0, r1, c0, c1 = bbox_of(iv)
        prob = predict(model_fn, self.data, (r0, r1, c0, c1), self.cfg, self.device, self.amp)
        c = metric.components(prob, self.masks["inner_truth"][r0:r1, c0:c1], iv[r0:r1, c0:c1],
                              px=float(self.data.schema["grid"]["pixel_m"]))
        return c.dti

    def _history_append(self, rows: list[dict]):
        p = self.ckpt.dir / "history.csv"
        old = pd.read_csv(p) if p.exists() else pd.DataFrame()
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["stage", "epoch"], keep="last")
        atomic_write_csv(p, df)

    def _fit(self, stage: str, model: torch.nn.Module, groups: list[dict], epochs: int, forward, eval_fn,
             dem_mode: bool, dropout: float, covered_only: bool) -> dict:
        t = self.cfg["training"]
        steps, accum, bs = int(t["steps_per_epoch"]), int(t["grad_accum"]), int(t["batch_size"])
        params = [p for g in groups for p in g["params"]]
        opt = torch.optim.AdamW(groups, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
        total = max(epochs * steps, 1)
        warm = max(int(0.05 * total), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        fp16 = self.amp is torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
        start, best, bad, best_state = 0, -float("inf"), 0, None
        ck = self.ckpt.load(stage, "last", map_location=self.device)
        if ck is not None:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            set_rng_state(ck["rng"])
            self.sampler.set_state(ck["sampler_rng"])
            start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad_epochs"]
            log.info("%s/%s: resuming after epoch %d (best inner-val DTI %.4f)", self.unit, stage, ck["epoch"], best)
        t_stage = time.time()
        budget = float(t["max_stage_minutes"]) * 60.0
        epoch = start - 1
        for epoch in range(start, epochs):
            t0 = time.time()
            model.train()
            agg = defaultdict(float)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                for _ in range(accum):
                    b = to_device(self.sampler.batch(bs, dem_mode, dropout), self.device)
                    if covered_only:
                        cov = (b["cov"] > 0).float()
                        for k in ("weight", "prox_w", "orient_w"):
                            b[k] = b[k] * cov
                    with torch.autocast(device_type=self.device.type, dtype=self.amp, enabled=self.amp is not None):
                        out = forward(b)
                    loss, info = losses.total(out, b, self.tg.prior, self.cfg)
                    scaler.scale(loss / accum).backward()
                    for k, v in info.items():
                        agg[k] += v / (steps * accum)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            model.eval()
            val = self.inner_val_dti(eval_fn)
            row = {"unit": self.unit, "stage": stage, "epoch": epoch, **agg, "inner_val_dti": val,
                   "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
            self._history_append([row])
            log.info("%s/%s epoch %d: loss %.4f inner-val DTI %.4f (%.0fs)", self.unit, stage, epoch, agg["loss"], val,
                     row["seconds"])
            if np.isfinite(val) and val > best + 1e-6:
                best, bad = val, 0
                self.ckpt.save(stage, "best", {"model": model.state_dict(), "epoch": epoch, "best": best})
            else:
                bad += 1
            self.ckpt.save(stage, "last", {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                           "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                                           "epoch": epoch, "step": (epoch + 1) * steps, "best": best,
                                           "bad_epochs": bad, "rng": rng_state(),
                                           "sampler_rng": self.sampler.get_state()})
            if epoch + 1 >= int(t["min_epochs"]) and bad >= int(t["patience"]):
                log.info("%s/%s: early stop at epoch %d", self.unit, stage, epoch)
                break
            if budget and time.time() - t_stage > budget and epoch + 1 < epochs:
                raise BudgetExhausted(f"{self.unit}/{stage}: max_stage_minutes reached after epoch {epoch}; "
                                      "re-run the cell to resume")
        bk = self.ckpt.load(stage, "best", map_location=self.device)
        if bk is None:
            raise RuntimeError(f"{self.unit}/{stage}: no best checkpoint (inner-val DTI never finite)")
        model.load_state_dict(bk["model"])
        extra = {"best_inner_val_dti": float(bk["best"]), "best_epoch": int(bk["epoch"]), "epochs_run": epoch + 1}
        self.ckpt.finalize(stage, model.state_dict(), extra)
        return extra

    # ------------------------------------------------------------------
    def run_reference(self) -> dict:
        """Stages A + B."""
        if self.ckpt.reference_exists():
            log.info("%s: immutable reference present - stage A/B skipped", self.unit)
            return read_json(self.ckpt.dir / "reference.json")
        ref = ReferenceNet(self.spec).to(self.device)
        if not self.ckpt.final_done("A"):
            fwd = lambda b: ref(b["terrain"], b["field"], b["ctx"])  # noqa: E731
            self._fit("A", ref, [{"params": list(ref.parameters())}], int(self.cfg["training"]["epochs"]["baseline"]),
                      fwd, reference_fn(ref), dem_mode=False, dropout=0.0, covered_only=False)
        st = self.ckpt.final_state("A", map_location=self.device)
        return self.ckpt.write_reference(st["model"], {"spec": self.spec, "best_inner_val_dti":
                                                       st.get("best_inner_val_dti"), "stage": "B"})

    def load_reference(self) -> tuple[ReferenceNet, str]:
        state, sha = self.ckpt.load_reference(map_location=self.device)
        ref = ReferenceNet(self.spec).to(self.device)
        ref.load_state_dict(state)
        ref.eval()
        return ref, sha

    @torch.no_grad()
    def parity(self, system: E6System, n: int = 8, gate_closed: bool = True) -> dict:
        """Max |E6 - reference| logit difference on fixed probe patches with the DEM gate closed (C) and exact
        equality of the system output where coverage == 0 (always)."""
        rng = np.random.default_rng(self.seed + 7)
        idx = rng.choice(len(self.sampler.all), min(n, len(self.sampler.all)), replace=False)
        P, ext = self.sampler.P, self.sampler.ext
        items = [self.data.inputs(int(r), int(c), P, ext) for r, c in self.sampler.all[idx]]
        b = to_device({k: np.stack([it[k] for it in items]).astype(np.float32) for k in items[0]}, self.device)
        system.eval()
        ref = system.reference(b["terrain"], b["field"], b["ctx"], aux=False)["logit"]
        out = {}
        if gate_closed:
            z = torch.zeros_like(b["cov"])
            e6 = system.e6(b["terrain"], b["field"], b["dem"], z, b["ctx"], aux=False)["logit"]
            out["gate_closed_max_abs_diff"] = float((e6 - ref).abs().max())
        sysout = system(b["terrain"], b["field"], b["dem"], b["cov"], b["cov"], b["ctx"], aux=False)["logit"]
        unc = b["cov"] <= 0
        out["uncovered_px"] = int(unc.sum())
        out["uncovered_exact"] = bool(torch.equal(sysout[unc], ref[unc]))
        return out

    def run_e6(self) -> dict:
        """Stages C, D, E. Requires the immutable reference."""
        ref, ref_sha = self.load_reference()
        e6 = E6Net(self.spec).to(self.device)
        system = E6System(e6, ref)
        t = self.cfg["training"]
        if not (self.ckpt.dir / "C_parity.json").exists():
            copied = e6.init_from_reference(ref)
            par = self.parity(system)
            par.update(copied_tensors=len(copied), reference_sha256=ref_sha)
            atomic_write_json(self.ckpt.dir / "C_parity.json", par)
            if par["gate_closed_max_abs_diff"] > 1e-3 or not par["uncovered_exact"]:
                raise RuntimeError(f"{self.unit}: stage C parity failed {par}")
            self.ckpt.save("C", "init", {"model": e6.state_dict()})
        par0 = read_json(self.ckpt.dir / "C_parity.json")
        if par0.get("reference_sha256") != ref_sha:
            raise RuntimeError(f"{self.unit}: C_parity.json was made from another reference")
        fwd = lambda b: system(b["terrain"], b["field"], b["dem"], b["cov"], b["gate_cov"], b["ctx"])  # noqa: E731
        ev = system_fn(system)
        if not self.ckpt.final_done("D"):
            init = self.ckpt.load("C", "init", map_location=self.device)
            e6.load_state_dict(init["model"])
            dem_names = {n for n, _ in e6.dem_parameters()}
            for n, p in e6.named_parameters():
                p.requires_grad_(n in dem_names)
            self._fit("D", system, [{"params": [p for n, p in e6.named_parameters() if n in dem_names]}],
                      int(t["epochs"]["dem"]), fwd, ev, dem_mode=True, dropout=0.0, covered_only=True)
        e6.load_state_dict(_e6_state(self.ckpt.final_state("D", map_location=self.device)["model"]))
        if not self.ckpt.final_done("E"):
            for p in e6.parameters():
                p.requires_grad_(True)
            dem_names = {n for n, _ in e6.dem_parameters()}
            lr = float(t["lr"])
            groups = [{"params": [p for n, p in e6.named_parameters() if n in dem_names], "lr": lr},
                      {"params": [p for n, p in e6.named_parameters() if n not in dem_names],
                       "lr": lr * float(t["finetune_lr_mult"])}]
            self._fit("E", system, groups, int(t["epochs"]["joint"]), fwd, ev, dem_mode=True,
                      dropout=float(t["dem_dropout"]), covered_only=True)
        fin = self.ckpt.final_state("E", map_location=self.device)
        e6.load_state_dict(_e6_state(fin["model"]))
        par = self.parity(system, gate_closed=False)
        par["reference_sha256"] = ref_sha
        atomic_write_json(self.ckpt.dir / "E_parity.json", par)
        if not par["uncovered_exact"]:
            raise RuntimeError(f"{self.unit}: uncovered pixels differ from the reference after stage E: {par}")
        return {"best_inner_val_dti": fin.get("best_inner_val_dti"), "parity": par}

    def load_system(self) -> tuple[E6System, str]:
        """Final E6System of this unit (identity-checked) and the sha256 of its stage-E checkpoint."""
        ref, _ = self.load_reference()
        e6 = E6Net(self.spec).to(self.device)
        e6.load_state_dict(_e6_state(self.ckpt.final_state("E", map_location=self.device)["model"]))
        sysm = E6System(e6, ref).to(self.device)
        sysm.eval()
        return sysm, read_json(self.ckpt.dir / "E_final.json")["sha256"]

    def is_complete(self) -> bool:
        return self.ckpt.reference_exists() and self.ckpt.final_done("E") and \
            (self.ckpt.dir / "E_parity.json").exists()
