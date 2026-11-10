"""Generic resumable trainer: AMP, gradient accumulation, clipping, cosine / plateau LR, early stopping on a
VALIDATION metric, per-epoch atomic checkpoints, OOM recovery (batch skipped + cache cleared, counted), history CSV.

The trainer never sees test data: callers pass train / validation loaders only; the best checkpoint is chosen on
``selection_metric`` from ``evaluate_fn(model, val_loader)``.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpointing import load_checkpoint, save_checkpoint
from .losses import masked_multitask_loss


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _forward(model, b):
    kw = {"x": b["x"]}
    if "dem" in b and hasattr(model, "dem"):
        kw.update(dem=b["dem"], dem_valid=b["dem_valid"], modality=b["modality"])
    out = model(**kw)
    return out if isinstance(out, dict) else {"logits": out}


class Trainer:
    def __init__(self, model, loss_fn, cfg: dict, out_dir, device=None, evaluate_fn=None):
        t = cfg["training"]
        self.model, self.loss_fn, self.cfg, self.t = model, loss_fn, cfg, t
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(int(t["epochs"]), 1)) \
            if t["scheduler"] == "cosine" else torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="max")
        self.amp = bool(t["amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.evaluate_fn = evaluate_fn
        self.history: list[dict] = []
        self.oom = 0

    def _resume(self) -> tuple[int, float]:
        last = self.dir / "last.pt"
        if last.is_file():
            st = load_checkpoint(last, self.model, self.opt, self.sched, self.scaler)
            hp = self.dir / "history.csv"
            self.history = pd.read_csv(hp).to_dict("records") if hp.is_file() else []
            best = max([h.get("val_metric", -np.inf) for h in self.history] or [-np.inf])
            return int(st["epoch"]) + 1, float(best)
        return 0, -np.inf

    def train_epoch(self, loader) -> dict:
        self.model.train()
        acc = int(self.t["grad_accum"])
        tot, n, parts_sum = 0.0, 0, {}
        self.opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            b = _to(batch, self.device)
            try:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    out = _forward(self.model, b)
                    loss, parts = masked_multitask_loss(out, b, self.loss_fn)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at batch {i}")
                self.scaler.scale(loss / acc).backward()
            except torch.cuda.OutOfMemoryError:
                self.oom += 1
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            if (i + 1) % acc == 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.t["clip_norm"]))
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            tot += float(loss.detach())
            n += 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
        return {"train_loss": tot / max(n, 1), **{f"train_{k}": v / max(n, 1) for k, v in parts_sum.items()},
                "oom_batches": self.oom}

    def fit(self, train_loader, val_loader, epochs: int | None = None, meta: dict | None = None) -> pd.DataFrame:
        start, best = self._resume()
        patience, bad = int(self.t["patience"]), 0
        for ep in range(start, int(epochs or self.t["epochs"])):
            t0 = time.time()
            rec = {"epoch": ep, **self.train_epoch(train_loader)}
            vm = self.evaluate_fn(self.model, val_loader) if self.evaluate_fn else {}
            rec.update({f"val_{k}": v for k, v in vm.items()})
            rec["val_metric"] = float(vm.get(self.t["selection_metric"].replace("val_", ""), np.nan)) if vm else np.nan
            rec["lr"] = self.opt.param_groups[0]["lr"]
            rec["seconds"] = time.time() - t0
            if isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.sched.step(rec["val_metric"] if np.isfinite(rec["val_metric"]) else 0.0)
            else:
                self.sched.step()
            self.history.append(rec)
            save_checkpoint(self.dir / "last.pt", self.model, self.opt, self.sched, self.scaler, ep, rec, meta)
            if np.isfinite(rec["val_metric"]) and rec["val_metric"] > best:
                best, bad = rec["val_metric"], 0
                save_checkpoint(self.dir / "best.pt", self.model, epoch=ep, metrics=rec, meta=meta)
            else:
                bad += 1
            tmp = self.dir / "history.csv.partial"
            pd.DataFrame(self.history).to_csv(tmp, index=False)
            tmp.replace(self.dir / "history.csv")
            if bad >= patience:
                break
        return pd.DataFrame(self.history)


@torch.no_grad()
def predict_loader(model, loader, device=None) -> list[dict]:
    dev = torch.device(device or next(model.parameters()).device)
    model.eval()
    outs = []
    for batch in loader:
        b = _to(batch, dev)
        o = _forward(model, b)
        outs.append({"prob": torch.sigmoid(o["logits"]).float().cpu().numpy(), "y": b["y"].cpu().numpy(),
                     "valid": b["valid"].cpu().numpy()})
    return outs


def pixel_eval_fn(model, loader) -> dict:
    """Validation metrics on valid pixels: PR-AUC (average precision) and ROC-AUC."""
    from ..evaluation.pixel_metrics import pr_auc, roc_auc

    outs = predict_loader(model, loader)
    if not outs:
        return {}
    p = np.concatenate([o["prob"][o["valid"] > 0] for o in outs])
    y = np.concatenate([o["y"][o["valid"] > 0] for o in outs])
    try:
        return {"pr_auc": pr_auc(y, p), "roc_auc": roc_auc(y, p), "n": int(y.size), "positives": int((y > 0.5).sum())}
    except ValueError:                        # no positive (or no negative) validation pixel: metric undefined
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "n": int(y.size), "positives": int((y > 0.5).sum())}
