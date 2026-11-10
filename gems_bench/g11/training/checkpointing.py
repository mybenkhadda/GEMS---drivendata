"""Atomic checkpoints with metadata, and save/reload parity checks."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, epoch: int = 0, metrics: dict | None = None,
                    meta: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "epoch": int(epoch), "metrics": metrics or {}, "meta": meta or {},
             "rng_torch": torch.get_rng_state()}
    for k, o in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if o is not None:
            state[k] = o.state_dict()
    tmp = p.with_name(p.name + ".partial")
    torch.save(state, tmp)
    os.replace(tmp, p)
    return p


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, strict: bool = True) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=strict)
    for k, o in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if o is not None and k in state:
            o.load_state_dict(state[k])
    return state


@torch.no_grad()
def reload_parity(model_factory, path, example: dict, atol: float = 1e-6) -> float:
    """Max |output(original) - output(reloaded)| on ``example`` (eval mode) - must be ~0."""
    a = model_factory()
    load_checkpoint(path, a)
    b = model_factory()
    load_checkpoint(path, b)
    a.eval()
    b.eval()
    oa, ob = a(**example), b(**example)
    oa = oa["logits"] if isinstance(oa, dict) else oa
    ob = ob["logits"] if isinstance(ob, dict) else ob
    return float((oa - ob).abs().max())
