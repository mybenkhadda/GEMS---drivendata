"""Resumable checkpoints with identity checks.

Per unit and stage: ``checkpoints/<unit>/<stage>_last.pt`` (every epoch), ``<stage>_best.pt`` (best inner-val
DTI), ``<stage>_final.pt`` + ``<stage>_final.json`` (completion marker with sha256). A checkpoint stores model,
optimizer, scheduler, AMP scaler, epoch / step, the RNG states (python, numpy, torch, cuda, patch sampler), the
identity (schema hash, raw fingerprint, dataset fingerprint, normalisation hash, split identity, model spec,
training config, code version), the best validation value and the history. A checkpoint whose identity differs
from the current one is never loaded: ``IdentityMismatch`` names the differing keys and the remedy.

The reference model (``reference.pt`` + ``reference.json``) is immutable: once written it is never overwritten;
a different identity raises.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from .cache import atomic_write_json, read_json, sha256_file


class IdentityMismatch(RuntimeError):
    pass


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _diff(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, unit_dir: Path, identity: dict):
        self.dir = Path(unit_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def path(self, stage: str, kind: str) -> Path:
        return self.dir / f"{stage}_{kind}.pt"

    def _check(self, ident: dict, where: str):
        d = _diff(ident, self.identity)
        if d:
            raise IdentityMismatch(f"{where}: identity differs in {d}. The inputs / config / code changed since this "
                                   f"checkpoint was written. Delete {self.dir} to retrain this unit.")

    # ---- per-stage --------------------------------------------------------
    def final_done(self, stage: str) -> bool:
        js = read_json(self.dir / f"{stage}_final.json")
        if js is None or not self.path(stage, "final").exists():
            return False
        self._check(js["identity"], f"{stage}_final.json")
        return js["sha256"] == sha256_file(self.path(stage, "final"))

    def save(self, stage: str, kind: str, payload: dict):
        atomic_torch_save({**payload, "identity": self.identity, "stage": stage}, self.path(stage, kind))

    def load(self, stage: str, kind: str, map_location="cpu") -> dict | None:
        p = self.path(stage, kind)
        if not p.exists():
            return None
        ck = torch.load(p, map_location=map_location, weights_only=False)
        self._check(ck["identity"], p.name)
        return ck

    def finalize(self, stage: str, model_state: dict, extra: dict):
        atomic_torch_save({"model": model_state, "identity": self.identity, "stage": stage, **extra},
                          self.path(stage, "final"))
        atomic_write_json(self.dir / f"{stage}_final.json",
                          {"identity": self.identity, "sha256": sha256_file(self.path(stage, "final")),
                           **{k: v for k, v in extra.items() if isinstance(v, (int, float, str, list, dict))}})

    def final_state(self, stage: str, map_location="cpu") -> dict:
        if not self.final_done(stage):
            raise FileNotFoundError(f"{self.dir.name}: stage {stage} not finalised")
        return torch.load(self.path(stage, "final"), map_location=map_location, weights_only=False)

    # ---- immutable reference ---------------------------------------------
    @property
    def reference_path(self) -> Path:
        return self.dir / "reference.pt"

    def reference_exists(self) -> bool:
        js = read_json(self.dir / "reference.json")
        if js is None:
            return False
        self._check(js["identity"], "reference.json")
        if sha256_file(self.reference_path) != js["sha256"]:
            raise IdentityMismatch(f"{self.reference_path} checksum differs from reference.json - the immutable "
                                   "reference was modified. Delete the unit folder to retrain.")
        return True

    def write_reference(self, state: dict, extra: dict) -> dict:
        if (self.dir / "reference.json").exists():
            raise IdentityMismatch(f"{self.dir}/reference.json exists: the reference is immutable")
        atomic_torch_save({"model": state, "identity": self.identity, **extra}, self.reference_path)
        rec = {"identity": self.identity, "sha256": sha256_file(self.reference_path),
               **{k: v for k, v in extra.items() if isinstance(v, (int, float, str, list, dict))}}
        atomic_write_json(self.dir / "reference.json", rec)
        try:
            os.chmod(self.reference_path, 0o444)
        except OSError:
            pass
        return rec

    def load_reference(self, map_location="cpu") -> tuple[dict, str]:
        if not self.reference_exists():
            raise FileNotFoundError(f"{self.dir.name}: no reference model")
        js = read_json(self.dir / "reference.json")
        ck = torch.load(self.reference_path, map_location=map_location, weights_only=False)
        return ck["model"], js["sha256"]
