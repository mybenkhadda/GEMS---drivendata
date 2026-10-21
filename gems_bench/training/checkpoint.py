"""Resumable training checkpoints: atomic, rotated, integrity-checked, Drive-synchronised on a time budget.

Files under ``experiments/<id>/checkpoints/``::

    <unit>_ep<NNN>.pt     full resume state (model, optimizer, scheduler, scaler, EMA teacher, RNG states,
                          epoch, best-so-far state, history, current batch / grad_accum, spec hash)
    <unit>_latest.json    pointer to the newest complete resume file (written after the .pt is in place)
    <unit>_model.pt       final model (best inner-val state) + spec; the only file inference needs

Safety rules: a resume file is written atomically; old resume files are pruned only when a newer one exists
locally AND (without Drive) or (its Drive copy is verified); the final model is pushed before any resume
file is removed. A corrupt latest file falls back to the previous one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..artifacts.atomic import atomic_torch_save, atomic_write_json


class CheckpointManager:
    def __init__(self, directory: Path, unit: str, *, keep: int = 2, sync=None, sync_every_min: float = 10.0,
                 spec_hash: str = ""):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.unit, self.keep, self.sync, self.spec_hash = unit.replace("/", "_"), keep, sync, spec_hash
        self.sync_every_s = sync_every_min * 60
        self._last_sync = 0.0

    # -- paths ----------------------------------------------------------------
    def resume_files(self) -> list[Path]:
        return sorted(self.dir.glob(f"{self.unit}_ep*.pt"))

    @property
    def pointer(self) -> Path:
        return self.dir / f"{self.unit}_latest.json"

    @property
    def final_path(self) -> Path:
        return self.dir / f"{self.unit}_model.pt"

    def has_final(self) -> bool:
        return self.final_path.exists()

    # -- write ------------------------------------------------------------------
    def save(self, state: dict, epoch: int, force_sync: bool = False) -> Path:
        state = {**state, "spec_hash": self.spec_hash, "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
        p = atomic_torch_save(state, self.dir / f"{self.unit}_ep{epoch:03d}.pt")
        atomic_write_json(self.pointer, {"file": p.name, "epoch": epoch, "spec_hash": self.spec_hash})
        if self.sync is not None and (force_sync or time.time() - self._last_sync >= self.sync_every_s):
            self._push([p, self.pointer])
            self._last_sync = time.time()
        self._prune()
        return p

    def save_final(self, payload: dict) -> Path:
        p = atomic_torch_save({**payload, "spec_hash": self.spec_hash}, self.final_path)
        if self.sync is not None:
            self._push([p])
        return p

    def clear_resume(self) -> None:
        """Remove resume files once the final model exists (and is on Drive when syncing)."""
        if not self.has_final() or (self.sync is not None and not self.sync.is_synced(self.final_path)):
            return
        for f in self.resume_files():
            f.unlink(missing_ok=True)
        self.pointer.unlink(missing_ok=True)

    def _push(self, files) -> None:
        rels = [Path(f).resolve().relative_to(self.sync.local_root.resolve()).as_posix() for f in files]
        try:
            self.sync.push(rels, verify="sha256")
        except OSError as e:                                     # keep training; local copy is intact
            print(f"[ckpt] Drive push failed ({e}); will retry at the next checkpoint")

    def _prune(self) -> None:
        files = self.resume_files()
        for f in files[:-self.keep] if len(files) > self.keep else []:
            if self.sync is None or self.sync.is_synced(f) or self.sync.is_synced(files[-1]):
                f.unlink(missing_ok=True)

    # -- read -------------------------------------------------------------------
    def latest(self, map_location="cpu") -> dict | None:
        import torch

        cands = []
        if self.pointer.exists():
            try:
                cands.append(self.dir / json.loads(self.pointer.read_text(encoding="utf-8"))["file"])
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        cands += [f for f in reversed(self.resume_files()) if f not in cands]
        for f in cands:
            if not f.exists():
                continue
            try:
                s = torch.load(f, map_location=map_location, weights_only=False)
            except Exception as e:                               # truncated / corrupt: try the previous one
                print(f"[ckpt] unreadable {f.name}: {e}")
                continue
            if self.spec_hash and s.get("spec_hash") not in ("", None, self.spec_hash):
                raise RuntimeError(f"{f.name} was written by a different spec ({s.get('spec_hash')} != "
                                   f"{self.spec_hash}); refuse to resume an incompatible run")
            return s
        return None

    def load_final(self, map_location="cpu") -> dict:
        import torch

        return torch.load(self.final_path, map_location=map_location, weights_only=False)
