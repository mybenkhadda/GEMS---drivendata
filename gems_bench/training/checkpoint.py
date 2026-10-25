"""Resumable training checkpoints: atomic, integrity-checked, Drive-synchronised on a time budget.

Two retention policies (``policy``; experiments choose it under the non-semantic ``runtime.checkpoint``)

``rotating`` (historical)  ``<unit>_ep<NNN>.pt`` full resume files, the newest ``keep`` retained locally. Every
                           epoch file pushed to Drive stays there (Drive is never pruned by this module).
``best_and_latest``        fixed names, replaced in place, so Drive holds at most one resume file per unit::

    <unit>_latest.pt      full resume state (model, optimizer, scheduler, scaler, EMA teacher, NumPy / torch /
                          CUDA RNG, epoch, best-so-far state, history, current batch / grad_accum, spec hash)
    <unit>_latest.prev.pt the previous resume file, kept locally until the new one is written (and, when
                          syncing, verified on Drive); a crash between the two writes resumes from it
    <unit>_best.pt        model-only state of the best inner-validation epoch (inference / inspection)

Common to both::

    <unit>_latest.json    pointer to the newest complete resume file + its sha256 (written after the .pt)
    <unit>_model.pt       final model (best inner-val state, or the last state for fixed-epoch discovery runs)
                          + spec; the only file inference needs

Safety rules: every file is written atomically (temporary + fsync + rename); Drive copies go through
``atomic_copy`` (``.partial`` + sha256 verification + rename), so a Drive file is only ever replaced by a verified
newer copy; local resume files are removed only after the final model exists and (when syncing) is verified on
Drive; a resume file whose sha256 differs from its pointer, or that cannot be read, falls back to the previous one;
a resume file written by a different spec is refused (``RuntimeError``), never silently loaded.

Checkpoint selection: CV units keep the epoch with the best *finite* inner-validation score (NaN is never an
improvement). Pseudo-discovery units have no inner-validation zone by design (fixed epochs = median best CV
epoch, Phase-3 rule): their final model is the last epoch and no ``_best.pt`` is written.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..artifacts.atomic import atomic_torch_save, atomic_write_json, sha256_file

POLICIES = ("rotating", "best_and_latest")


class CheckpointManager:
    def __init__(self, directory: Path, unit: str, *, keep: int = 2, sync=None, sync_every_min: float = 10.0,
                 spec_hash: str = "", policy: str = "rotating"):
        if policy not in POLICIES:
            raise ValueError(f"checkpoint policy {policy!r} not in {POLICIES}")
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.unit, self.keep, self.sync, self.spec_hash = unit.replace("/", "_"), keep, sync, spec_hash
        self.policy = policy
        self.sync_every_s = sync_every_min * 60
        self._last_sync = 0.0
        self._best_saved_ep = None

    # -- paths ----------------------------------------------------------------
    def resume_files(self) -> list[Path]:
        """Legacy rotating files (``_ep<NNN>.pt``), oldest first."""
        return sorted(self.dir.glob(f"{self.unit}_ep*.pt"))

    @property
    def pointer(self) -> Path:
        return self.dir / f"{self.unit}_latest.json"

    @property
    def latest_path(self) -> Path:
        return self.dir / f"{self.unit}_latest.pt"

    @property
    def prev_path(self) -> Path:
        return self.dir / f"{self.unit}_latest.prev.pt"

    @property
    def best_path(self) -> Path:
        return self.dir / f"{self.unit}_best.pt"

    @property
    def final_path(self) -> Path:
        return self.dir / f"{self.unit}_model.pt"

    def has_final(self) -> bool:
        return self.final_path.exists()

    def has_resume(self) -> bool:
        return self.latest_path.exists() or self.prev_path.exists() or bool(self.resume_files())

    # -- write ------------------------------------------------------------------
    def save(self, state: dict, epoch: int, force_sync: bool = False) -> Path:
        state = {**state, "spec_hash": self.spec_hash, "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "ckpt_policy": self.policy}
        if self.policy == "rotating":
            p = atomic_torch_save(state, self.dir / f"{self.unit}_ep{epoch:03d}.pt")
            atomic_write_json(self.pointer, {"file": p.name, "epoch": epoch, "spec_hash": self.spec_hash})
            if self._sync_due(force_sync):
                self._push([p, self.pointer])
            self._prune()
            return p
        if self.latest_path.exists():                            # keep the previous complete state until
            os.replace(self.latest_path, self.prev_path)         # the new one is written (and verified)
        p = atomic_torch_save(state, self.latest_path)
        atomic_write_json(self.pointer, {"file": p.name, "epoch": epoch, "spec_hash": self.spec_hash,
                                         "sha256": sha256_file(p), "policy": self.policy})
        files = [p, self.pointer]
        best = self._save_best(state)
        if best is not None:
            files.append(best)
        if self._sync_due(force_sync):
            if self._push(files):
                self.prev_path.unlink(missing_ok=True)           # the verified Drive copy replaces it
        elif self.sync is None:
            self.prev_path.unlink(missing_ok=True)
        return p

    def _save_best(self, state: dict) -> Path | None:
        """Model-only best inner-val state, rewritten only when the best epoch changed."""
        bs, ep = state.get("best_state"), state.get("best_ep")
        if bs is None or ep is None or ep == self._best_saved_ep:
            return None
        p = atomic_torch_save({"state_dict": bs, "best_epoch": ep, "best_inner": state.get("best"),
                               "spec_hash": self.spec_hash}, self.best_path)
        self._best_saved_ep = ep
        return p

    def save_final(self, payload: dict) -> Path:
        p = atomic_torch_save({**payload, "spec_hash": self.spec_hash}, self.final_path)
        if self.sync is not None:
            self._push([p])
        return p

    def clear_resume(self) -> None:
        """Remove local resume files once the final model exists (and is verified on Drive when syncing).
        Drive copies are kept (this module never deletes on Drive)."""
        if not self.has_final() or (self.sync is not None and not self.sync.is_synced(self.final_path)):
            return
        for f in self.resume_files() + [self.latest_path, self.prev_path, self.best_path]:
            f.unlink(missing_ok=True)
        self.pointer.unlink(missing_ok=True)

    def _sync_due(self, force: bool) -> bool:
        if self.sync is None:
            return False
        if force or time.time() - self._last_sync >= self.sync_every_s:
            self._last_sync = time.time()
            return True
        return False

    def _push(self, files) -> bool:
        rels = [Path(f).resolve().relative_to(self.sync.local_root.resolve()).as_posix() for f in files]
        try:
            self.sync.push(rels, verify="sha256")
            return True
        except OSError as e:                                     # keep training; local copy is intact
            print(f"[ckpt] Drive push failed ({e}); will retry at the next checkpoint")
            return False

    def _prune(self) -> None:
        files = self.resume_files()
        for f in files[:-self.keep] if len(files) > self.keep else []:
            if self.sync is None or self.sync.is_synced(f) or self.sync.is_synced(files[-1]):
                f.unlink(missing_ok=True)

    # -- read -------------------------------------------------------------------
    def _candidates(self) -> list[tuple[Path, str | None]]:
        cands: list[tuple[Path, str | None]] = []
        if self.pointer.exists():
            try:
                ptr = json.loads(self.pointer.read_text(encoding="utf-8"))
                cands.append((self.dir / ptr["file"], ptr.get("sha256")))
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        seen = {c[0] for c in cands}
        for f in [self.latest_path, self.prev_path] + list(reversed(self.resume_files())):
            if f not in seen:
                cands.append((f, None))
                seen.add(f)
        return cands

    def latest(self, map_location="cpu") -> dict | None:
        import torch

        for f, sha in self._candidates():
            if not f.exists():
                continue
            if sha and sha256_file(f) != sha:                    # pointer says otherwise: truncated / stale copy
                print(f"[ckpt] {f.name}: sha256 differs from {self.pointer.name}; trying the previous state")
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

    def final_ok(self, map_location="cpu") -> bool:
        """Integrity of the final model: readable, written by this spec (when known), has a state_dict."""
        if not self.has_final():
            return False
        try:
            obj = self.load_final(map_location)
        except Exception:
            return False
        if self.spec_hash and obj.get("spec_hash") not in ("", None, self.spec_hash):
            return False
        return "state_dict" in obj
