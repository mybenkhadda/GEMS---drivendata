"""``Experiment``: one YAML -> one folder -> four restartable stages (prepare, train, evaluate, export).

    from gems_bench.experiment import Experiment
    exp = Experiment.from_yaml("configs/experiments/b2_nnpu.yaml")
    exp.prepare(); exp.train(); exp.evaluate(); exp.export()       # or exp.run_all()

* A completed stage is skipped (``force=True`` re-runs it). A stage interrupted by a disconnect or by the
  runtime budget restarts and its runner skips completed *units* (folds / splits) and resumes the current
  unit from its checkpoint.
* Every stage ends with a Drive push of the experiment folder (and the model registry), so nothing
  research-critical lives only on ``/content``.
* Resuming checks the config hash and code commit recorded in the manifest (``resume_policy``).
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import pandas as pd

from .._version import git_commit, provenance
from ..artifacts.atomic import atomic_write_text
from ..artifacts.registry import ModelRegistry
from ..artifacts.store import RunDir
from ..artifacts.sync import DriveSync
from ..paths import ProjectPaths
from .config import ExperimentConfig
from .manifest import STAGES, ExperimentManifest


def _dataset_fingerprint(paths: ProjectPaths) -> dict:
    from gems_eda.config import resolve_raw

    out = {}
    for k in ("features", "faults", "submission"):
        p = resolve_raw(paths.root, k)
        out[k] = {"file": p.name, "bytes": p.stat().st_size if p.exists() else None,
                  "mtime": int(p.stat().st_mtime) if p.exists() else None}
    return out


class Experiment:
    def __init__(self, config: ExperimentConfig, paths: ProjectPaths | None = None, *, resume_policy: str = "strict",
                 sync: bool | None = None, ctx=None):
        """``ctx``: an already loaded ``P5Context`` to share between experiments of one session (saves the
        ~1 min data / fold / split reload); it must be the default validation design of this workspace."""
        self.config = config
        self.paths = (paths or ProjectPaths.discover()).ensure()
        self.id = config.experiment_id()
        self.rd = RunDir(self.paths.experiments / self.id)
        self.drive = None
        use_sync = config.get("drive.sync", True) if sync is None else sync
        if use_sync and self.paths.drive is not None:
            self.drive = DriveSync(self.paths.root, self.paths.drive)
            self._stage_existing_from_drive()
        commit = git_commit()
        self.manifest = ExperimentManifest.load_or_create(
            self.rd.path / "manifest.json", experiment_id=self.id, name=config.name, kind=config.kind,
            config_hash=config.hash(), git_commit=commit, package_version=provenance()["version"], seed=config.seed,
            dataset=_dataset_fingerprint(self.paths), paths=self.paths.as_dict())
        self.warnings = self.manifest.check_compatible(config.hash(), commit, resume_policy)
        cfg_file = self.rd.path / "config.yaml"
        if not cfg_file.exists():
            atomic_write_text(cfg_file, config.to_yaml())
        from ..training.gpu import detect

        self.manifest.add_session({"git_commit": commit, "gpu": detect().get("name"), "host": os.uname().nodename
                                   if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "")})
        from .runners import get_runner

        self.runner = get_runner(config.kind)(self)
        self._ctx = ctx
        self.registry = ModelRegistry(self.paths.registry)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, root: str | Path | None = None, drive: str | Path | None = None,
                  code: str | Path | None = None, **kw) -> "Experiment":
        return cls(ExperimentConfig.from_yaml(path), ProjectPaths.discover(root, code, drive), **kw)

    @classmethod
    def resume(cls, experiment_id: str, root=None, drive=None, code=None, **kw) -> "Experiment":
        """Reopen an experiment from its saved config.yaml (staged from Drive when missing locally)."""
        paths = ProjectPaths.discover(root, code, drive)
        local = paths.experiments / experiment_id / "config.yaml"
        if not local.exists() and paths.drive is not None:
            DriveSync(paths.root, paths.drive).stage_in([f"experiments/{experiment_id}"])
        cfg = ExperimentConfig.from_yaml(local)
        cfg.raw.setdefault("id", experiment_id)
        return cls(cfg, paths, **kw)

    def _stage_existing_from_drive(self) -> None:
        rel = f"experiments/{self.id}"
        if (self.paths.drive / rel).exists():
            self.drive.stage_in([rel])

    # -- context ------------------------------------------------------------
    @property
    def ctx(self):
        if self._ctx is None:
            from ..models.zoo import register_aliases
            from ..phase5_context import load_context5

            prof = self.config.get("features.profile_config")
            self._ctx = load_context5(self.paths.root, paths5=self.paths,
                                      profile_cfg=self.paths.code / prof if prof else None)
            register_aliases(self._ctx)
        return self._ctx

    # -- stages ---------------------------------------------------------------
    def deadline(self) -> float | None:
        m = float(self.config.get("runtime.max_stage_minutes", 0) or 0)
        return time.monotonic() + 60 * m if m > 0 else None

    def _run_stage(self, name: str, force: bool = False):
        st = self.manifest.stage(name)
        if st.get("status") == "done" and not force:
            print(f"[{self.id}] {name}: done -> skip (force=True to re-run)")
            return st.get("outputs")
        i = STAGES.index(name)
        for prev in STAGES[:i]:
            if self.manifest.stage(prev).get("status") != "done" and getattr(self.runner, f"requires_{prev}", prev == "prepare"):
                print(f"[{self.id}] {name}: running required stage {prev} first")
                self._run_stage(prev)
        self.manifest.set_stage(name, "running")
        self.rd.log(f"stage {name} start (attempt {self.manifest.stage(name)['attempts']})")
        from ..training.trainer import TrainingInterrupted

        try:
            outputs = getattr(self.runner, name)() or {}
        except TrainingInterrupted as e:
            self.manifest.set_stage(name, "interrupted", error=str(e))
            self.rd.log(f"stage {name} interrupted: {e}")
            self.persist()
            raise
        except BaseException as e:                      # KeyboardInterrupt included: record, push, re-raise
            status = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
            self.manifest.set_stage(name, status, error=f"{type(e).__name__}: {e}",
                                    traceback=traceback.format_exc()[-4000:])
            self.rd.log(f"stage {name} {status}: {type(e).__name__}: {e}")
            self.persist()
            raise
        self.manifest.set_stage(name, "done", outputs=outputs)
        self.rd.log(f"stage {name} done")
        self.persist()
        return outputs

    def prepare(self, force: bool = False):
        return self._run_stage("prepare", force)

    def train(self, force: bool = False):
        return self._run_stage("train", force)

    def evaluate(self, force: bool = False):
        return self._run_stage("evaluate", force)

    def export(self, force: bool = False):
        out = self._run_stage("export", force)
        self.rd.finalize_contract(self.config.kind)
        self.persist()
        return out

    def run_all(self):
        for s in STAGES:
            getattr(self, s)()
        return self.status()

    # -- persistence ------------------------------------------------------------
    def persist(self) -> None:
        if self.drive is None:
            return
        try:
            self.drive.push([self.paths.rel(self.rd.path), self.paths.rel(self.paths.registry)])
        except OSError as e:
            print(f"[{self.id}] Drive push failed ({e}); re-run exp.persist() before the session ends")

    def status(self) -> pd.DataFrame:
        rows = [{"stage": s, **{k: v for k, v in self.manifest.stage(s).items() if k in
                                ("status", "attempts", "started", "finished", "error")}} for s in STAGES]
        return pd.DataFrame(rows)

    def units(self) -> pd.DataFrame:
        return pd.DataFrame([{"unit": u, **v} for u, v in self.manifest.data.get("units", {}).items()])

    def __repr__(self) -> str:
        st = ", ".join(f"{s}={self.manifest.stage(s).get('status')}" for s in STAGES)
        return f"Experiment({self.id}: {self.config.kind}; {st})"
