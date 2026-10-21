"""Experiment manifest (``manifest.json``): identity, provenance, stage / unit state, produced maps.

Schema::

    experiment_id, name, kind, config_hash, git_commit, package_version, created, updated, seed,
    dataset      {raw file -> size / mtime fingerprint}, folds / discovery directories + fingerprints
    stages       {prepare|train|evaluate|export: {status, attempts, started, finished, error, outputs}}
    units        {cv_fold0|disc1|...: {status, finished, info}}   (resumable sub-steps of a stage)
    maps         {model name: {"oof": rel path, "disc": {split: rel path}}}  -> P5Context zoo
    sessions     [{started, env, gpu, git_commit}]                 one entry per Colab session
    protocol     evaluation protocol record + hash (written by the evaluate stage)

Stage status: pending | running | done | failed | interrupted. A "running" stage found at load time means the
previous session died; it is treated as interrupted and resumed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts.atomic import atomic_write_json

STAGES = ("prepare", "train", "evaluate", "export")


class IncompatibleResume(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class ExperimentManifest:
    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load_or_create(cls, path: Path, **identity) -> "ExperimentManifest":
        path = Path(path)
        if path.exists():
            m = cls(path, json.loads(path.read_text(encoding="utf-8")))
            for s, st in m.data.get("stages", {}).items():
                if st.get("status") == "running":
                    st["status"] = "interrupted"
            return m
        m = cls(path, {**identity, "created": _now(), "stages": {s: {"status": "pending", "attempts": 0} for s in STAGES},
                       "units": {}, "maps": {}, "sessions": []})
        m.save()
        return m

    def save(self) -> None:
        self.data["updated"] = _now()
        atomic_write_json(self.path, self.data)

    # -- compatibility ----------------------------------------------------
    def check_compatible(self, config_hash: str, git_commit: str, policy: str = "strict") -> list[str]:
        """``strict``: config hash must match and the code commit must match (dirty / unknown -> warning).
        ``config``: only the config hash must match. ``force``: warnings only (recorded)."""
        warn = []
        if self.data.get("config_hash") != config_hash:
            msg = f"config hash {config_hash} != manifest {self.data.get('config_hash')}"
            if policy != "force":
                raise IncompatibleResume(msg + ": use a new experiment id or resume_policy='force'")
            warn.append(msg)
        old = str(self.data.get("git_commit", "unknown"))
        if old != git_commit:
            msg = f"code commit {git_commit} != manifest {old}"
            if policy == "strict" and "unknown" not in (old, git_commit) and "+dirty" not in old + git_commit:
                raise IncompatibleResume(msg + ": install the recorded commit or resume_policy='config'")
            warn.append(msg)
        if warn:
            self.data.setdefault("resume_warnings", []).append({"at": _now(), "warnings": warn})
            self.save()
        return warn

    # -- stages / units -----------------------------------------------------
    def stage(self, name: str) -> dict:
        return self.data["stages"].setdefault(name, {"status": "pending", "attempts": 0})

    def set_stage(self, name: str, status: str, **kw) -> None:
        st = self.stage(name)
        if status == "running":
            st["attempts"] = st.get("attempts", 0) + 1
            st["started"] = _now()
            st.pop("error", None)
        if status in ("done", "failed", "interrupted"):
            st["finished"] = _now()
        st["status"] = status
        st.update(kw)
        self.save()

    def unit_done(self, unit: str) -> bool:
        return self.data["units"].get(unit, {}).get("status") == "done"

    def mark_unit(self, unit: str, info: dict | None = None, status: str = "done") -> None:
        self.data["units"][unit] = {"status": status, "finished": _now(), "info": info or {}}
        self.save()

    def add_maps(self, name: str, oof: str | None = None, disc: dict | None = None) -> None:
        self.data["maps"][name] = {"oof": oof, "disc": {str(k): v for k, v in (disc or {}).items()}}
        self.save()

    def add_session(self, env: dict) -> None:
        self.data.setdefault("sessions", []).append({"started": _now(), **env})
        self.save()
