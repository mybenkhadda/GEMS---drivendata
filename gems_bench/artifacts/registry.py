"""Formal model registry: baseline / candidate / validated / promoted / rejected.

Files under ``model_registry/`` (workspace root, mirrored to Drive)::

    registry.csv        current state, one row per model name
    transitions.jsonl   append-only log of every status change with its evidence
    evidence/<name>.json  latest evidence bundle (promotion decision, scorecard row, protocol hash)

Rules (docs/MODEL_REGISTRY.md):
* ``baseline`` is set once for pre-Phase-5 reference models and never changes automatically.
* ``candidate -> validated`` requires a complete scorecard (geographic AND pseudo-discovery, both
  known-fault policies) produced under the registered evaluation protocol.
* ``validated -> promoted`` requires a promotion decision with ``promote: True``
  (``evaluation.scoreboard.promotion_decision``); training loss, visual appeal or novelty are never evidence.
* Any status may move to ``rejected``; ``promoted -> validated`` demotes a former champion.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .atomic import atomic_save, atomic_write_json

STATUSES = ("baseline", "candidate", "validated", "promoted", "rejected")
ALLOWED = {
    None: {"baseline", "candidate"},
    "baseline": {"rejected"},
    "candidate": {"validated", "rejected"},
    "validated": {"promoted", "rejected", "candidate"},
    "promoted": {"validated", "rejected"},
    "rejected": {"candidate"},
}
COLUMNS = ["name", "status", "kind", "experiment_id", "oof", "discovery", "config_hash", "git_commit",
           "geo_dti", "disc_dti_excl", "disc_dti_retained_fp", "protocol_hash", "updated", "notes"]


class RegistryError(RuntimeError):
    pass


@dataclass
class ModelRegistry:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "evidence").mkdir(parents=True, exist_ok=True)

    @property
    def csv(self) -> Path:
        return self.root / "registry.csv"

    def table(self) -> pd.DataFrame:
        if not self.csv.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.csv)

    def get(self, name: str) -> dict | None:
        t = self.table()
        r = t[t.name == name]
        return None if r.empty else r.iloc[-1].to_dict()

    def status(self, name: str) -> str | None:
        r = self.get(name)
        return None if r is None else r["status"]

    # -- writes -------------------------------------------------------------
    def _write(self, row: dict) -> None:
        t = self.table()
        t = pd.concat([t[t.name != row["name"]], pd.DataFrame([{c: row.get(c) for c in COLUMNS}])], ignore_index=True)
        atomic_save(self.csv, lambda p: t.to_csv(p, index=False))

    def _log(self, entry: dict) -> None:
        with open(self.root / "transitions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def register(self, name: str, status: str, *, kind: str, experiment_id: str = "", oof: str = "",
                 discovery: dict | None = None, metrics: dict | None = None, config_hash: str = "",
                 git_commit: str = "", protocol_hash: str = "", notes: str = "", overwrite: bool = False) -> dict:
        if status not in ("baseline", "candidate"):
            raise RegistryError("new models enter the registry as 'baseline' or 'candidate'")
        prev = self.get(name)
        if prev is not None and not overwrite:
            return prev
        m = metrics or {}
        row = {"name": name, "status": status, "kind": kind, "experiment_id": experiment_id, "oof": oof,
               "discovery": json.dumps(discovery or {}), "config_hash": config_hash, "git_commit": git_commit,
               "geo_dti": m.get("geo_dti"), "disc_dti_excl": m.get("disc_dti_excl"),
               "disc_dti_retained_fp": m.get("disc_dti_retained_fp"), "protocol_hash": protocol_hash,
               "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "notes": notes}
        self._write(row)
        self._log({"name": name, "from": prev["status"] if prev else None, "to": status, "at": row["updated"],
                   "reason": "register", "evidence": m})
        return row

    def transition(self, name: str, to: str, *, reason: str, evidence: dict | None = None,
                   metrics: dict | None = None) -> dict:
        if to not in STATUSES:
            raise RegistryError(f"unknown status {to}")
        row = self.get(name)
        if row is None:
            raise RegistryError(f"{name} is not registered")
        frm = row["status"]
        if to not in ALLOWED.get(frm, set()):
            raise RegistryError(f"{name}: transition {frm} -> {to} not allowed")
        ev = evidence or {}
        if to == "validated":
            need = ("geo_dti", "disc_dti_excl", "disc_dti_retained_fp")
            have = {**{k: row.get(k) for k in need}, **(metrics or {})}
            missing = [k for k in need if have.get(k) is None or pd.isna(have.get(k))]
            if missing:
                raise RegistryError(f"{name}: validation needs a complete scorecard, missing {missing}")
        if to == "promoted":
            dec = ev.get("promotion_decision", {})
            if not dec.get("promote", False):
                raise RegistryError(f"{name}: promotion requires evidence['promotion_decision']['promote'] == True")
            for other in self.table().query("status == 'promoted'").name:
                if other != name:
                    self.transition(other, "validated", reason=f"demoted: {name} promoted")
        row.update({k: v for k, v in (metrics or {}).items() if k in COLUMNS})
        row.update(status=to, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._write(row)
        atomic_write_json(self.root / "evidence" / f"{name}.json", {"status": to, "reason": reason, **ev})
        self._log({"name": name, "from": frm, "to": to, "at": row["updated"], "reason": reason, "evidence": ev})
        return row

    def champion(self) -> dict | None:
        t = self.table()
        p = t[t.status == "promoted"]
        return None if p.empty else p.iloc[-1].to_dict()

    def history(self) -> pd.DataFrame:
        p = self.root / "transitions.jsonl"
        if not p.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()])
