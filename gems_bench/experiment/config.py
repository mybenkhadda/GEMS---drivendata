"""Experiment configuration: YAML with ``extends`` inheritance, dotted access and a semantic hash.

* ``extends: <path>`` (or a list) - base files resolved relative to the YAML, then to ``configs/``; deep merge,
  child wins, lists are replaced (not concatenated).
* ``config_hash`` covers everything that can change a result. Runtime knobs (``runtime``, ``drive``, ``notes``,
  ``description``) are excluded, so moving a run from a T4 to an A100 or changing the Drive path does not
  invalidate a resume; changing a loss weight, a channel or a fold design does.
* ``experiment_id`` = ``<name>__<hash8>`` unless ``id`` is set explicitly; a SCREEN / VALIDATE config (``fidelity``
  block, see ``pipeline.fidelity``) is ``<name>__<profile>__<hash8>``.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

NON_SEMANTIC = ("runtime", "drive", "notes", "description", "extends")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(ref: str, here: Path) -> Path:
    for cand in (here / ref, Path(ref)):
        if cand.exists():
            return cand.resolve()
    from ..paths import config_path

    p = config_path(ref) if not ref.startswith("configs/") else config_path(*Path(ref).parts[1:])
    if p.exists():
        return p
    raise FileNotFoundError(f"extends: {ref} not found (relative to {here} or configs/)")


def load_yaml_tree(path: Path, _seen: tuple = ()) -> dict:
    import yaml

    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"circular extends: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = raw.get("extends") or []
    bases = [bases] if isinstance(bases, str) else bases
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, load_yaml_tree(_resolve(b, path.parent), _seen + (path,)))
    return deep_merge(merged, {k: v for k, v in raw.items() if k != "extends"})


@dataclass
class ExperimentConfig:
    raw: dict
    source: Path | None = None
    _hash: str = field(default="", repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls(load_yaml_tree(Path(path)), Path(path))

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(copy.deepcopy(d))

    def get(self, dotted: str, default=None):
        cur = self.raw
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def kind(self) -> str:
        return self.raw["kind"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 2026))

    def semantic(self) -> dict:
        return {k: v for k, v in self.raw.items() if k not in NON_SEMANTIC}

    def hash(self) -> str:
        if not self._hash:
            s = json.dumps(self.semantic(), sort_keys=True, default=str)
            self._hash = hashlib.sha256(s.encode()).hexdigest()[:12]
        return self._hash

    @property
    def profile(self) -> str:
        """Fidelity profile (``pipeline.fidelity``); a config without a ``fidelity`` block is FULL."""
        return str((self.raw.get("fidelity") or {}).get("profile", "FULL")).upper()

    def experiment_id(self) -> str:
        if self.raw.get("id"):
            return self.raw["id"]
        if self.profile != "FULL":                      # a cheap run never shares a folder with a FULL run
            return f"{self.name}__{self.profile.lower()}__{self.hash()[:8]}"
        return f"{self.name}__{self.hash()[:8]}"

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True)
