"""Explicit promotion: a model becomes ``promoted`` only from a saved, passing promotion decision."""
from __future__ import annotations

import json

from ..artifacts.registry import ModelRegistry
from ..paths import ProjectPaths


def promote(name: str, experiment_id: str, paths: ProjectPaths | None = None) -> dict:
    paths = paths or ProjectPaths.discover()
    f = paths.experiments / experiment_id / "reports" / "promotion_decisions.json"
    if not f.exists():
        raise FileNotFoundError(f"no promotion decisions in {experiment_id}: run its evaluate stage first")
    decs = [d for d in json.loads(f.read_text(encoding="utf-8")) if d["candidate"] == name]
    if not decs:
        raise KeyError(f"{name} has no decision in {experiment_id}")
    dec = decs[-1]
    reg = ModelRegistry(paths.registry)
    if reg.status(name) == "candidate":
        raise RuntimeError(f"{name} is not validated yet (incomplete scorecard)")
    proto = json.loads((paths.experiments / experiment_id / "manifest.json").read_text(encoding="utf-8")).get("protocol", {})
    return reg.transition(name, "promoted", reason=f"promotion decision in {experiment_id}",
                          evidence={"promotion_decision": dec, "protocol": proto, "experiment_id": experiment_id})


def reject(name: str, reason: str, paths: ProjectPaths | None = None) -> dict:
    paths = paths or ProjectPaths.discover()
    return ModelRegistry(paths.registry).transition(name, "rejected", reason=reason)
