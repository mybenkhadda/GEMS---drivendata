"""Stage declarations: what each pipeline stage needs, produces, fingerprints, costs, and how it is checked.

A ``Stage`` is pure declaration plus four callables taking the ``PipelineContext``:

``run(pc)``                 build / resume the outputs (must be idempotent: completed pieces are skipped)
``validate(pc, root)``      content-level ``artifacts.validation.Report`` of the outputs under ``root``
                            (the local workspace before running, or the Drive mirror for a read-only plan)
``fingerprint(pc)``         the configuration + input identities a cached result must match to be reused
``enabled(pc)``             (bool, reason): configuration switches (expensive stages are off by default)

``outputs(pc)`` / ``stage_in(pc)`` list workspace-relative paths pushed to / staged from Drive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# resource classes: what running the stage costs (the executor refuses some classes in some modes)
LIGHT, CPU, CPU_HEAVY, GPU, NETWORK, NETWORK_HEAVY = "light", "cpu", "cpu-heavy", "gpu", "network", "network-heavy"
EXPENSIVE = (GPU, NETWORK_HEAVY)

# what a missing output means for recovery
RAW_REQUIRED, REGENERABLE, RETRAIN, OPTIONAL = "raw_required", "regenerable", "retrain", "optional"

# cache / reuse policies
REUSE_VALID = "reuse_valid"          # valid outputs + matching fingerprint -> skip
ALWAYS_RUN = "always_run"            # cheap, idempotent (environment checks)
DELEGATED = "delegated"              # the stage's own engine resumes (experiments, feature store): run checks state


def _no_paths(pc) -> list[str]:
    return []


def _enabled(pc) -> tuple[bool, str]:
    return True, ""


def _no_fp(pc) -> dict:
    return {}


@dataclass
class Stage:
    name: str
    title: str
    deps: tuple[str, ...] = ()
    run: Callable | None = None
    validate: Callable | None = None
    fingerprint: Callable = _no_fp
    enabled: Callable = _enabled
    outputs: Callable = _no_paths            # workspace-relative, pushed to Drive after the stage
    stage_in: Callable = _no_paths           # workspace-relative, staged from Drive before validation / run
    inputs: Callable = _no_paths             # artifacts consumed (provenance only)
    soft_deps: tuple[str, ...] = ()          # used when available, never block
    resource: str = LIGHT
    requirement: str = REGENERABLE
    reuse: str = REUSE_VALID
    versioned: bool = False                  # outputs live under a config-derived (versioned) location
    recovery: str = ""                       # human instruction when the stage cannot complete
    section: str = ""                        # notebook section the stage belongs to
    est_minutes: str = ""
    critical: bool = False                   # data-preparation backbone: under failure_policy "interactive" a
                                             # failure stops the run (nothing downstream can be valid)

    @property
    def expensive(self) -> bool:
        return self.resource in EXPENSIVE


@dataclass
class StageRegistry:
    stages: dict[str, Stage] = field(default_factory=dict)

    def add(self, s: Stage) -> Stage:
        if s.name in self.stages:
            raise ValueError(f"duplicate stage {s.name}")
        self.stages[s.name] = s
        return s

    def __getitem__(self, name: str) -> Stage:
        try:
            return self.stages[name]
        except KeyError:
            raise KeyError(f"unknown stage {name!r}; stages: {list(self.stages)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self.stages

    def __iter__(self):
        return iter(self.stages.values())

    def names(self) -> list[str]:
        return list(self.stages)

    def check(self) -> None:
        """Every dependency is declared and the graph is acyclic."""
        from .dependencies import topological_order

        for s in self:
            for d in (*s.deps, *s.soft_deps):
                if d not in self.stages:
                    raise ValueError(f"stage {s.name}: unknown dependency {d!r}")
        topological_order(self)

    def table(self):
        import pandas as pd

        return pd.DataFrame([{"stage": s.name, "title": s.title, "deps": ", ".join(s.deps),
                              "soft_deps": ", ".join(s.soft_deps), "resource": s.resource,
                              "requirement": s.requirement, "reuse": s.reuse, "versioned": s.versioned,
                              "critical": s.critical, "est": s.est_minutes} for s in self])
