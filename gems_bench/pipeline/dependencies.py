"""Dependency graph of the pipeline: topological order, ancestor closure, mode targets, blocking reasons."""
from __future__ import annotations

from .registry import StageRegistry


class CycleError(ValueError):
    pass


def topological_order(reg: StageRegistry, names: list[str] | None = None) -> list[str]:
    """Stable topological order (declaration order breaks ties); soft deps order but never require."""
    names = list(names) if names is not None else reg.names()
    keep = set(names)
    order, state = [], {}

    def visit(n: str, path: tuple[str, ...]) -> None:
        if state.get(n) == "done":
            return
        if state.get(n) == "active":
            raise CycleError(f"dependency cycle: {' -> '.join(path + (n,))}")
        state[n] = "active"
        s = reg[n]
        for d in (*s.deps, *s.soft_deps):
            if d in keep:
                visit(d, path + (n,))
        state[n] = "done"
        order.append(n)

    for n in reg.names():
        if n in keep:
            visit(n, ())
    return order


def ancestors(reg: StageRegistry, name: str, soft: bool = False) -> set[str]:
    out, todo = set(), [name]
    while todo:
        s = reg[todo.pop()]
        for d in (*s.deps, *(s.soft_deps if soft else ())):
            if d not in out:
                out.add(d)
                todo.append(d)
    return out


def descendants(reg: StageRegistry, name: str) -> set[str]:
    out = set()
    changed = True
    while changed:
        changed = False
        for s in reg:
            if s.name not in out and (name in s.deps or out.intersection(s.deps)):
                out.add(s.name)
                changed = True
    return out


def closure(reg: StageRegistry, targets: list[str]) -> list[str]:
    """Targets plus every hard ancestor, in topological order."""
    need = set(targets)
    for t in targets:
        need |= ancestors(reg, t)
    return topological_order(reg, [n for n in reg.names() if n in need])


# Mode -> target stages (executor adds ancestors). Ancestors of SETUP_ONLY / SUBMISSION_ONLY targets are
# validated and staged but not built (``ANCESTORS_VALIDATE_ONLY``).
MODE_TARGETS = {
    "SETUP_ONLY": ["bootstrap", "raw_data"],
    # PREPARE_DATA stops before any model training (baselines / advanced_models are validate-only there)
    "PREPARE_DATA": ["native_grid", "dem_inventory", "geological_eda", "segment_attributes", "spatial_folds",
                     "discovery_splits", "feature_store", "dem_pilot"],
    # the EDA stages are soft dependencies of feature_store / baselines, so they are named here explicitly
    # (cached -> SKIP_CACHED in seconds; never silently left out of a mode that needs eda:* features)
    "BASELINES": ["geological_eda", "segment_attributes", "feature_store", "baselines", "discovery_validation",
                  "reports"],
    "ADVANCED_RESEARCH": ["geological_eda", "segment_attributes", "feature_store", "baselines", "advanced_models",
                          "discovery_validation", "reports"],
    "FULL_PIPELINE": None,           # every stage
    "RESUME": None,                  # every stage; expensive ones only when previously started (see executor)
    "SUBMISSION_ONLY": ["full_inference", "submission"],
}
ANCESTORS_VALIDATE_ONLY = ("SETUP_ONLY", "SUBMISSION_ONLY")


def mode_targets(reg: StageRegistry, mode: str) -> list[str]:
    t = MODE_TARGETS[mode]
    return reg.names() if t is None else [n for n in t if n in reg]


def blocking(reg: StageRegistry, name: str, status: dict[str, str], usable: set[str]) -> list[str]:
    """Hard dependencies of ``name`` that are not usable (no valid output), with their status."""
    return [f"{d} ({status.get(d, 'not run')})" for d in reg[name].deps if d not in usable]
