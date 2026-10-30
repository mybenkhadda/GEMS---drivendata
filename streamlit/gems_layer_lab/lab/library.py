"""Feature library: versioned registry of saved formulas, dependency graph, import / export.

File: ``<outputs>/registry/feature_registry.json`` (atomic writes; every change keeps the previous version in
``history``)::

    {"schema": "gll-registry.1",
     "features": {"mag_grav_ratio": {"name": "mag_grav_ratio",
                                     "formula": "ratio(gradmag(rtp, 3), gradmag(iso_grav_anom, 3), 0.01)",
                                     "param_specs": {}, "parameters": {}, "raw_inputs": ["iso_grav_anom", "rtp"],
                                     "depends_on": [], "version": 1, "formula_hash": "...", "status": "experimental",
                                     "created": "...", "updated": "...", "description": "", "units": "...",
                                     "spatial_units": "sigma in pixels (1 px = 100 m)", "nodata": "...",
                                     "aoi_dependent": false, "halo_px": 14, "evaluations": [], "exports": [],
                                     "history": []}}}

A saved feature is evaluated by the same engine everywhere (app, ``lab.integration`` for P6 / M9, exporter): its
formula is compiled with its own PARAMS (``param_specs`` + frozen ``parameters``) and may reference raw layers and
other saved features. Cycles are refused when saving and detected when loading.
"""
from __future__ import annotations

import ast
import copy
import graphlib
import hashlib
import json
import os
import time
from pathlib import Path

SCHEMA = "gll-registry.1"
STATUSES = ("experimental", "candidate", "validated", "deprecated")
NODATA_NOTE = "NaN propagates; neighbourhood filters are NaN-normalised; invalid inputs never become values"


class LibraryError(ValueError):
    pass


def formula_names(formula: str) -> set:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise LibraryError(f"formula syntax: {e.msg}") from e
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def own_hash(formula: str, parameters: dict, param_specs: dict) -> str:
    tree = ast.dump(ast.parse(formula, mode="eval"), annotate_fields=False)
    return hashlib.sha256(json.dumps({"f": tree, "p": parameters, "s": param_specs}, sort_keys=True)
                          .encode()).hexdigest()[:16]


def params_literal(specs: dict, values: dict) -> str:
    """PARAMS source line of a saved feature (defaults = the saved values)."""
    if not specs:
        return ""
    d = {k: {"min": s["min"], "max": s["max"], "step": s["step"], "default": values.get(k, s["default"])}
         for k, s in specs.items()}
    return "PARAMS = " + json.dumps(d)


class FeatureLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = {"schema": SCHEMA, "features": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("schema") != SCHEMA:
                raise LibraryError(f"{self.path}: unsupported registry schema {self.data.get('schema')}")
            self.check_graph()

    # ------------------------------------------------------------------ persistence
    @property
    def features(self) -> dict:
        return self.data["features"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, name: str) -> dict:
        if name not in self.features:
            raise LibraryError(f"no saved feature {name!r}")
        return self.features[name]

    def program_source(self, name: str) -> str:
        e = self.get(name)
        head = params_literal(e.get("param_specs") or {}, e.get("parameters") or {})
        return (head + "\n" if head else "") + f"{name} = {e['formula']}"

    # ------------------------------------------------------------------ graph
    def edges(self) -> dict:
        return {n: set(e.get("depends_on") or []) for n, e in self.features.items()}

    def check_graph(self, edges: dict | None = None) -> list[str]:
        try:
            return list(graphlib.TopologicalSorter(edges if edges is not None else self.edges()).static_order())
        except graphlib.CycleError as e:
            raise LibraryError(f"circular feature dependency: {' -> '.join(map(str, e.args[1]))}") from e

    def closure(self, names) -> list[str]:
        """``names`` + every saved feature they depend on, in evaluation order."""
        need, todo = set(), list(names)
        while todo:
            n = todo.pop()
            if n in need or n not in self.features:
                continue
            need.add(n)
            todo += list(self.features[n].get("depends_on") or [])
        return [n for n in self.check_graph() if n in need]

    def dependents(self, name: str) -> list[str]:
        out, todo = set(), [name]
        while todo:
            n = todo.pop()
            for k, deps in self.edges().items():
                if n in deps and k not in out:
                    out.add(k)
                    todo.append(k)
        return sorted(out)

    def dot(self) -> str:
        """Graphviz DOT of the dependency graph (raw layers as boxes)."""
        lines = ["digraph features {", "rankdir=LR;", 'node [shape=ellipse, style=filled, fillcolor="#1f3b4d", '
                 'fontcolor=white];']
        raws = set()
        for n, e in self.features.items():
            lines.append(f'"{n}" [label="{n}\\nv{e.get("version", 1)} {e.get("status", "")}"];')
            for d in e.get("depends_on") or []:
                lines.append(f'"{d}" -> "{n}";')
            for r in e.get("raw_inputs") or []:
                raws.add(r)
                lines.append(f'"raw:{r}" -> "{n}";')
        for r in sorted(raws):
            lines.append(f'"raw:{r}" [label="{r}", shape=box, fillcolor="#3d3d3d"];')
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ edits
    def upsert(self, name: str, formula: str, *, raw_inputs, depends_on, param_specs=None, parameters=None,
               description="", status="experimental", units="", aoi_dependent=False, halo_px=0,
               kind="formula", plugin=None, source_program="") -> dict:
        """Create or update (new version when the content hash changes). Refuses cycles and unknown dependencies."""
        if not name.isidentifier() or name.startswith("_"):
            raise LibraryError(f"invalid feature name {name!r}")
        if status not in STATUSES:
            raise LibraryError(f"status must be one of {STATUSES}")
        missing = [d for d in depends_on if d not in self.features and d != name]
        if missing:
            raise LibraryError(f"{name} depends on unsaved features {missing}: save them first")
        if name in depends_on:
            raise LibraryError(f"{name} cannot reference itself")
        edges = {**self.edges(), name: set(depends_on)}
        self.check_graph(edges)
        h = own_hash(formula, parameters or {}, param_specs or {})
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        old = self.features.get(name)
        if old and old.get("formula_hash") == h:
            old.update(description=description or old.get("description", ""), status=status, units=units or
                       old.get("units", ""), updated=now)
            self.save()
            return old
        entry = {"name": name, "kind": kind, "formula": formula, "plugin": plugin, "param_specs": param_specs or {},
                 "parameters": parameters or {}, "raw_inputs": sorted(raw_inputs), "depends_on": sorted(depends_on),
                 "version": (old["version"] + 1) if old else 1, "formula_hash": h, "status": status,
                 "created": old["created"] if old else now, "updated": now, "description": description,
                 "units": units, "spatial_units": "sigma / size in pixels (1 px = 100 m on the GEMS grid)",
                 "nodata": NODATA_NOTE, "aoi_dependent": bool(aoi_dependent), "halo_px": int(halo_px),
                 "evaluations": old.get("evaluations", []) if old else [], "exports": old.get("exports", []) if old
                 else [], "source_program": source_program, "history": []}
        if old:
            entry["history"] = old.get("history", []) + [{k: old.get(k) for k in (
                "version", "formula", "parameters", "param_specs", "formula_hash", "updated", "status")}]
        self.features[name] = entry
        self.save()
        return entry

    def duplicate(self, name: str, new: str) -> dict:
        if new in self.features:
            raise LibraryError(f"{new} already exists")
        e = copy.deepcopy(self.get(name))
        e.update(name=new, version=1, history=[], evaluations=[], exports=[], created=time.strftime(
            "%Y-%m-%dT%H:%M:%S"), description=f"copy of {name} v{self.get(name)['version']}. " + e.get(
            "description", ""))
        self.features[new] = e
        self.save()
        return e

    def rename(self, old: str, new: str) -> list[str]:
        """Rename and rewrite references in dependents (each dependent gets a new version). Returns the dependents."""
        if new in self.features or not new.isidentifier():
            raise LibraryError(f"{new!r} exists or is not an identifier")
        e = self.get(old)
        deps = [d for d in self.features if old in (self.features[d].get("depends_on") or [])]
        self.features[new] = {**e, "name": new, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        del self.features[old]
        for d in deps:
            f = self.features[d]
            f["history"] = f.get("history", []) + [{"version": f["version"], "formula": f["formula"],
                                                    "updated": f["updated"], "note": f"before rename {old}->{new}"}]
            f["formula"] = rename_in_formula(f["formula"], old, new)
            f["depends_on"] = sorted(new if x == old else x for x in f["depends_on"])
            f["version"] += 1
            f["formula_hash"] = own_hash(f["formula"], f.get("parameters") or {}, f.get("param_specs") or {})
        self.check_graph()
        self.save()
        return deps

    def delete(self, name: str) -> None:
        dep = self.dependents(name)
        if dep:
            raise LibraryError(f"{name} is used by {dep}: delete or edit them first")
        self.get(name)
        del self.features[name]
        self.save()

    def set_status(self, name: str, status: str) -> None:
        if status not in STATUSES:
            raise LibraryError(f"status must be one of {STATUSES}")
        self.get(name)["status"] = status
        self.save()

    def record_evaluation(self, name: str, result: dict) -> None:
        e = self.get(name)
        e.setdefault("evaluations", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                "version": e["version"], **result})
        self.save()

    def record_export(self, name: str, path: str, manifest: str) -> None:
        e = self.get(name)
        e.setdefault("exports", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": e["version"],
                                            "formula_hash": e["formula_hash"], "path": path, "manifest": manifest})
        self.save()

    # ------------------------------------------------------------------ import / export
    def export_json(self, names=None) -> str:
        names = self.closure(names) if names else list(self.features)
        return json.dumps({"schema": SCHEMA, "features": {n: self.features[n] for n in names}}, indent=2, default=str)

    def import_json(self, text: str) -> dict:
        """Import entries; an existing name with different content becomes a new version. Returns {name: action}."""
        d = json.loads(text)
        if d.get("schema") != SCHEMA:
            raise LibraryError("not a gll-registry.1 export")
        inc = d["features"]
        edges = {**self.edges(), **{n: set(e.get("depends_on") or []) for n, e in inc.items()}}
        self.check_graph(edges)
        out = {}
        for n in graphlib.TopologicalSorter({k: v & set(inc) for k, v in
                                             {n: set(e.get("depends_on") or []) for n, e in inc.items()}.items()}
                                            ).static_order():
            e = inc[n]
            if n in self.features and self.features[n]["formula_hash"] == e.get("formula_hash"):
                out[n] = "unchanged"
                continue
            out[n] = "updated" if n in self.features else "added"
            self.upsert(n, e["formula"], raw_inputs=e.get("raw_inputs", []), depends_on=e.get("depends_on", []),
                        param_specs=e.get("param_specs"), parameters=e.get("parameters"),
                        description=e.get("description", ""), status=e.get("status", "experimental"),
                        units=e.get("units", ""), aoi_dependent=e.get("aoi_dependent", False),
                        halo_px=e.get("halo_px", 0), kind=e.get("kind", "formula"), plugin=e.get("plugin"))
        return out


def rename_in_formula(formula: str, old: str, new: str) -> str:
    """Replace Name tokens ``old`` by ``new`` (AST positions; strings / functions untouched)."""
    tree = ast.parse(formula, mode="eval")
    raw = formula.encode("utf-8")
    hits = sorted(((n.col_offset, n.end_col_offset) for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == old
                   and n.lineno == 1 and n.end_lineno == 1), reverse=True)
    if any(isinstance(n, ast.Name) and n.id == old and n.lineno != 1 for n in ast.walk(tree)):
        raise LibraryError("multi-line formula: rename manually")
    for a, b in hits:
        raw = raw[:a] + new.encode("utf-8") + raw[b:]
    return raw.decode("utf-8")
