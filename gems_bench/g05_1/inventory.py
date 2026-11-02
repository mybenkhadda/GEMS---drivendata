"""Stage A: preflight and dependency inventory (fails closed on any REQUIRED item).

``outputs/g05_1_system_recovery/<run_id>/preflight/dependency_inventory.csv`` - one row per dependency: component,
required, status (PASS / FAIL / ABSENT_OPTIONAL / WARN), detail, path. Includes the exact band resolution
(``bands.resolve``), the grid contract, the frozen-engine artifacts, and the optional G05 run chosen by manifest policy
(G05 source code is not in this repository: its outputs are read only through their manifest, schema discovered at
runtime, never assumed).
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from . import bands as BD

PASS, FAIL, OPT, WARN = "PASS", "FAIL", "ABSENT_OPTIONAL", "WARN"


def row(component, required, ok, detail="", path="", warn=False) -> dict:
    st = PASS if ok else (FAIL if required else (WARN if warn else OPT))
    return {"component": component, "required": bool(required), "status": st, "detail": str(detail)[:600],
            "path": str(path)}


def modules(required=("numpy", "pandas", "scipy", "rasterio", "lightgbm", "matplotlib", "yaml"),
            optional=("geopandas", "shapely", "pyarrow")) -> list[dict]:
    out = []
    for names, req in ((required, True), (optional, False)):
        for m in names:
            try:
                mod = importlib.import_module(m)
                out.append(row(f"python module {m}", req, True, getattr(mod, "__version__", "")))
            except Exception as e:                      # noqa: BLE001 - recorded, not raised
                out.append(row(f"python module {m}", req, False, f"{type(e).__name__}: {e}"))
    return out


def _manifest(d: Path) -> dict | None:
    for rel in ("manifests/run.json", "manifest.json", "run_manifest.json"):
        p = d / rel
        if p.exists():
            try:
                return {"file": rel, **json.loads(p.read_text(encoding="utf-8"))}
            except (OSError, json.JSONDecodeError):
                return {"file": rel, "unreadable": True}
    return None


def _complete(m: dict) -> bool:
    if not m or m.get("unreadable"):
        return False
    if str(m.get("status", "")).upper() in ("COMPLETE", "COMPLETED", "SUCCESS"):
        return True
    st = m.get("stages")
    if isinstance(st, dict) and st:
        return all(str(v).upper() in ("COMPLETE", "REUSED_VALID_CACHE", "SKIPPED_BY_VALID_DECISION")
                   for v in st.values())
    return False


def discover_g05(roots: list[Path], policy: str, override: str | None) -> dict:
    """Choose a G05 run. ``override`` wins (must exist); else ``latest_complete_manifest`` = the most recently
    written run dir whose manifest says complete; ``require_override`` refuses to guess."""
    runs = []
    for r in roots:
        if r.exists():
            for d in sorted(p for p in r.iterdir() if p.is_dir()):
                m = _manifest(d)
                runs.append({"run_id": d.name, "path": str(d), "manifest": (m or {}).get("file"),
                             "complete": _complete(m), "mtime": d.stat().st_mtime,
                             "written": (m or {}).get("written")})
    out = {"policy": policy, "override": override, "n_runs_seen": len(runs), "runs": runs}
    if override:
        hit = [x for x in runs if x["run_id"] == override]
        return {**out, "status": "SELECTED_OVERRIDE" if hit else "OVERRIDE_NOT_FOUND",
                "selected": hit[0] if hit else None}
    if policy == "require_override":
        return {**out, "status": "NO_OVERRIDE_GIVEN", "selected": None}
    ok = sorted((x for x in runs if x["complete"]), key=lambda x: (str(x.get("written") or ""), x["mtime"]))
    return {**out, "status": "SELECTED_LATEST_COMPLETE" if ok else "NO_COMPLETE_RUN", "selected": ok[-1] if ok else None}


def g05_schema(path: Path, max_files: int = 200) -> pd.DataFrame:
    """Files of a G05 run with their columns (csv / parquet headers only): runtime schema discovery."""
    rows = []
    for p in sorted(Path(path).rglob("*"))[:max_files]:
        if p.suffix not in (".csv", ".parquet", ".json"):
            continue
        cols = ""
        try:
            if p.suffix == ".csv":
                cols = ";".join(pd.read_csv(p, nrows=0).columns)
            elif p.suffix == ".parquet":
                import pyarrow.parquet as pq

                cols = ";".join(pq.read_schema(p).names)
        except Exception as e:                          # noqa: BLE001
            cols = f"unreadable: {type(e).__name__}"
        rows.append({"file": p.relative_to(path).as_posix(), "bytes": p.stat().st_size, "columns": cols})
    return pd.DataFrame(rows, columns=["file", "bytes", "columns"])


def engine_rows(G, cfg: dict) -> list[dict]:
    """Grid contract, exact bands and frozen-engine artifacts of the attached engine ``G``."""
    out = []
    d = G.ctx.data
    eg = cfg["inputs"]["expected_grid"]
    H, W = d.shape
    epsg = getattr(d.grid.crs, "to_epsg", lambda: None)()
    res = abs(float(d.profile["transform"].a))
    out.append(row("grid shape", True, (H, W) == (int(eg["height"]), int(eg["width"])), f"{H} x {W}"))
    out.append(row("grid CRS", True, epsg == int(eg["epsg"]), f"EPSG:{epsg}"))
    out.append(row("grid resolution", True, abs(res - float(eg["res_m"])) < 1e-6, f"{res} m"))
    out.append(row("feature band count", True, len(d.band_names) == int(eg["bands"]), f"{len(d.band_names)} bands"))
    out.append(row("template on the feature grid", True, d.template_matches(), "example_submission.tif"))
    for r in BD.resolution_table(cfg["inputs"]["raw_bands"], d.band_names):
        out.append(row(f"band {r['requested']} (exact)", True, r["status"] == "RESOLVED",
                       f"{r['status']} index {r['band_index']}; substring near misses NOT used: "
                       f"{r['substring_near_misses'] or '-'}"))
    w2 = G.bst.w2_inputs() if G.bst is not None else {}
    maps = w2.get("maps_sha256") or {}
    out.append(row("W2 cross-fitted maps", True, bool(maps), f"{len(maps)} maps"))
    g = G.bst.g02() if G.bst is not None else None
    out.append(row("G01 / G02 unit rasters", False, g is not None and bool(g.raster_units()),
                   f"units {sorted(g.raster_units()) if g is not None else []}", warn=True))
    fz = G.frozen_policy()
    out.append(row("frozen submitted W2 policy", True, bool(fz.get("cell")), f"{fz.get('cell')} ({fz.get('source')})"))
    return out


def fail_closed(rows: list[dict]) -> list[str]:
    return [f"{r['component']}: {r['detail']}" for r in rows if r["required"] and r["status"] == FAIL]
