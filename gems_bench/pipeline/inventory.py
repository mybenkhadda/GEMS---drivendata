"""Read-only inventory of every experiment folder: what exists, what is reusable, what is missing.

    from gems_bench.pipeline.inventory import experiment_inventory
    inv = experiment_inventory(pc)                  # or pipe.inventory()

One row per ``experiments/<id>/manifest.json`` (Drive when mounted, else the local workspace)::

    experiment | experiment_id | role | profile | configuration_hash | config_compatible | folds_done |
    discovery_done | evaluation_done | metrics_available | checkpoints | reusable | next_action | ...

* ``profile`` comes from the folder's ``config.yaml`` ``fidelity`` block; a folder without one is FULL (every run
  made before fidelity existed is a FULL run by construction: same YAML, same id).
* ``config_compatible``: the recorded config hash equals the hash the *current* code / configs / validation
  design would give the same experiment at the same profile (only with a pipeline context; else empty).
  An incompatible complete run is kept and reported as historical, never resumed or overwritten.
* Units come from the manifest (``cv_fold<k>`` / ``disc<d>``) and are cross-checked against the final model
  file of each unit (``checkpoints/<unit>_model.pt``) and its prediction raster. No checkpoint is loaded here.

Nothing is created, staged or written (the caller may save the returned table).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["experiment", "experiment_id", "role", "profile", "configuration_hash", "config_compatible",
           "folds_done", "folds_expected", "discovery_done", "discovery_expected", "train", "evaluation_done",
           "export_done", "metrics_available", "checkpoints", "maps", "complete", "reusable", "next_action",
           "git_commit", "updated", "gpu_train_min", "error"]


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_yaml(p: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _units(m: dict, prefix: str) -> list[int]:
    out = []
    for u, v in (m.get("units") or {}).items():
        mm = re.fullmatch(rf"{prefix}(\d+)", u)
        if mm and (v or {}).get("status") == "done":
            out.append(int(mm.group(1)))
    return sorted(out)


def _expected(cfg: dict, n_folds: int, n_splits: int) -> tuple[list[int], list[int]]:
    fid = cfg.get("fidelity") or {}
    folds = fid.get("geographic_folds")
    folds = list(range(n_folds)) if folds is None else [int(k) for k in folds]
    if cfg.get("kind") not in ("cnn5", None) or cfg.get("discovery", True) is False:
        splits: list[int] = [] if cfg.get("kind") == "cnn5" else list(range(n_splits))
    else:
        s = fid.get("discovery_splits")
        splits = list(range(n_splits)) if s is None else [int(d) for d in s]
    return folds, splits


def _gpu_minutes(d: Path) -> float:
    for name, col in (("runtime_profile", "train_s"), ("fold_metrics", "train_s")):
        p = d / "metrics" / f"{name}.csv"
        if p.exists():
            try:
                t = pd.read_csv(p)
                if col in t:
                    extra = 0.0
                    dr = d / "metrics" / "discovery_runs.csv"
                    if name == "fold_metrics" and dr.exists():
                        extra = float(pd.read_csv(dr).get("train_s", pd.Series(dtype=float)).sum())
                    return round((float(t[col].sum()) + extra) / 60.0, 1)
            except Exception:
                pass
    return float("nan")


def _compat_hash(pc, name: str, profile: str, cache: dict) -> str | None:
    if pc is None:
        return None
    key = (name, profile)
    if key not in cache:
        try:
            from .experiments import experiment_config

            cache[key] = experiment_config(pc, name, profile=profile).hash()
        except Exception:                                  # experiment no longer in configs/, invalid profile
            cache[key] = None
    return cache[key]


def inventory_root(pc=None, root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    if pc is not None:
        return pc.drive or pc.local
    from ..paths import ProjectPaths

    p = ProjectPaths.discover()
    return p.drive or p.root


def experiment_inventory(pc=None, root: Path | None = None, *, n_folds: int | None = None,
                         n_splits: int | None = None) -> pd.DataFrame:
    """Read-only table of every experiment folder under ``<root>/experiments`` (see module docstring)."""
    from ..experiment.manifest import STAGES
    from .experiments import role_of

    base = inventory_root(pc, root) / "experiments"
    nf = int(n_folds or (pc.cfg.get("validation.n_folds") if pc is not None else 3) or 3)
    ns = int(n_splits or (pc.cfg.get("validation.n_discovery") if pc is not None else 3) or 3)
    rows, cache = [], {}
    for mf in sorted(base.glob("*/manifest.json")) if base.exists() else []:
        d = mf.parent
        m = _read_json(mf)
        if m is None:
            rows.append({"experiment": d.name.split("__")[0], "experiment_id": d.name, "next_action":
                         "INSPECT (unreadable manifest.json)", "error": "unreadable manifest", "complete": False,
                         "reusable": False})
            continue
        cfg = _read_yaml(d / "config.yaml")
        name = m.get("name") or cfg.get("name") or d.name.split("__")[0]
        profile = str((cfg.get("fidelity") or {}).get("profile") or (m.get("fidelity") or {}).get("profile")
                      or "FULL").upper()
        st = {s: (m.get("stages", {}).get(s) or {}).get("status", "pending") for s in STAGES}
        exp_f, exp_s = _expected(cfg or {"kind": m.get("kind")}, nf, ns)
        f_done, s_done = _units(m, "cv_fold"), _units(m, "disc")
        ck = d / "checkpoints"
        finals = sorted(p.name.replace("_model.pt", "") for p in ck.glob("*_model.pt")) if ck.exists() else []
        # a unit counts only with its final model on disk (the manifest alone is not enough)
        f_ok = [k for k in f_done if f"cv_fold{k}" in finals or m.get("kind") != "cnn5"]
        s_ok = [k for k in s_done if f"disc{k}" in finals or m.get("kind") != "cnn5"]
        metrics = (d / "metrics" / ("scoreboard.csv" if profile == "FULL" else "fidelity_metrics.csv")).exists()
        maps = m.get("maps") or {}
        n_maps = sum(1 for v in maps.values() if v.get("oof") and (d / v["oof"]).exists())
        complete = all(st[s] == "done" for s in STAGES)
        h = _compat_hash(pc, name, profile, cache)
        compat = None if h is None else (h == m.get("config_hash"))
        missing_f = [k for k in exp_f if k not in f_ok]
        missing_s = [k for k in exp_s if k not in s_ok]
        failed = [s for s in STAGES if st[s] in ("failed", "interrupted")]
        if complete and compat is not False:
            nxt = "REPORT (complete; reuse results)" if profile == "FULL" else "REVIEW SHORTLIST (complete)"
        elif complete:
            nxt = "KEEP (historical: configuration changed since; not resumable)"
        elif compat is False:
            nxt = "KEEP (incomplete, incompatible configuration: start a new run id)"
        elif missing_f or missing_s:
            why = f" after {failed[0]} {st[failed[0]]}" if failed else ""
            nxt = f"RESUME train{why}: folds {missing_f or '-'} / splits {missing_s or '-'} missing"
        elif st["evaluate"] != "done":
            nxt = "RUN evaluate (CPU; training units complete)"
        else:
            nxt = "RUN export (CPU)"
        err = next(((m.get("stages", {}).get(s) or {}).get("error", "") for s in failed), "")
        rows.append({
            "experiment": name, "experiment_id": m.get("experiment_id", d.name), "role": role_of(name),
            "profile": profile, "configuration_hash": m.get("config_hash", ""), "config_compatible": compat,
            "folds_done": ",".join(map(str, f_ok)), "folds_expected": ",".join(map(str, exp_f)),
            "discovery_done": ",".join(map(str, s_ok)), "discovery_expected": ",".join(map(str, exp_s)),
            "train": st["train"], "evaluation_done": st["evaluate"] == "done", "export_done": st["export"] == "done",
            "metrics_available": metrics, "checkpoints": len(finals), "maps": n_maps, "complete": complete,
            "reusable": bool(complete and compat is not False), "next_action": nxt,
            "git_commit": str(m.get("git_commit", ""))[:12], "updated": m.get("updated", ""),
            "gpu_train_min": _gpu_minutes(d), "error": str(err)[:200],
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df["config_compatible"] = df["config_compatible"].astype(object)
        df = df.sort_values(["experiment", "profile", "updated"]).reset_index(drop=True)
    return df


def remaining_units(inv: pd.DataFrame) -> pd.DataFrame:
    """Training units still missing per incomplete, compatible experiment (for the dashboard)."""
    rows = []
    for r in inv.itertuples():
        if r.complete or r.config_compatible is False:
            continue
        ef = [x for x in str(r.folds_expected).split(",") if x not in ("", "nan")]
        df_ = [x for x in str(r.folds_done).split(",") if x not in ("", "nan")]
        es = [x for x in str(r.discovery_expected).split(",") if x not in ("", "nan")]
        ds = [x for x in str(r.discovery_done).split(",") if x not in ("", "nan")]
        rows.append({"experiment_id": r.experiment_id, "profile": r.profile,
                     "folds_remaining": len(set(ef) - set(df_)), "splits_remaining": len(set(es) - set(ds))})
    out = pd.DataFrame(rows, columns=["experiment_id", "profile", "folds_remaining", "splits_remaining"])
    out["units_remaining"] = out.folds_remaining + out.splits_remaining if len(out) else np.array([], int)
    return out


def last_checkpoint(root: Path) -> dict:
    """Newest checkpoint file under ``<root>/experiments`` (name, experiment, time); read-only."""
    base = Path(root) / "experiments"
    best = None
    for p in base.glob("*/checkpoints/*.pt") if base.exists() else []:
        t = p.stat().st_mtime
        if best is None or t > best[0]:
            best = (t, p)
    if best is None:
        return {"file": None}
    import time

    return {"file": best[1].name, "experiment_id": best[1].parent.parent.name,
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(best[0]))}
