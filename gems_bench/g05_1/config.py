"""Notebook CONFIG / RUN handling, YAML loading and the deterministic run id."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import G051_VERSION

REPO = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO / "configs" / "g05_1" / "recovery.yaml"

NB_DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "main", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "local_root": "/content/gems_work", "run_id": None, "reuse_cached": True, "sync_to_drive": True,
    "save_png": True, "run_synthetic_tests": True, "backfill_missing_g02": False, "g05_run_id": None,
    "overrides": {},             # deep-merged into configs/g05_1/recovery.yaml
    "geomoe_overrides": {},      # deep-merged into configs/geomoe/discovery_geomoe.yaml (candidate machinery)
    "manual_upload_only": True,
}
# First run: preflight, atlas, recall, registry, graph. Everything that fits or writes rasters is opt-in.
RUN_DEFAULTS = {"preflight": True, "atlas": True, "recall": True, "registry": True, "displacement": False,
                "graph": True, "training": False, "strict": False, "selection": None, "ablations": False,
                "reconstruction": False, "export": False}
# the run id is a hash of everything that changes the candidate inventory / evaluation contract; stage-only
# settings (rankers, selection, ...) enter the stage fingerprints and earlier outputs are archived, never overwritten
RUN_ID_KEYS = ("version", "inputs", "target_systems")


class G051ConfigError(ValueError):
    pass


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def notebook(CONFIG: dict | None, RUN: dict | None) -> tuple[dict, dict]:
    c = {**NB_DEFAULTS, **(CONFIG or {})}
    r = {**RUN_DEFAULTS, **(RUN or {})}
    if r["selection"] is None:
        r["selection"] = bool(r["training"])
    errs = []
    if not c["manual_upload_only"]:
        errs.append("manual_upload_only must stay True: nothing is uploaded automatically")
    unknown = sorted(set(RUN or {}) - set(RUN_DEFAULTS))
    if unknown:
        errs.append(f"unknown RUN keys {unknown}")
    if r["selection"] and not r["training"]:
        errs.append("RUN['selection'] needs RUN['training'] (no model scores otherwise)")
    if r["export"] and not r["reconstruction"]:
        errs.append("RUN['export'] needs RUN['reconstruction']")
    if r["export"] and not r["selection"]:
        errs.append("RUN['export'] needs RUN['selection'] (promotion gates)")
    if r["ablations"] and not r["training"]:
        errs.append("RUN['ablations'] needs RUN['training']")
    if errs:
        raise G051ConfigError("; ".join(errs))
    return c, r


def load_yaml(overrides: dict | None = None) -> dict:
    import yaml

    with open(DEFAULT_YAML, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    if str(y.get("version")) != G051_VERSION:
        raise G051ConfigError(f"{DEFAULT_YAML.name} version {y.get('version')} != code {G051_VERSION}")
    return merge(y, overrides or {})


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def run_id(cfg: dict, geomoe_overrides: dict, ref: str) -> str:
    """Deterministic: ``g051-<sha1(version, inputs, targets, geomoe overrides, ref)[:10]>``."""
    payload = {**{k: cfg.get(k) for k in RUN_ID_KEYS}, "geomoe_overrides": geomoe_overrides or {}, "ref": ref}
    return "g051-" + hashlib.sha1(canonical(payload).encode()).hexdigest()[:10]
