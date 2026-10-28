"""Milestone 1 configuration (``configs/leaderboard_focus/m1.yaml``) and the shared status vocabulary (pure).

Stage statuses: COMPLETE (every check passed), BLOCKED (a check failed or a required artifact is missing: the reason is
recorded), INCOMPLETE (an upstream stage has not run), FAILED (an exception; traceback recorded). The submission
gate adds READY_FOR_UPLOAD, which only ``submission.export`` may assign.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

M1_VERSION = "m1.1"
M1_CODE_REV = "m1.1-r1"                          # checked by every notebook start-up cell (not fingerprinted)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "leaderboard_focus" / "m1.yaml"
COMPLETE, BLOCKED, INCOMPLETE, FAILED = "COMPLETE", "BLOCKED", "INCOMPLETE", "FAILED"
READY = "READY_FOR_UPLOAD"
STATUSES = (COMPLETE, BLOCKED, INCOMPLETE, FAILED, READY)
STAGES = ("environment", "artifacts", "selection", "validation", "parity", "submissions", "leaderboard")


class M1ConfigError(ValueError):
    pass


def merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_m1_config(config: dict | None = None, path: Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = merge(yaml.safe_load(f), config)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    ids = [a["id"] for a in cfg["artifacts"]]
    if len(ids) != len(set(ids)):
        raise M1ConfigError(f"duplicate artifact ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    for a in cfg["artifacts"]:
        if not a.get("path"):
            raise M1ConfigError(f"artifact {a['id']}: no path")
        req = a.get("required_files")
        if req is not None:
            if a.get("expect_count") is not None:
                raise M1ConfigError(f"artifact {a['id']}: required_files and expect_count are exclusive")
            if not req or len(set(req)) != len(req) or any("/" in str(n) or "*" in str(n) for n in req):
                raise M1ConfigError(f"artifact {a['id']}: required_files must be distinct plain file names")
    ch = cfg["champion"]
    clash = set(ch["candidates"]) & set(ch.get("extra_candidates") or {})
    if clash:
        raise M1ConfigError(f"extra candidates shadow configured ones: {sorted(clash)}")
    for n, e in (ch.get("extra_candidates") or {}).items():
        if e.get("role") != "historical" and e.get("fixed"):
            raise M1ConfigError(f"{n}: a fixed cell not chosen by a nested procedure must have role 'historical'")
    v = cfg["validation"]
    if not 0 < float(v["parity_min_spearman"]) <= 1:
        raise M1ConfigError("validation.parity_min_spearman in (0, 1]")
    if cfg["output_dir"] in (cfg["p6_output_dir"], cfg["m9_output_dir"]):
        raise M1ConfigError("output_dir must differ from every read-only source directory")


def artifact_path(entry: dict, cfg: dict) -> str:
    """Drive-relative glob pattern of one artifact entry ({p6} / {m9} expanded)."""
    return str(entry["path"]).format(p6=cfg["p6_output_dir"], m9=cfg["m9_output_dir"])
