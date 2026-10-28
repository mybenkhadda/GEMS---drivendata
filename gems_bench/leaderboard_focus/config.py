"""Milestone 1 configuration (``configs/leaderboard_focus/m1.yaml``) and the shared status vocabulary (pure).

Stage statuses: COMPLETE (every check passed), WARNING (computed; non-blocking caveats recorded), BLOCKED (a check
failed or a required artifact is missing: the reason is recorded), INCOMPLETE (an upstream stage has not run or is
not usable), FAILED (an exception; traceback recorded), SKIP_CACHED (reported when a COMPLETE / WARNING result is
reused: the manifest keeps its own status), NOT_REQUIRED (not part of the requested mode / outputs). The submission
gate adds READY_FOR_UPLOAD, which only ``submission.export`` may assign.

Four questions are never conflated: did a stage compute (stage status), did the champion pass promotion
(``gates``: SELECTION_COMPLETE, DISCOVERY_ELIGIBLE, GEOGRAPHIC_GATE_PASS, VALIDATION_PASS, PARITY_PASS ->
READY_FOR_EXPORT), did the prediction pass parity (parity stage), did the file pass export validation
(READY_FOR_UPLOAD).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

M1_VERSION = "m1.1"
M1_CODE_REV = "m1.2-r1"                          # checked by every notebook start-up cell (not fingerprinted)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "leaderboard_focus" / "m1.yaml"
COMPLETE, BLOCKED, INCOMPLETE, FAILED = "COMPLETE", "BLOCKED", "INCOMPLETE", "FAILED"
WARNING, SKIP_CACHED, NOT_REQUIRED = "WARNING", "SKIP_CACHED", "NOT_REQUIRED"
READY = "READY_FOR_UPLOAD"
STATUSES = (COMPLETE, WARNING, BLOCKED, INCOMPLETE, FAILED, SKIP_CACHED, NOT_REQUIRED, READY)
PASSING = (COMPLETE, WARNING)                     # a downstream stage may consume these
STAGES = ("environment", "artifacts", "selection", "validation", "parity", "promotion", "submissions", "leaderboard",
          "report")
MODES = ("FULL", "RECOVER", "VALIDATE", "SUBMIT")
GEO_GATE_MODES = ("mandatory", "advisory")


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
    req = list(ch.get("required_grids") or [])
    if not req or [g for g in req if g not in ch["grids"]]:
        raise M1ConfigError(f"champion.required_grids {req} must be non-empty and within champion.grids {ch['grids']}")
    v = cfg["validation"]
    if not 0 < float(v["parity_min_spearman"]) <= 1:
        raise M1ConfigError("validation.parity_min_spearman in (0, 1]")
    if v.get("recorded_mismatch", "warning") not in ("warning", "blocking"):
        raise M1ConfigError("validation.recorded_mismatch must be 'warning' or 'blocking'")
    pr = cfg.get("promotion") or {}
    if pr.get("geographic_gate") not in GEO_GATE_MODES:
        raise M1ConfigError(f"promotion.geographic_gate must be one of {GEO_GATE_MODES}")
    if pr["geographic_gate"] == "advisory" and not str(pr.get("override_note") or "").strip():
        raise M1ConfigError("promotion.geographic_gate 'advisory' needs promotion.override_note (why a champion "
                            "below the geographic floor may be exported); the floor is never lowered silently")
    if cfg["output_dir"] in (cfg["p6_output_dir"], cfg["m9_output_dir"]):
        raise M1ConfigError("output_dir must differ from every read-only source directory")


NOTEBOOK_DEFAULTS = {"mode": "FULL", "reuse_cached": True, "overwrite_existing": False, "strict_provenance": True,
                     "required_grids": ["boost_a"], "train_cnn": False, "train_transformer": False,
                     "run_extra_seeds": False, "save_figures": True, "save_csv": True, "save_parquet": True,
                     "save_geotiff": True, "sync_to_drive": True}


def notebook_config(CONFIG: dict) -> dict:
    """The master notebook CONFIG with defaults, checked (M1 never trains, CSV tables are the stage contract)."""
    c = {**NOTEBOOK_DEFAULTS, **CONFIG}
    c["mode"] = str(c["mode"]).upper()
    if c["mode"] not in MODES:
        raise M1ConfigError(f"CONFIG['mode'] {c['mode']!r} not in {MODES}")
    on = [k for k in ("train_cnn", "train_transformer", "run_extra_seeds") if c.get(k)]
    if on:
        raise M1ConfigError(f"{on}: Milestone 1 recovers frozen artifacts only; training is outside its scope")
    if not c["save_csv"]:
        raise M1ConfigError("save_csv=False is not supported: CSV tables are the inputs of the later stages")
    if not c["strict_provenance"]:
        print("[m1] WARNING strict_provenance=False: a changed B0 / B1 experiment id is accepted (recorded)")
    return c


def overrides_from_notebook(c: dict, m1_config: dict | None = None) -> dict:
    """M1 config overrides implied by the notebook CONFIG (explicit ``m1_config`` keys win)."""
    g = list(c["required_grids"])
    base = {"champion": {"grids": g, "required_grids": g},
            "outputs": {k: bool(c[k]) for k in ("save_figures", "save_parquet", "save_geotiff", "sync_to_drive")}}
    return merge(base, m1_config)


def artifact_path(entry: dict, cfg: dict) -> str:
    """Drive-relative glob pattern of one artifact entry ({p6} / {m9} expanded)."""
    return str(entry["path"]).format(p6=cfg["p6_output_dir"], m9=cfg["m9_output_dir"])
