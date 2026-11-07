"""Section 02 - real-file preflight: every artifact, exact checkpoint provenance, no synthetic fallback."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from . import inventory as inv
from .cache import atomic_write_csv, atomic_write_json, read_json
from .config import stable_hash
from .contracts import PreflightError, StageResult


def _check(rows, name, ok, required, detail=""):
    rows.append({"role": "check", "key": name, "path": "", "required": required, "exists": True, "size": None,
                 "mtime_ns": None, "sha256": "",
                 "status": "OK" if ok else ("FAILED" if required else "WARN"), "detail": detail})
    return bool(ok)


def v1_run_key(src_cfg: dict, bank_key: str) -> str:
    """Exactly GeoRegimeExperiment.prepare(): sha256({config minus drive/scratch roots, bank})[:12]."""
    semantic = {k: v for k, v in src_cfg.items() if k not in ("drive_root", "scratch_root")}
    return hashlib.sha256(json.dumps({"config": semantic, "bank": bank_key}, sort_keys=True).encode()).hexdigest()[:12]


def v1_config_hash(src_cfg: dict) -> str:
    """Exactly the checkpoint contract's config_hash."""
    return hashlib.sha256(json.dumps(src_cfg, sort_keys=True).encode()).hexdigest()[:12]


def run_preflight(run) -> StageResult:
    import torch

    from gems_bench.georegime.data import Bank
    from gems_bench.georegime.model import GeoRegimeNet

    P, cfg = run.paths, run.cfg
    checks: list[dict] = []
    failures: list[str] = []
    limitations: list[str] = []
    schema = read_json(P.atlas / "06_ml_ready/schemas/feature_stack_schema.json")
    v1 = read_json(P.src("config.json"))
    bank_key_cfg = (v1 or {}).get("bank_fingerprint")
    table = inv.inventory(inv.expected_sources(P, cfg, schema, bank_key_cfg), run.hashes)
    for r in table.itertuples():
        if r.required and r.status != "OK":
            failures.append(f"{r.role}:{r.key} {r.status} ({r.path})")

    # --- V1 configuration
    src_cfg = (v1 or {}).get("config")
    if not _check(checks, "v1_config_readable", isinstance(src_cfg, dict), True, str(P.src("config.json"))):
        failures.append("V1 config.json unreadable")
        src_cfg = {}

    # --- the V1 feature bank (validates Atlas grid, CRS, band names, masks and the official template)
    bank = None
    try:
        bank = Bank(P.atlas, P.d("cache", "bank"), P.scratch / "bank")
        _check(checks, "atlas_bank_grid", True, True,
               f"shape {bank.shape}, {len(bank.names)} channels, eligible {int(bank.valid.sum())}, "
               f"mapped {int(bank.labels.sum())}")
    except Exception as e:                                                      # noqa: BLE001 - reported verbatim
        failures.append(f"V1 Bank construction failed: {type(e).__name__}: {e}")
        _check(checks, "atlas_bank_grid", False, True, f"{type(e).__name__}: {e}")
    if bank is not None and bank_key_cfg:
        if not _check(checks, "bank_fingerprint_matches_training", bank.key == bank_key_cfg, True,
                      f"current {bank.key} vs V1 config {bank_key_cfg}"):
            failures.append("Atlas inputs changed since V1 training (bank fingerprint differs)")
    if bank is not None and src_cfg:
        key = v1_run_key(src_cfg, bank.key)
        _check(checks, "v1_run_key_reproduced", key == P.source_run.name, False,
               f"recomputed {key} vs directory {P.source_run.name}")

    # --- checkpoints: contract, completion, schema, normalizer, strict state-dict load
    ckpt: dict = {}
    u = cfg["units"]
    units = u["geographic"] + u["discovery"] + [u["full"]]
    want_cfg_hash = v1_config_hash(src_cfg) if src_cfg else None
    for unit in units:
        p = P.checkpoint(unit)
        if not p.is_file() or bank is None or not src_cfg:
            continue
        try:
            state = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:                                                  # noqa: BLE001
            failures.append(f"{unit}: checkpoint unreadable ({type(e).__name__}: {e})")
            _check(checks, f"{unit}_checkpoint_load", False, True, str(e))
            continue
        c = state.get("contract", {})
        ok_epoch = int(state.get("epoch", -1)) + 1 == int(src_cfg.get("epochs", -1))
        ok_unit = c.get("unit") == unit
        ok_feat = c.get("features") == bank.key and state.get("feature_schema") == bank.names
        ok_type = state.get("model_type") == "GeoRegimeNet_v1"
        norm = state.get("normalizer")
        ok_norm_len = isinstance(norm, dict) and len(norm.get("median", [])) == len(bank.names) + 3
        nj = read_json(P.normalizer(unit))
        norm_json = "absent" if nj is None else ("equal" if nj == norm else "DIFFERS")
        try:
            g = bank.group_indices
            m = GeoRegimeNet(len(g["mag"]), len(g["grav"]), len(g["terrain"]), int(src_cfg["width"]))
            m.load_state_dict(state["model"], strict=True)
            ok_load, load_msg = True, f"{sum(t.numel() for t in m.parameters())} parameters"
            del m
        except Exception as e:                                                  # noqa: BLE001
            ok_load, load_msg = False, f"{type(e).__name__}: {e}"
        for name, ok, req, det in (
                ("epochs_complete", ok_epoch, True, f"epoch {state.get('epoch')} + 1 vs {src_cfg.get('epochs')}"),
                ("contract_unit", ok_unit, True, str(c.get("unit"))),
                ("feature_schema", ok_feat, True, f"contract features {c.get('features')}"),
                ("model_type", ok_type, True, str(state.get("model_type"))),
                ("normalizer_shape", ok_norm_len, True, f"{len((norm or {}).get('median', []))} channels"),
                ("normalizer_json", norm_json != "DIFFERS", True, norm_json),
                ("config_hash", c.get("config_hash") == want_cfg_hash, False, f"{c.get('config_hash')} vs {want_cfg_hash}"),
                ("strict_state_dict", ok_load, True, load_msg)):
            if not _check(checks, f"{unit}_{name}", ok, req, det) and req:
                failures.append(f"{unit}: {name} failed ({det})")
        if norm_json == "absent":
            limitations.append(f"{unit}: no {unit}_normalizer.json - the checkpoint's embedded normalizer is used")
        ckpt[unit] = {"path": str(p), "sha256": run.hashes.sha(p), "size": p.stat().st_size,
                      "mtime_ns": p.stat().st_mtime_ns, "train_mask_sha": c.get("train_mask_sha"),
                      "normalizer": norm, "normalizer_sha": stable_hash(norm), "epoch": int(state.get("epoch", -1))}
        del state

    # --- V1 prediction metadata (the V1 cache key ignores the ROI: record what it was produced from)
    v1pred = {}
    for unit in units:
        meta = read_json(P.v1_prediction(unit).with_suffix(".json"))
        tif = P.v1_prediction(unit)
        if not tif.is_file():
            continue
        rec = {"tif": str(tif), "meta": meta}
        try:
            import rasterio

            with rasterio.open(tif) as ds:
                rec["grid_ok"] = bank is not None and ds.shape == bank.shape and ds.transform == bank.transform \
                    and ds.crs.to_epsg() == 32611
        except Exception as e:                                                  # noqa: BLE001
            rec["grid_ok"] = False
            rec["error"] = str(e)
        if unit in ckpt and meta:
            rec["same_checkpoint_state"] = (meta.get("checkpoint_size") == ckpt[unit]["size"]
                                            and meta.get("checkpoint_mtime_ns") == ckpt[unit]["mtime_ns"])
            _check(checks, f"{unit}_v1_prediction_from_current_checkpoint", rec["same_checkpoint_state"], False,
                   "size/mtime recorded by V1 vs current checkpoint")
        req = unit == u["full"]
        if not _check(checks, f"{unit}_v1_prediction_grid", rec["grid_ok"], req, rec.get("error", "")) and req:
            failures.append(f"{unit}: V1 prediction raster grid mismatch")
        v1pred[unit] = rec

    cuda = bool(torch.cuda.is_available())
    _check(checks, "cuda_available", cuda or cfg["inference"]["allow_cpu"], False,
           "GPU" if cuda else "CPU fallback (full-area inference is slow but allowed)")
    if not cuda:
        limitations.append("no CUDA device: full-area inference runs on CPU")
    for r in table.itertuples():
        if not r.required and r.status in ("OPTIONAL_MISSING", "INACCESSIBLE"):
            limitations.append(f"optional {r.role}:{r.key} {r.status}")

    full = pd.concat([table, pd.DataFrame(checks)], ignore_index=True)
    status = "PASSED" if not failures else "FAILED"
    fps = {r.path: {"size": r.size, "mtime_ns": r.mtime_ns, "sha256": r.sha256} for r in table.itertuples() if r.exists}
    fingerprints = {"files": fps, "bank_key": getattr(bank, "key", None), "bank_key_v1_config": bank_key_cfg,
                    "checkpoints": {k: {kk: v[kk] for kk in ("sha256", "train_mask_sha", "normalizer_sha", "epoch")}
                                    for k, v in ckpt.items()},
                    "code_commit": cfg.get("code_commit"), "code_hash": run.code_hash}
    out = [atomic_write_csv(P.d("00_inventory") / "source_inventory.csv", full),
           atomic_write_json(P.d("00_inventory") / "source_fingerprints.json", fingerprints),
           atomic_write_json(P.d("00_inventory") / "preflight.json",
                             {"status": status, "failures": failures, "limitations": limitations, "cuda": cuda,
                              "bank": bank.preflight() if bank is not None else None, "v1_predictions": v1pred,
                              "config_hash": run.cfg_hash})]
    run.bank, run.src_cfg, run.ckpt, run.v1pred = bank, src_cfg, ckpt, v1pred
    run.source_fp = stable_hash(fingerprints)
    res = StageResult("preflight", status, {"table": full}, {"n_failures": len(failures), "failures": failures,
                                                             "cuda": cuda, "source_fingerprint": run.source_fp},
                      out, limitations)
    if failures:
        raise PreflightError("PREFLIGHT FAILED - mandatory artifacts missing or inconsistent:\n  - "
                             + "\n  - ".join(failures) + f"\nSee {out[0]}")
    return res
