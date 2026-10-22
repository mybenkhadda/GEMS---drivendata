"""Established baselines: find and verify, retrain on request, or report NOT_AVAILABLE. Never fabricate.

A reported score is not a model. For every baseline the pipeline checks, in the canonical legacy folder
(the one the model zoo resolves), that the run *finished* (``run.json`` status done), that its OOF map is on
the native grid, which pseudo-discovery maps exist, whether fold-model checkpoints exist (needed for
full-raster inference), and - for runs the pipeline regenerated - that they were trained on the current fold
/ split files (sha256 recorded in ``PROVENANCE.json``). Statuses:

``AVAILABLE``      OOF + every discovery map (or label-free / CV-only by design)
``CV_ONLY``        OOF present, pseudo-discovery maps missing or incomplete
``INCOMPATIBLE``   regenerated on different fold / split files -> never reused
``INTERRUPTED``    a run folder exists but its run.json is not 'done'
``NOT_AVAILABLE``  nothing usable; retrain by listing it in ``stages.train_baselines``

Retraining uses the *legacy* training code with the committed configuration snapshot
(``configs/baselines/legacy_specs/<name>.json`` = the original run's config.json: hyper-parameters, channels,
sampling; no results). The result is written to the canonical folder only when no finished run is there,
and is labelled ``origin: regenerated_by_master_pipeline`` with ``historical_equivalence: false``: library
versions, hardware and nondeterministic GPU kernels mean it is a new run under the same recipe, not the
historical model.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts.validation import read_json, validate_discovery_run, validate_legacy_run

PROVENANCE = "PROVENANCE.json"


@dataclass(frozen=True)
class BaselineRecipe:
    name: str
    kind: str                  # map | lgbm_p2 | lgbm_rs | cnn3 | cnn3_p4 | cnn4
    phase: str                 # canonical CV folder: outputs/<phase>/experiments/<exp_id>
    exp_id: str
    disc_key: str | None       # discovery folders disc<d>_<disc_key> (None = label-free / not applicable)
    gpu: bool
    note: str


RECIPES = {r.name: r for r in (
    BaselineRecipe("B0_lineament", "map", "benchmark", "b0_lineament_fused_s2026", None, False,
                   "label-free Phase-1 fused lineament score; floor for every model and B1 background evidence"),
    BaselineRecipe("G1", "lgbm_p2", "benchmark", "lgbm_G1_s2026", "lgbm_G1_s2026", False, "terrain LightGBM (Phase 2)"),
    BaselineRecipe("G3", "lgbm_p2", "benchmark", "lgbm_G3_s2026", "lgbm_G3_s2026", False,
                   "engineered multi-modal LightGBM (Phase 2)"),
    BaselineRecipe("RS05_G3_noSS_RS", "lgbm_rs", "phase4", "p4_RS05_G3_noSS_RS_s2026", "RS05_G3_noSS_RS", False,
                   "G3 without seismicity/strain + remote-sensing structural features (Phase 4)"),
    BaselineRecipe("CNN_A_dti", "cnn3", "phase3", "p3_CNN_A_dti_p256_s2026", "CNN_A_dti", True,
                   "CNN-A, 19 raw bands, distance-aware DTI surrogate"),
    BaselineRecipe("DL00_CNN_B_dti", "cnn3_p4", "phase4", "p4_DL00_CNN_B_dti_p256_s2026", "DL00_CNN_B_dti", True,
                   "CNN-B inputs, distance-aware DTI surrogate"),
    BaselineRecipe("DL02_CNN_TMGB_dti", "cnn3_p4", "phase4", "p4_DL02_CNN_TMGB_dti_p256_s2026", "DL02_CNN_TMGB_dti",
                   True, "reduced modalities (terrain, magnetic, gravity, basement)"),
    BaselineRecipe("P4_B_gated", "cnn4", "phase4", "p4_P4_B_gated_p256_s2026", "P4_B_gated", True,
                   "gated terrain / subsurface dual encoder (Phase 4)"),
)}

AVAILABLE, CV_ONLY, INCOMPATIBLE, INTERRUPTED, NOT_AVAILABLE = (
    "AVAILABLE", "CV_ONLY", "INCOMPATIBLE", "INTERRUPTED", "NOT_AVAILABLE")


def legacy_spec(name: str) -> dict | None:
    """Committed training-configuration snapshot of a legacy run (None when absent)."""
    from ..paths import config_path

    try:
        p = config_path("baselines", "legacy_specs", f"{name}.json")
    except FileNotFoundError:
        return None
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cv_rel(r: BaselineRecipe) -> str:
    return f"outputs/{r.phase}/experiments/{r.exp_id}"


def disc_rels(r: BaselineRecipe, n_splits: int) -> list[str]:
    if not r.disc_key:
        return []
    phase = "benchmark" if r.kind == "lgbm_p2" else "phase4"
    return [f"outputs/{phase}/experiments/disc{d}_{r.disc_key}" for d in range(n_splits)]


def all_rels(n_splits: int, names=None) -> list[str]:
    out = []
    for n in names or RECIPES:
        r = RECIPES[n]
        out += [cv_rel(r), *disc_rels(r, n_splits)]
    return out


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def baseline_status(root: Path, name: str, *, grid: dict | None, n_splits: int, folds_sha: str | None = None,
                    splits_sha: str | None = None) -> dict:
    r = RECIPES[name]
    root = Path(root)
    cv_dir = root / cv_rel(r)
    row = {"model": name, "kind": r.kind, "gpu": r.gpu, "cv_dir": cv_rel(r), "origin": "", "status": NOT_AVAILABLE,
           "discovery": "n/a" if not r.disc_key else "0/%d" % n_splits, "checkpoints": False, "detail": ""}
    if not cv_dir.exists():
        row["detail"] = "no run folder"
        return row
    prov = read_json(cv_dir / PROVENANCE, None)
    row["origin"] = "regenerated" if prov else "legacy"
    rep = validate_legacy_run(cv_dir, grid=grid)
    if not rep.ok:
        bad = rep.failed()[0]
        row["status"] = INTERRUPTED if "run.json" in bad.path and bad.status != "missing" else NOT_AVAILABLE
        row["detail"] = rep.summary()
        return row
    m = cv_dir / "models"
    row["checkpoints"] = m.is_dir() and any(f.is_file() for f in m.rglob("*"))
    if prov:
        if folds_sha and prov.get("folds_sha256") not in (None, folds_sha):
            row.update(status=INCOMPATIBLE, detail="regenerated on different fold files (fold_map sha256 differs)")
            return row
    else:
        cfg = read_json(cv_dir / "config.json", {}) or {}
        f = str(cfg.get("folds") or "")
        row["detail"] = f"legacy run; fold design by name: {Path(f).name or 'not recorded'}"
    if not r.disc_key:
        row["status"] = AVAILABLE
        return row
    ok = 0
    for rel in disc_rels(r, n_splits):
        d = root / rel
        if validate_discovery_run(d, grid=grid).ok:
            dp = read_json(d / PROVENANCE, None)
            if dp and splits_sha and dp.get("splits_sha256") not in (None, splits_sha):
                continue
            ok += 1
    row["discovery"] = f"{ok}/{n_splits}"
    row["status"] = AVAILABLE if ok == n_splits else CV_ONLY
    return row


def status_table(root: Path, *, grid: dict | None, n_splits: int, folds_sha=None, splits_sha=None) -> pd.DataFrame:
    return pd.DataFrame([baseline_status(root, n, grid=grid, n_splits=n_splits, folds_sha=folds_sha,
                                         splits_sha=splits_sha) for n in RECIPES])


# ---------------------------------------------------------------------------
# retraining (legacy code paths, snapshot configs)
# ---------------------------------------------------------------------------
def _ensure_features(ctx, names: list[str]) -> None:
    unknown = [n for n in names if n not in ctx.store.specs]
    if unknown:
        from .context import PipelineStop

        raise PipelineStop(f"features not defined in the store: {unknown[:8]}",
                           "check the feature families / configs/phase4/rs_features.yaml of this checkout")
    missing = [n for n in names if not ctx.store.has(n)]
    if missing:
        print(f"[baselines] building {len(missing)} missing feature caches")
        ctx.store.build(missing)


def _gpu_spec(spec, cls):
    from ..training.gpu import autoconfig

    gpu = autoconfig(patch_px=spec.patch_px, effective_batch=spec.batch_size * max(getattr(spec, "grad_accum", 1), 1),
                     n_channels=len(spec.channels))
    allowed = {f.name for f in fields(cls)} & {"amp_dtype", "infer_batch", "batch_size", "grad_accum", "data_on_gpu",
                                               "cudnn_benchmark", "channels_last"}
    return replace(spec, **{k: v for k, v in gpu.items() if k in allowed}), gpu


def _write_provenance(root: Path, rels: list[str], name: str, extra: dict) -> None:
    from .._version import provenance
    from ..artifacts.atomic import atomic_write_json

    for rel in rels:
        d = Path(root) / rel
        if d.exists() and not (d / PROVENANCE).exists():
            atomic_write_json(d / PROVENANCE, {"model": name, "origin": "regenerated_by_master_pipeline",
                                               "historical_equivalence": False, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               **provenance(), **extra})


def train_baseline(pc, name: str, *, folds_sha: str | None, splits_sha: str | None) -> dict:
    """(Re)train one baseline into its canonical folder (only if no finished run is there); returns a summary."""
    from .context import PipelineStop

    r = RECIPES[name]
    root = pc.local
    st = baseline_status(root, name, grid=pc.grid(), n_splits=pc.bench_cfg.n_discovery_folds, folds_sha=folds_sha,
                         splits_sha=splits_sha)
    if st["status"] == AVAILABLE:
        return {"model": name, "action": "already available"}
    if st["status"] == INCOMPATIBLE:
        raise PipelineStop(f"{name}: existing regenerated run used other fold files",
                           f"keep it for reference; move {cv_rel(r)} aside on Drive (never deleted automatically) "
                           "and re-run to train under the current folds")
    snap = legacy_spec(name)
    if snap is None:
        raise PipelineStop(f"{name}: no configuration snapshot in configs/baselines/legacy_specs/",
                           "restore the legacy run folder from the local project instead (docs/ARTIFACT_RECOVERY.md)")
    if r.gpu:
        pc.require_gpu(f"baseline {name}")
    ctx = pc.ctx5()
    cfg = pc.bench_cfg
    data, n_splits = ctx.data, ctx.splits.n_splits
    t0 = time.perf_counter()
    cv_done = st["status"] == CV_ONLY
    gpu = {}
    if r.kind == "map":
        from ..gbm import run_map_baseline

        _ensure_features(ctx, ["eda:fused_score"])
        run_map_baseline(r.exp_id, np.asarray(ctx.store.get("eda:fused_score")), data, ctx.folds, cfg,
                         snap.get("description", "Phase-1 fused lineament score (200 m -> 100 m)"))
    elif r.kind == "lgbm_p2":
        from ..gbm import fit_full_gbm, median_best_iteration, run_gbm_cv

        names = list(snap["features"])
        _ensure_features(ctx, names)
        run = run_gbm_cv(r.exp_id, snap.get("group", name), names, data, ctx.store, ctx.folds, cfg,
                         sample_scale=float(snap.get("sample_scale", 1.0)))
        rounds = median_best_iteration(run)
        for d in range(n_splits):
            fit_full_gbm(f"disc{d}_{r.exp_id}", names, data, ctx.store, cfg, data.valid, rounds,
                         ignore_mask=ctx.splits.ignore(d), predict_area=ctx.splits.eval_zone(d) & data.valid,
                         extra_config={"discovery_split": d, "cv_run": r.exp_id})
    elif r.kind == "lgbm_rs":
        from ..p3 import lgbm3
        from ..phase4_training import run_summary

        names, v = list(snap["features"]), dict(snap["variant"])
        _ensure_features(ctx, names)
        lgbm3.run_variant_cv(ctx, r.exp_id, names, v, tag=name)
        ctx.cfg4.setdefault("reuse", {}).setdefault("oof", {})[name] = r.exp_id
        its = [x for x in run_summary(ctx, r.exp_id).get("best_iters", []) if x == x]
        rounds = max(1, int(np.median(its))) if its else int(v.get("rounds", 400))
        for d in range(n_splits):
            lgbm3.fit_full_variant(ctx, f"disc{d}_{name}", names, v, rounds, ignore_mask=ctx.splits.ignore(d),
                                   predict_area=ctx.splits.eval_zone(d) & data.valid)
    elif r.kind in ("cnn3", "cnn3_p4"):
        from ..config import BenchManifest
        from ..p3 import cnn3
        from ..phase4_training import run_cnn_control, run_cnn_control_discovery, spec_from_config

        spec, gpu = _gpu_spec(spec_from_config(snap, cnn3.CNN3Spec), cnn3.CNN3Spec)
        _ensure_features(ctx, list(spec.channels))
        if not cv_done:
            if r.kind == "cnn3":                   # Phase-3 run: canonical folder outputs/phase3/experiments
                p3 = replace(ctx, paths=ctx.p3paths, manifest=BenchManifest(ctx.p3paths))
                cnn3.run_cnn3_cv(p3, r.exp_id, spec, input_config=snap.get("input_config", name))
            else:
                run_cnn_control(ctx, name, spec)
        ctx.cfg4.setdefault("reuse", {}).setdefault("oof", {})[name] = r.exp_id
        run_cnn_control_discovery(ctx, name, spec, r.exp_id)
    elif r.kind == "cnn4":
        from ..phase4_training import P4Spec, run_cnn4_cv, run_cnn4_discovery_all, spec_from_config

        spec, gpu = _gpu_spec(spec_from_config(snap, P4Spec), P4Spec)
        _ensure_features(ctx, list(spec.channels))
        if not cv_done:
            run_cnn4_cv(ctx, r.exp_id, spec)
        ctx.cfg4.setdefault("reuse", {}).setdefault("oof", {})[name] = r.exp_id
        run_cnn4_discovery_all(ctx, name, spec, r.exp_id)
    else:
        raise ValueError(r.kind)
    rels = [cv_rel(r), *disc_rels(r, n_splits)]
    _write_provenance(root, rels, name, {"spec_snapshot": f"configs/baselines/legacy_specs/{name}.json",
                                         "folds_sha256": folds_sha, "splits_sha256": splits_sha,
                                         "fold_dir": pc.fold_rel, "discovery_dir": pc.discovery_rel,
                                         "gpu": pc.gpu().get("name"), "throughput_overrides": gpu})
    ctx._maps.pop(name, None)
    return {"model": name, "action": "trained", "minutes": round((time.perf_counter() - t0) / 60, 1), "rels": rels}
