"""Catalogue of the established (pre-Phase-5) models: names, experiment folders, kinds, reported numbers.

The "reported" numbers are copied from the saved scorecards (outputs/phase4/cnn_controls_scorecard.csv,
rs_lgbm_scorecard.csv, P4 area_crossfit.csv, PHASE3_FINDINGS.md) for orientation only; the authoritative
values are always recomputed by ``evaluation.scoreboard`` under one protocol. ``None`` = never measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ZooEntry:
    name: str
    kind: str                        # lgbm | cnn3 | cnn4 | ensemble | map
    experiment: str                  # folder under outputs/<phase>/experiments (or ensemble file)
    phase: str
    reported_geo_dti: float | None
    reported_disc_excl: float | None
    reported_disc_retained_fp: float | None
    note: str = ""


BASELINES = (
    ZooEntry("G1", "lgbm", "lgbm_G1_s2026", "benchmark", 0.1327, 0.1001, 0.0849, "terrain LightGBM (Phase 2)"),
    ZooEntry("RS05_G3_noSS_RS", "lgbm", "p4_RS05_G3_noSS_RS_s2026", "phase4", 0.1429, 0.1186, 0.0945,
             "G3 without seismicity/strain + RS terrain/potential/cross-modal features"),
    ZooEntry("CNN_A_dti", "cnn3", "p3_CNN_A_dti_p256_s2026", "phase3", 0.1302, 0.1788, 0.0918,
             "CNN-A (19 raw bands), distance-aware DTI surrogate"),
    ZooEntry("DL00_CNN_B_dti", "cnn3", "p4_DL00_CNN_B_dti_p256_s2026", "phase4", 0.1496, 0.1675, 0.0807,
             "CNN-B inputs, DTI surrogate"),
    ZooEntry("DL02_CNN_TMGB_dti", "cnn3", "p4_DL02_CNN_TMGB_dti_p256_s2026", "phase4", 0.1215, 0.1709, 0.0998,
             "reduced modalities (terrain, magnetic, gravity, basement)"),
    ZooEntry("P4_B_gated", "cnn4", "p4_P4_B_gated_p256_s2026", "phase4", 0.1584, None, None,
             "gated terrain/subsurface dual encoder; pseudo-discovery NOT yet run (Experiment A1)"),
    ZooEntry("P4_B_concat", "cnn4", "p4_P4_B_concat_p256_s2026", "phase4", None, None, None, "concat dual encoder"),
    ZooEntry("P4_B_ctrl", "cnn4", "p4_P4_B_ctrl_p256_s2026", "phase4", None, None, None,
             "single encoder, parameter-matched control"),
)
BY_NAME = {z.name: z for z in BASELINES}

# Phase-3/4 names resolve through the reuse alias table. Notebook 04 set most of these at runtime only
# (phase4_training.run_rs_cv / run_cnn_control), so a fresh context needs them registered again.
EXTRA_OOF_ALIASES = {z.name: z.experiment for z in BASELINES if z.phase in ("phase3", "phase4")}


def experiment_dir(ctx, name: str) -> Path:
    z = BY_NAME[name]
    if z.phase == "benchmark":
        return ctx.bench.experiments / z.experiment
    return ctx.run_dir(z.experiment)


def load_fold_models(ctx, name: str, device=None) -> list:
    """Fold models of a zoo CNN: [(spec, model)] (cnn4) or the cnn3 loader's output. LightGBM: Booster list."""
    z = BY_NAME[name]
    d = experiment_dir(ctx, name)
    if z.kind == "cnn4":
        from ..phase4_training import load_cnn4_fold_models

        return load_cnn4_fold_models(d, device)
    if z.kind == "cnn3":
        from ..p3.cnn3 import load_cnn3_fold_models
        from ..persist import ExperimentRun

        root = d.parent
        paths = ctx.paths if root == ctx.paths.experiments else (ctx.p3paths if root == ctx.p3paths.experiments else ctx.bench)
        return load_cnn3_fold_models(ExperimentRun(paths, d.name, "cnn"), device)
    if z.kind == "lgbm":
        import lightgbm as lgb

        return [lgb.Booster(model_file=str(p)) for p in sorted((d / "models").glob("fold*.txt"))]
    raise ValueError(f"{name}: no fold models for kind {z.kind}")


def register_aliases(ctx) -> None:
    """Make every zoo name resolvable by ctx.oof / ctx.discovery_pred (Phase-4 alias tables)."""
    oof = ctx.cfg4.setdefault("reuse", {}).setdefault("oof", {})
    for n, exp in EXTRA_OOF_ALIASES.items():
        oof.setdefault(n, exp)
