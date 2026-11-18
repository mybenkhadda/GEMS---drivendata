"""Staged research ladder R00-R17 (Phase O) and the controlled training-spec registry (Phases D, E, K, L, M, N).

Stages declare their dependencies, whether they retrain, the config switch that enables them and the profiles that
include them. ``plan(cfg)`` returns the ordered, dependency-closed list of stages a run will execute and why each
other stage is skipped. FAST covers R00-R05 from existing artifacts only; R08-R15 never run unless the mode is TRAIN.

Training specs change ONE factor against a named reference (``ref``), so every comparison is attributable; the
FULL profile runs the registry, not a Cartesian product.

UNTESTED: written without execution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .training import TrainSpec


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    switch: str                     # G141Config stage switch
    deps: tuple = ()
    retrain: str = "No"             # No | Inference only | Yes | Optional | Depends
    profiles: tuple = ("FAST", "STANDARD", "FULL")


STAGES = [
    Stage("R00", "Artifact inventory and preflight", "artifact_inventory"),
    Stage("R01", "Raster and export parity", "geometry_audit", ("R00",)),
    Stage("R02", "Existing G14 map diagnostics + static leakage audit", "prediction_diagnostics", ("R00",)),
    Stage("R03", "Representation, area, masking-order and calibration sweep", "representation_sweep", ("R00",)),
    Stage("R04", "DTI component decomposition and failure classes", "metric_decomposition", ("R03",)),
    Stage("R05", "Champion comparison and complementarity", "champion_comparison", ("R00",)),
    Stage("R05b", "Geological failure atlas", "failure_atlas", ("R04",), profiles=("STANDARD", "FULL")),
    Stage("R06", "Existing checkpoint counterfactuals + inner-validation audit", "checkpoint_counterfactuals",
          ("R00",), "Inference only", ("STANDARD", "FULL")),
    Stage("R07", "Corrected discovery splits", "discovery_splits", ("R00",), profiles=("STANDARD", "FULL")),
    Stage("R08", "Known-context-only model (D1)", "train_models", ("R07",), "Yes", ("FULL",)),
    Stage("R09", "Geophysics-only CNN (D2 / M1)", "train_models", ("R07",), "Yes", ("FULL",)),
    Stage("R10", "Conditional discovery CNN (M2) + corrected targets (M3)", "train_models", ("R07",), "Yes", ("FULL",)),
    Stage("R11", "Loss, negative-supervision, augmentation and feature ablations", "train_models", ("R09", "R10"),
          "Yes", ("FULL",)),
    Stage("R12", "Champion residual recovery (blend sweep; M6 if a nested champion exists)", "complementarity",
          ("R05",), "Optional", ("FAST", "STANDARD", "FULL")),
    Stage("R13", "Multiscale CNN (N1 / N2)", "train_models", ("R11",), "Yes", ("FULL",)),
    Stage("R14", "Regional transformer (N3) and directional attention (N5)", "transformers", ("R13",), "Yes", ("FULL",)),
    Stage("R15", "Full GeoFaultFormer (N4)", "transformers", ("R14",), "Yes", ("FULL",)),
    Stage("R16", "Final discovery-calibrated ensemble", "complementarity", ("R12",), "Depends", ("FULL",)),
    Stage("R17", "Submission preparation", "submission", ("R03",), "No additional training if final maps exist",
          ("FAST", "STANDARD", "FULL")),
]
STAGE = {s.id: s for s in STAGES}


def plan(cfg) -> pd.DataFrame:
    """Ordered stage plan for a config: RUN / SKIP with the reason (mode capability, profile, switch, dependency)."""
    rows, will = [], set()
    for s in STAGES:
        enabled = cfg.stage_enabled(s.switch)
        in_profile = cfg.profile in s.profiles or s.switch in cfg.stages
        missing = [d for d in s.deps if d not in will and STAGE[d].switch != s.switch]
        if not in_profile:
            why = f"not in profile {cfg.profile}"
        elif not enabled:
            why = f"switch '{s.switch}' off or mode {cfg.mode} lacks its capability"
        elif missing:
            why = f"dependencies not scheduled: {missing} (runs on existing outputs if present)"
        else:
            why = ""
        run = in_profile and enabled
        if run:
            will.add(s.id)
        rows.append({"stage": s.id, "title": s.title, "retraining": s.retrain, "action": "RUN" if run else "SKIP",
                     "reason": why, "deps": ",".join(s.deps)})
    return pd.DataFrame(rows)


# =============================================================================== training specs
M1 = TrainSpec("M1_geophysics_only", supervision="geophysics_only", feature_set="geophysics",
               objective="weighted_bce", augmentation="conservative", notes="D2: every target-derived input removed")
REGISTRY: dict[str, tuple[TrainSpec, str, str]] = {}


def _add(spec: TrainSpec, ref: str, factor: str):
    REGISTRY[spec.name] = (spec, ref, factor)


_add(TrainSpec("M0_original_g14", supervision="original", feature_set="all", objective="g14_original",
               augmentation="original", separate_targets=False, negative_policy="distance_masked",
               notes="control: G14 shortcut, combined targets, G14 loss and augmentation"), "", "control")
_add(TrainSpec("D1_known_only", supervision="original", feature_set="known_only", objective="g14_original",
               augmentation="original", separate_targets=False, notes="shortcut ceiling: context channels only"),
     "M0_original_g14", "inputs: known context only")
_add(M1, "M0_original_g14", "supervision: no known context, separated targets")
_add(TrainSpec("M2_conditional_discovery", supervision="conditional", feature_set="all", objective="weighted_bce",
               augmentation="conservative", known_dropout=0.2,
               notes="Y_target = Y_visible minus Y_context, re-drawn every epoch"), "M1_geophysics_only",
     "supervision: conditional context / target systems")
_add(replace(M1, name="M3_combined_targets", separate_targets=False), "M1_geophysics_only",
     "targets: G14 combined train|inner raster (S03 reproduced)")
# E2 objectives (M4 picks among them on discovery evidence)
for obj in ("bce", "bce_dice", "bce_tversky", "focal", "bce_focal_tversky", "distance_aware", "dti_surrogate", "cldice",
            "background_sparsity", "g14_original", "corrected_composite"):
    _add(replace(M1, name=f"E2_{obj}", objective=obj), "M1_geophysics_only", f"objective: {obj}")
_add(replace(M1, name="E2_nnpu", objective="nnpu", negative_policy="pu"), "M1_geophysics_only", "objective: nnPU risk")
# M5 negative supervision
for pol, extra in (("all_unlabeled", {}), ("reduced_weight", {"neg_weight": 0.3}),
                   ("hard_negative", {"hard_negative_frac": 0.2})):
    _add(replace(M1, name=f"M5_{pol}", negative_policy=pol, **extra), "M1_geophysics_only", f"negatives: {pol}")
# M6 residual (requires a nested champion map)
_add(replace(M1, name="M6_champion_residual", residual_champion=True), "M1_geophysics_only",
     "loss emphasis on champion misses / champion false positives")
# K feature ablations
for fs in ("magnetic", "gravity", "terrain", "conductivity", "geodesy", "structural", "raw_only", "engineered_only"):
    _add(replace(M1, name=f"K_{fs}", feature_set=fs), "M1_geophysics_only", f"inputs: {fs}")
# L augmentation ablations
for aug in ("none", "noise_only", "sensor_dropout_only", "modality_dropout_only", "rotation_safe", "original"):
    _add(replace(M1, name=f"L_{aug}", augmentation=aug), "M1_geophysics_only", f"augmentation: {aug}")
# N architectures (all on the corrected local CNN's features, supervision, folds and optimisation)
for arch, tag in (("n1_large_rf", "N1"), ("n2_local_regional", "N2"), ("g14_regional_swin", "N3"),
                  ("g14_full", "N4"), ("n5_directional", "N5"), ("n5_isotropic", "N5_control")):
    _add(replace(M1, name=f"{tag}_{arch}", architecture=arch), "M1_geophysics_only", f"architecture: {arch}")

STAGE_SPECS = {
    "R08": ["D1_known_only", "M0_original_g14"],
    "R09": ["M1_geophysics_only"],
    "R10": ["M2_conditional_discovery", "M3_combined_targets"],
    "R11": [k for k in REGISTRY if k.startswith(("E2_", "M5_", "K_", "L_"))],
    "R12": ["M6_champion_residual"],
    "R13": ["N1_n1_large_rf", "N2_n2_local_regional"],
    "R14": ["N3_g14_regional_swin", "N5_n5_directional", "N5_control_n5_isotropic"],
    "R15": ["N4_g14_full"],
}
PROFILE_SPEC_LIMIT = {"FAST": 0, "STANDARD": 3, "FULL": 10_000}


def specs_for(stage: str, profile: str, only: tuple = ()) -> list[TrainSpec]:
    names = [n for n in STAGE_SPECS.get(stage, []) if not only or n in only]
    return [REGISTRY[n][0] for n in names[:PROFILE_SPEC_LIMIT[profile]]]


def matrix() -> pd.DataFrame:
    return pd.DataFrame([{"spec": n, "reference": ref, "changed_factor": f, "supervision": s.supervision,
                          "feature_set": s.feature_set, "objective": s.objective, "negatives": s.negative_policy,
                          "augmentation": s.augmentation, "architecture": s.architecture,
                          "separate_targets": s.separate_targets, "stage": next((k for k, v in STAGE_SPECS.items()
                                                                                if n in v), "")}
                         for n, (s, ref, f) in REGISTRY.items()])
