# Experiment ladder and training-spec registry

`gems_bench/g14_1/experiments.py` (`STAGES`, `plan(cfg)`, `REGISTRY`, `matrix()`). No stage has been run.

| stage | experiment | retraining | profiles | switch |
|---|---|---|---|---|
| R00 | artifact inventory + preflight | No | all | artifact_inventory |
| R01 | raster / export parity | No | all | geometry_audit |
| R02 | existing map diagnostics + static leakage | No | all | prediction_diagnostics |
| R03 | representation / area / masking / calibration sweep | No | all | representation_sweep (+ calibration) |
| R04 | DTI decomposition + failure classes | No | all | metric_decomposition |
| R05 | champion comparison | No | all | champion_comparison |
| R05b | failure atlas | No | STANDARD, FULL | failure_atlas |
| R06 | checkpoint counterfactuals + inner audit | Inference only | STANDARD, FULL | checkpoint_counterfactuals |
| R07 | corrected discovery splits | No | STANDARD, FULL | discovery_splits |
| R08 | D1 known-only, M0 control | Yes | FULL | train_models |
| R09 | M1 geophysics-only | Yes | FULL | train_models |
| R10 | M2 conditional discovery, M3 combined targets | Yes | FULL | train_models |
| R11 | E2 objectives, M5 negatives, K features, L augmentation | Yes | FULL | train_models |
| R12 | champion residual recovery (blend sweep) | Optional | all | complementarity |
| R13 | N1 large-RF CNN, N2 local + regional CNN | Yes | FULL | train_models |
| R14 | N3 regional Swin, N5 directional (+ isotropic control) | Yes | FULL | transformers |
| R15 | N4 full GeoFaultFormer | Yes | FULL | transformers |
| R16 | discovery-calibrated ensemble | Depends | FULL | complementarity |
| R17 | submission | No (if final maps exist) | all | submission (mode SUBMIT) |

R00–R05 never need R08–R15. FAST runs R00–R05 and R12 from existing artifacts. FULL runs the registry, not a
Cartesian product.

## Specs (each changes one factor against its reference)

| spec | reference | factor |
|---|---|---|
| M0_original_g14 | none | control: shortcut, combined targets, G14 loss and augmentation |
| D1_known_only | M0 | inputs: known context only |
| M1_geophysics_only | M0 | no known context, separated targets |
| M2_conditional_discovery | M1 | conditional context / target systems |
| M3_combined_targets | M1 | G14 combined train ∪ inner target raster |
| E2_{bce, bce_dice, bce_tversky, focal, bce_focal_tversky, distance_aware, dti_surrogate, cldice, background_sparsity, g14_original, corrected_composite, nnpu} | M1 | objective |
| M5_{all_unlabeled, reduced_weight, hard_negative} | M1 | negative supervision |
| M6_champion_residual | M1 | loss emphasis on champion misses / FPs (needs a nested champion map) |
| K_{magnetic, gravity, terrain, conductivity, geodesy, structural, raw_only, engineered_only} | M1 | inputs |
| L_{none, noise_only, sensor_dropout_only, modality_dropout_only, rotation_safe, original} | M1 | augmentation |
| N1 n1_large_rf, N2 n2_local_regional, N3 g14_regional_swin, N4 g14_full, N5 n5_directional, N5_control n5_isotropic | M1 | architecture |

M4 (the discovery-aligned loss) is the winner of E2 on discovery cross-fitted DTI with a paired bootstrap. It is not
fixed in advance.

The mandatory comparison holds for every N spec: it shares features, supervision, folds, evaluation masks and
optimisation with M1. Only `architectures.build` differs.
