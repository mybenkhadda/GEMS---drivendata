"""G14.1: critical failure analysis, discovery-calibrated recovery and corrected training for GEMS fault discovery.

Status: UNTESTED. Every module was written without executing any code, test or notebook.

Package map (adapted to the repository; G13 / G14 / P6 functionality is reused, not duplicated):

=====================  ===============================================================================
config                 typed config, named execution modes, profiles (FAST / STANDARD / FULL), validation
artifacts              artifact registry, discovery across local + Drive roots, provenance, grid checks,
                       Drive restoration, stage ledger (resumability), preflight report
geometry               raster geometry and submission parity (CRS / transform / shape / flips / shifts),
                       resolution-aware metric kernel, hard-coded assumption audit
leakage                known-context shortcut audit, target/context overlap, checkpoint counterfactuals,
                       context-dependence curves
probability            raw probability-map diagnostics (G1)
representations        representation catalogue with exact area accounting, known-masking order (G2, G5)
calibration            soft calibration (temperature, power, quantile, logistic, isotonic, regime) fitted
                       out-of-fold; cross-fitted representation x area x calibration sweep (G3, G4)
metric_audit           DTI decomposition, localisation, failure classification (F01-F10), inner-validation
                       methodology (full-region aggregation, ranking consistency)
champion               champion discovery and loading, compatibility, complementarity, residual blending
discovery              validation protocols, conditional discovery units (context / target systems),
                       separated train / validation targets, negative policies
losses                 selectable loss registry (13 objectives) + per-loss gradient diagnostics
architectures          large-receptive-field CNN, local + regional CNN, directional attention (N1, N2, N5)
training               corrected trainer: channel subsets, conditional sampling, augmentation policies,
                       full-region inner DTI checkpoint selection
atlas                  geological failure atlas (window selection, panels, index)
figures                the 24 research figures
decision               hypothesis evidence table and outcome classification (A-G)
submission             one robust submission function (discovery-calibrated first, provisional otherwise)
experiments            R00-R17 stage ladder, dependencies, M0-M6 / N1-N6 / ablation registry
pipeline               notebook orchestrator with stage-level resumability
=====================  ===============================================================================
"""
G14_1_VERSION = "g14.1.0-untested"
OUT_SUBDIR = "outputs/G14_1_CRITICAL_RECOVERY"
OUTPUT_DIRS = ("inventory", "geometry_audit", "leakage_audit", "probability_diagnostics", "metric_decomposition",
               "representation_sweep", "failure_atlas", "champion_comparison", "complementarity", "discovery_splits",
               "corrected_models", "architecture_comparison", "calibration", "submissions", "reports", "manifests")
