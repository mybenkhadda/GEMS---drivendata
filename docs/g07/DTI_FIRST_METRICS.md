# G07 DTI-first diagnostics — run guide

This branch changes **measurement**, not the model-training loss. It does not claim a leaderboard improvement or imply that a local DEM score predicts leaderboard DTI.

## Notebook

\`notebooks/colab/GEMS_G07_MULTISCALE_FAULT_DISCOVERY.ipynb\`

- \`CONFIG.ref = g07-dti-first-metrics\` while reviewing this branch.
- \`DTI_FIRST_MODE=True\` is the default: load previously synced G07 artifacts, evaluate scores only, and do **not** retrain, download DEM, generate datasets, export or submit.
- This deliberately skips synthetic tests *only for the cached read-only diagnostic path*. It does **not** mark them passed; the full mode retains the original strict test gate.
- If there is no saved \`dem_features/dem_features_100m.tif\`, the DTI feature section explains why it cannot run. Do not fabricate a fallback or relax DEM masks.
- Set \`DTI_FIRST_MODE=False\` to restore full G07 stage flags. Resolve any upstream test/DEM/training issues independently before using that mode for scientific validation.

## Section 09A — DEM feature DTI

\`gems_bench/g07/dti_screening.py::run_feature_dti\` reads the canonical V1/V3 held-out evaluation units and the frozen W2 predictions.

Every feature has a **predeclared geological transformation** (high or absolute); orientation cos/sin are refused until an independently specified reference direction exists. No direction/weight/budget is optimized against withheld labels.

Two protocols are **not interchangeable**:

1. \`DEM_COVERAGE_RESTRICTED\`: a fresh 300 m scorer with \`eval_mask = unit.eval_mask & feature_valid_coverage\`. W2 and each feature are selected over exactly the same finite eligible cells and exact-k fraction. It is a **diagnostic**, with boundary censoring, not a competition score.
2. \`FULL_V1_V3\`: original full evaluation unit, original known-fault exclusion, and full unit exact-k budget. W2 score ranks remain unchanged outside DEM coverage. Predeclared 15% and 35% blends of W2/DEM **ranks** are rescored over the full area. Blended ranks are not calibrated probabilities.

Both report weighted TP, FP, FN, number of truth pixels, selected area, full pooled DTI computed **from summed components**, ΔDTI against W2, and per-fault-system recovered/lost diagnostics.

All held-out feature/weight results are **SCREENING_ONLY**, even when a fold set is complete. Never choose a fusion weight or feature by maximizing the displayed held-out score and then call that score independent. Such selection requires new train-only/nested cross-validation before promotion.

Artifacts:

- \`evaluation/dti_screening/coverage.csv\`
- \`evaluation/dti_screening/feature_local_rows.csv\`
- \`evaluation/dti_screening/feature_local_summary.csv\`
- \`evaluation/dti_screening/fusion_full_rows.csv\`
- \`evaluation/dti_screening/fusion_full_summary.csv\`
- \`evaluation/dti_screening/system_recovery.csv\`
- \`evaluation/dti_screening/missing_inputs.csv\`
- \`evaluation/dti_screening/manifest.json\`
- \`figures/dti_screening/*.{png,svg}\`

**Do not compare** a local restricted DTI with a full-region DTI. Only compare candidates that use the same mask, protocol, prediction fraction and holdout.

## Section 16A — cached model DTI without retraining

\`cached_models_dti\` uses the same official-style 300 m scorer and exact-k policy across all available W2 and G07 model predictions. Incomplete \`(model,scenario)\` pairs remain per-unit diagnostics. Only complete fold sets get pooled scenario DTI. This stage works even when the separate G07 promotion stage is blocked.

\`continuous_probability_dti\` additionally evaluates explicitly named G07-B0/B1 score maps **only if** all finite scores lie in [0,1], without an arbitrary 1% area cap. Missing scores count as 0 as in the canonical scorer. Numeric range alone does **not** establish calibration. Raw DEM feature ranks and W2 rank maps are not assumed calibrated probabilities and therefore are not evaluated using the continuous method or paired against B0/B1 continuous outputs.

The binary exact-k and continuous DTI numbers represent different prediction policies. They are not directly comparable as evidence of model improvement without an explicitly matched evaluation procedure.

Additional artifacts:

- \`evaluation/dti_screening/cached_model_status.csv\`
- \`evaluation/dti_screening/cached_model_unit_rows.csv\`
- \`evaluation/dti_screening/cached_model_complete_pooled.csv\`
- \`evaluation/dti_screening/cached_model_paired_vs_w2.csv\`
- \`evaluation/dti_screening/continuous_probability_unit_rows.csv\`
- \`evaluation/dti_screening/continuous_probability_pooled.csv\`
- \`evaluation/dti_screening/continuous_probability_exclusions.csv\`

## Scientific guardrails

- Preserve V1 withheld-system and V3 geographic masks, pixel-exact known-fault exclusions and scorer TP-support convention.
- Use the 300 m distance kernel, alpha=0.2, beta=0.8, no extrapolation into missing DEM.
- Report selected area exactly; 1% is an **internal comparison convention**, not the organizer's mandatory area fraction.
- Pool by **summing weighted components first**. Do not average zone DTIs.
- Use *training-visible labels* for any future model/direction/weight calibration. This notebook is only post-hoc descriptive research.
- Under the current ~0.071% usable DEM footprint and single represented V1 system, even high local DTI is insufficient to justify promotion.
- The repository metric is an organizer-aligned approximation; no official hidden labels are available, and leaderboard parity cannot be asserted from this notebook.

## Validation

Synthetic checks in \`tests/test_g07_dti_screening.py\` cover label-free ranking, global exact-k area, incomplete-fold pooling and restricted-domain score semantics.

**Not executed during code authoring.** Run them in Colab if needed.

## Known issue addressed

\`gems_bench/g07/pipeline.py::evaluate\` previously attempted to sort an empty promotion candidate frame on a nonexistent \`v1_gain\` column. The branch guards the sort, preserving already-computed DTI tables even when no candidate is eligible.
