# Postprocessing policy (sections 14, 15)

## Shortlist
The shortlist is fixed in `config["transforms"]`, and `validate` enforces at most four policies including identity.

| name | q | note |
|---|---|---|
| `identity` | p | unchanged |
| `gamma_0.7` | p^0.7 | mild recall expansion |
| `gamma_1.4` | p^1.4 | mild concentration |
| `geo_support` | clip(p (0.9 + 0.2 S), 0, 1) | factor bounded to [0.9, 1.1] |

`S` is the mean of the Atlas layers `xm_evidence_mean` and `xm_multiscale_persistence`, robustly rescaled (5th–95th percentile over the eligible footprint) to [0, 1]. Where no layer is finite, S = 0.5, giving a factor of 1, i.e. unchanged. S is label-free: feature names containing fault / label / dist / cluster / truth / proximity are refused. It is saved as `06_postprocessing/geo_support_S.tif` with provenance in `geo_support_S.json`.

What is not allowed:
* known-fault distance;
* label-derived cluster IDs;
* hidden truth;
* floors;
* binary or area-budget masks;
* a 1 km suppression.

## Freezing
`06_postprocessing/policy_registry.json` is written before any score is computed. If the policies change later, it gets a new `frozen_utc`, so the change is visible.

## Scoring (section 14)
Every policy is scored on every discovery fold (`DISCOVERY_FULL_AREA`, primary exclusion) and on every geographic full-area fold. Outputs:
* `transform_components.csv`;
* `transform_blocks.csv` (10 km block components, used for the bootstrap);
* `transform_regions.csv`;
* `transform_response_curves.csv`.

The all-fold numbers are descriptive and are never used to choose a policy for a fold.

## Cross-fitting (section 15)
For each discovery fold f, the chosen policy is the one with the highest pooled DTI (summed components) over the other folds. Ties go to the earlier policy, identity first. That choice is applied unchanged to f.

The cross-fitted pooled DTI is the sum of these untouched held-out components. Uncertainty comes from a paired 10 km block bootstrap: blocks are resampled within each fold, 1000 replicates, 90% CI of the pooled difference against identity.

## Promotion gates
The gates are fixed in `config["gates"]`; all must pass.
1. Cross-fitted pooled gain ≥ 3% relative to identity.
2. No held-out fold loses more than 5% relative.
3. The same policy is chosen on every fold and on all folds (stability).
4. The chosen policy is not identity.
5. Geographic full-area folds lose no more than 5% under the all-fold choice.
6. The lower bound of the bootstrap 90% CI of the pooled gain is > 0.
7. The policy is label-free.
8. Provenance: splits verified, all discovery full-area rasters valid, and reproduction status `REPRODUCED` (or `V1_RASTER_ABSENT`).

If any gate fails: **`NO RECOVERY PROMOTION — DO NOT SUBMIT`**. There is no W2, top-1% or binary fallback.

Outputs: `crossfit_results.csv`, `promotion_gates.csv`, `crossfit_regional.csv`, `crossfit_bootstrap.csv`, `recovery_decision.json`.

## Export (section 16)
The export is disabled by default (`enable_submission_export=False`). Even when enabled it writes only if every check passes:
* opt-in;
* provenance;
* rasters;
* a promoted decision with a non-identity policy;
* the V1 full raster unchanged since preflight (sha256).

The transform is applied to the original V1 full-model prediction and written with V1's `_write_submission_masked`: finite float32 [0, 1] everywhere, with an internal mask equal to the official footprint. The file is then reopened and checked against the template for:
* CRS, transform and shape;
* finite [0, 1] values;
* mask equal to the footprint;
* continuous values (more than two distinct);
* not equal to the V1 raster.

A file that fails validation is renamed `*.FAILED_VALIDATION.tif`, and the result stays "NO RECOVERY PROMOTION". Outputs: `10_submission/submission_georegime_recovery_v2.tif`, `submission_validation.json`, `submission_sha256.txt`. There is no automatic upload.
