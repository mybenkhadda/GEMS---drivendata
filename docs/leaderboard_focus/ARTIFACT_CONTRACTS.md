# M1 artifact contracts

## Geospatial contract

* **Grid:** the GeoDAWN feature grid at 100 m, EPSG:32611. `example_submission.tif` must match it
  (`data.template_matches()`; `final_submission.inspect_spec` raises otherwise).
* **Submission raster:**
  * one float32 band;
  * template transform and shape;
  * values in [0, 1];
  * NaN exactly where the template is NaN (`gems3.submission.finalise(outside_valid='template')`);
  * 0 where the template is finite but no positive pixel was selected.
* **Unit crops:** `Unit.sl` is the bounding box of the evaluation mask plus a kernel pad. Every `.npy` map has the
  unit crop shape, and M1 checks it.

## Required artifacts (Drive-relative; `configs/leaderboard_focus/m1.yaml` → `artifacts`)

| id | path | owner | needed for |
|---|---|---|---|
| w2_candidate | `outputs/p6_fast_track/stack/candidate.json` | P6 W2 | W2 feature schema, PROMOTE gate |
| w2_rows | `stack/features/rows.parquet` | P6 W2 | provenance hash |
| w2_stack_grid | `stack/grid.csv` | P6 W2 | reproduction of the recorded 0.224640 |
| w2_maps | `stack/maps/{geo,disc_official}_{0,1,2}.npy` | P6 W2 | zone-stitched champion, recomputation |
| boost_a_grid / manifest | `boost/benchmark/{grid.csv, manifest.json}` | P6 boost | selection, recomputation |
| m9_selection | `outputs/m9_geofault_production/baseline/baseline_selection.json` | M9 | fusion weight w |
| b0 / b1 manifests and maps | `experiments/b0_reference__*/`, `experiments/b1_bg_weight__*/` (`predictions/{oof,disc0..2}.tif`) | Phase 5 | fusion, W2 inputs |
| geo_folds / disc_splits | `outputs/benchmark/folds/*` | pipeline | units |
| template | `data/raw/geodawn/example_submission.tif` | DrivenData | export |

The following are optional; M1 runs without them, but the parity diagnosis is then partial:

* `stack/maps/full_score.tif` (historical);
* `recovery/export_parity/{decision.json, train_inference_parity.csv, policy_comparison.csv, unit_overlap.csv}`;
* `submissions/p6_stack_fusion*.tif`.

CNN checkpoint directories are **never** staged: M1 uses no checkpoint.

## Fingerprints and cache invalidation

| stage | fingerprint covers | changes when |
|---|---|---|
| artifacts | id + sha256 (or size + mtime above 256 MB) of every inspected file | any input file changes |
| selection | inventory fp, candidate specs, selection rule, recorded values, recovery decision | an input or a rule changes |
| validation | selection fp, tolerances, recorded values | selection or tolerances change |
| parity | validation fp, recovery decision | – |
| submissions | parity fp, champion recipe, submission config | the recipe changes (a new file name, never an overwrite) |

## Schemas

* **`champion_candidates.csv`:**
  * `candidate_id, label, role, family, source_grid, deploy, status, status_reason, artifacts`;
  * the recomputed nested numbers: `disc_dti, disc_worst, disc_worst_zone, geo_dti, objective`;
  * `deploy_cell, deployable, geo_floor, geo_floor_met`;
  * the records: `recorded_disc_dti, recorded_disc_worst, recorded_geo_dti, recorded_objective`;
  * the decision: `eligible, eligibility_reasons, parity_status, export_feasible, decision, is_champion, disc_per_zone`.
* **`champion_recipe.yaml`:** `recipe_id`, `candidate`, `deploy_kind`, `selection{objective, rule, selection_id}`,
  `policy{grid_candidate, rep, frac, blend, fusion_w_all_zones}`, `representation`, `sigma_px`, `score`,
  `construction`, `eligibility`, `ranking`, `calibration`, `raster`, `sources{hashes}`.
* **`validation/per_zone_metrics.csv`:** one row per (task, zone), holding the saved and recomputed TP / FP / FN /
  n_truth, both DTIs and a status.
* **`submissions/submission_manifest.json`:** `status`, `reasons`, `file`, `sha256`, `candidate`, `recipe_id`,
  `recipe_fp`, `cell`, `offline`, `p6_readiness`, `coverage`, `uploaded: false`.
* **`leaderboard/leaderboard_history.csv`:** the `reporting.LB_COLUMNS`.

## Ownership

| directory | owner | M1 access |
|---|---|---|
| `outputs/p6_fast_track/` | P6 | read-only; M1 never writes there |
| `outputs/p6_boost/` | p6g | untouched |
| `outputs/leaderboard_focus/m1_champion_recovery/` | M1 | the only writable root; p6g writes its `submissions/` and `submission/` there because M1 sets the p6g root to it |
