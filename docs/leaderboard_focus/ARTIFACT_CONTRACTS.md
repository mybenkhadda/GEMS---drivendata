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
| w2_maps | `stack/maps/{geo,disc_official}_{0,1,2}.npy` (required by name; extras listed) | P6 W2 | zone-stitched champion, recomputation |
| boost_a_grid / manifest | `boost/benchmark/{grid.csv, manifest.json}` | P6 boost | selection, recomputation (the ONLY grid read: `champion.required_grids`) |
| m9_selection | `outputs/m9_geofault_production/baseline/baseline_selection.json` | M9 | fusion weight w |
| b0 / b1 manifests and maps | `experiments/b0_reference__*/`, `experiments/b1_bg_weight__*/` (`predictions/{oof,disc0..2}.tif` required by name; other GeoTIFFs such as `oof_fold<k>.tif` accepted and listed as `extra_files`; exactly one experiment directory with the recorded `expected_id`) | Phase 5 | fusion, W2 inputs |
| geo_folds / disc_splits | `outputs/benchmark/folds/*` | pipeline | units |
| template | `data/raw/geodawn/example_submission.tif` | DrivenData | export |

The following are optional; M1 runs without them, but the parity diagnosis is then partial:

* `stack/maps/full_score.tif` (historical);
* `boost/benchmark/status.json` (grid provenance);
* `recovery/export_parity/{decision.json, train_inference_parity.csv, policy_comparison.csv, unit_overlap.csv}`;
* `submissions/p6_stack_fusion*.json` (historical recipe sidecar; the `.tif` is inventoried, never staged).

## Grid loading contract (`p6/full_benchmark.load_grids`)

* `load_grids(fg)` (P6 callers): every configured grid; an absent or damaged one is recorded `usable: false` with its
  `grid_status` and never stops the caller.
* `load_grids(fg, sources=["boost_a"], required=["boost_a"])` (M1): only boost A is read; boost B/C/D, `rec_x3` and
  `w1_seeds` are not opened. Unknown names raise `KeyError`.
* `grid_status`: `OK`, `MISSING`, `EMPTY_FILE` (0 bytes / whitespace, e.g. the observed 1-byte `seeds/grid.csv`),
  `NO_HEADER`, `MALFORMED` (parser / encoding error), `ZERO_ROWS` (header only), `SCHEMA_INVALID` (required column
  absent or non-numeric official components). A required grid that is not OK raises `GridUnavailable` → the stage is
  BLOCKED with `selection/grid_provenance.{json,csv}`. Oracle rows always raise `OracleLeak`.

## Staging

Stage 1 copies exactly the `staged: true` files (W2 maps and metadata, boost-A grid and manifest, recovery tables,
M9 selection, historical sidecar). The session re-stages exactly that list (`session.stage_frozen_inputs`) instead of
`p6.full_gpu.stage_inputs` (which copies whole `stack/`, `boost/`, `seeds/`, `submissions/` directories). Never staged:
boost B/C/D, `recovery/segments`, `seeds/`, `stack/models`, CNN checkpoint directories, historical submission
rasters. CNN prediction rasters are read through the pipeline's own staging.

## Fingerprints and cache invalidation

Every stage fingerprint (except `recipe`) also includes the stage's implementation hash (`session.IMPL_MODULES`:
the M1 and P6 modules the stage's numbers depend on) and the numpy / pandas / scipy / rasterio / lightgbm /
scikit-learn versions.

| stage | fingerprint covers | changes when |
|---|---|---|
| artifacts | id + sha256 (or size + mtime above 256 MB) of every inspected file | any input file changes |
| selection | inventory fp, candidate specs, selection rule, recorded values, recovery decision, grid sources / config | an input, a rule or the selection code changes |
| validation | selection fp, tolerances, recorded values | selection, tolerances or validation code change |
| parity | validation fp, recovery decision | validation or parity code changes |
| promotion | selection / validation / parity fps, promotion + selection config | never cached (re-evaluated each run) |
| submissions | parity fp, champion recipe, submission config, promotion fp | the recipe changes (a new file name, never an overwrite) |
| recipe id | the recipe content only | the recipe changes (code edits never rename the file) |

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
* **`selection/grid_provenance.csv`:** `source, required, usable, grid_status, detail, path, bytes, rows, sha256,
  stage_status`.
* **`selection/promotion_gates.csv`:** `gate, status (PASS / WARNING / BLOCKED / PENDING), mandatory, evidence`.
* **`validation/recorded_reproduction.csv`:** `recorded_id, source, objective, protocol, metric, recorded,
  reproduced, abs_diff, tolerance, status (REPRODUCED / NOT_REPRODUCED / NOT_AVAILABLE), comparable_to_champion,
  comparability_note, note`.
* **`validation/protocol_comparison.csv`:** `value_id, kind, protocol, metric, value, comparable_to_champion, note`.
* **`submissions/submission_manifest.json`:** `status`, `reasons`, `file`, `sha256`, `candidate`, `recipe_id`,
  `recipe_fp`, `cell`, `offline`, `p6_readiness`, `coverage`, `promotion`, `uploaded: false`.
* **`manifests/<stage>.json`:** `stage, status, reasons, warnings, fingerprint, impl, inputs, outputs, m1_version,
  code_rev, code, seconds, written` (+ `traceback` when FAILED); every write is also kept in `manifests/history/`.
* **`leaderboard/leaderboard_history.csv`:** the `reporting.LB_COLUMNS`.

## Ownership

| directory | owner | M1 access |
|---|---|---|
| `outputs/p6_fast_track/` | P6 | read-only; M1 never writes there |
| `outputs/p6_boost/` | p6g | untouched |
| `outputs/leaderboard_focus/m1_champion_recovery/` | M1 | the only writable root; p6g writes its `submissions/` and `submission/` there because M1 sets the p6g root to it |
