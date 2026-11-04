# G07 session handoff

**Code revision:** `g07.1-r2` (repair of `g07.1-r1`, 2026-11-04). **Status: authored, uncommitted, NOTHING executed.** No Python, notebook cell, test, Ruff run, download, raster processing, training or evaluation has been run locally. The test suite has never been run. No G07 metric, runtime or DTI value exists. Checks so far: reading the code, tracing the tests by hand, IDE diagnostics.

**Next action:** the user commits, pushes and runs the notebook top to bottom on a Colab GPU runtime (`COLAB_EXECUTION.md`). The first thing to read is section 03 (`audit/pytest_output.txt`).

## Files

* Package `gems_bench/g07/` (21 modules), config `configs/g07/discovery.yaml`, notebook `notebooks/colab/GEMS_G07_MULTISCALE_FAULT_DISCOVERY.ipynb` (sections 00-30), docs `docs/g07/*.md`.
* Tests: `tests/test_g07_{dem_io,dem_alignment,dem_features,surface_candidates,supervision,splits,models,exact_budget,metric_parity,inference,persistence,pipeline}.py` (synthetic).
* Also modified, earlier: `gems_bench/g06/models_segment.py` (`class_weights` balancing).

## What r2 repaired

| area | defect in r1 | now |
|---|---|---|
| coverage | `finalize` used the native resolution while counting work pixels: coverage capped at 0.25 with `work_res_m=2`, below `min_coverage`, so no footprint, no patches, no U-Net | `finalize(work res)` |
| coverage | fine and coarse tiers added up in the mosaic | `dem_coverage` = fine tier; separate `dem_coverage_coarse` selects patches |
| DEM sources | CRS check on the string's EPSG suffix would refuse compound CRSs; unknown vertical unit stayed LOCAL and failed late; horizontal unit kept as a "hint" | strict status + codes; `dem_source_validation.json`; explicit vertical metadata only |
| windows | `analysis_window` mutated lower bounds first; padding could turn an outside tile into a window | raw range, emptiness test, then independent padding / clipping |
| tile edges | one-sided smoothing ran along tile edges; a NaN band diluted its mosaic cell | aligned neighbour halos, edge guard, per-band mosaic weights, seam audit |
| gating | NO_GO made GPU training unreachable | `STRICT_RESEARCH` / `EXPLORATORY_TRAINING`, `training_gate.json`, SCREENING_ONLY propagated |
| GPU | silent CPU fallback; no diagnostics; 3-head model could not be reloaded; target maps recomputed on the full grid per sample | `validate_gpu_runtime`, `gpu_preflight`, finite checks, OOM guidance, history + curves, checkpoints, persisted input contract |
| patches | NaN stored as 0 so the validity channel was always 1; scaler fitted on the first patches; inner fold 0 could have no DEM patch; patches straddled folds | NaN kept, loss mask, random training patches, fold chosen on visible positives, straddlers dropped, `patch_readiness.csv` |
| persistence | tile products never reached Drive (`*/tiles/*` excluded); memory-bound merge | explicit push; disk-backed merge; cached parts; budgets + PARTIAL |
| failures | stages reported COMPLETE with failed fits; notebook continued past failures | `require_stage`, `validate_model_registry`, `execution_summary` |
| candidates | synthetic geometry came from `eval_truth` of a unit, whichever fit used it | generated per fit from `visible`, with provenance and assertions |
| feasibility | positive-only rho in the decision; system ids merged across units, V1 + V3 pooled | signed rho, |rho| screen, (scenario, unit, id) keys, V1 gates, V3 reported |
| evaluation | incomplete fold sets could enter a pooled DTI | `evaluation_status.csv`, only complete pairs pool |
| promotion | portfolio rows failed the exact-budget audit by construction | raw selections audited by their area |
| speed | nested fits always trained; classical models before the U-Net; headers re-read per stage | nested only with candidates; U-Net first; catalog cached per session |

## External artifacts

| artifact | used by | if missing |
|---|---|---|
| `data/raw/geodawn/*` etc. (`inputs.sources`) | all | preflight FAILED |
| `data/raw/dem_1m/**/*.tif` | DEM stages | BLOCKED_INPUT |
| `outputs/g06_geological_feature_store/g06-cda34ef1f1` | datasets, B0-B2 | g06_restore FAILED (never rebuilt here) |
| `outputs/p6_fast_track/stack/maps/<task>_<zone>.npy` | A0, comparison | evaluate BLOCKED_INPUT |
| `outputs/p6_fast_track/stack/maps/full_score.tif` | full_inference, export | blocked (off in the first run) |

## Unverified assumptions and open risks (none measured)

1. **Tests may fail for test reasons.** They were traced by hand only. `TESTS_FAILED` blocks training: read the failure before anything else. Tests that depend on GDAL behaviour: `Resampling.sum` conservation in the coverage tests.
2. **Runtime of DEM features.** Rough reading of the code suggests tens of minutes per 10 km tile on Colab CPUs, so pilot and full builds will likely hit their budgets (`pilot_budget_hours` 2, `fine_budget_hours` 4, `coarse_budget_hours` 2) and return PARTIAL. If windows are slow, `dem_tile_px` can rise (1024 halves the halo overhead) without invalidating tiles.
3. **DEM headers.** Whether real tiles state a vertical unit is unknown: section 04 will say. If not, the user must verify the metadata and set the override.
4. **Feasibility outcome.** Unknown. In the default strict mode NO_GO stops the notebook at section 08 by design; `require_go=False` is the user's decision.
5. **Memory.** `dem_signal_table` loads all DEM bands over the covered box of a unit; evaluation loads maps per unit. Standard-RAM GPU runtimes may struggle.
6. **Disk.** Patches take 1.5 MB each at the defaults; the build refuses if it would not fit and memmaps above `max_push_gib` stay local.
7. **Non-overlapping patches** (`patch_stride_cells = 24`) can leave prediction seams at patch edges; `12` overlaps and blends at 4 x the disk.
8. **U-Net cannot be promoted yet:** it has V1 fits only, so the V3 guard stays unmet. Add V3 fits to `surface.unet.fits` when it looks useful.
9. **Elevation offsets between projects** cannot be removed by the guard; read `tables/dem_seam_audit.csv`.
10. **DEM competition eligibility is not verified** (`dem.competition_eligibility_verified: false`): DEM-based export is refused until the rules are checked.
11. **Cost of the candidate stage** grows with outer x training units (per-fit synthetic tables); never measured.

## Inspect after the first run, in this order

1. `audit/pytest_output.txt`, `audit/gpu_runtime.json`
2. `dem_catalog/dem_source_validation.json`, `dem_invalid_sources.csv`, `tables/dem_coverage.csv`
3. `dem_catalog/feasibility_decision.json`, `training_gate.json`, `reports/DEM_FEASIBILITY.md`
4. `dem_features/dem_features_manifest.json`, `tables/dem_seam_audit.csv`
5. `datasets/surface/patch_readiness.csv`
6. `models/surface_unet_registry.csv`, `figures/training/`, `models/surface_unet/training_history.csv`
7. `evaluation/metrics/evaluation_status.csv`, `exact_budget.csv`, `metrics.csv`, `evaluation/promotion/promotion_decisions.csv`
8. `manifests/execution_summary.csv`, `runtime_errors.json`, `auxiliary_errors.json`

## Decisions after the first run

* Tests fail: fix them first, nothing else counts.
* Gate blocked on NO_GO: the user decides between exploratory training and more DEM coverage. Thresholds stay.
* U-Net trains: compare with W2 on V1 at exact budgets, read the systems recovered. Worth keeping: add V3 fits.
* Then candidates: `RUN.update(GENERATE_CANDIDATES=True, TRAIN_CANDIDATE_RANKER=True, EVALUATE=True)`.

The 0.40 leaderboard DTI is the longer-term target and has not been reached or measured.
