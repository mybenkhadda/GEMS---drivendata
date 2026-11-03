# G07 session handoff

**Code revision:** `g07.1-r1` (authored 2026-11-03). **Status: authored, uncommitted, NOTHING executed** - no
Python, notebook cell, test, Ruff run, download, raster processing, training or evaluation was run locally. No metric,
runtime or DTI value exists for G07.

## Files

* Package `gems_bench/g07/` (21 modules: `__init__, config, contracts, inventory, dem_io, dem_features,
  terrain_structure, candidate_generation, training_labels, datasets, models_surface, models_subsurface,
  models_candidates, crossfit, confidence, representation, evaluation, inference, visualization, persistence,
  pipeline`).
* Notebook `notebooks/colab/GEMS_G07_MULTISCALE_FAULT_DISCOVERY.ipynb` (sections 00-30).
* Config `configs/g07/discovery.yaml`.
* Tests `tests/test_g07_{dem_io, dem_alignment, dem_features, surface_candidates, supervision, splits, models,
  exact_budget, metric_parity, inference, persistence}.py` (synthetic only, unrun).
* Docs `docs/g07/*.md` (10 files).
* Modified: `gems_bench/g06/models_segment.py` - `class_weights` balancing fix (half of the total weight is fixed
  before the per-class rescale; previously sequential rescaling skewed the balance).

## External artifacts needed (Drive)

| artifact | used by | if missing |
|---|---|---|
| `data/raw/geodawn/*` (raster, faults, template) | all | preflight FAILED |
| `data/raw/dem_1m/tnm_urls.json` and / or DEM tiles | DEM stages | `dem_feasibility` BLOCKED_INPUT (geology path continues) |
| COMPLETE G06 feature store (`outputs/g06_*/<run_id>/features/`) | datasets, B0-B2, D1 | datasets FAILED |
| P6 held-out maps `outputs/p6_fast_track/stack/maps/<task>_<zone>.npy` | A0, D1W, D2, E1 | W2 rows missing, portfolio blocked |
| P6 `stack/maps/full_score.tif` | full_inference / export | full-region stage blocked |
| folds / splits used by G06 | isolation | datasets FAILED |

## Reused components

G06 contracts, feature store, splits, labels, evaluation and persistence (aliased); `gems_bench.dem.*` inventory,
download (guarded, disabled by default) and morphology; `gems3.submission.finalise / validate`,
`gems_bench.submission.write_submission`; FastScorer / M7 protocol keys.

## Blockers and unverified assumptions

* Whether DEM tiles / `tnm_urls.json` are present on Drive is **unknown**; coverage of the study area unknown.
* USGS 3DEP is public domain (DATA_PROVENANCE); **competition eligibility of external DEM data is NOT verified**
  (`dem.competition_eligibility_verified: false`) - DEM-based maps cannot be exported until it is set after a rules
  check.
* Vertical units assumed metres unless the header says otherwise (unknown unit -> BLOCKED_INPUT).
* Memory / time of fine-tier processing at 2 m is estimated, not measured.
* The feasibility thresholds (`feasibility.go_if`) are design choices, predeclared, not tuned.
* The legacy `score >= threshold` behaviour is assumed to have produced tie excess for W2; `exact_budget.csv` will
  measure it.
* Static IDE diagnostics only; the code has never been imported.

## Colab execution order

1. Run 1: PREFLIGHT, DEM_INVENTORY, DEM_COVERAGE, DEM_FEASIBILITY, REPORT (synthetic tests run first in Colab).
2. GO -> DEM_FEATURES, BUILD_DATASETS, TRAIN_SURFACE_CLASSIC, TRAIN_SUBSURFACE, EVALUATE.
   NO_GO / BLOCKED_INPUT -> BUILD_DATASETS, TRAIN_SUBSURFACE, EVALUATE (geology path B0-B2).
3. GENERATE_CANDIDATES, TRAIN_CANDIDATE_RANKER, EVALUATE.
4. Conditional TRAIN_SURFACE_UNET (GPU) only if A1 / A2 add systems vs W2.
5. FULL_INFERENCE; EXPORT_SUBMISSION only after reviewing promotion decisions. Never auto-upload.

## Inspect after the first run

`audit/tests.csv` + `pytest_output.txt`; `manifests/runtime_errors.json`; `dependencies/missing_dependencies.json`;
`dem_catalog/dem_inventory.csv`, `source_licenses.csv`; `tables/dem_coverage.csv` (eligible share);
`tables/dem_fault_signal.csv` (AUC, `rho_with_coverage` artefact screen); `dem_catalog/feasibility_decision.json`;
`reports/DEM_FEASIBILITY.md` and `figures/dem_feasibility/`.
