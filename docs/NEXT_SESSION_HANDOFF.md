# Next-session handoff (2026-10-22, after the first master-notebook run)

## What the Colab run at 8ac074d showed

Completed: `bootstrap`, `raw_data`, `native_grid`, `spatial_folds`, `discovery_splits`, `reports`.

| symptom | root cause | fix |
|---|---|---|
| `[geological_eda] FAILED: NameError: name 'version' is not defined`; 8 stages BLOCKED | `eda_rebuild._rebuild_paths` declared `version: str = version` inside a nested dataclass. A class body that assigns a name cannot read the enclosing function's variable of that name | field renamed `rebuild_version`, constructed with keywords; version string validated |
| `[dem_inventory] BLOCKED: native_grid (not run)` | notebook §8 (DEM inventory) ran before §9 (native grid) | notebook and registry reordered: native_grid → dem_inventory; a test enforces notebook order = registry order |
| `cuda: False` | CPU runtime (`torch 2.11.0+cpu`); the `accelerator: GPU` metadata does not attach a GPU | GPU stages DISABLED with instructions; GPU baselines "not attempted"; torch CUDA build reported |

## Changes in this session (code authored, nothing executed)

* `gems_bench/pipeline/eda_rebuild.py`: NameError fix; dependency preflight (geopandas / pyogrio / shapely /
  rasterio / skimage) before the 15–45 min run; `REBUILD_COMPLETE.json` reuse of a finished rebuild;
  `INSTALL_STATE.json` completion of our own interrupted install; per-group output validation before install;
  native segments: EDA group must be complete, gpkg / parquet seg_id consistency, label coverage > 0,
  `native_segments_provenance.json` (sha256 of the source segment set).
* `gems_bench/segments.py`: empty-geometry guard before `rasterize`.
* `gems_bench/artifacts/validation.py`: `validate_native_segments(..., eda_dir=)` → `INCOMPATIBLE` when the EDA
  changed after the segment rasters were made.
* `gems_bench/pipeline/stages.py`: registry in notebook order; `critical` flags; `feature_store` hard deps =
  folds + splits (EDA soft; `phase2_eda` family deferred with a precise error when EDA rasters are missing);
  `baselines` hard deps = folds + splits (feature_store / segments soft), so G1 never waits for EDA / RS / DEM;
  `advanced_models` needs feature_store; GPU stages gated by `pc.cuda_ready()`; GPU baselines skipped (noted) on
  CPU; `eda_run` stages finished / interrupted rebuild folders from Drive; `seg_run` drops the cached context.
* `gems_bench/pipeline/executor.py`: `failure_policy` (`interactive` raises `StageFailed` on a critical failure;
  `batch` groups blocked stages under root failures); `StageResult.location` (file:line), full traceback, root
  causes; `failure_report()`; failure location persisted in the stage record.
* `gems_bench/pipeline/__init__.py`: `pipe.overview()`, `pipe.explain(stage)`, `pipe.failure_report()`;
  `run_stage` prints concise BLOCKED lines naming the root.
* `gems_bench/pipeline/context.py`: `StageFailed`, `cuda_ready()`, `has_segments()`; `ctx5` loads segments only
  when present.
* `gems_bench/pipeline/{config.py,dependencies.py}`, `configs/pipeline/master.yaml`: `failure_policy` key
  (default `batch`); BASELINES / ADVANCED_RESEARCH name the EDA stages explicitly.
* `gems_bench/pipeline/baselines.py`: recipes with `eda:*` inputs stop with an explicit message when the EDA is
  missing (other baselines continue).
* `gems_bench/training/gpu.py`, `runtime/colab.py`: report `torch` version + CUDA build, `cuda_available`.
* `notebooks/colab/GEMS_MASTER_PIPELINE.ipynb`: PREPARE_DATA default + interactive policy; token-free clone,
  env-header auth for private forks, git errors raised, branch vs pinned SHA printed, stale-kernel guard;
  `pipe.overview()`; corrected section order; GPU guidance; resume / inspect / update instructions.
* `notebooks/colab/00_setup_and_drive.ipynb`: signed DrivenData URLs removed (Colab Secrets `GEMS_URL_*`);
  `00`–`07`: token no longer embedded in the clone URL (env header), remote URL reset.
* Tests (authored, **not run**): `tests/test_pipeline_recovery.py` (new); `tests/test_pipeline_engine.py`
  (registry-graph assertions updated for the soft EDA dependency).
* Docs: `MASTER_COLAB_GUIDE.md`, `ARTIFACT_RECOVERY.md`, `PIPELINE_DEPENDENCIES.md`, this file.

## Run next

1. Locally: `pytest tests/test_pipeline_recovery.py tests/test_pipeline_engine.py tests/test_pipeline_artifacts.py`.
2. Colab: **Runtime → Restart session**, open the master notebook from GitHub (`ref='main'`), keep
   `run_mode='PREPARE_DATA'`, run all. Expect `raw_data` / `native_grid` / `spatial_folds` / `discovery_splits`
   = `SKIP_CACHED`, `geological_eda` = BUILD (15–45 min), then `segment_attributes`, `feature_store`.
3. Then `run_mode='BASELINES'`, `train_baselines=['G1']` (CPU).

## Unverified at runtime (check on the next run)

* The full `gems_eda` s2→s6 chain on Colab's current library versions (skeletonisation, GeoPackage writing via
  pyogrio, memory of `s3_profiles` / `s5_crossmodal` at 2048 px).
* `segment_id_raster` CRS comparison between the GeoPackage (pyproj) and the native grid (rasterio CRS).
* `feature_store` end-to-end (Phase 2 + `eda:*` + Phase 4 RS used): never reached yet.
* G1 retraining through `pc.ctx5()` without segments, and the scoreboard on a partial zoo.
* The env-header Git auth for private forks (`GIT_CONFIG_COUNT` needs git ≥ 2.31; Colab ships newer).
* `_external()` frame filter in `error_location` on Colab's Python layout (`/usr/lib/python3.x`,
  `/usr/local/lib/python3.x/dist-packages`).

## Remaining risks

* The signed DrivenData URLs removed from notebook 00 remain in Git history (commits 32c0ad5 / 1a9601d). They
  expired 24 h after 2026-10-21 16:12 UTC, and history was not rewritten (no force-push).
* A rebuilt EDA is a new version (`historical_equivalence: false`). Legacy OOF maps made with historical EDA
  features are not directly comparable unless the historical `outputs/eda/` is restored.
* Regenerated folds on Drive may differ from the historical local folds. Compare `fold_map.tif` sha256 before
  mixing legacy runs (`ARTIFACT_RECOVERY.md`).
* `SubmissionRunner` still supports only `cnn5` and LightGBM champions (pre-existing).
