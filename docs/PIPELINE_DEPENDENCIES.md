# Pipeline dependencies

These dependencies were resolved from the library code (the loaders and readers that are actually called), not
from notebook order. The stage graph is declared in `gems_bench/pipeline/stages.py::build_registry`, and the
engine is in `gems_bench/pipeline/executor.py`. The master notebook calls `pipe.step(...)` in exactly the
registry's topological order (enforced by `tests/test_pipeline_recovery.py::test_notebook_calls_every_stage_after_its_prerequisites`).

## Execution order

| # | stage | hard deps | soft deps (ordered first, never blocking) | critical |
|---|---|---|---|---|
| 1 | `bootstrap` | – | – | yes |
| 2 | `raw_data` | bootstrap | – | yes |
| 3 | `native_grid` | raw_data | – | yes |
| 4 | `dem_inventory` | native_grid | – | |
| 5 | `geological_eda` | native_grid | – | yes |
| 6 | `segment_attributes` | geological_eda | – | yes |
| 7 | `spatial_folds` | native_grid | – | yes |
| 8 | `discovery_splits` | native_grid | – | yes |
| 9 | `feature_store` | spatial_folds, discovery_splits | geological_eda, segment_attributes | yes |
| 10 | `dem_pilot` | dem_inventory, feature_store | – | |
| 11 | `baselines` | spatial_folds, discovery_splits | feature_store, segment_attributes | |
| 12 | `advanced_models` | feature_store, baselines | – | |
| 13 | `discovery_validation` | baselines | advanced_models, segment_attributes | |
| 14 | `ensemble` | discovery_validation | – | |
| 15 | `full_inference` | feature_store | ensemble, advanced_models, baselines | |
| 16 | `submission` | full_inference | – | |
| 17 | `reports` | bootstrap | every stage | |

`critical` = data-preparation backbone. Under `failure_policy: interactive` (the notebook default) a failure of a
critical stage prints the full diagnostics and stops the run with `StageFailed`.

```
bootstrap → raw_data → native_grid ─┬─ dem_inventory ───────────────────────────┐
                                    ├─ geological_eda → segment_attributes ┄┄┐  └→ dem_pilot (+ feature_store)
                                    ├─ spatial_folds ────────────────────────┼→ feature_store
                                    └─ discovery_splits ─────────────────────┘
spatial_folds + discovery_splits (┄feature_store, ┄segment_attributes) → baselines → discovery_validation → ensemble
feature_store + baselines → advanced_models         feature_store → full_inference → submission
reports (deps: bootstrap; soft: every stage)                        ┄ = soft dependency
```

## Why each edge exists

| stage | hard dependency | reason in code |
|---|---|---|
| `native_grid` | `raw_data` | `data.load_native` reads the features, faults and template files via `gems_eda.config.resolve_raw` |
| `dem_inventory` | `native_grid` | study-area intersection uses the native valid mask (it was called before `native_grid` in the 8ac074d notebook, hence the `BLOCKED`) |
| `geological_eda` | `native_grid` | `gems_eda` sections read the same raw rasters; the grid check makes sure they are the competition release |
| `segment_attributes` | `geological_eda` | `segments.segment_id_raster` burns `outputs/eda/datasets/known_fault_segments.gpkg`; `segments.load_segment_table` reads `known_fault_segments.parquet` + `segment_archetypes.parquet` |
| `spatial_folds`, `discovery_splits` | `native_grid` | `folds.make_geo_folds` / `make_discovery_splits` take the valid footprint and labels |
| `feature_store` | `spatial_folds`, `discovery_splits` | features are built through `load_context5` → `p3.context.load_context`, which **raises** without fold and split folders. Segments are attached only when present (`PipelineContext.has_segments`); they are used by segment-stratified evaluation, not by feature computation |
| `feature_store` family `phase2_eda` | EDA groups `lineaments`, `crossmodal` | `eda:*` features read `outputs/eda/features/{lineament_maps,crossmodal_orientation}.tif` (`stages.FAMILY_EDA_GROUPS`). If they are missing, this family is deferred and named in a `PipelineStop`; every other family is still built |
| `dem_pilot` | `dem_inventory`, `feature_store` | the pilot runner uses the research context; the selection optionally uses G1 / CNN maps |
| `baselines` | `spatial_folds`, `discovery_splits` | each recipe builds the features it needs (`baselines._ensure_features`); a recipe with `eda:*` inputs (B0_lineament, any `eda:` channel) stops with an explicit message when the EDA is missing, and other baselines continue |
| `advanced_models` | `feature_store`, `baselines` | experiments need the full feature bank; every `cnn5` base spec is `P4_B_gated`. B1's background evidence is the label-free `B0_lineament` map: `experiments.ensure_label_free_map` finds and verifies it (staged from Drive) or builds it with its original CPU recipe before B1 starts (`auto_provision_maps`). A supervised model's map is never substituted. The stage is `always_run` when enabled: `advanced_run` reads the experiment inventory, plans (`REUSE_FULL` / `DONE` / `RESUME` / `RUN` / `NOT_NEEDED`) at `experiments.profile`, and trains only `RUN` / `RESUME` items |
| `discovery_validation` | `baselines` | scores every model with an OOF map under one protocol |
| `ensemble` | `discovery_validation` | members need an OOF map plus every discovery map |
| `full_inference` | `feature_store` | the registry champion's fold models predict from feature stacks |
| `submission` | `full_inference` | exports the full-region map |

### Experiment-level dependencies (inside `advanced_models`)

| experiment | needs | at SCREEN / VALIDATE also needs |
|---|---|---|
| `b0_reference` | `P4_B_gated` spec (snapshot fallback); its own control `P4_B_gated` map for the FULL delta (`INCOMPARABLE` when absent) | – (it *is* the matched control) |
| `b1_bg_weight` | `B0_lineament` map (auto-provisioned, CPU) | `b0_reference` at the same profile |
| `b2_*`, `c1_geometry_heads`, `d2_*`, `e1_gsn_parity` | – | `b0_reference` at the same profile |
| `c2_*`, `c3_*` | – | `c1_geometry_heads` at the same profile |
| `e2_*` … `e5_*` | – | `e1_gsn_parity` at the same profile |

A SCREEN or VALIDATE result never feeds a downstream stage: `discovery_validation`, `ensemble` and
`full_inference` see only FULL maps (`manifest["maps"]`).

### G1 readiness without the advanced feature bank

`G1` (terrain LightGBM) needs only: raw data → native grid → folds + splits → the G1 feature list from
`configs/baselines/legacy_specs/G1.json` (built on demand). It does **not** need the EDA, the Phase 4 RS
features, Phase 5 profiles or the DEM pilot. In the notebook, the EDA and the feature store still run first,
because they are soft dependencies. If either fails, G1 is still trainable in `BASELINES` mode under the batch
policy, or by calling `pipe.step('baselines')` after fixing nothing else.

## Artifact classes (what a missing output means)

| class | artifacts | when missing |
|---|---|---|
| **raw required** | `gems-geodawn-numerical-features.tif`, `existing_faults.tif`, `example_submission.tif` | download (fresh URL) or upload; nothing can be regenerated without them |
| **optional raw** | USGS DEM URL JSON (`tnm_urls.json` / `tnm_items.json`) | the DEM branch is disabled |
| **regenerable, deterministic** | native-grid record, Phase 1 minimal EDA set, segment rasters, feature caches | rebuilt from raw data (EDA ≈ 15–45 min CPU, features ≈ 10–90 min CPU) |
| **regenerable splits** | `geo_k3_b20km_buf13km_s2026/`, `discovery_d3_link1km_s2026/` | generated once with the Phase 2 protocol; never regenerated over existing files |
| **retrain** | baseline runs (OOF + discovery maps + fold models), Phase 5 experiments, full-region maps | retrained only when listed or enabled; otherwise reported as `NOT_AVAILABLE` |
| **optional historical** | Phase 3 ablations, Phase 4 drop-group runs, SSL runs, historical figures | skipped; never fabricated |

Recovery states used in `docs/ARTIFACT_RECOVERY.md`: `VALID_EXISTING` (→ `SKIP_CACHED`), `MISSING_REGENERABLE`
(→ `BUILD`), `PARTIAL` (→ `RESUME`), `INCOMPATIBLE` (preserved; new version), `MISSING_CHECKPOINT` (retrain only
when enabled), `OPTIONAL` (→ `DISABLED`, blocks nothing mandatory).

## Execution rules

| observed state | decision |
|---|---|
| a hard dependency has no usable output | `BLOCKED`: names the dependency, its outcome and the **root failure** upstream, and carries its recovery |
| outputs valid + fingerprint equal | `SKIP_CACHED` |
| outputs valid, no record | `SKIP_CACHED` + record `origin: adopted_existing` |
| outputs valid, record made under another configuration | `INCOMPATIBLE`: nothing is overwritten |
| content incompatible (grid / parameters / source segment set differ) | `INCOMPATIBLE` |
| missing | `BUILD` |
| partial or invalid | `RESUME` (invalid local files renamed `*.invalid-<ts>` first) |
| disabled in config, or a GPU stage without CUDA | `DISABLED` (dependents blocked only if the outputs are missing) |
| exception | `FAILED` + error, `file:line`, full traceback, recovery action (all stored in the stage record) |
| `KeyboardInterrupt` / training budget | `INTERRUPTED`; the run stops cleanly and the next run resumes |

### Failure policies (`failure_policy`, CONFIG key)

* **`interactive`** (notebook default): a failure prints error, `where: file:line in function`, recovery, the
  stages it blocks and the full traceback. If the stage is critical (or a critical stage is `INCOMPATIBLE`),
  `StageFailed` is raised, which stops "Run all" before any dependent cell. Completed artifacts stay on local
  disk and on Drive.
* **`batch`** (library / CLI default; previous behaviour): stages independent of the failure continue. Per-stage
  `BLOCKED` lines are suppressed, and one report groups the blocked stages under their root failure:

  ```
  [pipeline] failure report (fix the root, then re-run: completed stages are skipped)
    ROOT geological_eda FAILED: NameError: name 'version' is not defined  [gems_bench/pipeline/eda_rebuild.py:47 in RebuildPaths]
         recovery: ...
         blocked (2): segment_attributes, ...
  ```

`pipe.explain('<stage>')` prints the same information later, also from a record written in an earlier session.
`pipe.failure_report()` returns the grouped table.

Fingerprints per stage: `native_grid` uses raw file sizes; `geological_eda` uses the EDA config and raw sizes;
folds and splits use their protocol parameters (also encoded in the folder name; non-default parameters get a
suffix such as `_c4`); `feature_store` uses the families, extra features and the advanced switch, plus a per-feature
spec hash in `data/feature_store/feature_provenance.json`; experiments use the experiment config hash (in the
experiment id). Code revisions are recorded in every record but are not compared.

## Versioning

* **Splits**: folder names encode the protocol. A changed protocol creates a new folder, and experiments then
  carry `validation.folds_dir` / `discovery_dir`, so their ids change too.
* **Experiments**: `experiment_id = <name>__<config hash>`. Changing a semantic setting (loss, channels,
  members, model list) creates a new folder; runtime knobs (GPU overrides, stage budget, DEM download approval)
  do not.
* **Fixed-path artifacts** (EDA, segment rasters, feature caches): never overwritten when valid. A rebuilt EDA is
  versioned under `outputs/eda_rebuild/<version>/` and installed group by group only into empty slots. Native
  segment rasters record the sha256 of the segment set they came from (`native_segments_provenance.json`); a
  later EDA change makes them `INCOMPATIBLE` instead of silently mismatched.

## Mode → targets

| mode | targets (plus their hard ancestors) |
|---|---|
| `SETUP_ONLY` | bootstrap, raw_data (no download; ancestors validated only) |
| `PREPARE_DATA` | native_grid, dem_inventory, geological_eda, segment_attributes, spatial_folds, discovery_splits, feature_store, dem_pilot. **No training**: baselines and later stages are only validated |
| `BASELINES` | geological_eda, segment_attributes, feature_store, baselines, discovery_validation, reports |
| `ADVANCED_RESEARCH` | geological_eda, segment_attributes, feature_store, baselines, advanced_models, discovery_validation, reports |
| `FULL_PIPELINE`, `RESUME` | every stage |
| `SUBMISSION_ONLY` | full_inference, submission (ancestors validated and staged, not built) |

The EDA stages are named explicitly in `BASELINES` / `ADVANCED_RESEARCH` because they are only soft
dependencies of `feature_store`. When cached, they take seconds (`SKIP_CACHED`).
