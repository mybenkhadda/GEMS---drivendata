# Pipeline dependencies

These dependencies were resolved from the library code (the loaders and readers that are actually called), not
from notebook order. The stage graph is declared in `gems_bench/pipeline/stages.py::build_registry`, and the
engine is in `gems_bench/pipeline/executor.py`.

## Stage graph

```
bootstrap
  └─ raw_data
       └─ native_grid
            ├─ geological_eda ── segment_attributes ─┐
            ├─ spatial_folds ────────────────────────┼─ feature_store ─┬─ baselines ─┬─ advanced_models
            ├─ discovery_splits ─────────────────────┘                 │             └─ discovery_validation (~advanced_models)
            └─ dem_inventory ── dem_pilot (+ feature_store)            │                    └─ ensemble
                                                                       └─ full_inference (~ensemble, ~advanced_models, ~baselines)
                                                                              └─ submission
reports  (deps: bootstrap; soft: every stage)
```

`~` marks a soft dependency: it orders the stages and its outputs are used when present, but it never blocks.

## Why each edge exists

| stage | hard dependency | reason in code |
|---|---|---|
| `native_grid` | `raw_data` | `data.load_native` reads the features, faults and template files via `gems_eda.config.resolve_raw` |
| `geological_eda` | `native_grid` | `gems_eda` sections read the same raw rasters; the grid check makes sure they are the competition release |
| `segment_attributes` | `geological_eda` | `segments.segment_id_raster` burns `outputs/eda/datasets/known_fault_segments.gpkg`; `segments.load_segment_table` reads `known_fault_segments.parquet` + `segment_archetypes.parquet` |
| `spatial_folds`, `discovery_splits` | `native_grid` | `folds.make_geo_folds` / `make_discovery_splits` take the valid footprint and labels |
| `feature_store` | `segment_attributes`, `spatial_folds`, `discovery_splits` | features are built through `load_context5` → `load_context4` → `p3.context.load_context`, which **raises** without fold and split folders and (with segments) reads the segment table; `eda:*` features read `outputs/eda/features/{lineament_maps,crossmodal_orientation}.tif` |
| `dem_inventory` | `native_grid` | study-area intersection uses the native valid mask |
| `dem_pilot` | `dem_inventory`, `feature_store` | the `dem_pilot` runner uses the research context; the selection optionally uses G1 / CNN maps when available |
| `baselines` | `feature_store` | retraining reads feature caches; status checks validate maps on the native grid |
| `advanced_models` | `baselines` | the B1 background evidence is the `B0_lineament` map; every `cnn5` base spec is `P4_B_gated` (legacy `config.json` or the committed snapshot); controls are scored when available |
| `discovery_validation` | `baselines` | scores every model with an OOF map under one protocol (`a1_zoo_discovery` with `complete_discovery: []`) |
| `ensemble` | `discovery_validation` | members need an OOF map plus every discovery map |
| `full_inference` | `feature_store` | the registry champion's fold models predict from feature stacks |
| `submission` | `full_inference` | exports the full-region map |

## Artifact classes (what a missing output means)

| class | artifacts | when missing |
|---|---|---|
| **raw required** | `gems-geodawn-numerical-features.tif`, `existing_faults.tif`, `example_submission.tif` | download (fresh URL) or upload; nothing can be regenerated without them |
| **optional raw** | USGS DEM URL JSON (`tnm_urls.json` / `tnm_items.json`) | the DEM branch is disabled |
| **regenerable, deterministic** | native-grid record, Phase 1 minimal EDA set, segment rasters, feature caches | rebuilt from raw data (EDA ≈ 15–45 min CPU, features ≈ 10–90 min CPU) |
| **regenerable splits** | `geo_k3_b20km_buf13km_s2026/`, `discovery_d3_link1km_s2026/` | generated once with the Phase 2 protocol; never regenerated over existing files |
| **retrain** | baseline runs (OOF + discovery maps + fold models), Phase 5 experiments, full-region maps | retrained only when listed or enabled; otherwise reported as `NOT_AVAILABLE` |
| **optional historical** | Phase 3 ablations, Phase 4 drop-group runs, SSL runs, historical figures | skipped; never fabricated |

## Execution rules

| observed state | decision |
|---|---|
| a hard dependency has no usable output | `BLOCKED` (names the dependency and its outcome, and carries its recovery) |
| outputs valid + fingerprint equal | `SKIP_CACHED` |
| outputs valid, no record | `SKIP_CACHED` + record `origin: adopted_existing` |
| outputs valid, record made under another configuration | `INCOMPATIBLE`: nothing is overwritten |
| content incompatible (grid / parameters differ) | `INCOMPATIBLE` |
| missing | `BUILD` |
| partial or invalid | `RESUME` (invalid local files renamed `*.invalid-<ts>` first) |
| disabled in config | `DISABLED` (dependents blocked only if the outputs are missing) |
| exception | `FAILED` + error, traceback tail, recovery action; independent stages continue |
| `KeyboardInterrupt` / training budget | `INTERRUPTED`; the run stops cleanly and the next run resumes |

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
  versioned under `outputs/eda_rebuild/<version>/` and installed group by group only into empty slots.

## Mode → targets

| mode | targets (plus their hard ancestors) |
|---|---|
| `SETUP_ONLY` | bootstrap, raw_data (no download; ancestors validated only) |
| `PREPARE_DATA` | native_grid, geological_eda, segment_attributes, spatial_folds, discovery_splits, feature_store, dem_inventory, dem_pilot |
| `BASELINES` | feature_store, baselines, discovery_validation, reports |
| `ADVANCED_RESEARCH` | feature_store, baselines, advanced_models, discovery_validation, reports |
| `FULL_PIPELINE`, `RESUME` | every stage |
| `SUBMISSION_ONLY` | full_inference, submission (ancestors validated and staged, not built) |
