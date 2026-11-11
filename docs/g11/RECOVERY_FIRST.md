# G11 recovery-first pipeline: process once, save, then train

**Notebook:** `notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb` (sections 00–20).
**Modules:** `recovery.py`, `feature_versions.py`, `persist.py`, `dem_engine.py`, `feature_dataset.py`, `splits_v2.py`, `training_data.py`. The pipeline orchestration is in `pipeline.py`.

## 1. What the interrupted run actually left on Drive

I inspected Drive metadata through a read-only connector on 2026-11-11. Nothing was downloaded, changed or deleted.

| artifact | state |
|---|---|
| `manifests/features.json` | **COMPLETE** at 2026-11-10 21:39:44 UTC; commit `1618b0e`; config hash `74e4ac4e8ffd64d1`; 651 features (645 predictors). |
| `03_features/{geophysical, terrain_100m, targets}/*.npy` | About 651 layers. Each is 49,116,768 B: one float32 grid of 3730 × 3292 plus the header. Together with the registry, `segment_features.parquet` and one native window, that is about 654 files and 29.8 GiB. |
| `manifests/splits.json` | COMPLETE but **invalid**: train 5,065,055, validation 5,317, stack 318, **test 0** px; embargo 6,400 m. |
| `manifests/train_E0.json` | COMPLETE: raw-band XGBoost on the invalid split; validation AUC-PR was NaN. |
| `manifests/train_E1.json` | **RUNNING**. This is where the session died. |
| `G11_PERSISTENT_PROCESSED_DATA/snapshot_20261110_214635/features/geophysical/elevation_aniso_800m.npy.partial` | A 0-byte copy started at 21:46, outside the G11 code. It is left untouched and is not used. |
| `01_audit/audit_report.md` | 392 DEM tiles VERIFIED_METRES and 264 UNKNOWN. The UNKNOWN tiles are excluded, which loses 12.3 % of the DEM-covered fault pixels. |

So the feature stage itself had finished. The interruption hit training, and the real blocker is the broken split. Recovery therefore verifies the layers once and reuses them. Only the features of the requested new DEM version (10 m) are computed.

## 2. Why the split failed

The v1 generator merged every block touched by a "fault system". It defined fault systems after dilating the mapped faults by half the embargo, i.e. 3.2 km. With about 61,000 fault pixels spread over the area, that dilation connects almost all faults into one component. Almost every 20 km block therefore joined a single group, which went to `train`. The few fault-free groups left were then mostly consumed by the 6.4 km embargo.

## 3. Feature identity (what can and cannot invalidate features)

| dependency class | spec hash covers | does not cover |
|---|---|---|
| geophysical | GeoDAWN sha256, grid signature, `feature_engineering.geophysics`, grid validity / nodata, G06 registry sha | splits, losses, models, training, sampling, evaluation |
| dem | geophysical spec, usable DEM catalog content signature, `dem.{stats_resolution_m, context_scales_m, allow_unknown_vertical, seam_tolerance_m, priority}` | same |
| label | fault raster sha256, grid, `faults`, `labels.{buffers_m, near_band_m}` | same |

DEM-class layer ids carry the resolution (`dem100_slope_mean@r10m`). New 10 m layers are written to `03_features/terrain_100m/r10m/`. The legacy 2 m layers keep their original files and become `@r2m` ids. The two resolutions therefore never share a name in the pool, a file, a manifest row or a dataset version.

## 4. Policies

| `feature_policy` | behaviour |
|---|---|
| `REUSE_VERIFIED` (default) | If `datasets/<version>/COMPLETE.json` exists and its spec equals the current spec, the dataset is loaded and **nothing is computed**. Otherwise: recover, compute only the missing layers, finalize, validate. The same version with a different spec is **refused**, with instructions. |
| `RESUME_MISSING` | As above, but every layer is re-read and re-hashed (FULL verification), and anything invalid is repaired. |
| `FORCE_NEW_VERSION` | Every layer is recomputed into a new, version-private namespace. It is refused if the version already exists. It is never the default. |

`force_recompute=True` re-runs cached downstream stages only. It never recomputes a verified feature layer.

## 5. Persistence protocol (every chunk / layer)

Each chunk or layer goes through these steps:

1. Write it to local scratch.
2. Validate it locally: shape, dtype, no inf, valid-fraction semantics.
3. Copy it to Drive as `<name>.partial<ext>`.
4. Re-read it from Drive and check size and sha256 against the local file.
5. Atomically rename it.
6. Write its completion manifest: a chunk sidecar JSON, or a row in `03_features/layer_manifest.parquet`.

A restart skips only artifacts that have a completion manifest and whose `stat` matches it. Later launches never re-hash 30 GB: each layer gets a `stat` check and a random sample is re-hashed (8 layers by default). `recovery_verify="FULL"` re-hashes everything.

## 6. DEM engine (10 m)

1. **Tile cache.** Each usable 1 m tile is read once from Drive and decimated to 10 m by area averaging over the full-resolution source (overviews are not used by default). The result goes to `<local_root>/dem_tile_cache/r10m/`. A restart on the same VM reuses the cache.
2. **Chunks.** The grid is cut into chunks of 128 × 128 cells (12.8 km). Each chunk gets a halo of the largest context scale, and only chunks with usable DEM are processed. The statistics are vectorized over the 10 × 10 pixels of each 100 m cell.
3. **Commit.** Each chunk is committed to `03_features/dem_chunks/r10m/<chunk>.npz`, together with a sidecar listing its source tiles, resolution, spec and validation.
4. **Assembly.** When every chunk is committed, the 42 `dem100_*@r10m` layers are assembled, re-hashing each chunk first. Then `terrain_scarp_rank@r10m` and the 16 geo × DEM multimodal layers are computed.

`feature_time_budget_min` lets one session stop cleanly. The next launch resumes from the first uncommitted chunk.

## 7. Datasets (`03_features/datasets/<version>/`)

`dataset_manifest.json` holds the cubes (`geophysical` and `dem_100m_summary` name lists), the spec, the grid, the DEM resolution, the availability codes and the file hashes. The other files are:

- `feature_schema.parquet`: channel, layer id, file, sha256, unit, scale, resolution;
- `label_schema.parquet`: targets only;
- `valid_mask.npy`;
- `dem_availability.npy`: 0 = no tile, 1 = tile present but excluded, 2 = usable;
- `feature_availability.parquet`;
- `native_window_index.parquet`;
- `fault_geometry_registry.parquet`;
- `processing_config.json`;
- `source_provenance.json`;
- `validation_report.json`;
- `COMPLETE.json`, written last and only after validation passes.

Layers are referenced in the pool, not copied. A dataset never contains split assignments.

After the default run there will be two versions:
- `g11_features_v1` (DEM 10 m);
- `g11_features_legacy_r2m` (the verified 2 m layers of the interrupted run).

## 8. Split v2 (`05_splits/v2/`)

1. Blocks are 20 km.
2. Small fault systems are kept whole. A small system is an undilated 8-connected component with an extent of at most 20 km. Longer regional systems may cross split boundaries, separated by the full embargo, and are listed in `split_report.json`.
3. Block groups are clustered into 24 compact regions.
4. 2,000 seeded region → split assignments are scored on area, fault and DEM shares after a coarse embargo. The best 12 are recomputed exactly, and the first one that passes every gate is used.

The gates are:
- every partition holds at least half its target area;
- fault positives of at least max(200 px, 40 % of target);
- at least 3 independent blocks per split;
- the embargo is respected;
- enough 128-px patch windows (train ≥ 20, validation ≥ 4);
- usable-DEM faults in every split;
- no small fault system leaks across splits;
- no target-derived predictor;
- no final-test tuning.

`COMPLETE.json` is written only when every gate passes. Only then is the old root-level split marked `SPLIT_V1_INVALID.json`; its files are kept. Models, predictions, evaluation, candidates and submissions live under the namespace `<feature_version>__split_<split_version>`.

## 9. Experiments

| id | model | inputs |
|---|---|---|
| E0 | XGBoost | 100 m geophysics plus G06 indices |
| E1 | XGBoost | E0 plus the DEM 100 m summaries and geo × DEM layers |
| E2 | SegFormer | top-k E0 channels; plus loss ablation and the optional G07 fine-tune |
| E3 | gated multiresolution fusion | geophysics plus DEM patches and a modality mask; plus mode ablation |
| E4 | segment ranker (LightGBM) | candidate geophysical support, DEM morphology and topology |
| E5 | pixel ensemble | fitted on stack, selected on validation |
| B0 | XGBoost | raw-band reference |

Training data (`06_models/<ns>/training_data/<exp>_<key>/`) holds:
- persisted indices per split;
- the train-only scaler;
- class weights;
- the sampling configuration.

When a model directory belongs to another configuration, its checkpoints are **archived**, never deleted. A run resumes from `last.pt` only for the same fingerprint.

## 10. Exact Colab instructions

1. Push the code to `g11-colab-production` (see the final message for the commands). Then open the notebook from that branch on a GPU runtime (T4, L4 or A100).
2. Leave CONFIG at its defaults (`FULL_RESUME`, `REUSE_VERIFIED`, `g11_features_v1`, DEM 10 m, split v2) and run all cells.
3. In §04, check the counts. RECOVERABLE should be close to 651, INCOMPLETE and INCOMPATIBLE should be near zero, and about 59 DEM-class layers should appear as MISSING (the 10 m version).
4. If §05 is interrupted, reconnect, re-run §00–§03 and then §05. Committed chunks are skipped.
5. Later launches: §04–§07 finish in minutes (stat checks plus a sampled re-hash), and nothing is extracted again.
6. To use the 2 m legacy features immediately instead of computing 10 m, set `dem_statistics_resolution_m=2.0` and `feature_version="g11_features_legacy_r2m"`.
7. Experiments that change only splits, losses or hyperparameters: change those settings and re-run. The features stay reused.

## 11. Limitations

- Nothing in this design has run on the real data yet.
- The legacy DEM resolution (2 m) is not recorded inside the legacy files. It is taken from the YAML and notebook default at commit `1618b0e`, and adopting the legacy layers depends on it.
- Legacy adoption also requires the feature-relevant config sections at `1618b0e` to equal the current ones. §02's clone is shallow, so `git show` needs the commit to be fetchable. If it is not, the layers are classified INCOMPATIBLE unless `recovery.accept_unverified_legacy_config: true` is set.
- 264 DEM tiles with UNKNOWN vertical units remain excluded.
- The 10 m tile cache, the run time and Drive throughput are estimates.
