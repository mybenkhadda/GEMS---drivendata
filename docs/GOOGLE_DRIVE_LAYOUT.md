# Google Drive layout

```
MyDrive/GEMS_2/                         (= $GEMS_DRIVE_ROOT; same relative layout as /content/gems_work)
├── data/
│   ├── raw/geodawn/        gems-geodawn-numerical-features.tif, example_submission.tif
│   ├── raw/faults/         existing_faults.tif
│   ├── raw/dem_1m/         tnm_urls.json, <project>/USGS_1M_*.tif (pilot only)
│   ├── external/  intermediate/
│   ├── processed/          splits/<design>/, dem_1m_100m/{tiles/, dem_features_100m.tif}
│   └── feature_store/      phase5/ (p5:* and dem:* .npy + registry)
├── outputs/                Phase 1-4 artifacts uploaded once from the local project (read-only)
│   ├── eda/  benchmark/  phase3/  phase4/
├── experiments/<EXPERIMENT_ID>/        config.yaml, manifest.json, checkpoints/, metrics/, predictions/,
│                                       figures/, logs/, reports/, runs/
├── submissions/            competition GeoTIFFs + provenance JSON
├── model_registry/         registry.csv, transitions.jsonl, evidence/
├── reports/                leaderboard_log.csv, cross-experiment figures
├── cached_artifacts/  environment/     sync ledger, environment snapshots
```

## One-time upload from the local project (`Downloads/GEMS`)

| local | Drive | size | needed for |
|---|---|---|---|
| `gems-geodawn-numerical-features.tif`, `example_submission.tif` | `data/raw/geodawn/` | 0.42 GB | everything |
| `existing_faults.tif` | `data/raw/faults/` | <1 MB | everything |
| `outputs/benchmark/` (folds, datasets, cache/features, experiments, ensembles) | `outputs/benchmark/` | 3.6 GB | everything |
| `outputs/phase3/experiments/` | `outputs/phase3/experiments/` | ~0.8 GB | zoo (CNN-A, G3_D2) |
| `outputs/phase4/` (cache/features, experiments) | `outputs/phase4/` | 4.4 GB | zoo, RS features, A1 |
| `outputs/eda/manifest.json`, `outputs/eda/datasets/` | `outputs/eda/` | small | segment attributes |
| `configs/phase4/*.yaml` | (comes with the Git checkout) | – | – |

`outputs/eda/figures` and other large figure folders are optional.

## Rules

* Colab reads from `/content/gems_work` (fast); Drive is the durable copy. `colab.stage_core()` / `stage_legacy()`
  copy Drive → local, skipping unchanged files (ledger by size + mtime).
* Every experiment stage, every finished CV fold / discovery split / DEM tile, and every checkpoint (at most every
  `ckpt_sync_min` minutes) is pushed to Drive through `.partial` + verification + atomic rename.
* Nothing in this package deletes files on Drive. Local files are deleted only through
  `DriveSync.safe_delete_local`, which first compares checksums with the Drive copy.
* No Drive IDs or credentials anywhere in code or notebooks: paths only.
