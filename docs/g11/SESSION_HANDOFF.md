# G11 session handoff

## 2026-11-11: Recovery-first pipeline ("process once, save, then train")

**Drive findings (metadata only, read-only connector)**
- The interrupted run's feature stage is **COMPLETE** (651 layers).
- Its split is invalid (test 0 px).
- E0 trained on that invalid split; E1 was RUNNING when the session died.
- Details in [RECOVERY_FIRST.md §1](RECOVERY_FIRST.md).
- Root cause of the split failure: fault-system merging after a 3.2 km dilation.

**Added**
- `recovery.py`: `recover_feature_stage`.
- `feature_versions.py`: specs, layer ids, expected-layer plan, `LayerManifest`.
- `persist.py`: verified commit protocol.
- `dem_engine.py`: tile cache, chunks, `process_missing_feature_chunks`, `validate_feature_chunk`, `commit_feature_chunk`, `resume_feature_processing`, assembly.
- `feature_dataset.py`: `finalize_feature_dataset`, `load_feature_dataset`, `validate_feature_dataset`, `PoolView`.
- `splits_v2.py`: region-based generator plus hard gates.
- `training_data.py`: `build_training_dataset`.
- `tests/test_recovery.py`.
- `docs/g11/RECOVERY_FIRST.md`.

**Changed**
- `pipeline.py` rewritten around `feature_policy`, with:
  - section-scoped fingerprints, so loss, split and hyperparameter changes no longer invalidate features;
  - the run namespace `<feature_version>__split_<split_version>` for every downstream artifact;
  - `predict_all` regeneration;
  - checkpoint archiving.
- Experiment ids renumbered to the brief: E0 geophysics XGB, E1 + DEM, E2 SegFormer, E3 gated, E4 ranker, E5 ensemble, B0 raw baseline.
- `dem_reader.dem_chunk_stats` split out of the aggregation.
- `notebook_config`: new modes and keys; DEM statistics default 10 m.
- Master notebook regenerated (§00–§20); phase notebooks regenerated.
- YAML gained `features`, `recovery`, `spatial_splits.version/v2` and the DEM engine keys.

**Not executed:** anything on the real data. No real recovery, DEM chunk, split, model or submission exists from this code.

## Next step
1. Commit and push to `g11-colab-production` (needs the user's authorization).
2. Run the master notebook with the defaults.
3. In §04, check the class counts and the INCOMPATIBLE reasons. The legacy config check against commit `1618b0e` must say MATCH.
4. §05 computes the 59 DEM-class layers at 10 m. It can be split over sessions with `feature_time_budget_min`.
5. §09: if the split gates fail, read `split_gates_FAILED.parquet`.

## Decisions (check them)
- **The legacy 2 m DEM resolution is an assumption.** It comes from the YAML and notebook default at `1618b0e`, because the files themselves don't record it.
- **Legacy layers are adopted only when** all of these hold:
  - the stage manifest is COMPLETE;
  - the sources are unchanged;
  - the registry hash matches;
  - the finite fraction equals the recorded value;
  - the write time precedes completion;
  - the layer name is in the current plan;
  - the feature config sections are equal at the legacy commit.
- **Default = 10 m new version.** The 2 m layers become `g11_features_legacy_r2m` (usable at once by setting DEM 2.0 and that version name).
- **Regional fault systems longer than one block may span splits.** The embargo separates the parts. Small systems never span splits.
- **The ingest stage no longer writes targets.** Labels are versioned layers.
