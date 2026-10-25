# Artifact recovery

What to do when an artifact is missing, partial, or incompatible. The master pipeline detects each case and
prints the matching recovery action. This page explains the actions in more detail.

## Minimal bootstrap: the four raw files

With only these on Drive, the pipeline rebuilds everything else that is deterministic:

```
MyDrive/GEMS_2/data/raw/geodawn/gems-geodawn-numerical-features.tif
MyDrive/GEMS_2/data/raw/faults/existing_faults.tif
MyDrive/GEMS_2/data/raw/example_submission.tif        (data/raw/geodawn/ also accepted)
MyDrive/GEMS_2/data/raw/tnm_items.json                (optional; copied to data/raw/dem_1m/tnm_urls.json)
```

Run with `run_mode='PREPARE_DATA'` (the notebook default). In order, it produces the grid record, the DEM
inventory, the minimal Phase 1 EDA set, the segment rasters, folds and splits (reused when already on Drive),
and the feature caches. It stops before any model training.

## Recovery states

| state | meaning | pipeline decision |
|---|---|---|
| `VALID_EXISTING` | present and passes content checks (GeoTIFF grid / CRS / bands, Parquet columns + rows, GeoPackage features, `.npy` shape + dtype, manifest entries, recorded parameters) | `SKIP_CACHED`; adopted with provenance if it has no record |
| `MISSING_REGENERABLE` | absent, derivable from raw inputs | `BUILD` |
| `PARTIAL` | some files of a set present | `RESUME`: completed units are kept. EDA groups are resumed only when the partial state was left by the pipeline's own install (`INSTALL_STATE.json`) |
| `INCOMPATIBLE` | valid content made under another grid / protocol / feature definition / segment set | preserved untouched; a changed protocol gets a new versioned folder |
| `MISSING_CHECKPOINT` | a model / baseline without its run folder or fold models | retrained only when explicitly listed (`train_baselines`, `run_advanced_models`) |
| `OPTIONAL` | DEM inventory / pilot, historical ablations | `DISABLED`; never blocks mandatory work |

No placeholder file, dummy record or fabricated metric is ever written to make a check pass.

## Phase 1 EDA rebuild in detail

* **Where:** computed in `outputs/eda_rebuild/<version>/` (version = `vYYYYmmdd_HHMMSS`), then each missing
  group is copied into `outputs/eda/` (atomic `.partial` + rename). Groups: `segments` (segments parquet + gpkg,
  nodes, archetypes), `lineaments` (`lineament_maps.tif`), `crossmodal` (`crossmodal_orientation.tif`).
* **Finished rebuild:** `REBUILD_COMPLETE.json` (groups, `max_dim`, `seed`, commit). If the session dies after
  the computation, the next run installs from that folder (staged back from Drive) **without recomputing**,
  provided the parameters match.
* **Interrupted install:** `outputs/eda/INSTALL_STATE.json` is `in_progress` during the copy and `complete` after
  the manifest. A group left partial by an `in_progress` install is completed from the recorded rebuild folder.
  A partial group **without** that record is historical: it stays blocked (never mixed).
* **Preflight:** the rebuild checks `geopandas`, `pyogrio`, `shapely`, `rasterio` and `skimage` before the
  15–45 min computation. Without geopandas, `gems_eda.io.write_lines` would silently write a WKT Parquet
  instead of the required GeoPackage.
* **Provenance:** `outputs/eda/manifest.json` (`origin: rebuilt`, `rebuild_version`) and
  `outputs/eda/REBUILD_PROVENANCE.json` (`historical_equivalence: false`). A rebuild is a new artifact version
  made with the same code and defaults as notebook 01, not a claim of byte identity with the historical run.

## Native segment attributes

`segment_attributes` writes `outputs/benchmark/datasets/native_segment_ids.tif` (native 100 m grid, one `seg_id`
band, NaN = no segment), `segment_attributes.parquet` (one row per real Phase 1 segment: geometry, archetype and
strata columns) and `native_segments_provenance.json` (sha256 of the source segment files, label-pixel coverage,
CRS, shape). It stops with a recovery message when:

* the EDA `segments` group is not complete (`run geological_eda first`);
* only one of the two files exists and its source is unknown (a regenerated partner could disagree on `seg_id`);
* the files were derived from another segment set (EDA rebuilt since). They are then `INCOMPATIBLE`: move the
  three files aside on Drive and re-run;
* a `seg_id` in the GeoPackage is missing from the parquet, or no label pixel lies within 300 m of a segment
  (CRS mismatch).

## Per-artifact recovery

| artifact | detected as | automatic action | manual alternative |
|---|---|---|---|
| raw GeoTIFFs | missing / invalid (HTML error page, truncated, wrong band count) | download with a fresh URL from Colab Secrets / the URL cell | upload the files to `data/raw/` on Drive |
| `outputs/eda/manifest.json`, `outputs/eda/datasets/`, `outputs/eda/features/{lineament_maps,crossmodal_orientation}.tif` | missing group | `geological_eda` rebuilds the missing groups only (same `gems_eda` code and defaults as notebook 01) and installs them with `origin: rebuilt` | upload the historical `outputs/eda/` from the local project |
| a **partial** EDA group (for example segments parquet present, gpkg missing) | `partial` | stage fails: groups are never mixed, because segment ids must come from one run | upload the complete historical group, or move the partial files out of `outputs/eda/` on Drive, then re-run |
| an EDA group left partial by an interrupted **pipeline** install (`INSTALL_STATE.json` = `in_progress`) | `partial` | completed from the recorded `outputs/eda_rebuild/<version>/` | – |
| `outputs/benchmark/datasets/{native_segment_ids.tif, segment_attributes.parquet}` | missing | `segment_attributes` derives them from the EDA set | upload from the local project |
| native segment files from an older EDA | `incompatible` | none (never overwritten) | move the three files aside on Drive, re-run |
| folds / discovery splits | missing | generated once with the Phase 2 protocol | upload `outputs/benchmark/folds/` |
| folds / discovery splits already on Drive (current state) | valid | **reused**: grid, band count, roles and recorded parameters are checked; never regenerated | – |
| a **partial** fold or split folder | `partial` / `incompatible` | stage fails; existing files are never overwritten | restore the missing files, or move the folder aside on Drive to regenerate |
| feature caches (`.npy`) | missing / invalid / wrong shape | only the missing features are computed; invalid local files are quarantined | upload `outputs/benchmark/cache/features`, `outputs/phase4/cache/features` |
| a feature cached under another definition (spec hash differs) | `incompatible` | none (never overwritten) | restore the previous feature definition, or move the cache file aside |
| baseline run folders (OOF, discovery maps, fold models) | `NOT_AVAILABLE` / `CV_ONLY` / `INTERRUPTED` | retrained only when listed in `train_baselines` (from `configs/baselines/legacy_specs/`; labelled `historical_equivalence: false`) | upload the legacy run folders (see below) |
| regenerated baseline on other folds | `INCOMPATIBLE` | never reused | move the folder aside and retrain under the current folds |
| `P4_B_gated` `config.json` (base of every `cnn5` experiment) | absent | `base_spec_dict` falls back to the committed snapshot automatically | – |
| Phase 5 experiment state | interrupted | `Experiment` resumes at the next unit and from the last checkpoint | – |
| `B0_lineament` map needed by B1 (previous run: `b1_bg_weight needs the B0_lineament map, which is not available`) | `NOT_AVAILABLE` | `ensure_label_free_map`: staged from Drive and verified (run finished, native grid, fold provenance), else rebuilt with the original recipe (rank-normalised `eda:fused_score`, CPU, minutes) and pushed | `train_baselines=['B0_lineament']` |
| a completed FULL experiment while `experiment_profile` is SCREEN / VALIDATE | `REUSE_FULL` | reused for reporting; never retrained | – |
| a resume checkpoint whose sha256 differs from `<unit>_latest.json` | corrupt / truncated | falls back to `<unit>_latest.prev.pt` or the older `_ep<NNN>.pt` files | – |
| a resume checkpoint written by another spec | incompatible | refused (`RuntimeError`), never loaded | start a new experiment id (change of profile = new folder automatically) |
| a final unit model (`<unit>_model.pt`) that cannot be read | invalid | that unit is retrained; other finished units are kept | – |
| SCREEN / VALIDATE folders (`<name>__screen__<hash>`) | – | never adopted as FULL results, never scored on the leaderboard | delete only by hand on Drive if no longer wanted |

### Inspect before you spend GPU time

```python
pipe.inventory()          # every experiment folder: profile, completed folds / splits, metrics, reusable, next action
pipe.plan_experiments()   # what this session would train at experiments.profile
pipe.compare_results()    # reports/experiment_screening: metrics, paired deltas, resource usage, shortlist, figures
```

All three are read-only.
| promoted model | none | `full_inference` stops: promotion is a manual decision | `promote('<model>', '<experiment_id>')` after reading `promotion_decisions.csv` |

## Restoring legacy results from the local project (optional, saves GPU time)

Historical Phase 2–4 results exist only in the local project (`Downloads/GEMS/outputs/`, about 9 GB). They cannot
be regenerated identically. Uploading them makes the baselines `AVAILABLE` with `origin: legacy`, and
avoids retraining.

| local | Drive | gives |
|---|---|---|
| `outputs/benchmark/experiments/{lgbm_G1_s2026, lgbm_G3_s2026, b0_lineament_fused_s2026, disc*_lgbm_G1_s2026, disc*_lgbm_G3_s2026}` | same path | G1, G3, B0 floor |
| `outputs/phase3/experiments/p3_CNN_A_dti_p256_s2026` | same path | CNN-A (CV) |
| `outputs/phase4/experiments/{p4_RS05_G3_noSS_RS_s2026, p4_DL00_CNN_B_dti_p256_s2026, p4_DL02_CNN_TMGB_dti_p256_s2026, p4_P4_B_gated_p256_s2026, disc*_RS05_G3_noSS_RS, disc*_CNN_A_dti, disc*_DL00_CNN_B_dti, disc*_DL02_CNN_TMGB_dti}` | same path | RS05, CNN-A / CNN-B / DL02 discovery, gated dual encoder |
| `outputs/benchmark/cache/features`, `outputs/phase4/cache/features` | same path | skips feature computation |

Legacy runs were trained on the historical fold files. If the folds on Drive were regenerated (as in the
observed state), check that they are identical to the local ones before mixing legacy maps with new runs:

```bash
sha256sum outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/fold_map.tif           # local
sha256sum /content/drive/MyDrive/GEMS_2/outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/fold_map.tif
```

If they differ, legacy OOF maps were made on other folds. Either upload the local fold folders, replacing the
regenerated ones by hand (the pipeline will not do this), or retrain the baselines under the regenerated folds.
Do not compare models across the two designs. Legacy runs record their fold design only by folder name; runs
regenerated by the pipeline record the fold-map sha256 in `PROVENANCE.json`, and they are rejected
(`INCOMPATIBLE`) when the folds change.

## Inspecting failures

```python
pipe.explain('geological_eda')  # outcome, error, file:line, recovery, root cause, full traceback (also from a past session)
pipe.failure_report()           # each root failure and the stages it blocks
pipe.status()                   # state + reason + recovery per stage
pipe.record('feature_store')    # full record: error, location, traceback tail, fingerprint, outputs, code commit
pipe.events(50)                 # transition log (environment/pipeline/events.jsonl on Drive)
```

The record is saved at `environment/pipeline/stages/<stage>.json` on Drive. A `FAILED` stage whose location is
in `gems_bench/` or `gems_eda/` is a code defect: fix it, push to GitHub, restart the Colab session and run again
from section 1. Completed stages are skipped.

## Things the pipeline never does

It never deletes Drive files, overwrites valid or historical artifacts, regenerates existing folds, fabricates
scores, maps, or checkpoints, uploads submissions, or stores URLs or tokens. Moving a conflicting artifact aside
is always a manual step.
