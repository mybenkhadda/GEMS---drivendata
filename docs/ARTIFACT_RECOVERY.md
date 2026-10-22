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

Run with `run_mode='PREPARE_DATA'` (or the default `RESUME`). In order, it produces the grid record, the minimal
Phase 1 EDA set, the segment rasters, folds and splits (reused when already on Drive), and the feature caches.

## Per-artifact recovery

| artifact | detected as | automatic action | manual alternative |
|---|---|---|---|
| raw GeoTIFFs | missing / invalid (HTML error page, truncated, wrong band count) | download with a fresh URL from Colab Secrets / the URL cell | upload the files to `data/raw/` on Drive |
| `outputs/eda/manifest.json`, `outputs/eda/datasets/`, `outputs/eda/features/{lineament_maps,crossmodal_orientation}.tif` | missing group | `geological_eda` rebuilds the missing groups only (same `gems_eda` code and defaults as notebook 01) and installs them with `origin: rebuilt` | upload the historical `outputs/eda/` from the local project |
| a **partial** EDA group (for example segments parquet present, gpkg missing) | `partial` | stage fails: groups are never mixed, because segment ids must come from one run | upload the complete historical group, or move the partial files out of `outputs/eda/` on Drive, then re-run |
| `outputs/benchmark/datasets/{native_segment_ids.tif, segment_attributes.parquet}` | missing | `segment_attributes` derives them from the EDA set | upload from the local project |
| folds / discovery splits | missing | generated once with the Phase 2 protocol | upload `outputs/benchmark/folds/` |
| a **partial** fold or split folder | `partial` / `incompatible` | stage fails; existing files are never overwritten | restore the missing files, or move the folder aside on Drive to regenerate |
| feature caches (`.npy`) | missing / invalid / wrong shape | only the missing features are computed; invalid local files are quarantined | upload `outputs/benchmark/cache/features`, `outputs/phase4/cache/features` |
| a feature cached under another definition (spec hash differs) | `incompatible` | none (never overwritten) | restore the previous feature definition, or move the cache file aside |
| baseline run folders (OOF, discovery maps, fold models) | `NOT_AVAILABLE` / `CV_ONLY` / `INTERRUPTED` | retrained only when listed in `train_baselines` (from `configs/baselines/legacy_specs/`; labelled `historical_equivalence: false`) | upload the legacy run folders (see below) |
| regenerated baseline on other folds | `INCOMPATIBLE` | never reused | move the folder aside and retrain under the current folds |
| `P4_B_gated` `config.json` (base of every `cnn5` experiment) | absent | `base_spec_dict` falls back to the committed snapshot automatically | – |
| Phase 5 experiment state | interrupted | `Experiment` resumes at the next unit and from the last checkpoint | – |
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
pipe.status()                 # state + reason + recovery per stage
pipe.record('feature_store')  # full record: error, traceback tail, fingerprint, outputs, code commit
pipe.events(50)               # transition log (environment/pipeline/events.jsonl on Drive)
```

## Things the pipeline never does

It never deletes Drive files, overwrites valid or historical artifacts, regenerates existing folds, fabricates
scores, maps, or checkpoints, uploads submissions, or stores URLs or tokens. Moving a conflicting artifact aside
is always a manual step.
