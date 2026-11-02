# G06 dataset contracts

All datasets live under `<run>/datasets/`; features and labels are separate files joined by ids; every model applies
an explicit column allowlist and `contracts.assert_label_free_columns` (refuses `y_*`, `truth`, `target`,
`label_conf`, `withheld`, `system_id`, `recovered`, `near_share`, `cls`, `weight`, `inner_fold`, ...). Manifests:
`datasets/dataset_manifests/<kind>.json` (schema version `g06.datasets.1`, feature-schema hash, fingerprint,
provenance). `load_dataset(..., expected_schema_hash=...)` refuses a dataset built on another feature schema.

## D1 pixel

| file | columns | label scope |
|---|---|---|
| `pixels/pool_meta.parquet` | `row_id, row, col, x, y, stratum, in_label, in_evidence, in_background, block, incl_prob` | stratum membership only (the label stratum is drawn from ALL mapped faults) |
| `pixels/pool_features.parquet` | `row_id` + every store channel except `phys_*` | label-free |
| `pixels/labels/<fit>.parquet` | `row_id, fit_id, stratum, cls, target, weight, label_conf, system_id, system_px, dist_km, target_dist_km, inner_fold, inner_buffer` | VISIBLE labels of the fit only |

`cls`: 1 POSITIVE, 0 UNLABELED, 2 LIKELY_BACKGROUND (IGNORE rows are not stored). Rows of the label stratum on a
hidden fault are dropped; a hidden-fault pixel kept through the background / evidence strata is an ordinary
unlabelled row with that stratum. `system_id` is the system of a POSITIVE (visible) row, -1 otherwise; `target` is
1 for POSITIVE, else 0. `pixels/label_summary.csv`: class counts / weights per fit.

## D2 patch

`patches/patch_index.parquet`: `patch_id, r0, c0, size, footprint_share, fold, eval_share_<d>` (label-free).
Arrays: `patches/stack_<hash>.npy` memmap (C, H, W) of `patch.channels`, written once by
`models_patch.create_patch_dataset`; targets are built per fit from the visible labels at training time.

## D3 segment

| file | content | scope |
|---|---|---|
| `segments/units/<unit>.parquet` | `unit, scenario, zone, label, row, col, r_full, c_full, seg_id` + `seg_*, on_*, geo_*, ori_*, kn_*, ev_*, unc_*` | label-free; `kn_*` = permitted known context of the unit; `ev_*` = honest maps of the unit's base |
| `segments/units/<unit>_labels.parquet` | `unit, label, near_share, cls, target, label_conf, system_id` | EVALUATION SIDE (the unit's own truth) - training target for OTHER units only |
| `segments/units/<unit>_lab.npz` | `lab` (crop int32 label raster), `window` | label-free |
| `segments/nested_known/<sc>_<outer>__<zone>.parquet` | `label, kn_*` recomputed without the outer unit's truth | strict cross-fitting |

Candidate pools (label-free): top 2 % of honest W2, top 2 % of the CNN B0 / B1 fusion, top 2 % of a physics
edge feature (`segment_candidates.pools`), deduplicated (`p6.segment_recovery.dedup_inventory`), split to <= 60 px,
>= 5 px. `cls`: POSITIVE (near share >= 0.5 within 3 px), NEGATIVE (< 0.1), AMBIGUOUS (excluded, `target` NaN).

## D4 connection

`connections/units/<unit>.parquet`: `unit, label_a, label_b, ra, ca, rb, cb, gap_km, con_angle_agree,
con_gap_align_a / _b, con_len_a_km / _b_km, con_gap_<feature>, ev_gap_w2_rank` (label-free).
`connections/units/<unit>_labels.parquet`: `target` (1 = both fragments POSITIVE on the same truth system and the gap
>= 50 % near truth; 0 = a NEGATIVE fragment or two different systems; NaN = ambiguous), `same_system`,
`gap_near_share` - evaluation side.

## Model inputs never allowed

* any truth-derived column (above);
* for pixel / patch models: any known-fault context (`known_*`, `tip_*`) - the target IS the visible faults;
* for S1: any W2 / CNN column (`ev_w2_*`, `ev_cnn_*`, `on_w2`, `on_cnn`, `unc_w2_rank_std`);
* OOF and full-fit predictions are never mixed: prediction files carry `fit_id`, unit and fingerprint in their path
  and json sidecar; FULL-fit rasters are research only.
