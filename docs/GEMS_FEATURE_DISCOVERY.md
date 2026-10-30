# GEMS geological feature discovery & fault association (gfd.1-r1)

**State: authored, NOT executed** (no notebook cell, test or data job was run). No result exists yet.

## Files

| path | role |
|---|---|
| `notebooks/colab/GEMS_GEOLOGICAL_FEATURE_DISCOVERY.ipynb` | orchestration, sections 00-13 |
| `gems_eda/feature_discovery/__init__.py` | version (`FD_CODE_REV = "gfd.1-r1"`), lazy `FeatureDiscoveryExperiment` |
| `gems_eda/feature_discovery/catalog.py` | hypothesis catalog (R01-R21 from the Layer Lab program, R25-R29 magnetic, R31-R42 extra, `neg_rtp`) |
| `gems_eda/feature_discovery/metrics.py` | full-grid vectorised metrics: both-direction association, recovery (exact / 300 m / component / branch / localisation index), unique components, sorted-block regional, distance response + class, sorted-strata matched controls, fold masks, held-out with frozen polarity, PROMOTE / EXPERIMENTAL / REDUNDANT / REJECT rules |
| `gems_eda/feature_discovery/experiment.py` | `FeatureDiscoveryExperiment`: load / validate, formulas, layers, sections, figures, checkpoints, export, Layer Lab publishing |
| `gems_eda/feature_discovery/report.py` | synthesis written only from saved tables |
| `tests/test_feature_discovery.py` | synthetic unit tests + end-to-end FULL run on the Layer Lab demo raster |
| `streamlit/gems_layer_lab/lab/ops.py` | + `tilt_angle` (only change to the Layer Lab) |

Reuse: the formula language, ops, Engine (windowed reads, halos, content-hash cache, leakage guard), block bootstrap,
fault-normal profiles and complementarity provenance come from `streamlit/gems_layer_lab/lab`; held-out regions from
`outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/fold_map.tif` (band 1 fold id, bands 2-4 roles per fold:
0 train, 1 val, 2 buffer, 3 inner_val, 4 inner_buffer; dev = 0 / 3, eval = 1); canonical DTI from
`gems_bench.metric.dti_components` (optional diagnostic).

## Run in Colab

1. Push the repository; set `REF` in the config cell to the commit SHA.
2. Run 00 (mount, clone, install, build `E`), optionally the test cell, then 01 → 13 in order.
3. `mode='FAST'` + `fast_window` for a quick pass; `mode='FULL'` for final measurements (whole native grid, ~5–8 GB RAM;
   features are evaluated in batches of 10 and persisted as NPY).

Custom formulas (section 04):

```python
E.register_formulas('''
mag_edge = gradmag(rtp, sigma=3)
mag_texture = local_std(rtp, 5)
my_mag_ratio = ratio(mag_edge, mag_texture, 0.05)
my_terrain_mag = normdiff(gradmag(det_elev_slope, 3), mag_edge, 0.05)
''')
E.compute_features(); E.plot_feature("my_mag_ratio"); E.evaluate_against_faults("my_mag_ratio")
E.compare_features(["rtp", "R25_magnetic_edge", "R26_magnetic_residual", "my_mag_ratio"])
```

## Outputs — `GEMS_2/outputs/geological_feature_discovery/`

`configs/config.json` · `formulas/` (catalog_used.py, custom_*.json, frozen_catalog.py + frozen_statistics.json,
layer_index.json, feature_registry.json) · `layers/full/<name>.npy` + `.json` provenance (formula, hash, parameters,
source bands, CRS, transform, window, units, nodata, AOI dependence, reload line) · `layers/geotiff/` (exports) ·
`tables/`: raw_fault_association, **global_fault_association, feature_polarity_comparison, fault_distance_responses,
fault_response_classes, regional_fault_association, feature_scale_comparison, feature_rank_by_region,
region_vs_neg_rtp, fault_recovery_by_budget, unique_fault_recovery, feature_complementarity, w2_missed_fault_signatures,
heldout_validation, heldout_selected_on_dev, heldout_combinations, canonical_dti_diagnostic, feature_recommendations**
(CSV + Parquet) · `datasets/` (fault_normal_profiles, regional_maps.npz, unique recovery map) · `figures/*.png|svg` +
`figures/data/*.csv` · `checkpoints/<section>.json` · `manifests/` (data_report, run_manifest) ·
`reports/FEATURE_DISCOVERY_SYNTHESIS.md`.

Reload: run 00 + 01; sections with a matching checkpoint are reused; `E.load_table(name)`, `E.get(feature)`.

## Scientific rules implemented

Both polarities reported in exploration; held-out polarity chosen on the fold's training side and frozen; budgets on
the full valid population of the evaluated region; spatial-block bootstrap (2 km) CIs; regional cells below
`min_pos` / `min_neg` are INSUFFICIENT_DATA (NaN, never 0); distance / fault-normal diagnostics are retrospective;
combinations tested only after single features, pair chosen on dev; unlabelled ≠ false fault; no official DTI
except the optional canonical-scorer diagnostic on thresholded maps.

## Integration with P6 / W2 / M9

PROMOTE features are exported as native GeoTIFFs (`layers/geotiff/`) and published into the Layer Lab registry
(`outputs/geological_layer_lab/registry`) with frozen normalisation, so P6 / M9 can call
`lab.integration.FeatureStore(root).compute(name)` or read the GeoTIFF, then run the official DTI benchmark at equal
area budget on the existing held-out zones.

## Prior hypothesis evidence (not a result of this notebook)

A Layer Lab session on ONE 12 km AOI (14 361 px, 290 fault px; `labs/2026-10-30T00-05_export.csv`) ranked `-rtp`,
inverse terrain texture (R08 / R18) and R19 / R20 highest. This motivated the catalog; it is a single-AOI exploratory
observation with unvalidated direction choice.

## Known risks

Never executed; FULL-mode RAM / runtime unmeasured (5 M valid px x ~55 features; unique / complementarity pairs are the
slowest step); branch metrics depend on the `gems_eda.morphology` tracer working on a lightweight grid shim; the band
names of the real raster must match the catalog (`tmi_vg`, `iso_grav_anom_vg`, ...), otherwise section 03 reports the
unknown name.
