# M6 results guide: where every result is and how to read it

Run section **18b** of `notebooks/colab/GEMS_MASTER_PIPELINE.ipynb`, or `pipe.m6(M6_CONFIG).run()`. Outputs are
written to `/content/gems_work/reports/m6/` and pushed to `/content/drive/MyDrive/GEMS_2/reports/m6/`. Every
figure has its plotting data next to it as `<figure>_data.csv`.

```text
reports/m6/
├── artifact_inventory.csv     provenance_audit.csv     model_compatibility.csv     missing_artifacts.csv
├── M6_RESEARCH_REPORT.md      (built only from the tables below)
├── manifests/   analysis_config.json, provenance.json, artifact_inventory.json, ensemble_weights.json,
│                ensemble_primary.json, recommendation.json, unit_log.csv, units/<unit>.json (fingerprint + outputs)
├── tables/      (CSV + Parquet)
├── processed/   scorecard/, scorecard_ensembles/, uncertainty/, calibration/, postprocessing/,
│                calibrated_predictions/, ensemble_predictions/<candidate>/{oof,disc<d>}.tif, postprocessed_predictions/
└── figures/     *.png + *_data.csv, geological_case_studies/<kind>_<source>_<id>.png
```

## Read in this order

1. **`artifact_inventory.csv`**: `status` per model (COMPLETE / PARTIAL / INCOMPATIBLE / MISSING / INVALID) and
   `reasons`. M6 stops with `M6ArtifactError` if an analysed model is not COMPLETE / PARTIAL or lacks a discovery
   map.
2. **`tables/reference_check.csv`**: the previous review's numbers vs persisted (`<experiment>/metrics/scoreboard.csv`)
   vs recomputed. `DIFFERENT` rows are findings to explain, not errors to hide.
3. **`tables/model_comparison.csv`** (`figures/model_comparison.png`): one protocol for every model.
   `geo_dti_area` is the cross-fitted geographic DTI, `disc_dti_excl` / `disc_dti_retained_fp` are the
   pseudo-discovery policies, and `short_recall_w` is < 2 km recall at the matched 5 % budget.
4. **`tables/fold_metrics.csv`, `discovery_metrics.csv`, `paired_summary.csv`, `spatial_bootstrap.csv`**:
   - `sign_pattern` `+++` means the candidate beats the reference on all three units.
   - `boot_q05..boot_q95` is the 90 % block-bootstrap interval of the pooled difference.
   - `p_delta_positive` is the bootstrap share of positive differences.
   - Figures: `geo_dti_per_fold`, `paired_dti_differences`, `discovery_dti_per_split`, `worst_fold_comparison`,
     `short_fault_recall_comparison`, `score_sensitivity`.
5. **`tables/calibration_results.csv`**: `dti_crossfit` per strategy.
   - `calibration_invariance.csv` should say `IDENTICAL`: binary masks of raw and rank scores coincide.
   - `threshold_transfer.csv` shows `area_ratio_heldout_vs_dev` far from 1 for an absolute cut: the score scale
     differs between regions, and the per-fold area fraction is the safer rule.
   - Figure: `area_budget_curves.png` (vertical lines are the cross-fitted budgets).
6. **`tables/ensemble_candidates.csv`**: members, method, normalisation, per-fold weights, all metrics,
   `delta_*_vs_<B0|B1>`, `weight_dispersion` (stability of the nested choice), `relative_inference_cost`,
   `extension_justified` (larger sets only).
   - `ensemble_index_audit.csv`: `heldout_labels_used_for_selection` must be all `False`.
   - Figures: `score_vs_cost.png`, `ensemble_weight_choices.png`.
7. **Complementarity.**
   - `model_complementarity.csv`: correlation, IoU, a-only / b-only / both area, far-field disagreement, per
     budget and fold.
   - `fault_system_analysis.csv` (one row per mapped segment) and `fault_system_summary.csv`: both / B0-only /
     B1-only / neither, by length, fold, strike sector, sinuosity, junction, edge distance and contrast tercile.
   - `discovery_system_classes.csv`, `fp_distance_profile.csv`, and `error_attribution.csv` (descriptive
     hypotheses about missed faults).
   - Figures: `model_disagreement.png`, `short_fault_analysis.png`, `geological_case_studies/`.
8. **`tables/postprocessing_ablation.csv`**: `delta_vs_identity` (nested DTI change) next to
   `delta_short_recall_w`, `delta_fragmentation_per_100km`, `delta_n_junctions` and `delta_far_share`.
9. **`tables/promotion_decisions.csv`** (`control == "ALL"` = overall verdict) and
   **`manifests/recommendation.json`**.

## Reproducing a case-study panel

`tables/case_study_windows.csv` holds `kind, source (oof | disc<d>), id, r0, c0, size, frac, model_a, model_b,
ensemble`. To redraw one panel:

```python
from gems_bench.evaluation.complementarity import fig_case_study
w = m6.ws.load_table('case_study_windows').iloc[0].to_dict()
fig_case_study(m6.ctx, w, {m: m6.ctx.oof(m) for m in m6.pair}, context_band='det_elev')
```

## Resuming and recomputing

Each unit (`scorecard`, `paired`, `calibration`, `ensembles`, `complementarity`, `postprocessing`, `gate`) records
the fingerprint of its inputs, its configuration and the M6 version. A matching unit whose outputs exist is
reused. Use `overwrite_existing=True` to recompute everything. Changing one config block recomputes that unit and
the units downstream of it. The audit always re-runs, because it is read-only and cheap.

## What not to conclude

- A local DTI is not a leaderboard score.
- Agreement far from mapped faults is an unmapped prediction, not a discovered fault.
- Three folds and three splits do not support population-level significance claims.
