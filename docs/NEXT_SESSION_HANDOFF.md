# Next-session handoff (2026-10-25: M6 model selection on existing FULL runs)

Code authored only. **Nothing was executed**: no notebook, script, test, training, raster processing or download.
No M6 number exists yet. The scoreboard quoted in `docs/M6_RESEARCH_PLAN.md` comes from the previous notebook
review; it lives in `configs/m6/model_selection.yaml` under `reference_observations` and is used only for
`reference_check.csv`. Branch: `m6-model-selection` (from `origin/main` 8bab38b).

## What M6 is

M6 is section **18b** of the master notebook (`pipe.m6(M6_CONFIG).run()`). It is CPU-only, reads the saved FULL
predictions of `b0_reference`, `b1_bg_weight`, `b2_nnpu_p010` and `c1_geometry_heads` (model names
`P5_B0_gated`, `P5_B1_bgw`, `P5_B2_nnpu_p010`, `P5_C1_heads`), and writes only to `reports/m6/` (pushed to Drive)
and `cache/m6/` (local `.npy`).

| unit | module | output tables (reports/m6/tables unless noted) |
|---|---|---|
| audit (always re-run) | `evaluation/comparative.py` | `../artifact_inventory.csv`, `../provenance_audit.csv`, `../model_compatibility.csv`, `../missing_artifacts.csv` |
| scorecard | `scoreboard.evaluate_models` (reused) | `processed/scorecard/*` |
| paired | `comparative.py` | `model_comparison`, `fold_metrics`, `discovery_metrics`, `paired_deltas`, `paired_summary`, `spatial_bootstrap`, `reference_check`, `persisted_scores` |
| calibration | `evaluation/calibration.py` (+ `p3.transforms`) | `calibration_results`, `calibration_per_fold`, `calibration_invariance`, `threshold_transfer`, `score_distribution` |
| ensembles | `discovery/nested_selection.py` (+ `ensemble5` conventions) | `ensemble_candidates`, `ensemble_selection_*`, `ensemble_index_audit`, `ensemble_bootstrap` |
| complementarity | `evaluation/complementarity.py`, `evaluation/geological_errors.py` | `model_complementarity`, `fault_system_analysis`, `fault_system_summary`, `discovery_system_classes`, `fp_distance_profile`, `error_attribution`, `case_study_windows` |
| postprocessing | `postprocess/topology.py` | `postprocessing_ablation`, `postprocessing_selection_audit` |
| gate | `evaluation/selection_gate.py` (wraps section-17 `promotion_decision`) | `promotion_decisions`, `manifests/recommendation.json` |
| report | `pipeline/m6_report.py` | `M6_RESEARCH_REPORT.md` |

Orchestration: `gems_bench/pipeline/m6.py` provides the config loader with unknown-key rejection,
`Workspace` (atomic CSV / Parquet / JSON / GeoTIFF, per-unit fingerprints and outputs), `PredictionCache`
(memory-mapped `.npy`), `M6Analysis` and `pipe.m6()`.

## Decisions made this session

1. The ensemble selector lives in `gems_bench/discovery/nested_selection.py` (next to `ensemble5.py`). A
   `gems_bench/ensemble/` package would shadow the existing `gems_bench/ensemble.py` module.
2. The only core change: `P4Context.discovery_pred` / `has_discovery` first check `_maps["_disc:<model>:<d>"]`,
   so memory-mapped and derived discovery maps are served without re-reading GeoTIFFs. Behaviour is unchanged
   when the key is absent.
3. `M6_CONFIG` has its own schema (`configs/m6/model_selection.yaml`) instead of extending `PipelineConfig`.
4. Prespecified primary candidate: B0+B1 nested weighted rank with `geo_masked` discovery weights. The final
   recommendation rule is "highest primary discovery DTI among candidates that PROMOTE against both B0 and B1",
   otherwise "no candidate passes".
5. A leakage subtlety was fixed in design: a fold's coarse weight choice is restricted to the coarse grid, because
   weights added by another fold's refinement were centred using this fold's labels.
6. The audit is strictly read-only: `RegistryView` reads `registry.csv` (`ModelRegistry` would create folders).
7. Rasters: `save_rasters: selected` stores only the primary ensemble (OOF + 3 discovery maps), the area-budget
   masks of B0 / B1 and the best postprocessed masks. Other candidates are reproducible from
   `manifests/ensemble_weights.json`.

## Run next

1. Locally: `pytest tests/test_m6.py`, then the earlier suites (`tests/test_fidelity.py`,
   `test_pipeline_engine.py`, `test_pipeline_recovery.py`, `test_phase5_infrastructure.py`,
   `test_phase5_science.py`). None of these has ever been run.
2. Colab (a CPU runtime is enough; high-RAM recommended): open the notebook from `m6-model-selection` (or `main`
   after merge), run cells 1–4 and sections 7–12 (cached), then section 18b. Expect the audit to show B0, B2, C1
   as COMPLETE. **B1 is the open question:** the 88c97ed log showed B1 failing on `B0_lineament`, but the
   reviewed scoreboard has B1 numbers, so a FULL B1 run must exist on Drive. If the audit says MISSING / PARTIAL,
   the error names the gap.
3. Read `reports/m6/M6_RESEARCH_REPORT.md`, starting with "Central question" and `reference_check`.

## Unverified at runtime (everything)

* Every module above. Specific risks:
  * `fold_metrics` assumes the `frac_fold<k>` columns of `budget.crossfit_area`.
  * `discovery_metrics` assumes the `disc_per_split` columns produced by `evaluate_models`.
  * CSV round-trip of the scorecard (dtypes of `zone`, `target_frac`).
* Runtime estimates (not measured):
  * consolidated scorecard: several minutes;
  * ensemble curves: 2 normalisations × 5–13 weights × 14 budgets × 3 folds;
  * postprocessing: 14 operations × 3 budgets × 3 folds, plus about 30 full-grid EDTs per model for short recall;
  * isotonic calibration: 2 M-pixel fits.
* `link_endpoints` / `directional_close` speed on full fold crops; `segment_attributes` band gradient on the full raster.
* `fig_case_study` windowed reads of `det_elev` and the auto-picked geophysics band (`mag` / `grav` / `bouguer` name
  match; set `complementarity.contrast_bands` if none matches).
* Parquet export of object columns (`_parquet_safe` stringifies; failures only print).
* Drive push of `reports/m6` after each unit (`pc.push`).

## Known limitations / risks (unchanged from before, still open)

* `b0_reference`'s own control `P4_B_gated` is absent from Drive. Irrelevant for M6, which compares B0 and B1
  directly.
* Rebuilt EDA ≠ historical EDA; regenerated folds vs local folds; signed URLs in history (expired).
* G1 / G3 baselines are still `NOT_AVAILABLE` unless retrained (`train_baselines=['G1','G3']`, CPU). M6
  inventories them and includes them only if COMPLETE.
* `SubmissionRunner` supports only `cnn5` / LightGBM champions. An M6 ensemble champion would need an ensemble
  submission path: fold models of both members plus the recorded weights, applied with the all-fold weight rule.
  It is not implemented.
