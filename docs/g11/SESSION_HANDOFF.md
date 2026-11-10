# G11 session handoff

## 2026-11-10 (b): Colab + Drive master execution system
- **Added:**
  - the master notebook `notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb` (sections 00–20);
  - modules `colab_runtime.py`, `notebook_config.py` (CONFIG adapter, plan, `StepRunner`), `dem_units.py`, `external_predictions.py`, `smoke.py`, `fusion.py`, `reports.py`, `eda/model_evidence.py`;
  - tests `test_colab_infra.py` and `test_real_data.py`.
- **Changed:**
  - output layout `00_runtime … 11_reports`;
  - stage records now carry start/completion time, FAILED status and validation results;
  - FeatureStore uses family sub-folders plus `localize()`;
  - DEM catalog spans several folders, de-duplicates tiles and audits units;
  - smoke grid cropping and grid-aware reads;
  - ingestion and EDA deliverables from the brief;
  - training ablations, per-family model folders, `07_predictions/`;
  - fusion stage E5;
  - leaderboard with the required columns;
  - submission names and the `official_DTI` status;
  - local DEM mosaic for E4.
- **Real Drive layout read:** via a read-only Drive connector, metadata only. See [DATA_PATHS.md](DATA_PATHS.md). The G07 arrays' grid comes from their `manifest.json`.
- **Executed:** the local synthetic suite only (102 tests, 97 passed, 5 real-data tests skipped).
- **Not executed:** the real-data smoke test, which needs the Drive mount in Colab. No real result exists.
- **Not committed, not pushed.** The notebook defaults to `ref=g11-colab-production`, which does not exist on GitHub yet; section 02 stops with the deployment commands until it is pushed.

## Next step
1. Push the branch, using the commands printed by notebook §02.
2. Run the master notebook with `run_mode="SMOKE_TEST"` on a GPU runtime.
3. Read `01_audit/audit_report.md`: DEM unit classes, G07 compatibility, smoke window.
4. Read `02_ingestion/coverage_report.json`: fault coverage lost to unit exclusions.
5. Only then run `FULL`.

## Decisions (check them)
- DEM tiles of unknown or conflicting vertical unit are excluded. USGS 3DEP NAVD88 tiles should classify as VERIFIED_METRES through the compound CRS; this is unverified on the real tiles.
- G07, M9 and P6 maps are EDA-only, never fusion members or targets, because their splits differ from the G11 split.
- Model fusion (E5) uses only G11 maps: fitted on stack, selected on validation.
- E4 uses a local 25 m mosaic.
- The SegFormer loss ablation and gated-fusion mode ablation are on by default (3 extra variants each). Set `training.loss_ablation: []` and `models.gated_fusion.mode_ablation: []` to save time.
- In SMOKE_TEST, `candidates` and `submit` are not in the plan; the full chain runs in FULL.

## Open risks
- Drive throughput for the roughly 21 GB of DEM.
- Feature store size, about 12–17 GB (estimate).
- G07 grid nesting: if the 243350 / 4508550 origin does not nest in the master grid, the arrays are excluded, with the reason in the compatibility CSV.
- PU labels: unlabeled pixels are background, not negatives.
- No official evaluator.
