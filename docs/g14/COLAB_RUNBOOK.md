# G14 Colab runbook

Notebook: `notebooks/colab/GEMS_G14_GEOFAULTFORMER_MASTER.ipynb` (clones ref `g14-geofaultformer`).
**Never executed.** The first run is the first test. Expect to fix things, and run the stages in this order.

## 0. Safety defaults

* `run_mode = "AUDIT_ONLY"`.
* `train_enabled`, `inference_enabled` and `submission_enabled` are all False.
* `TRAIN_FINAL_UNIT`, `RUN_GRAPH` and `RUN_ENSEMBLE` are False, and `CONFIRM = ''`.
* Every heavy cell checks the mode and its flag, then prints `SKIPPED (...)` otherwise.

Change settings only in the `OVERRIDES` dict of cell 2.

Run modes, in increasing order:

1. `AUDIT_ONLY`
2. `VALIDATE_DATA`
3. `TRAIN`
4. `EVALUATE`
5. `SUBMIT`

## 1. Runtime

* Use a GPU runtime (T4 works; A100 / L4 is better for T2+).
* Mount Drive when asked. Data is found under `drive_root` (`/content/drive/MyDrive/GEMS_2`).
* The setup cell clones the branch, pip-installs missing packages and puts the repo on `sys.path`.
* Before anything else, run the unexecuted tests once in a scratch cell and fix any failure:
  `!cd /content/GEMS && python -m pytest -q tests/test_g14_core.py tests/test_g14_model.py`.

## 2. Data validation (`VALIDATE_DATA`)

```python
OVERRIDES = {"run_mode": "VALIDATE_DATA"}
```

* Cells 6–9 do the following:
  * inventory existing artifacts;
  * build the channel stack (`shared/stack`, about 3 levels × C channels, float16);
  * run `check_contract`;
  * build the units and write `folds/` (`fold_audit.csv`, `system_crossing.csv`, sysmap);
  * render dataset previews.
* G13 features are not on Drive. Choose one of these:
  * set `data.build_g13_features_if_missing=True` (slow, rebuilds the G13 store);
  * use `data.g13_features_dir` pointing to an existing store;
  * accept raw + G14 + known channels only (see the `A_feat_raw_only` ablation).
* **Stop if:** any contract row is FAIL, `assert_no_leakage` raises, or `permitted_known_in_heldout > 0`.

## 3. Baseline reproduction (T0)

* Use `split.backend = "phase5"` for comparison with W2 (needs the Phase-5 context and the W2 maps from the P6 outputs).
* Use `split.backend = "g13"` for the G13 top-3 models.
* Set:

  ```python
  OVERRIDES = {"run_mode": "EVALUATE", "inference_enabled": True, "experiment": "T0_champion"}
  ```

* Check that the reference scores land near their documented values: W2 disc ≈ 0.24 (phase5), and G13 F_fixed
  disc ≈ 0.085 / geo ≈ 0.154 (g13). A large gap means the evaluation wiring is wrong. Fix it before training anything.

## 4. Ladder T1–T5 (`TRAIN` then `EVALUATE`)

```python
OVERRIDES = {"run_mode": "TRAIN", "train_enabled": True, "inference_enabled": True, "experiment": "T1_local_cnn"}
```

* Cell 12 trains one model per CV unit: 3 geo folds and 3 disc splits. Checkpoints are resumable. The wall-clock
  budget is `train.max_minutes_per_unit` (default 60).
* Cell 13 writes the held-out maps. The evaluation cells write `scorecard.json` and the bootstrap.
* Then repeat with `T2_regional`, `T3_global`, `T4_fusion`, `T5_topology`. Each experiment has its own directory,
  `experiments/<name>__<hash>`.
* Promote a step only if its paired bootstrap CI against the previous rung excludes 0 on at least one task and is
  not worse on the other.
* Rough costs are guesses, not measurements: T1 is perhaps 0.5–1.5 h per unit on a T4, and T2/T3 2–3×.

## 5. T6 graph recovery and T7 ensemble

* `RUN_GRAPH = True` (cell 17): the rules come first, and the classifier is optional. Keep it only if the
  cross-fitted DTI ≥ T5.
* `RUN_ENSEMBLE = True` (cell 18): members are T-maps on the same backend. Weights and gates for zone z use other
  zones only.

## 6. Final evaluation (g13 backend only)

* Choose the operating point from the cross-fitted CV results. Then run `evaluate_locked_test` **once**.
* The record is appended to `locked_test_log.jsonl`. Do not re-tune afterwards. A second look is labelled
  `SECOND_LOOK`.

## 7. Submission (manual upload only)

```python
OVERRIDES = {"run_mode": "SUBMIT", "inference_enabled": True, "submission_enabled": True}
TRAIN_FINAL_UNIT = True      # trains on all permitted geography (excludes nothing at test time)
CONFIRM = 'WRITE_SUBMISSION'
```

* Writes `submissions/<name>.tif` plus validation JSON and provenance.
* The notebook never calls the competition API. Upload the file by hand.

## 8. Sync

* Results are mirrored to `drive_root/outputs/G14_GEOFAULTFORMER`.
* Large memmaps (the stack, full maps) stay local unless `sync` is told otherwise.
