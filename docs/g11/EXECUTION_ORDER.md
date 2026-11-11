# Execution order (master notebook)

| § | section | call | gate |
|---|---|---|---|
| 00 | configuration | `CONFIG` | Validated in §03 by `notebook_config.build_plan`. |
| 01 | Drive mount | mount, GPU, RAM, disk | Drive root must exist. |
| 02 | repository checkout | `ls-remote`, fetch, detached checkout, pip | Stops if `ref` is missing, lacks the modules, or the checkout is dirty. |
| 03 | environment validation | synthetic tests (LEVEL 1), plan, runtime, `P.audit()`, `P.ingest()` | Tests must pass; audit fails fast. |
| 04 | existing feature recovery | `P.recover()` | Classifies only; never deletes. |
| 05 | missing-chunk processing | `P.process_features()` | Computes only unverified layers; DEM work is chunk-resumable. |
| 06 | feature dataset finalization | `P.finalize_features()` | Every expected layer must be verified; `COMPLETE.json` is written after validation. |
| 07 | feature integrity tests | `P.validate_features()` | QUICK (FULL under RESUME_MISSING). |
| 08 | EDA | `P.eda()` | Optional; saved features only. |
| 09 | geographic split reconstruction | `P.splits()` (v2) | Raises if no candidate passes every gate. |
| 10 | split integrity tests | gates + leakage report | Required. |
| 11 | training dataset preparation | `P.training_data(exp)` | Persisted indices, scaler, class weights. |
| 12 | XGBoost | `P.train(["E0", "E1"])` | Required. |
| 13 | SegFormer | `P.train(["E2"])` | Optional. |
| 14 | multiresolution fusion | `P.train(["E3"])` | Optional. |
| 15 | segment ranker (E4) | `predict_all`, `fusion`, `evaluate(validation)`, `infer` | Selection is written before any test metric. |
| 16 | full-grid inference | `P.predict_all()` | Missing maps are regenerated from the saved models. |
| 17 | candidate recovery | budgets and ranker tables | |
| 18 | offline evaluation | `P.evaluate(final=True)` | Refused without `selection.json`. |
| 19 | submission | `P.submit()` | Template validation; no upload. |
| 20 | report | `P.report(PLAN, INFO)` | |

`StepRunner` failure policy:
- **Required** steps stop the notebook: audit, ingest, recover, process / finalize / validate features, splits, validate splits, training data, XGBoost, inference, evaluate, submit.
- **Optional** steps are recorded as FAILED and skipped.
- `STOP_ON_ANY_FAILURE` stops on everything.

## Run modes

| mode | steps |
|---|---|
| FULL_RESUME | all |
| RECOVER_ONLY | audit, recover |
| PROCESS_FEATURES_ONLY | audit, ingest, recover, process |
| FINALIZE_FEATURES_ONLY | audit, ingest, recover, finalize, validate |
| SPLITS_ONLY | audit, validate features, splits, split gates |
| TRAIN_ONLY | audit, validate features, splits, gates, training data, all trainers |
| INFER_ONLY | audit, validate features, splits, gates, inference, ranker, candidates |
| AUDIT_ONLY / INGEST_ONLY / EDA_ONLY / EVALUATE_ONLY / SUBMISSION_ONLY / SMOKE_TEST | as named |

The training, inference and split modes never call feature extraction.
