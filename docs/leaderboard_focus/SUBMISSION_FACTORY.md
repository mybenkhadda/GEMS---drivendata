# GEMS submission factory

One Colab notebook turns the frozen B0 / B1 / W2 artifacts into validated, read-back competition GeoTIFFs and tells
you which file to upload. Nothing is trained, re-inferred, tuned or uploaded.

* Notebook: `notebooks/leaderboard_focus/GEMS_SUBMISSION_FACTORY.ipynb`
* Code: `gems_bench/leaderboard_focus/factory.py` (`Factory`), `factory_report.py`, `factory_selftest.py`
* Outputs: `MyDrive/GEMS_2/outputs/leaderboard_focus/submission_factory/` (local mirror
  `/content/gems_work/outputs/leaderboard_focus/submission_factory/`)

## How to run

1. Colab -> open the notebook from GitHub -> Runtime type **CPU** (High-RAM recommended).
2. Edit only the CONFIG cell (defaults produce files when the frozen artifacts pass the checks). Pin `ref` to the
   pushed commit for a reproducible run.
3. Run all cells top to bottom (or `RUN_ALL()` after the environment cell).
4. After a disconnect: restart the session and run everything again. Finished candidates print `SKIP_CACHED`.

## What happens

| step | reused code | output |
|---|---|---|
| self-tests | synthetic arrays through the real writer / validator / stitcher / ranking | `diagnostics/selftests.csv` |
| [1/6] preflight | `preflight.run`, `artifacts.run_environment`, `full_benchmark.read_csv_checked` | `diagnostics/preflight.{csv,json}`, `artifact_status.csv`, `reports/PREFLIGHT.md` |
| [2/6] frozen models | `M1Session.attach`, `champion.recover` (nested single-policy table of every candidate), `final_submission.inspect_spec`, `parity.fold_assignment / zone_overlaps` | `selection/`, `diagnostics/source_compatibility.csv` |
| [3/6] recovery | `champion.recipe_spec` + `M1Session.fp` (same recipe id as M1: `0a8692b2ab` for the champion if no source changed) | `manifests/candidates/<id>.json` |
| official evaluation | `validation.recompute` (official scorer; components vs the frozen grid) | `metrics/` |
| [4/6] technical validation | `final_submission.build_map` (zone-stitched), `validation.budget_rows`, `recipe_consistency`, `array_sha` | per-candidate checks |
| [5/6] GeoTIFF | `final_submission.recipe / write_file` (never overwrites another recipe) | `submissions/submission_<candidate>_<recipe_id>.tif` + sidecar |
| [6/6] read-back | `final_submission.validate` (gems3 validate + `audit_export`) + `factory.readback_checks` on the file on disk | `submissions/<stem>_validation.json` |
| ranking / reports | `factory.rank`, `factory_report.write_all` | `submissions/submission_manifest.*`, `submission_sha256.txt`, `submission_ranking.csv`, `README_UPLOAD.md`, `figures/`, `reports/` |

## Candidates

| id | source (`configs/p6/full_gpu.yaml`) | builder | depends on |
|---|---|---|---|
| `w2_champion` | `a_w2_rep_frac` | stack (W2 held-out maps) | shared + W2 |
| `w2_crest` | `a_w2_crest` | stack | shared + W2 |
| `w2_skeleton` | `a_w2_skeleton` | stack | shared + W2 |
| `b0_b1_fallback` | `ref_m9_fallback` | fusion, nested per-zone weights as evaluated | shared only |

Every number comes from the same nested single-policy protocol (one cell per outer zone chosen on the discovery labels
of the other zones, applied to both tasks; same folds, eligible area and scorer). Recorded historical values are kept
in `metrics/comparative_metrics.csv` with their protocol; the `separate`-protocol geographic values of crest
(0.185609) and skeleton (0.168841) are flagged as not comparable. Identical rasters are recorded as `DUPLICATE` and get
no file.

## Verdicts

| field | meaning | blocks upload |
|---|---|---|
| `TECHNICAL_VALIDATION` | frozen components reproduced, recipe consistent, fold geometry, coverage, budget, mask | yes |
| `EXPORT_PARITY` | deployed zone score == evaluated score (Spearman >= 0.999) and read-back band == validated array | yes |
| `OFFICIAL_FORMAT` | read-back file: 1 band float32, EPSG / resolution / shape / transform of the template, NaN nodata, values in [0, 1], size | yes |
| `GEOGRAPHIC_TARGET_MET` | single-policy geo DTI >= 0.1758 (internal research target) | only in `strict` mode |
| `READY_FOR_UPLOAD` | all hard checks passed on the file read back from disk | - |
| `RECOMMENDATION_RANK` | predeclared offline ordering of READY files | - |

Statuses: `READY_FOR_UPLOAD`, `VALIDATED_NOT_WRITTEN` (DIAGNOSTIC mode / file cap), `HELD_BY_STRICT_TARGET`,
`DUPLICATE`, `INVALID`, `BLOCKED`, `UNAVAILABLE`, `FAILED`. Errors: `diagnostics/runtime_errors.{csv,json}`.

## Which file to upload

Open `submissions/README_UPLOAD.md`. `PRIMARY` = rank 1 of the predeclared rule (p6g-eligible first, then discovery
DTI, then worst discovery zone). At most one `ALTERNATIVE`: the READY file with a higher single-policy geographic DTI
than the primary (its public score shows whether the leaderboard rewards geographic robustness or discovery strength).
Check `sha256sum <file>` against `submission_sha256.txt`, upload by hand (3 scored submissions per rolling 7 days),
then record the score in the last notebook cell (`F.record_result`). Offline DTIs are not leaderboard scores; public
scores never change a selection.

## Boundaries

* Mapped faults are used only as the official eligibility mask (no positive pixel on a mapped fault); no held-out
  label decides which pixels score higher.
* The fusion weights of `b0_b1_fallback` are the frozen nested per-zone M9 weights the benchmark used; nothing is
  re-tuned.
* The geographic DTI is the nested held-out estimate of the deployed policy; the stitched file itself is not re-scored
  against geographic labels (the geo-unit evaluation counts mapped faults as truth, the file masks them).
