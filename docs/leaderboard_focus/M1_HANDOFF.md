# M1 handoff

**State (2026-10-28):** authored, NOTHING executed. No Python, test, notebook, Drive access or upload ran in the
authoring session. No M1 number exists. The tests are written but were never run.

**Fix (2026-10-28, first Colab run):** notebook 00 was BLOCKED by `b0_maps` / `b1_maps: COUNT_MISMATCH (7 files,
expected 4)`, because `training.cv` also writes `oof_fold<k>.tif`. The maps are now checked by `required_files`
(`oof.tif`, `disc0-2.tif`) with pinned `expected_id`; extras are accepted, hashed and listed in `extra_files`; several
matching experiment directories are AMBIGUOUS. Authored only; the new tests have not been run.

## Work completed

1. Read-only audit (`M1_AUDIT.md`), including the saved Colab outputs of the p6g run. That run:
   * passed its audit;
   * was interrupted during W1 seed training;
   * never reached benchmark, selection or submission.
2. A thin M1 layer over p6g in `mode: recovery`:
   * W2-family selection on the frozen boost-A grid;
   * independent recomputation and recorded-value reproduction;
   * historical parity diagnosis;
   * zone-stitched export with M1 gates;
   * a leaderboard log.

## Files (all new; no existing file modified)

* `gems_bench/leaderboard_focus/`:
  `__init__`, `config`, `artifacts`, `session`, `champion`, `validation`, `parity`, `submission`, `reporting`
* `configs/leaderboard_focus/m1.yaml`
* `notebooks/leaderboard_focus/`: `README.md` and notebooks `00`–`04`
* `docs/leaderboard_focus/`:
  `README`, `M1_AUDIT`, `ARCHITECTURE`, `M1_EXECUTION_GUIDE`, `ARTIFACT_CONTRACTS`, `VALIDATION_AND_PARITY`,
  `SUBMISSION_GUIDE`, `TROUBLESHOOTING`, `M1_HANDOFF`
* `tests/test_leaderboard_focus.py`

## Key decisions

* **Champion space.** The candidates are p6g names (`ref_p6_w2`, `a_w2_crest`, `a_w2_skeleton`, `a_w2_rep_frac`,
  `a_hybrid`, `rec_p6r_choice`, `ref_m9_fallback`) plus the historical cell (role historical, never selected). The
  rule is p6g `full_benchmark.choose` with objective `disc`, worst zone ≥ 0.153414 and the relative geo rule.
* **Geo floor.** 0.1758 is reported per candidate and does not block by default (`geo_floor_blocking`; this is the
  user's decision).
* **Construction.** Zone-stitched (`final_submission.build_map`). The p6g root is set to the M1 root, so no P6 /
  p6g directory is written.
* **Strictness.** Strict parity: a fusion-w WARNING blocks. No silent fallback to a reference when nothing is
  eligible.
* **File naming.** `submission_m1_<family>_<recipe_id>.tif`; a different recipe never overwrites an existing file.

## Known blockers and risks (unverified until Colab)

* **The geo gate question.** The single-policy W2 geo was about 0.106 in the P6.1 run. If you require 0.1758, no W2
  policy is likely to qualify.
* **Blend.** If the champion's cell has blend < 1, strict parity blocks it (fusion weight per zone vs all zones).
* **Recorded-value reproduction.** It assumes the Drive grids are the ones that produced the records. Any drift is
  BLOCKED.
* **First-run API mismatches:**
  * `Pipeline.from_notebook` with the reduced CONFIG;
  * `bst.w2_inputs()` / `bst.sel` when the boost selection files are missing;
  * the `EP.agreement` key names in the parity detail;
  * the sidecar fields of the historical submission.

## Artifact locations

* M1 outputs: `MyDrive/GEMS_2/outputs/leaderboard_focus/m1_champion_recovery/`, with sub-folders `environment`,
  `selection`, `validation`, `parity`, `figures`, `submissions`, `leaderboard`, `reports` and `manifests`.
* Read-only inputs: `outputs/p6_fast_track/{stack, boost/benchmark, recovery/export_parity}`, `experiments/b0_reference__*`,
  `experiments/b1_bg_weight__*`.

## How to run

Follow the commit / push steps in `M1_EXECUTION_GUIDE.md`. Then, in Colab:

1. Run `00` → `01` → `02` → `03`.
2. Upload by hand, only if the status is READY_FOR_UPLOAD.
3. Run `04` with `LEADERBOARD_ENTRY`.

## Milestone 1 is complete when (all of them)

1. `manifests/artifacts.json`, `selection.json`, `validation.json` and `parity.json` are COMPLETE.
2. `submissions/submission_manifest.json` → `status: READY_FOR_UPLOAD`, with `export_parity_report.json` →
   `equal: true`.
3. The manifests and reports exist on Drive.

The public score (notebook 04) is recorded afterwards.

## Milestone 2 (GeoSelect) entry criteria

* M1 is complete, and the champion recipe is frozen as the baseline (`selection/champion_recipe.yaml`).
* At least one public score is recorded for the M1 file. It is used for context only, never for selection.
* The M1 parity checks are reused as the acceptance gate for any GeoSelect export.

## Paste-ready summary for the next session

```text
GEMS M1 (leaderboard focus) AUTHORED 2026-10-28, nothing executed. Code gems_bench/leaderboard_focus/ (config, artifacts,
session, champion, validation, parity, submission, reporting), config configs/leaderboard_focus/m1.yaml, notebooks
notebooks/leaderboard_focus/00-04, docs docs/leaderboard_focus/ (start with M1_AUDIT.md), tests
tests/test_leaderboard_focus.py (never run). No existing file modified. M1 = p6g (mode recovery, root =
outputs/leaderboard_focus/m1_champion_recovery) restricted to W2 family + fusion fallback on the frozen boost-A grid;
export = final_submission.export_one zone-stitched + M1 gates (recomputed cells == grid, recorded values reproduced,
strict zone parity, written band sha == validated array). Last real run (300d99e): p6g audit OK, W1 seed training
interrupted, no benchmark/submission. Historical W2 export blocked by rank-domain mismatch + in-sample refit + mixed
policy (not used by M1). Open user decision: geo floor 0.1758 blocking (default no). Before claiming anything read
manifests/*.json on Drive. Do not start GeoSelect until M1 is READY_FOR_UPLOAD.
```
