# Leaderboard focus: Milestone 1 notebook

**Use `GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb`.** It is the single canonical Milestone-1 notebook (code rev `m1.2-r1`):
one configuration cell, one `Runner` (`gems_bench/leaderboard_focus/workflow.py`), one cell per stage.

| stage | cell | runtime | gate to continue |
|---|---|---|---|
| 0 | configuration + bootstrap (Drive, strict clone, version lock) | CPU | – |
| 1 | artifact inventory + selective staging | CPU | COMPLETE |
| 2 | frozen champion selection (boost-A grid only) | CPU, High-RAM | COMPLETE (selection only) |
| 3 | official validation | CPU, High-RAM | COMPLETE or WARNING |
| 4 | export parity + promotion gates | CPU, High-RAM | READY_FOR_EXPORT |
| 5 | submission export (no upload) | CPU, High-RAM | READY_FOR_UPLOAD |
| 6 | leaderboard review (after a **manual** upload) | CPU | – |
| 7 | final report | CPU | – |

`CONFIG['mode']`: `FULL` · `RECOVER` · `VALIDATE` · `SUBMIT` (`docs/leaderboard_focus/M1_EXECUTION_GUIDE.md`).

**After a disconnect:** restart the session, run Stage 0, then the stage cells. Finished stages print `SKIP_CACHED`;
BLOCKED stages are always re-evaluated.

**Where to look** (`MyDrive/GEMS_2/outputs/leaderboard_focus/m1_champion_recovery/`):
* `reports/M1_EXECUTIVE_SUMMARY.md`, `reports/M1_NEXT_ACTION.md`, `reports/M1_FAILURE_DIAGNOSTICS.md`;
* `manifests/<stage>.json` (+ `manifests/history/`);
* `selection/promotion_gates.csv`, `validation/protocol_comparison.csv`, `submissions/README.md`.

**Stage 1 and the CNN maps:** `b0_maps` / `b1_maps` require `oof.tif`, `disc0.tif`, `disc1.tif`, `disc2.tif` by name
in exactly one experiment directory with the recorded id; other GeoTIFFs (the per-fold `oof_fold<k>.tif`) are
accepted and listed in `extra_files`. Nothing is deleted or renamed.

## Superseded

`00_ENVIRONMENT_AND_ARTIFACTS`, `01_CHAMPION_RECOVERY`, `02_OFFICIAL_VALIDATION`, `03_SUBMISSION_EXPORT`,
`04_LEADERBOARD_REVIEW` are kept for history (saved outputs untouched). Their `EXPECT['code_rev']` is `m1.1-r1`, so
they stop at the version check against the current code by design.

Docs: `docs/leaderboard_focus/README.md`.
