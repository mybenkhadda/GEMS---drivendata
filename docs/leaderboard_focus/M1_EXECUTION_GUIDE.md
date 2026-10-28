# M1 execution guide

## Before Colab (local, by you; the authoring session ran none of this)

```bash
git switch -c m1-champion-recovery
pytest -q tests/test_leaderboard_focus.py
ruff check gems_bench/leaderboard_focus tests/test_leaderboard_focus.py
git add gems_bench/leaderboard_focus configs/leaderboard_focus notebooks/leaderboard_focus docs/leaderboard_focus \
        tests/test_leaderboard_focus.py
git commit -m "M1: champion recovery, official validation, zone-stitched export, leaderboard log"
git push -u origin m1-champion-recovery && git rev-parse HEAD
```

Put the pushed ref or SHA in `CONFIG['ref']` and `EXPECT['commit']` of every notebook.

## Notebook 00: environment and artifacts

* **Runtime:** CPU, a few minutes. The time depends on the sha256 of the parquet / grids.
* **Steps:**
  1. Configuration.
  2. Clone and version lock.
  3. `run_environment`.
* **Outputs:** `environment/environment.json`, `artifact_inventory.{csv,json}`, `staging_manifest.json`,
  `manifests/artifacts.json`.
* **Continue when:** `ARTIFACT STATUS: COMPLETE`.
* **Resume:** re-run. Staging is skipped while the inventory fingerprint is unchanged.

## Notebook 01: champion recovery

* **Runtime:** CPU, High-RAM.
  * Attaching (cached data staging) takes about 5–10 min, judging from the last P6 run's sync times.
  * Selection takes minutes (saved grid rows only).
* **Steps:**
  1. Configuration.
  2. Clone and `M1Session.attach`.
  3. `champion.recover`.
* **Outputs:** `selection/*`, `reports/recovery_report.md`.
* **Continue when:** `SELECTION STATUS: COMPLETE`.
* **On failure:** read `champion_selection.json` → `reasons` and `blocked_candidates.csv`.

## Notebook 02: official validation and parity

* **Runtime:** CPU, High-RAM. The dry zone-stitched build crops every geo zone, so allow 5–20 min. This is the
  p6g planning figure, not measured.
* **Steps:**
  1. Configuration.
  2. Attach.
  3. `validation.validate`.
  4. `parity.diagnose`.
* **Outputs:** `validation/*`, `parity/*`, `figures/*`, `reports/validation_report.md`.
* **Continue when:** both stages are COMPLETE.

## Notebook 03: submission export

* **Runtime:** CPU, High-RAM; 5–20 min (the build runs again, then write and validate).
* **Steps:**
  1. Configuration.
  2. Attach.
  3. `submission.export`.
* **Outputs:** `submissions/*`.
* **Continue when:** `SUBMISSION STATUS: READY_FOR_UPLOAD`. Then upload by hand (`SUBMISSION_GUIDE.md`).

## Notebook 04: leaderboard review

* **Runtime:** CPU, seconds.
* **Use:** fill `LEADERBOARD_ENTRY` after the manual upload and run the notebook.
* **Outputs:** `leaderboard/*`.

## Resume and failure handling (all notebooks)

* Each stage writes `manifests/<stage>.json` with its status: COMPLETE, BLOCKED, INCOMPLETE or FAILED.
* On re-run, a stage whose fingerprint and outputs match is reloaded, not recomputed.
* A changed input changes the fingerprint and the stage is recomputed.
* An exception is recorded as FAILED (with the traceback) and re-raised.
* An upstream stage that is not COMPLETE gives INCOMPLETE.
