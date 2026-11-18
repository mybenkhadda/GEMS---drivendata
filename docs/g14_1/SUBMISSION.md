# Submission generation (Phase R)

One function, `gems_bench.g14_1.submission.generate_submission`, is called from one notebook cell (25). It is UNTESTED.

## Steps

1. Cell 01: `CONFIG['mode'] = 'SUBMIT'` (`CONFIG['stages']['submission']` defaults to on in SUBMIT mode). Rerun
   cells 01 and 05.
2. Cell 25: set `SUBMIT_MODEL`, then run the cell. The options are:
   * `'g14'`: the source experiment's `maps/full.npy`;
   * `'g14_1:<spec>'`: a G14.1 final map from `pipe.train_final('<spec>')` in mode TRAIN;
   * `'g13:<model>'`: a G13 `full.npy`;
   * `'p6_w2'`: `full_score.tif`;
   * an explicit `.npy` / `.tif` path.
3. Optional: `SUBMIT_KW = {'rep': 'sparse_soft', 'frac': 0.01}` forces a MANUAL operating point.

## What the function does

1. Finds the full-region map (local or Drive, used in place). If it is missing, it raises one error that names the
   stage producing it.
2. Finds the operating point of **the same model**:
   * first `calibration/selected_operating_points.json` (R03 / R05 / R09+), discovery task first;
   * then, for `'g14'`, the G14 scorecard (`disc_*`, then `geo_*`).

   An operating point of another model is refused unless `allow_transfer=True` (labelled `TRANSFERRED_*`).
3. Status labels:
   * `DISCOVERY_CALIBRATED`;
   * `PROVISIONAL_GEO` (geographic calibration only, as in the G14 submission);
   * `MANUAL`;
   * `TRANSFERRED_*`.
4. Renders the representation on the official domain:
   * eligible = data footprint minus pixel-exact known faults;
   * known faults are written as 0;
   * NaN outside the template;
   * soft calibration is re-applied for temperature / power.
5. Writes `<local>/g14_1/submissions/<model>__<rep>_<frac>__<status>.tif` with the G13 writer, then runs:
   * the G13 validator;
   * the G14.1 parity check (shift, flips, binarisation, footprint, known-pixel positives);
   * a provenance JSON with map sha, operating point, area accounting, cfg hash and code revision.
6. Copies both files to `MyDrive/GEMS_2/outputs/G14_1_CRITICAL_RECOVERY/submissions/` and prints the local and Drive
   paths.

## Rules

* Several candidates can be written for offline comparison, each named by model, operating point and status.
* Nothing is uploaded, and the competition API is never called.
* Do not iterate on leaderboard feedback. The G14 PROVISIONAL_GEO submission (0.0189) is a data point, not a
  calibration target.
* There is no confirmation token and no flag juggling: the mode is the only gate.
