# Submission guide

## 1. Is the file ready?

Open `MyDrive/GEMS_2/outputs/leaderboard_focus/m1_champion_recovery/submissions/submission_manifest.json`. The file
is ready only if `"status": "READY_FOR_UPLOAD"`.

`BLOCKED` and `INCOMPLETE` files must not be uploaded; `reasons` says why.

## 2. Locate the GeoTIFF

The `file` field names the GeoTIFF: `submission_m1_<family>_<recipe_id>.tif`.

Next to it:

* the `.json` sidecar (recipe);
* `_validation.json`;
* `_diagnostics.csv`;
* `_preview.png`;
* `submission_sha256.txt`.

## 3. Upload manually

1. Download the `.tif` from Drive.
2. Check `sha256sum <file>` against `submission_sha256.txt`.
3. On DrivenData, open competition 306 → **Submissions** and upload the file. The limit is 3 scored submissions per
   rolling 7 days.
4. Note the submission id, the public score and the date shown.

Nothing in M1 uploads automatically.

## 4. Record the score

In notebook 04, fill `LEADERBOARD_ENTRY` (`submission_id`, `submission_file`, `public_dti`, `submitted_at`, `notes`)
and run the notebook.

The entry is refused when the file is not the READY M1 file, its sha256 changed, or the id or file was already
recorded.

## 5. Compare offline and public results responsibly

* The public board scores an undisclosed spatial subset. The offline number is a nested pseudo-discovery estimate on
  3 zones.
* A difference smaller than 0.005 is no evidence. Larger gaps have several possible causes: pseudo-discovery
  under-represents corrections near known faults, the zone variance is large, and the public subset is a chunk.
* Never select or tune anything on the public score. The next candidate comes from the offline protocol.
