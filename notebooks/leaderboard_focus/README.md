# Leaderboard focus: Milestone 1 notebooks

Purpose: one trustworthy, deployment-consistent DrivenData submission recovered from the frozen W2 artifacts. Nothing
is retrained or re-inferred.

| # | notebook | runtime | produces | gate to continue |
|---|---|---|---|---|
| 00 | `00_ENVIRONMENT_AND_ARTIFACTS.ipynb` | CPU | environment record, artifact inventory, staging manifest | inventory COMPLETE |
| 01 | `01_CHAMPION_RECOVERY.ipynb` | CPU, High-RAM | champion candidates, selection, recipe, provenance | selection COMPLETE |
| 02 | `02_OFFICIAL_VALIDATION.ipynb` | CPU, High-RAM | recomputed official metrics, recorded-value reproduction, parity diagnosis, figures | validation and parity COMPLETE |
| 03 | `03_SUBMISSION_EXPORT.ipynb` | CPU, High-RAM | one zone-stitched GeoTIFF with manifest, validation and sha256 | `READY_FOR_UPLOAD` |
| 04 | `04_LEADERBOARD_REVIEW.ipynb` | CPU | leaderboard history (after a **manual** upload) | – |

Every notebook is independently runnable once its prerequisites exist on Drive. Each one:

* clones the pinned commit;
* checks `EXPECT['code_rev']`;
* attaches with one call (`M1Session.attach`; notebook 04 uses `M1Session.light`);
* reloads finished stages by fingerprint.

All outputs live in `MyDrive/GEMS_2/outputs/leaderboard_focus/m1_champion_recovery/`.

**After a disconnect:** re-run the notebook from the top. Finished stages print `reloaded`.

**Where to look:**
* `manifests/<stage>.json`: status and reasons;
* `reports/*.md`: readable summaries;
* `submissions/README.md`: the upload instructions.

Docs: `docs/leaderboard_focus/README.md`.
