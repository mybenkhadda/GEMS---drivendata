# Troubleshooting

| problem | symptom | fix |
|---|---|---|
| Colab disconnection | the runtime resets | re-run the notebook from the top; finished stages print `reloaded` (fingerprint match) |
| missing Drive artifacts | notebook 00 `MISSING` / `COUNT_MISMATCH` | restore `outputs/p6_fast_track/...` or the experiment folders. M1 never recomputes W2 or CNN maps. |
| invalid checkpoint | – | M1 uses no checkpoint. A model error means the wrong notebook. |
| changed experiment id | `ID_CHANGED` for b0 / b1 | confirm the model; then set `expected_id` deliberately in `M1_CONFIG['artifacts']` |
| incorrect commit | `AssertionError` on `EXPECT` or `git … failed` | pin `CONFIG['ref']` to the pushed SHA; the notebooks never fall back to another branch |
| insufficient memory | the kernel dies in 01–03 | use a High-RAM runtime; restart and re-run (the stages resume) |
| invalid GeoTIFF | 03 BLOCKED with a failed official / audit check | read `submission_validation.json`; check the template (`submission/official_spec.json`) |
| W2 parity blocker | the historical rows in `parity_report.csv` are BLOCKED | expected and documented: the champion does not use the historical export. Only champion rows block. |
| zone parity WARNING | 02 BLOCKED: "zone score parity not PASS" | the champion's blend is below 1 and the fusion w differs per zone. This needs an explicit decision (`strict_parity: false`), which is recorded. |
| stale feature fingerprints | recomputed ≠ grid in 02 | an input changed after boost A. Restore the frozen version (compare `champion_provenance.json` with `artifact_inventory.json`). |
| cache mismatch | "fingerprint matches but outputs missing" | the stage is recomputed automatically. Never edit manifests by hand. |
| missing submission template | `template MISSING` in 00 | run any pipeline notebook's `raw_data` step once, or restore `data/raw/geodawn/example_submission.tif` |
| file name collision | `FileExistsError` in 03 | two other recipes already own the name. Archive them by hand; M1 never overwrites. |
| leaderboard entry refused | `REJECTED:` in 04 | fix the named field, or upload the exact validated file |
