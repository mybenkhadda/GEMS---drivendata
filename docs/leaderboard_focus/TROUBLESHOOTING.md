# Troubleshooting

Start with `reports/M1_FAILURE_DIAGNOSTICS.md` (written by Stage 7, and also when a stage raises): every non-passing
stage with its reasons, matched against the known failure modes (`reporting.FAILURE_MODES`), plus tracebacks.

## Known failure modes

| code | detected at | symptom | status | fix |
|---|---|---|---|---|
| ART_COUNT | stage 1 | `COUNT_MISMATCH` on a count-based artifact | BLOCKED | restore the frozen folder. (The observed `b0_maps / b1_maps: 7 files, expected 4` is fixed: those use `required_files`.) |
| ART_REQUIRED_FILE | stage 1 | `MISSING_REQUIRED_FILES` (note lists what was found) | BLOCKED | restore the named file (`oof.tif`, `disc0-2.tif`, `geo_0-2.npy`, `disc_official_0-2.npy`) from the original run |
| ART_MISSING | stage 1 | `MISSING` | BLOCKED | restore it; template: run the pipeline `raw_data` step once |
| ART_AMBIGUOUS | stage 1 | `AMBIGUOUS: several experiments match` | BLOCKED | M1 never combines or picks; resolve the duplicate experiment folder by hand |
| ART_ID | stage 1 | `ID_CHANGED` for b0 / b1 | BLOCKED | confirm the model, then set `expected_id` explicitly in `M1_CONFIG['artifacts']` |
| STAGING | engine attach | `staged input(s) absent locally` | INCOMPLETE | re-run stage 1 (re-stages and verifies) |
| GRID_REQUIRED | stage 2 / 3 | `required grid(s) unusable: boost_a EMPTY_FILE / NO_HEADER / MALFORMED / ZERO_ROWS / SCHEMA_INVALID / MISSING` | BLOCKED | restore `outputs/p6_fast_track/boost/benchmark/grid.csv` (facts in `selection/grid_provenance.csv`). No fallback grid, no recorded score. |
| (fixed) | stage 2 / 3 | `pandas.errors.EmptyDataError: No columns to parse` from `load_grids` | – | the shared loader now reads only `required_grids`; an empty optional grid (the 1-byte `seeds/grid.csv`) is recorded `EMPTY_FILE` and never read by M1. Do not narrow `fg.cfg["grids"]` in a notebook. |
| UPSTREAM | every stage | `upstream stages not usable` | INCOMPLETE | run mode `FULL` (or the earlier stage) |
| STALE_CODE | every stage | `produced by different stage code` | INCOMPLETE | re-run that stage; `FULL` does it automatically |
| TABLE | stage inputs | `<table> is absent or empty` | BLOCKED | re-run the stage that writes it |
| RECIPE | stages 3 / 5 / promotion | `recipe ids disagree` | BLOCKED | re-run selection; never edit selection files by hand |
| PREFLIGHT | preflight | `preflight BLOCKED (diagnostics/DRIVE_READINESS_REPORT.md)` on an engine stage | INCOMPLETE | a required frozen input is missing / unreadable / malformed / misaligned: restore the file named in the report; the engine is not attached until it passes |
| GEO_FLOOR | promotion | `GEOGRAPHIC_GATE_PASS BLOCKED: geo_dti … < floor 0.1758` | BLOCKED | a decision, not a bug (`VALIDATION_AND_PARITY.md` → comparable numbers): keep it unexported, or opt in to the exploratory baseline `M1_CONFIG = {'exploratory': {'enabled': True, 'reason': '<why>'}}` and run mode `SUBMIT` (separate file, `EXPLORATORY_READY`, never the champion). The older `promotion.geographic_gate: advisory` + `override_note` exports under the champion name. |
| EXPLORATORY | stage 5b | `exploratory export needs technical validity PASS and GEOGRAPHIC_GATE_PASS as the only failing gate` | BLOCKED | another gate fails (discovery, validation, parity): fix that first; the exploratory path never bypasses technical checks |
| EXPLORATORY_STALE | stage 5b | `promotion gates absent or stale` | INCOMPLETE | run the promotion stage (mode `SUBMIT` or `FULL` runs it first) |
| DISC_ELIG | promotion | `DISCOVERY_ELIGIBLE BLOCKED` | BLOCKED | the champion fails the predeclared rule; read `eligibility_reasons` |
| COMPONENTS | stage 3 | `recomputed components differ from the saved grid` | BLOCKED | a frozen input changed after boost A: compare `champion_provenance.json` with `artifact_inventory.json` |
| ZONE_PARITY | stage 3 / 4 | `zone-stitched score parity not PASS` | BLOCKED | blend < 1 champion (per-zone vs all-zone fusion w): explicit decision `validation.strict_parity: false` (recorded) |
| (fixed, M1.3) | stage 3 | `per-zone area budget / known-fault mask violated on zones [0, 1, 2]` with `positive_on_known_px = 0` and `rel_error ≈ -0.5` | – | the M1.1 two-sided check expected the thinned area to equal `frac`; the contract is now the upper bound used by `audit_export` (`VALIDATION_AND_PARITY.md` → per-zone area budget). The prediction is unchanged. |
| BUDGET | stage 3 | `per-zone area budget / known-fault mask violated on zones [...]: {zone: cause}` | BLOCKED | a real violation under the M1.3 rule: read `cause` (`AREA_OVER_ALLOCATION`, `KNOWN_FAULT_MASK_LEAK`, empty map) in `validation/zone_budget.csv` and `diagnostics/root_cause_summary.json`; never widen `budget_tolerance` to pass |
| FORENSICS | stage 3 | warning `forensics diagnostics failed` | WARNING | a diagnostic raised, the validation verdict is unaffected: traceback in `diagnostics/forensics_error.txt` |
| TEMPLATE | stage 3 | `official template check` | BLOCKED | restore `data/raw/geodawn/example_submission.tif`; see `submission/official_spec.json` |
| RECORDED | stage 3 | `… NOT_REPRODUCED` | WARNING | compare the grid sha256; read `protocol_comparison.csv` before concluding anything |
| EXPORT_SHA | stage 5 | `written band == array validated …` failed | BLOCKED | re-run the export; never upload that file |
| NAME | stage 5 | `FileExistsError` | FAILED | two other recipes own the name; archive them by hand (M1 never overwrites) |
| ORACLE | stage 2 | `OracleLeak` | FAILED | label-built rows reached a grid; the grid cannot be used for selection |

## Runtime problems

| problem | symptom | fix |
|---|---|---|
| Colab disconnection | the runtime resets | restart the session, run Stage 0, then the stage cells; finished stages print `SKIP_CACHED` |
| module already imported | `gems_bench already imported` | Runtime → Restart session, then Stage 0 |
| incorrect commit | `AssertionError` on `EXPECT` or `git … failed` | pin `CONFIG['ref']` / `EXPECT['commit']` to the pushed SHA; never falls back to another branch |
| old notebook | `code rev m1.3-r1 != m1.1-r1` in `00`–`04` | by design: those notebooks are superseded; use `GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb` |
| stale clone | `code rev m1.2-r1 != m1.3-r1` in the canonical notebook | `CONFIG['ref']` points at a commit without the M1.3 repair: pin the pushed M1.3 SHA |
| insufficient memory | the kernel dies in stages 2–5 | High-RAM runtime; restart and re-run (stages resume) |
| training flag set | `M1ConfigError: … training is outside its scope` | set `train_cnn`, `train_transformer`, `run_extra_seeds` to False |
| leaderboard entry refused | `REJECTED:` in stage 6 | fix the named field, or upload the exact validated file |
| W2 historical parity blocker | historical rows BLOCKED in `parity_report.csv` | expected and documented: the champion does not use the historical export; only champion rows block |
