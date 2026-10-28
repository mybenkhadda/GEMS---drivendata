# M1 handoff

**State (2026-10-28, code rev m1.2-r1):** consolidation authored; NOTHING executed in the authoring sessions (no
Python, test, linter, notebook, Drive access or upload). The new and changed tests were written, never run.

## Observed Colab history (from the user's runs; not re-run here)

1. Notebook 00 BLOCKED: `b0_maps / b1_maps: COUNT_MISMATCH (7 files, expected 4)` → fixed by `required_files`
   (`training.cv` also writes `oof_fold<k>.tif`).
2. `champion.recover` → `FB.load_grids` → `pd.read_csv` → `EmptyDataError` on `seeds/grid.csv` (1 byte); boost B/C/D
   absent; boost A (99,538 B, 28 cols) and rec_x3 valid. A notebook-only workaround narrowed `fg.cfg["grids"]`.
3. The same error in `02_OFFICIAL_VALIDATION` via `reproduce_recorded` (the workaround did not reach the loader).
4. With the workaround, selection returned COMPLETE: champion `a_w2_rep_frac`, recipe `0a8692b2ab`, discovery 0.242156,
   geo 0.105687; `geo_floor_met = False` for every evaluated candidate.

## What changed (m1.2)

* **Shared loader** `p6/full_benchmark.py`: `load_grids(fg, sources=None, required=None)`, `read_csv_checked`,
  `read_grid_csv`, `GridUnavailable`, `resolve_specs(..., candidates=None)`. `sources=None` keeps the P6 behaviour, but a
  damaged optional grid is now recorded instead of crashing. M1 callers (`champion.recover`, `validation.
  reproduce_recorded`) read only `champion.grids` and require `champion.required_grids` (`[boost_a]`). The
  `fg.cfg["candidates"]` swap in `champion._resolve` is gone.
* **One notebook** `notebooks/leaderboard_focus/GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb` + `workflow.Runner` (modes FULL /
  RECOVER / VALIDATE / SUBMIT). Notebooks 00–04 are marked superseded.
* **Statuses** COMPLETE / WARNING / BLOCKED / INCOMPLETE / FAILED / SKIP_CACHED / NOT_REQUIRED (+ READY_FOR_UPLOAD).
  `StageInvalid` → BLOCKED; a BLOCKED result is never served from cache; manifests carry `impl` (stage code hash) and
  `require` refuses results of different stage code; every manifest is also kept in `manifests/history/`.
* **Promotion gates** `gates.py` (SELECTION_COMPLETE, DISCOVERY_ELIGIBLE, GEOGRAPHIC_GATE_PASS, VALIDATION_PASS,
  PARITY_PASS → READY_FOR_EXPORT); the geographic gate is **mandatory** by default (`promotion.geographic_gate`;
  `advisory` requires an `override_note`). Export runs only on READY_FOR_EXPORT and otherwise writes no file.
* **Protocol labels**: recorded values carry `protocol` (separate / fixed) and `comparable_to_champion`;
  `validation/protocol_comparison.csv`; a non-reproduced record is a WARNING by default (`recorded_mismatch`).
* **Staging**: the session re-stages exactly the inventory's staged files (no `fg.stage_inputs`, so no `seeds/`,
  historical submissions or boost B/C/D). W2 maps are also required by name.
* **Reports**: `reports/M1_EXECUTIVE_SUMMARY.md`, `M1_FAILURE_DIAGNOSTICS.md` (failure catalogue
  `reporting.FAILURE_MODES`), `M1_NEXT_ACTION.md`, `stage_status.csv`.

## Expected outcome of the next run (prediction from recorded numbers, not a result)

Stages 1–2 COMPLETE with the same champion; stages 3–4 run to their own status; promotion BLOCKED on
`GEOGRAPHIC_GATE_PASS` (0.105687 < 0.1758); no GeoTIFF. The comparable P6.1 single-policy records show no W2 policy
(and not the M9 fusion, 0.158416) at the floor, so the floor question is a **user decision**, not a bug to fix.

## Open decisions / risks (need Colab)

* **Geographic floor:** keep mandatory (no submission from this champion) or `advisory` + note.
* **Strict parity** blocks a blend < 1 champion (not expected: the recorded champion cell has blend 1 for `w2`, unverified).
* **First-run API risks:** `stage_frozen_inputs` replaces `fg.stage_inputs` — if some P6 code path reads a file
  outside the staged list (e.g. `p6.budget` set by `Boost.stage`, `probes/`), it will fail with FileNotFoundError /
  AttributeError; add the file to `m1.yaml → artifacts` with `staged: true`. `EP.agreement` key names; historical
  sidecar fields.

## Milestone 1 is complete when (all of them, verified in Colab)

1. `manifests/{artifacts, selection, validation, parity, promotion}.json` are COMPLETE / WARNING (warnings read).
2. `submissions/submission_manifest.json` → `READY_FOR_UPLOAD`; `export_parity_report.json` → `equal: true`.
3. The file is uploaded by hand and its public score recorded in Stage 6 (context only, never for selection).

## Paste-ready summary for the next session

```text
GEMS M1 consolidation (code rev m1.2-r1) AUTHORED 2026-10-28, nothing executed, tests unrun
(tests/test_leaderboard_focus.py, tests/test_m1_consolidation.py). Canonical notebook:
notebooks/leaderboard_focus/GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb (Runner in gems_bench/leaderboard_focus/workflow.py,
modes FULL/RECOVER/VALIDATE/SUBMIT); notebooks 00-04 superseded. Shared fix: p6/full_benchmark.load_grids(fg,
sources, required) + read_csv_checked/GridUnavailable (EmptyDataError on the 1-byte seeds/grid.csv can no longer reach
M1; P6 callers unchanged). New: gates.py promotion gates (geo floor 0.1758 MANDATORY by default; advisory needs
override_note), protocol-labelled recorded values (separate-protocol geo not comparable with single-policy geo),
impl-hash cache keys, BLOCKED never cached, manifests/history, targeted staging, reports/M1_*.md. Last observed:
champion a_w2_rep_frac recipe 0a8692b2ab disc 0.242156 geo 0.105687 -> expect promotion BLOCKED on
GEOGRAPHIC_GATE_PASS (user decision). Read reports/M1_NEXT_ACTION.md + manifests/*.json on Drive before claiming
anything. Do not start Milestone 2 until READY_FOR_UPLOAD is verified.
```
