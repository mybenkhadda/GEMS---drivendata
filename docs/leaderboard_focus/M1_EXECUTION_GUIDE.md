# M1 execution guide

## Before Colab (local, by you; the authoring sessions ran none of this)

```bash
git switch -c m1-3-repair
pytest -q tests/test_m1_3_repair.py tests/test_leaderboard_focus.py tests/test_m1_consolidation.py tests/test_p6_full_gpu.py
ruff check gems_bench/leaderboard_focus gems_bench/p6/final_submission.py tests/test_m1_3_repair.py
git add gems_bench/leaderboard_focus gems_bench/p6/final_submission.py configs/leaderboard_focus \
        notebooks/leaderboard_focus docs/leaderboard_focus tests/test_m1_3_repair.py tests/test_m1_consolidation.py
git commit -m "M1.3: Drive preflight, upper-bound budget contract, budget/mask forensics, opt-in exploratory export"
git push -u origin m1-3-repair && git rev-parse HEAD
```

In `GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb`, set `CONFIG['ref']` and `EXPECT['commit']` to the pushed SHA.

## Running

1. Colab: CPU runtime, **High-RAM**. Fresh session (Runtime → Restart session).
2. Edit only the Stage 0 configuration cell (`mode`, `M1_CONFIG`, later `LEADERBOARD_ENTRY`).
3. Run all cells top to bottom. Each stage cell prints one line: `[m1] <stage> <STATUS> | <first reason>`.

| mode | stages run | use |
|---|---|---|
| `FULL` | 0c, 1, 2, 3, 4 (parity + promotion), 5, 5b, 6, 7 | normal run; downstream stages record INCOMPLETE / BLOCKED instead of running on a blocker |
| `RECOVER` | 0c, 1, 2, promotion view, 7 | re-select only |
| `VALIDATE` | 0c, 1, 3, 4, 7 | re-validate the persisted selection (**the M1.3 repair run**) |
| `SUBMIT` | 0c, 1, promotion, 5, 5b, 6, 7 | export the persisted validated recipe (after e.g. a promotion decision) |
| `PREFLIGHT` | 0c, 7 | read-only Drive check only (no engine, minutes) |

A stage outside the mode prints `NOT_REQUIRED`; its persisted manifest is not touched. The preflight (0c) and stage 1
run in every engine mode. A BLOCKED preflight makes every engine stage record INCOMPLETE without attaching the engine.
Stage 5b is `NOT_REQUIRED` (no engine attach) while `exploratory.enabled` is false (default).

## M1.3 repair run (recommended order)

1. `mode: 'PREFLIGHT'` → read `diagnostics/DRIVE_READINESS_REPORT.md`. Expect the last run's budget re-read
   (`budget_root_cause_preflight.csv`) to show all three zones `ok` with `CHECK_DEFINITION_MISMATCH`.
2. `mode: 'VALIDATE'` → validation re-runs (its stage code changed); parity runs for the first time; promotion is
   re-evaluated. Selection is reused (`SKIP_CACHED`: its fingerprint does not include the changed modules).
3. Decide on the geographic gate. To get a leaderboard baseline without promoting the model:
   `M1_CONFIG = {'exploratory': {'enabled': True, 'reason': '<why>'}}`, `mode: 'SUBMIT'`.

## Expected outcome (INFERRED from the observed M1.2 outputs, not a result)

* Preflight: COMPLETE or WARNING (the M1.2 stage manifests carry the old validation `impl`, which is reported).
* Stage 2: SKIP_CACHED, the same champion `a_w2_rep_frac` / recipe `0a8692b2ab`.
* Stage 3: every check that passed in M1.2 is unchanged. The budget passes under the upper-bound contract
  (0.46–0.49 % ≤ 1.02 %, 0 px on mapped faults), so validation should be COMPLETE. Not yet observed.
* Stage 4 parity: never run before. Its outcome is unknown.
* Promotion: technical validity PASS only if validation and parity pass. Internal promotion stays BLOCKED on
  `GEOGRAPHIC_GATE_PASS` (0.105687 < 0.1758).
* Stage 5: BLOCKED, no champion file. Stage 5b, only if opted in and technically valid: `EXPLORATORY_READY`.

## Cache and resume

* Every stage manifest records its fingerprint: inventory hashes, configuration, selection rule, recorded values,
  upstream fingerprints, and an implementation hash of the code the stage depends on (`session.IMPL_MODULES`) plus
  numeric dependency versions. An unrelated code edit does not invalidate a stage.
* A COMPLETE / WARNING stage is reused (`SKIP_CACHED`) only when the fingerprint matches and every output exists.
* A BLOCKED stage is always re-evaluated.
* A downstream stage refuses an upstream result produced by different stage code (`produced by different stage code`)
  and the promotion gates refuse validation / parity results produced for another selection (`stale`).
* After a disconnect: run Stage 0 again (fresh session), then the stage cells. Outputs are on Drive after every stage.

## Outputs

`MyDrive/GEMS_2/outputs/leaderboard_focus/m1_champion_recovery/`:
`environment/`, `artifact_inventory.{csv,json}`, `staging_manifest.json`, `selection/` (incl. `grid_provenance.csv`,
`promotion_gates.csv`), `validation/` (incl. `protocol_comparison.csv`), `parity/`, `figures/` (PNG + `_data.csv`),
`submissions/`, `leaderboard/`, `reports/` (`M1_EXECUTIVE_SUMMARY.md`, `M1_FAILURE_DIAGNOSTICS.md`,
`M1_NEXT_ACTION.md`, `M1_3_REPAIR_REPORT.md`, `stage_status.csv`), `manifests/` (+ `history/`, `preflight.json`,
`diagnostics.json`).

M1.3 additions:
* `diagnostics/` (preflight): `drive_inventory.{csv,json,parquet}`, `schema_checks.csv`,
  `artifact_compatibility.csv`, `artifact_dependency_graph.json`, `DRIVE_READINESS_REPORT.md`,
  `budget_root_cause_preflight.csv`, `root_cause_summary_preflight.json`.
* `diagnostics/` (validation): `budget_root_cause.{csv,parquet}`, `representation_comparison.{csv,parquet}`,
  `mask_alignment.csv`, `root_cause_summary.json`, `evaluated_vs_exported_maps.npz` (uint8 code map 0 none / 1 both /
  2 evaluated only / 3 deployed only + `zone_id`: the full-grid raster behind the comparison figure).
* `figures/`: `per_zone_area_budget`, `per_zone_area_budget_preflight`, `eligible_mask_comparison`,
  `evaluated_vs_exported`, `geographic_score_comparison` (each PNG + `_data.csv`).
* `submissions/exploratory_*` (opt-in only).

Every earlier output is kept. Rewritten stage files are versioned through `manifests/history/`, and submission files
are never overwritten.
