# M1 execution guide

## Before Colab (local, by you; the authoring sessions ran none of this)

```bash
git switch -c m1-consolidation
pytest -q tests/test_leaderboard_focus.py tests/test_m1_consolidation.py tests/test_p6_full_gpu.py
ruff check gems_bench/leaderboard_focus gems_bench/p6/full_benchmark.py tests/test_m1_consolidation.py
git add gems_bench/leaderboard_focus gems_bench/p6/full_benchmark.py configs/leaderboard_focus \
        notebooks/leaderboard_focus docs/leaderboard_focus tests/test_leaderboard_focus.py tests/test_m1_consolidation.py
git commit -m "M1: single resumable notebook, shared grid loader, promotion gates"
git push -u origin m1-consolidation && git rev-parse HEAD
```

In `GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb`, set `CONFIG['ref']` and `EXPECT['commit']` to the pushed SHA.

## Running

1. Colab: CPU runtime, **High-RAM**. Fresh session (Runtime → Restart session).
2. Edit only the Stage 0 configuration cell (`mode`, `M1_CONFIG`, later `LEADERBOARD_ENTRY`).
3. Run all cells top to bottom. Each stage cell prints one line: `[m1] <stage> <STATUS> | <first reason>`.

| mode | stages run | use |
|---|---|---|
| `FULL` | 1, 2, 3, 4 (parity + promotion), 5, 6, 7 | normal run; downstream stages record INCOMPLETE / BLOCKED instead of running on a blocker |
| `RECOVER` | 1, 2, promotion view, 7 | re-select only |
| `VALIDATE` | 1, 3, 4, 7 | re-validate the persisted selection |
| `SUBMIT` | 1, promotion, 5, 6, 7 | export the persisted validated recipe (after e.g. a promotion decision) |

A stage outside the mode prints `NOT_REQUIRED`; its persisted manifest is not touched. Stage 1 runs in every mode: it
is the input check that every later stage relies on (the staging itself is skipped while nothing changed).

## Expected first run with the current Drive state

From the last observed Colab run (not re-run in this session):

* Stage 1: COMPLETE (after the required-file fix).
* Stage 2: COMPLETE — champion `a_w2_rep_frac`, recipe `0a8692b2ab` (recipe id is code-independent, so it should
  recur if the inputs are unchanged), discovery 0.242156, geo 0.105687.
* Stage 3 / 4: not yet observed. They run to their own status.
* Promotion: with the default mandatory geographic gate, BLOCKED on `GEOGRAPHIC_GATE_PASS` (0.105687 < 0.1758).
* Stage 5: BLOCKED, no file. `reports/M1_NEXT_ACTION.md` states the decision you need to make.

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
`M1_NEXT_ACTION.md`, `stage_status.csv`), `manifests/` (+ `history/`).
