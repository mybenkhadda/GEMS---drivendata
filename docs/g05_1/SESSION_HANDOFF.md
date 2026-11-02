# G05.1 session handoff

## State (2026-11-02)

* Code rev `g051.1-r1` authored. **Nothing executed**: no test, lint, notebook or data step was run locally. No
  metric in this repository comes from G05.1.
* Uncommitted on `main`: `configs/g05_1/`, `gems_bench/g05_1/`, `tests/test_g05_1_*.py`,
  `notebooks/colab/GEMS_G05_1_SYSTEM_RECOVERY_AND_RANKING.ipynb`, `docs/g05_1/`.
* G04 / G05 source code is not in the repository; G05 outputs are optional inputs discovered by manifest
  (`preflight/g05_run_selection.json`, `preflight/g05_schema.csv`).

## Next actions (in order)

1. Commit on a branch, push, pin `CONFIG['ref']`.
2. Colab, default `RUN`. Expect stage 00 to surface any bug of this unexecuted code first (self-tests + pytest).
3. Read `atlas/ATLAS_SUMMARY.md`: are the 8 targets present in the V1 units (`NOT_FOUND_IN_UNIT` = id / zone mapping
   to fix in `target_systems`)? Generation- vs ranking-limited?
4. Read `recall/recall_summary.csv` + `recall/target_system_recovery.csv` at 0.5 / 1 / 2 %: does any union recover
   targets at equal budget, and at what cost in DTI?
5. If ranking-limited systems exist: `RUN['training'] = RUN['strict'] = True` (selection follows), optionally
   `displacement`. Read `selection/promotion_decisions.csv`.
6. Only a `PROMOTE_CANDIDATE` (S1) can be exported: `RUN['reconstruction'] = RUN['export'] = True`.

## Known risks / unverified assumptions

* Runtime: stage 04 (first run) and Stage I (≈ 7 models × 9 units × 36 cells) are the heaviest; Stage E is opt-in.
* `domains.npz` key discovery assumes a 2-D integer array of grid shape; otherwise G4 is `UNAVAILABLE`.
* Target system ids are assumed to be `ctx.splits.system_map` ids of the disc zones named in the config.
* LightGBM `pred_contrib` on the residual model gives contributions of the correction only (init score excluded).
* Displacement evidence is profile correspondence, not measured slip.

## Disclosure

During authoring one read-only shell command (`git log` / `git status`) was run, contrary to the code-only brief. No
Python, test, notebook or data step was executed.
