# Cache and resume

## Rules
1. **One namespace.** Every artifact lives under `GEMS_2/e6_v2_from_raw/`. Nothing else is read except the raw
   inputs (RAW_DATA_CONTRACT.md). There is no fallback to earlier experiment outputs: a missing artifact raises
   `MissingArtifact` with its recovery procedure.
2. **Atomic files.** Each file is written to `.tmp-<pid>-<name>` in its folder and moved into place with
   `os.replace`. Readers refuse `.tmp-*` / `*.partial` / `*.tmp` files.
3. **Manifests.** A stage is complete only when its `manifest.json` exists, its `fingerprint` equals the expected
   fingerprint and every listed output exists with the recorded size (sha256 recorded for files <= 256 MB). A
   stale manifest (fingerprint differs) is recomputed and the reason is logged (`reports/pipeline.log`).
4. **Raw fingerprint.** Unchanged raw files (same path, size, mtime) reuse their checksum from
   `manifests/inventory_hash_cache.json`.

## Stage by stage
| stage | unit of resume | after a disconnect |
|---|---|---|
| inventory / preflight | whole stage (cheap after the first hash) | re-run |
| base / terrain / potential | whole stage per manifest | re-run; complete stages are skipped |
| DEM windows | one window (`<id>.npz`, then `<id>.json`) | re-run section 05: done windows skipped; the window plan is deterministic |
| DEM assembly | whole stage | re-run section 06 |
| datasets / splits | whole stage | re-run |
| training | one epoch per unit and stage (`<stage>_last.pt`) | re-run the section: the unit resumes after its last finished epoch with model, optimizer, scheduler, scaler, RNG (python, numpy, torch, cuda) and patch-sampler RNG restored |
| reference | immutable file | never rewritten |
| evaluation | one unit (`predictions/units/<unit>_eval.npy`, keyed by checkpoint sha) | re-run section 12 |
| full inference | one fold map (sidecar keyed by checkpoint sha + dataset fingerprint) | re-run section 13 |
| export | whole stage | re-run section 14 |

## Session budgets
- `dem.budget_minutes`: the DEM window loop stops submitting new windows after this time, finishes the windows in
  flight and returns `{"status": "incomplete"}`; section 05 then raises with the number of windows left.
- `training.max_stage_minutes`: after an epoch that exceeds the budget, training raises `BudgetExhausted`; re-run.

## Local scratch
`/content/e6v2_scratch/datasets/` holds a copy of `datasets/*.npy` (staged once per dataset fingerprint, sizes
checked); it is a speed copy, never a source of truth.

## Identity mismatches
A checkpoint whose identity differs from the current one is never loaded (`IdentityMismatch` lists the differing
keys). If the change is intended (new data, new config), delete `checkpoints/<unit>/` and re-run. The reference is
immutable on purpose: a new reference means a new unit history.

## Disk (estimates from array shapes, not measured)
terrain 0.7 GB, potential 2.2 GB, DEM 100 m 1.1 GB, datasets about 2.2 GB, windows about 2000 x 0.2 MB, QA
windows about 3 x 70 MB, checkpoints about 6 units x (A, D, E: last + best + final) x 10-60 MB, predictions
3 x 50 MB.
