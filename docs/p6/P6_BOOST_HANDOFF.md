# P6 boost — handoff (revision r2, 2026-10-27)

Design and stage definitions: `P6_BOOST.md`. This file covers the state, the r2 repairs, the blockers and the exact
next steps. **Nothing in r2 was executed**: no tests, no Ruff, no notebook cell, no data. The 84 passing tests and
clean Ruff reported for r1 apply to commit `4fa5b48` only, not to r2.

## 1. Source state (from `.git` metadata, read without git commands)

| ref | commit | note |
|---|---|---|
| local `p6-boost` | `4fa5b48` "P7 Performance Boost…" | r1 boost (code + notebook 3B), parent `794a7bd` |
| `origin/main` | `453bde3` | `4fa5b48` was pushed **to main**, then main fast-forwarded to `453bde3` (content not inspected; probably a Colab notebook save) |
| `origin/p6-boost` | **absent** | the notebook's `CONFIG['ref'] = 'p6-boost'` cannot resolve in Colab until it is pushed |

r2 is uncommitted in the working tree on `p6-boost`.

* Modified: `gems_bench/p6/boost.py`, `gems_bench/m9/research/g02_io.py`, `gems_bench/m9/research/g02_experiment.py`,
  `configs/p6/boost.yaml`, `tests/test_p6_boost.py`, `docs/p6/P6_BOOST.md`.
* Added: `gems_bench/p6/boost_audit.py`, `scripts/p6/patch_p6_boost_notebook.py`, this file.
* The notebook is **not** modified yet: it is too large for the editor tools. Run the patch script (step 2 in §4).

## 2. Defects confirmed in r1 and repaired in r2

| # | Defect (r1) | Repair (r2) |
|---|---|---|
| 1 | `resolve_run` picked the newest of several complete G02 runs | only run → used; several → `AmbiguousRunError`; corrupt config = incomplete; `list_runs` survives corrupt manifests |
| 2 | backfill reused the run config but never checked that today's inputs match the run's | `G02Experiment.input_drift()`; `Boost.backfill` refuses on any changed / missing / extra input fingerprint |
| 3 | features cached in `Boost._raw` before a backfill stayed in use after it | cache cleared after backfill |
| 4 | toggling `backfill_missing` alone changed the geology fingerprint | execution switches excluded (`EXEC_SWITCHES`); G02 run identity (id, fingerprint, manifest sha) included instead |
| 5 | cross-fitted variant maps were keyed without the G02 run, so another run could reuse stale S1–S3 maps | G02 identity in the variant fingerprint (only for variants that use G02 features, so other caches stay valid) |
| 6 | a variant with a unit skipped ("too few rows") was scored on fewer zones and could be ACCEPTed | such a variant is BLOCKED (not matched) |
| 7 | `_done` trusted the manifest only; a corrupt JSON crashed `read_json` | reload needs readable manifest + fingerprint + every required output (`STAGE_OUTPUTS`) + unchanged protocol keys (masks / truth / budget domain) when recorded |
| 8 | no stage status | `<stage>/status.json` PENDING / BLOCKED / RUNNING / COMPLETE / FAILED; failures and spent budget recorded; geology = BLOCKED when every G02 variant is |
| 9 | `selection.export: true` + notebook `EXPORT = True` exported automatically on PASS, and every rerun refit and wrote a new timestamped file + log row | export opt-in (`BOOST_RUN['export']`); `selection/exports.json` keyed by candidate fingerprint → a rerun reuses the file |
| 10 | promotion never looked at training / inference parity | `preflight/train_inference_parity.json` required; a BLOCKED row on a candidate feature (or `*`) fails provenance |
| 11 | `_provenance_ok` ignored W2 input changes | stage decisions made on other W2 map hashes fail provenance |
| 12 | interrupted cross-fit lost earlier importances | `importance.csv` merged atomically on resume |
| 13 | benchmark had no `decision.json`; selection had no manifest | both written |

Confirmed by reading the code, still open (not "fixed"):

* **W2 submission provenance.** `stack.full_region` takes rep + blend from the discovery choice and frac from the geo
  choice (`area_frac_source: geo`). The uploaded W2 file is therefore `skeleton_1px @ 5 %`, a cell no nested evaluation
  selected. `preflight/submission_provenance.json` rebuilds it from the writer sidecar `submissions/<stem>.json`.
  The result is VERIFIED only when the sidecar and file exist and agree. `selected_cell: false` flags the problem.
  W2 itself is unchanged.
* **Training / inference shift of W2 features** (recorded as WARNING, not BLOCKED):
  * CNN: held-out unit maps in training, full-region maps at inference.
  * `cnn_disagree`: ranked per unit in training, over the whole footprint at inference.
  * Known-fault features: permitted set in training, every mapped fault at inference (by design).
  * G02 references: training geography in training, whole footprint at inference.
  * Fusion weight: per zone in evaluation, one pooled `w` at export.
  * `preflight.parity_numeric: true` measures the CNN part in Colab (Spearman + top-1 % Jaccard per discovery unit).
    A Spearman below 0.2 BLOCKs.
* **DEM** (code audit; `feature_provenance.json → dem_integration`):
  * The pipeline reads the tnm JSON index, resolves the 1 m TIFFs and aggregates them to the 100 m grid. This is
    M7 `dem_morph`, and it covers pilot tiles only.
  * The DEM is used **only by the W4 gate**. It is in no training row and no inference.
  * `boost_audit.dem_features` is an optional, label-free adapter (NaN outside coverage) and belongs to no candidate.
  * Elevation, relief and ridge/valley layers do not exist without new tile processing.

## 3. Artifacts

Existing (Drive, from earlier runs):

* W2: `stack/candidate.json`, `stack/features/rows.parquet`, `stack/maps/*.npy` (6).
* W2 submission + sidecar.
* G02 run `g02_23686a63cb`: FAST_RESEARCH, rasters for `geo_0/1` and `disc_official_0/1` only.

Whether any r1 boost stage already ran in Colab is unknown: check `boost/*/manifest.json`.

New in r2, all under `outputs/p6_fast_track/boost/`:

* `preflight/`:
  * `g02_attachment_audit.json`, `g02_backfill_plan.json`
  * `feature_inventory.csv`, `feature_compatibility.csv`, `feature_provenance.json`
  * `raster_alignment_audit.csv`, `dem_coverage.csv`
  * `train_inference_parity.json/.csv`
  * `evaluation_contract.json`, `submission_provenance.json`
  * `manifest.json`, `status.json`
* `<stage>/status.json` and `benchmark/decision.json`.
* `selection/manifest.json`, `selection/exports.json`.
* `reports/`:
  * `boost_summary.csv`, `per_zone_comparison.csv`, `system_recovery.csv`, `experiment_status.csv`
  * `leaderboard_readiness.json`, `boost_report.md`
  * `dti_by_area.png` + `_data.csv`, `artifact_registry.csv`

Reload rule: a stage recomputes only when its fingerprint, required outputs or protocol keys change. Variant maps stay
cached per variant fingerprint; G02 backfill resumes per tile. The geology fingerprint changes once in r2 (it now
includes G02 identity). A, C and D fingerprints are unchanged.

## 4. Colab instructions

Local, once:

```bash
git switch p6-boost
git fetch origin && git show --stat origin/main          # see what 453bde3 changed (notebook?)
git merge --ff-only origin/main                          # p6-boost has no commit beyond 4fa5b48 (stash r2 if refused)
python scripts/p6/patch_p6_boost_notebook.py --check     # then without --check (idempotent, outputs kept)
pytest -q tests/test_p6_boost.py tests/test_p6.py tests/test_g02_structural_geology.py
ruff check gems_bench/p6 gems_bench/m9/research tests/test_p6_boost.py scripts/p6
git add gems_bench/p6/boost.py gems_bench/p6/boost_audit.py gems_bench/m9/research/g02_io.py \
        gems_bench/m9/research/g02_experiment.py configs/p6/boost.yaml tests/test_p6_boost.py \
        scripts/p6/patch_p6_boost_notebook.py docs/p6/P6_BOOST.md docs/p6/P6_BOOST_HANDOFF.md \
        notebooks/colab/GEMS_P6_FAST_TRACK.ipynb
git commit -m "P6 boost r2: preflight contracts, strict G02 resolution, cache validity, explicit export"
git push -u origin p6-boost
git rev-parse HEAD                                        # pin this SHA in Colab: CONFIG['ref'] and EXPECT['commit']
```

Colab (CPU, high-RAM). Run in this order and stop at the first failure:

| step | cell | switch | expect | est. |
|---|---|---|---|---|
| 0 | config, clone, **0b start-up check**, context | — | prints ref, SHA, module path, config fp; raises on mismatch | 5 min |
| 1 | 3B.0 `bst.stage()` | — | `candidate.json`, `rows.parquet`, 6 maps | 2 min |
| 2 | 3B.P preflight | `preflight` | `areas` table; G02 decision `PARTIAL` (zone 2 missing) expected; W2 file `selected_cell: false` expected | ≤ 10 min, RAM ≈ full-grid labels + one DEM band |
| 3 | 3B-A | `benchmark` | `benchmark/verdict.json` | ≈ 0.3 CPU-h (r1 estimate, unmeasured) |
| 4 | 3B-B0 backfill | `backfill_g02` | only if plan `PENDING/READY` and `fits_budget`; writes `geo_2`, `disc_official_2` | ≤ 2 × 1698 s upper bound (measured full unit evals); raster-only unmeasured |
| 5 | 3B.P again, then 3B-B | `geology` | S1–S3 RUN instead of BLOCKED | ≈ 0.5 CPU-h |
| 6 | 3B-C, 3B-D | `robustness`, `hybrid` | ablation + decision | ≈ 0.4 + 0.5 CPU-h |
| 7 | 3B-E | `selection` (`export` False first) | `selection/decision.json`; gate reasons | minutes |
| 8 | 3B-E again | `export` True | only if `replace_champion` and `slot_worthy`; validated GeoTIFF, logged sha256 | full-region features (+ `g02_full` if G02 is used: heaviest, unmeasured) |
| 9 | 3B-R | `report` | `reports/*` | minutes |

Stop when the start-up check raises, when preflight reports FAILED or a BLOCKED evaluation contract, when stage A
says `w2_gain_is_ranking: false` (see §5), or when the `wb` budget is spent. Nothing is uploaded by code.

## 5. Experiment decisions

* **After A**:
  * Read `ranking_gain.csv`. A ranking gain of at least 0.005 that improves at least 2/3 zones means the W2 gain is
    ranking. Otherwise it is mostly representation and budget, and the cheaper move is to re-export the fallback / W2
    at the matched best cell.
  * Compare `cell:w2_submitted_file` with `cell:w2_disc_choice`.
* **G02 backfill is justified** when all of these hold:
  * the attachment decision is `PARTIAL`, not `INCOMPATIBLE`;
  * input drift is ok;
  * the estimate fits the `wb` budget left;
  * W2 zone 2 (0.153414) is still the weakest zone in A.
* **C and D are justified** when the B decision is final (ACCEPT or REJECT, not BLOCKED). D additionally needs
  `system_coverage.csv` to show fallback-only systems (r1 recorded 3 lost).
* **A submission is justified** when stage E's gate passes:
  * discovery DTI > max(0.224640, matched W2);
  * worst zone ≥ 0.153414;
  * geographic DTI ≥ 0.1758;
  * provenance clean, parity has no BLOCKED row, validation is clean;
  * and the gain is +0.010 before using a leaderboard slot.

## 6. Ranked backlog (plan only; no measured gains)

| id | experiment | expected leverage | cost | needs | leakage risk | stop |
|---|---|---|---|---|---|---|
| R1 | Candidate-pool recall: share of withheld systems reachable in the 2–5 % pools of B0 / B1 / W2 | decides whether reranking can matter at all | < 0.2 CPU-h (saved maps) | stage A | none (diagnostic) | < 50 % reachable → stop reranking work, go to R2 |
| R3 | Fallback-only vs W2-only systems: geometry, orientation, known distance | targets D1 | < 0.2 CPU-h | A + C diagnostics | diagnostic only | no consistent pattern |
| R2 | Graph tracing / endpoint continuation from W2 crests | raises recall beyond the pool | 1–2 CPU-h | R1 low recall | label-free if rules are fixed in advance | no gain over D2 at matched area |
| R5 | Full-region parity (numeric) + calibration of the exported cell | protects the leaderboard | < 0.5 CPU-h | preflight numeric | none | Spearman ≥ 0.6 everywhere |
| R4 | G01 / G02 / DEM label-safe fusion (new predeclared stage) | unknown | 1–3 CPU-h + `g02_full` | B result, DEM coverage | low (label-free maps) | B shows no G02 gain |
| R6 | M9 recovery (cube key `9ef4f2149217`, `m9_xmod_mag_grav_tol@300m_1km.npy`) | unknown | GPU | repairs first | — | W3 stays off until repaired |

## 7. Paste-ready summary for the next session

> P6 boost r2 is on local branch `p6-boost`, uncommitted on top of `4fa5b48`. `origin/main` is `453bde3`, and
> `origin/p6-boost` does not exist. Nothing has been executed.
>
> * r2 adds `gems_bench/p6/boost_audit.py`: preflight, contracts, provenance and reports.
> * r2 repairs 13 integration defects (`docs/p6/P6_BOOST_HANDOFF.md §2`).
> * The notebook must be patched with `scripts/p6/patch_p6_boost_notebook.py` before the commit.
>
> Next steps: run the §4 local commands, then Colab 0 → 3B.P → 3B-A. Read `preflight/*.json` and
> `benchmark/verdict.json` before claiming any number. The G02 zone-2 backfill is opt-in.
>
> Recorded baselines:
>
> | candidate | disc | worst zone | geo |
> |---|---:|---:|---:|
> | fallback | 0.161576 | — | 0.182832 |
> | W2 | 0.224640 | 0.153414 | 0.180843 |
>
> The uploaded W2 file is `skeleton_1px @ 5 %`, a cell no nested evaluation selected.
