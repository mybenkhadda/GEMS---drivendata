# P6 boost: prove the W2 gain, restore G02 structure, robust weak zone, fault-system hybrids

Notebook section **3B** of `notebooks/colab/GEMS_P6_FAST_TRACK.ipynb` · code `gems_bench/p6/boost.py` · config
`configs/p6/boost.yaml` · outputs `MyDrive/GEMS_2/outputs/p6_fast_track/boost/` · tests `tests/test_p6_boost.py`.

**Status (2026-10-27): code authored, nothing run on real data.** Only synthetic tests (84 passed across
`test_p6.py`, `test_p6_boost.py`, `test_g02_structural_geology.py`) and Ruff ran locally. Every number below is either
a recorded W2 / fallback value or a plan.

## 1. Starting point (recorded, origin/main 794a7bd)

| Model | Disc DTI | Worst disc zone | Geo DTI | Status |
|---|---:|---:|---:|---|
| P5 B0+B1 fallback | 0.161576 | — | 0.182832 | valid fallback |
| P6 W2 LightGBM stacker | 0.224640 | 0.153414 (zone 2) | 0.180843 | offline PROMOTE |

W2 trained on 11 features (no G02, no G01). Its nested choices: disc `skeleton_1px @ 1 %` (blend 1.0 in zones 0 / 1,
**0.5 in zone 2**), geo `crest_1px @ 5 %`. Two systems gained and three lost against the fallback.

**Fact found while reading the code:** `stack.full_region` takes `rep` from the discovery choice and `frac` from the
geo choice (`area_frac_source: geo`), so the W2 submission file is `skeleton_1px @ 5 %`. No nested evaluation selected
that cell. Stage A scores it directly (`ranking_gain.csv → cell:w2_submitted_file`,
`selection/leaderboard_compare.csv → disc_dti_exported_cell`). Boosted exports use `area_frac_source: matched`
(rep and frac from the same nested choice).

## 2. Root cause of `W2: no G02 run`

Confirmed from the saved outputs and the code:

1. `G02Experiment.prepare()` writes `LATEST.json` **one level above** the run folders (`set_latest(self.parent, …)`).
2. `G02Experiment.push()` synced only `self.store.root`; every `[sync out]` line of the G02 notebook names
   `…/g02_23686a63cb`. `LATEST.json` stayed on the G02 runtime's local disk.
3. The P6 runtime staged the run folder, but `latest_run()` found no `LATEST.json`, so `attach()` raised
   `no G02 run … (known: ['g02_23686a63cb'])`. The run itself was listed and complete.

Fix: `g02_io.resolve_run()` resolves an explicit id first, then `LATEST.json`, then the newest complete run folder.
`attach()` uses it and records `attach_info`, and `push()` now also pushes `LATEST.json`. The boost config names the
run explicitly (`geology.g02_run_id: g02_23686a63cb`).

**A second blocker remains after the fix.** The run is FAST_RESEARCH with `max_units_per_task: 2`. It has rasters for
`geo_0, geo_1, disc_official_0, disc_official_1` only, so **zone 2, the weak zone, has no G02 map in either task**.
W2 (and boost) use a feature only when every unit has it, so S1–S3 are `BLOCKED` until zone 2 is backfilled.
`G02Experiment.unit_raster(u)` (opt-in: `BOOST_CONFIG = {'geology': {'backfill_missing': True}}`) computes only the
per-unit raster, reusing the run's own config, its training-geography references and its NPZ layout. It adds no
metrics and changes no verdict. The raster metadata carries `backfill: True`. Measured full unit evaluations took
486–1698 s each. A raster-only backfill is cheaper (unmeasured).

`stack.run` now refuses to proceed when `stack/candidate.json` exists with a different feature list. Without that, a
W2 re-run after the attach fix could silently pair the reused 11-feature maps with a G02 feature list.

## 3. Plan: leverage, cost, dependencies, stop rules

| Stage | Question / leverage | Est. CPU | Depends on | Stop / accept rule |
|---|---|---:|---|---|
| A benchmark | Is +0.063 ranking or 1 % skeleton? Decides every later claim | 0.3 h | W2 maps, CNN OOF, M9 selection | `w2_gain_is_ranking` needs ranking gain ≥ 0.005, ≥ 2 / 3 zones improved, W2 > fusion in ≥ 60 % of matched cells |
| B geology | G02-C / G02-A / G01 persistence on top of W2 | 0.5 h (+ ≤ 1 h backfill) | A; G02 rasters for **all** units | accept: Δdisc ≥ +0.005 vs S0, worst ≥ S0 − 0.005, geo ≥ 0.1758; else BLOCKED / REJECT |
| C robustness | Zone 2 (0.153) without zone-specific rules | 0.4 h | B decision | accept: disc > base, worst ≥ base, geo ≥ 0.1758 |
| D hybrid | Recover the 3 lost systems without new FP | 0.5 h | C decision | accept: disc > base, worst ≥ base, FP weight ≤ +10 % at the same cells |
| E selection | Replace W2? Spend a slot? | 0.1 h (+ export) | A–D | disc > max(0.224640, matched W2), worst ≥ 0.153414, geo ≥ 0.1758, provenance + validation clean; slot needs +0.010 |

Budget: workstream `wb` = 4 CPU-h (`fast_track.yaml`, global CPU cap 20 h now fully allocated). `BudgetExceeded` stops
a stage once the cap is spent. RAM: same order as W2 (16–24 GB, unmeasured). Six units of raw features stay cached in
memory between stages. Use a high-RAM CPU runtime. No GPU.

## 4. One protocol for every candidate

`Boost._eval_q` is the only scorer. It uses the unit's evaluation mask, `m9.elig(u, 0)` as the budget domain, the M7
representations (`crest_1px`, `skeleton_1px`), `m7.protocol.score_row` (official components) and
`m7.protocol.system_hits`. Grid: fracs {0.5, 1, 1.5, 3, 5} %, blends {0.5, 1.0}.

* **Fixed cells** (`matched_comparison.csv`) are unselected: pooled DTI, worst / best zone, weighted
  precision / recall, FP and FN shares of the DTI denominator, area, length, fragmentation, systems.
* **Nested** numbers come from `gems3.metric_opt.nested_select`. Each zone's (rep, frac[, blend]) is chosen on the
  other zones and scored on the held-out zone. No quoted number is selected on its own test zone.
* A **hybrid** never exceeds the area of its base map at the same (blend, rep, frac). For `skeleton_1px` that area is
  the post-skeleton pixel count, not the nominal budget.
* **Variants share W2's sampled rows and weights** (`stack/features/rows.parquet`); only columns change. `rows(check=
  True)` recomputes W2's own columns from the current features and refuses to continue on any drift. New columns
  come from `stack.assemble`, exactly as at prediction.
* A missing feature is never zero-filled (`KeyError` / `BLOCKED`). Leakage guards come from `stack`: CNN provenance,
  G02 metadata (task, zone, window, `labels_stored`), permitted known map, buffered leave-one-zone-out training.

## 5. Stages

**A (`a_benchmark`).** Rankers: fusion fallback (champion per-zone `w`), CNN B0, CNN B1, W2, fixed 0.5 hybrid, plus
the published fallback's held-out maps. Nested spaces per ranker: crest only, skeleton only, rep + frac, and
`w2_space` (blend + rep + frac, W2's own space on the matched fracs). `ranking_gain.csv` is a ladder:

* published → matched grid
* representation (crest → crest + skeleton)
* ranking (fusion → W2, at crest, at skeleton, and both)
* hybridization (W2 → W2 space)
* total
* matched-cell deltas, and the three named cells: W2 disc choice, W2 geo choice, W2 submitted file

**B (`b_geology`).** Attach audit per (feature, unit): raster present, key present, backfill flag, `labels_stored`,
window match, normalisation reference (`processed/refs_<unit>.json`: training geography), source channels, support
halo, wavelengths, resampling (none) and full-region availability.

* S0 is W2 (its maps are reused).
* S1 adds G02-C; S2 adds G02-A; S3 adds G01 persistence `comp:P`.
* G02-F is **excluded**: the fusion is evaluated lazily with a label-selected weight and never persisted, and its
  components are already candidate features.

**C (`c_robustness`).** Diagnostics per zone (labels used for diagnosis only):

1. Truth mean rank of B0, B1, fusion, W2 and the base.
2. Feature quantiles and KS distance against the other zones.
3. Known-fault distance at truth vs evaluation pixels.
4. Positive / negative / ambiguous row composition and truth density.
5. Known density and G02-C mean.
6. Prediction length and fragmentation.
7. Orientation total-variation distance to the truth.
8. Distance of predictions to truth and to permitted known faults.
9. Systems against the fallback.

Two predeclared variants:

* **C1** ranks known-fault distance, density and tip distance inside each unit (label-free).
* **C2** sets the known-geometry group missing together on half the training rows.

**D (`d_hybrid`).**

* **D1** keeps the base map's top 75 % and adds fallback-only lines: lines with no pixel within 300 m of the
  full-budget base map, ordered by mean (fusion, CNN disagreement, G02-C). Remaining base lines fill any leftover area.
* **D2** reselects a 2× pool by a fixed weighted rank of mean score, length, linearity, endpoint continuation support
  and G02-C. Segments shorter than 5 px are down-weighted (× 0.5), never removed. Junction pixels are restored when
  the budget allows. Without G02-C the geology term is dropped.
* **D3** compares the precision of low- vs high-confidence segments in low vs high geological context (G02-C, else
  magnetic–gravity strike agreement). It is a diagnostic only.

**E (`e_select`).** Candidates are the fallback, W2 (recorded plus matched) and each stage's accepted choice.

* Gate (`boost_gate`): disc > max(recorded W2, matched W2), worst ≥ 0.153414, geo ≥ 0.1758, provenance clean
  (including the base stage of a hybrid), validation clean.
* After PASS only: refit on all primary rows, then write full-region features. G02 maps come from `stack.g02_full`,
  resumable tiles and the heaviest step.
* The GeoTIFF is written with `m7.export._write` (validated, never overwritten) and logged with its sha256 through
  `probes.update_logs`.
* `leaderboard_compare.csv` lists fallback / W2 / boosted with sha256, offline numbers and any typed public score.
  **No upload is automatic.**

## 6. Colab execution order

1. Push branch `p6-boost` (the notebook's `CONFIG['ref']` now points to it). Use a CPU high-RAM runtime.
2. Section 0 (setup), then skip to **3B**. W0–W2 cells are not re-run: W2 is frozen.
3. 3B.0 `bst.stage()` must report `candidate.json`, `rows.parquet` and 6 maps.
4. 3B-A → read `benchmark/verdict.json`.
5. 3B-B with `backfill_missing: False` first. It writes the attach audit; S1–S3 are expected BLOCKED (zone 2). To
   evaluate geology, set `BOOST_CONFIG = {'geology': {'backfill_missing': True}}` and re-run 3B-B. This backfills
   `geo_2` and `disc_official_2` into the G02 run folder and pushes them.
6. 3B-C, then 3B-D.
7. 3B-E. If `replace_champion` and `slot_worthy`, upload the file by hand and type the public score into
   `reports/p6/leaderboard_log.csv` (sha256 key).
8. W3 stays disabled (`RUN_W3 = False`).

Restart: after a disconnect run section 0 and 3B.0, then any stage. A stage whose fingerprint is unchanged reloads
from `boost/<stage>/` and recomputes nothing. Variant maps are keyed by their own fingerprint (`maps/<name>_<fp8>/`).

## 7. Artifacts (`outputs/p6_fast_track/boost/`)

* `benchmark/`
  * `grid`, `matched_comparison`, `per_zone`, `nested_selection`, `nested_per_zone`, `ranking_gain`,
    `system_coverage`, `inputs_provenance` (all `.csv`)
  * `dti_by_area.png` / `.svg` and `dti_by_area_data.csv`
  * `verdict.json`, `manifest.json`
  * The brief's `dtI_by_area` is written as `dti_by_area`.
* `geology/`
  * `g02_attach.json`, `g02_audit.csv`, `availability.csv`, `row_consistency.csv`
  * `grid`, `matched_comparison`, `per_zone`, `ablation`, `feature_importance`, `systems`, `crossfit_audit` (all `.csv`)
  * `model_manifests.json`, `maps/<variant>_<fp8>/*.npy` + `importance.csv`, `decision.json`, `manifest.json`
* `robustness/`
  * `zone_ranking`, `feature_shift`, `known_distance`, `composition`, `prediction_geometry`, `systems` (all `.csv`)
  * `zone_diagnostics.png` + `_data.csv`
  * the same ablation / decision files as geology
* `hybrid/`
  * `grid`, `matched_comparison`, `d3_geological_context`, `segment_selection` (all `.csv`)
  * ablation / decision files
* `selection/`
  * `candidates.csv`, `leaderboard_compare.csv`, `decision.json`
  * after PASS: `models/<name>_full.joblib`, `maps/<name>_full_score.tif`, and the submission in
    `p6_fast_track/submissions/`

Every `manifest.json` holds the fingerprint, config, P6 config fingerprint, inputs (W2 map / rows sha256, G02 attach
info), decisions, code provenance, seconds and reload instructions. CSVs gain Parquet copies when `save_parquet` is set.

## 8. Remaining blockers and unverified points

* Zone-2 G02 rasters do not exist. S1–S3 (and any G02 term in D) are BLOCKED until the opt-in backfill runs. Backfill
  cost is unmeasured for raster-only mode.
* The G02 run is FAST_RESEARCH: 2 wavelengths, 4 azimuths. A backfill reuses that config, so all units match, but it
  is not the FULL_EVALUATION definition.
* A G02 feature in the full-region export needs `stack.g02_full` over the whole grid (unmeasured; G02-A dominates).
* The W2 training–inference shift remains: discovery-split CNN maps in training, OOF / full maps at inference.
  G02 references are fitted on training geography per unit but on the whole footprint at full region.
* The matched grid omits fracs 0.02 / 0.075, so `w2_space` may differ slightly from the recorded 0.224640. The gate
  uses the larger of the two.
* With three zones, "worst zone" is a single number. Small differences (< 0.005) are within noise.
* W3 needs its M9 cube key (`9ef4f2149217`) and `m9_xmod_mag_grav_tol@300m_1km.npy` repaired separately.

## 9. Next decisive metric

`benchmark/verdict.json → w2_gain_is_ranking` and `ranking_gain.csv (ranking vs representation)`. Then
`geology/ablation.csv` after backfill: S1 Δdisc and zone-2 DTI.
