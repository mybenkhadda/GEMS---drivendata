# G14 repository audit

**How this was produced:** static reading of `main` at `da9d17a` (code, docs, configs) and of the outputs saved
inside `notebooks/colab/GEMS_G13_GEODISCOVERY_MASTER.ipynb` (a real Colab run, commit `a29bc2d`, Tesla T4, 51 GB
RAM). Nothing was executed. Every number below is quoted from a saved output or a document, with its source.
**None is a leaderboard score unless labelled "public".**

## 1. What exists

| pipeline | code | status (from docs / saved outputs) |
|---|---|---|
| Official metric | `gems_bench/metric.py`, `gems3/official.py` | verified against an independent KD-tree reference (`tests/test_gems3.py`) |
| G05 / G05.1 candidate recovery | `gems_bench/g05_1/` | public score **0.0942** recorded for a G05 submission (`docs/g05_1/README.md`) |
| G06 geological feature store | `gems_bench/g06/` | store `g06-cda34ef1f1` referenced by G07 |
| G07 multiscale / DEM discovery | `gems_bench/g07/` (21 modules) | "authored, NOTHING executed" (`docs/g07/SESSION_HANDOFF.md`) |
| M9 GeoFault production | `gems_bench/m9/`, `docs/M9_GEOFAULT_PRODUCTION.md` | B0 / B1 CNN maps (`P5_B0_gated`, `P5_B1_bgw`) reused by P6 |
| P6 W2 stacker + recovery | `gems_bench/p6/` | ran on real data; nested disc **0.242** (`C_w2_skeleton`), geo 0.106, META 0.2246 (`docs/p6/P6_RECOVERY_TO_BOOST_AUDIT.md`) |
| P7 GeoFaultFormer | `gems_bench/p7/` | CNN + window attention + ConvGRU ladder; "code authored, nothing executed" (`docs/p7/P7_GEOFAULTFORMER_HANDOFF.md`) |
| G12 deep signatures | `gems_bench/g12/` | XGBoost on 90+ engineered features |
| **G13 GeoDiscovery** | `gems_bench/g13/` | ran on real data (FULL_RESEARCH) up to `fusion`; `dti_opt` never finished (see §3) |

## 2. G13 audit (verified from code)

| item | G13 implementation |
|---|---|
| Data sources | `gems-geodawn-numerical-features.tif` (19 bands), `existing_faults.tif`, `example_submission.tif`, found by content (`g13/contracts.discover_inputs`); band names canonicalised from descriptive GDAL text (`canonical_band_name`, commit `0c8b5fb`) |
| Raster inventory | `band_inventory.csv`, `coverage_matrix.csv`; grid 3730 x 3292 at 100 m, EPSG:32611; 60 894 mapped fault px (1.17 % of valid) |
| Features | 297 label-free features in 10 families (FFT potential-field products, multiscale Hessian / structure tensor, monogenic phase, Gabor, cross-modal, regional z, validity); robust-scaled float16 memmaps (`g13/features.py`) |
| Feature selection | quality rejection (coverage, degeneracy, nodata-boundary dependence, |Spearman| >= 0.98 duplicates) and A10 = top XGBoost importances |
| Targets | pixels within 1 px of a mapped trace (binary), XGBoost |
| Negative sampling | uniform beyond a 300 m ignore ring; PU variants: distance-stratified, soft negatives, hard negatives (`g13/models.draw_sample`) |
| Normalisation | global label-free robust scaling of every feature over all valid pixels (transductive on inputs, label-free) |
| Augmentation | none for tabular models; D4 + axial orientation transform in the optional U-Net (`g13/deep.py`, never run) |
| Splits | `g13/splits.py`: `folds.make_geo_folds` (4 folds, 20 km blocks, 13 km buffer), one fold = locked TEST (fold 0 in the run), 3 discovery splits (systems linked within 1 km) inside the CV region |
| Models | XGBoost / LightGBM per unit (3 CV folds, 3 discovery splits, 1 submission model) |
| Losses | log-loss (trees) |
| Validation metrics | `disc_official` and `geo` DTI at cross-fitted area budgets (`g13/evaluation.crossfit`), block bootstrap |
| Candidate extraction / segment recovery | `g13/topology.py`: top-a mask, small-component removal, orientation-aware gap linking (reused `postprocess.topology`), skeleton graph, classification vs mapped faults |
| Output formats | `submissions/*.tif` (template grid, float32, NaN outside template, 0 on known pixels) + JSON validation |
| Artifacts | `/content/gems_g13_work` (local) mirrored to `MyDrive/GEMS_2/outputs/G13_GEODISCOVERY`; **features and most predictions are not mirrored** (`Workspace.PERSIST_EXCLUDE`) |
| Drive / Colab dependencies | Drive mount, `drive_root` discovery, GPU optional (XGBoost CUDA if torch sees CUDA) |

## 3. What the real G13 run shows (saved outputs, offline proxies)

| model | geo proxy | disc proxy | chosen area |
|---|---:|---:|---|
| S0 raw XGBoost (19 bands) | 0.0873 | 0.0205 | 12 % (grid edge) |
| S1 all 316 features | 0.1308 | 0.0490 | geo 8 %, disc 12 % (edge) |
| A10 compact | 0.1275 | **0.0546** | |
| A11 complementary | **0.1331** | 0.0505 | |
| A4 terrain only | 0.1213 | | |
| F_fixed_weights (cross-fitted, disc) | | 0.0851 | weights (1, 0, 0) |
| F_fixed_weights (geo) | 0.1539 | | (0.75, 0.25, 0) |
| F_gated_moe | 0.1443 | 0.0716 | gate not kept |

* **Paired bootstrap vs raw (disc):** terrain +0.025 [0.017, 0.035], structural +0.018, A10 +0.034 [0.016, 0.049];
  magnetic +0.005; gravity, conductivity, depth ≈ 0.
* **PU learning (vs standard):** soft negatives −0.018 [−0.027, −0.010], hard negatives −0.007 [−0.012, −0.001],
  distance-stratified ≈ 0. Incomplete-label re-weighting did not help a tree model.
* **EDA:** 13 of 316 features are "discriminative" against every control; the largest |Cliff δ| is 0.24
  (terrain–depth joint edge, terrain valley). Scale-space gradient AUC tops at 0.595 (detrended elevation, 3.2 km);
  RTP ≈ 0.55, conductivity ≈ 0.52. Variogram correlation lengths reach the 12.8 km maximum lag for every band.
* **Regimes:** k = 8 chosen, but seed stability (ARI) never exceeds 0.52, below G13's own 0.75 rule (fallback).
* **dti_opt** printed "running" and the saved notebook stops there: the 171-representation tournament over full-grid
  units did not complete in the session (cost, not an error message).
* Area budgets at the 12 % edge mean the optimum of weak models may lie above G13's grid.

## 4. Diagnosis that shapes G14

1. **The strongest honest model in the repository is P6 W2** (disc 0.242, Phase-5 units), whose backbone is CNN
   probability maps plus the geometry of the *permitted* known-fault map (`p6/stack.known_features`). A pixel
   LightGBM scored 0.090 there (`p6/stack.py` docstring). G13 deliberately used no known-fault input and reached
   0.055–0.085. `existing_faults.tif` is a test-time input, so permitted known-fault context is legitimate and G14
   uses it per unit.
2. **Signals are weak and broad at 100 m** (AUC ≤ 0.6, 12.8 km correlation lengths): spatial context beyond a local
   patch is plausibly where information lies, which motivates the regional / global views.
3. **P7 already designed a hybrid CNN / attention model and never ran.** G14 does not fork P7. It differs in the
   multi-resolution three-view input, mask-aware modality fusion, permitted known-fault channels, and it reuses the
   G13 data / split / evaluation stack that has actually run.
4. **Scores from different protocols are not comparable** (G05 public 0.0942, W2 nested 0.242 on Phase-5 units, G13
   on G13 units). G14 scores every model, including T0 references, on the same units of one backend.

## 5. Reuse map

| need | reused (unchanged) | new in G14 |
|---|---|---|
| inputs, grid checks | `g13.contracts.discover_inputs/load/check_grid` | channel schema, availability masks, pyramids (`g14.contract`) |
| features | G13 feature store (if present), `g13.signal` | MSDI / GSDI / TLI / CSCI composites, aspect vector, curvature, roughness |
| known-fault context | `p6.stack.known_features` (+ strike of nearest trace) | per-unit permitted maps (`g14.splits.TrainUnit.permitted_known`) |
| splits | `g13.splits.Splits` (locked TEST), `folds.GeoFolds`, Phase-5 context | `UnitSystem`, orientation-aware fault grouping, regime holdout, fold manifest |
| scoring | `gems3.official` units / FastScorer, `g13.evaluation.crossfit/area_table/bootstrap` | `g14.evaluation` (wider area grid, crest representation, diagnostics, locked-TEST log) |
| post-processing | `gems3.representation`, `g13.dti_opt.render`, `g13.topology` | `g14.graph` evidence-gated linking |
| export | `g13.submission.export_candidate/validate` | `g14.submission` guards |
| model, losses, training, inference | — | `g14.model`, `g14.losses`, `g14.train`, `g14.inference` |

Nothing existing was modified. G06, G07, G13, M9 and P6 / P7 files are untouched.

## 6. Verified vs proposed

* **Verified (by reading code and saved outputs):** sections 1–3 and the reuse targets' signatures (each imported
  symbol was located with `git grep`).
* **Proposed / unverified:** every G14 module, test and notebook cell; all expected behaviours in docs/g14.
