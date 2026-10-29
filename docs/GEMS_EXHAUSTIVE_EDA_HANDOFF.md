# GEMS exhaustive geological discovery EDA — handoff (code rev gde.1-r1)

**State: authored, never executed.** No figure, table, statistic or verdict exists yet. Everything below describes
what the code *will* compute when it runs in Colab; nothing here is a result.

## How to run

1. Commit and push `gems_eda/discovery/`, `tests/test_geological_eda.py`,
   `notebooks/colab/GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb` and this file. Pin `CONFIG['ref']` (and optionally
   `EXPECT['commit']`) to the pushed SHA.
2. Open the notebook in Colab, set `mode`:
   * `QUICK_LOOK` (start here): 600 fault stations, 100 bootstrap draws, scales 0.5 / 2 / 10 km.
   * `FULL_EDA`: 5000 stations, 500 draws, all scales 0.2–20 km.
   * `RESUME`: FULL_EDA settings, validated caches always reused.
3. Run top to bottom. Stage 00 runs the synthetic self-tests and `pytest tests/test_geological_eda.py`; a failure
   (`TESTS_FAILED`) blocks every downstream stage. A CPU runtime is enough: no stage trains or runs a network
   (the notebook metadata requests a GPU; switch the runtime to CPU if you want to save quota).
4. Outputs: `/content/drive/MyDrive/GEMS_2/outputs/geological_discovery_eda/` (synced to Drive after each stage).

## Architecture

`gems_eda.discovery.run.ExhaustiveEDA` subclasses `gems_bench.reset.run.FaultDiscoveryReset` (itself the GeoMoE
`DiscoveryGeoMoE`), so validated stage caching, dependency blocking (BLOCKED_UPSTREAM / NOT_EVALUATED), stage
integrity, Drive sync, the frozen B0 / B1 / W2 engine, honest zone-stitched maps and append-only reviews are reused
unchanged. Data stages read the three competition rasters at native 100 m through `gems_eda`
(`load_feature_stack(max_dim=None)`), never the engine.

| module | content |
|---|---|
| `core.py` | config + modes, Cliff's delta / AUC, cluster bootstrap over fault systems, status rule, regional robustness, axial statistics, figure `Registry` (PNG + SVG + data CSV + interpretation JSON + `FIGURE_CATALOG.csv`) |
| `data.py` | data loading, inventory, label-free domains, fault stations, matched controls, profiles, station / structure metrics, crests, label-free lineaments |
| `viz.py` | map panels in UTM km, scale bars, ECDF, effect bars |
| `parts_ab.py` | A landscape F001–F012, B geometry F013–F022 |
| `parts_cd.py` | C signatures F023–F036, D scales F037–F047 |
| `parts_efg.py` | E cross-modal F048–F059, F label bias F060–F069, G regional F070–F078 |
| `parts_hi.py` | H frozen-model forensics F079–F090, I candidate atlas F091–F098, DEM pilot F099 |
| `synthesis.py` | dashboards D1–D10, findings, hypothesis register, modeling decision, reports, executive summary |
| `selftest.py` | 20 synthetic self-tests (also run by pytest) |

Stages: `00_tests → e01_inventory → e02_landscape (A) → e03_geometry (B) → e04_profiles → e05_signatures (C) →
e06_scales (D) → e07_crossmodal (E) → e08_label_bias (F) → e09_regional (G) → 01_attach → 03_scenarios →
e11_forensics (H) → e12_candidates (I) → e13_dem (opt-in) → e14_integrity → e15_synthesis`. A figure that raises is
recorded in `manifests/figure_failures/<stage>.json`; the stage fails only when more than half of its figures fail.

## Statistical contract

* Matched controls: same 3 x 3 region and physical setting (terrain-slope x magnetic-gradient terciles), 2–10 km
  from mapped traces (sensitivity 1–10 and 5–20 km), falling back to region then global (recorded as
  `match_level`). Controls are *unlabelled*, not confirmed non-faults.
* Effects: Cliff's delta with a cluster bootstrap over fault systems (mapped-fault connected components; controls
  move with their fault). Region-level tests (lineaments) cluster over regions and say so.
* Status: SUPPORTED when the 95 % CI excludes ±`min_effect` (0.10); NOT_SUPPORTED when the CI lies inside it;
  otherwise INCONCLUSIVE (including fewer than 5 groups). DESCRIPTIVE figures make no test claim; NOT_AVAILABLE
  figures write no image and no placeholder data.
* Every interpretation sentence is assembled from the figure's own `Result` (status + numbers); none is hardcoded.
* Circularity: lineaments and the candidate atlas are built from label-free crests; mapping status is attached
  afterwards for analysis only. F077 selects the signature on the other regions and tests it on the held-out one.

## Model forensics (part H)

Uses `reset.atlas.honest_maps` (held-out V3 zone ranks of B0 / B1 / W2 + the frozen submitted W2 binary map). Each
mapped branch inside the evaluated zones is classed `detected` (≥ 50 % within 300 m), `localisation_failure`
(≥ 50 % within 2 km), `missed_ranked_high` (mean held-out W2 rank ≥ 0.9) or `detection_failure`; frozen-W2
components > 300 m from mapped traces are `near_fp_0.3_2km`, `far_fp_with_lineament` or `far_fp_no_evidence`.
F087 decomposes the official components per V3 zone (`E.score`) and per region (`m7.protocol.block_components`).
Outputs: `tables/model_failure_taxonomy.csv`, `datasets/model_error_cases.parquet`.

## Candidate atlas and reviews (part I)

Categories A–G (unmapped multi-family lineaments, unmapped single-family, far W2 predictions with lineament support,
missed mapped branches, suspected non-faults (terrain-only), short fragments, random controls), seeded, region
round-robin, ≥ 5 km apart (`reset.atlas.select`). Per candidate: `atlas/panels/<id>_evidence_only.png` (no mapped
faults, no model output) and `<id>_diagnostic.png`; `atlas/candidate_catalog.parquet`, `atlas/atlas_index.csv`,
`atlas/candidate_geometries.gpkg`, `atlas/review_template.csv`, `atlas/atlas_index.html`. Reviews go through the
widget, `E.record_review` or `E.import_reviews` into `atlas/reviewed_annotations.csv`, which is append-only and never
overwritten. Only EVIDENCE_ONLY reviews citing no mapped-fault / model source count as INDEPENDENT_REVIEWED.

## Modeling decision

`reports/MODELING_DECISION.md` answers 15 questions from the cited figures and derives one recommendation by fixed
rules (`synthesis.decision`): REPRESENTATION_AND_BUDGET (localisation + high-rank misses exceed pure detection
failures), NEW_EVIDENCE_FEATURES (missed branches carry evidence comparable to detected ones), LABEL_REVIEW_AND_PU
(unmapped lineaments resemble mapped ones), KEEP_W2_AND_REVIEW_CANDIDATES, or INSUFFICIENT_EVIDENCE (forensics not
run). Each comes with an invalidation criterion. The public W2 score (0.0942) is an external reference only, never a
tuning target; nothing is uploaded to DrivenData.

## Known limitations / first things to check in the Colab run

* The stage-00 pytest has never run; expect to fix small API mismatches on first execution (the self-test details
  in `audit/selftests.csv` name the failing function).
* Part H / I W2-derived content needs `01_attach` and `03_scenarios`; without the factory they are
  BLOCKED_UPSTREAM and categories C / D of the atlas are recorded as unavailable.
* The honest-map grid must equal the EDA grid; a mismatch raises (never resampled silently).
* DEM: `dem.enabled=False` by default; when enabled only a cached M7 mosaic / cached tiles are used, never a full
  download. Missing coverage is NOT_AVAILABLE, never evidence of absence.
* Crest thresholds (top 3 %), the 300 m / 2 km / rank 0.9 forensic cut-offs and the 5 km domain blocks are analysis
  choices recorded in each interpretation's limitations.
