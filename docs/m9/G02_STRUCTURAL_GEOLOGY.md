# G02 — geological fault evidence: displacement, truncation and structural continuity

Research experiment on top of M9 caches and G01 code (not an M9 production stage). Entry point:
`notebooks/colab/GEMS_G02_STRUCTURAL_GEOLOGY_EXPERIMENT.ipynb`. Configuration: `configs/m9/g02_structural_geology.yaml`.
Code: `gems_bench/m9/research/{geology_displacement,geology_truncation,geology_continuity,geology_network,
geology_hypotheses,g02_metrics,g02_figures,g02_io,g02_experiment,g01_review}.py`. Tests:
`tests/test_g02_structural_geology.py`.

**Status: authored, never executed. No G02 (or G01) number exists yet.**

## 1. Why: what distinguishes a fault from other boundaries

High wavelet energy (G01) marks *contrasts*. Faults produce contrasts, but so do lithological contacts, intrusion
margins, dykes, folds, survey seams and erosional edges. What is more specific to faulting is **disruption of
pre-existing structure**:

| signature | fault | lithological contact | intrusive margin | survey artefact |
|---|---|---|---|---|
| fabric continues across the line, **offset** | yes (strike slip / oblique) | no | no | no (unless line shift) |
| persistent ridges **end** along a common line | yes | at unconformities | at the body margin | at a resolution seam |
| lineament **cuts** the regional grain | often | rarely (follows layering) | locally | grid-parallel |
| level **step** across the line, parallel to grain | sometimes (juxtaposition) | yes | yes (closed) | yes (levelling) |
| curved, closed boundary around a compact body | no | no | yes | no |
| straight over many km, several scales | often | sometimes | no | yes |

None of these is diagnostic alone. G02 measures each one, label-free, and tests by held-out evaluation which ones add
information beyond G01. **Apparent displacement is not geological slip**: potential-field pattern offsets cannot
separate strike slip from dip slip on a dipping contact, a change of source depth or a pre-existing bend; slip
magnitude is not identifiable here.

## 2. Evidence families

Conventions (all modules): pixel (r, c) -> E = c px, N = -r py; azimuth clockwise from grid north, axial
(phi = phi + 180), identical to M9's wavelet strike. Oriented operations run on an explicitly rotated grid
(`RotFrame`, rows along the line, columns across it) so profiles, shifts and running means are vectorised and O(1) per
pixel in the profile length. Large wavelengths are processed on a decimated grid (step = lambda / 4). Pixels must be
square within 2 %.

### G02-A displacement (`geology_displacement.py`)
* Signal: signed DoG band-pass of the raw field (RTP magnetic, isostatic gravity) peaking at lambda.
* For each tested azimuth: profiles parallel to the line at -h2, -h1, +h1, +h2 (h1 = lambda / 2, h2 = lambda); NCC
  over |u| <= 1.5 lambda with a bounded shift search (<= 1.5 km and <= 0.75 window).
* **Step vs slope.** Continuous oblique fabric gives shifts linear in normal separation (no intercept); a fault adds a
  step to the across shift only. offset a = s_X - 2 h1 (s_L + s_R) / (2 (h2 - h1)); gain = NCC(s_X) - NCC(prediction).
* Score = NCC_across x NCC_sides x gain x resolvable(a) x (1 - periodicity ambiguity) x side consistency x energy gate,
  max over azimuths and wavelengths. Saved separately: `disp[<field>]:offset_m` (signed; > 0 apparent sinistral),
  `:azimuth`, `:lambda_km`, `:rho_x`, `:ambiguity` (second NCC peak), `:azimuth_ambiguity`.
* Not detectable (score 0, never negative evidence): offsets beyond the search, offsets by a multiple of a periodic
  fabric, fabric nearly parallel to the line, weak signal.

### G02-B truncation (`geology_truncation.py`)
* Ridges = energy percentile >= 0.8 and across-strike maximum, skeletonised; endpoints by crossing number; tip
  direction from the local skeleton centroid.
* Endpoint weight = abruptness (energy drop 0.75 lambda past the tip): fading ridges (weak signal) carry little
  weight; endpoints within 400 m of invalid data are excluded and counted in `trunc[...]:edge_endpoints`.
* Alignment: endpoints whose tips point AT the line from either side, >= 30 deg oblique, counted in two one-sided
  strips and compared with the isotropic local density of the same endpoints (Poisson z). Also saved: energy step and
  fine-minus-coarse (resolution-seam) step across the local strike.

### G02-C continuity (`geology_continuity.py`)
Ridge-position persistence across scales (within G01 tolerances), cross-scale strike stability, along-strike ridge
occupancy, straightness; narrow (<= 1 km) and broad bands scored separately and combined by max (short fine-scale
faults are never averaged away). G01's J is deliberately excluded (no double counting in the incremental test).

### G02-D network (`geology_network.py`)
Hysteresis skeleton of a label-free evidence map (thresholds = top 3 % / 8 % of the unit's valid area), segments
between junction clusters, traced polylines. Per segment: length, straightness, axis, linearity, curvature, attached
map means, junction degree and largest intersection angle, continuation (colinear branch or en-echelon stepover within
1.5 km), regional trend alignment (measured, not scored), fragmentation. Score = evidence x length factor x
straightness x (0.75 + 0.25 continuation): **connectivity is rewarded, never forced**. Vectors: GeoJSON (+ GPKG).

### G02-E competing explanations (`geology_hypotheses.py`)
Support maps for fault (disruption x persistence), lithological contact (step x parallel to grain x no disruption),
intrusive boundary (blob nearby x curved x weak disruption), structurally controlled terrain (surface expression
aligned with the lineament; corroborating) and ambiguous (strong lineament not explained, or artefact flags). Dominant
hypothesis, margin and uncertainty are saved. G02-E = fault + 0.5 ambiguous. Hypothesis **scores**, not probabilities.

## 3. Comparisons

| ID | candidate | how it is chosen |
|---|---|---|
| B0 | simple magnetic gradient (`edge`) | fixed |
| B1 | best G01 magnetic-only map among G01-A, G01-A@lambda | nested on other zones |
| B2 | best G01 fusion among G01-C/D/E/F | nested |
| G02-A..E | the five families (+ [mag]/[grav], [narrow]/[broad], net[source] variants) | fixed |
| `F:<base>+<X>` | combine(base, X) with weight w in {0.25, 0.5} | w nested |
| G02-F | best `F:<base>+<X>@w` over bases, families, weights | nested |
| M9pred | optional existing M9 prediction (`baselines.m9_prediction_npy`) | fixed |

Contrast kinds (`tables/contrasts`): `baseline` (B1 vs B0, B2 vs B1/B0, G01-E vs G01-D), `standalone` (X vs B0,
random, B2), **`incremental` (F:base+X vs base — the decision criterion)**, `field` (magnetic vs gravity), `band`
(narrow vs broad), `network` (G02-D vs network on the edge), `fusion` (G02-F vs B2), `m9_increment`.

## 4. Validation and leakage prevention
* Units: geographic folds (`geo`) and discovery splits (`disc_official`, known faults excluded pixel-exactly) from
  the M7 protocol; domain = `m9.elig(u, 0)` & 400 m interior, identical for all candidates.
* Percentile tables (G01 channels and raw fields) fitted on the unit's training geography; `assert_disjoint` raises
  on any overlap. G02 maps never take a label argument.
* Equal-area selections (exact k pixels) with seeded tie-break noise shared by every map.
* Distance-tolerant precision / recall / F1 / enrichment at 0-1000 m; 20 km spatial blocks; paired block bootstrap.
* Every choice (B1, B2, weights, G02-F) is made on the task's other zones, **excluding zones whose evaluation
  footprint overlaps** (`tables/unit_overlap`). Decision thresholds are fixed in configuration.
* Strata by truth-trace length come from the label raster (no segment attributes exist); region = held-out zone.
* Pixel ranking and lineament geometry (`tables/lineament_metrics`) are reported separately.

## 5. Modes
* **FAST_RESEARCH** (default): 2 units per task, wavelengths 0.8 / 1.6 km, 4 azimuths, 100 bootstrap draws, 3 budgets,
  3 distances, one fusion base. Exploratory: recommendations are always UNVERIFIED (with the provisional verdict).
* **FULL_EVALUATION**: every unit, every cached wavelength, 6 azimuths, 500 draws, all budgets / distances, bases
  G01-C / G01-E / G01-F. Different fingerprint, different folder: the two never overwrite each other.

## 6. Artifacts (`outputs/m9_geofault_production/research/G02_structural_geology/g02_<fp>/`)

| folder | content |
|---|---|
| `config/` | `g02_config.json`, `g01_config_used.json`, `inputs.json` (input fingerprints, scales, halo, warnings) |
| `processed/` | per unit: `blocks_*`, `complement_*`, `strata_*`, `correlations_*`, `lambda_usage_*`, `lineament_*`, `official_*` (CSV + Parquet), `refs_*.json`, `footprint_*.npz`, `unit_*.json`, `done_*.json`; `tmp_<unit>/` memmaps while a unit runs |
| `rasters/units/<unit>.npz` | every map (float16) + domain + interior; labels never stored |
| `lineaments/` | `<unit>_<source>.geojson` / `.gpkg`, `segments_*`, `nodes_*` tables |
| `tables/` | summaries, nested choices, contrasts, per-zone deltas, recommendations, complementarity, strata, correlations, wavelength usage, lineament metrics, official-style, unit overlap, G01 review |
| `figures/` | 15 figures (PNG + PDF + CSV of plotted data) |
| `reports/` | `G02_REPORT.md`, `G01_REVIEW.md`, `summary.json` |
| `manifests/manifest.json` | every artifact with sources, parameters and content fingerprint |

## 7. Decision framework (`g02_metrics.decide`)
Per family, from the incremental contrasts at the primary budget (3 %) and distance (300 m):
* **KEEP** — SUPPORTED on `disc_official` over at least one base and not REFUTED on `geo` for that base; FULL mode;
  >= 3 zones.
* **REJECT** — never SUPPORTED, and REFUTED somewhere or P(delta > 0) < 0.5 everywhere.
* **INVESTIGATE** — otherwise.
* **UNVERIFIED** — FAST mode, missing contrast, or < 3 zones.
G02-F is judged on G02-F vs B2 with the same rule. Plausible-looking maps never produce KEEP.
Multiple comparisons: KEEP over "any of three bases" is lenient; treat a KEEP over one base only as a candidate for
confirmation on new geography.

## 8. Integration (only after a KEEP from FULL_EVALUATION)
Outputs are designed for later use as: independent candidate maps (`rasters/units`), extra feature channels (same
grid, float16), post-processing evidence for M7 / M9 (fusion `F:M9pred+X` is already evaluated when an M9 prediction
is supplied), lineament ranking (`lineaments/segments_*`), continuity refinement (G02-C / G02-D), ensemble members.
G02 never modifies M9, its checkpoints or any submission.

## 9. Known limitations
* Fixed functional forms and constants (set before any run); negative results refute these rules, not the geology.
* G02-A cannot see offsets of periodic fabric by one period, offsets beyond 1.5 km, or fabric parallel to the fault.
* G02-B confuses fault truncation with intrusive / unconformable terminations (G02-E weighs them).
* Network segments are cut at the unit window; junction detection on skeletons can create short spurs.
* Truth-length strata use raster components (crossing traces merge).
* Held-out zones were inspected in earlier GEMS work: not an untouched test set.

## 10. Results (fill from `reports/G02_REPORT.md` after FULL_EVALUATION)

| family | standalone vs B0 | increment over best base | recommendation |
|---|---|---|---|
| G02-A | — | — | UNVERIFIED |
| G02-B | — | — | UNVERIFIED |
| G02-C | — | — | UNVERIFIED |
| G02-D | — | — | UNVERIFIED |
| G02-E | — | — | UNVERIFIED |
| G02-F | — | — | UNVERIFIED |
