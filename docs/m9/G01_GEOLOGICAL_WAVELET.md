# G01: physics-informed wavelet fault discovery

Notebook `notebooks/colab/GEMS_G01_GEOLOGICAL_WAVELET_EXPERIMENT.ipynb` · code `gems_bench/m9/research/` ·
configuration `configs/m9/g01_wavelet_geology.yaml` · tests `tests/test_g01_geology_wavelet.py` · handoff
`docs/m9/G01_SESSION_HANDOFF.md`.

**Status: authored, never executed.** Every result section below is a placeholder until the notebook has run.

---

## 1. How to run it in Colab

1. **Prerequisite (once).** The M9 layer cache must contain the section-03 log-Gabor layers
   (`cache/m9/layers/m9_spec_{magnetic,gravity}_wav_*@<lam>km.npy`) and, for G01-F, the section-02 cross-modal layers
   (`m9_xmod_mag_grav_tol@*`, `m9_xmod_mag_grav_orient@*`). They exist after `m9.s02_features()` / `m9.s03_spectral()`
   in the M9 notebook. M9 does not push `cache/m9` to Drive itself; G01 stages it from
   `<drive_root>/cache/m9/layers` when it is there. If it is in neither place, set
   `G01_CONFIG['build_missing'] = {'wavelets': True, 'crossmodal': True}`: G01 then computes **only** the missing
   layers with M9's own functions (`spectral.wavelet_layers`, `features.crossmodal_layers`) and pushes them.
2. Open `notebooks/colab/GEMS_G01_GEOLOGICAL_WAVELET_EXPERIMENT.ipynb` in Colab (a CPU runtime is enough).
3. Check `CONFIG['ref']` (branch / commit), `CONFIG['drive_root']` (default `/content/drive/MyDrive/GEMS_2`).
4. Run the cells top to bottom:
   * section 0: Drive mount, install, pipeline context, `m9 = pipe.m9(M9_CONFIG)` (no M9 section runs),
     `g01 = G01Experiment(m9, G01_CONFIG)`;
   * section 1 `g01.prepare()`; section 2 `g01.exploratory()`; section 3 `g01.evaluate()`;
     section 4 `g01.metrics()`; section 5 `g01.figures()`; section 6 `g01.report()` + `g01.push()`.
5. After a disconnect: rerun section 0, then any section (finished steps are reused), or
   `G01Experiment.attach(m9).results()` to read a finished run.

Equivalent one-liner after section 0: `g01.run()`.

## 2. Hypotheses

| ID | map | hypothesis | contrasts (primary budget 3 %, distance 300 m) |
|---|---|---|---|
| G01-A | M = mean over scales of the magnetic energy percentile | magnetic structure is independently informative | A vs edge, A vs random |
| G01-B | G, same for gravity | density boundaries carry complementary evidence | B vs edge, B vs random |
| G01-C | (M + G + J) / 3 | joint evidence beats either field alone | C vs A, C vs B |
| G01-D | (M + G + J_D) / 3, J_D = J x strike gate | strike alignment separates coherent structures | D vs C |
| G01-E | D x (0.5 + 0.5 P) | multiscale continuity helps | E vs D |
| G01-F | mean percentile of M9 `mag_grav_tol@*`, `mag_grav_orient@*` | established engineered evidence | E vs F, C vs F, F vs edge |

Each candidate adds exactly one ingredient to the previous one, so each contrast isolates one physical idea. G01-E is
not assumed to be best. No weight is learned.

## 3. Physical justification

* **Wavelet energy.** M9's steerable one-sided log-Gabor filters give, per wavelength λ, the quadrature energy of
  band-limited linear structure, its dominant strike (wavevector + 90°) and its directional coherence. Faults juxtapose
  rocks of different magnetisation / density, so they appear as band-limited, oriented contrasts. So do lithological
  contacts, intrusion margins and dykes: **energy is evidence of a boundary, not of a fault.**
* **Percentiles.** The cached energy is a monotone transform of log1p(E / median E); G01 maps it to percentiles of a
  reference distribution, which makes the fields comparable and bounded. For held-out evaluation the reference is the
  unit's training geography (never the held-out zone).
* **Spatial tolerance.** Magnetic and gravity expressions of one structure need not coincide: gravity sources are
  deeper and broader, the gravity survey is coarser, a dipping contact shifts its anomaly down-dip, and an edge
  observed at wavelength λ is located to about λ/4. G01 therefore allows a lateral offset
  tol(λ) = clip(0.25 λ, 200 m, 1200 m) (metres, converted with the grid's real pixel size) and discounts it smoothly:
  T[x](p) = max_q x(q) exp(-|p - q|² / 2 tol²), searched to 2 tol. Co-location J = max(√(M T[G]), √(G T[M])).
* **Axial orientation.** Strike is axial (θ ≡ θ + 180°). Agreement A = (1 + cos 2(θ_m - θ_g)) / 2 is computed from
  unit doubled-angle vectors. M9's stored vectors are already multiplied by coherence, so G01 uses that once, as
  confidence κ = √(r_m r_g) of the tolerance-smoothed fields; gate = 1 - κ (1 - A). Undefined orientation (κ = 0) never
  penalises; magnetic and gravity ridges are not required to coincide (smoothing at the tolerance scale).
* **Persistence.** A structure rooted at depth shows at several wavelengths. P_mag, P_grav = share of scales with
  percentile ≥ q; P = max(P_mag, P_grav) so a fault visible in one field only is not penalised for the other;
  P_cross (both fields, tolerant) is kept as a separate component.
* **Ambiguity is preserved.** Components saved separately: magnetic boundary (M), gravity boundary (G), co-location (J),
  orientation agreement / confidence / gate, persistence (P_mag, P_grav, P_cross), surface expression (S = THDR
  percentile of detrended elevation; not used by any candidate in G01).
* **Not probabilities.** Outputs are evidence scores in [0, 1]; no probability calibration is performed.

## 4. Evaluation protocol (separate from the official score)

* **Units.** All geographic folds (`geo`) and discovery splits (`disc_official`), independent of the M9 PILOT/FULL
  training profile (`evaluation.zones = all`).
* **Domain.** `m9.elig(u, 0)` (evaluation zone, template finite, known faults excluded on `disc_official`) ∩ pixels
  ≥ 400 m from the data edge (wavelet edge artefacts). Identical for all candidates and baselines.
* **Budget.** Exactly k = round(f |domain|) pixels per map, f ∈ {0.5, 1, 2, 3, 5, 10} %; deterministic tie-break.
* **Distances.** d ∈ {0, 100, 200, 300, 500, 1000} m (Euclidean, real pixel size).
* **Metrics.** precision@d, recall@d, F1@d, enrichment@d over random placement, precision ratio vs simple edge.
* **Baselines.** `edge` = THDR percentile of RTP at σ = 200 m; `random` = seeded uniform.
* **Uncertainty.** Paired spatial-block bootstrap (20 km blocks, resampled within zone, 500 draws, 90 % intervals).
* **Ablations.** Per wavelength (A/B/C/D@λ), fixed tolerances (C|tol = 0, 300, 800, 1600 m) at 3 % budget; best
  variant chosen leave-one-zone-out only.
* **Leakage safeguards.** No label enters any score (the builder has no label argument); percentile tables fitted on
  training geography only (`assert_disjoint`); budgets are area fractions; no tuned post-processing; nested choice for
  any variant selection; fault labels shown only on evaluation figures.
* **Decision rule** (`decision`, fixed before the run): SUPPORTED if Δ > 0, P(Δ > 0) ≥ 0.9 and zone win share ≥ 0.67;
  REFUTED if Δ < 0 and P(Δ > 0) ≤ 0.1; else INCONCLUSIVE.
* **Caveat.** These zones were used throughout GEMS development: they are not an untouched test set.

## 5. Outputs

`outputs/m9_geofault_production/research/G01_wavelet_geology/` (local and Drive):

```text
LATEST.json                          newest run id
g01_<fingerprint10>/
  manifest.json                      every artifact -> sources, parameters, content fingerprint
  config/g01_config.json             resolved configuration
  config/inputs.json                 grid, wavelengths, input layer fingerprints, warnings
  tables/channel_audit.*             present / missing / duplicate / unaligned cached channels
  tables/wavelength_tolerances.*     tolerance and smoothing per scale, metres and pixels
  tables/units.*                     per unit: domain, truth, reference sizes, protocol key
  tables/summary_pooled.*            pooled metrics (+ 90 % block intervals) per task / map / budget / distance
  tables/summary_per_zone.*          per-region results
  tables/hypothesis_contrasts.*      paired bootstrap deltas + verdicts
  tables/bootstrap_vs_references.*   every map vs G01-A, G01-F, edge
  tables/ablations.*, nested_choice.*
  tables/official_style_*            official-style DTI of the same masks (orientation only)
  tables/blocks_all.*                pooled block counts (reusable)
  processed/blocks_<unit>.*, official_<unit>.*, refs_<unit>.json, unit_<unit>.json, done_*.json
  rasters/units/<unit>.npz           all maps + components + domain per unit (window offsets, protocol key)
  rasters/exploratory_full_region.npz  label-free full-region maps + interior mask
  figures/01..12.{png,pdf}, figures/*_data.csv
  reports/G01_REPORT.md, reports/summary.json
```

Unit rasters carry their full-grid window and protocol key, so they can be compared pixel-for-pixel with M7 / M8 / M9
prediction maps of the same units.

---

## 6. Report template

### 6.1 Hypothesis
_State which contrasts were pre-registered (table in section 2) and the decision rule._

### 6.2 Data and inputs
_Wavelengths used: …; M9 cross-modal layers: …; grid / warnings: … (from `config/inputs.json`)._

### 6.3 Results (placeholders)

| task | contrast | metric | Δ | 90 % CI | P(Δ > 0) | zone win share | verdict |
|---|---|---|---|---|---|---|---|
| geo | A vs edge | F1@300 m | … | … | … | … | … |
| disc_official | C vs A | F1@300 m | … | … | … | … | … |
| … | | | | | | | |

_Pooled precision / recall / enrichment per map (`summary_pooled`, 3 %, 300 m): …_

_Distance dependence (figure 11): do gains appear at 0–100 m or only ≥ 500 m? …_

_Budget dependence (figure 10): …_

_Per-wavelength ablation and nested choice: which scale carries the signal? …_

_Tolerance ablation: is 0 m worse than the physically motivated tolerance? …_

_Per-region spread (`summary_per_zone`): …_

_Official-style DTI (orientation only): …_

### 6.4 Geological interpretation
_Figures 9 and 12: successes (which structural settings?), failures (contacts, intrusions, survey seams, edges?)._

### 6.5 Limitations
* Wavelet energy marks any contrast; contacts and intrusions are false positives by construction.
* Fixed equal weights: a negative result refutes this rule, not every fusion.
* Persistence floor 0.5 may penalise single-scale faults.
* Square tolerant neighbourhoods reach √2 farther along diagonals (use `footprint: disk` to check).
* M9 wavelengths assume a 100 m grid.
* Zones previously inspected; block bootstrap conditional on the selections.

### 6.6 Recommendation
_Which G0x next (section 7), and whether any G01 component should enter M9 as an input channel._

---

## 7. Next experiments (documented, not implemented)

Ranked by expected geological value × validation feasibility ÷ compute cost.

| rank | ID | idea | geological value | cost | validation feasibility |
|---|---|---|---|---|---|
| 1 | **G03** | fault vs lithological contact as competing hypotheses: faults truncate / offset fabrics and juxtapose unrelated units; contacts follow fabric and are often curved. Score asymmetry across the lineament (fabric continuity, profile shape, terrain break) | high: attacks G01's main false-positive mode | low–moderate (profiles along candidate normals; M9 `candidates` code reusable) | high: same held-out protocol; contacts unlabeled, so measure FP reduction at equal recall |
| 2 | **G06** | fuse G01 components with M7 / M9 prediction maps (cross-fitted, held-out zones only) | high for the competition | low (maps exist; M9 `fusion` patterns) | high: identical units and protocol keys |
| 3 | **G02** | displaced / truncated magnetic fabrics: detect where oriented fabric terminates or is offset along a line (strike-slip signature) | high (direct fault criterion) | moderate (oriented continuity transforms) | moderate: few labelled offsets |
| 4 | **G04** | structural network coherence: connectivity, intersections, Riedel / conjugate angles among G01 lineaments | moderate–high | moderate (graph extraction; M9 `tracing` reusable) | moderate: system-level recall metrics exist |
| 5 | **G05** | potential-field source-depth consistency: Euler / tilt-depth along candidates; faults show coherent depth steps | moderate | high (Euler windows) and fragile (noise, interference) | low: no depth ground truth |

Start G03 only if G01 shows that at least one of A–E beats `edge` (otherwise the wavelet evidence itself is in doubt).
