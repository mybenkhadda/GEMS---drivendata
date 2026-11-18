# G14.1 static code audit

Scope: `g14-geofaultformer` at `b8a580e` (G14 plus the user's three post-run fixes). Also read: G13 evaluation and
submission, P6 W2 stacker, `gems3.official`, `gems3.representation`, `gems_bench.metric`, `g13.contracts`.
Method: **reading only**. No line of code was executed. "Confirmed statically" means that the code path was
read end to end and the mechanism follows from it. It never means "measured".

## Observed facts (supplied by the user from the Colab run; not reproduced here)

T1_local_cnn (`da3f44d05d84`), 5 219 365 parameters:

| quantity | value |
|---|---|
| inner surrogate, best (geo_1 / geo_2) | 0.851 / 0.839 |
| geo_1 final loss (train / val) | 0.0368 / 0.0392 |
| geographic cross-fitted DTI | 0.0807 |
| worst geographic zone | 0.0752 |
| geographic optimism | 0.0048 |
| operating point | crest_w1 at 8 % |
| official leaderboard | 0.0189 |
| full map probability range | 5.9e-7 to 0.99995 |

The submission was calibrated on the **geographic** scorecard because no discovery scorecard existed.

## Summary of root causes (ranked)

| id | severity | finding | statically confirmed | explains |
|---|---|---|---|---|
| S01 | **Critical** | Training targets are also inputs: known-fault context is built from a map that **contains every training positive** | yes | inner 0.85 vs held-out 0.08; LB 0.019 |
| S05 | **Critical** | Protocol mismatch: geo-only units trained; discovery task requested; submission calibrated on geo truth (all faults, no known mask) | yes | 8 % area; LB transfer failure |
| S03 | High | Inner validation is neither independent nor representative (same geography, its faults in the context, enriched patches) | yes | inner 0.85 |
| S04 | High | Inner "surrogate" is a per-batch mean of ratios on positive-centred crops; not a region DTI | yes | inner 0.85 |
| S06 | High | crest_w1 area accounting: 8 % is the *requested* crest share; dilation makes the real area larger and is never recorded | yes (mechanism), size needs runtime | FP mass |
| S07 | High | The export is binary; DTI uses continuous p; soft or sparse-soft maps were never evaluated | yes | calibration |
| S02 | High | Modality and sensor dropout never drop `known_context`; nothing forces geophysical learning | yes | S01 amplification |
| S08 | Medium | Crests are computed *before* known-fault exclusion; eligible crests are secondary ridges next to masked faults | yes | FP near known faults |
| S09 | Medium | Distance-centreline BCE trains a ~600 m probability band; under S01 the band surrounds known faults and survives pixel-exact masking | yes | LB |
| S11 | Medium | Negative design: background sparsity pushes probability down on lineaments > 1 km from mapped faults, i.e. where discovery targets live | yes (design) | discovery recall |
| S10 | Medium | Final model's context drops faults in the locked TEST fold (`lab & ~ex`), unlike every other region | yes | regional shift |
| S12 | Medium | D4 augmentation of sign-dependent scalar channels (`*:across_asym_d1000`) is not value-transformed | partly (sign convention unverified) | E |
| S13 | Low | Hard-coded 100 m (kernel 3 px, 0.1 km/px, `pixel_m=(100,100)`) | yes, consistent with the verified 100 m grid | none currently |
| S15 | Medium | Run-mode ordering lets EVALUATE work in TRAIN; SUBMIT forbids training flags; submission needs four artifacts and fails with KeyErrors | yes | usability |
| S16 | Medium | `card['disc_rep']` KeyError when only geo units exist; no provisional path | yes | observed error |
| S19 | fixed | Two unpacking crashes (soft_cldice; sensor dropout) fixed in `62f26dd`, `3c56975` | yes | regression tests added |
| S21 | Low | `data.labels` restricted to `valid`; known faults with no raw data are not zeroed by export (scorer masks them anyway) | yes | parity check only |

Geometry: **no static evidence of a raster mismatch**. Training, inference and export share the template grid,
which `g13.contracts.check_grid` asserts at load (CRS, transform, shape, 100 m). The other dimensions in the
repository are the 200 m EDA working grid, 1865 × 1646 (`outputs/eda/EDA_FINDINGS.md`, `gems_bench/data/__init__.py`),
which G14 never uses. A round-trip parity check of the exported file is still required (R01).

---

## Detailed findings

Each finding lists: file and function, relevant code, severity, mechanism, expected DTI effect, static status, runtime
verification, correction.

### S01: known-fault context contains the training targets (Critical)

**Where:**

* `gems_bench/g14/splits.py`:
  * `UnitSystem.geo_unit` (L108–115): `permitted = lab & ~(val(k) | buffer(k)) & ~test` and `labels = lab & (train | inner)`.
  * `disc_unit` (L117–131): `permitted = vis`, `labels = vis & (train | inner)`.
  * `final_unit` (L161–169): `permitted = lab & ~ex`, `labels = lab & region`.
* `gems_bench/g14/multiscale.py`: `UnitInputs.__init__` (L59–70): `known_context(unit.permitted_known, data.valid)`.

```python
permitted = lab & ~(self.sp.val(k) | self.sp.buffer(k)) & ~self.test_exclusion()   # ⊇ lab & (train | inner)
return TrainUnit("geo", k, train, inner, lab & (train | inner), ..., permitted, held)
```

**Mechanism:** for every training positive pixel, `known:dist_km == 0`, `known:density > 0` and the strike channels
equal the target orientation. Learning the identity "dist_km < 0.05 → fault" minimises every loss term. The leakage
audit (`assert_no_leakage`) only checks that **held-out** truth is absent from context. It never checks that the
**training target** is absent from the training inputs, which is the shortcut. W2 (P6) avoided this: its positives
are withheld systems and its context is the retained map (`p6/stack.py` docstring).

**Expected effect:**

* Inner validation: its faults are in the context too, so the surrogate is high (0.85).
* Geographic held-out zone: no context fault within the 13 km buffer, so the model falls back to weak geophysics (0.08).
* Official map: context = all known faults. Peaks sit on known faults, which the scorer masks pixel-exactly. The budget
  then goes to the shoulders of known faults (S09) and to residual noise, so LB ≈ 0.02.

**Static status:** confirmed (code path).

**Runtime verification:**

* R06 counterfactuals with the existing checkpoint (context disabled / neutral / shifted / subset).
* D1: a known-context-only model reaching a similar inner surrogate.
* R02: share of the selected area within 300 m of known faults in `full.npy`.

**Correction:**

* `g14_1.discovery.ConditionalDiscoveryUnit`: context and target fault systems are disjoint, re-drawn each epoch.
* Or the geophysics-only model (M1).
* `g14_1.leakage.target_context_overlap` fails a unit whose training targets appear in its context.

### S05: protocol and calibration mismatch (Critical)

**Where:**

* `config.SplitCfg.protocol = "spatial_cv_embargo"` and `tasks = ("geo", "disc_official")`.
* `EvalCfg.primary_task = "disc_official"`.
* `pipeline.units()` → `units_for(protocol)` trains geo units only.
* `evaluation.score` writes `geo_*` keys only.
* Notebook cell 20: `card[f'{key}_rep']`.

**Mechanism:** only geo maps exist, so the discovery scorecard is empty and the submission had to use `geo_rep` /
`geo_frac`. Geo truth = every mapped fault in the held-out fold (dense; rediscovery, no known mask). Official truth =
*new* faults only, with the known faults masked. An area optimum on dense truth (8 %) is far too large for sparse
truth. `G14Config.validate` never checks that the training protocol produces maps for the primary task.

**Expected effect:** large FP mass on the official map; geo DTI does not transfer.

**Static status:** confirmed.

**Runtime verification:**

* R07 builds discovery units.
* R03 sweeps area on any discovery maps available (G13 `disc*.npy`, P6 W2 `disc_official_*`).
* Compare the area optimum geo vs disc.

**Correction:**

* `g14_1.config` protocols `geo_only | disc_only | geo_and_disc`, with validation of protocol × tasks × primary task
  × available maps.
* `g14_1.submission` labels a geo-only operating point PROVISIONAL.

### S03: inner validation is not independent (High)

**Where:**

* `train.fit_unit` L138: `tg = unit_targets(unit.labels & (unit.train | unit.inner_val), ...)`.
* L168: `MultiScaleDataset(..., unit.inner_val, ..., positive_frac=0.5)`.
* `splits._block_select`: 1-in-5 blocks of 20 km.
* g13 roles: `inner_val` blocks inside the CV training geography.

**Mechanism:**

1. Inner blocks are interleaved with training blocks; faults run continuously across block edges, and spatial
   autocorrelation reaches 12.8 km (G13 EDA).
2. Inner-val faults are in the known context (S01).
3. The combined target raster lets inner-val labels shape training supervision:
   * the soft target within 3 px;
   * `dist_px`, therefore the ignore ring (3 px) and reliable negatives (10 px);
   * orientation weights;
   * endpoint / junction maps.

   This leak is small, but it is real.

**Expected effect:** optimistic inner metric and unreliable checkpoint selection.

**Static status:** confirmed.

**Runtime verification:** R10 / F4: compare enriched, uniform, tiled and full-region inner DTI, and their rank
correlation with discovery DTI.

**Correction:** `g14_1.discovery.separate_targets`:

* training targets come from `labels & train` only;
* inner-val targets from `labels & inner_val` only;
* a label-free boundary embargo of `ignore_ring_px` removes training pixels next to inner-val blocks.

### S04: the inner surrogate is not a DTI (High)

**Where:** `train.evaluate_inner` L243–260 and `losses.dti_surrogate`.

**Mechanism:**

* `1 - dti_surrogate` is computed per batch of 8 crops (128 px).
* Half of the crops are centred on positives.
* FP is counted only inside `inner_val ∩ crop`, with no neighbours across crops.
* The batch values are averaged (a mean of ratios).

The official DTI pools TP / FP / FN over the whole region, including the vast fault-free background.

**Expected effect:** an inflated, unstable number that is not comparable with DTI.

**Static status:** confirmed.

**Correction:** `g14_1.metric_audit.full_region_inner_dti`. It runs tiled inference over the inner-val region, then
computes the official components with `metric.MaskedScorer` (ratio of sums). Enriched patches are kept as a labelled
diagnostic.

### S06: crest_w1 area is under-reported (High)

**Where:**

* `evaluation.operating_tables` L102–110 and `submission.operating_point_map`.
* `gems3.representation.crest_top` and `op_dilate`.

**Mechanism:**

* `crest_top` selects `ceil(frac × eligible)` crest pixels, or all crest pixels if there are fewer.
* A radius-1 disk dilation then grows the 1-px crest to about 3 px across.
* The scorecard records `target_frac` only. The real predicted area, and the share of it inside known pixels, is never
  stored.

**Expected effect:** the submitted area can be about 2–3 × the stated 8 %, adding FP mass.

**Static status:** the mechanism is confirmed; the factor needs the map.

**Runtime verification:** `g14_1.representations.apply_rep` records requested, pre-dilation, post-dilation and scored
area (R03).

### S07: binary export discards the ranking (High)

**Where:** `submission.operating_point_map` (top / crest only) and `g13.dti_opt.render`.

**Mechanism:** official FP = Σ p·(1 − k). A binary map puts p = 1 on every selected pixel. A sparse soft map (top-a
pixels keep their calibrated p) or a power transform can trade FP mass for TP. None of these was evaluated.

**Static status:** confirmed.

**Runtime verification:** R03 / G4 compare raw, power, sparse-soft and calibrated maps under cross-fitting.

### S02: dropout never removes the shortcut (High)

**Where:** `multiscale._radiometric` L242: `names = [m for m in mods if m != "known_context"]`.

**Mechanism:** whole-modality and rectangular sensor dropout exclude known context explicitly, so the model never
learns to work without it.

**Correction:** `g14_1.training` augmentation policies, including `known_context_dropout`.

### S08: crest before known masking (Medium)

**Where:** `evaluation.operating_tables`: `crest_map(np.where(u.ev, s, 0), u.ev)` is computed on the full score, then
known pixels are removed from eligibility in `render`.

**Mechanism:** under S01 the main ridges lie on known faults, which are excluded after crest detection. The remaining
eligible crests are secondary maxima beside or between known faults. Dilation then extends them toward the masked
faults.

**Runtime verification:** R03 G5 masking-order comparison.

### S09: wide probability band (Medium)

**Where:**

* `losses.distance_centerline`: BCE on `soft = max(1 − d/3 px, 0)`.
* `multiscale.unit_targets`.

**Mechanism:** the model learns p ≈ 0.67 at 1 px and 0.33 at 2 px from a trace. On the final map those traces are
known faults; after pixel-exact masking the band shoulders remain the highest-ranked eligible pixels.

**Runtime verification:** R02 probability by distance-to-known-fault curve.

### S10: final context excludes the TEST fold (Medium)

**Where:** `splits.final_unit`: `permitted = lab & ~ex`.

**Mechanism:**

* Inside the locked-TEST fold, the submission model sees no known faults, unlike the rest of the map.
* That makes behaviour heterogeneous across the official area.
* It also contradicts the docstring ("all mapped faults are context at inference").

**Correction:** the final context must be every mapped fault (a test-time input), and the training targets must be
disjoint from it (S01).

### S11: negatives are anti-discovery (Medium)

**Where:**

* `multiscale.MultiScaleDataset.__getitem__`: `neg_weight`, `reliable_neg`.
* `losses.background_sparsity`.

**Mechanism:** every unlabeled pixel more than 300 m from a visible fault is a negative, and pixels more than 1 km
away also feed the sparsity term. Undiscovered faults are, by definition, unlabeled and usually far from mapped ones.
The objective therefore explicitly suppresses them (positive-unlabeled problem).

**Correction:** `g14_1.losses` PU risk (nnPU), reduced negative weights, distance-masked negatives (M5).

### S12: sign-dependent scalars under D4 (Medium, partly unverified)

**Where:** `contract.G13_CHANNELS` (`structural:*:across_asym_d1000`) is registered as `scalar`;
`multiscale.transform_channels`.

**Mechanism:** an across-strike asymmetry has a sign tied to a reference direction. A flip or rotation moves the
pixels but keeps the sign, which can contradict the new geometry. The G13 definition must be checked at runtime
(`g14_1.diagnostics.feature_audit` lists channel kinds).

**Correction:** conservative augmentation policy (rotation-safe channels only, or no flips).

### S13: hard-coded 100 m assumptions (Low)

**Where:**

| file | assumption |
|---|---|
| `g14/config.py` | `kernel_radius_px = 3`, `ignore_ring_px = 3`, `reliable_negative_px = 10` |
| `g14/multiscale.py` | `orient_w = dist_px <= 3` |
| `g14/evaluation.py` | `* 0.1` km per px |
| `g14/splits.py` | `pixel_m = 100` |
| `p6/stack.py` | `known_features(px_km = 0.1)` |
| `gems3/official.py` | `pixel_m = (100, 100)`, `_dilate(sampling = (100, 100))` |
| `metric.py` | default `pixel_m = (100, 100)` |
| `g14/config.py` | `embargo_m = 13000` |

**Status:** correct for the verified grid, because `check_grid` raises unless the pixel is 100 m.
`g14_1.geometry.kernel_radius_px(transform)` derives the radius from the affine transform, and
`assumption_audit()` re-checks every item at runtime.

### S15 / S16: execution-mode friction (Medium)

* `pipeline.MODE_ORDER` gives TRAIN and EVALUATE equal rank.
* `config.validate` rejects `run_mode="SUBMIT"` with `train_enabled=True` (correct), but the notebook required
  toggling five flags across runtimes.
* The submission path needs `final_0` checkpoint + manifest + `full.npy` + scorecard with `disc_*` keys. Each missing
  item produced an exception.

**Correction:**

* `g14_1.config.MODES` named execution modes set compatible flags.
* `g14_1.submission.generate_submission` discovers maps and scorecards, labels provisional calibrations, and fails
  only on genuinely missing artifacts.

### S19: fixed crashes (regression tests)

* `losses.soft_cldice`: `p, t, w = a, b` (two values into three names) was fixed in `62f26dd`.
* `multiscale._radiometric`: `for name, (f, _, _, S) in zip(LEVEL_NAMES, self.views)` (views are 2-tuples) was fixed
  in `3c56975`.

Both are covered by `tests/test_g14_1_regressions.py`.

### S21: labels restricted to the data footprint (Low)

**Where:** `g13.contracts.load`: `labels = (kn >= 1) & valid`.

**Mechanism:** known faults where the template is finite but no raw band has data are not in `data.labels`. Export
zeroes only `data.labels`. The official scorer masks every known pixel, so the score is unaffected.

**Runtime verification:** `geometry.submission_parity` reports positive values on the raw known raster.

## Items verified as correct (no change needed)

* **Metric:** radius 300 m, α 0.2, β 0.8, pixel-exact exclusion that cannot support TP. Same code path as the
  verified `metric.dti_components`.
* **Export geometry:** template profile, float32, NaN outside the template, validated by `g13.submission.validate`.
* **Held-out map construction:** each unit's model predicts only its own held-out zone, with its own context.
* **Locked TEST:** single use, logged.

## Not verifiable statically

* Which G13 features the checkpoint actually consumed. The manifest `channels` field records it; see
  `g14_1.artifacts` and the feature provenance table.
* The size of the crest dilation factor, the share of mass near known faults, and the LB-vs-offline relation. These
  are all R02–R04 outputs.
