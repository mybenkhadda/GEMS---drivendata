# GEMS official metric audit (GEMS 3.0, 2026-10-26)

Status: **structurally complete, nothing executed.** Every statement below is labelled with its evidence. No
official scorer exists in public, so code-level parity with the leaderboard cannot be verified. What exists is
parity with the problem statement plus the organizers' written answers on the forum.

## Sources inspected

| source | what it settles |
|---|---|
| Problem description, `drivendata.org/competitions/306/.../page/967/` | formula, kernel, α/β, file format, external-data rule |
| Forum t/11516 "are known USGS/INGENIOUS faults masked when scoring" (organizer chrisk-dd, 2026-09-16 and 09-21) | known-fault mask, truth set, near-known false positives, corrections |
| Forum t/11550 "Leaderboard aggregation" (organizer, 2026-10-01) | pooling |
| Forum t/11556 "final round rules" (organizer, 2026-10-07) | weighting, exclusion raster in phase 2, label revisions |
| Forum t/11536 "Where do you draw the line" (organizer, 2026-09-23) | definition of "new fault" |
| Forum t/11527 "How were the new test faults identified" (organizer, 2026-09-23) | nothing: the data sources and fault types behind the test labels are not disclosed |
| `github.com/drivendataorg/gems-prize-reference-solution` (notebook `unet-mc-cv-reference-solution.ipynb`) | no scorer; labels `>= 1` are faults; output written with the labels' CRS / transform; comment: "it may be advantageous to threshold this map for better scoring" |
| Forum t/11505, t/11524, t/11520 | 3 scored submissions per rolling 7-day window (midnight UTC); the best final-leaderboard submission is auto-selected |

## The contract

| item | official | internal before GEMS3 | status |
|---|---|---|---|
| kernel | k(d) = max(1 − d/R, 0), R = 300 m, d between pixel centres | same (`metric.kernel_offsets`; k(300 m) = 0) | verified (statement) |
| α, β | 0.2 (FP), 0.8 (FN) | same | verified (statement) |
| TP / FP / FN | TP = Σ_g max_x p k; FP = Σ_{p>0} p (1 − max_g k); FN = Σ_g (1 − max_x p k) | same | verified (statement) |
| ε | present, value unpublished | 1e-6 | unspecified, negligible |
| truth | **new faults only** ("any fault pixel not already captured by USGS/INGENIOUS"; can include new geometry of an existing system) | geo DTI: all mapped faults | verified (organizer) |
| known-fault mask | **pixel-exact, identical to the training fault labels**; masked pixels "do not count towards penalty terms" | pseudo-discovery `excl`: retained faults **dilated 300 m** | verified (organizer). **Mismatch** |
| near-known FP | a prediction near a known trace but far from new truth is "fully penalized, i.e., the buffer does not apply to known faults" | `excl` forgave everything within 300 m of a retained fault | verified (organizer). **Mismatch** |
| masked predictions as TP support | "it should not matter whether these known faults are included with predictions or not" | excluded pixels could support nearby TP | **inferred**: masked predictions have no effect. Both conventions are implemented |
| new truth near known | allowed (corrections / modifications) | pseudo-discovery withholds whole systems (linked within 1 km) | verified (organizer); proxy limitation |
| aggregation | one Tversky index pooled over all public-subset pixels (private: same; final: the whole GeoDAWN area) | pooled components across folds / splits | verified (organizer) |
| weighting | none beyond the kernel; same metric for public, private and final | none | verified (organizer) |
| format | single-band float32 GeoTIFF, EPSG:32611, 100 m, training bounds, values in [0, 1], null / NaN outside the bounds | same | verified (statement) |
| NaN inside the bounds | not stated | NaN outside the data footprint | unverified. GEMS3 writes **0 where the template is finite** and NaN only where the template is NaN |
| clipping / out-of-range values | not stated | clipped locally | unverified. Submissions stay in [0, 1] (validated) |
| public / private split | chunking not disclosed | – | unknown |
| phase 2 | labels can be added, removed or changed after expert review; the same exclusion raster | – | verified (organizer) |

## Mismatches found and how they are fixed

1. **Known-fault mask geometry.** The historical pseudo-discovery policy `excl` excluded a 300 m dilation of the
   retained faults. The official mask is pixel-exact, and predictions 100–300 m off a known trace are fully
   penalised. `excl` therefore overstated every model that spreads probability around known faults (a CNN trained
   on mapped faults does exactly that). New task `disc_official` (`gems_bench/gems3/official.py`): truth =
   withheld systems, exclusion = retained faults pixel-exact, with `exclude_supports_tp=False`. It is the primary
   development task. `disc_excl` is kept for parity with M6 numbers only.
2. **Masked predictions as TP support.** `metric.dti_components`, `metric.MaskedScorer` and `p3.curves.BinaryCurve`
   take a new `exclude_supports_tp` flag. The default `True` keeps every historical number unchanged; `False` is the
   official reading (`metric.OFFICIAL_EXCLUDE_SUPPORTS_TP`).
3. **Truth definition.** Geographic DTI scores rediscovery of *mapped* faults, and the leaderboard does not.
   Geographic DTI stays a reported generalisation check, and selection uses `disc_official`.

Regression tests: `tests/test_gems3.py`, sections 1–3 (parity of the fast scorer with `MaskedScorer` and with
the independent KD-tree `parity.reference_dti`; max-based TP; kernel boundary; masked-prediction invariance in
all three scorers; the four synthetic conventions; adjacent known / new faults; unit construction).

## Consequences for submissions

* **Known-fault fill** (writing 1 on known pixels): it can never lower the score, because masked pixels carry no
  penalty. It helps only if masked predictions support TP of nearby new truth, which the organizer's "should not
  matter" excludes. Default: off. `submission.known_fill: true` exports a second, otherwise identical file. One
  upload of the pair settles the reading. That is a scorer-convention check, not a probe of hidden labels.
* **Probability mass around known faults** is a real false-positive cost unless new truth is there. Hence the
  representation-stage `known_ring` operation (zero predictions within 1–3 px of known faults, known pixels
  untouched), selected on `disc_official`. Caveat: new truth *can* lie next to known traces (corrections). The
  pseudo-discovery proxy under-represents that class, so a selected ring may be over-aggressive on the hidden
  test. This is the main known bias of the proxy.
* **Binary vs soft**: for a fixed truth set the DTI-optimal map is binary (`p3/curves.py` docstring: the score's
  super-level sets are convex in p). Soft outputs can only win through uncertainty about the truth.
  `metric_opt` tests this empirically (`raw`, `soft_top`, `halo_soft` against binary representations) on
  held-out zones. Nothing assumes either answer.
* **Pooling over all public pixels** means the total predicted area competes against the whole public region's
  sparse new-fault truth. The optimal area fraction therefore depends on the hidden truth density, which is
  unknown. The submission budget is a development choice (`submission.area_frac_source`, default: the geo
  all-zone fraction). This is the largest unverified assumption in the submission path. Do not tune it with
  repeated leaderboard probes.

## What remains unverified

* Whether masked predictions support TP (inferred: no).
* NaN handling inside the bounds (avoided: 0 is written wherever the template is finite).
* The ε value, and the public / private chunking.
* Whether hidden-test new faults resemble withheld mapped systems. The organizers did not disclose the sources;
  the forum suggests lidar scarps and geophysics both play a role.
