# Geometry, submission parity and metric units

Status: static findings plus UNTESTED diagnostics (`gems_bench/g14_1/geometry.py`, stage R01).

## Why 3730 × 3292?

That is the native competition grid. The repository records it in several places:

* `docs/DATA_PROVENANCE.md`: 3730 × 3292 px, 100 m, EPSG:32611, 19 bands.
* `gems_eda/discovery/data.py`, `gems_bench/geological_atlas/config.py`: the same grid.
* `docs/leaderboard_focus/M1_HANDOFF.md`: template spec "EPSG 32611, 100 m, 3730×3292, float32, NaN nodata".

The other dimension found in the repository, **1865 × 1646 at 200 m**, is the historical EDA working grid. Its sources
are `outputs/eda/EDA_FINDINGS.md`, `gems_bench/data/__init__.py` (Phase-1 rasters, 2× block-averaged) and
`gems_bench/features/__init__.py` (EDA products resampled bilinearly back to 100 m, flagged `eda_resampled`).
G14 does not read those products.

Static chain for G14:

1. Input. `g13.contracts.load` opens features, known faults and template, and `check_grid` asserts the same CRS,
   transform and shape for all three, EPSG:32611 and 100 m pixels. It raises otherwise.
2. Every array is (row = north→south, col = west→east), as read by rasterio. No resampling happens at 100 m.
3. G14 coarse views are block means at factors 2 / 8 on the same origin. Inference writes back on the 100 m grid.
4. Export: `g13.submission.write` copies the template profile (CRS, transform, shape), writes float32 with NaN
   outside the template, and sets 0 on `data.labels`. `validate` checks CRS, transform, shape and 100 m.

**Static conclusion:** no evidence of a raster mismatch in the G14 path. It still has to be verified on the real file
(R01): read the submitted GeoTIFF back and compare it with the map it should contain.

## R01 diagnostics

| check | function | detects |
|---|---|---|
| grid table | `grid_table` | CRS, shape, pixel size and origin offset of every raster vs the template |
| identity | `submission_parity(expected=...)` | max abs difference, share of differing pixels |
| shift | `phase_correlation` | integer and sub-pixel displacement (parabolic peak refinement) |
| flips / transposition | `d4_match` | best D4 transform by correlation |
| binarisation / clipping | `submission_parity` | number of unique values, share of exact 0 / 1, min / max |
| footprint | `submission_parity` | NaN inside the template, finite values outside |
| known masking | `submission_parity` | positives on `data.labels` and on the raw known raster (S21) |
| label registration | `ridge_offset_to_known` | mean signed vector from high-score ridges to the nearest known pixel (a constant non-zero vector suggests that labels and inputs are mis-registered) |
| round trip | `round_trip` | write the official way, read back, compare |

## Metric kernel in pixels

`geometry.kernel_radius_px(transform)` uses R = 300 m divided by the pixel size from the affine transform, separately
for rows and columns. The kernel has offsets only where d < R. At 100 m that gives radius 3 px, with the farthest
offset at (2, 2) (283 m); (3, 0) is exactly 300 m and has weight 0. At 50 m the radius would be 6 px.

G14.1 code takes `pixel_m` from the transform everywhere it builds a distance:

* `pipeline.pixel_m` raises for non-square pixels;
* the loss radius is `300 / pixel_m`;
* the discovery masks, the conditional groups and the atlas all use metres.

## Hard-coded assumptions (S13)

`geometry.HARD_CODED` lists every 100 m assumption found in the G14, G13, P6 and metric code: the kernel of 3 px,
ignore ring 3 px, reliable negatives 10 px, orientation band 3 px, 0.1 km/px, `pixel_m = 100`, local view 128 px and
the 13 km embargo.

`assumption_audit(transform)` marks each item CONSISTENT or INCONSISTENT against the real pixel. It also flags that
the 13 km embargo is shorter than the 12.8 km variogram range only if the range is larger; G13 reported 12.8 km as
the *maximum lag* measured, so the real range may be longer.

## Metric definitions (unchanged, verified in repository tests)

TP = Σ over truth pixels of max_x p(x)·k(d), with k = 1 − d / 300 m. FP = Σ p·(1 − k_d). FN = n − TP.
DTI = TP / (TP + 0.2·FP + 0.8·FN).

`disc_official` masks known faults pixel-exactly, and they cannot support TP (`OFFICIAL_EXCLUDE_SUPPORTS_TP = False`,
an organiser-statement reading). G14.1 never re-implements the metric: it calls `gems_bench.metric` /
`gems3.official`.
