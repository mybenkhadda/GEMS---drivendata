# Feature schema

Channel lists are code (`contracts.py`); their hash (`schema_hash`) is part of every dataset and checkpoint identity.
All "scale s" operators are Gaussians with sigma = s / 2. Kinds: `scalar` (invariant under 90-degree rotation and
flips) or `axial_cos` / `axial_sin` (doubled-angle pair).

## Branch 1 - terrain (14, from `det_elev` / `det_elev_slope` at 100 m)
`elev`, `slope_raw`, `grad_{300,1000,3000}m` (gradient magnitude, m/m), `log_{300,1000,3000}m` (Laplacian of
Gaussian, 1/m), `relief_{1000,3000}m` (max - min), `tpi_{500,2000}m` (z - Gaussian mean),
`dog_300_1000m`, `dog_1000_3000m`. Edge guard: NaN within 2 sigma (or half the window) of the footprint edge.
Cache: `cache/terrain/{feature_stack.npy (14,H,W) f32, feature_registry.csv, normalization.json, manifest.json}`.

## Branch 2 - potential fields (44)
Per field (magnetic `rtp` nT -> prefix `mag`; gravity `iso_grav_anom` mGal -> prefix `grav`), 17 channels:
`_field`, `_vd` (|k| F), `_thd` (sqrt(fx^2 + fy^2)), `_tilt` (atan2(fz, THD + eps), rad), `_as` (analytic signal),
`_uc{500,1000,2000,5000}` (exp(-|k| h) F), `_res{h}` (field - continuation), `_thd_uc{h}` (THD after continuation).
`k = 2 pi f` (rad/m), so derivatives are in unit/m. FFT preparation: nearest fill of nodata, de-mean, 25 % reflect
padding, 32 px cosine taper, crop back; derivative products NaN within 1 km of the footprint edge, continuation
products within min(h, 2 km).
Subsurface (10, raw, normalised only): `raw_iso_grav_anom_hg`, `raw_iso_grav_anom_vg`, `raw_iso_grav_anom_slope`,
`raw_depth_to_base_surf`, `raw_cond_surf`, `raw_geod_2ndinv`, `raw_geod_shearrate`, `raw_geod_dilaterate`,
`raw_deq_n100a15`, `raw_ieq_n100a15`.
Cache: `cache/potential_fields/{magnetic_features.npy, gravity_features.npy, subsurface_features.npy,
feature_registry.csv, normalization.json, manifest.json}`.

## Branch 3 - DEM (22 statistics per 100 m cell + coverage)
2 m operators (`dem_features.py`, gradient sigma 2 m, guard = 4 sigma of each operator):
slope; 8-azimuth |directional derivative| smoothed at 5 m -> edge energy (mean) and anisotropy
((max - min) / (max + min)); structure tensor at 5 m and 20 m -> coherence ((l1 - l2) / (l1 + l2))^2 and the
doubled-angle strike (20 m); principal curvatures and the second derivative along the gradient at 10 m
(crest = max(-k_p, 0), toe = max(k_p, 0)); TPI at 50 m and 200 m; scarp = |z(p + 10 m n) - z(p - 10 m n)| across
the 20 m gradient axis; roughness = local std of z - Gaussian(z, 10 m); edge pixel = slope >= 20 deg and
5 m coherence >= 0.5.

| channel | statistic over the usable 2 m pixels of the cell |
|---|---|
| edge_energy_p50 / _p90 / _max | percentiles / max of edge energy |
| edge_anisotropy | mean |
| coherence_p90_5m, coherence_p90_20m | 90th percentile |
| orientation_cos2, orientation_sin2 (axial) | 20 m-coherence-weighted mean strike vector |
| orientation_R | length of that mean vector |
| slope_p50 / _p90 / _std | degrees |
| curvature_min / curvature_max | p10 of k_min / p90 of k_max (1/m) |
| tpi_50m / tpi_200m | p90 of abs(TPI) (m) |
| scarp_strength | p90 of the scarp step (m) |
| crest_response / toe_response | p90 |
| edge_fraction | share of edge pixels |
| roughness / roughness_p90 | mean / p90 (m) |

A statistic is NaN where < 25 % of the cell's pixels pass its guard. `dem_coverage` = share of 2 m pixels with data,
farther than the gradient guard from missing data and farther than 60 m from a seam between DEM projects.
Cache: `cache/dem/features_100m/{dem_features_100m.npy (22,H,W) f32, normalization.json}`,
`cache/dem/coverage_100m/dem_coverage_100m.npy`.

## Normalisation (`normalization.py`)
scalar: `clip((x - q50) / (q95 - q05), -6, 6)`; axial: unchanged; NaN -> 0. Statistics from a deterministic
2 M-pixel sample of the footprint (DEM: covered cells), inputs only - never labels, identical for every fold (a
label-free transductive choice). Datasets: `datasets/{terrain,field,dem}.npy` float16 on the grid padded to a
multiple of 8. `dem_coverage` is forced to 0 where any DEM statistic is NaN (count in `dataset_schema.json`).

## Augmentation contract
Only 90-degree rotations and flips. Scalars are unchanged; axial pairs: rotation (c, s) -> (-c, -s), flip
(c, s) -> (c, -s); the orientation target follows the same rule. No intensity, scale or noise augmentation.
