# Feature engineering

Every feature is one tiled float32 GeoTIFF on the 100 m grid under `03_features/<family>/` with a registry row in
`03_features/feature_registry/feature_registry.parquet` (feature_id, family, parent_source, unit, scale_m,
operator, aggregation, valid_fraction, nodata_policy, orientation_convention, normalization, output_path,
fingerprint, status, meaning, stage). Orientation convention: axial strike azimuth a in [0, 180) degrees clockwise
from north, stored as (cos 2a, sin 2a). Geographic axes: x = east, y = north.

## raw (`03_features/raw_bands/`)
Every GeoDAWN band, nodata -> NaN; feature_id = band name.

## magnetic (RTP) and gravity (isostatic anomaly) - `magnetic.build_potential_family`
FFT with k = 2 pi f (rad/m): nearest fill, de-mean, 25 % reflect padding, 32 px cosine taper, crop back. NaN
outside the footprint and within 1 km of its edge (continued fields: max(1 km, min(h, 2 km))).

| feature | definition |
|---|---|
| `<p>_field` | raw band |
| `<p>_thd`, `<p>_vd` | sqrt(fx^2+fy^2), F^-1(\|k\| F), unit/m. Gravity: compared with `iso_grav_anom_hg/_vg`; Spearman >= 0.98 -> status `alias_of_delivered` (not stored twice) |
| `<p>_tilt` | atan2(fz, THD + eps) |
| `<p>_analytic_signal` | sqrt(fx^2+fy^2+fz^2) |
| `<p>_dxx`, `_dyy`, `_dxy` | second horizontal derivatives |
| `<p>_boundary_c2/_s2` | strike perpendicular to grad f (axial) |
| `<p>_uc{500,1000,2000,5000}` | exp(-\|k\| h) F |
| `<p>_residual{h}` | field - uc{h} |
| `<p>_thd_uc{h}` | THD of the continued field |
| `<p>_edge_persistence` | share of levels (0, 0.5, 1, 2, 5 km) with THD rank >= 0.9 |
| `<p>_edge_stability` | mean of min(rank) over consecutive levels |
| `<p>_boundary_strength` | mean THD rank over levels |
| `<p>_edge_density_2km` | share of rank >= 0.9 pixels within 2 km |
| `<p>_coherence/_anisotropy/_strike_c2/_strike_s2_{300,1000,3000}m` | structure tensor |
| `<p>_orientation_consistency` | coherence-weighted resultant of the per-scale strikes |

Spectra: `04_eda/tables/<family>_radial_spectrum.csv` (azimuthal power vs k) and `<family>_band_energy.csv`
(shares for 0.2-1, 1-3, 3-10, 10-30, > 30 km). No source-depth estimates.

## terrain (100 m, `terrain.build`)
Scales 300, 1000, 2000, 3000 m (sigma = s / 2): `terrain_slope`, `_profile_curvature`, `_plan_curvature`,
`_ridge`, `_valley`, `_tpi`, `_relief`, `_roughness`, `_slope_break`, `_coherence`, `_strike_c2/_s2` per scale;
`terrain_dz_east/north_1km`; `terrain_dog_<a>_<b>m`; `terrain_elevation`, `terrain_slope_delivered`. NaN within
2 sigma of the footprint edge. Nothing at 100 m resolves metre-scale scarps.

## dem (`dem.py`, `dem_features.py`, `dem_aggregation.py`)
FAST_10M (default) or DETAILED_2M (opt-in); parameters in `config.DEFAULTS["dem"]`. Windows of 50 x 50 cells with
a halo >= the largest operator support (1.2 km FAST, 420 m DETAILED). Guards: 4 sigma of each operator from missing
data, 60 m from DEM project seams. 100 m statistics (`dem_<name>` in `03_features/dem_multiscale/<MODE>/`):
edge_p50 / p90 / max, edge_anisotropy, coherence_p90 at both structure scales, orient_c2 / s2 / R
(coherence-weighted), slope_p50 / p90 / std, profile_curvature_min / max (p05 / p95), plan_curvature_absp90,
tpi_<r>m_absp90, relief_mean, ridge_p90, valley_p90, scarp_p90, roughness_mean, edge_fraction (slope >= 20 deg and
coherence >= 0.5, label-free), elevation_mean; DETAILED adds crest_p90 and toe_p90. FAST computes no fine-scale
product. Support fractions: `02_geospatial/coverage/dem_support_<MODE>.tif`; measured coverage
`dem_coverage_<MODE>.tif`; seam share `dem_seam_fraction_<MODE>.tif`. Missing DEM is NaN, never zero elevation.

## cross_modal (`cross_modal.build`)
Groups: magnetic, gravity, topography (100 m terrain + DEM = ONE group). `xm_edge_<a>_<b>`, `xm_orient_<a>_<b>`,
`xm_available_modalities`, `xm_evidence_count`, `xm_evidence_fraction`, `xm_evidence_mean`,
`xm_multiscale_persistence`, `xm_disagreement`, `xm_continuity`. `parent_source` lists the contributing groups.
Descriptive evidence, not fault probabilities.

## Export flags
Feature rasters are always GeoTIFF and summary tables always CSV, because later stages and figures read them;
`export_parquet` controls the Parquet copies. `export_geotiff` / `export_csv` therefore do not switch anything off.
