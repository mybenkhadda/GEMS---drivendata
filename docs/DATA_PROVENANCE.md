# Data provenance

| dataset | origin | grid | notes |
|---|---|---|---|
| GeoDAWN multiband raster (`gems-geodawn-numerical-features.tif`) | DrivenData GEMS competition | 3730 × 3292 px, 100 m, EPSG:32611, 19 bands | terrain = detrended elevation + slope; magnetic (6), gravity (4), basement depth, conductivity, strain (3), seismicity (2); valid footprint 42 % |
| Mapped faults (`existing_faults.tif`) | DrivenData GEMS (USGS-derived, incomplete) | same grid | value ≥ 1 = fault; 60,988 px. Incomplete by construction: unmapped ≠ negative |
| Submission template (`example_submission.tif`) | DrivenData GEMS | same grid | defines the output grid; local audit compares footprint with the data mask |
| USGS TNM 1 m DEM | USGS 3D Elevation Program via The National Map | 1 m, NAD83 / UTM 11N (verify per tile) | URL list supplied by the user; tiles named `USGS_1M_<zone>_x<E/10km>y<N/10km>_<project>`; public domain (USGS) |

## Derived data (all reproducible from code + raw data)

* Phase-2 feature cache (`outputs/benchmark/cache/features`), Phase-4 RS cache (`outputs/phase4/cache/features`):
  label-independent transforms of the raw bands (provenance in their registries).
* Phase-5 profile features (`data/feature_store/phase5`): label-independent; orientation from measured fields only.
* DEM aggregates (`data/processed/dem_1m_100m`): per-tile JSON records the source tile, project and morphology
  parameters; band `source_project` and `coverage` are kept in the mosaic.
* Folds / splits: Phase-2 `geo_k3_b20km_buf13km_s2026`, `discovery_d3_link1km_s2026`; Phase-5 designs under
  `data/processed/splits/` (parameters in each `*_params.json`).

## Transductive inputs

Label-independent rasters are computed once on the whole map and reused in every fold (the hidden-test input
raster is the same). This is allowed by the protocol as long as no label enters the computation and the fold
buffer (13 km) covers each feature's support (`evaluation.spatial_cv.buffer_audit`).
