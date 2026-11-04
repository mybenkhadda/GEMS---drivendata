# G07 DEM data contract

## Source and location

USGS 3D Elevation Program 1 m DEM tiles from The National Map (`docs/DATA_PROVENANCE.md`): `USGS_1M_<zone>_x<E/10km>y<N/10km>_<project>.tif`, public domain (USGS). The tiles are on Drive, not in the repository:

* Tiles: `MyDrive/GEMS_2/data/raw/dem_1m/**/*.tif` (Upper Humboldt, Reno-Carson and other Nevada projects).
* URL list: `MyDrive/GEMS_2/data/raw/dem_1m/tnm_urls.json` (also `data/raw/tnm_items.json`, `data/raw/dem_1m/tnm_items.json`).

Existing files are never re-downloaded. `dem.download.enabled: true` is an explicit, guarded opt-in (`max_tiles`, `max_gib`).

## Source status (`dem_io.open_dem_catalog`)

| status | meaning | used as data |
|---|---|---|
| `LOCAL` | file exists and passes every check below | **yes** |
| `LOCAL_INVALID` | file present but excluded | no |
| `LOCAL_OUTSIDE_STUDY_AREA` | valid header, no intersection with the grid or data footprint | no |
| `CATALOG_ONLY` | listed in the URL list, no local file | no |

## Checks

| check | rule | exclusion code |
|---|---|---|
| readable | the GeoTIFF opens | `UNREADABLE` |
| resolution | known, <= `max_accepted_res_m` (3 m), square pixels. The 100 m `det_elev` band is never a substitute | `RESOLUTION_UNKNOWN`, `RESOLUTION_TOO_COARSE`, `NON_SQUARE_PIXELS` |
| CRS | present, projected, linear unit metre (compound 3DEP CRSs are accepted; aggregation reprojects to the analysis grid exactly) | `CRS_MISSING`, `CRS_NOT_PROJECTED_METRIC`, `CRS_UNUSABLE` |
| transform | finite, north-up, positive sizes | `INVALID_TRANSFORM` |
| vertical unit | explicit metadata, see below | `UNKNOWN_VERTICAL_UNIT` |
| study area | intersects the grid and the GeoDAWN footprint | `NO_GRID_INTERSECTION`, `NO_FOOTPRINT_INTERSECTION` |

A missing nodata value and a resolution other than the 1 m target are warnings (`warnings` column), not exclusions. Overlapping projects: one source per tile id, latest year.

## Vertical units

Read from, in order: band units, tags (`UNITTYPE`, `units`, `VERTICAL_UNITS`, `VERTICAL_UNIT`), the **vertical** component of a compound CRS, the `<altunits>` of the USGS XML next to the tile. `vertical_unit_source` records which. **The horizontal CRS unit is never used.**

If none states a unit, the tile is `LOCAL_INVALID` with `UNKNOWN_VERTICAL_UNIT` and `dem_catalog/dem_source_validation.json` gives the remedy. `dem.vertical_unit_override` is an explicit, user-confirmed assumption: it applies only where metadata is silent (a stated unit always wins), it is recorded as `vertical_unit_source = override`, and it is never set by default.

## Outputs

`dem_catalog/dem_inventory.csv` (every source), `dem_catalog/dem_invalid_sources.csv` (every excluded file with codes and reasons), `dem_catalog/dem_source_validation.json` (counts and remedy), `tables/dem_coverage.csv`, `dem_catalog/dem_coverage.geojson`, `tables/dem_seam_audit.csv`.

## Coverage

The share of the ELIGIBLE fault-discovery region (footprint & template & not a known fault) and of every V1 evaluation zone covered by LOCAL tiles (and by the catalog) - never a plain bounding-box overlap with the grid. `dem_coverage` in the feature store is the fine-tier share of each 100 m cell; `dem_coverage_coarse` is the merged coarse DEM's share and selects the U-Net patches.

## Processing

1. **Coarse DEM:** NaN-aware block means per tile (cached, pushed), merged at `coarse_res_m` through a disk-backed array. All the U-Net needs.
2. **Fine tier:** windows of `dem_tile_px` + halo >= operator support, optional block mean to `work_res_m`. Pixels beyond the tile are filled from exactly aligned neighbouring tiles of the same CRS and resolution, so a window crosses tile edges seamlessly.
3. **Coarse tier:** windows of `coarse_tile_px` over the merged DEM.
4. **Aggregation** to 100 m with exact area weights; coverage is counted in WORK pixels. Mosaic: each band is weighted by the coverage of the tiles that really carry it.

### Tile boundaries invent no lineaments

* Inside the data, windows read across tile edges: results equal one continuous raster.
* At a real data edge, normalised convolution would bend the smoothed surface (slope error about 11 % at 2 sigma from the edge), and slope, curvature, scarp and lineament responses would run along the 10 km tile grid. Features are therefore NaN within `edge_guard_sigma` (4, the Gaussian truncation) x their scale of missing data, x 3 for structure-tensor products, x 4 for roughness.
* A real offset between two projects cannot be removed this way. `tables/dem_seam_audit.csv` compares each feature in cells on tile boundaries with other covered cells and flags ratios beyond 1.5: read it before trusting a DEM feature.

## If no DEM is usable

`DEM_FEASIBILITY` = BLOCKED_INPUT with exact reasons; DEM-dependent stages are BLOCKED_INPUT in both training modes. The geology path (B experiments, candidates without the surface pool, evaluation) still runs when the notebook is not stopped (`STRICT_NOTEBOOK = False`, or DEM flags off).
