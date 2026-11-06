# Raw-data contract

The pipeline reads exactly these inputs, all under `paths.drive_root` (`/content/drive/MyDrive/GEMS_2`):

| source | file | searched in (`raw.search_dirs` / `raw.dem_dirs`) |
|---|---|---|
| GeoDAWN numerical features | `gems-geodawn-numerical-features.tif` | `""`, `data/raw`, `data/raw/geodawn`, `data/raw/faults` |
| fault labels | `existing_faults.tif` | same |
| official template | `example_submission.tif` | same |
| USGS 1 m DEM | `**/*.tif` (+ FGDC `.xml` next to each tile) | `data/raw/dem_1m` |

Refused: any path under `raw.forbidden_prefixes` (`outputs`, `experiments`, `data/feature_store`, `data/processed`,
`model_registry`, the namespace itself), any `.tmp-*`, `*.partial`, `*.tmp` file. If a GeoDAWN file exists in several
search folders with different sizes, the preflight blocks.

## GeoDAWN raster
- expected: EPSG:32611, 100 m, 3730 x 3292 (documented; the header is authoritative, a different shape is a warning),
  north-up (rotation blocks), 19 bands.
- band name = tag `band_name`, else the band description up to ` - `.
- required bands: `det_elev`, `det_elev_slope`, `rtp`, `iso_grav_anom` and the ten `potential.subsurface_bands`.
- nodata: not finite, equal to the declared nodata, or `|x| >= 1e30`.
- footprint (`valid`): >= 50 % of the bands valid.
- **excluded on purpose**: `mag_anom`, `tmi`, `tmi_hg`, `tmi_vg`, `tc` - total-field magnetic products whose anomaly
  shape depends on the inclination / declination of the Earth's field; rotating or mirroring them produces
  physically impossible patterns, which would break the augmentation contract. Magnetic information enters through
  the reduced-to-pole `rtp` and its FFT features.

## Fault labels
- fault pixel = value >= 1 (`raw.label_min_value`); reprojected nearest-neighbour onto the GeoDAWN grid if its grid
  differs (warning). Labels outside the footprint are dropped.

## Template
- must have exactly the GeoDAWN grid (blocker otherwise). template-valid = finite and != nodata.
- the submission is written on the template's CRS / transform / shape.

## DEM tiles
Header checks per tile (`dem_sources.validate_tiles`): readable, has a CRS, native resolution in [0.5, 1.5] m,
>= 1 band, determinable vertical unit. Vertical-unit rule (first that answers):
1. GeoTIFF band `units`; 2. vertical CRS in the WKT; 3. FGDC XML `<altunits>`; 4. `dem.vertical_unit_override`.
Unknown -> preflight blocker. Factors: metre 1, foot 0.3048, US survey foot 1200/3937.
Elevations outside [-500, 9000] m after conversion are nodata; a source without declared nodata uses -999999.

Overlap priority: newer project year, then project name, then file name (project / year parsed from the USGS file
name `..._x<..>y<..>_<project>`, else the parent folder name).

## Inventory
`manifests/raw_data_inventory.{json,csv}`: source_id, type, path, size, checksum, checksum_kind, crs, resolution,
shape, bands, nodata, bounds, vertical_unit, validation_status. GeoDAWN files and metadata XML: full sha256. DEM
tiles: `sha256(size, first MiB, last MiB)` (labelled in `checksum_kind`) - full hashes of hundreds of GB of tiles per
session are not affordable. `raw_fingerprint` = hash of (source_id, checksum, status) of every row.
