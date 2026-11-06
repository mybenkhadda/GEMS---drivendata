# Data sources

Status: authored 2026-11-06; **nothing executed** (no local run, no Colab run). All facts below are rules the code
applies; actual file counts, sizes and band lists appear only after sections 01-03 run.

## Original raw sources (discovery rules, `inventory.discover_sources`)

All paths relative to `drive_root` = `/content/drive/MyDrive/GEMS_2`.

| role | rule |
|---|---|
| `geodawn_features` | `gems-geodawn-numerical-features.tif`, first hit in `data/raw/geodawn`, `data/raw`, `data/raw/faults`, `""` |
| `fault_labels` | `existing_faults.tif`, same folder order |
| `official_template` | `example_submission.tif`, same folder order |
| `*:duplicate_candidate` | further hits of the same name (recorded, not snapshotted) |
| `dem_tile` | `data/raw/dem_1m/**/*.tif` and `*.tiff` |
| `dem_metadata` | `data/raw/dem_1m/**/*.xml` (not `.aux.xml`), FGDC metadata, copied next to its tile |
| `dem_catalog` | `data/raw/dem_1m/tnm_urls.json`, `data/raw/dem_1m/tnm_items.json`, `data/raw/tnm_items.json`, `data/raw/tnm_urls.json` |
| `provenance_document` | `data/raw/**/*LICENSE*`, `*license*`, `*PROVENANCE*`, `*provenance*`, `*README*`, `*readme*` |

Never sources: any path component containing `outputs`, `experiments`, `feature_store`, `processed`,
`model_registry`, `e6_v2_from_raw`, `GEMS_GEOLOGICAL_DATA_ATLAS`; any `.partial` / `.tmp-` file.
Access status per file: `OK`, `MISSING` (not found), `INACCESSIBLE:<class>` (stat failed; never reported as missing
or invalid). `Errno 107` stops discovery with remount instructions.

## GeoDAWN raster
Band registry built from the raster itself (`inventory.band_registry`): index, `band_name` tag or the description
before ` - `, tags, modality (known-name table, else `unknown`), unit (metadata, else a documented default flagged
`documented default`, else `unknown`), dtype, nodata, valid fraction and p01/p50/p99/min/max/mean/std on every 4th
pixel. Nodata = non-finite, the declared nodata, or |x| >= 1e30. Valid footprint = >= 50 % of bands valid.
The candidate channels (`det_elev`, `rtp`, `iso_grav_anom`, ...) are expected, not assumed: a missing `rtp` or
`iso_grav_anom` stops its family with an explicit error; total-field bands are never treated as RTP.

## Fault labels
Value >= 1 = fault; reprojected nearest-neighbour if its grid differs; label-raster footprint = its finite pixels.

## USGS 1 m DEM (classified on the snapshot copies, `inventory.dem_inventory`)
| status | meaning |
|---|---|
| VALID | readable, projected CRS, 0.5-3 m pixels, finite elevations in a decimated read, known vertical unit, overlaps the GeoDAWN footprint |
| UNREADABLE | open / read failure, no projected CRS, resolution outside range, or no finite value |
| UNKNOWN_VERTICAL_UNIT | no unit in band units, tags (`UNITTYPE`, `units`, `VERTICAL_UNIT(S)`), vertical CRS or FGDC `<altunits>`; `dem.vertical_unit_override` is your explicit decision |
| OUTSIDE_STUDY_AREA | valid tile without footprint overlap |
| DUPLICATE_OVERLAP | same 10 km tile id as a newer project (kept as a gap filler at lower priority) |
| CATALOG_ONLY | listed in the TNM catalog, not downloaded |

`has_metadata` flags tiles without XML. Tile id, project and year are parsed from
`USGS_1M_<zone>_x<..>y<..>_<project>.tif`.

## Optional new DEM tiles
`enable_new_dem_downloads=False` by default. When enabled: only
`https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/...` URLs from the snapshot catalog; greedy, label-free
selection by newly covered footprint cells (one per tile id, newest project); `acquisition.max_new_tiles` and
`max_gib` budgets; files saved into `00_raw_snapshot/dem_1m/<project>/`. Licence recorded: USGS 3DEP 1 m DEM,
public domain (U.S. Government work).
