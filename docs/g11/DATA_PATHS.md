# Data paths

Root: `data_root = /content/drive/MyDrive/GEMS_2`.

## Verified on 2026-11-10
These were read through a read-only Google Drive connector: file names and metadata only, no data downloaded.

| role | real location | config key | notes |
|---|---|---|---|
| GeoDAWN raster | `data/raw/geodawn/gems-geodawn-numerical-features.tif` (419 MB) | `source_artifacts.geodawn_raster` | |
| fault labels | `data/raw/faults/existing_faults.tif` | `source_artifacts.fault_raster` | Raster only; no fault vector layer seen, so segments are traced from the raster skeleton. |
| official template | `data/raw/example_submission.tif` | `source_artifacts.template` | |
| 1 m DEM | `data/dem_1m/` (with `downloads/`, `fast_download_report.csv`, `usgs_1m_catalog.csv`, `dem_1m_inventory.csv`) and `data/raw/dem_1m/` | `source_artifacts.dem_dirs` | Both folders are scanned and tiles de-duplicated by file name. Tiles come from UTM zones **10 and 11** (`USGS_1M_10_*`, `USGS_1M_11_*`) and are reprojected to EPSG:32611. No `.xml` sidecars were seen. |
| G07 GeoFormer | `outputs/G07_GEOFORMER/`: `T{1,2,4}_*_{stack,test}_foldmasked_v2.npy`, `T1_segformer_full.npy`, `manifest.json`, `T1_segformer.pt`, `g07_fault_evidence.tif`, … | `optional_patterns.g07_arrays`, `g07_prediction` | The arrays are 1865 × 1646 float32. **`manifest.json` documents crs EPSG:32611 and transform [200, 0, 243350, 0, -200, 4508550].** Several Drive folders share the name `G07_GEOFORMER`. |
| G10 | `outputs/G10_UNIFIED/` (`g07_foldmasked_source_audit.csv`, `heldout_foldmasked_proxy_diagnostics.csv`, …) | `optional_patterns.g10` | |
| G08 / G09.4 / G06 | `outputs/G08_DEM_1M_EDA/`, `outputs/G08_DEM_FAULT_RECOVERY/`, `outputs/G09_4_STREAMING_GEOLOGICAL_EDA/`, `outputs/G06_GEOLOGICAL_FAULTS/`, `outputs/g06_geological_feature_store/` | optional patterns | |
| M9 / P6 / GeoRegime | `outputs/m9_geofault_production/`, `outputs/p6_boost/`, `outputs/p6_fast_track/`, `GEMS_GEOREGIME_V1/` | `optional_patterns.m9`, `p6`, `georegime` | |

The audit re-verifies all of these at runtime (notebook §03). Paths are never assumed from this table.

## G11 outputs found on 2026-11-11 (metadata only)
- `outputs/G11_MULTISCALE_FAULT_DISCOVERY/`: the first real run, at commit `1618b0e`, with a complete 651-layer feature store. Details in [RECOVERY_FIRST.md §1](RECOVERY_FIRST.md).
- `outputs/G11_MULTISCALE_FAULT_DISCOVERY_SMOKE/`: from an earlier smoke run.
- `G11_PERSISTENT_PROCESSED_DATA/snapshot_20261110_214635/`: a copy started outside the G11 code. It holds one 0-byte `.partial` file and is ignored.
