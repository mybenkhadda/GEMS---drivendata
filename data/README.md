# data/ — not in Git

Everything under `data/` is excluded from version control. Raw and processed data live on Google Drive
(`MyDrive/GEMS_2/data/...`) and are staged to the Colab disk (`/content/gems_work/data/...`) by
`gems_bench.runtime.colab.stage_core()`.

| path | content | source |
|---|---|---|
| `data/raw/geodawn/gems-geodawn-numerical-features.tif` | 19-band GeoDAWN raster, 100 m, EPSG:32611 (~420 MB) | DrivenData GEMS |
| `data/raw/geodawn/example_submission.tif` | submission template grid | DrivenData GEMS |
| `data/raw/faults/existing_faults.tif` | released mapped-fault raster (value ≥ 1 = fault) | DrivenData GEMS |
| `data/raw/dem_1m/tnm_urls.json` | USGS TNM 1 m DEM download URL list | USGS The National Map |
| `data/raw/dem_1m/<project>/USGS_1M_*.tif` | pilot 1 m DEM tiles (downloaded only after explicit approval) | USGS 3DEP |
| `data/feature_store/phase5/` | Phase-5 feature cache (`p5:*`, `dem:*` .npy) + registry | built by notebook 01 / 05 |
| `data/processed/splits/<design>/` | A3 validation designs (fold / split GeoTIFF + params) | built by notebook 02 |
| `data/processed/dem_1m_100m/` | per-tile and mosaic 1 m morphology aggregated to 100 m | built by notebook 05 |
| `data/external/` | optional third-party files (e.g. a local copy of the official scorer) | user |

Locally the original layout keeps working: the three raw GeoTIFFs at the project root are found first
(`gems_eda.config.resolve_raw`). Never commit data, credentials or checkpoints; see `.gitignore`.
