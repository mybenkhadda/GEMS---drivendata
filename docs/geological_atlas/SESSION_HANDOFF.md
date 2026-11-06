# Geological Data Atlas V1 - session handoff

## 2026-11-06 - phase 0: code authored, nothing executed
- Written: `gems_bench/geological_atlas/` (30 modules + `__init__.py`), notebook
  `notebooks/colab/GEMS_GEOLOGICAL_DATA_ATLAS.ipynb` (sections 00-16), `docs/geological_atlas/` (8 docs).
- Executed: **nothing** - no local Python, tests, notebook or Colab run. No atlas folder exists on Drive yet; no
  number, figure or candidate exists yet.
- Not committed.

## Decisions taken while authoring (check them)
1. CSV summary tables and GeoTIFF features are always written (downstream stages read them); `export_csv` /
   `export_geotiff` therefore do not switch anything off.
2. DEM tile checksums are full SHA-256 in the snapshot (every byte is copied anyway).
3. Coverage in section 02 is measured from a 10 m decimated read of each VALID tile; section 08 measures it again
   at the work resolution with guards.
4. DUPLICATE_OVERLAP tiles are used only to fill pixels the preferred (newest) project leaves empty.
5. Terrain and DEM count as ONE topography evidence group in the cross-modal layer.
6. Gravity THD / VD are not stored when they equal the delivered `_hg` / `_vg` bands (Spearman >= 0.98).
7. Lineament rules (92nd evidence percentile, 3 km, linearity 0.85) are fixed defaults, chosen without the labels.

## Next step
Run sections 00-02 and read `01_inventory/manifests/preflight.csv`, `snapshot_report.json` and
`01_inventory/dem_inventory.csv` (vertical-unit sources, UNKNOWN_VERTICAL_UNIT count) before continuing.

## Open risks (first execution of all code)
- Expect first-run errors (rasterio `WarpedVRT` / `read(out_shape, resampling)` details, shapes, pandas dtypes).
- Snapshot doubles the raw storage on Drive (the report shows the bytes before copying).
- FAST_10M runtime and RAM unmeasured; DETAILED_2M is much slower.
