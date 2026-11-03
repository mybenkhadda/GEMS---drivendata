# G07 DEM data contract

## Source

USGS 3D Elevation Program 1 m DEM tiles from The National Map (TNM), already documented in
`docs/DATA_PROVENANCE.md`: tiles `USGS_1M_<zone>_x<E/10km>y<N/10km>_<project>.tif`, NAD83 / UTM 11N (EPSG:26911,
verify per tile), metres (NAVD88), public domain (USGS). The repository does **not** contain the tiles or the URL list.

**Where to put them (Drive):**
* URL list (TNM Access API JSON or a list of URLs): `MyDrive/GEMS_2/data/raw/dem_1m/tnm_urls.json`
  (also accepted: `data/raw/tnm_items.json`, `data/raw/dem_1m/tnm_items.json`). Obtain it from
  https://apps.nationalmap.gov/downloader/ (Elevation Products 3DEP, 1 meter DEM, area = the GeoDAWN footprint).
* Tiles: `MyDrive/GEMS_2/data/raw/dem_1m/<project>/<file>.tif`, or set `dem.download.enabled: true` (guarded: at most
  `max_tiles` tiles / `max_gib` GiB, tiles intersecting the eligible region first, resumable, pushed to Drive).

## Checks (fail closed; ``dem_io.open_dem_catalog`` + ``contracts.validate_dem_source``)

| check | rule |
|---|---|
| availability | only LOCAL files with readable headers are data; CATALOG_ONLY tiles are reported, never assumed |
| resolution | header resolution recorded; > `max_accepted_res_m` (3 m) = LOCAL_INVALID. The 100 m `det_elev` band is never a substitute (`assert_not_100m_substitute`) |
| CRS | EPSG 26911 / 32611 / 6340 accepted (NAD83 vs WGS84 UTM 11N differ by ~1-2 m; the aggregation reprojects exactly) |
| vertical unit | from header units / tags; unknown -> problem; `dem.vertical_unit_override` only after checking metadata (recorded as ASSUMED) |
| nodata | declared value and non-finite values -> NaN; never filled with zeros |
| overlap | one source per tile id (latest year) |
| licence | `dem.research_use_ok` (public domain); `dem.competition_eligibility_verified` must be set by the user before any submission that depends on DEM features |

## Coverage

Reported as the share of the ELIGIBLE fault-discovery region (footprint & template & not a known fault) and of every
V1 evaluation zone covered by local valid DEM (and by the catalog), `tables/dem_coverage.csv` +
`dem_catalog/dem_coverage.geojson` - never a bounding-box overlap.

## Processing

Fine tier: native tiles read in work windows (`dem_tile_px` + halo >= the operator support, NaN beyond the tile),
optional NaN-aware block mean to `work_res_m`; coarse tier: one merged `coarse_res_m` DEM (seamless halos). Features
are computed at these resolutions and only then aggregated to 100 m with exact area weights (coverage written per
cell). Per-tile products are cached; a disconnect never recomputes a finished tile.

## If no DEM is available

`DEM_FEASIBILITY` = BLOCKED_INPUT with exact instructions; DEM-dependent stages are BLOCKED_INPUT; the geology path
(B experiments, candidates without the surface pool, ranker, evaluation) runs normally.
