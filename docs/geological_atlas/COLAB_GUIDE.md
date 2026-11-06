# Colab guide

Notebook: `notebooks/colab/GEMS_GEOLOGICAL_DATA_ATLAS.ipynb` (clones `main` of `mybenkhadda/GEMS---drivendata`).
Runtime: CPU is enough (no GPU stage); High-RAM is recommended (FFT on padded 100 m grids, full-grid arrays).

1. Raw files on Drive in their original folders (DATA_SOURCES.md); DEM tiles with their XML metadata if available.
2. Push this code to `main` (or set `REF` in cell 00 to a commit).
3. Run the notebook top to bottom:

| section | calls |
|---|---|
| 00 | `CONFIG`, mount, clone, `AtlasPipeline(CONFIG)` |
| 01 | `P.health()`, `P.discover()`, `P.preflight()`, `P.snapshot()` |
| 02 | `P.geospatial()`, `P.dem_inventory()`, optional `P.acquire()` |
| 03 | `P.raw_bands()` |
| 04 | `P.faults()` |
| 05-07 | `P.magnetic()`, `P.gravity()`, `P.terrain()` |
| 08 | `P.dem()`, `P.dem_examples()` |
| 09 | figures only (`F.persistence`, `F.strike_roses`) |
| 10 | `P.cross_modal()` |
| 11 | `P.proximity()`, `P.profiles()` |
| 12 | `P.quality()` |
| 13 | `P.lineaments()`, interactive map |
| 14 | `P.ml_ready()`, `GeologicalAtlas.open(...)` |
| 15-16 | `P.write_reports()`, `P.artifact_report()` |

Section 03 must run before 06 (gravity aliases point at the exported raw bands). Section 08 may need several
sessions in DETAILED_2M; FAST_10M is the default. Every runtime and output size is unmeasured until this runs.

Reload later without recomputation:
```python
from gems_bench.geological_atlas import GeologicalAtlas
atlas = GeologicalAtlas.open("/content/drive/MyDrive/GEMS_2/GEMS_GEOLOGICAL_DATA_ATLAS_V1")
features = atlas.load_features(families=["magnetic", "gravity", "terrain", "dem"])
coverage = atlas.load_coverage()
lineaments = atlas.load_lineaments()
```
