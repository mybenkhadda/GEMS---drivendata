# GEMS Geological Feature Discovery Studio (Streamlit)

Interactive laboratory to **create, test, compare and export geological features** that could improve fault detection
on the GEMS native 100 m grid. Upgrade of the Geological Layer Lab (the old `lab/formulas.py`, `lab/export_full.py`
and their tests are kept unchanged for backward compatibility).

## Launch

| where | how |
|---|---|
| Colab | open `Launch_GEMS_Layer_Lab_Colab.ipynb`: mount Drive → clone the repo (pin `REF`) → `pip install -r requirements.txt` → start Streamlit → `serve_kernel_port_as_window(8501)` |
| Windows | `start_windows.bat` (or `python -m streamlit run app.py`), sidebar root = your synced GEMS_2, e.g. `G:/My Drive/GEMS_2` |
| Linux / macOS | `./start_linux_mac.sh` |

**Rasters (sidebar file selectors):** *Features GeoTIFF*, *Known faults GeoTIFF* and an optional *Submission
template* are dropdowns of the `.tif` files found next to `app.py` and under the GEMS_2 root (`data/raw/...`), plus
**📁 Custom path…** (a text box with a placeholder path). The `gems-geodawn-numerical-features.tif` and
`existing_faults.tif` shipped in this folder are selected by default. *GEMS_2 root* only decides where outputs
are saved (empty = this folder). Without real rasters, toggle **Synthetic demo data** → *Generate demo rasters*.

## Seven pages

1. **Map Explorer** – overview with drag-box AOI, up to 4 synchronised maps (shared zoom), hover values, colour map /
   percentile stretch, fault overlay, statistics with units, raw vs transformed, **cross-section explorer** (multi-signal
   aligned profiles, fault-crossing markers, display-only normalisation, CSV / PNG / SVG).
2. **Python Feature Laboratory** – multi-line formula editor, `PARAMS` sliders, ▶ Run Features, every variable becomes a
   feature (maps, diagnostics for NaN / Inf / constant / outliers / division instability, comparison with its sources,
   diff vs the previous parameter setting, quick fault evaluation, freezing of AOI statistics, save to the library),
   trusted plugins and the function reference.
3. **Fault Association** – leaderboard (point-biserial, Spearman, AUROC, AP, AP lift, recall @ 0.5 / 1 / 2 / 5 / 10 %,
   median difference, Cliff's delta, robust effect, near-trace enrichment, sign, support) with fixed / frozen
   (development split) / flagged-exploratory direction; distributions; distance-band quantile envelopes; fault-normal
   profiles; formal comparison (2 km block bootstrap, shifted-target negative controls, physically matched controls).
4. **Local & Regional** – tile heatmaps (2 × 2 … 10 × 10, 1–20 km tiles, label-free EDA domains), INSUFFICIENT_DATA tiles,
   click-to-inspect tile vs global, moving-window point-biserial map.
5. **Tournament** – template + parameter grid, EXPLORE (all combos, regional stability, null) vs VALIDATE (spatial folds:
   direction + parameters chosen on development folds only); resumable from `tournaments/<id>/`.
6. **W2 / M9 Complementarity** – baseline vs feature at identical area budgets: recovered by both / one / neither,
   unique feature recoveries, extra unlabelled area, rank-average and half-and-half marginal recall, category map,
   provenance badge (held-out only with `<raster>.provenance.json`). Not DTI.
7. **Library & Export** – registry, versions / history, dependency graph, import / export / duplicate / rename / delete /
   status, AOI GeoTIFF download, full-footprint native GeoTIFF export, P6 / M9 snippet.

## Working example (page 2)

```python
PARAMS = {"sigma": {"min": 1, "max": 20, "step": 1, "default": 3},
          "eps": {"min": 0.001, "max": 0.2, "step": 0.001, "default": 0.01}}
mag = gradmag(rtp, sigma=sigma)
grav = gradmag(iso_grav_anom, sigma=sigma)
ratio_mg = ratio(mag, grav, eps=eps)
terrain = highpass(det_elev, sigma=8)
feature_A = abs(terrain) * ratio_mg
mag_fabric = orientation(rtp, sigma=5)
terrain_fabric = orientation(det_elev, sigma=5)
fabric_agreement = axial_agreement(mag_fabric, terrain_fabric)
feature_B = custom_disruption(det_elev, det_elev_slope, sigma=4)     # trusted plugin (plugins/example_features.py)
```

▶ Run Features → all variables appear; move the `sigma` slider → only dependent variables recompute.

## Two authoring modes

* **Mode A – restricted formulas** (`lab/featurelang.py`): AST interpreter; assignments, numbers, + − × ÷, constant
  powers, comparisons (NaN-propagating), `& | ~`, calls of `lab.ops.OPS` and registered plugins. No imports,
  attributes, subscripts, loops, comprehensions, lambdas or unknown functions. `PARAMS` is a strict literal. Mapped
  faults (`target`) and W2 / M9 maps (`prediction`) are refused as inputs.
* **Mode B – trusted plugins** (`lab/plugins.py`, `plugins/*.py`): full Python with `@register_feature(name, inputs,
  parameters, halo_px, ...)`, loaded only from local directories (`plugins/`, `$GEMS_LAB_PLUGIN_DIR`, sidebar). **Not
  sandboxed**; never upload untrusted code. Call: one array per declared input (positional), then keyword parameters.

## Conventions

sigma / size in **pixels** (1 px = 100 m); gradients per pixel (east / north); angles in degrees clockwise from north,
axial [0, 180); NaN = invalid and propagates (neighbourhood filters NaN-normalised). AOI-dependent normalisations
(`zscore` without `center` / `scale`, `percentile_rank`, `minmax` without `lo` / `hi`, …) work interactively but must be
frozen (page 2) on a development AOI before full-grid export. Full table: page 2 → Function reference.

## Outputs — `GEMS_2/outputs/geological_layer_lab/`

`registry/feature_registry.json` (+ `last_program.py`, dependency graph) · `tables/<run>/` (CSV + Parquet) ·
`figures/<run>/` (PNG + SVG + `data/*.csv`) · `tournaments/<id>/` (results.csv / .parquet, manifest.json) ·
`transects/` · `exports/<name>_v<k>_<hash>/` (native GeoTIFFs + `export_manifest.json`, never overwritten) ·
`manifests/`. The original GEMS rasters are never written.

## Reuse in P6 / M9

```python
import sys; sys.path.insert(0, "/content/GEMS/streamlit/gems_layer_lab")
from lab.integration import FeatureStore
fs = FeatureStore("/content/drive/MyDrive/GEMS_2")
x = fs.compute("feature_A")            # full native grid (tiled, same engine), float32, NaN = invalid
fs.export(["feature_A"])               # or: python -m lab.export_native --root ... --features feature_A
```

## Tests (synthetic)

`python -m pytest -q tests` from this directory (`tests/test_layer_lab.py` legacy, `tests/test_feature_studio.py` new).

## Scientific guardrails

Unmapped ≠ non-fault; AUROC / AP rank mapped faults above unlabelled pixels and are not DTI; pixels are spatially
dependent (use block bootstrap / negative controls); choose formula, direction, parameters and thresholds on
development data only (frozen mode, VALIDATE tournaments); distance-to-known-fault and fault-normal profiles are
retrospective diagnostics, never predictors; complementarity needs held-out baseline provenance for any claim; the
official scorer remains the only DTI.
