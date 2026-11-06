# G07 FAST COARSE × DTI — research-backed, low-compute screening

This is a narrow, bounded alternative to processing ~250 native 1 m DEM
tiles at 2 m with expensive terrain-structure filters and U-Net training.
It uses previously completed, cached **10 m** DEM parts, and creates a
small **100 m** analysis-grid feature store for W2-relative diagnostic DTI.

## How to start in Colab

Notebook: \`notebooks/colab/GEMS_G07_MULTISCALE_FAULT_DISCOVERY.ipynb\`
(branch: \`g07-dti-first-metrics\`)

In the config cell:

\`\`\`python
DTI_FIRST_MODE = True
FAST_COARSE_MODE = True
FAST_COARSE_MAX_TILES = 8
\`\`\`

Then initialize \`P\` normally, skip the original heavy G07 feasibility and
\`DEM_FEATURES\` steps (RUN flags are disabled in DTI-first mode), and execute
sections **08B** and **09B**. You may use the same Colab runtime; once updated
source is pulled, restart the Python runtime if older modules were already
imported.

**STOP and interrupt any currently running \`[pilot] fine tier ...\` cell.**
Do not delete \`dem_features/coarse_parts/\` or completed \`fine_*.tif\`.

No downloads, no native 1 m feature extraction, no GPU model training, and
no submission export are called by FAST_COARSE. The existing G07 B0/B1 vs W2
diagnostic remains in section 16A, independent of DEM.

If the notebook reports MISSING_COARSE_CACHE, it will not silently build
1 m products; stage existing saved \`dem_features/coarse_parts/\` or use the
available merged 10 m DEM to create coarse parts via a separate dedicated
conversion step. The 37.9% 1 m source *footprint* is not the same as 10 m
fast-feature coverage. The first successful fast run is intentionally a
sample of no more than eight cached tile IDs.

## Research → implementation mapping

| Research idea (from supplied review) | FAST implementation | Caveat |
|---|---|---|
| Multi-illuminated DEM lineaments | Four azimuths: 315, 45, 135, 225 deg; hillshade range | Descriptive proxy, not tectonic proof |
| Multiscale DEM edge detection | 30/60/150 m smoothing, slope break, curvature | Small scarps below 10–30 m can be lost |
| Continuous wavelet transforms | 30/150 m Difference-of-Gaussians bandpass \`dem_fast_dog_s60\` | NOT a CWT or implementation of the published wavelet method |
| Geological structural orientation | 60 m structure-tensor coherence | Direction-free coherence alone cannot determine fault strike |
| Terrain roughness / local relief | 60 m normalized local roughness | Topography is not proof of a fault |
| Magnetic / gravity edge CNN | Inventory already computed G06 RTP, gravity, tilt, structure / gradient feature names | No new CNN or recomputation |
| Distance-tolerant / imbalanced segmentation | Optional train-only soft target within 300 m, masked BCE + Tversky / optional focal Tversky | OFF by default; does not implement the canonical DTI exactly |
| Cross-region validation | Existing V1 withheld systems / V3 geographic units reused in 09B | Incomplete or data-sparse folds are NOT independently validated |
| Pretrained U-Net / DeepLabv3+ / SAM / SegFormer | Research backlog only | No heavy training unless fast DTI demonstrates added value |

The coefficients 0.2 (false positives) / 0.8 (false negatives) and 300 m
support radius are aligned with GEMS' existing competition-style metric.
The scorer is an approximation based on the repo's understanding of the
competition; official hidden-evaluation parity cannot be claimed.

## Pipeline & artifacts

\`\`\`text
Existing USGS 1m DEM (already downloaded)
        |
        +-- existing validated 10m cached parts (already created by old pilot)
                  |
                  +-- FAST research features at 10m (seven channels)
                           |
                           +-- 100m grid aggregate, strict no-data edge guards
                                 |
                                 +-- W2 baseline (same held-out unit / mask / budget)
                                 +-- W2 + 15% normalized DEM rank, only inside DEM support
                                           |
                                           +-- 300m DTI, weighted TP/FP/FN
                                           +-- recovered/lost fault systems
                                           +-- local vs full-region, explicitly separate
\`\`\`

Outputs (all synced to Drive under the G07 run):

- \`dem_features/fast_coarse/<fingerprint>/tiles/fast_<tile>.tif/.json\`
- \`dem_features/fast_coarse/<fingerprint>/dem_features_100m_fast.tif\`
- \`dem_features/fast_coarse/tiles_processed.csv\`
- \`dem_features/fast_coarse/manifest.json\`
- \`evaluation/dti_fast_coarse/coverage.csv\`
- \`evaluation/dti_fast_coarse/feature_local_rows.csv\`
- \`evaluation/dti_fast_coarse/feature_local_summary.csv\`
- \`evaluation/dti_fast_coarse/fusion_full_rows.csv\`
- \`evaluation/dti_fast_coarse/fusion_full_summary.csv\`
- \`evaluation/dti_fast_coarse/system_recovery.csv\`
- \`evaluation/dti_fast_coarse/g06_boundary_feature_inventory.csv\`
- \`figures/fast_coarse/fast_coarse_features.{png,svg}\`

**Selection discipline:** fixed feature signs, fixed 1% area, fixed 15% fusion
weight (before viewing held-out results). V1/V3 labels are used for scoring
only; do not choose a winner based on V1/V3 then report it as independent.
The result tier is SCREENING_ONLY. DEM features can be insufficient if cached
coarse data cover too few distinct fault systems.

## Research backlog if fixed-policy ΔDTI is positive

1. Increase geographically representative cached 10 m coverage (not native
   2 m processing), and repeat at the *same* fixed settings.
2. Compare W2 with existing G06 geophysical boundary candidates (THD/HGM,
   tilt derivative, anisotropy, RTP and grav) on matched units. Reuse cached
   feature store; do not calculate redundant transforms.
3. Run stricter held-out spatial/system evaluation for any promising
   geophysical or DEM fusion. Pool weighted DTI components only on **complete**
   units.
4. If and only if the gains persist, consider a 100 m pre-trained U-Net with
   masked BCE + SoftTversky + optional FocalTversky; compare against W2.
5. Defer DeepLabv3+, U-Net++, ViT fusion, SegFormer, SAM adapters and synthetic
   pretraining until a lightweight model establishes a real improvement.

## Sources provided for the research plan

The following articles were supplied as an external research shortlist.
Their methods are *candidate inspirations*, not evidence of measured G07
improvement; exact bibliographic details have not been independently audited.

- *A Large-scale Benchmark on Geological Fault Delineation Models* (2025):
  https://arxiv.org/html/2505.08585
- *Seismic Fault SAM* (2024):
  https://arxiv.org/abs/2407.14121
- *Lineament extraction from multi-illuminated DEM hillshades* (2024):
  https://link.springer.com/article/10.1007/s11004-024-10146-5
- *Improved U-Net for automatic seismic fault identification* (2023):
  https://link.springer.com/article/10.1007/s11600-023-01200-7
- *CRACKS* (2024):
  https://arxiv.org/abs/2408.11185
- *Edge recognition of magnetic anomaly sources with CNNs* (2025):
  https://link.springer.com/article/10.1007/s11600-025-01537-1

## Validation status

Code and notebook structure were authored and reviewed via GitHub source
readback only. **No scripts, tests, notebooks, downloads, or model training
were run locally.** Synthetic tests are added to
\`tests/test_g07_fast_coarse.py\`. Runtime success, speeds and DTI improvements
must be measured in Colab.
