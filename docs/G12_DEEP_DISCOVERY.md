# GEMS G12 Deep Geological Signature Discovery

## What this adds

G12 is a **fully independent feature/EDA/ablation/submission workflow** in the existing `gems_bench` package on `main`, not a dependency on the unavailable G11 branch. It runs on the original 19-band GeoDAWN raster and the two official label/template rasters. No legacy recovery, and no training-data labels in engineered features.

### Feature engineering

- Mask-normalized multiscale smoothing at **200, 600 and 1,200 m**; anomaly z contrast, gradient log-energy, orientation tensor coherence, Laplacian response and tilt-angle proxy.
- Directional **cross-contact asymmetry**, sampling both sides of local gradient normals, and **axial orientation persistence** from fine to coarse scales.
- Magnetic, gravity, conductivity, DEM/elevation and depth families where available, with pairwise magnetic–gravity, magnetic–conductivity, gravity–conductivity and magnetic–terrain **aligned-edge coincidence** and relative response discordance.
- Explicit validity/coverage features to surface tile-footprint artifacts and missing DEM instead of silently treating missing inputs as zero.
- **Chunked, 96 px halo** raster computation, atomic intermediate write, cache/source fingerprint checks, aligned GeoTIFF metadata.

### Evaluation and analysis

- Geographic block train/validation/test masks with an **embargo between training and holdout geography**. The held-out test is not used in model selection.
- Paired raw-only and family incremental XGBoost ablations. All models use identical settings, splits and sampling policy. Case-control *sampled* PR-AUC is labeled as a diagnostic, NOT a population metric.
- Full validation-grid continuous DTI using the existing `gems_bench.metric` implementation. **Labels are held-out known faults, not official private new-fault labels: this is a proxy.**
- Fault evidence atlas with mapped traces, unverified lineaments and missed mapped traces, plus robust per-feature contrast table. Unlabeled lineaments must not be declared true/false until geological review.
- A complete official-grid **single-band float32** submission, NaN outside the template, known faults masked pixel-exactly, confidence range [0,1]. It is not uploaded automatically.

### Input contract

Set the three official paths: `gems-geodawn-numerical-features.tif`, `existing_faults.tif`, `example_submission.tif`. All must share shape, transform and EPSG:32611 100 m grid. Raw geophysical raster must carry original band descriptions (`rtp`, `iso_grav_anom`, etc); absent optional modalities are skipped and recorded.

### Run in Colab

Open [`notebooks/colab/GEMS_G12_DEEP_DISCOVERY.ipynb`](../notebooks/colab/GEMS_G12_DEEP_DISCOVERY.ipynb). It mounts Drive, clones `main`, installs non-Torch dependencies and discovers unique official filenames under `GEMS_2`. If more than one matching filename exists, specify the paths in `CFG` rather than using a possibly incorrect duplicate. Set `CFG['groups']` to `('raw','all')` for the fastest initial benchmark. The default complete ablation is `('raw','mag','grav','cross','all')`.

### CLI equivalent

```bash
python -m pip install -e '.[m9]'
python -m gems_bench.g12.deep \
  --raw '/path/gems-geodawn-numerical-features.tif' \
  --known '/path/existing_faults.tif' \
  --template '/path/example_submission.tif' \
  --out '/path/G12_DEEP_DISCOVERY' \
  --groups raw,mag,grav,cross,all
```

### Outputs

`physics_g12.tif`, `physics_g12.manifest.json`, `models/*.json`, `predictions/*.tif`, `ablation_results.json`, `atlas/feature_contrast.json`, `atlas/fault_evidence_atlas.png`, `submission_g12_deep.tif`, `report.json`.

### Important limitations

Synthetic unit tests validate feature shape/coverage, grid handling, spatial embargo, cache integrity and submission masking. No full-data run, notebook GPU execution or leaderboard submission has been performed here. The offline DTI does not replicate the private label distribution. Geographically adjacent validation and test blocks can share geological settings; a **true whole-fault heldout evaluation** and detailed regime-stratified diagnostics should be added before interpreting it as a reliable novelty estimate. `n_estimators=280` and five ablations may take time on a Colab CPU; start with two models if needed.

Test with `pytest -q tests/test_g12_deep.py`.