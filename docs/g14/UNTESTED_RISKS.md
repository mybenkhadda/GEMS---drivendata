# G14 untested risks

**NO CODE, TESTS, NOTEBOOKS, TRAINING, INFERENCE OR EXPERIMENTS WERE EXECUTED.** The code was written and reviewed
by reading only. Expect first-run failures. Below are the places most likely to break, the symptom, and where to look.

| # | area | risk | symptom | where |
|---|---|---|---|---|
| 1 | imports | a G13 / P6 / gems3 signature assumed from reading differs at runtime | ImportError / TypeError in cell 1 or 7 | `pipeline.py`, `evaluation.py` |
| 2 | synthetic fixtures | tests assume `g13.synthetic.make` geometry (400 px grid, conductivity gap rows 300–340) | test assertion failures unrelated to G14 logic | `tests/test_g14_core.py` |
| 3 | memory | full stack at level 1 is C × 3730 × 3292 float16 (≈ 25 MB per channel) | Colab OOM when C > ~150 | `contract.build_stack`; reduce channels |
| 4 | known context | `p6.stack.known_features` cost on full grid per unit | slow unit setup | `multiscale.UnitInputs` |
| 5 | lattice / alignment | off-by-one in view tops when centre is not divisible by 8 | misaligned context (silent) | `view_tops`; `test_views_share_centre_and_exact_fraction` |
| 6 | Swin padding | regional size must be divisible by patch × window × 2^3 | shape error in SwinEncoder | `model.SwinEncoder`; keep 256 / 4 / 8 |
| 7 | AMP | fp16 overflow in attention on T4 | NaN loss | `train.amp_setup`; set `train.amp=False` (bf16 is used automatically where supported) |
| 8 | clDice | soft skeleton iterations vs thin traces at 100 m | loss ≈ constant | `losses.soft_skeleton`; lower weight |
| 9 | DTI surrogate | equivalence to official metric only argued, not checked | mismatch | `test_dti_surrogate_matches_official_on_binary` |
| 10 | augmentation | axial / vector sign conventions (image y down vs north up) | worse training with augment on | `multiscale.axial_update` / `vector_update`; ablate `augment=False` |
| 11 | phase5 backend | context loading paths / label equality check | ValueError at `system()` | `splits.UnitSystem` |
| 12 | G13 split rebuild | rebuilding splits with G13 defaults may not reproduce the run's fold ids | different units than G13 run | load the saved splits from `g13_artifact_dirs` first |
| 13 | inference cost | full-grid tiling with TTA + MC dropout | hours per map | lower `mc_samples`, disable TTA |
| 14 | ensemble / lgbm | OOF blend feature construction | shape errors | `ensemble.oof_blend` |
| 15 | regime gate | simplified fixed 5 % budget | weak gating | `ensemble.regime_gate` |
| 16 | submission | `export_candidate` expectations (NaN/0 conventions) | validation fails | `submission.prepare_submission` |
| 17 | checkpoint resume | rng state restore on CUDA | resume error | `train.fit_unit` |
| 18 | pretrained weights | `strict=False` loading may silently load nothing | no improvement | check reported missing keys |
| 19 | time budget | wall-clock stop leaves the best checkpoint, not the latest | expected; verify the manifest | `train.fit_unit` |
| 20 | scientific | context and fusion may not help on these weak signals | T2–T4 ≈ T1 | the experiment ladder is designed to show this honestly |
