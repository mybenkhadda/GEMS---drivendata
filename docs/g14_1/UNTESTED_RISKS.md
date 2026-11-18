# Remaining blockers and uncertainties

**NO NOTEBOOKS, TESTS, PYTHON CODE, TRAINING, INFERENCE, OR EXPERIMENTS HAVE BEEN EXECUTED.**

## Blockers that need user action or data

1. **Hash reproduction.** The semantic config of `T1_local_cnn__da3f44d05d84` has to be reproduced. If the original
   notebook used semantic OVERRIDES, copy them into `G14_OVERRIDES`; otherwise `g14_config_for` raises. Every stage
   that uses the G14 data layer depends on this, R00 inventory included, because it loads the official inputs.
2. **Channel stack for R06 and R08+.** `stack_l*.npy` was never synced. Rebuilding it must give the same schema hash.
   If the original run used a G13 feature store that is no longer present, the rebuild differs and checkpoint loading
   is refused.
3. **Discovery maps.** The G14 run produced geographic maps only. Without discovery maps, every G14 operating point is
   PROVISIONAL_GEO; G13 `disc*.npy` maps (if present) provide the discovery area optimum for comparison.
4. **M6.** Needs `corrected_models/champion_nested/<unit>.npy`: champion ranks from a champion that never saw the
   unit's held-out zone. None exists yet.

## Likely first-run failures (code read but never run)

| area | risk |
|---|---|
| `pipeline.G141Pipeline` | attribute or signature mismatches with G13 / G14 / P6 objects read but not called |
| R03 cost | 6 representations × 20 areas × 2 orders × units: crest maps are computed once per unit and order, but `hysteresis` / `segment_rank` in STANDARD are slower |
| R06 cost | full-region inference of the inner region per intervention (6 × units); STANDARD limits it to one unit |
| `calibration.isotonic` | large samples; sklearn memory |
| `atlas` | many figures; raw bands are loaded at full resolution |
| `training.fit_corrected` | DataLoader workers with large closures (`num_workers=0` is the fallback); per-epoch context recomputation (EDT on the full grid, a few seconds) |
| `training.mine_hard_negatives` | builds a probe TrainSpec; not exercised by tests |
| `architectures.LocalRegionalCNN` | regional encoder downsampling of a 256 px view to 16 px before alignment (coarse) |
| `losses.nnpu` | simplified nnPU (no gradient-ascent step); the class prior (default 0.01) is a guess |
| notebook | JSON hand-written; open it once in Colab to confirm it parses |

## Scientific uncertainties

* The static mechanism of the shortcut is certain. How much of the LB failure it explains is not: that is R02 / R06
  / D1.
* Geophysical signals were weak in every earlier experiment (G13 AUC ≤ 0.6). Removing the shortcut may give an honest
  but low score. The 0.40 target is not implied by anything here.
* The organiser-statement reading of the known mask (`exclude_supports_tp=False`) is unverified against a public
  scorer.
* The discovery proxy (withheld mapped systems) may differ from the hidden new faults: mapped systems are biased
  toward visible, long structures.

## Process note

Twice during authoring, `python3` was invoked with an **empty** script by mistake: once while inspecting the G14
notebook, and once during G14.1. No repository code, test or notebook ran. All other command-line operations were
`git`.
