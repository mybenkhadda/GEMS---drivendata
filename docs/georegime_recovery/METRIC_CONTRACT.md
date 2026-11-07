# Metric contract

The implementation is the repository metric, `gems_bench/metric.py` (last changed in faec7ba). There is no new metric. `metrics.check_constants` refuses to start if the config differs from the module constants.

```text
k(d)  = max(1 - d/R, 0),  R = 300 m (3 px)          alpha = 0.2, beta = 0.8, eps = 1e-6
M(g)  = max_x p(x) k(d(x,g))                        (local_weighted_max)
TP_w  = sum_{g in G} M(g)       FN_w = |G| - TP_w
FP_w  = sum_{x in eval & ~excl} p(x) [1 - k(dist(x, G))]
DTI   = TP_w / (TP_w + alpha FP_w + beta FN_w + eps)
```

| property | implementation |
|---|---|
| Known-fault exclusion | `exclude_mask`, pixel-exact. `exclude_supports_tp = OFFICIAL_EXCLUDE_SUPPORTS_TP = False`: an excluded pixel is zeroed before the local max, so it is neither truth, FP, nor TP support. This is an organizer-statement reading, not verified against a public scorer; it is recorded in `03_validation/scorer_settings.json` and in every report. |
| Eligible mask | `eval_mask`: predictions outside it are zeroed before the local max, so no TP can come from outside the footprint. |
| Boundary support | The crop pads by ceil(R / px) + 1 pixels, so support near the evaluation edge is computed with zeros outside. |
| NaN / invalid | `clean_prediction`: non-finite values become 0, then values are clipped to [0, 1]. |
| Pooling | Component sums (`DTIComponents.__add__`, `pool`). |
| Weighted recall / precision | `tp/|G|`, `tp/(tp+fp)` (`as_dict`). |
| Prediction mass | `p[eval].sum()` after zeroing. |

## Pixel maps (`metrics.ComponentMaps`)
The maps use the same `MaskedScorer` triple, the same zeroing and the same kernel:
* `tp_support` = M(g) on G;
* `fp_mass` = p(1 - k) on eval & ~excl;
* `fn_support` = 1 - M(g).

Every call checks that the map sums equal `MaskedScorer.components` within 1e-6 relative, and raises `MetricConsistencyError` otherwise. Regional and block DTIs are computed from summed map components.

Every figure and table that shows a DTI labels it "offline DTI proxy (NOT the official leaderboard DTI)".
