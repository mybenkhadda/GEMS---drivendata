"""Section 10 - are the two specialists complementary, or one output on different confidence scales?

Heads from the cached full-area inference: ``global`` = sigmoid(global_logits), ``expert0/1`` =
sigmoid(experts), ``route0`` = router weight of expert 0, ``specialist_weight`` = mixture strength and the
final ``prob``. Evidence used (fixed in ``config['experts']`` before any result):

* rank agreement (Spearman) and top-1 % overlap (Jaccard) between heads on block-sampled pixels
  (one pixel per 1 km block - neighbouring pixels are not independent samples);
* scale difference at equal rank (median log ratio) - high rank agreement + different scale = "scale only";
* how much the mixture changes the global head (|prob - global|, top-1 % membership change);
* router behaviour: entropy, regional assignment, and its rank correlation with the label-free
  magnetic-minus-gravity persistence prior it was trained toward (a router that only reproduces its prior
  has not learned a new partition);
* withheld-label performance of every head (full-area discovery protocol) and per structural environment.
A textured router map is never taken as evidence of specialisation by itself.
"""
from __future__ import annotations

import numpy as np

from . import metrics as MX

HEADS = ("global", "expert0", "expert1", "prob")
PAIRS = (("expert0", "expert1"), ("global", "expert0"), ("global", "expert1"), ("global", "prob"),
         ("expert0", "prob"), ("expert1", "prob"))


def block_sample(valid: np.ndarray, block_px: int, seed: int):
    """One random valid pixel per ``block_px`` block (deterministic)."""
    rng = np.random.default_rng(seed)
    h, w = valid.shape
    r = rng.random(valid.shape, dtype=np.float32)
    r[~valid] = 2.0
    nbc = (w + block_px - 1) // block_px
    ri, ci = np.indices(valid.shape, dtype=np.int32)
    bid = ((ri // block_px) * nbc + ci // block_px).ravel()
    order = np.lexsort((r.ravel(), bid))
    _, first = np.unique(bid[order], return_index=True)
    pick = order[first]
    pick = pick[r.ravel()[pick] < 2.0]
    return np.unravel_index(pick, valid.shape)


def regime_prior(bank) -> np.ndarray:
    """V1 ``Bank.regime_prior`` on the whole grid (label-free magnetic - gravity edge persistence)."""
    m = np.asarray(bank.array[bank.names.index("mag_edge_persistence")], np.float32)
    g = np.asarray(bank.array[bank.names.index("grav_edge_persistence")], np.float32)
    return np.clip(0.5 + 1.5 * (np.nan_to_num(m, nan=0) - np.nan_to_num(g, nan=0)), 0.05, 0.95)


ENV_BINS = ((0.0, 0.35, "gravity-dominant edges"), (0.35, 0.65, "mixed / weak edges"), (0.65, 1.01, "magnetic-dominant edges"))


def _jaccard_top(a, b, mask, frac):
    ta = a >= np.quantile(a[mask], 1 - frac)
    tb = b >= np.quantile(b[mask], 1 - frac)
    ta &= mask
    tb &= mask
    u = (ta | tb).sum()
    return float((ta & tb).sum() / u) if u else np.nan


def analyse_unit(unit, heads: dict, valid, labels, region, prior, sample, cfg, units=None, exclusion="training_visible"):
    from scipy.stats import spearmanr

    E = cfg["experts"]
    elig = valid & ~labels
    rr, cc = sample
    sim, div, route = [], [], []
    for h in HEADS:
        v = heads[h][elig]
        div.append({"unit": unit, "head": h, "mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                    "q99": float(np.nanquantile(v, 0.99)), "q999": float(np.nanquantile(v, 0.999)),
                    "std": float(np.nanstd(v)), "mass": float(np.nansum(v)), "frac_above_0.5": float(np.nanmean(v >= 0.5))})
    for a, b in PAIRS:
        x, y = heads[a][rr, cc], heads[b][rr, cc]
        ok = np.isfinite(x) & np.isfinite(y)
        rho = float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() > 10 else np.nan
        pr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else np.nan
        lr = np.log(np.clip(y[ok], 1e-6, 1)) - np.log(np.clip(x[ok], 1e-6, 1))
        sim.append({"unit": unit, "head_a": a, "head_b": b, "spearman": rho, "pearson": pr,
                    "jaccard_top": _jaccard_top(np.nan_to_num(heads[a]), np.nan_to_num(heads[b]), elig, E["top_fraction"]),
                    "median_log_ratio_b_over_a": float(np.median(lr)) if lr.size else np.nan,
                    "mean_abs_diff": float(np.nanmean(np.abs(heads[a][elig] - heads[b][elig]))), "n_sample": int(ok.sum())})
    r0, s = heads["route0"][elig], heads["specialist_weight"][elig]
    rc = np.clip(r0, 1e-6, 1 - 1e-6)
    H = -(rc * np.log2(rc) + (1 - rc) * np.log2(1 - rc))
    dmix = np.abs(heads["prob"][elig] - heads["global"][elig])
    rel = dmix > E["material_change"] * np.maximum(heads["global"][elig], 1e-3)
    pri = prior[rr, cc]
    rs = heads["route0"][rr, cc]
    ok = np.isfinite(rs)
    rho_prior = float(spearmanr(rs[ok], pri[ok]).correlation) if ok.sum() > 10 else np.nan
    q = {"route0_mean": float(np.nanmean(r0)), "route0_std": float(np.nanstd(r0)),
         "route0_frac_above_0.5": float(np.nanmean(r0 > 0.5)), "route_entropy_mean_bits": float(np.nanmean(H)),
         "strength_mean": float(np.nanmean(s)), "strength_median": float(np.nanmedian(s)),
         "strength_q90": float(np.nanquantile(s, 0.9)), "mixing_abs_change_mean": float(np.nanmean(dmix)),
         "mixing_frac_material_change": float(np.nanmean(rel)),
         "mixing_top1pct_membership_change": 1 - _jaccard_top(np.nan_to_num(heads["prob"]),
                                                              np.nan_to_num(heads["global"]), elig, E["top_fraction"]),
         "router_vs_label_free_prior_spearman": rho_prior}
    for k in np.unique(region[valid & (region >= 0)]):
        m = elig & (region == k)
        q[f"route0_mean_region_{int(k)}"] = float(np.nanmean(heads["route0"][m]))
        q[f"strength_mean_region_{int(k)}"] = float(np.nanmean(heads["specialist_weight"][m]))
    for lo, hi, name in ENV_BINS:
        m = elig & (prior >= lo) & (prior < hi)
        q[f"route0_mean_env_{name}"] = float(np.nanmean(heads["route0"][m])) if m.any() else np.nan
    for k, v in q.items():
        route.append({"unit": unit, "quantity": k, "value": v})
    s01 = [r for r in sim if r["head_a"] == "expert0" and r["head_b"] == "expert1"][0]
    if s01["spearman"] >= E["redundant_spearman"] and s01["jaccard_top"] >= E["redundant_jaccard"]:
        verdict = "REDUNDANT_OR_SCALE_ONLY"
    elif q["strength_median"] < E["weak_mixing_strength"] and q["mixing_top1pct_membership_change"] < 0.1:
        verdict = "WEAK_MIXING (specialists barely change the global head)"
    elif s01["spearman"] < E["complementary_spearman"] and s01["jaccard_top"] < E["complementary_jaccard"]:
        verdict = "DIVERGENT_HEADS (necessary, not sufficient, for complementarity)"
    else:
        verdict = "PARTIALLY_OVERLAPPING"
    summary = {"unit": unit, "verdict": verdict, "spearman_e0_e1": s01["spearman"], "jaccard_top_e0_e1": s01["jaccard_top"],
               "median_log_ratio_e1_over_e0": s01["median_log_ratio_b_over_a"], "strength_median": q["strength_median"],
               "route_entropy_mean_bits": q["route_entropy_mean_bits"],
               "mixing_top1pct_membership_change": q["mixing_top1pct_membership_change"],
               "router_vs_label_free_prior_spearman": rho_prior}
    perf, env = [], []
    if units is not None and unit in units and units[unit].protocol == "WITHHELD_SYSTEM":
        U = units[unit]
        cm = MX.ComponentMaps(U.truth, valid, U.exclusion(exclusion))
        for h in HEADS:
            mp = cm.maps(heads[h])
            c = mp["components"]
            perf.append(MX.row(c, unit=unit, head=h, protocol="DISCOVERY_FULL_AREA", exclusion=exclusion))
            G = mp["truth"]
            for lo, hi, name in ENV_BINS:
                m = (prior >= lo) & (prior < hi)
                n = int((G & m).sum())
                env.append({"unit": unit, "head": h, "environment": name, "n_truth": n,
                            "recall_w": float(mp["tp_support"][G & m].sum() / n) if n else np.nan,
                            "fp_mass_share": float(mp["fp_mass"][m].sum() / max(mp["fp_mass"].sum(), 1e-12))})
        best = max(perf, key=lambda r: r["dti"])
        summary.update(best_head_withheld=best["head"], dti_prob=[r for r in perf if r["head"] == "prob"][0]["dti"],
                       dti_global=[r for r in perf if r["head"] == "global"][0]["dti"])
    return {"similarity": sim, "diversity": div, "routing": route, "summary": summary, "performance": perf,
            "environment": env}
