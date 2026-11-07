"""Section 15 - cross-fitted policy choice, block-bootstrap uncertainty and pre-registered promotion gates.

For each discovery fold f: the policy with the highest POOLED DTI (summed TP/FP/FN) on the other folds is
chosen (ties -> earlier in the config list, identity first) and applied unchanged to f. Pooled
cross-fitted DTI = sum of the untouched held-out components. The gates in ``config['gates']`` were fixed
before any score; if any fails - or the choice is identity - the decision is
``NO RECOVERY PROMOTION - DO NOT SUBMIT``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import dti_from

NO_PROMOTION = "NO RECOVERY PROMOTION — DO NOT SUBMIT"


def _pooled(df) -> float:
    return dti_from(df.tp_w.sum(), df.fp_w.sum(), df.fn_w.sum())


def decide(comp: pd.DataFrame, blocks: pd.DataFrame, regions: pd.DataFrame, policies, cfg: dict,
           provenance_ok: bool, provenance_detail: str, label_safe: dict) -> dict:
    G = cfg["gates"]
    order = {p["name"]: i for i, p in enumerate(policies)}
    sel = cfg["transforms"]["selection_protocol"]
    disc = comp[comp.protocol == sel]
    folds = sorted(disc.unit.unique())
    names = [p["name"] for p in policies if p["name"] in set(disc.policy)]

    def best(df):
        scores = {n: _pooled(df[df.policy == n]) for n in names}
        return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0], scores

    rows = []
    for f in folds:
        (choice, other), _ = best(disc[disc.unit != f])
        held = disc[disc.unit == f]
        c, i = held[held.policy == choice], held[held.policy == "identity"]
        rows.append({"fold": f, "chosen_policy": choice, "dti_other_folds": other,
                     "dti_heldout_chosen": _pooled(c), "dti_heldout_identity": _pooled(i),
                     "delta_dti": _pooled(c) - _pooled(i), "tp_w": float(c.tp_w.sum()), "fp_w": float(c.fp_w.sum()),
                     "fn_w": float(c.fn_w.sum())})
    per = pd.DataFrame(rows)
    per["relative_change"] = per.delta_dti / per.dti_heldout_identity.replace(0, np.nan)
    cf = dti_from(per.tp_w.sum(), per.fp_w.sum(), per.fn_w.sum())
    ident = _pooled(disc[disc.policy == "identity"])
    (all_choice, all_score), all_scores = best(disc)
    # block bootstrap of the pooled cross-fitted minus identity difference (paired, resampled within folds)
    rng = np.random.default_rng(int(cfg["seed"]))
    B = int(G["n_bootstrap"])
    deltas = np.full(B, np.nan)
    if len(per):
        prep = []
        for r in per.itertuples():
            bc = blocks[(blocks.unit == r.fold) & (blocks.policy == r.chosen_policy) & (blocks.protocol == sel)]
            bi = blocks[(blocks.unit == r.fold) & (blocks.policy == "identity") & (blocks.protocol == sel)]
            bc, bi = bc.set_index("block").sort_index(), bi.set_index("block").sort_index()
            prep.append((bc[["tp_w", "fp_w", "fn_w"]].to_numpy(), bi[["tp_w", "fp_w", "fn_w"]].to_numpy()))
        for b in range(B):
            sc, si = np.zeros(3), np.zeros(3)
            for xc, xi in prep:
                ix = rng.integers(0, len(xc), len(xc))
                sc += xc[ix].sum(0)
                si += xi[ix].sum(0)
            deltas[b] = dti_from(*sc) - dti_from(*si)
    a = (1 - float(G["ci"])) / 2
    ci = (float(np.nanquantile(deltas, a)), float(np.nanquantile(deltas, 1 - a))) if np.isfinite(deltas).any() else (np.nan, np.nan)
    # secondary: geographic full-area folds with the all-fold choice
    geo = comp[comp.protocol == "GEOGRAPHIC_FULL_AREA"]
    geo_id = _pooled(geo[geo.policy == "identity"]) if len(geo) else np.nan
    geo_ch = _pooled(geo[geo.policy == all_choice]) if len(geo) else np.nan
    geo_rel = (geo_ch - geo_id) / geo_id if len(geo) and geo_id > 0 else np.nan
    # regional effect of the all-fold choice (pooled over discovery folds)
    reg = []
    rd = regions[regions.protocol == sel] if len(regions) else regions
    for r in sorted(rd.region.unique()) if len(rd) else []:
        x = rd[rd.region == r]
        reg.append({"region": int(r), "dti_identity": _pooled(x[x.policy == "identity"]),
                    "dti_choice": _pooled(x[x.policy == all_choice])})
    reg = pd.DataFrame(reg)
    if len(reg):
        reg["delta_dti"] = reg.dti_choice - reg.dti_identity
    unanimous = per.chosen_policy.nunique() == 1 and (per.chosen_policy == all_choice).all() if len(per) else False
    rel_gain = (cf - ident) / ident if ident > 0 else np.nan
    gates = [
        ("cross-fitted pooled gain >= %g relative" % G["min_relative_gain"],
         bool(np.isfinite(rel_gain) and rel_gain >= G["min_relative_gain"]), f"{rel_gain:+.4f}" if np.isfinite(rel_gain) else "n/a"),
        ("no held-out fold loses more than %g relative" % G["max_fold_relative_loss"],
         bool(len(per) and (per.relative_change.fillna(-np.inf) >= -G["max_fold_relative_loss"]).all()),
         ", ".join(f"{r.fold}: {r.relative_change:+.4f}" for r in per.itertuples())),
        ("same policy chosen on every fold and on all folds", bool(unanimous) or not G["require_unanimous_choice"],
         f"per fold {per.chosen_policy.tolist()}, all folds {all_choice}"),
        ("chosen policy is not identity", all_choice != "identity" and bool(len(per)) and (per.chosen_policy != "identity").all(),
         all_choice),
        ("geographic full-area folds not degraded beyond %g" % G["max_geographic_relative_loss"],
         bool(np.isfinite(geo_rel) and geo_rel >= -G["max_geographic_relative_loss"]),
         f"{geo_rel:+.4f}" if np.isfinite(geo_rel) else "n/a"),
        ("bootstrap %d%% CI lower bound of pooled gain > 0" % round(100 * G["ci"]),
         bool(np.isfinite(ci[0]) and ci[0] > 0) or not G["require_ci_lower_above_zero"], f"[{ci[0]:+.5f}, {ci[1]:+.5f}]"),
        ("transform is label-free", bool(label_safe.get(all_choice, False)), str(label_safe.get(all_choice))),
        ("checkpoint / split / full-area inference provenance verified", bool(provenance_ok), provenance_detail),
    ]
    gt = pd.DataFrame(gates, columns=["gate", "passed", "detail"])
    promote = bool(gt.passed.all())
    decision = f"PROMOTE {all_choice} (subject to the export guard)" if promote else NO_PROMOTION
    pooled_row = {"fold": "POOLED", "chosen_policy": "per-fold", "dti_other_folds": np.nan,
                  "dti_heldout_chosen": cf, "dti_heldout_identity": ident, "delta_dti": cf - ident,
                  "tp_w": per.tp_w.sum(), "fp_w": per.fp_w.sum(), "fn_w": per.fn_w.sum(),
                  "relative_change": rel_gain}
    table = pd.concat([per, pd.DataFrame([pooled_row])], ignore_index=True)
    return {"decision_table": table, "gates": gt, "regional": reg,
            "bootstrap": pd.DataFrame({"delta_dti": deltas}),
            "stats": {"decision": decision, "promote": promote, "final_policy": all_choice if promote else "identity",
                      "all_fold_choice": all_choice, "all_fold_scores": all_scores, "crossfit_pooled_dti": cf,
                      "identity_pooled_dti": ident, "relative_gain": rel_gain, "ci_low": ci[0], "ci_high": ci[1],
                      "unanimous": bool(unanimous), "geographic_relative_change": geo_rel,
                      "fold_delta_min": float(per.delta_dti.min()) if len(per) else np.nan,
                      "fold_delta_max": float(per.delta_dti.max()) if len(per) else np.nan}}
