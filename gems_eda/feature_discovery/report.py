"""Section 13: scientific synthesis written ONLY from saved tables. A question whose table is missing says NOT RUN;
no number here is typed by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import FD_CODE_REV


def _md(df: pd.DataFrame, n=12) -> str:
    if df is None or not len(df):
        return "_NOT RUN (table absent)_"
    d = df.head(n)
    try:
        return d.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        return "```\n" + d.to_string(index=False) + "\n```"


def synthesis_markdown(E) -> str:
    t = E.load_table
    raw, G, C = t("raw_fault_association"), t("global_fault_association"), t("fault_response_classes")
    S, U, H, Hs, R = t("feature_scale_comparison"), t("unique_fault_recovery"), t("heldout_validation"), \
        t("heldout_selected_on_dev"), t("feature_recommendations")
    W = t("w2_missed_fault_signatures")
    out = [f"# Geological feature discovery — synthesis ({FD_CODE_REV}, mode {E.c['mode']})", "",
           "All values below are read from the saved tables of this run. Exploratory = direction / feature chosen on "
           "the same pixels (optimistic); held-out = polarity frozen on GEMS training-side regions, scored on the "
           "held-out fold with 2 km spatial-block CIs. Unlabelled pixels are not confirmed non-faults; no number "
           "here is an official DTI.", ""]
    out += ["## 1. Raw variables associated with mapped faults", "",
            _md(raw[["feature", "auroc_high", "auroc_low", "ap_lift_high", "ap_lift_low", "n_fault_px"]]
                if len(raw) else raw), ""]
    if len(G):
        base = set(raw.feature) if len(raw) else set()
        dv = G[~G.feature.isin(base)]
        best_raw = raw.best_auroc_EXPLORATORY.max() if len(raw) and "best_auroc_EXPLORATORY" in raw else np.nan
        out += ["## 2. Are changes (gradients / contrasts) more informative than absolute values?", "",
                f"Best exploratory AUROC — raw bands: {best_raw:.3f}; derived features: "
                f"{dv.best_auroc_EXPLORATORY.max():.3f} ({dv.iloc[0].feature if len(dv) else 'n/a'}). Held-out "
                "evidence: section 5.", ""]
        out += ["## 3. Transformations revealing fault-like structure (exploratory ranking)", "",
                _md(dv[["feature", "family", "best_auroc_EXPLORATORY", "better_direction_EXPLORATORY",
                        "ap_lift_high", "ap_lift_low"]]), ""]
    else:
        out += ["## 2-3. NOT RUN (global association absent)", ""]
    out += ["## 4. Distance of the strongest response", "", _md(C), ""]
    out += ["## 5. Unique recovery beyond the references (1 % budget, components within 300 m)", "",
            _md(U.sort_values("feature_only", ascending=False)[["feature", "reference", "feature_only",
                                                                 "reference_only", "unique_share_of_components"]]
                if len(U) else U), ""]
    out += ["## 6. Generalisation across regions / domains", "",
            _md(S[S.partition.isin(["region_3x3", "tile_20km", "label_free_domain"])] if len(S) else S, 20), "",
            "Held-out folds (polarity frozen on development side):", "",
            _md(H[["fold", "feature", "status", "direction", "dev_auroc", "eval_auroc", "eval_auroc_ci_lo",
                   "eval_auroc_ci_hi", "eval_recall_1pct"]] if len(H) else H, 30), "",
            "Feature selected on development data per fold (honest estimate of picking the best feature):", "",
            _md(Hs), ""]
    if len(W):
        out += ["## W2-missed mapped faults", "", _md(W), ""]
    out += ["## 7. Features recommended for P6 / W2 / M9 DTI benchmarking", "",
            _md(R[["feature", "class", "reason", "score"]] if len(R) else R, 40), "",
            "PROMOTE requires: held-out CI above 0.5 in >= 2/3 folds, consistent polarity, regional same-polarity "
            "share >= threshold and unique component recovery beyond the best reference. Promoted features must "
            "still be benchmarked with the official scorer at equal area budget in P6 before any claim.", ""]
    return "\n".join(out)
