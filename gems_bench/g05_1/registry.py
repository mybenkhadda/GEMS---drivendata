"""Stage D: fault-hypothesis registry (one row per candidate segment of every scenario unit).

Families (``FAMILY``) - every column is label-free; the only truth-derived column is ``y_near_truth`` (evaluation /
training target, never a feature, guarded by ``assert_label_free``):
  D1 ranking evidence     W2 / CNN / fusion segment means, W2 unit rank, W2 local contrast, W2 budget persistence
  D2 geophysics           magnetic / gravity gradients and agreement, multiscale persistence, raw-band across-trace
                          contrasts (exact band identifiers), tilt angle of the magnetic field
  D3 geometry             trace length, tortuosity, linearity, strike, dispersion, curvature, width, branches,
                          survey-axis artifact indicator, short flag
  D4 known-network        distance / density of permitted known faults, tip distance and alignment, network evidence
  D5 regional             regional orientation agreement, spatial domain (when the EDA domain raster exists)
  D6 structural           G02 continuity / displacement, G01 persistence, Stage-E displacement-style evidence
  D7 uncertainty          CNN disagreement, number of proposers, expert-rank dispersion, availability flags
Graph features (Stage F, prefix ``g_``) are their own family ``F_graph``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import displacement as DP
from . import geometry as GM

KEYS = ("candidate_id", "scenario", "zone", "unit", "label")
TARGET_COLUMNS = ("y_near_truth",)
FAMILY_LAYERS = {
    "D1_ranking": ("mean_w2", "mean_cnn_b0", "mean_cnn_b1", "mean_fusion", "w2_unit_rank", "w2_local_contrast",
                   "w2_persistence"),
    "D2_magnetic": ("mean_mag_hgrad", "mean_tilt_deg", "abs_tilt_deg", "xc_mag_anom", "xc_rtp", "xc_tmi_hg",
                    "xc_tmi_vg"),
    "D2_gravity": ("mean_grav_hgrad", "xc_iso_grav_anom", "xc_iso_grav_anom_hg", "xc_iso_grav_anom_vg"),
    "D2_multiscale": ("mean_pf_persistence", "mean_mg_cross", "mean_mg_orient_agree"),
    "D3_geometry": ("chain_length_km", "endpoint_dist_km", "tortuosity", "linearity", "strike_dispersion_deg",
                    "curvature_deg_per_km", "line_r2", "width_px", "n_branch_px", "n_endpoints", "survey_artifact",
                    "short"),
    "D4_known": ("mean_known_dist_km", "mean_known_density", "min_tip_dist_km", "mean_tip_along_cos",
                 "mean_network"),
    "D5_regional": ("g_regional_agreement", "domain"),
    "D6_structural": ("mean_g02c", "mean_g02a", "mean_g01_persistence", "disp_rtp_excess", "disp_rtp_consistent",
                      "disp_iso_grav_anom_excess", "disp_iso_grav_anom_consistent"),
    "D7_uncertainty": ("mean_cnn_disagree", "n_proposers", "expert_rank_std", "n_experts_available"),
    "F_graph": ("g_deg_cont", "g_deg_branch", "g_component_size", "g_component_length_km", "g_support",
                "g_best_neighbor_conf", "g_compat_mean", "g_known_tip_dist_km", "g_isolated"),
}
FAMILY = {c: f for f, cs in FAMILY_LAYERS.items() for c in cs}
ABLATION_MAP = {"D1_ranking_cnn": ("mean_cnn_b0", "mean_cnn_b1", "mean_fusion"),
                "D2_magnetic": FAMILY_LAYERS["D2_magnetic"], "D2_gravity": FAMILY_LAYERS["D2_gravity"],
                "D2_multiscale": FAMILY_LAYERS["D2_multiscale"], "F_graph": FAMILY_LAYERS["F_graph"],
                "D4_known": FAMILY_LAYERS["D4_known"], "D5_regional": FAMILY_LAYERS["D5_regional"],
                "D7_uncertainty": FAMILY_LAYERS["D7_uncertainty"]}
# W2 itself is the base of every residual ranker: these are never residual features (they would let the model
# re-learn the base); W2 persistence / contrast are kept as descriptors of W2's own map shape.
BASE_COLUMNS = ("mean_w2", "w2_unit_rank")
DEFINITIONS = {
    "w2_unit_rank": "average-rank ECDF of the segment W2 mean inside its unit",
    "w2_local_contrast": "segment mean of (W2 - W2 box mean over 2 x ring_km) on its pixels",
    "w2_persistence": "number of W2 pixel-map top-k budgets (registry.persistence_budgets) that select >= half of "
                      "the segment",
    "mean_tilt_deg": "mean tilt angle atan2(tmi_vg, |tmi_hg|) on the trace, degrees",
    "abs_tilt_deg": "mean |tilt angle| (small = near a magnetic edge)",
    "survey_artifact": "strike within registry.survey_axis_tol_deg of a survey / grid axis",
    "short": "chain length < registry.short_km",
    "expert_rank_std": "std of the Stage-C expert unit ranks of the segment",
    "n_experts_available": "number of Stage-C experts with a finite rank on the segment",
    "domain": "spatial domain id of the segment centroid (EDA domains.npz), categorical; NaN when absent",
}


class LeakageError(RuntimeError):
    pass


def assert_label_free(cols) -> None:
    bad = [c for c in cols if str(c).startswith("y_") or "truth" in str(c) or str(c) in ("system_id", "recovered")]
    if bad:
        raise LeakageError(f"truth-derived columns among features: {bad}")


def feature_columns(R: pd.DataFrame, drop_families=(), include_graph: bool = True) -> list[str]:
    """Residual-ranker feature columns present in ``R`` (W2 base columns excluded)."""
    drop = {c for f in drop_families for c in ABLATION_MAP.get(f, FAMILY_LAYERS.get(f, ()))}
    cols = [c for c in FAMILY if c in R and c not in BASE_COLUMNS and c not in drop and
            (include_graph or FAMILY[c] != "F_graph")]
    assert_label_free(cols)
    return cols


def dictionary() -> pd.DataFrame:
    rows = []
    for f, cs in FAMILY_LAYERS.items():
        for c in cs:
            d = DEFINITIONS.get(c)
            if d is None:
                if c.startswith("mean_"):
                    d = f"segment mean of the GeoMoE unit layer '{c[5:]}' (finite pixels)"
                elif c.startswith("xc_"):
                    d = f"|mean(+3 px side) - mean(-3 px side)| of raw band '{c[3:]}' / unit std (exact identifier)"
                elif c.startswith("disp_"):
                    d = "Stage-E displacement-style evidence (shift correspondence minus same-side control)"
                elif c.startswith("g_"):
                    d = "Stage-F candidate-graph feature (label-free edges)"
                else:
                    d = "trace geometry (gems_bench.g05_1.geometry)"
            rows.append({"feature": c, "family": f, "definition": d, "label_free": True,
                         "residual_feature": c not in BASE_COLUMNS})
    rows.append({"feature": "y_near_truth", "family": "TARGET", "definition": "share of the segment within 300 m of "
                 "the unit's held-out truth (target only; never a feature)", "label_free": False,
                 "residual_feature": False})
    return pd.DataFrame(rows)


def provenance() -> pd.DataFrame:
    src = {"D1_ranking": "GeoMoE unit layers (frozen W2 stack map, B0 / B1 honest CNN maps) via stage-04 segments; "
                         "W2 pixel map from Boost.stack_rank",
           "D2_magnetic": "GeoMoE layers + raw bands of the features GeoTIFF read by exact identifier",
           "D2_gravity": "GeoMoE layers + raw bands (exact identifier: iso_grav_anom != iso_grav_anom_hg)",
           "D2_multiscale": "GeoMoE potential-field persistence / cross-modal agreement",
           "D3_geometry": "gems_bench.g05_1.geometry on the stage-04 label raster",
           "D4_known": "permitted known faults of the scenario (p6.stack.permitted_known convention)",
           "D5_regional": "Stage-F regional agreement; EDA domains raster (optional)",
           "D6_structural": "frozen G01 / G02 unit rasters; Stage-E displacement on raw bands",
           "D7_uncertainty": "CNN disagreement layer, inventory proposers, Stage-C expert ranks",
           "F_graph": "gems_bench.g05_1.graph (label-free edges)"}
    return pd.DataFrame([{"feature": c, "family": f, "source": src[f], "computed_in": "Stage D/E/F (Colab)",
                          "label_free": True, "leakage_note": "SCREENING caveat: W2 / CNN values come from models "
                          "trained with other zones' truth (see VALIDATION_CONTRACT.md)" if f == "D1_ranking" else
                          ("permitted known context only" if f == "D4_known" else "")}
                         for f, cs in FAMILY_LAYERS.items() for c in cs])


# =============================================================================== per-unit computation
def local_contrast(a: np.ndarray, ev: np.ndarray, ring_px: int) -> np.ndarray:
    """a - box mean of a over (2 ring_px + 1)^2 inside ``ev`` (NaN outside)."""
    x = np.where(ev & np.isfinite(a), a, 0.0).astype(np.float64)
    w = (ev & np.isfinite(a)).astype(np.float64)
    k = 2 * int(ring_px) + 1
    m = ndimage.uniform_filter(x, k, mode="constant") / np.maximum(ndimage.uniform_filter(w, k, mode="constant"),
                                                                   1e-9)
    return np.where(ev & np.isfinite(a), a - m, np.nan).astype(np.float32)


def persistence_count(lab: np.ndarray, w2: np.ndarray, el: np.ndarray, budgets) -> np.ndarray:
    """Per label (1..n): number of budgets whose W2 top-k selection covers >= half of the segment."""
    from ..gems3.representation import top_mask

    n = int(lab.max())
    cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    out = np.zeros(n + 1)
    for f in budgets:
        q = top_mask(np.where(el, w2, np.nan), el, float(f))
        hit = np.bincount(lab[q], minlength=n + 1).astype(float)
        out += (hit / np.maximum(cnt, 1)) >= 0.5
    return out[1:]


def segment_pass(lab: np.ndarray, labels, fields: dict, xc_fields: dict, dc: dict, disp_labels: set,
                 rcfg: dict) -> pd.DataFrame:
    """One loop over segments: geometry, across-trace contrasts of raw bands, displacement evidence (only for
    ``disp_labels``: a label-free compute budget, others NOT_COMPUTED)."""
    P = GM.segment_pixels(lab)
    std = {k: float(np.nanstd(v[0][v[1] < 0.5])) if (v[1] < 0.5).any() else np.nan for k, v in xc_fields.items()}
    rows = []
    for k in labels:
        rr, cc = P.get(int(k), (np.zeros(0, int), np.zeros(0, int)))
        g = GM.features(rr, cc)
        row = {"label": int(k), **g}
        if len(rr) >= 3:
            order = GM.path_order(rr, cc)
            r, c, nr, nc = DP.trace_frame(rr, cc, order)
            for b, prep in xc_fields.items():
                A, B = DP.sample(prep, r + 3 * nr, c + 3 * nc), DP.sample(prep, r - 3 * nr, c - 3 * nc)
                d = np.nanmean(A) - np.nanmean(B) if np.isfinite(A).any() and np.isfinite(B).any() else np.nan
                row[f"xc_{b}"] = abs(d) / std[b] if np.isfinite(d) and std[b] and np.isfinite(std[b]) else np.nan
            for f, fs in fields.items():
                if int(k) in disp_labels:
                    e = DP.segment_evidence(fs, rr, cc, dc, order)
                    row[f"disp_{f}_status"] = e["status"]
                    row[f"disp_{f}_excess"] = e.get("excess", np.nan)
                    row[f"disp_{f}_gain"] = e.get("gain", np.nan)
                    row[f"disp_{f}_shift_px"] = e.get("shift_px", np.nan)
                    row[f"disp_{f}_consistent"] = float(e["consistent"]) if "consistent" in e else np.nan
                else:
                    row[f"disp_{f}_status"] = "NOT_COMPUTED_BUDGET"
        rows.append(row)
    D = pd.DataFrame(rows)
    D["survey_artifact"] = GM.survey_artifact(D.strike_deg, rcfg["survey_axes_deg"], float(rcfg["survey_axis_tol_deg"]))
    D["short"] = D.chain_length_km < float(rcfg["short_km"])
    return D


def domain_of(dom: np.ndarray | None, rows, cols) -> np.ndarray:
    if dom is None:
        return np.full(len(rows), np.nan)
    r = np.clip(np.round(np.asarray(rows, float)).astype(int), 0, dom.shape[0] - 1)
    c = np.clip(np.round(np.asarray(cols, float)).astype(int), 0, dom.shape[1] - 1)
    return dom[r, c].astype(float)
