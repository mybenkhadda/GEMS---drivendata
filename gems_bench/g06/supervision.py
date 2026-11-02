"""Trustworthy training targets for incomplete fault labels.

The mapped faults are INCOMPLETE: an unmapped pixel is not a known negative. Pixel classes (per training
visibility, computed from VISIBLE labels only)::

    POSITIVE            visible mapped-fault pixel inside the training region
    IGNORE              outside the training region / footprint, or within ``ignore_km`` of a visible fault
                        (position uncertainty of the mapped trace)
    UNLABELED           ``ignore_km`` < distance to visible faults < ``likely_background_km``  (PU: weight u < 1)
    LIKELY_BACKGROUND   distance >= ``likely_background_km`` (still unlabelled, but far from every visible fault)

Hidden (withheld) faults are NOT given a class of their own: a hidden fault pixel inside the region is simply an
unlabelled pixel at its natural rate - marking or avoiding it would use the held-out truth.

Segment and connection targets are EVALUATION-SIDE quantities of a unit (proximity of a candidate to that unit's
truth); they are used to train only on OTHER units (cross-fitting) and never as an inference feature.
Targets live in label tables, physically separate from feature tables; ``validate_label_visibility`` refuses
truth-derived feature columns and, for pixel models trained on visible faults, known-fault context columns (the
distance to a visible fault would be the target itself).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .contracts import LeakageError, assert_label_free_columns, validate_target_separation
from .feature_context import KNOWN_COLUMNS

POSITIVE, UNLABELED, LIKELY_BACKGROUND, IGNORE = 1, 0, 2, -1
CLASS_NAMES = {POSITIVE: "POSITIVE", UNLABELED: "UNLABELED", LIKELY_BACKGROUND: "LIKELY_BACKGROUND",
               IGNORE: "IGNORE"}
SEG_POS, SEG_NEG, SEG_AMB = "POSITIVE", "NEGATIVE", "AMBIGUOUS"


def visible_distance_km(visible: np.ndarray, px_km: float = 0.1) -> np.ndarray:
    v = np.asarray(visible, bool)
    if not v.any():
        return np.full(v.shape, np.inf, np.float32)
    return (ndimage.distance_transform_edt(~v) * px_km).astype(np.float32)


def pixel_classes(visible: np.ndarray, region: np.ndarray, valid: np.ndarray, dist_km: np.ndarray, cfg: dict
                  ) -> np.ndarray:
    """int8 class map (codes above) from visible labels, training region, footprint and distance to visible faults."""
    ign, far = float(cfg["ignore_km"]), float(cfg["likely_background_km"])
    reg = np.asarray(region, bool) & np.asarray(valid, bool)
    vis = np.asarray(visible, bool)
    c = np.full(vis.shape, IGNORE, np.int8)
    c[reg & (dist_km >= far)] = LIKELY_BACKGROUND
    c[reg & (dist_km > ign) & (dist_km < far)] = UNLABELED
    c[reg & vis] = POSITIVE
    return c


def build_pixel_supervision(visible: np.ndarray, region: np.ndarray, valid: np.ndarray, cfg: dict) -> dict:
    """{classes, dist_km, target_dist_km} full-grid maps of one training visibility."""
    d = visible_distance_km(visible)
    cls = pixel_classes(visible, region, valid, d, cfg)
    cap = float(cfg.get("target_dist_cap_km", 5.0))
    return {"classes": cls, "dist_km": d, "target_dist_km": np.minimum(d, cap).astype(np.float32)}


def class_confidence(cls: np.ndarray, cfg: dict) -> np.ndarray:
    conf = {POSITIVE: 1.0, LIKELY_BACKGROUND: float(cfg.get("likely_background_confidence", 0.9)),
            UNLABELED: float(cfg.get("unlabeled_confidence", 0.5)), IGNORE: 0.0}
    out = np.zeros(np.shape(cls), np.float32)
    for k, v in conf.items():
        out[np.asarray(cls) == k] = v
    return out


def build_orientation_targets(visible: np.ndarray, sigma_px: float = 3.0, min_coherence: float = 0.5) -> dict:
    """Strike of the visible traces (doubled angle) on the trace pixels where the local orientation is stable."""
    k = np.asarray(visible, np.float64)
    if not k.any():
        z = np.full(k.shape, np.nan, np.float32)
        return {"orient_cos2": z, "orient_sin2": z.copy(), "orient_coherence": z.copy()}
    gr = ndimage.gaussian_filter(k, sigma_px, order=(1, 0))
    gc = ndimage.gaussian_filter(k, sigma_px, order=(0, 1))
    ge, gn = gc, -gr
    jee = ndimage.gaussian_filter(ge * ge, sigma_px)
    jen = ndimage.gaussian_filter(ge * gn, sigma_px)
    jnn = ndimage.gaussian_filter(gn * gn, sigma_px)
    tr = jee + jnn
    disc = np.sqrt(np.maximum((jee - jnn) ** 2 + 4 * jen * jen, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(tr > 0, (disc / tr) ** 2, 0.0)
    strike = 0.5 * np.arctan2(2 * jen, jee - jnn) + np.pi / 2
    ok = np.asarray(visible, bool) & (coh >= min_coherence)
    return {"orient_cos2": np.where(ok, np.cos(2 * strike), np.nan).astype(np.float32),
            "orient_sin2": np.where(ok, np.sin(2 * strike), np.nan).astype(np.float32),
            "orient_coherence": np.where(np.asarray(visible, bool), coh, np.nan).astype(np.float32)}


def build_segment_supervision(lab: np.ndarray, eval_truth: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Per candidate label: near-truth share, class (POSITIVE / NEGATIVE / AMBIGUOUS), binary target (NaN for
    ambiguous), label confidence. EVALUATION SIDE: cross-unit training targets only."""
    from ..geomoe.candidates import near_share

    n = int(np.asarray(lab).max()) if np.asarray(lab).size else 0
    if not n:
        return pd.DataFrame(columns=["label", "near_share", "cls", "target", "label_conf"])
    sh = near_share(np.asarray(lab, np.int32), np.asarray(eval_truth, bool), float(cfg["tol_px"]))
    pos, neg = float(cfg["positive_share"]), float(cfg["negative_share"])
    cls = np.where(sh >= pos, SEG_POS, np.where(sh < neg, SEG_NEG, SEG_AMB))
    tgt = np.where(cls == SEG_POS, 1.0, np.where(cls == SEG_NEG, 0.0, np.nan))
    conf = np.clip(np.abs(sh - 0.5) * 2.0, 0.0, 1.0)
    return pd.DataFrame({"label": np.arange(1, n + 1), "near_share": sh, "cls": cls, "target": tgt,
                         "label_conf": conf})


def dominant_system(lab: np.ndarray, sysmap: np.ndarray, eval_truth: np.ndarray, tol_px: float) -> np.ndarray:
    """System id of the truth nearest to each segment (mode over its pixels within ``tol_px``); -1 if none."""
    lab = np.asarray(lab, np.int32)
    n = int(lab.max()) if lab.size else 0
    out = np.full(n, -1, np.int64)
    t = np.asarray(eval_truth, bool) & (np.asarray(sysmap) >= 0)
    if not n or not t.any():
        return out
    d, (ir, ic) = ndimage.distance_transform_edt(~t, return_indices=True)
    rr, cc = np.nonzero(lab)
    near = d[rr, cc] <= tol_px
    sid = np.asarray(sysmap)[ir[rr, cc], ic[rr, cc]]
    df = pd.DataFrame({"label": lab[rr, cc][near], "sid": sid[near]})
    if len(df):
        m = df.groupby("label").sid.agg(lambda s: s.value_counts().index[0])
        out[m.index.to_numpy() - 1] = m.to_numpy()
    return out


def build_connection_supervision(pairs: pd.DataFrame, seg_sup: pd.DataFrame, dom: np.ndarray,
                                 gap_near_share: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Connection target per candidate pair: 1 = both fragments lie on the SAME truth system and the gap is near that
    truth; 0 = at least one fragment is a clear negative; NaN = ambiguous. Evaluation side only."""
    if not len(pairs):
        return pairs.assign(target=np.nan, cls=SEG_AMB, label_conf=0.0)
    s = seg_sup.set_index("label")
    a, b = pairs.label_a.to_numpy(), pairs.label_b.to_numpy()
    sa, sb = s.loc[a, "cls"].to_numpy(), s.loc[b, "cls"].to_numpy()
    same = (dom[a - 1] >= 0) & (dom[a - 1] == dom[b - 1])
    pos = same & (sa == SEG_POS) & (sb == SEG_POS) & (gap_near_share >= float(cfg.get("gap_positive_share", 0.5)))
    neg = (sa == SEG_NEG) | (sb == SEG_NEG) | ((sa == SEG_POS) & (sb == SEG_POS) & ~same)
    tgt = np.where(pos, 1.0, np.where(neg, 0.0, np.nan))
    cls = np.where(pos, SEG_POS, np.where(neg, SEG_NEG, SEG_AMB))
    return pairs.assign(target=tgt, cls=cls, label_conf=np.where(np.isnan(tgt), 0.0, 1.0), same_system=same,
                        gap_near_share=gap_near_share)


def validate_label_visibility(kind: str, feature_columns, target_columns=()) -> None:
    """Feature / target separation for a dataset ``kind`` (pixel | patch | segment | connection | meta).

    Pixel / patch models are trained on the VISIBLE faults themselves: any known-fault context column is refused
    (it would encode the target). Segment / connection / meta models are trained on other units' truth with the
    unit's permitted known context: those columns are allowed there."""
    validate_target_separation(list(feature_columns), list(target_columns))
    assert_label_free_columns(feature_columns, f"{kind} features")
    if kind in ("pixel", "patch"):
        bad = [c for c in feature_columns if c in KNOWN_COLUMNS or str(c).startswith(("known_", "tip_"))]
        if bad:
            raise LeakageError(f"{kind} model trained on visible faults cannot use known-fault context {bad}")
