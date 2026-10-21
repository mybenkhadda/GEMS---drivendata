"""Official GEMS metric: distance-weighted Tversky index (DTI), plus diagnostics.

Definition (DrivenData problem description, competition 306)::

    k(d)  = max(1 - d / R, 0),               R = 300 m (3 px at 100 m)
    TP_w  = sum_{g in G} max_{x: d(x,g) <= R} p(x) k(d(x,g))
    FP_w  = sum_{x: p(x) > 0} p(x) [1 - max_{g in G} k(d(x,g))]
    FN_w  = sum_{g in G} [1 - max_{x: d(x,g) <= R} p(x) k(d(x,g))]
    DTI   = TP_w / (TP_w + alpha FP_w + beta FN_w + eps),   alpha = 0.2, beta = 0.8

``G`` is the set of ground-truth fault pixels, ``p`` in [0, 1] the predicted
probability, ``d`` the Euclidean distance between pixel centres in metres.
Note FN_w = |G| - TP_w and ``max_g k(d(x, g)) = k(dist(x, G))`` (k is decreasing).

Implementation choices that are *not* fixed by the organizers (all explicit arguments):

* ``eps`` — unspecified; default 1e-6.
* NaN / nodata predictions are treated as 0 (no fault); values are clipped to [0, 1].
* ``eval_mask`` — pixels outside it contribute neither ground truth nor false positives
  (their predictions are zeroed before the local max, so they cannot create TP either).
* ``exclude_mask`` — optional known-fault exclusion: excluded pixels are removed from G
  and from the FP sum, but their predictions *can* still support TP of nearby G pixels.
  The organizers do not state that known USGS faults are excluded from the hidden-test
  scoring; this option exists for pseudo-discovery experiments and sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

DTI_ALPHA = 0.2
DTI_BETA = 0.8
DTI_RADIUS_M = 300.0
DTI_EPS = 1e-6


@dataclass(frozen=True)
class DTIComponents:
    tp: float
    fp: float
    fn: float
    n_truth: int
    pred_mass: float
    alpha: float = DTI_ALPHA
    beta: float = DTI_BETA
    eps: float = DTI_EPS

    @property
    def dti(self) -> float:
        return float(self.tp / (self.tp + self.alpha * self.fp + self.beta * self.fn + self.eps))

    def __add__(self, other: "DTIComponents") -> "DTIComponents":
        return DTIComponents(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn,
                             self.n_truth + other.n_truth, self.pred_mass + other.pred_mass,
                             self.alpha, self.beta, self.eps)

    def as_dict(self) -> dict:
        return {"dti": self.dti, "tp_w": self.tp, "fp_w": self.fp, "fn_w": self.fn,
                "n_truth": self.n_truth, "pred_mass": self.pred_mass,
                "recall_w": self.tp / max(self.n_truth, 1),
                "precision_w": self.tp / max(self.tp + self.fp, 1e-12)}


def pool(components: list[DTIComponents]) -> DTIComponents:
    out = components[0]
    for c in components[1:]:
        out = out + c
    return out


# ---------------------------------------------------------------------------
# kernel machinery
# ---------------------------------------------------------------------------
def kernel_offsets(radius_m: float, pixel_m: tuple[float, float]) -> list[tuple[int, int, float]]:
    """(dr, dc, k) for every offset with k(d) > 0, i.e. d < R. Includes (0, 0, 1)."""
    py, px = pixel_m
    rr, rc = int(np.floor(radius_m / py)), int(np.floor(radius_m / px))
    out = []
    for dr in range(-rr, rr + 1):
        for dc in range(-rc, rc + 1):
            d = float(np.hypot(dr * py, dc * px))
            k = 1.0 - d / radius_m
            if k > 0:
                out.append((dr, dc, k))
    return out


def local_weighted_max(p: np.ndarray, radius_m: float, pixel_m: tuple[float, float]) -> np.ndarray:
    """M(y) = max_x p(x) k(d(x, y)) — the TP support each pixel would receive if it were truth."""
    offs = kernel_offsets(radius_m, pixel_m)
    r = max(max(abs(a) for a, _, _ in offs), max(abs(b) for _, b, _ in offs))
    h, w = p.shape
    P = np.pad(p, r, mode="constant", constant_values=0.0)
    M = np.zeros_like(p, dtype=np.float64)
    for dr, dc, k in offs:
        np.maximum(M, k * P[r + dr:r + dr + h, r + dc:r + dc + w], out=M)
    return M


def _bbox(mask: np.ndarray, pad: int) -> tuple[slice, slice]:
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    if not rows.any():
        return slice(0, 0), slice(0, 0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return (slice(max(r0 - pad, 0), min(r1 + pad, mask.shape[0])),
            slice(max(c0 - pad, 0), min(c1 + pad, mask.shape[1])))


def clean_prediction(pred: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, 1.0)


def dti_components(pred: np.ndarray, truth: np.ndarray, *,
                   eval_mask: np.ndarray | None = None,
                   exclude_mask: np.ndarray | None = None,
                   radius_m: float = DTI_RADIUS_M,
                   pixel_m: tuple[float, float] = (100.0, 100.0),
                   alpha: float = DTI_ALPHA, beta: float = DTI_BETA, eps: float = DTI_EPS,
                   crop: bool = True) -> DTIComponents:
    """Exact DTI components on a raster (see module docstring for conventions)."""
    truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {truth.shape}")
    ev = np.ones(truth.shape, bool) if eval_mask is None else np.asarray(eval_mask, bool)
    ex = np.zeros(truth.shape, bool) if exclude_mask is None else np.asarray(exclude_mask, bool)
    if crop:
        pad = int(np.ceil(radius_m / min(pixel_m))) + 1
        sl = _bbox(ev, pad)
        pred, truth, ev, ex = pred[sl], truth[sl], ev[sl], ex[sl]
    p = clean_prediction(pred)
    p[~ev] = 0.0
    G = truth & ev & ~ex
    n = int(G.sum())
    fp_region = ev & ~ex
    if n == 0:
        fp = float(p[fp_region].sum())
        return DTIComponents(0.0, fp, 0.0, 0, float(p[ev].sum()), alpha, beta, eps)
    M = local_weighted_max(p, radius_m, pixel_m)
    tp = float(M[G].sum())
    dist = ndimage.distance_transform_edt(~G, sampling=pixel_m)
    kd = np.clip(1.0 - dist / radius_m, 0.0, None)
    fp = float((p * (1.0 - kd))[fp_region].sum())
    return DTIComponents(tp, fp, float(n - tp), n, float(p[ev].sum()), alpha, beta, eps)


def dti(pred, truth, **kw) -> float:
    return dti_components(pred, truth, **kw).dti


class MaskedScorer:
    """Precomputes crop, G and the FP weight map for one (truth, eval, exclude) triple.

    ``components(pred)`` is identical to :func:`dti_components` but reuses the distance
    transform, which dominates the cost when many prediction maps are scored on one fold.
    """

    def __init__(self, truth, eval_mask=None, exclude_mask=None, radius_m=DTI_RADIUS_M,
                 pixel_m=(100.0, 100.0), alpha=DTI_ALPHA, beta=DTI_BETA, eps=DTI_EPS):
        truth = np.asarray(truth, bool)
        ev = np.ones(truth.shape, bool) if eval_mask is None else np.asarray(eval_mask, bool)
        ex = np.zeros(truth.shape, bool) if exclude_mask is None else np.asarray(exclude_mask, bool)
        self.sl = _bbox(ev, int(np.ceil(radius_m / min(pixel_m))) + 1)
        self.ev, ex = ev[self.sl], ex[self.sl]
        self.G = truth[self.sl] & self.ev & ~ex
        self.fp_region = self.ev & ~ex
        self.n = int(self.G.sum())
        self.radius_m, self.pixel_m = radius_m, pixel_m
        self.alpha, self.beta, self.eps = alpha, beta, eps
        if self.n:
            dist = ndimage.distance_transform_edt(~self.G, sampling=pixel_m)
            self.fp_w = (1.0 - np.clip(1.0 - dist / radius_m, 0.0, None))[self.fp_region]
        else:
            self.fp_w = np.ones(int(self.fp_region.sum()))

    def components(self, pred: np.ndarray) -> DTIComponents:
        p = clean_prediction(pred[self.sl])
        p[~self.ev] = 0.0
        fp = float((p[self.fp_region] * self.fp_w).sum())
        mass = float(p[self.ev].sum())
        if not self.n:
            return DTIComponents(0.0, fp, 0.0, 0, mass, self.alpha, self.beta, self.eps)
        tp = float(local_weighted_max(p, self.radius_m, self.pixel_m)[self.G].sum())
        return DTIComponents(tp, fp, float(self.n - tp), self.n, mass, self.alpha, self.beta, self.eps)


def dti_by_zone(pred: np.ndarray, truth: np.ndarray, zones: np.ndarray, *, valid: np.ndarray,
                exclude_mask: np.ndarray | None = None, pixel_m=(100.0, 100.0), **kw) -> pd.DataFrame:
    """DTI per integer zone (folds, regions, blocks); zone < 0 is skipped. Adds a pooled row."""
    rows, comps = [], []
    for z in np.unique(zones[valid & (zones >= 0)]):
        c = dti_components(pred, truth, eval_mask=valid & (zones == z), exclude_mask=exclude_mask,
                           pixel_m=pixel_m, **kw)
        comps.append(c)
        rows.append({"zone": int(z), **c.as_dict()})
    if comps:
        rows.append({"zone": "pooled_components", **pool(comps).as_dict()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# diagnostics (not the competition score)
# ---------------------------------------------------------------------------
def tolerant_recall_precision(pred: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray,
                              tol_m: float, threshold: float, pixel_m=(100.0, 100.0)) -> dict:
    sl = _bbox(eval_mask, int(np.ceil(tol_m / min(pixel_m))) + 1)
    p, t, ev = clean_prediction(pred[sl]), truth[sl] & eval_mask[sl], eval_mask[sl]
    hit = (p >= threshold) & ev
    if not t.any() or not hit.any():
        return {"tol_m": tol_m, "threshold": threshold, "recall": 0.0 if t.any() else np.nan,
                "precision": np.nan if not hit.any() else 0.0, "pred_frac": float(hit[ev].mean())}
    d_to_hit = ndimage.distance_transform_edt(~hit, sampling=pixel_m)
    d_to_truth = ndimage.distance_transform_edt(~t, sampling=pixel_m)
    return {"tol_m": tol_m, "threshold": threshold,
            "recall": float((d_to_hit[t] <= tol_m).mean()),
            "precision": float((d_to_truth[hit] <= tol_m).mean()),
            "pred_frac": float(hit[ev].mean())}


def sampled_auc_ap(pred: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, rng: np.random.Generator,
                   n: int = 400_000, pos_tol_m: float = 100.0, neg_min_m: float = 1000.0,
                   pixel_m=(100.0, 100.0)) -> dict:
    """ROC-AUC / AP of near-truth (<= pos_tol) vs far (>= neg_min) pixels on a bounded sample."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    sl = _bbox(eval_mask, int(np.ceil(neg_min_m / min(pixel_m))))
    t, ev = truth[sl] & eval_mask[sl], eval_mask[sl]
    if not t.any():
        return {"roc_auc": np.nan, "avg_precision": np.nan}
    d = ndimage.distance_transform_edt(~t, sampling=pixel_m)
    p = clean_prediction(pred[sl])
    pos, neg = np.flatnonzero(ev & (d <= pos_tol_m)), np.flatnonzero(ev & (d >= neg_min_m))
    pos = rng.choice(pos, min(len(pos), n // 4), replace=False)
    neg = rng.choice(neg, min(len(neg), n - len(pos)), replace=False)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[p.ravel()[pos], p.ravel()[neg]]
    return {"roc_auc": float(roc_auc_score(y, s)), "avg_precision": float(average_precision_score(y, s)),
            "auc_pos_tol_m": pos_tol_m, "auc_neg_min_m": neg_min_m}


def calibration_table(pred: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, tol_m: float = 300.0,
                      bins: int = 10, pixel_m=(100.0, 100.0)) -> pd.DataFrame:
    """Reliability: share of pixels within ``tol_m`` of truth per predicted-probability bin."""
    sl = _bbox(eval_mask, int(np.ceil(tol_m / min(pixel_m))) + 1)
    t, ev = truth[sl] & eval_mask[sl], eval_mask[sl]
    d = ndimage.distance_transform_edt(~t, sampling=pixel_m) if t.any() else np.full(t.shape, np.inf)
    p = clean_prediction(pred[sl])[ev]
    near = (d <= tol_m)[ev]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "near": near}).groupby("bin").agg(
        p_mean=("p", "mean"), near_rate=("near", "mean"), n=("p", "size")).reset_index()
    df["bin_lo"], df["bin_hi"] = edges[df.bin], edges[df.bin + 1]
    return df


def full_diagnostics(pred: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, rng: np.random.Generator,
                     tolerances=(100.0, 200.0, 300.0), threshold: float = 0.5, pixel_m=(100.0, 100.0),
                     exclude_mask: np.ndarray | None = None) -> dict:
    c = dti_components(pred, truth, eval_mask=eval_mask, exclude_mask=exclude_mask, pixel_m=pixel_m)
    out = c.as_dict()
    for tol in tolerances:
        rp = tolerant_recall_precision(pred, truth, eval_mask, tol, threshold, pixel_m)
        out[f"recall@{tol:g}m"], out[f"precision@{tol:g}m"] = rp["recall"], rp["precision"]
        out["pred_frac"] = rp["pred_frac"]
    out.update(sampled_auc_ap(pred, truth, eval_mask, rng, pixel_m=pixel_m))
    p = clean_prediction(pred)[eval_mask]
    out["mean_p"] = float(p.mean()) if p.size else np.nan
    return out


# ---------------------------------------------------------------------------
# output transforms: DTI depends on probability *scale*, so tune on OOF only
# ---------------------------------------------------------------------------
def apply_output_transform(p: np.ndarray, gamma: float = 1.0, scale: float = 1.0, floor: float = 0.0) -> np.ndarray:
    """q = clip(scale * p**gamma, 0, 1), with q = 0 where p < floor. NaN stays NaN."""
    q = np.clip(scale * np.power(np.clip(p, 0, 1), gamma), 0.0, 1.0)
    q = np.where(p < floor, 0.0, q)
    return np.where(np.isfinite(p), q, np.nan).astype(np.float32)


DEFAULT_TRANSFORM_GRID = {"gamma": (0.5, 1.0, 2.0, 3.0), "scale": (0.5, 1.0, 1.5, 2.0, 3.0),
                          "floor": (0.0, 0.05, 0.1, 0.2, 0.4)}


def transform_component_table(oof: np.ndarray, truth: np.ndarray, folds: np.ndarray, valid: np.ndarray,
                              grid: dict | None = None, pixel_m=(100.0, 100.0),
                              exclude_mask: np.ndarray | None = None) -> pd.DataFrame:
    """DTI components for every transform x fold (fold < 0 skipped)."""
    grid = grid or DEFAULT_TRANSFORM_GRID
    rows = []
    fold_ids = [int(f) for f in np.unique(folds[valid & (folds >= 0)])]
    scorers = {f: MaskedScorer(truth, valid & (folds == f), exclude_mask, pixel_m=pixel_m) for f in fold_ids}
    for g in grid["gamma"]:
        for s in grid["scale"]:
            for fl in grid["floor"]:
                q = apply_output_transform(oof, g, s, fl)
                for f in fold_ids:
                    c = scorers[f].components(q)
                    rows.append({"gamma": g, "scale": s, "floor": fl, "fold": f,
                                 "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn})
    return pd.DataFrame(rows)


def _dti_from(df: pd.DataFrame, alpha=DTI_ALPHA, beta=DTI_BETA, eps=DTI_EPS) -> float:
    tp, fp, fn = df.tp_w.sum(), df.fp_w.sum(), df.fn_w.sum()
    return float(tp / (tp + alpha * fp + beta * fn + eps))


def cross_fitted_transform(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For each fold pick the transform maximising pooled DTI on the *other* folds.

    Returns (per-fold choice with its held-out DTI, transform chosen on all folds for submission).
    """
    keys = ["gamma", "scale", "floor"]
    rows = []
    for f in sorted(table.fold.unique()):
        others = table[table.fold != f].groupby(keys).apply(_dti_from).rename("dti_other").reset_index()
        best = others.sort_values("dti_other", ascending=False).iloc[0]
        held = table[(table.fold == f) & (table.gamma == best.gamma) & (table.scale == best.scale)
                     & (table.floor == best.floor)]
        rows.append({"fold": f, **{k: best[k] for k in keys}, "dti_other": best.dti_other,
                     "tp_w": held.tp_w.sum(), "fp_w": held.fp_w.sum(), "fn_w": held.fn_w.sum(),
                     "dti_heldout": _dti_from(held)})
    per_fold = pd.DataFrame(rows)
    allf = table.groupby(keys).apply(_dti_from).rename("dti_all").reset_index()
    final = allf.sort_values("dti_all", ascending=False).iloc[0].to_dict()
    final["dti_crossfit_pooled"] = _dti_from(per_fold)
    return per_fold, final
