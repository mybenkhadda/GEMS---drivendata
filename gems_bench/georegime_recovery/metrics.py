"""DTI scoring - thin, checked wrappers around the repository metric (``gems_bench.metric``); no new metric.

* :func:`score` = ``dti_components`` with ``exclude_supports_tp = OFFICIAL_EXCLUDE_SUPPORTS_TP`` (False: an
  excluded known-fault pixel neither counts as truth / FP nor supports TP of nearby truth);
* :class:`ComponentMaps` produces the pixel maps (TP support M(g) on truth, FP mass p(1-k), FN 1-M(g)) with
  the SAME kernel (``local_weighted_max``), cleaning (``clean_prediction``) and zeroing as ``MaskedScorer``,
  and verifies that their sums equal ``MaskedScorer.components`` - otherwise it raises;
* pooling sums TP / FP / FN (``gems_bench.metric.pool``); a mean of fold DTIs is never reported as pooled.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from gems_bench import metric as M

from .contracts import MetricConsistencyError

LABEL = "offline DTI proxy (NOT the official leaderboard DTI)"


def check_constants(cfg: dict) -> dict:
    m = cfg["metric"]
    if (m["radius_m"], m["alpha"], m["beta"]) != (M.DTI_RADIUS_M, M.DTI_ALPHA, M.DTI_BETA):
        raise MetricConsistencyError(f"config metric {m} differs from gems_bench.metric constants")
    return scorer_settings()


def scorer_settings() -> dict:
    return {"implementation": "gems_bench.metric.dti_components / MaskedScorer / local_weighted_max",
            "radius_m": M.DTI_RADIUS_M, "alpha": M.DTI_ALPHA, "beta": M.DTI_BETA, "eps": M.DTI_EPS,
            "kernel": "k(d) = max(1 - d/300 m, 0); M(g) = max_x p(x) k(d(x,g))",
            "exclude_supports_tp": M.OFFICIAL_EXCLUDE_SUPPORTS_TP,
            "exclude_supports_tp_note": "organizer-statement reading; NOT verified against a public reference scorer",
            "nan_policy": "NaN / non-finite predictions -> 0, values clipped to [0, 1]",
            "eval_mask_policy": "predictions outside the evaluation mask are zeroed before the local max (no TP from outside)",
            "pooling": "sum of TP_w, FP_w, FN_w over units (gems_bench.metric.pool)",
            "label": LABEL}


def score(pred, truth, eval_mask, exclude) -> M.DTIComponents:
    return M.dti_components(pred, truth, eval_mask=eval_mask, exclude_mask=exclude,
                            exclude_supports_tp=M.OFFICIAL_EXCLUDE_SUPPORTS_TP)


def pool(comps):
    comps = [c for c in comps if c is not None]
    return M.pool(comps) if comps else None


def row(comp: M.DTIComponents, **keys) -> dict:
    return {**keys, **comp.as_dict(), "metric_label": LABEL}


class ComponentMaps:
    """Fixed (truth, eval, exclude) triple; ``maps(pred)`` returns full-size pixel component maps + totals."""

    def __init__(self, truth, eval_mask, exclude, pixel_m=(100.0, 100.0)):
        self.shape = truth.shape
        self.scorer = M.MaskedScorer(truth, eval_mask, exclude, pixel_m=pixel_m,
                                     exclude_supports_tp=M.OFFICIAL_EXCLUDE_SUPPORTS_TP)
        s = self.scorer
        self.sl = s.sl
        if s.n:
            d = ndimage.distance_transform_edt(~s.G, sampling=pixel_m)
            self.k = np.clip(1.0 - d / M.DTI_RADIUS_M, 0.0, None)            # k(dist to truth), crop frame
            self.dist_m = d
        else:
            self.k = np.zeros(s.G.shape)
            self.dist_m = np.full(s.G.shape, np.inf)

    def components(self, pred) -> M.DTIComponents:
        return self.scorer.components(pred)

    def maps(self, pred, check: bool = True) -> dict:
        s = self.scorer
        p = M.clean_prediction(pred[self.sl])
        p[s.zero] = 0.0
        fp = np.where(s.fp_region, p * (1.0 - self.k), 0.0)
        if s.n:
            Mx = M.local_weighted_max(p, s.radius_m, s.pixel_m)
            tp = np.where(s.G, Mx, 0.0)
        else:
            tp = np.zeros_like(p)
        comp = s.components(pred)
        if check:
            for name, a, b in (("tp", tp.sum(), comp.tp), ("fp", fp.sum(), comp.fp)):
                if abs(a - b) > 1e-6 * max(1.0, abs(b)):
                    raise MetricConsistencyError(f"{name} map sum {a} != MaskedScorer {b}")
        full = {}
        for k, v in (("tp_support", tp), ("fp_mass", fp)):
            out = np.zeros(self.shape, np.float32)
            out[self.sl] = v
            full[k] = out
        G = np.zeros(self.shape, bool)
        G[self.sl] = s.G
        full["truth"] = G
        full["fn_support"] = np.where(G, 1.0 - full["tp_support"], 0.0).astype(np.float32)
        full["components"] = comp
        return full

    def dist_to_truth_full(self) -> np.ndarray:
        out = np.full(self.shape, np.inf, np.float32)
        out[self.sl] = self.dist_m
        return out


def block_ids(shape, block_px: int):
    h, w = shape
    nbc = (w + block_px - 1) // block_px
    r, c = np.indices(shape)
    return ((r // block_px) * nbc + (c // block_px)).astype(np.int32), nbc


def block_sums(maps: dict, bid: np.ndarray, nblocks: int) -> dict:
    tp = np.bincount(bid.ravel(), weights=maps["tp_support"].ravel(), minlength=nblocks)
    fp = np.bincount(bid.ravel(), weights=maps["fp_mass"].ravel(), minlength=nblocks)
    n = np.bincount(bid.ravel(), weights=maps["truth"].ravel().astype(np.float64), minlength=nblocks)
    return {"tp_w": tp, "fp_w": fp, "n_truth": n, "fn_w": n - tp}


def dti_from(tp, fp, fn) -> float:
    return float(tp / (tp + M.DTI_ALPHA * fp + M.DTI_BETA * fn + M.DTI_EPS))
