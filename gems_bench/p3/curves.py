"""Exact DTI of *binary* threshold masks for every threshold at once.

Why binary masks matter
-----------------------
For a fixed truth set G the DTI can be written f(p) = TP(p) / (0.2 TP(p) + 0.2 FP(p) + 0.8 n)
(using FN = n - TP). TP is a sum of maxima of linear functions of p (convex), FP is linear.
For any level t the set {f >= t} is {TP (1 - 0.2 t) - 0.2 t FP - 0.8 t n >= 0}, whose left side
is convex in p, so it is maximised at a vertex of [0, 1]^N: **the DTI-optimal prediction is a
binary map**. Probability calibration (in the ML sense) is therefore irrelevant to the score;
what matters is the pixel ranking and the area budget at which it is cut. Phase-2's rank
ensemble "won" with a transform that turned the top 20 % of pixels into exactly 1.0 — a binary
mask — while single models were only offered soft transforms (see audit.py).

Algorithm (exact, O(N log N) for the whole curve)
-------------------------------------------------
Kernel offsets with d <= R fall on a few distance levels d_0 = 0 < d_1 < ... (100, 141, 200,
224, 283, 300 m at 100 m pixels) with k_j = max(1 - d_j / R, 0). For every truth pixel g let
c_j(g) = max p over offsets at distance <= d_j. For the mask {p >= t}:

    TP_t(g) = max{k_j : c_j(g) >= t} = sum_j (k_j - k_{j+1}) 1[c_j(g) >= t]     (k_{J+1} = 0)
    recall@tol(t) = mean_g 1[c_{j(tol)}(g) >= t]
    FP_t = sum_{x: p(x) >= t, x in eval \\ exclude} (1 - k(dist(x, G)))

so all thresholds are answered by sorting c_j and the FP weights once. Conventions match
``metric.MaskedScorer`` exactly (eval mask zeroes predictions outside it; excluded pixels
neither count as truth nor as FP, but may still support TP) — see tests/test_phase3_scoring.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from ..metric import DTI_ALPHA, DTI_BETA, DTI_EPS, DTI_RADIUS_M, _bbox

DEFAULT_AREA_FRACS = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40)


def distance_levels(radius_m: float, pixel_m: tuple[float, float]) -> list[tuple[float, list[tuple[int, int]]]]:
    """[(distance_m, [(dr, dc), ...]), ...] for all offsets with d <= radius, ascending distance."""
    py, px = pixel_m
    rr, rc = int(np.floor(radius_m / py)), int(np.floor(radius_m / px))
    lv: dict[float, list] = {}
    for dr in range(-rr, rr + 1):
        for dc in range(-rc, rc + 1):
            d = round(float(np.hypot(dr * py, dc * px)), 6)
            if d <= radius_m + 1e-9:
                lv.setdefault(d, []).append((dr, dc))
    return sorted(lv.items())


@dataclass
class _Sorted:
    values: np.ndarray         # ascending
    def count_ge(self, t: np.ndarray) -> np.ndarray:
        return len(self.values) - np.searchsorted(self.values, t, side="left")


class BinaryCurve:
    """All-threshold DTI / recall / area of the masks {pred >= t} for one (truth, eval, exclude) triple."""

    def __init__(self, pred: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, exclude_mask: np.ndarray | None = None,
                 *, far_mask: np.ndarray | None = None, radius_m: float = DTI_RADIUS_M,
                 pixel_m: tuple[float, float] = (100.0, 100.0), tolerances=(100.0, 200.0, 300.0),
                 alpha: float = DTI_ALPHA, beta: float = DTI_BETA, eps: float = DTI_EPS):
        self.alpha, self.beta, self.eps, self.radius_m, self.pixel_m = alpha, beta, eps, radius_m, pixel_m
        ev_full = np.asarray(eval_mask, bool)
        pad = int(np.ceil(radius_m / min(pixel_m))) + 1
        self.sl = _bbox(ev_full, pad)
        ev = ev_full[self.sl]
        ex = np.zeros_like(ev) if exclude_mask is None else np.asarray(exclude_mask, bool)[self.sl]
        p = np.asarray(pred, np.float64)[self.sl]
        p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 0.0)
        p = np.where(ev, p, -np.inf)                       # outside eval: never in a mask, never supports TP
        self.ev_px = int(ev.sum())
        G = np.asarray(truth, bool)[self.sl] & ev & ~ex
        self.G_rc = np.argwhere(G)
        self.n = len(self.G_rc)
        # -- truth-side cumulative maxima per distance level
        levels = distance_levels(radius_m, pixel_m)
        r = max(abs(o[0]) for _, offs in levels for o in offs)
        P = np.pad(p, r, mode="constant", constant_values=-np.inf)
        gr, gc = self.G_rc[:, 0] + r, self.G_rc[:, 1] + r
        self.k = np.array([max(1.0 - d / radius_m, 0.0) for d, _ in levels])
        self.d = np.array([d for d, _ in levels])
        cum = np.full(self.n, -np.inf)
        self.c: list[np.ndarray] = []
        for _, offs in levels:
            ring = np.max(np.stack([P[gr + dr, gc + dc] for dr, dc in offs]), 0) if self.n else np.empty(0)
            cum = np.maximum(cum, ring)
            self.c.append(cum.copy())
        self.c_sorted = [_Sorted(np.sort(c)) for c in self.c]
        self.dk = self.k - np.r_[self.k[1:], 0.0]          # telescoping weights
        self.tol_level = {t: int(np.searchsorted(self.d, t + 1e-9, side="right") - 1) for t in tolerances}
        # -- prediction side
        fp_region = ev & ~ex
        if self.n:
            dist = ndimage.distance_transform_edt(~G, sampling=pixel_m)
            kd = np.clip(1.0 - dist / radius_m, 0.0, None)
        else:
            kd = np.zeros(ev.shape)
        pv = p[fp_region]
        order = np.argsort(pv, kind="stable")
        self.fp_vals = pv[order]
        self.fp_cum = np.r_[0.0, np.cumsum((1.0 - kd[fp_region])[order])]   # prefix sums (ascending p)
        self.near_cum = np.r_[0.0, np.cumsum(kd[fp_region][order])]
        self.ev_vals = np.sort(p[ev])
        if far_mask is not None:
            fm = np.asarray(far_mask, bool)[self.sl] & ev
            self.far_vals = np.sort(p[fm])
        else:
            self.far_vals = None
        self.pixel_km2 = pixel_m[0] * pixel_m[1] / 1e6

    # -- thresholds -----------------------------------------------------------
    def threshold_for_area(self, fracs) -> np.ndarray:
        """Threshold whose mask covers ~frac of the eval pixels (ties may enlarge the area)."""
        fracs = np.atleast_1d(np.asarray(fracs, float))
        n = len(self.ev_vals)
        k = np.clip(np.ceil(fracs * n).astype(int), 1, n)
        return self.ev_vals[n - k]

    def _suffix(self, cum: np.ndarray, vals: np.ndarray, t: np.ndarray) -> np.ndarray:
        i = np.searchsorted(vals, t, side="left")
        return cum[-1] - cum[i]

    def at(self, thresholds) -> pd.DataFrame:
        t = np.atleast_1d(np.asarray(thresholds, float))
        tp = np.zeros(len(t))
        for dk, cs in zip(self.dk, self.c_sorted):
            if dk > 0:
                tp += dk * cs.count_ge(t)
        fp = self._suffix(self.fp_cum, self.fp_vals, t)
        near = self._suffix(self.near_cum, self.fp_vals, t)
        area = len(self.ev_vals) - np.searchsorted(self.ev_vals, t, side="left")
        fn = self.n - tp
        dti = tp / (tp + self.alpha * fp + self.beta * fn + self.eps)
        df = pd.DataFrame({"threshold": t, "eval_px": self.ev_px, "area_px": area, "area_frac": area / max(self.ev_px, 1),
                           "area_km2": area * self.pixel_km2, "tp_w": tp, "fp_w": fp, "fn_w": fn,
                           "n_truth": self.n, "near_mass": near, "dti": dti,
                           "recall_w": tp / max(self.n, 1)})
        for tol, j in self.tol_level.items():
            df[f"recall@{tol:g}m"] = self.c_sorted[j].count_ge(t) / max(self.n, 1) if self.n else np.nan
        if self.far_vals is not None:
            far = len(self.far_vals) - np.searchsorted(self.far_vals, t, side="left")
            df["far_area_km2"] = far * self.pixel_km2
        return df

    def at_area(self, fracs=DEFAULT_AREA_FRACS) -> pd.DataFrame:
        fracs = np.atleast_1d(np.asarray(fracs, float))
        df = self.at(self.threshold_for_area(fracs))
        df.insert(0, "target_frac", fracs)
        return df

    def best(self, fracs=None) -> dict:
        """Best DTI over a dense area grid (in-sample on this zone's labels: diagnostics only)."""
        fracs = np.geomspace(1e-3, 0.6, 80) if fracs is None else fracs
        df = self.at_area(fracs)
        return df.loc[df.dti.idxmax()].to_dict()

    def per_truth_tp(self, threshold: float) -> np.ndarray:
        """TP support of every truth pixel (order of ``truth_rc``) for the mask {p >= threshold}."""
        tp = np.zeros(self.n)
        for dk, c in zip(self.dk, self.c):
            if dk > 0:
                tp += dk * (c >= threshold)
        return tp

    def per_truth_hit(self, threshold: float, tol: float) -> np.ndarray:
        return self.c[self.tol_level[tol]] >= threshold

    @property
    def truth_rc(self) -> np.ndarray:
        """Full-grid (row, col) of the scored truth pixels."""
        return self.G_rc + np.array([self.sl[0].start, self.sl[1].start])


# ---------------------------------------------------------------------------
# pooling across zones and cross-fitted selection
# ---------------------------------------------------------------------------
def dti_from(df: pd.DataFrame, alpha=DTI_ALPHA, beta=DTI_BETA, eps=DTI_EPS) -> float:
    tp, fp, fn = df.tp_w.sum(), df.fp_w.sum(), df.fn_w.sum()
    return float(tp / (tp + alpha * fp + beta * fn + eps))


def zone_curves(pred: np.ndarray, truth: np.ndarray, zones: np.ndarray, valid: np.ndarray, *,
                exclude_mask: np.ndarray | None = None, far_mask: np.ndarray | None = None,
                fracs=DEFAULT_AREA_FRACS, mode: str = "per_zone") -> pd.DataFrame:
    """Binary-mask components per zone (zone < 0 skipped) on an area-fraction grid.

    mode="per_zone": each zone is cut at its own quantile (label-free, transductive — the same rule
    can be applied to any map). mode="global": one threshold value per fraction, taken from the
    pooled valid distribution (what a single global cut of a stitched map does).
    """
    zone_ids = [int(z) for z in np.unique(zones[valid & (zones >= 0)])]
    rows = []
    if mode == "global":
        v = np.sort(np.nan_to_num(pred[valid & (zones >= 0)], nan=0.0))
        k = np.clip(np.ceil(np.asarray(fracs) * len(v)).astype(int), 1, len(v))
        gthr = v[len(v) - k]
    for z in zone_ids:
        c = BinaryCurve(pred, truth, valid & (zones == z), exclude_mask, far_mask=far_mask)
        df = c.at(gthr) if mode == "global" else c.at(c.threshold_for_area(fracs))
        df.insert(0, "target_frac", np.asarray(fracs, float))
        df.insert(0, "zone", z)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["mode"] = mode
    return out


def pooled(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum components over zones for every ``by`` key and recompute the ratio metrics."""
    df = df.copy()
    rec = [c for c in df.columns if c.startswith("recall@")]
    for col in rec:                                    # truth-weighted average == pooled recall
        df[col] = df[col].fillna(0) * df.n_truth
    num = [c for c in ("tp_w", "fp_w", "fn_w", "n_truth", "eval_px", "area_px", "area_km2", "near_mass",
                       "far_area_km2", *rec) if c in df]
    out = df.groupby(by, dropna=False)[num].sum().reset_index()
    out["dti"] = out.tp_w / (out.tp_w + DTI_ALPHA * out.fp_w + DTI_BETA * out.fn_w + DTI_EPS)
    out["recall_w"] = out.tp_w / out.n_truth.clip(lower=1)
    for col in rec:
        out[col] = out[col] / out.n_truth.clip(lower=1)
    if "eval_px" in out:
        out["area_frac"] = out.area_px / out.eval_px.clip(lower=1)
    return out


def crossfit_select(table: pd.DataFrame, params: list[str], zone_col: str = "zone") -> tuple[pd.DataFrame, dict]:
    """For each zone pick the parameter maximising pooled DTI on the *other* zones; score it on the zone.

    Returns (per-zone choices with held-out components, summary). The summary also reports the
    in-sample optimum (selected on all zones) so the selection optimism is visible.
    """
    rows = []
    zones = sorted(table[zone_col].unique())
    for z in zones:
        other = table[table[zone_col] != z].groupby(params).apply(dti_from).rename("dti_other").reset_index()
        best = other.sort_values("dti_other", ascending=False).iloc[0]
        sel = np.ones(len(table), bool)
        for p in params:
            sel &= (table[p] == best[p]).to_numpy()
        held = table[(table[zone_col] == z) & sel]
        rows.append({zone_col: z, **{p: best[p] for p in params}, "dti_other": float(best.dti_other),
                     "tp_w": held.tp_w.sum(), "fp_w": held.fp_w.sum(), "fn_w": held.fn_w.sum(),
                     "dti_heldout": dti_from(held)})
    per = pd.DataFrame(rows)
    allz = table.groupby(params).apply(dti_from).rename("dti_all").reset_index().sort_values("dti_all", ascending=False)
    top = allz.iloc[0].to_dict()
    summ = {"dti_crossfit": dti_from(per), "dti_insample": float(top["dti_all"]),
            "optimism": float(top["dti_all"]) - dti_from(per),
            "fold_min": float(per.dti_heldout.min()), "fold_max": float(per.dti_heldout.max()),
            "params_all": {p: top[p] for p in params},
            "params_per_zone": per[[zone_col] + params].to_dict("records")}
    # was the chosen value on the edge of the searched grid? (grid truncation)
    edge = []
    for p in params:
        if not pd.api.types.is_numeric_dtype(table[p]):
            continue
        vals = np.sort(table[p].unique())
        if len(vals) > 2:
            edge += [bool(v in (vals[0], vals[-1])) for v in per[p]]
    summ["choice_on_grid_edge"] = bool(any(edge))
    return per, summ
