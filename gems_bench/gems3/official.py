"""Evaluation units under the verified competition contract (docs/GEMS_OFFICIAL_METRIC_AUDIT.md).

What the organizers stated (community.drivendata.org, competition 306):

* t/11516 (2026-09-16, 09-21): hidden-test truth is *new* faults only. Known USGS/INGENIOUS pixels are masked
  "pixel-exact - it is identical to the provided set of training fault labels"; they "do not count towards penalty
  terms"; a prediction near a known trace but far from new truth "will be fully penalized, i.e., the buffer does
  not apply to known faults"; new truth can lie within 300 m of a known trace (corrections); "it should not
  matter whether these known faults are included with predictions or not".
* t/11550 (2026-10-01): one Tversky index pooled over all pixels of the public subset (private: same; final
  round: entire GeoDAWN area). t/11556 (2026-10-07): no weighting beyond the published distance kernel; the same
  exclusion raster in both rounds.

Tasks (one ``Unit`` per fold / split; every array is cropped to the unit's evaluation bbox + kernel pad)::

    geo               truth = all mapped faults in fold k; no known mask              (historical geographic DTI)
    disc_official     truth = withheld systems of split d; retained faults masked pixel-exactly and their
                      predictions ignored (exclude_supports_tp=False)                 (hidden-test analogue)
    disc_excl         M6 / Phase-2 'excl': retained faults dilated 300 m excluded, may support TP   (parity only)
    disc_retained_fp  retained faults not masked: predictions on them are FP          (pessimistic sensitivity)

Pseudo-discovery limitation: withheld units are whole *systems* (linked within 1 km), so the class "new truth
within 300 m of a known trace" (corrections, splays, tip extensions) is under-represented compared with the
hidden test. ``disc_official`` is therefore a proxy, never a leaderboard estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from ..metric import (DTI_ALPHA, DTI_BETA, DTI_EPS, DTI_RADIUS_M, OFFICIAL_EXCLUDE_SUPPORTS_TP, DTIComponents,
                      _bbox, clean_prediction, dti_components, kernel_offsets)

TASKS = {
    "geo": "geographic folds: truth = all mapped faults of the held-out fold; no known-fault mask",
    "disc_official": "pseudo-discovery: truth = withheld systems; retained faults masked pixel-exactly, "
                     "masked predictions ignored (official reading)",
    "disc_excl": "pseudo-discovery, Phase-2/M6 'excl': retained faults dilated 300 m excluded, masked "
                 "predictions may support TP",
    "disc_retained_fp": "pseudo-discovery: retained faults not masked; predictions on them count as FP",
}
PAD_PX = int(np.ceil(DTI_RADIUS_M / 100.0)) + 1


class FastScorer:
    """``metric.MaskedScorer`` with TP gathered at the truth pixels only (identical components; tested).

    MaskedScorer takes the kernel max over the whole crop for every map; the representation search scores
    thousands of maps per unit, so here TP = sum_g max_j k_j P[g + o_j] is a gather on |G| x |offsets| values
    and FP one product with a precomputed weight raster."""

    def __init__(self, truth, eval_mask, exclude_mask=None, *, exclude_supports_tp: bool = True,
                 radius_m: float = DTI_RADIUS_M, pixel_m=(100.0, 100.0), alpha=DTI_ALPHA, beta=DTI_BETA, eps=DTI_EPS):
        truth, ev = np.asarray(truth, bool), np.asarray(eval_mask, bool)
        ex = np.zeros(ev.shape, bool) if exclude_mask is None else np.asarray(exclude_mask, bool)
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.ev = ev
        self.zero = ~ev | ex if not exclude_supports_tp else ~ev
        G = truth & ev & ~ex
        self.n = int(G.sum())
        fp_region = ev & ~ex
        offs = kernel_offsets(radius_m, pixel_m)
        self.r = max(max(abs(a) for a, _, _ in offs), max(abs(b) for _, b, _ in offs))
        h, w = ev.shape
        self.W = w + 2 * self.r
        gr, gc = np.nonzero(G)
        self.g_rc = (gr, gc)
        self.k = np.array([k for _, _, k in offs])[:, None]
        self.idx = np.stack([(gr + dr + self.r) * self.W + (gc + dc + self.r) for dr, dc, _ in offs])
        if self.n:
            dist = ndimage.distance_transform_edt(~G, sampling=pixel_m)
            self.fp_w = np.where(fp_region, 1.0 - np.clip(1.0 - dist / radius_m, 0.0, None), 0.0)
        else:
            self.fp_w = fp_region.astype(np.float64)

    def components(self, pred: np.ndarray) -> DTIComponents:
        p = clean_prediction(pred)
        p[self.zero] = 0.0
        fp = float((p * self.fp_w).sum())
        mass = float(p[self.ev].sum())
        if not self.n:
            return DTIComponents(0.0, fp, 0.0, 0, mass, self.alpha, self.beta, self.eps)
        tp = float(self.support(p, cleaned=True).sum())
        return DTIComponents(tp, fp, float(self.n - tp), self.n, mass, self.alpha, self.beta, self.eps)

    def support(self, pred: np.ndarray, cleaned: bool = False) -> np.ndarray:
        """TP support max_x p(x) k(d) of every scored truth pixel (np.nonzero order of the scored truth)."""
        if not self.n:
            return np.zeros(0)
        p = pred if cleaned else clean_prediction(pred)
        if not cleaned:
            p[self.zero] = 0.0
        return (self.k * np.pad(p, self.r).ravel()[self.idx]).max(0)


@dataclass
class Unit:
    task: str
    zone: int                          # fold k (geo) or split d (disc)
    sl: tuple                          # crop of the full grid
    ev: np.ndarray                     # cropped evaluation mask
    truth: np.ndarray                  # cropped truth (before exclusion)
    exclude: np.ndarray | None
    supports_tp: bool
    known: np.ndarray | None           # cropped known faults visible to this unit (retained), else None
    _scorer: FastScorer | None = field(default=None, repr=False)

    @property
    def is_disc(self) -> bool:
        return self.task.startswith("disc")

    @property
    def key(self) -> str:
        return f"{self.task}:{self.zone}"

    def crop(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a)[self.sl]

    @property
    def scorer(self) -> FastScorer:
        if self._scorer is None:
            self._scorer = FastScorer(self.truth, self.ev, self.exclude, exclude_supports_tp=self.supports_tp)
        return self._scorer

    def release(self) -> None:
        self._scorer = None

    def components(self, q: np.ndarray):
        return self.scorer.components(np.asarray(q, np.float64))

    def curve(self, p: np.ndarray, far_mask: np.ndarray | None = None):
        from ..p3.curves import BinaryCurve

        return BinaryCurve(p, self.truth, self.ev, self.exclude, far_mask=far_mask,
                           exclude_supports_tp=self.supports_tp)

    def eval_truth(self) -> np.ndarray:
        """Scored truth pixels (truth inside eval, minus exclusion)."""
        t = self.truth & self.ev
        return t & ~self.exclude if self.exclude is not None else t


def _dilate(mask: np.ndarray, metres: float) -> np.ndarray:
    return ndimage.distance_transform_edt(~mask, sampling=(100.0, 100.0)) <= metres


def _unit(task, zone, ev_full, truth_full, exclude_full, supports, known_full) -> Unit:
    sl = _bbox(ev_full, PAD_PX)
    c = (lambda a: None if a is None else np.ascontiguousarray(np.asarray(a, bool)[sl]))
    return Unit(task, int(zone), sl, c(ev_full), c(truth_full), c(exclude_full), bool(supports), c(known_full))


def build_units(ctx, tasks) -> list[Unit]:
    """Units for ``tasks`` on the context's folds and discovery splits (read-only; masks recomputed, cheap)."""
    lab, valid = ctx.data.labels, ctx.data.valid
    out = []
    for task in tasks:
        if task == "geo":
            fm = ctx.folds.fold_map
            for k in ctx.fold_ids:
                out.append(_unit("geo", k, valid & (fm == k), lab, None, True, None))
            continue
        for d in range(ctx.splits.n_splits):
            G = ctx.splits.withheld(d)
            retained = lab & ~G
            ev = ctx.splits.eval_zone(d) & valid
            if task == "disc_official":
                out.append(_unit(task, d, ev, G, retained, OFFICIAL_EXCLUDE_SUPPORTS_TP, retained))
            elif task == "disc_excl":
                out.append(_unit(task, d, ev, G, _dilate(retained, 300.0) & ~G, True, retained))
            elif task == "disc_retained_fp":
                out.append(_unit(task, d, ev, G, None, True, retained))
            else:
                raise KeyError(task)
    return out


def unit_score(ctx, model: str, u: Unit) -> np.ndarray | None:
    """Cropped honest score of ``model`` for ``u``: OOF for geo folds, the split's discovery map for disc."""
    p = ctx.oof(model) if u.task == "geo" else ctx.discovery_pred(model, u.zone)
    return None if p is None else np.asarray(p, np.float32)[u.sl]


# =============================================================================== contract + audits
CONTRACT = [
    # item, official, internal (before GEMS3), status, source
    ("kernel", "k(d) = max(1 - d/R, 0), R = 300 m, d between pixel centres", "same (metric.kernel_offsets)",
     "VERIFIED (problem description)", "page/967"),
    ("alpha_beta", "alpha = 0.2 (FP), beta = 0.8 (FN)", "same", "VERIFIED (problem description)", "page/967"),
    ("tp_fp_fn", "TP = sum_g max_x p k; FP = sum_x p (1 - max_g k); FN = sum_g (1 - max_x p k)", "same",
     "VERIFIED (problem description)", "page/967"),
    ("eps", "present, value not published", "1e-6", "UNSPECIFIED", "page/967"),
    ("truth", "new faults only (not in USGS/INGENIOUS)", "geo task: all mapped faults (proxy)",
     "VERIFIED (organizer)", "t/11516"),
    ("known_mask_geometry", "pixel-exact, identical to the training fault labels, no buffer",
     "disc 'excl' policy: retained faults DILATED 300 m (over-forgiving)", "VERIFIED (organizer) - MISMATCH fixed by "
     "disc_official", "t/11516"),
    ("known_mask_fp", "masked pixels do not count towards penalty terms", "same", "VERIFIED (organizer)", "t/11516"),
    ("near_known_fp", "a prediction near a known trace but far from new truth is fully penalised",
     "'excl' forgave everything within 300 m of a retained fault", "VERIFIED (organizer) - MISMATCH fixed", "t/11516"),
    ("masked_pred_supports_tp", "'should not matter whether known faults are included with predictions or not' "
     "=> masked predictions have no effect", "excluded predictions supported nearby TP",
     "INFERRED from organizer wording; no public scorer - both conventions reported", "t/11516"),
    ("new_truth_near_known", "new truth may lie within 300 m of a known trace (corrections)",
     "pseudo-discovery withholds whole systems: under-represents this class", "VERIFIED (organizer) - proxy limitation",
     "t/11516"),
    ("aggregation", "one Tversky index pooled over all public-subset pixels; private same; final = whole area",
     "pooled components over folds / splits", "VERIFIED (organizer)", "t/11550"),
    ("weighting", "no weighting beyond the distance kernel", "none", "VERIFIED (organizer)", "t/11556"),
    ("format", "single-band float32 GeoTIFF, EPSG:32611, 100 m, training bounds, values in [0, 1], null/NaN "
     "outside bounds", "same (submission.write_submission)", "VERIFIED (problem description)", "page/967"),
    ("nan_inside_bounds", "not stated", "NaN outside the data footprint; GEMS3 default writes 0 where the template "
     "is finite", "UNVERIFIED", "-"),
    ("clip", "values must be in [0, 1]; no statement on clipping", "clipped locally", "UNVERIFIED", "-"),
    ("public_private_split", "chunking / subsets not disclosed", "-", "UNKNOWN", "t/11550"),
    ("reference_scorer", "no scorer published; the reference notebook notes thresholding may help",
     "-", "VERIFIED absent (github.com/drivendataorg/gems-prize-reference-solution)", "reference solution"),
]


def contract_table() -> pd.DataFrame:
    return pd.DataFrame(CONTRACT, columns=["item", "official", "internal_before_gems3", "status", "source"])


def _line(shape, r, c0, c1):
    a = np.zeros(shape, bool)
    a[r, c0:c1] = True
    return a


def masking_cases() -> list[dict]:
    """Synthetic cases where the known-fault conventions disagree (40 x 40 px at 100 m)."""
    sh = (40, 40)
    known = _line(sh, 20, 5, 35)
    new_adj = _line(sh, 21, 5, 35)                     # a correction 100 m off a known trace
    new_far = _line(sh, 5, 5, 35)                      # an independent new fault
    cases = [
        {"case": "predict_known_only", "truth": new_adj, "known": known, "pred": known.astype(float),
         "note": "TP only if masked predictions support TP"},
        {"case": "predict_new_exactly", "truth": new_adj, "known": known, "pred": new_adj.astype(float),
         "note": "every convention: DTI 1"},
        {"case": "ring_far_from_new", "truth": new_far, "known": known, "pred": (new_far | _line(sh, 22, 5, 35)).astype(float),
         "note": "ring 200 m off a known trace: FP officially, forgiven by 300 m dilated exclusion"},
        {"case": "known_fill_plus_new", "truth": new_far, "known": known, "pred": (new_far | known).astype(float),
         "note": "known pixels set to 1: neutral officially"},
    ]
    return cases


def masking_audit() -> pd.DataFrame:
    """DTI of every synthetic case under four conventions (computed in Colab; mirrored by tests)."""
    rows = []
    for c in masking_cases():
        t, k, p = c["truth"], c["known"], c["pred"]
        G = t & ~k
        conv = {
            "official_pixel_exact_no_support": dict(exclude_mask=k, exclude_supports_tp=False),
            "pixel_exact_with_support": dict(exclude_mask=k, exclude_supports_tp=True),
            "excl_dilated_300m_with_support": dict(exclude_mask=_dilate(k, 300.0) & ~G, exclude_supports_tp=True),
            "known_is_fp": dict(exclude_mask=None),
        }
        for name, kw in conv.items():
            comp = dti_components(p, G, **kw)
            rows.append({"case": c["case"], "convention": name, **comp.as_dict(), "note": c["note"]})
    return pd.DataFrame(rows)


def official_vs_internal(ctx, models: list[str], fracs=(0.01, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    """Binary top-a DTI of each model's discovery maps under every disc convention (pooled over splits)."""
    from ..p3.curves import pooled

    rows = []
    for task in ("disc_excl", "disc_official", "disc_retained_fp"):
        for u in build_units(ctx, [task]):
            for m in models:
                p = unit_score(ctx, m, u)
                if p is None:
                    continue
                df = u.curve(p).at_area(fracs)
                df.insert(0, "model", m)
                df.insert(0, "zone", u.zone)
                df.insert(0, "task", task)
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    out = pooled(long, ["task", "model", "target_frac"])
    out["unit"] = "pooled_splits"
    return pd.concat([out, long.assign(unit="per_split")], ignore_index=True)
