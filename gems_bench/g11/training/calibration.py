"""Probability calibration fitted ONLY on the permitted calibration split (``calibration.region``, default stack).

Methods: none, platt (logistic regression on the logit), isotonic. Reliability (expected calibration error and a
binned table) is reported per region; with mapped-fault labels the "observed rate" is the mapped-fault rate, which
under-states true fault prevalence (incomplete maps).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class CalibrationLeakage(RuntimeError):
    pass


@dataclass
class Calibrator:
    method: str
    model: object = None
    fitted_split: str | None = None

    def fit(self, p: np.ndarray, y: np.ndarray, split: str, allowed: str = "stack") -> "Calibrator":
        if split != allowed:
            raise CalibrationLeakage(f"calibration must be fitted on '{allowed}', got '{split}'")
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = np.clip(p[ok], 1e-6, 1 - 1e-6), y[ok]
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p))[:, None], y)
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        elif self.method != "none":
            raise ValueError(f"unknown calibration method {self.method}")
        self.fitted_split = split
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        fin = np.isfinite(p)
        out = np.full(p.shape, np.nan, np.float32)
        if not fin.any():
            return out
        x = np.clip(p[fin], 1e-6, 1 - 1e-6)
        if self.method == "none":
            out[fin] = x
        elif self.method == "platt":
            out[fin] = self.model.predict_proba(np.log(x / (1 - x))[:, None])[:, 1]
        else:
            out[fin] = self.model.predict(x)
        return out


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin_lo": e[b], "bin_hi": e[b + 1], "mean_p": float(p[m].mean()), "observed": float(y[m].mean()),
                         "n": int(m.sum())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return pd.DataFrame(rows), float(ece)


def reliability_by_region(p, y, region, bins: int = 10) -> pd.DataFrame:
    out = []
    for r in np.unique(region[np.isfinite(region)] if region.dtype.kind == "f" else region):
        m = region == r
        tab, ece = reliability(p[m], y[m], bins)
        out.append({"region": r, "ece": ece, "n": int(m.sum()), "prevalence": float(np.nanmean(y[m]))})
    return pd.DataFrame(out)
