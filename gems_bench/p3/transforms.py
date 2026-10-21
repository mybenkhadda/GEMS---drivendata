"""Output-transform families, all selected out-of-fold.

Every family maps a raw score map to [0, 1]. Parametric families are evaluated on a grid; the
parameter for geographic fold k is chosen on the *other* folds' labels (``curves.crossfit_select``)
and scored on fold k, so no reported number uses its own evaluation labels for selection.
The isotonic family is fitted on the other folds' pixels directly (cross-fitted fit, no grid).

Families
--------
raw              identity (clip to [0, 1])
linear           q = clip(s p)
temperature      q = sigmoid((logit p - b) / T)
isotonic         monotone piecewise map p -> E[k(dist to truth)] fitted on other folds (ML calibration)
power_floor      Phase-2 family q = clip(s p^gamma), 0 below a floor (Phase-2 grid + extended grid)
rank_power_p2    Phase-2 rank family on *global ordinal* ranks (exact Phase-2 rank-ensemble transform)
rank_power       same grid on per-fold *average* ranks (ties and position bias removed)
soft_threshold   linear ramp of width w around the per-fold top-a quantile
hard_threshold   binary top-a mask per fold (exact all-threshold engine, curves.BinaryCurve)

Per-fold quantiles are label-free and transductive: at submission time the same rule is applied
to the full map, so the OOF and submission transforms are the same *rule* (not the same values).
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from ..ensemble import rank_normalise
from ..metric import MaskedScorer, apply_output_transform
from .curves import DEFAULT_AREA_FRACS, crossfit_select, zone_curves

P2_GRID = {"gamma": (0.5, 1.0, 2.0, 3.0), "scale": (0.5, 1.0, 1.5, 2.0, 3.0), "floor": (0.0, 0.05, 0.1, 0.2, 0.4)}
EXT_GRID = {"gamma": (1.0, 2.0, 4.0), "scale": (3.0, 6.0, 12.0), "floor": (0.4, 0.55, 0.7)}
RANK_GRID = {"gamma": (1.0, 4.0, 16.0), "scale": (0.5, 1.0, 2.0), "floor": (0.8, 0.9, 0.95, 0.98, 0.99)}

FAMILY_GRIDS: dict[str, list[dict]] = {
    "raw": [{}],
    "linear": [{"s": s} for s in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)],
    "temperature": [{"T": T, "b": b} for T in (0.25, 0.5, 1.0, 2.0) for b in (-1.0, 0.0, 1.0, 2.0)],
    "power_floor": [dict(zip(("gamma", "scale", "floor"), v)) for v in itertools.product(*P2_GRID.values())],
    "power_floor_ext": [dict(zip(("gamma", "scale", "floor"), v)) for v in itertools.product(*EXT_GRID.values())],
    "rank_power_p2": [dict(zip(("gamma", "scale", "floor"), v)) for v in itertools.product(*RANK_GRID.values())],
    "rank_power": [dict(zip(("gamma", "scale", "floor"), v)) for v in itertools.product(*RANK_GRID.values())],
    "soft_threshold": [{"a": a, "w": w} for a in (0.05, 0.10, 0.15, 0.20, 0.30) for w in (0.01, 0.05)],
}
ALL_FAMILIES = ["raw", "linear", "temperature", "isotonic", "power_floor", "power_floor_ext",
                "rank_power_p2", "rank_power", "soft_threshold", "hard_threshold"]


def quantile_map(p: np.ndarray, valid: np.ndarray, zones: np.ndarray | None = None, method: str = "average") -> np.ndarray:
    """Rank of p scaled to [0, 1], computed separately inside every zone (zone < 0 -> NaN)."""
    from scipy.stats import rankdata

    out = np.full(p.shape, np.nan, np.float32)
    if zones is None:
        return rank_normalise(p, valid, method)
    for z in np.unique(zones[valid & (zones >= 0)]):
        m = valid & (zones == z)
        v = np.nan_to_num(p[m], nan=0.0)
        r = rankdata(v, method=method) - 1
        out[m] = (r / max(len(v) - 1, 1)).astype(np.float32)
    return out


def _logit(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def apply_family(family: str, p: np.ndarray, prm: dict, u: np.ndarray | None = None) -> np.ndarray:
    """Apply one transform; ``u`` is the per-zone quantile map (needed by rank / threshold families)."""
    if family == "raw":
        q = np.clip(p, 0, 1)
    elif family == "linear":
        q = np.clip(prm["s"] * p, 0, 1)
    elif family == "temperature":
        q = 1.0 / (1.0 + np.exp(-(_logit(p) - prm["b"]) / prm["T"]))
    elif family in ("power_floor", "power_floor_ext"):
        q = apply_output_transform(p, prm["gamma"], prm["scale"], prm["floor"])
    elif family in ("rank_power_p2", "rank_power"):
        q = apply_output_transform(u, prm["gamma"], prm["scale"], prm["floor"])
    elif family == "soft_threshold":
        q = np.clip((u - (1.0 - prm["a"])) / prm["w"] + 0.5, 0, 1)
    elif family == "hard_threshold":
        q = (u >= 1.0 - prm["a"]).astype(np.float32)
    else:
        raise KeyError(family)
    return np.where(np.isfinite(p), q, np.nan).astype(np.float32)


class FoldScorers:
    """One MaskedScorer per geographic fold, shared by every model and transform (distance transforms once)."""

    def __init__(self, truth, fold_map, valid, exclude_mask=None):
        self.truth, self.fold_map, self.valid = truth, fold_map, valid
        self.ids = [int(f) for f in np.unique(fold_map[valid & (fold_map >= 0)])]
        self.s = {f: MaskedScorer(truth, valid & (fold_map == f), exclude_mask) for f in self.ids}

    def table(self, q: np.ndarray, **tags) -> list[dict]:
        rows = []
        for f in self.ids:
            c = self.s[f].components(q)
            rows.append({**tags, "zone": f, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "mass": c.pred_mass})
        return rows


def isotonic_crossfit(p: np.ndarray, truth: np.ndarray, fold_map: np.ndarray, valid: np.ndarray, scorers: FoldScorers,
                      rng: np.random.Generator, n_fit: int = 2_000_000, pixel_m=(100.0, 100.0)) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit p -> E[k(dist)] on the other folds (kernel credit as target), apply to fold k. Returns (rows, map)."""
    from scipy import ndimage
    from sklearn.isotonic import IsotonicRegression

    kd = np.clip(1 - ndimage.distance_transform_edt(~truth, sampling=pixel_m) / 300.0, 0, None)
    q = np.full(p.shape, np.nan, np.float32)
    rows = []
    for f in scorers.ids:
        tr = np.flatnonzero((valid & (fold_map >= 0) & (fold_map != f)).ravel())
        tr = rng.choice(tr, min(n_fit, len(tr)), replace=False)
        x, y = np.nan_to_num(p.ravel()[tr]), kd.ravel()[tr]
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)
        m = valid & (fold_map == f)
        q[m] = iso.predict(np.nan_to_num(p[m])).astype(np.float32)
    rows += scorers.table(q, family="isotonic", param="{}")
    return pd.DataFrame(rows), q


def family_table(model: str, p: np.ndarray, truth: np.ndarray, fold_map: np.ndarray, valid: np.ndarray,
                 scorers: FoldScorers, families=ALL_FAMILIES, rng=None, reuse_p2_grid: pd.DataFrame | None = None,
                 area_fracs=DEFAULT_AREA_FRACS) -> pd.DataFrame:
    """Per-fold DTI components for every (family, parameter) of one OOF map."""
    rng = rng or np.random.default_rng(2026)
    u_zone = quantile_map(p, valid, fold_map, "average")
    u_glob = rank_normalise(p, valid, "ordinal")
    rows = []
    for fam in families:
        if fam == "isotonic":
            t, _ = isotonic_crossfit(p, truth, fold_map, valid, scorers, rng)
            rows += t.to_dict("records")
            continue
        if fam == "hard_threshold":
            zc = zone_curves(p, truth, fold_map, valid, fracs=area_fracs, mode="per_zone")
            for r in zc.itertuples():
                rows.append({"family": fam, "param": json.dumps({"a": float(r.target_frac)}), "a": float(r.target_frac),
                             "zone": int(r.zone), "tp_w": r.tp_w, "fp_w": r.fp_w, "fn_w": r.fn_w, "mass": r.area_px})
            continue
        if fam == "power_floor" and reuse_p2_grid is not None and len(reuse_p2_grid):
            for r in reuse_p2_grid.itertuples():          # Phase-2 transform_grid.csv: same family, already scored
                prm = {"gamma": r.gamma, "scale": r.scale, "floor": r.floor}
                rows.append({"family": fam, "param": json.dumps(prm), **prm, "zone": int(r.fold),
                             "tp_w": r.tp_w, "fp_w": r.fp_w, "fn_w": r.fn_w, "mass": np.nan})
            continue
        u = u_glob if fam == "rank_power_p2" else u_zone
        for prm in FAMILY_GRIDS[fam]:
            q = apply_family(fam, p, prm, u)
            rows += scorers.table(q, family=fam, param=json.dumps(prm), **prm)
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    return df


def compare_families(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fitted selection per (model, family). Returns (summary, per-fold choices)."""
    summ, per_all = [], []
    for (model, fam), g in table.groupby(["model", "family"], sort=False):
        per, s = crossfit_select(g, ["param"])
        prm_cols = [c for c in g.columns if c not in ("model", "family", "param", "zone", "tp_w", "fp_w", "fn_w", "mass")]
        edge = False
        for c in prm_cols:
            vals = np.sort(g[c].dropna().unique())
            if len(vals) > 2:
                chosen = [json.loads(x).get(c) for x in per.param]
                edge |= any(v is not None and v in (vals[0], vals[-1]) for v in chosen)
        per.insert(0, "family", fam)
        per.insert(0, "model", model)
        per_all.append(per)
        row = {"model": model, "family": fam, "n_params": g.param.nunique(), "dti_crossfit": s["dti_crossfit"],
               "dti_insample": s["dti_insample"], "optimism": s["optimism"], "fold_min": s["fold_min"],
               "fold_max": s["fold_max"], "choice_on_grid_edge": edge, "param_all_folds": json.dumps(s["params_all"])}
        for r in per.itertuples():
            row[f"dti_fold{r.zone}"] = r.dti_heldout
            row[f"param_fold{r.zone}"] = r.param
        summ.append(row)
    return pd.DataFrame(summ), pd.concat(per_all, ignore_index=True)
