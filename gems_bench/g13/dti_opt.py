"""DTI-oriented output optimisation (sections 29-30).

1. ``synthetic_sensitivity``: controlled experiments on a synthetic fault showing how the official metric responds
   to prediction width, displacement, background confidence, confidence gamma, false-positive length and recall.
2. ``tournament``: every output representation (raw, transforms, binary area budgets, crest lines, topology
   operations, soft outputs, known-fault ring, regime-balanced budgets) is scored on every validation unit; the
   choice for each unit is made on the *other* units (cross-fitted), the complete search table is saved, and the
   chosen configuration is evaluated **once** on the untouched TEST partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3 import representation as R
from ..metric import apply_output_transform, dti_components
from ..postprocess.topology import link_endpoints, remove_small
from .evaluation import crossfit, pooled_dti


# =============================================================================== synthetic sensitivity
def _line(shape=(160, 160), r=80, c0=20, c1=140):
    a = np.zeros(shape, bool)
    a[r, c0:c1] = True
    return a


def synthetic_sensitivity() -> pd.DataFrame:
    """DTI of controlled degradations of a perfect prediction of a 12 km straight fault (100 m pixels)."""
    t = _line()
    d = ndimage.distance_transform_edt(~t)
    rows = []

    def add(exp, x, p):
        c = dti_components(p.astype(np.float64), t)
        rows.append(
            {
                "experiment": exp,
                "x": float(x),
                "dti": c.dti,
                "tp_w": c.tp,
                "fp_w": c.fp,
                "fn_w": c.fn,
                "pred_mass": c.pred_mass,
            }
        )

    for w in range(0, 6):  # ribbon half-width (px)
        add("width_px", w, (d <= w).astype(float))
    for s in range(0, 6):  # lateral displacement (px)
        add("displacement_px", s, np.roll(t, s, axis=0).astype(float))
    for b in (0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):  # constant background confidence everywhere
        add("background", b, np.maximum(t.astype(float), b))
    soft = np.exp(-d / 2.0)  # a blurred probability ribbon
    for g in (0.25, 0.5, 1, 2, 3, 4, 6):
        add("gamma", g, soft**g)
    for L in (0, 0.25, 0.5, 1.0, 2.0):  # extra false-positive trace length (x truth length), far away
        fp = np.zeros_like(t)
        n = int(L * t.sum())
        fp[20, 10 : 10 + min(n, 140)] = True
        if n > 140:
            fp[140, 10 : 10 + min(n - 140, 140)] = True
        add("fp_length_ratio", L, (t | fp).astype(float))
    for rec in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        p = t.copy()
        cols = np.flatnonzero(t.any(0))
        p[:, cols[int(rec * len(cols)) :]] = False
        add("recall", rec, p.astype(float))
    for v in (0.25, 0.5, 0.75, 1.0):  # binary support at confidence v
        add("constant_confidence", v, t * v)
    return pd.DataFrame(rows)


# =============================================================================== representation grid
def representation_grid(cfg) -> list[dict]:
    """Candidate output representations, identity-like options first (ties prefer the simpler choice)."""
    reps = [{"name": "raw", "rep": {"select": "raw"}, "frac": None, "transform": None}]
    for g, s, f in (
        (1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (1.0, 2.0, 0.0),
        (2.0, 1.0, 0.1),
        (1.0, 1.5, 0.2),
        (3.0, 1.0, 0.0),
    ):
        reps.append({"name": f"transform_g{g}_s{s}_f{f}", "rep": {"select": "raw"}, "frac": None, "transform": (g, s, f)})
    for fr in cfg.area_fracs:
        reps.append({"name": "top", "rep": {"select": "top"}, "frac": fr, "transform": None})
        reps.append(
            {
                "name": "crest_w1",
                "rep": {"select": "crest", "post": [["dilate", {"radius_px": 1}]]},
                "frac": fr,
                "transform": None,
            }
        )
        reps.append(
            {
                "name": "top_remove_small6",
                "rep": {"select": "top"},
                "frac": fr,
                "transform": None,
                "topology": ("remove_small", 6),
            }
        )
        reps.append({"name": "top_link4", "rep": {"select": "top"}, "frac": fr, "transform": None, "topology": ("link", 4)})
        reps.append(
            {"name": "top_regime_balanced", "rep": {"select": "top"}, "frac": fr, "transform": None, "regime_balanced": True}
        )
        if cfg.allow_soft_outputs:
            reps.append(
                {
                    "name": "soft_top_v0.5",
                    "rep": {"select": "top", "post": [["soft_top", {"v_min": 0.5}]]},
                    "frac": fr,
                    "transform": None,
                }
            )
            reps.append(
                {
                    "name": "two_level_h1_0.3",
                    "rep": {"select": "top", "post": [["two_level", {"halo_px": 1, "halo_value": 0.3}]]},
                    "frac": fr,
                    "transform": None,
                }
            )
    out = []
    for r in reps:
        for ring in cfg.known_ring_px:
            rr = dict(r)
            rr["ring_px"] = int(ring)
            rr["key"] = f"{r['name']}|frac={r['frac']}|ring={ring}"
            out.append(rr)
    return out


def render(
    spec: dict,
    score: np.ndarray,
    ev: np.ndarray,
    known: np.ndarray | None,
    regimes: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
    crest: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one representation to a (cropped or full) score map. Labels are never read (``known`` = mapped faults
    visible to this unit, used only by the known-fault ring)."""
    s = np.where(ev, np.nan_to_num(score, nan=0.0), 0.0).astype(np.float32)
    rep = R.with_ring(spec["rep"], spec.get("ring_px", 0))
    elig = ev if eligible is None else eligible & ev
    if spec.get("ring_px", 0) > 0 and known is not None and known.any():
        ring = ndimage.binary_dilation(known, structure=R._disk(spec["ring_px"])) & ~known
        elig = elig & ~ring
    if known is not None:
        elig = elig & ~known  # known pixels are masked by the scorer: no budget there
    if spec.get("transform") is not None:
        g, sc, fl = spec["transform"]
        s = apply_output_transform(s, g, sc, fl)
        s = np.nan_to_num(s, nan=0.0)
    if spec.get("regime_balanced") and regimes is not None:
        q = np.zeros(s.shape, np.float32)
        for z in np.unique(regimes[elig]):
            zm = elig & (regimes == z)
            q = np.maximum(q, R.top_mask(s, zm, spec["frac"]).astype(np.float32))
        q = R.apply({"select": "raw", "post": [p for p in rep.get("post", []) if p[0] == "known_ring"]}, q, ev, known=known)
        return np.where(ev, q, 0.0).astype(np.float32)
    q = R.apply(rep, s, ev, spec.get("frac"), known=known, eligible=elig, crest=crest)
    topo = spec.get("topology")
    if topo is not None:
        m = q > 0
        if topo[0] == "remove_small":
            m = remove_small(m, topo[1])
        elif topo[0] == "link":
            m = link_endpoints(m, topo[1]) & ev
        q = m.astype(np.float32)
    return q


def tournament(score_by_unit: dict, units, cfg, regimes: np.ndarray | None = None, log=print) -> dict:
    """Score every representation on every unit and cross-fit the choice per task.

    ``score_by_unit``: unit key -> full-grid score (OOF for geo units, the split's map for disc units)."""
    grid = representation_grid(cfg)
    rows = []
    for u in units:
        s = u.crop(score_by_unit[u.key])
        reg = u.crop(regimes) if regimes is not None else None
        crest = (
            R.crest_map(np.where(u.ev, np.nan_to_num(s), 0.0), u.ev)
            if any(g["rep"].get("select") == "crest" for g in grid)
            else None
        )
        for spec in grid:
            q = render(spec, s, u.ev, u.known, reg, u.scored_mask(), crest)
            c = u.components(q)
            rows.append(
                {
                    "task": u.task,
                    "zone": u.zone,
                    "key": spec["key"],
                    "name": spec["name"],
                    "frac": spec["frac"],
                    "ring_px": spec["ring_px"],
                    "tp_w": c.tp,
                    "fp_w": c.fp,
                    "fn_w": c.fn,
                    "pred_mass": c.pred_mass,
                    "pos_px": int((q > 0).sum()),
                }
            )
        u.release()
    table = pd.DataFrame(rows)
    results = {"table": table, "by_task": {}}
    for task, t in table.groupby("task"):
        per, summ = crossfit(t, ["key"])
        results["by_task"][task] = {"per_zone": per, "summary": summ}
        best = (
            t.groupby("key")[["tp_w", "fp_w", "fn_w"]]
            .sum()
            .pipe(lambda g: g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6))
        ).sort_values(ascending=False)
        results["by_task"][task]["insample_top10"] = best.head(10)
        log(
            f"[dti_opt] {task}: cross-fitted DTI {summ['dti_crossfit']:.4f} (in-sample {summ['dti_insample']:.4f}); "
            f"chosen on all zones: {summ['params_all']['key']}"
        )
    return results


def spec_by_key(cfg, key: str) -> dict:
    for s in representation_grid(cfg):
        if s["key"] == key:
            return s
    raise KeyError(key)


def family_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Best pooled DTI per representation family and task (descriptive, in-sample)."""
    g = table.groupby(["task", "name", "frac", "ring_px"], dropna=False)[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    return g.sort_values(["task", "dti"], ascending=[True, False]).groupby(["task", "name"]).head(1)


def mass_curves(score_by_unit: dict, units, fracs=np.geomspace(0.002, 0.3, 30)) -> pd.DataFrame:
    """DTI, TP, FP vs predicted area (binary top-a) pooled per task: the prediction-mass analysis on real labels."""
    rows = []
    for u in units:
        df = u.curve(u.crop(score_by_unit[u.key])).at_area(fracs)
        df["task"], df["zone"] = u.task, u.zone
        rows.append(df)
    t = pd.concat(rows)
    g = t.groupby(["task", "target_frac"])[["tp_w", "fp_w", "fn_w", "area_px"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    return g


def width_displacement_real(score_by_unit: dict, units, frac: float) -> pd.DataFrame:
    """Held-out effect of widening (dilation) and shifting the chosen binary mask (real labels)."""
    rows = []
    for u in units:
        s = u.crop(score_by_unit[u.key])
        base = R.top_mask(np.nan_to_num(s), u.scored_mask(), frac)
        for w in (0, 1, 2):
            m = ndimage.binary_dilation(base, structure=R._disk(w)) if w else base
            c = u.components(m.astype(float) * u.ev)
            rows.append(
                {"task": u.task, "zone": u.zone, "experiment": "dilate_px", "x": w, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn}
            )
        for sh in (1, 2, 3):
            m = np.roll(base, sh, axis=0)
            c = u.components(m.astype(float) * u.ev)
            rows.append(
                {"task": u.task, "zone": u.zone, "experiment": "shift_px", "x": sh, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn}
            )
        u.release()
    t = pd.DataFrame(rows)
    g = t.groupby(["task", "experiment", "x"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    return g


__all__ = [
    "synthetic_sensitivity",
    "representation_grid",
    "render",
    "tournament",
    "spec_by_key",
    "family_summary",
    "mass_curves",
    "width_displacement_real",
    "pooled_dti",
]
