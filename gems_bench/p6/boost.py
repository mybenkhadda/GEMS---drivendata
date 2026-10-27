"""P6 boost: prove, then extend, the W2 stacker gain under one matched protocol (docs/p6/P6_BOOST.md).

    bst = p6.boost(BOOST_CONFIG)
    bst.a_benchmark()      # A  matched ranking x representation x budget benchmark (nested)
    bst.b_geology()        # B  G02 attach audit (+ optional zone backfill) and S0-S3 structural ablation
    bst.c_robustness()     # C  weak-zone diagnostics + two predeclared robustness variants
    bst.d_hybrid()         # D  controlled union / connectivity-aware lineament selection at matched area
    bst.e_select()         # E  gate against the W2 champion, optional validated export, leaderboard comparison

One protocol for every candidate (``_eval_q``): the unit's evaluation mask, ``m9.elig(u, 0)`` as the area budget
domain, the M7 representations, ``m7.protocol.score_row`` official components and ``system_hits``. Fixed cells
(rep, frac[, blend]) are reported unselected; every quoted "nested" number is ``gems3.metric_opt.nested_select``
(leave-one-zone-out choice, scored on the held-out zone). No zone-specific rule, threshold or weight exists here.

Reuse, not re-modelling: CNN ranks come from ``gems3.official.unit_score`` (OOF / split discovery maps), W2 scores from
``stack/maps`` (read-only), rows and cross-fitting from ``p6.stack`` (identical sampled rows and weights for every
variant: only columns change). Every stage writes ``boost/<stage>/manifest.json`` with its fingerprint and reloads
instead of recomputing while the fingerprint is unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from . import stack as ST

BOOST_VERSION = "p6b.1"                    # enters every stage fingerprint: bump only to invalidate all stage caches
BOOST_CODE_REV = "p6b.1-r3"               # code revision checked by the notebook start-up cell (not fingerprinted)
#   r3 (P6.1): export through export_parity's single recipe contract, robust system_set, unit_sources / binary
DEFAULT_BOOST_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "p6" / "boost.yaml"
COMP = ["tp_w", "fp_w", "fn_w", "n_truth"]
STAGES = ("benchmark", "geology", "robustness", "hybrid", "selection")
STATUSES = ("PENDING", "BLOCKED", "RUNNING", "COMPLETE", "FAILED")
EXEC_SWITCHES = {"geology": ("backfill_missing", "backfill_max_units")}   # never part of a stage fingerprint
# Files a finished stage must have before a fingerprint match may reload it (``<stage>/<name>``).
STAGE_OUTPUTS = {
    "benchmark": ("grid.csv", "matched_comparison.csv", "per_zone.csv", "nested_selection.csv", "nested_per_zone.csv",
                  "ranking_gain.csv", "system_coverage.csv", "verdict.json"),
    "geology": ("g02_attach.json", "g02_audit.csv", "availability.csv", "row_consistency.csv", "ablation.csv",
                "decision.json"),
    "robustness": ("zone_ranking.csv", "feature_shift.csv", "known_distance.csv", "composition.csv",
                   "prediction_geometry.csv", "ablation.csv", "decision.json"),
    "hybrid": ("grid.csv", "matched_comparison.csv", "d3_geological_context.csv", "segment_selection.csv",
               "ablation.csv", "decision.json"),
    "selection": ("candidates.csv", "leaderboard_compare.csv", "decision.json"),
}


def missing_outputs(stage_dir: Path, stage: str) -> list[str]:
    """Required outputs of ``stage`` absent (or empty) under ``stage_dir``: a stage with any is never reloaded."""
    d = Path(stage_dir)
    return [n for n in STAGE_OUTPUTS.get(stage, ()) if not (d / n).exists() or (d / n).stat().st_size == 0]


def load_boost_config(config: dict | None = None, path: Path | None = None) -> dict:
    from .run import _merge

    with open(path or DEFAULT_BOOST_CONFIG, encoding="utf-8") as f:
        cfg = _merge(yaml.safe_load(f), config or {})
    if len(cfg["geology"]["variants"]) > 4 or len(cfg["robustness"]["variants"]) > 2:
        raise ValueError("boost is bounded: at most 4 geology variants and 2 robustness variants")
    return cfg


# =============================================================================== pure: scoring tables
def dti(tp, fp, fn):
    """Official DTI from summed components (same expression as ``nested_select``)."""
    return tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6)


def pooled(G: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pooled components, DTI, weighted precision / recall and the FP / FN shares of the DTI denominator."""
    g = G.groupby(by, dropna=False)[COMP].sum().reset_index()
    den = g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6
    g["dti"] = g.tp_w / den
    g["precision_w"] = g.tp_w / (g.tp_w + g.fp_w).clip(lower=1e-9)
    g["recall_w"] = g.tp_w / (g.tp_w + g.fn_w).clip(lower=1e-9)
    g["fp_share"], g["fn_share"] = 0.2 * g.fp_w / den, 0.8 * g.fn_w / den
    return g


def matched_comparison(G: pd.DataFrame, cell: list[str] | None = None) -> pd.DataFrame:
    """Every fixed cell, unselected: pooled metrics, worst / best zone, area, line length, fragmentation, systems."""
    cell = cell or ["task", "candidate", "blend", "rep", "frac"]
    G = G.assign(blend=G.blend.fillna(-1.0)) if "blend" in G else G
    P = pooled(G, cell)
    z = G.assign(dti_zone=dti(G.tp_w, G.fp_w, G.fn_w)).groupby(cell, dropna=False)
    agg = {"worst_zone_dti": ("dti_zone", "min"), "best_zone_dti": ("dti_zone", "max"), "n_zones": ("zone", "nunique"),
           "area_px": ("area_px", "sum"), "area_frac_eligible": ("area_frac_eligible", "mean")}
    for c, how in (("length_km", "sum"), ("n_segments", "sum"), ("short_share", "mean"), ("n_recovered", "sum"),
                   ("n_systems", "sum")):
        if c in G:
            agg[c] = (c, how)
    out = P.merge(z.agg(**agg).reset_index(), on=cell)
    return out.assign(blend=out.blend.where(out.blend >= 0)) if "blend" in out else out


def nested(G: pd.DataFrame, params: list[str]) -> tuple[pd.DataFrame, dict]:
    """``nested_select`` + the held-out choice's systems and area (looked up on the chosen row)."""
    from ..gems3.metric_opt import nested_select

    if not len(G):
        return pd.DataFrame(), {"dti_crossfit": np.nan, "fold_min": np.nan}
    per, summ = nested_select(G, params)
    if len(per):
        keys = ["zone"] + params
        extra = [c for c in ("systems", "n_recovered", "area_px", "area_frac_eligible", "length_km", "candidate")
                 if c in G and c not in keys]
        per = _merge_on_str(per, G.drop_duplicates(keys)[keys + extra], keys)
    return per, summ


def _merge_on_str(per: pd.DataFrame, look: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    a, b = per.copy(), look.copy()
    for k in keys:
        a[f"_{k}"], b[f"_{k}"] = a[k].astype(str), b[k].astype(str)
    b = b.drop(columns=keys)
    out = a.merge(b, how="left", on=[f"_{k}" for k in keys])
    return out.drop(columns=[f"_{k}" for k in keys])


def system_set(s) -> set[int]:
    """System ids of a ``systems`` cell: ";"-joined integer ids of ``system_hits`` ("3;7"). Empty / missing cells
    give an empty set; a CSV reload that turned a single id into a float ("3.0") is accepted. Any other token
    (e.g. "0c", "3.5") is not a system id and raises ``ValueError`` (never coerced, never dropped)."""
    out = set()
    for x in str(s).split(";"):
        t = x.strip()
        if t in ("", "nan", "None", "<NA>"):
            continue
        try:
            v = float(t)
        except ValueError:
            raise ValueError(f"systems cell {s!r}: token {t!r} is not an integer system id") from None
        if not np.isfinite(v) or v != int(v):
            raise ValueError(f"systems cell {s!r}: token {t!r} is not an integer system id")
        out.add(int(v))
    return out


def systems_vs(per: pd.DataFrame, ref: dict) -> dict:
    """Recovered systems of nested per-zone choices against a reference ``{zone: set}`` (the fallback)."""
    out = {"recovered": 0, "reference": 0, "gained": 0, "lost": 0, "gained_ids": "", "lost_ids": ""}
    g, lo = [], []
    for r in per.itertuples() if len(per) and "systems" in per else []:
        s, f = system_set(r.systems), ref.get(int(r.zone), set())
        out["recovered"] += len(s)
        out["reference"] += len(f)
        g += [f"{int(r.zone)}:{i}" for i in sorted(s - f)]
        lo += [f"{int(r.zone)}:{i}" for i in sorted(f - s)]
    out.update(gained=len(g), lost=len(lo), gained_ids=";".join(g), lost_ids=";".join(lo))
    return out


def line_stats(q: np.ndarray, px_km: float = 0.1, short_px: int = 5) -> dict:
    """Area-as-length (pixel count x pixel size), 8-connected segments, mean segment length, share of short ones."""
    m = np.asarray(q) > 0
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    if not n:
        return {"length_km": 0.0, "n_segments": 0, "mean_segment_km": 0.0, "short_share": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    return {"length_km": float(m.sum() * px_km), "n_segments": int(n), "mean_segment_km": float(sizes.mean() * px_km),
            "short_share": float((sizes < short_px).mean())}


def ranking_gain(S: pd.DataFrame, M: pd.DataFrame, cfg: dict, prim: str) -> tuple[pd.DataFrame, dict]:
    """Ladder fallback -> fusion -> W2 -> hybrid on nested numbers, plus matched-cell ranking deltas.

    ``S``: nested summaries (task, space, dti_crossfit, fold_min, per_zone dict zone -> dti). ``M``: matched cells."""
    def v(task, space, col="dti_crossfit"):
        r = S[(S.task == task) & (S.space == space)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def zones(task, space):
        r = S[(S.task == task) & (S.space == space)]
        return dict(r.per_zone.iloc[0]) if len(r) else {}

    steps = [("published_to_matched_grid", "fallback_published", "fallback_fusion|crest_1px"),
             ("representation", "fallback_fusion|crest_1px", "fallback_fusion|rep+frac"),
             ("ranking_at_crest", "fallback_fusion|crest_1px", "w2|crest_1px"),
             ("ranking_at_skeleton", "fallback_fusion|skeleton_1px", "w2|skeleton_1px"),
             ("ranking", "fallback_fusion|rep+frac", "w2|rep+frac"),
             ("hybridization", "w2|rep+frac", "w2_space|blend+rep+frac"),
             ("total", "fallback_published", "w2_space|blend+rep+frac")]
    rows = []
    for task in sorted(S.task.unique()):
        for name, a, b in steps:
            za, zb = zones(task, a), zones(task, b)
            dz = {z: zb[z] - za[z] for z in za if z in zb}
            rows.append({"task": task, "component": name, "from": a, "to": b, "from_dti": v(task, a),
                         "to_dti": v(task, b), "gain": v(task, b) - v(task, a),
                         "worst_gain": v(task, b, "fold_min") - v(task, a, "fold_min"),
                         "zones_improved": int(sum(d > 0 for d in dz.values())), "n_zones": len(dz),
                         "zone_gains": json.dumps({str(k): round(float(x), 6) for k, x in dz.items()})})
        m = M[M.task == task]
        a = m[m.candidate == "fallback_fusion"].set_index(["rep", "frac"]).dti
        b = m[m.candidate == "w2"].set_index(["rep", "frac"]).dti
        d = (b - a).dropna()
        rows.append({"task": task, "component": "matched_cells_w2_minus_fusion", "from": "fallback_fusion",
                     "to": "w2", "gain": float(d.mean()) if len(d) else np.nan,
                     "worst_gain": float(d.min()) if len(d) else np.nan, "zones_improved": int((d > 0).sum()),
                     "n_zones": int(len(d)), "zone_gains": json.dumps({f"{r}@{f:g}": round(float(x), 6)
                                                                       for (r, f), x in d.items()})})
        for nm, c in (cfg.get("named_cells") or {}).items():
            fa = a.get((c["rep"], float(c["frac"])), np.nan)
            fb = b.get((c["rep"], float(c["frac"])), np.nan)
            rows.append({"task": task, "component": f"cell:{nm}", "from": "fallback_fusion", "to": "w2",
                         "from_dti": fa, "to_dti": fb, "gain": fb - fa,
                         "zone_gains": json.dumps({"rep": c["rep"], "frac": float(c["frac"])})})
    R = pd.DataFrame(rows)
    g = cfg["genuine"]
    rk = R[(R.task == prim) & (R.component == "ranking")]
    mc = R[(R.task == prim) & (R.component == "matched_cells_w2_minus_fusion")]
    comp = R[(R.task == prim) & R.component.isin(["representation", "ranking", "hybridization",
                                                     "published_to_matched_grid"])]
    gain = float(rk.gain.iloc[0]) if len(rk) else np.nan
    win = float(mc.zones_improved.iloc[0] / max(mc.n_zones.iloc[0], 1)) if len(mc) else np.nan
    zi = int(rk.zones_improved.iloc[0]) if len(rk) else 0
    verdict = {"task": prim, "ranking_gain": gain, "zones_improved": zi, "cell_win_share": win,
               "largest_component": str(comp.loc[comp.gain.idxmax(), "component"]) if len(comp.dropna(
                   subset=["gain"])) else None,
               "components": {r.component: r.gain for r in comp.itertuples()},
               "w2_gain_is_ranking": bool(np.isfinite(gain) and gain >= float(g["min_ranking_gain"])
                                          and zi >= int(g["min_zones_improved"])
                                          and win >= float(g["min_cell_win_share"])),
               "rule": g}
    return R, verdict


# =============================================================================== pure: gates
def accept(c: dict, base: dict, rules: dict, geo_floor: float) -> tuple[bool, list[str]]:
    """Variant vs its base under the identical protocol: pooled gain, worst-zone harm, geographic floor."""
    why = []
    d, b = c.get("disc_dti"), base.get("disc_dti")
    if c.get("status") == "BLOCKED":
        return False, ["BLOCKED: " + str(c.get("reason", ""))]
    if d is None or not np.isfinite(d) or b is None or not np.isfinite(b):
        return False, ["no discovery DTI"]
    if d < b + float(rules.get("min_gain", 0.0)) or (float(rules.get("min_gain", 0.0)) == 0.0 and d <= b):
        why.append(f"disc {d:.4f} not above base {b:.4f} + {float(rules.get('min_gain', 0.0)):.3f}")
    wb = float(base.get("disc_worst", -np.inf)) - float(rules.get("max_worst_drop", 0.0))
    w = c.get("disc_worst")
    if w is None or not np.isfinite(w) or w < wb - 1e-12:
        why.append(f"worst zone {w} < {wb:.4f}")
    g = c.get("geo_dti")
    if g is None or not np.isfinite(g) or g < geo_floor:
        why.append(f"geo {g} < floor {geo_floor:.4f}")
    if c.get("fp_increase") is not None and "max_fp_increase" in rules and \
            c["fp_increase"] > float(rules["max_fp_increase"]):
        why.append(f"false-positive weight +{c['fp_increase']:.1%} > {float(rules['max_fp_increase']):.0%}")
    return not why, why


def boost_gate(c: dict, champ: dict, rules: dict) -> tuple[bool, list[str], bool]:
    """Replacement gate of the brief: disc > champion, worst >= champion worst, geo >= floor, provenance and
    validation clean. Third value: worth a leaderboard slot (gain >= ``slot_min_gain`` or corrects a flaw)."""
    why = []
    d, w, g = (c.get(k) for k in ("disc_dti", "disc_worst", "geo_dti"))
    if d is None or not np.isfinite(d):
        return False, ["no discovery DTI"], False
    need = max(float(champ.get("disc_dti", -np.inf)), float(champ.get("disc_dti_matched", -np.inf)))
    if not d > need:
        why.append(f"disc {d:.4f} <= champion {need:.4f}")
    if w is None or not np.isfinite(w) or w < float(champ.get("disc_worst", -np.inf)) - 1e-12:
        why.append(f"worst zone {w} < champion {champ.get('disc_worst')}")
    if g is None or not np.isfinite(g) or g < float(rules["geo_floor"]):
        why.append(f"geo {g} < {float(rules['geo_floor']):.4f}")
    if not c.get("provenance_ok", False):
        why.append("provenance not verified")
    if c.get("validation_failed"):
        why.append("submission validation failed")
    ok = not why
    slot = ok and (d - need >= float(rules["slot_min_gain"]) - 1e-12 or bool(c.get("corrects_validation_flaw")))
    return ok, why, bool(slot)


def variant_status(present: dict, need: list[str]) -> tuple[str, list[str]]:
    """``present``: unit key -> set of feature names available. RUN only if every unit has every needed feature."""
    miss = sorted(f"{u}:{k}" for u, have in present.items() for k in need if k not in have)
    return ("RUN" if not miss else "BLOCKED"), miss


def group_dropout(X: np.ndarray, cols: list[int], p: float, rng) -> np.ndarray:
    """Copy of ``X`` with the ``cols`` group set missing (NaN) together on a random ``p`` share of rows."""
    X = np.array(X, np.float32, copy=True)
    if cols and p > 0:
        drop = rng.random(len(X)) < p
        X[np.ix_(drop, cols)] = np.nan
    return X


# =============================================================================== pure: zone diagnostics
def mean_truth_rank(score: np.ndarray, truth: np.ndarray, ev: np.ndarray) -> float:
    """Mean per-unit rank of the truth pixels (0.5 = random ranking; AUC-like, label used for diagnosis only)."""
    from ..gems3.representation import rank01

    t = np.asarray(truth, bool) & ev
    if not t.any():
        return np.nan
    return float(rank01(np.nan_to_num(np.asarray(score, np.float32), nan=0.0), ev)[t].mean())


def axial_orientation(mask: np.ndarray, win: int = 7) -> np.ndarray:
    """Local axial orientation (rad, [0, pi), from +column toward +row) of line pixels from windowed second moments;
    NaN off the mask."""
    m = np.asarray(mask, bool)
    out = np.full(m.shape, np.nan, np.float32)
    if not m.any():
        return out
    f = m.astype(np.float64)
    rr, cc = np.indices(m.shape, dtype=np.float64)
    rr -= rr.mean()
    cc -= cc.mean()

    def u(a):
        return ndimage.uniform_filter(a, win, mode="constant")

    n = np.maximum(u(f), 1e-12)
    mr, mc = u(f * rr) / n, u(f * cc) / n
    srr, scc, src = u(f * rr * rr) / n - mr ** 2, u(f * cc * cc) / n - mc ** 2, u(f * rr * cc) / n - mr * mc
    th = 0.5 * np.arctan2(2.0 * src, scc - srr)
    out[m] = np.mod(th[m], np.pi)
    return out


def orientation_hist(theta: np.ndarray, n_bins: int = 12) -> np.ndarray:
    v = np.asarray(theta)[np.isfinite(theta)]
    h, _ = np.histogram(v, bins=n_bins, range=(0.0, np.pi))
    return h / max(h.sum(), 1)


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance in [0, 1] between two orientation histograms."""
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def distance_km(src: np.ndarray, dst: np.ndarray, px_km: float = 0.1) -> np.ndarray:
    """Distance (km) from each ``src`` pixel to the nearest ``dst`` pixel (inf when ``dst`` is empty)."""
    s, d = np.asarray(src, bool), np.asarray(dst, bool)
    if not d.any():
        return np.full(int(s.sum()), np.inf)
    return ndimage.distance_transform_edt(~d)[s] * px_km


def quantiles(v: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95), prefix: str = "q") -> dict:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return {f"{prefix}{int(q * 100):02d}": (float(np.quantile(v, q)) if v.size else np.nan) for q in qs}


# =============================================================================== pure: lineament hybrids
def branches(mask: np.ndarray) -> np.ndarray:
    """Labelled branches of a 1-px line map (junction neighbourhoods removed, every branch kept)."""
    from ..p3.lineaments import split_branches

    m = np.asarray(mask, bool)
    if not m.any():
        return np.zeros(m.shape, np.int32)
    lab, _, _ = split_branches(m.astype(np.uint8), 1)
    return lab.astype(np.int32)


def seg_table(lab: np.ndarray, score: np.ndarray, layers: dict, line_mask: np.ndarray, ep_radius: int = 5) -> pd.DataFrame:
    """Per labelled segment: size, mean score, linearity (1 - minor / major moment), mean of each layer and the
    endpoint continuation support (best score of non-line pixels within ``ep_radius`` of the segment's tips)."""
    n = int(lab.max())
    if n == 0:
        return pd.DataFrame(columns=["label", "n_px", "mean_score", "linearity", "endpoint_support"])
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    cnt = np.bincount(lb, minlength=n + 1)[1:].astype(np.float64)

    def mean(v):
        return np.bincount(lb, weights=np.asarray(v, np.float64), minlength=n + 1)[1:] / np.maximum(cnt, 1)

    r, c = rr.astype(np.float64), cc.astype(np.float64)
    mr, mc = mean(r), mean(c)
    vrr, vcc, vrc = mean(r * r) - mr ** 2, mean(c * c) - mc ** 2, mean(r * c) - mr * mc
    tr, det = vrr + vcc, vrr * vcc - vrc ** 2
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, np.maximum(tr / 2.0 - disc, 0.0)
    lin = np.where(cnt >= 3, 1.0 - l2 / np.maximum(l1, 1e-9), 1.0)
    sc = np.nan_to_num(np.asarray(score, np.float64), nan=0.0)
    T = pd.DataFrame({"label": np.arange(1, n + 1), "n_px": cnt.astype(int), "mean_score": mean(sc[rr, cc]),
                      "linearity": lin})
    for k, a in layers.items():
        if a is not None:
            T[f"mean_{k}"] = mean(np.nan_to_num(np.asarray(a, np.float64), nan=0.0)[rr, cc])
    ep = ST.endpoints(lab > 0)
    sup = ndimage.maximum_filter(np.where(np.asarray(line_mask, bool), -np.inf, sc), size=2 * int(ep_radius) + 1)
    er, ec = np.nonzero(ep)
    if er.size:
        v = np.where(np.isfinite(sup[er, ec]), sup[er, ec], 0.0)
        k = np.bincount(lab[er, ec], minlength=n + 1)[1:]
        T["endpoint_support"] = np.bincount(lab[er, ec], weights=v, minlength=n + 1)[1:] / np.maximum(k, 1)
    else:
        T["endpoint_support"] = 0.0
    return T


def connectivity_z(T: pd.DataFrame, weights: dict, short_px: int, short_penalty: float) -> np.ndarray:
    """Predeclared segment score: weighted mean of pool ranks (score, length, linearity, endpoint support,
    geology when present); segments shorter than ``short_px`` are down-weighted, never removed."""
    cols = {"score": "mean_score", "length": "n_px", "linearity": "linearity", "endpoint": "endpoint_support",
            "geology": "mean_geology"}
    z, ws = np.zeros(len(T)), 0.0
    for k, w in weights.items():
        c = cols.get(k)
        if c in T and float(w) > 0:
            z += float(w) * T[c].rank(pct=True).to_numpy()
            ws += float(w)
    z = z / max(ws, 1e-9)
    return z * np.where(T.n_px.to_numpy() < int(short_px), float(short_penalty), 1.0)


def fill_budget(lab: np.ndarray, order, k: int, base: np.ndarray | None = None) -> np.ndarray:
    """``base`` plus whole segments in ``order`` while the total stays <= ``k`` pixels (skips, never splits)."""
    n = int(lab.max())
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out = np.zeros(lab.shape, bool) if base is None else np.asarray(base, bool).copy()
    cum, keep = int(out.sum()), np.zeros(n + 1, bool)
    for lb in order:
        s = int(sizes[int(lb)])
        if lb and s and cum + s <= k:
            keep[int(lb)] = True
            cum += s
    return out | keep[lab]


def controlled_union(base_part, base_full, fb, fb_score, base_score, k: int, tol_px: int = 3) -> np.ndarray:
    """D1: the base map's strongest share (``base_part``) + fallback-only lines (no pixel within ``tol_px`` of the
    full-budget base map) by mean independent evidence, then the base map's remaining lines by mean score, while the
    area stays <= ``k`` (the base map's own area at the same cell). No label enters any ordering."""
    from ..postprocess.topology import _disk

    bf = np.asarray(base_full, bool)
    bp = np.asarray(base_part, bool) & bf
    near = ndimage.binary_dilation(bf, structure=_disk(int(tol_px))) if tol_px > 0 else bf
    s8 = np.ones((3, 3), bool)
    lf, nf = ndimage.label(np.asarray(fb, bool) & ~near, structure=s8)
    lb, nb = ndimage.label(bf & ~bp, structure=s8)
    of = np.arange(1, nf + 1)[np.argsort(-np.asarray(ndimage.mean(np.nan_to_num(fb_score), lf, np.arange(1, nf + 1))),
                                         kind="stable")] if nf else np.zeros(0, int)
    ob = np.arange(1, nb + 1)[np.argsort(-np.asarray(ndimage.mean(np.nan_to_num(base_score), lb, np.arange(1, nb + 1))),
                                         kind="stable")] if nb else np.zeros(0, int)
    lab = lf.astype(np.int64)
    lab[lb > 0] = lb[lb > 0] + nf
    return fill_budget(lab, list(of) + list(ob + nf), int(k), base=bp)


def connectivity_select(pool, score, k: int, geology=None, *, weights: dict, short_px: int = 5,
                        short_penalty: float = 0.5, ep_radius: int = 5) -> tuple[np.ndarray, pd.DataFrame]:
    """D2: re-select ``k`` pixels of an over-generated line pool by segment coherence (``connectivity_z``); junction
    pixels adjacent to kept branches are restored when the budget allows. Returns (mask, segment table)."""
    pool = np.asarray(pool, bool)
    lab = branches(pool)
    T = seg_table(lab, score, {"geology": geology}, pool, ep_radius)
    if not len(T):
        return np.zeros(pool.shape, bool), T
    w = dict(weights) if geology is not None else {a: b for a, b in weights.items() if a != "geology"}
    T["z"] = connectivity_z(T, w, short_px, short_penalty)
    out = fill_budget(lab, T.label.to_numpy()[np.argsort(-T.z.to_numpy(), kind="stable")], int(k))
    j = pool & (lab == 0) & ndimage.binary_dilation(out, structure=np.ones((3, 3), bool))
    if out.sum() + j.sum() <= k:
        out |= j
    T["kept"] = np.bincount(lab[out & (lab > 0)], minlength=int(lab.max()) + 1)[1:] > 0
    return out, T


def context_table(T: pd.DataFrame, precision: np.ndarray, ctx_col: str) -> pd.DataFrame:
    """D3: precision (share of segment pixels near held-out truth; diagnosis only) of low / high confidence segments
    split by a label-free geological-context column."""
    t = T.assign(precision=precision)
    t["confidence"] = np.where(t.mean_score >= t.mean_score.median(), "high", "low")
    t["context"] = np.where(t[ctx_col] >= t[ctx_col].median(), "high", "low")
    t["_hit"] = t.precision * t.n_px
    g = t.groupby(["confidence", "context"]).agg(n_segments=("n_px", "size"), px=("n_px", "sum"),
                                                 _hit=("_hit", "sum")).reset_index()
    g["precision_px"] = g._hit / g.px.clip(lower=1)
    return g.drop(columns="_hit").assign(context_layer=ctx_col)


# =============================================================================== Colab side
class Boost:
    """Stages A-E on the P6 workspace (``<p6 output_dir>/boost``); W0-W2 outputs are read, never written."""

    def __init__(self, p6, config: dict | None = None, *, config_path: Path | None = None):
        self.p6 = p6
        self.cfg = load_boost_config(config, config_path)
        self.root = p6.ws.root / self.cfg["subdir"]
        self._raw, self._cnn, self._el, self._fb = {}, {}, {}, {}
        self._sel = self._T0 = self._w2in = None

    # ------------------------------------------------------------------------------- plumbing
    def say(self, msg: str) -> None:
        print(f"[boost] {msg}")

    def sub(self, stage: str) -> str:
        return f"{self.cfg['subdir']}/{stage}"

    def save_table(self, df: pd.DataFrame, name: str, stage: str):
        return self.p6.ws.save_table(df, name, self.sub(stage))

    def save_json(self, obj, stage: str, name: str):
        return self.p6.ws.save_json(json.loads(json.dumps(obj, default=_js)), f"{self.sub(stage)}/{name}.json")

    def load_table(self, name: str, stage: str) -> pd.DataFrame:
        return self.p6.ws.load_table(name, self.sub(stage))

    def read_json(self, stage: str, name: str) -> dict | None:
        """``<stage>/<name>.json`` or None; a corrupt (partially written) file is reported and treated as absent."""
        p = self.root / stage / f"{name}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.say(f"WARNING {p} unreadable ({type(e).__name__}): treated as absent")
            return None

    def fp(self, *parts) -> str:
        return self.p6.fp(BOOST_VERSION, *parts)

    def protocol_keys(self) -> dict:
        """{unit: M7 protocol key}: evaluation mask, truth, exclusions, budget domain and template of every unit."""
        return {u.key: self.p6.m9.pkey(u) for u in self.units()}

    def _done(self, stage: str, fp: str) -> bool:
        """Reload only when the manifest is readable and finished under ``fp``, every required output exists, and
        (manifests that recorded them) the units' protocol keys are unchanged."""
        m = self.read_json(stage, "manifest") or {}
        if m.get("fingerprint") != fp or m.get("status") != "done":
            return False
        miss = missing_outputs(self.root / stage, stage)
        if miss:
            self.say(f"{stage}: fingerprint matches but outputs {miss} are missing -> recomputed")
            return False
        old = (m.get("inputs") or {}).get("protocol_keys")
        if old and old != self.protocol_keys():
            self.say(f"{stage}: evaluation protocol keys changed (masks / truth / budget domain) -> recomputed")
            return False
        self.say(f"{stage}: fingerprint {fp[:10]} unchanged -> reloaded (nothing recomputed)")
        return True

    def set_status(self, stage: str, status: str, **info) -> None:
        """``<stage>/status.json``: PENDING / BLOCKED / RUNNING / COMPLETE / FAILED (+ reason, fingerprint)."""
        if status not in STATUSES:
            raise ValueError(f"status {status!r} not in {STATUSES}")
        self.save_json({"stage": stage, "status": status, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "code_rev": BOOST_CODE_REV, **info}, stage, "status")

    def _begin(self, stage: str, fp: str) -> float:
        self.set_status(stage, "RUNNING", fingerprint=fp)
        return time.perf_counter()

    def _finish(self, stage: str, fp: str, t0: float, inputs: dict, decisions: dict, *, status: str = "COMPLETE",
                reason: str = "") -> None:
        from .._version import provenance

        inputs = {**inputs, "protocol_keys": self.protocol_keys()}
        self.save_json({"stage": stage, "status": "done", "stage_status": status, "fingerprint": fp,
                        "boost_version": BOOST_VERSION, "code_rev": BOOST_CODE_REV, "config": self.cfg,
                        "p6_config_fp": self.p6.fp(self.p6.cfg), "inputs": inputs,
                        "decisions": decisions, "code": provenance(), "seconds": round(time.perf_counter() - t0, 1),
                        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "reload": f"p6.boost(BOOST_CONFIG).{_METHOD[stage]}() reloads while the fingerprint holds"},
                       stage, "manifest")
        self.set_status(stage, status, fingerprint=fp, reason=reason, source_commit=provenance()["git_commit"])

    def section(self, stage: str, fn):
        """Budgeted step; any exception leaves ``status.json`` = FAILED (BLOCKED when the CPU cap is spent)."""
        from .run import BudgetExceeded

        try:
            return self.p6.section("wb", "cpu", fn, what=f"boost_{stage}")
        except BudgetExceeded as e:
            self.set_status(stage, "BLOCKED", reason=f"BudgetExceeded: {e}")
            raise
        except Exception as e:
            self.set_status(stage, "FAILED", reason=f"{type(e).__name__}: {e}")
            raise

    def preflight(self) -> dict:
        """Dependency / contract / provenance audit without heavy computation (``boost_audit.run_preflight``)."""
        from . import boost_audit

        return boost_audit.run_preflight(self)

    def report(self) -> dict:
        """Final decision-support report from saved outputs only (``boost_audit.build_report``)."""
        from . import boost_audit

        return boost_audit.build_report(self)

    def stage(self) -> dict:
        """Bring W2 / boost / probe / ledger outputs back from Drive (a fresh runtime has none of them locally),
        then re-read the compute ledger so a later push cannot overwrite the Drive copy with an empty one."""
        from .run import Budget

        out = self.p6.cfg["output_dir"]
        self.p6.m9.stage_in([f"{out}/{s}" for s in ("stack", self.cfg["subdir"], "probes", "submissions",
                                                    "manifests", "selection")])
        self.p6.budget = Budget(self.p6.ws.root / "manifests" / "compute_ledger.csv", self.p6.cfg["budget"])
        st = self.p6.ws.root / "stack"
        need = {"stack/candidate.json": (st / "candidate.json").exists(),
                "stack/features/rows.parquet": (st / "features" / "rows.parquet").exists(),
                "stack/maps/*.npy": len(list((st / "maps").glob("*.npy"))) if (st / "maps").exists() else 0}
        self.say(f"W2 inputs: {need}")
        return need

    # ------------------------------------------------------------------------------- shared inputs
    @property
    def sel(self) -> dict:
        if self._sel is None:
            self._sel = self.p6.selection()
        return self._sel

    @property
    def prim(self) -> str:
        return self.p6.prim

    def units(self) -> list:
        return [u for t in self.cfg["tasks"] for u in self.p6.m9.m7.units_of(t)]

    def disc_units(self) -> list:
        return [u for u in self.units() if u.task == self.prim]

    def elig(self, u) -> np.ndarray:
        if u.key not in self._el:
            self._el[u.key] = np.asarray(self.p6.m9.elig(u, 0), bool)
        return self._el[u.key]

    def cnn(self, u) -> tuple[np.ndarray, np.ndarray]:
        """Honest B0 / B1 maps of the unit (OOF on geo folds, the split's discovery map on disc units)."""
        from ..gems3.official import unit_score

        if u.key not in self._cnn:
            ms = list(self.p6.m9.models)[:2]
            a = [unit_score(self.p6.ctx, m, u) for m in ms]
            if any(x is None for x in a):
                raise FileNotFoundError(f"{u.key}: honest CNN map missing ({ms})")
            self._cnn[u.key] = tuple(np.asarray(x, np.float32) for x in a)
        return self._cnn[u.key]

    def champion_w(self, u) -> float:
        return float(ST._champion_zone(self.sel, u.task, self.p6.cfg["probes"]["champion"], u.zone).get("w", 0.5))

    def fusion(self, u) -> np.ndarray:
        from ..gems3 import representation as R

        b0, b1 = self.cnn(u)
        return R.fuse(R.rank01(b0, u.ev), R.rank01(b1, u.ev), self.champion_w(u))

    def w2_map(self, u) -> np.ndarray:
        p = self.p6.ws.root / "stack" / "maps" / f"{u.task}_{u.zone}.npy"
        if not p.exists():
            raise FileNotFoundError(f"W2 held-out map {p} missing: run section 3 (W2) first")
        a = np.load(p)
        if a.shape != u.ev.shape:
            raise ValueError(f"W2 map {p.name} shape {a.shape} != unit crop {u.ev.shape}")
        return a

    def w2_inputs(self) -> dict:
        """Provenance of the frozen W2 inputs (candidate.json + sha256 of every held-out map + rows.parquet)."""
        if self._w2in is None:
            from ..artifacts.atomic import sha256_file

            st = self.p6.ws.root / "stack"
            cand = json.loads((st / "candidate.json").read_text(encoding="utf-8")) if (st / "candidate.json").exists() \
                else {}
            maps = {p.name: sha256_file(p) for p in sorted((st / "maps").glob("*.npy"))}
            rows = st / "features" / "rows.parquet"
            self._w2in = {"candidate": {k: cand.get(k) for k in ("features", "params_all", "disc_dti", "disc_worst",
                                                                  "geo_dti", "submission_file")},
                          "maps_sha256": maps, "rows_sha256": sha256_file(rows) if rows.exists() else None}
        return self._w2in

    def champion(self) -> dict:
        c = dict(self.cfg["champion"]["recorded"])
        got = (self.w2_inputs().get("candidate") or {})
        if got.get("disc_dti") is not None:
            c.update({k: float(got[k]) for k in ("disc_dti", "disc_worst", "geo_dti") if got.get(k) is not None})
            c["source"] = "stack/candidate.json"
        else:
            c["source"] = "recorded (configs/p6/boost.yaml)"
        return c

    @staticmethod
    def stack_rank(u, m: np.ndarray) -> np.ndarray:
        from ..gems3 import representation as R

        return R.rank01(np.nan_to_num(m, nan=0.0), u.ev)

    # ------------------------------------------------------------------------------- one protocol
    def _eval_q(self, u, q: np.ndarray, **tags) -> dict:
        from ..m7.protocol import score_row, system_hits

        gc = self.cfg["grid"]
        el = self.elig(u)
        q = np.where(u.ev, np.nan_to_num(q, nan=0.0), 0.0).astype(np.float32)
        row = score_row(u, q, el, self.p6.m9.pkey(u), **tags)
        row["dti"] = float(dti(row["tp_w"], row["fp_w"], row["fn_w"]))
        row.update(line_stats(q, float(gc["pixel_km"]), int(gc["short_segment_px"])))
        self._last_hits = pd.DataFrame()
        if u.is_disc:
            hs = system_hits(self.p6.ctx, u, q, self.p6.m9.sh)
            self._last_hits = hs                                 # per-system support of this map (X2 inventory)
            rec = sorted(int(x) for x in hs.system_id[hs.recovered]) if len(hs) else []
            row.update(n_systems=int(len(hs)), n_recovered=len(rec), systems=";".join(map(str, rec)))
        return row

    def binary(self, u, s: np.ndarray, rep: str, frac: float, crest=None) -> np.ndarray:
        """Boolean map of one (rep, frac) cell of a ranking score under the shared protocol (``_apply``)."""
        return np.asarray(self._apply(u, np.asarray(s, np.float32), rep, frac, crest)) > 0

    def unit_sources(self, u) -> dict:
        """Label-free rank maps of one unit, exactly as stage A builds them: fallback fusion (champion per-zone w),
        CNN B0 / B1 ranks and the frozen W2 held-out map rank."""
        from ..gems3 import representation as R

        b0, b1 = self.cnn(u)
        return {"fallback_fusion": self.fusion(u), "cnn_b0": R.rank01(b0, u.ev), "cnn_b1": R.rank01(b1, u.ev),
                "w2": self.stack_rank(u, self.w2_map(u))}

    def _apply(self, u, s: np.ndarray, rep: str, frac: float, crest=None) -> np.ndarray:
        from ..gems3 import representation as R

        sh = self.p6.m9.sh
        return R.apply(sh["baseline"]["representations"][rep], s, u.ev, float(frac), known=u.known, crest=crest,
                       sigma_px=float(sh["evaluation"]["ridge_sigma_px"]), eligible=self.elig(u))

    def _crest(self, u, s):
        from ..gems3 import representation as R

        return R.crest_map(s, u.ev, float(self.p6.m9.sh["evaluation"]["ridge_sigma_px"]))

    def eval_score(self, u, s: np.ndarray, **tags) -> list[dict]:
        """Every matched (rep, frac) cell of one ranking score on one unit."""
        gc = self.cfg["grid"]
        s = np.asarray(s, np.float32)
        crest = self._crest(u, s)
        return [self._eval_q(u, self._apply(u, s, rn, f, crest), rep=rn, frac=float(f), **tags)
                for rn in gc["reps"] for f in gc["area_fracs"]]

    def eval_variant(self, u, smap: np.ndarray, name: str) -> list[dict]:
        F, S = self.fusion(u), self.stack_rank(u, smap)
        rows = []
        for lam in self.cfg["grid"]["blends"]:
            rows += self.eval_score(u, ((1.0 - float(lam)) * F + float(lam) * S).astype(np.float32), candidate=name,
                                    blend=float(lam))
        return rows

    def fallback_systems(self) -> dict:
        """{disc zone: recovered system ids} of the published fallback's held-out map."""
        from ..m7.baseline import heldout_map
        from ..m7.protocol import system_hits

        if not self._fb:
            champ = self.p6.cfg["probes"]["champion"]
            for u in self.disc_units():
                qf = heldout_map(self.p6.view, u.task, champ, u.zone)
                hs = system_hits(self.p6.ctx, u, qf, self.p6.m9.sh) if qf is not None else pd.DataFrame()
                self._fb[int(u.zone)] = set(int(x) for x in hs.system_id[hs.recovered]) if len(hs) else set()
        return self._fb

    def summarize(self, G: pd.DataFrame, name: str, params: list[str],
                  space: str | None = None) -> tuple[dict, pd.DataFrame]:
        """Nested numbers of one candidate (or space) on both tasks + systems against the fallback."""
        out = {"candidate": name, "space": space or f"{name}|{'+'.join(params)}"}
        pers = []
        for task in self.cfg["tasks"]:
            g = G[G.task == task]
            per, s = nested(g, params)
            pre = "disc" if task == self.prim else "geo"
            out.update({f"{pre}_dti": s.get("dti_crossfit"), f"{pre}_worst": s.get("fold_min"),
                        f"{pre}_insample": s.get("dti_insample"), f"{pre}_optimism": s.get("optimism"),
                        f"{pre}_params_all": s.get("params_all")})
            if len(per):
                out[f"{pre}_per_zone"] = {int(r.zone): float(r.dti_heldout) for r in per.itertuples()}
                pers.append(per.assign(task=task, space=out["space"]))
        P = pd.concat(pers, ignore_index=True) if pers else pd.DataFrame()
        if len(P):
            out.update({f"systems_{k}": v for k, v in
                        systems_vs(P[P.task == self.prim], self.fallback_systems()).items()})
        out["params_all"] = {self.prim: out.get("disc_params_all"), "geo": out.get("geo_params_all")}
        return out, P

    # =========================================================================== A
    def a_benchmark(self) -> dict:
        return self.section("benchmark", self._a)

    def _a(self) -> dict:
        bc, gc = self.cfg["benchmark"], self.cfg["grid"]
        champ = self.p6.cfg["probes"]["champion"]
        fp = self.fp("benchmark", bc, gc, self.cfg["tasks"], self.w2_inputs(),
                     {t: (self.sel.get(t) or {}).get(champ) for t in self.cfg["tasks"]})
        if not self._done("benchmark", fp):
            t0 = self._begin("benchmark", fp)
            from ..gems3 import representation as R
            from ..m7.baseline import heldout_map

            rows, prov = [], []
            for u in self.units():
                b0, b1 = self.cnn(u)
                F = self.fusion(u)
                W = self.stack_rank(u, self.w2_map(u))
                cands = {"fallback_fusion": (F, 0.0), "cnn_b0": (R.rank01(b0, u.ev), np.nan),
                         "cnn_b1": (R.rank01(b1, u.ev), np.nan), "w2": (W, 1.0),
                         "hybrid": ((1.0 - float(bc["hybrid_blend"])) * F + float(bc["hybrid_blend"]) * W,
                                    float(bc["hybrid_blend"]))}
                for name in bc["candidates"]:
                    s, lam = cands[name]
                    rows += self.eval_score(u, s, candidate=name, blend=lam)
                qf = heldout_map(self.p6.view, u.task, champ, u.zone)
                if qf is not None:
                    z = next((r for r in (self.sel.get(u.task, {}).get(champ) or {}).get("per_zone", [])
                              if int(r["zone"]) == int(u.zone)), {})
                    rows.append(self._eval_q(u, qf, candidate="fallback_published", blend=np.nan,
                                             rep=str(z.get("rep", "published")), frac=float(z.get("frac", np.nan))))
                prov.append({"unit": u.key, "cnn": "oof" if u.task == "geo" else f"discovery_pred:{u.zone}",
                             "w2_map": f"stack/maps/{u.task}_{u.zone}.npy",
                             "w2_sha256": self.w2_inputs()["maps_sha256"].get(f"{u.task}_{u.zone}.npy"),
                             "champion_w": self.champion_w(u), "fallback_heldout": qf is not None,
                             "eligible_px": int(self.elig(u).sum()), "eval_px": int(u.ev.sum())})
                u.release()
            G = pd.DataFrame(rows)
            self.save_table(G, "grid", "benchmark")
            self.save_table(pd.DataFrame(prov), "inputs_provenance", "benchmark")
            self._a_tables(G)
            v = self.read_json("benchmark", "verdict") or {}
            self.save_json({"stage": "benchmark", "w2_gain_is_ranking": v.get("w2_gain_is_ranking"),
                            "largest_component": v.get("largest_component"), "ranking_gain": v.get("ranking_gain"),
                            "rule": v.get("rule"), "note": "diagnostic stage: selects nothing, promotes nothing"},
                           "benchmark", "decision")
            self._finish("benchmark", fp, t0, {"w2": self.w2_inputs(), "units": [p["unit"] for p in prov]},
                         {"w2_gain_is_ranking": v.get("w2_gain_is_ranking")})
            self.p6.push()
        return {"grid": self.load_table("grid", "benchmark"), "matched": self.load_table("matched_comparison", "benchmark"),
                "nested": self.load_table("nested_selection", "benchmark"),
                "ranking_gain": self.load_table("ranking_gain", "benchmark"),
                "verdict": self.read_json("benchmark", "verdict")}

    def _a_tables(self, G: pd.DataFrame) -> None:
        bc = self.cfg["benchmark"]
        fx = G[G.candidate != "fallback_published"]
        M = matched_comparison(fx)
        self.save_table(M, "matched_comparison", "benchmark")
        Z = fx.assign(dti=dti(fx.tp_w, fx.fp_w, fx.fn_w))
        self.save_table(Z.drop(columns=["systems"], errors="ignore"), "per_zone", "benchmark")
        summ, pers = [], []
        spaces = [(c, sc) for c in bc["candidates"] for sc in ("crest_1px", "skeleton_1px", "rep+frac")]
        for cand, scope in spaces:
            g = fx[fx.candidate == cand]
            if scope != "rep+frac":
                g = g[g.rep == scope]
            s, P = self.summarize(g, cand, ["frac"] if scope != "rep+frac" else ["rep", "frac"], f"{cand}|{scope}")
            summ.append(s)
            pers.append(P)
        if {"w2", "hybrid"} <= set(bc["candidates"]):
            s, P = self.summarize(fx[fx.candidate.isin(["w2", "hybrid"])], "w2_space", ["blend", "rep", "frac"],
                                  "w2_space|blend+rep+frac")
            summ.append(s)
            pers.append(P)
        fb = G[G.candidate == "fallback_published"]
        if len(fb):
            s = {"candidate": "fallback_published", "space": "fallback_published"}
            for task in self.cfg["tasks"]:
                g = fb[fb.task == task]
                pre = "disc" if task == self.prim else "geo"
                zs = {int(r.zone): float(dti(r.tp_w, r.fp_w, r.fn_w)) for r in g.itertuples()}
                s.update({f"{pre}_dti": float(dti(g.tp_w.sum(), g.fp_w.sum(), g.fn_w.sum())) if len(g) else np.nan,
                          f"{pre}_worst": min(zs.values()) if zs else np.nan, f"{pre}_per_zone": zs})
            s.update({f"systems_{k}": v for k, v in systems_vs(fb[fb.task == self.prim], self.fallback_systems()).items()})
            summ.append(s)
        N = pd.concat([p for p in pers if len(p)], ignore_index=True) if any(len(p) for p in pers) else pd.DataFrame()
        self.save_table(pd.DataFrame([_flat(s) for s in summ]), "nested_selection", "benchmark")
        self.save_table(N, "nested_per_zone", "benchmark")
        long = [{"task": task, "space": s["space"], "dti_crossfit": s.get(f"{pre}_dti", np.nan),
                 "fold_min": s.get(f"{pre}_worst", np.nan), "per_zone": s.get(f"{pre}_per_zone") or {}}
                for s in summ for task, pre in ((self.prim, "disc"), ("geo", "geo"))]
        RG, verdict = ranking_gain(pd.DataFrame(long), M, bc, self.prim)
        self.save_table(RG, "ranking_gain", "benchmark")
        self.save_table(self._coverage(fx, N), "system_coverage", "benchmark")
        verdict["note"] = ("nested numbers use the matched fracs only; W2's recorded 0.224640 also searched "
                           "fracs 0.02 / 0.075, so w2_space here may differ slightly")
        self.save_json(verdict, "benchmark", "verdict")
        self._fig_area(M)

    def _coverage(self, G: pd.DataFrame, N: pd.DataFrame) -> pd.DataFrame:
        fb = self.fallback_systems()
        rows = []
        d = G[G.task == self.prim]
        for r in d.itertuples():
            s, f = system_set(getattr(r, "systems", "")), fb.get(int(r.zone), set())
            rows.append({"kind": "fixed_cell", "candidate": r.candidate, "blend": r.blend, "rep": r.rep, "frac": r.frac,
                         "zone": int(r.zone), "recovered": len(s), "fallback": len(f), "gained": len(s - f),
                         "lost": len(f - s), "gained_ids": ";".join(map(str, sorted(s - f))),
                         "lost_ids": ";".join(map(str, sorted(f - s)))})
        n = N[N.task == self.prim] if len(N) else N
        for r in n.itertuples() if len(n) else []:
            s, f = system_set(getattr(r, "systems", "")), fb.get(int(r.zone), set())
            rows.append({"kind": "nested_choice", "candidate": r.space, "blend": getattr(r, "blend", np.nan),
                         "rep": getattr(r, "rep", None), "frac": r.frac, "zone": int(r.zone), "recovered": len(s),
                         "fallback": len(f), "gained": len(s - f), "lost": len(f - s),
                         "gained_ids": ";".join(map(str, sorted(s - f))), "lost_ids": ";".join(map(str, sorted(f - s)))})
        return pd.DataFrame(rows)

    def _fig_area(self, M: pd.DataFrame, stage: str = "benchmark") -> None:
        import matplotlib.pyplot as plt

        reps, tasks = self.cfg["grid"]["reps"], [self.prim, "geo"]
        fig, ax = plt.subplots(len(tasks), len(reps), figsize=(5.2 * len(reps), 3.8 * len(tasks)), squeeze=False)
        S = self.load_table("nested_selection", "benchmark")
        for i, t in enumerate(tasks):
            for j, rp in enumerate(reps):
                a = ax[i][j]
                m = M[(M.task == t) & (M.rep == rp)]
                for c, g in m.groupby("candidate"):
                    g = g.sort_values("frac")
                    a.plot(100 * g.frac, g.dti, marker="o", ms=3, label=c)
                fb = S[S.candidate == "fallback_published"]
                col = "disc_dti" if t == self.prim else "geo_dti"
                if len(fb) and col in fb:
                    a.axhline(float(fb[col].iloc[0]), ls="--", c="k", lw=0.8, label="fallback (published)")
                a.set(title=f"{t} / {rp}", xlabel="area budget (% of eligible)", ylabel="pooled DTI (fixed cell)")
                a.grid(alpha=0.3)
        ax[0][0].legend(fontsize=7)
        fig.suptitle("Matched budgets: pooled DTI per fixed (rep, area) cell, no selection")
        cols = ["task", "candidate", "blend", "rep", "frac", "dti", "worst_zone_dti", "precision_w", "recall_w",
                "area_px", "length_km"]
        self.p6.ws.save_fig(fig, "dti_by_area", M[[c for c in cols if c in M]], sub=self.sub(stage))
        plt.close(fig)

    # =========================================================================== B
    def g02(self):
        """Attached G02 run (explicit ``geology.g02_run_id``; attach computes nothing and never backfills)."""
        if self.p6.__dict__.get("g02") is None:
            from ..m9.research.g02_experiment import G02Experiment

            try:
                self.p6.__dict__["g02"] = G02Experiment.attach(self.p6.m9, self.cfg["geology"]["g02_run_id"])
                self.g02_error = None
            except FileNotFoundError as e:                       # includes AmbiguousRunError
                self.p6.__dict__["g02"] = None
                self.g02_error = f"{type(e).__name__}: {e}"
        return self.p6.__dict__["g02"]

    def g02_ident(self, names: list[str] | None = None) -> dict | None:
        """Identity of the attached G02 run (id, fingerprint, manifest hash) when ``names`` use a G02 / G01 feature
        (None otherwise, so fingerprints of variants without G02 features are unaffected)."""
        if names is not None and not set(names) & set(self.cfg["geology"]["keys"]):
            return None
        g = self.g02()
        if g is None:
            return None
        a = getattr(g, "attach_info", {}) or {}
        return {k: a.get(k) for k in ("run_id", "fingerprint", "manifest_sha256")}

    def g02_audit(self) -> tuple[dict, pd.DataFrame]:
        """Attach result + per (feature, unit) availability and construction provenance (no map is computed)."""
        gcf = self.cfg["geology"]
        g = self.g02()
        info = dict(getattr(g, "attach_info", {}) or {}) if g is not None else {"error": getattr(self, "g02_error", "")}
        info["root_cause"] = ("LATEST.json is written one level above the run folders and G02Experiment.push() synced "
                              "only the run folder, so a fresh runtime staged g02_<fp>/ without LATEST.json and "
                              "attach() found no run id (fixed: g02_io.resolve_run + push of LATEST.json).")
        rows = []
        if g is None:
            return info, pd.DataFrame(rows)
        st = g.store
        man = (st.load_json("manifests/manifest.json") or {}).get("artifacts") or {}
        inp = st.load_json("config/inputs.json") or {}
        have = set(g.raster_units())
        for u in self.units():
            key = f"{u.task}_{u.zone}"
            rel = f"rasters/units/{key}.npz"
            present = set(st.raster_keys(rel)) if key in have else set()
            meta = {}
            if key in have:
                _, meta = st.load_raster_keys(rel, [])
            refp = (man.get(f"processed/refs_{key}.json") or {}).get("params") or {}
            for feat, rk in gcf["keys"].items():
                rows.append({"feature": feat, "raster_key": rk, "unit": u.key, "raster": key in have,
                             "key_present": rk in present, "backfill": bool(meta.get("backfill", False)),
                             "labels_stored": meta.get("labels_stored"), "window_matches": (
                                 [int(x) for x in meta.get("window", [])] == [u.sl[0].start, u.sl[0].stop,
                                                                              u.sl[1].start, u.sl[1].stop])
                             if meta else None, "normalization_reference": refp.get("reference", "unknown"),
                             "reference_px": refp.get("ref_px"), "mode": meta.get("mode", info.get("mode")),
                             "source_channels": len(inp.get("input_fingerprints") or {}),
                             "support_halo_m": inp.get("halo_m"),
                             "g02_wavelengths_km": ";".join(str(w["lam_km"]) for w in inp.get("g02_wavelengths", [])),
                             "resampling": "none (native 100 m M9 grid; window = unit crop)",
                             "withheld_labels_used": "no (label-free maps; percentile references on the unit's "
                                                     "training geography, assert_disjoint with the evaluation zone)",
                             "full_region": "computable label-free (stack.g02_full, whole footprint as reference)"})
        rows.append({"feature": "g02f", "raster_key": "F:<base>+<family>@w", "unit": "*", "raster": False,
                     "key_present": False, "withheld_labels_used": "fusion weight / family chosen with labels",
                     "full_region": "not persisted", "status": "EXCLUDED",
                     "reason": "G02-F is evaluated lazily from persisted maps with a label-selected weight; its "
                               "components (G01-E, G02-x) are themselves candidate features of a tree model"})
        return info, pd.DataFrame(rows)

    def backfill(self, max_units: int | None = None) -> list[str]:
        """Compute the missing per-unit G02 rasters with the attached run's own configuration. Explicit opt-in only
        (``geology.backfill_missing``); refused when the current inputs differ from the run's recorded input
        fingerprints. Existing rasters are never rewritten; an interrupted unit resumes from its finished tiles."""
        g = self.g02()
        if g is None:
            raise FileNotFoundError(f"no attachable G02 run: {getattr(self, 'g02_error', '')}")
        drift = g.input_drift()
        if not drift["ok"]:
            raise RuntimeError(f"G02 backfill refused: current inputs differ from run {g.attach_info.get('run_id')} "
                               f"({drift}); the backfilled rasters would not share the run's feature definitions")
        have = set(g.raster_units())
        todo = [u for u in self.units() if f"{u.task}_{u.zone}" not in have][: int(max_units or
                                                                                    self.cfg["geology"]["backfill_max_units"])]
        sec = [float((g.store.load_json(f"processed/unit_{k}.json") or {}).get("seconds", np.nan)) for k in have]
        est = np.nanmean(sec) if sec else np.nan
        self.say(f"backfill {len(todo)} unit(s) {[u.key for u in todo]}; measured {np.round(sec, 0).tolist()} s per "
                 f"evaluated unit (raster-only backfill is cheaper; estimate <= {len(todo) * est / 3600:.2f} CPU-h)")
        g.m9.load()
        out = []
        for u in todo:
            g.unit_raster(u)
            out.append(u.key)
        g.attach_info["units_with_rasters"] = g.raster_units()
        self._raw.clear()                                        # unit features cached before the backfill are stale
        return out

    def raw(self, u) -> dict:
        """W2's unit features (``stack.unit_raw``: CNN ranks, potential fields, permitted known geometry) + the
        configured G02 / G01 layers of this unit (metadata-checked) when the raster exists."""
        if u.key not in self._raw:
            self.g02()
            raw, prov = ST.unit_raw(self.p6, u, list(self.p6.m9.models)[:2])
            for k in ST.G02:
                raw.pop(k, None)
            g, gp = ST.g02_unit(self.p6, u, dict(self.cfg["geology"]["keys"]))
            raw.update({k: v for k, v in g.items() if np.isfinite(v[u.ev]).any()})
            raw["_prov"] = {**prov, **gp}
            self._raw[u.key] = raw
        return self._raw[u.key]

    def view_raw(self, u, view: str | None, raw: dict | None = None, ev: np.ndarray | None = None) -> dict:
        raw = self.raw(u) if raw is None else raw
        ev = u.ev if ev is None else ev
        if view == "rank_known":
            raw = {**raw, **{k: ST.rank_in(raw[k], ev) for k in self.cfg["robustness"]["rank_known"] if k in raw}}
        elif view not in (None, "", "none"):
            raise KeyError(f"unknown feature view {view!r}")
        return raw

    def rows0(self) -> pd.DataFrame:
        if self._T0 is None:
            p = self.p6.ws.root / "stack" / "features" / "rows.parquet"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: the matched ablation reuses W2's sampled rows")
            self._T0 = pd.read_parquet(p)
        return self._T0

    def w2_features(self) -> list[str]:
        f = (self.w2_inputs().get("candidate") or {}).get("features")
        if not f:
            raise FileNotFoundError("stack/candidate.json has no feature list: run W2 first")
        return list(f)

    def rows(self, names: list[str], view: str | None = None, check: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """W2's sampled rows and weights with the columns of ``names`` under ``view``: W2 columns are reused, new or
        transformed columns are read from the unit features exactly as at prediction (``stack.assemble``).
        ``check`` recomputes W2's own columns and reports the largest difference (reproducibility audit)."""
        T0 = self.rows0()
        base = self.w2_features()
        tv = set(self.cfg["robustness"]["rank_known"]) if view == "rank_known" else set()
        keep = ["task", "zone", "r_full", "c_full", "y", "w"]
        T = T0[keep + [n for n in base if n in names and n not in tv]].copy()
        need = [n for n in names if n not in T]
        audit = []
        for n in need:
            T[n] = np.float32(np.nan)
        for u in self.units():
            m = ((T.task == u.task) & (T.zone == int(u.zone))).to_numpy()
            if not m.any():
                continue
            H, W = u.ev.shape
            idx = (T.r_full.to_numpy()[m] - u.sl[0].start) * W + (T.c_full.to_numpy()[m] - u.sl[1].start)
            pos = ST.ev_position(u.ev, idx.astype(np.int64))
            vr = self.view_raw(u, view)
            if need:
                miss = [n for n in need if n not in vr]
                if miss:
                    raise KeyError(f"{u.key}: features {miss} absent (never substituted)")
                T.loc[m, need] = ST.assemble(vr, u.ev, need)[pos]
            if check:
                X = ST.assemble(self.raw(u), u.ev, base)[pos]
                for j, n in enumerate(base):
                    a, b = X[:, j], T0.loc[m, n].to_numpy(np.float32)
                    same_nan = bool((np.isnan(a) == np.isnan(b)).all())
                    d = np.abs(a - b)
                    audit.append({"unit": u.key, "feature": n, "rows": int(m.sum()), "nan_pattern_equal": same_nan,
                                  "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0})
        A = pd.DataFrame(audit)
        if check and len(A) and ((A.max_abs_diff > 1e-4) | ~A.nan_pattern_equal).any():
            bad = A[(A.max_abs_diff > 1e-4) | ~A.nan_pattern_equal]
            raise RuntimeError(f"W2 rows are not reproducible from the current features: {bad.to_dict('records')[:4]}")
        return T, A

    def crossfit(self, stage: str, name: str, names: list[str], view: str | None = None,
                 dropout: dict | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
        """Leave-one-zone-out LightGBM exactly as W2 (rows, weights, buffer, fixed hyper-parameters); only
        ``names`` / ``view`` / ``dropout`` differ. Maps cached under ``<stage>/maps/<name>_<fp8>/``."""
        sc = self.p6.cfg["stack"]
        parts = [stage, name, names, view, dropout, sc["lgbm"], sc["exclude_buffer_km"], self.w2_inputs()]
        gid = self.g02_ident(names)
        fpv = self.fp(*parts, gid) if gid else self.fp(*parts)  # a different G02 run never reuses cached maps
        d = self.root / stage / "maps" / f"{name}_{fpv[:8]}"
        seed = int(self.p6.cfg["seed"])
        shape = self.p6.ctx.data.shape
        T = None
        scores, audit = {}, []
        ip = d / "importance.csv"
        imps = [pd.read_csv(ip)] if ip.exists() else []
        cols = [names.index(c) for c in (self.cfg["robustness"]["known_group"] if dropout else []) if c in names]
        for i, u in enumerate(self.units()):
            mp = d / f"{u.task}_{u.zone}.npy"
            if mp.exists():
                scores[u.key] = np.load(mp)
                audit.append({"variant": name, "unit": u.key, "status": "reused"})
                continue
            if T is None:
                T, _ = self.rows(names, view)
            tr = ST.train_index(T, u, shape, float(sc["exclude_buffer_km"]))
            y = T.y.to_numpy()[tr]
            a = {"variant": name, "unit": u.key, "n_train": int(tr.sum()), "n_pos": int((y == 1).sum()),
                 "n_neg": int((y == 0).sum())}
            if a["n_pos"] < 50 or a["n_neg"] < 50:
                audit.append({**a, "status": "too few training rows: unit skipped"})
                continue
            t0 = time.perf_counter()
            X = T.loc[tr, names].to_numpy(np.float32)
            if dropout:
                X = group_dropout(X, cols, float(dropout["p"]), np.random.default_rng(seed + 31 * i))
            m = ST.make_model(sc["lgbm"], seed)
            m.fit(X, y, sample_weight=T.w.to_numpy()[tr])
            scores[u.key] = ST.predict_unit(m, self.view_raw(u, view), u.ev, names)
            d.mkdir(parents=True, exist_ok=True)
            from ..artifacts.atomic import atomic_save

            atomic_save(mp, lambda t, a_=scores[u.key]: np.save(t, a_))
            imps.append(pd.DataFrame({"variant": name, "unit": u.key, "feature": names,
                                      "gain": m.booster_.feature_importance("gain").astype(float)}))
            audit.append({**a, "status": "trained", "seconds": round(time.perf_counter() - t0, 1)})
        I = pd.concat(imps, ignore_index=True).drop_duplicates(["unit", "feature"]) if imps else pd.DataFrame()
        if len(I) and (not ip.exists() or any(a.get("status") == "trained" for a in audit)):
            from ..artifacts.atomic import atomic_save   # resumed runs merge the earlier units' importances

            d.mkdir(parents=True, exist_ok=True)
            atomic_save(ip, lambda t: I.to_csv(t, index=False))
        man = {"variant": name, "features": names, "view": view, "dropout": dropout, "fingerprint": fpv,
               "maps_dir": d.relative_to(self.p6.ws.root).as_posix(), "lgbm": sc["lgbm"], "seed": seed,
               "rows": "stack/features/rows.parquet (W2 sample, identical rows and weights)",
               "exclude_buffer_km": sc["exclude_buffer_km"]}
        return scores, pd.DataFrame(audit), I, man

    def run_variants(self, stage: str, specs: list[dict], base_name: str, base_maps: dict | None = None) -> dict:
        """Cross-fit + matched evaluation + nested summaries of variants; ``specs``: [{name, names, view, dropout,
        status, reason}]. ``base_maps`` (W2 maps) are evaluated as ``base_name`` without retraining."""
        rows, audits, imps, mans, summ, pers = [], [], [], {}, [], []
        for sp in specs:
            if sp.get("status") == "BLOCKED":
                summ.append({"candidate": sp["name"], "status": "BLOCKED", "reason": sp.get("reason"),
                             "features": sp["names"]})
                continue
            if sp["name"] == base_name and base_maps is not None:
                scores = base_maps
                mans[sp["name"]] = {"variant": sp["name"], "features": sp["names"], "maps_dir": "stack/maps (W2)",
                                    "note": "W2 held-out maps reused (no retraining)"}
            else:
                scores, A, I, man = self.crossfit(stage, sp["name"], sp["names"], sp.get("view"), sp.get("dropout"))
                audits.append(A)
                imps.append(I)
                mans[sp["name"]] = man
            lost = [u.key for u in self.units() if u.key not in scores]
            if lost:                                             # a pooled number on fewer zones is not matched
                summ.append({"candidate": sp["name"], "status": "BLOCKED", "features": sp["names"],
                             "reason": f"units without a cross-fitted map: {lost} (see crossfit_audit)"})
                continue
            g = []
            for u in self.units():
                if u.key in scores:
                    g += self.eval_variant(u, scores[u.key], sp["name"])
                    u.release()
            G = pd.DataFrame(g)
            rows.append(G)
            s, P = self.summarize(G, sp["name"], ["blend", "rep", "frac"])
            s.update(status="RUN", features=sp["names"], view=sp.get("view"), dropout=sp.get("dropout"))
            summ.append(s)
            pers.append(P)
        G = pd.concat([r for r in rows if len(r)], ignore_index=True) if rows else pd.DataFrame()
        return {"grid": G, "summary": summ, "per_zone": pd.concat([p for p in pers if len(p)], ignore_index=True)
                if any(len(p) for p in pers) else pd.DataFrame(),
                "audit": pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
                "importance": pd.concat([i for i in imps if len(i)], ignore_index=True)
                if any(len(i) for i in imps) else pd.DataFrame(), "manifests": mans}

    def _decide(self, stage: str, out: dict, base_name: str, rules: dict) -> dict:
        geo_floor = float(self.cfg["selection"]["geo_floor"])
        S = {s["candidate"]: s for s in out["summary"]}
        base = S.get(base_name, {})
        rows = []
        for name, s in S.items():
            if name == base_name:
                rows.append({**_flat(s), "decision": "BASE", "reasons": ""})
                continue
            ok, why = accept(s, base, rules, geo_floor)
            rows.append({**_flat(s), "decision": "ACCEPT" if ok else ("BLOCKED" if s.get("status") == "BLOCKED"
                                                                      else "REJECT"),
                         "reasons": "; ".join(why), "gain_vs_base": (s.get("disc_dti") or np.nan) -
                         (base.get("disc_dti") or np.nan)})
        D = pd.DataFrame(rows)
        acc = D[D.decision == "ACCEPT"]
        best = acc.sort_values("disc_dti", ascending=False).iloc[0].candidate if len(acc) else base_name
        dec = {"stage": stage, "base": base_name, "chosen": str(best), "accepted": list(acc.candidate),
               "rules": rules, "geo_floor": geo_floor, "chosen_summary": S.get(best, {}),
               "spec": next((m for k, m in out["manifests"].items() if k == best), None)}
        self.save_table(D, "ablation", stage)
        self.save_json(dec, stage, "decision")
        return dec

    def _save_variant_tables(self, stage: str, out: dict) -> None:
        if len(out["grid"]):
            self.save_table(out["grid"], "grid", stage)
            self.save_table(matched_comparison(out["grid"]), "matched_comparison", stage)
        self.save_table(out["per_zone"], "per_zone", stage)
        self.save_table(out["audit"], "crossfit_audit", stage)
        I = out["importance"]
        if len(I):
            g = I.groupby(["variant", "feature"]).gain.mean().reset_index()
            g["share"] = g.gain / g.groupby("variant").gain.transform("sum")
            self.save_table(g.sort_values(["variant", "gain"], ascending=[True, False]), "feature_importance", stage)
        sy = [{"variant": s["candidate"], **{k[8:]: v for k, v in s.items() if k.startswith("systems_")}}
              for s in out["summary"] if s.get("status") == "RUN"]
        self.save_table(pd.DataFrame(sy), "systems", stage)
        self.save_json(out["manifests"], stage, "model_manifests")

    def b_geology(self) -> dict:
        return self.section("geology", self._b)

    def _b(self) -> dict:
        gcf = self.cfg["geology"]
        info, AU = self.g02_audit()
        if gcf.get("backfill_missing") and info.get("run_id"):
            info["backfilled"] = self.backfill()
            info, AU = self.g02_audit()
        self.save_json(info, "geology", "g02_attach")
        self.save_table(AU, "g02_audit", "geology")
        present = {u.key: set(AU[(AU.unit == u.key) & AU.key_present.astype(bool)].feature) if len(AU) else set()
                   for u in self.units()}                     # from the audit: a reload loads no feature
        gfp = {k: v for k, v in gcf.items() if k not in EXEC_SWITCHES["geology"]}   # switches never invalidate
        fp = self.fp("geology", gfp, self.cfg["grid"], self.w2_inputs(), sorted((k, sorted(v)) for k, v in present.items()),
                     self.g02_ident())
        if not self._done("geology", fp):
            t0 = self._begin("geology", fp)
            _, RA = self.rows(self.w2_features(), check=True)
            self.save_table(RA, "row_consistency", "geology")
            base = self.w2_features()
            specs = []
            for name, add in gcf["variants"].items():
                st, miss = variant_status(present, list(add))
                specs.append({"name": name, "names": base + [a for a in add if a not in base], "status": st,
                              "reason": (f"missing {miss} (G02 run {info.get('run_id')} mode {info.get('mode')}; "
                                         "set geology.backfill_missing = true)") if miss else ""})
            self.save_table(pd.DataFrame([{k: (";".join(v) if isinstance(v, list) else v) for k, v in s.items()}
                                          for s in specs]), "availability", "geology")
            w2 = {u.key: self.w2_map(u) for u in self.units()}
            out = self.run_variants("geology", specs, "S0", w2)
            self._save_variant_tables("geology", out)
            dec = self._decide("geology", out, "S0", gcf["accept"])
            blocked = [s["name"] for s in specs if s["name"] != "S0" and s["status"] == "BLOCKED"]
            n_var = sum(s["name"] != "S0" for s in specs)
            self._finish("geology", fp, t0, {"g02": info, "w2": self.w2_inputs(), "present": present},
                         {"chosen": dec["chosen"], "accepted": dec["accepted"], "blocked": blocked},
                         status="BLOCKED" if n_var and len(blocked) == n_var else "COMPLETE",
                         reason=f"variants BLOCKED: {blocked}" if blocked else "")
            self.p6.push()
        return {"attach": info, "audit": AU, "ablation": self.load_table("ablation", "geology"),
                "decision": self.read_json("geology", "decision")}

    # =========================================================================== C
    def base_spec(self, after: str) -> dict:
        """The champion-so-far: the last accepted variant of the stages before ``after`` (W2 = S0 otherwise)."""
        order = [s for s in ("geology", "robustness", "hybrid") if STAGES.index(s) < STAGES.index(after)]
        spec = {"name": "S0", "names": self.w2_features(), "features": self.w2_features(), "view": None,
                "dropout": None, "stage": "w2", "kind": "lgbm"}
        for st in order:
            d = self.read_json(st, "decision") or {}
            if d.get("chosen") and d["chosen"] != d.get("base") and d.get("spec"):
                spec = {**d["spec"], "name": d["chosen"], "names": d["spec"]["features"], "stage": st,
                        "kind": d["spec"].get("kind", "lgbm")}
        return spec

    def base_maps(self, spec: dict) -> dict:
        if spec["name"] == "S0":
            return {u.key: self.w2_map(u) for u in self.units()}
        d = self.p6.ws.root / spec["maps_dir"]
        return {u.key: np.load(d / f"{u.task}_{u.zone}.npy") for u in self.units()}

    def c_robustness(self) -> dict:
        return self.section("robustness", self._c)

    def _c(self) -> dict:
        rc = self.cfg["robustness"]
        spec = self.base_spec("robustness")
        fp = self.fp("robustness", rc, self.cfg["grid"], spec, self.w2_inputs())
        if not self._done("robustness", fp):
            t0 = self._begin("robustness", fp)
            bm = self.base_maps(spec)
            self.diagnostics(spec, bm)
            specs = [{"name": spec["name"], "names": spec["names"], "view": spec.get("view"),
                      "dropout": spec.get("dropout")}]
            for name, v in rc["variants"].items():
                specs.append({"name": name, "names": spec["names"], "view": v.get("view", spec.get("view")),
                              "dropout": v.get("dropout", spec.get("dropout"))})
            out = self.run_variants("robustness", specs, spec["name"], bm)
            self._save_variant_tables("robustness", out)
            dec = self._decide("robustness", out, spec["name"], rc["accept"])
            self._finish("robustness", fp, t0, {"base": spec, "w2": self.w2_inputs()},
                         {"chosen": dec["chosen"], "accepted": dec["accepted"]})
            self.p6.push()
        return {"ablation": self.load_table("ablation", "robustness"),
                "decision": self.read_json("robustness", "decision"),
                "diagnostics": {n: self.load_table(n, "robustness") for n in
                                ("zone_ranking", "feature_shift", "known_distance", "composition", "prediction_geometry")}}

    def diagnostics(self, spec: dict, bm: dict) -> None:
        """Nine label-aware DIAGNOSTICS per zone (no output feeds a model, a threshold or a choice)."""
        from scipy.stats import ks_2samp

        rc, gc = self.cfg["robustness"], self.cfg["grid"]
        px = float(gc["pixel_km"])
        rng = np.random.default_rng(int(self.p6.cfg["seed"]))
        n_s = int(rc["diag_sample_px"])
        feats = ["mag_hgrad", "grav_hgrad", "mag_tilt", "mg_orient_agree", "known_dist_km", "known_density",
                 "tip_dist_km"] + [k for k in self.cfg["geology"]["keys"]]
        N = self.load_table("nested_per_zone", "benchmark")
        T0 = self.rows0()
        rank_r, shift, known, comp, geom, samples = [], [], [], [], [], {}
        for u in self.units():
            raw, ev, truth = self.raw(u), u.ev, u.eval_truth()
            F, b0, b1 = self.fusion(u), *self.cnn(u)
            for nm, s in (("cnn_b0", b0), ("cnn_b1", b1), ("fallback_fusion", F), ("w2", self.w2_map(u)),
                          (spec["name"], bm[u.key])):
                rank_r.append({"task": u.task, "zone": int(u.zone), "ranker": nm,
                               "truth_mean_rank": mean_truth_rank(s, truth, ev), "truth_px": int(truth.sum())})
            for f in feats:
                if f not in raw:
                    continue
                v = np.asarray(raw[f], np.float32)[ev]
                v = v[np.isfinite(v)]
                smp = rng.choice(v, min(n_s, v.size), replace=False) if v.size else v
                samples.setdefault((u.task, f), {})[int(u.zone)] = smp
                shift.append({"task": u.task, "zone": int(u.zone), "feature": f, **quantiles(v), "mean": float(v.mean())
                              if v.size else np.nan})
            kd = np.asarray(raw["known_dist_km"], np.float32)
            known.append({"task": u.task, "zone": int(u.zone), "where": "evaluation", **quantiles(kd[ev])})
            known.append({"task": u.task, "zone": int(u.zone), "where": "truth", **quantiles(kd[truth])})
            r = T0[(T0.task == u.task) & (T0.zone == int(u.zone))]
            area_km2 = float(ev.sum()) * px * px
            comp.append({"task": u.task, "zone": int(u.zone), "eval_px": int(ev.sum()),
                         "eligible_px": int(self.elig(u).sum()), "truth_px": int(truth.sum()),
                         "truth_km_per_100km2": float(truth.sum() * px / max(area_km2, 1e-9) * 100.0),
                         "rows_pos": int((r.y == 1).sum()), "rows_neg": int((r.y == 0).sum()),
                         "rows_ambiguous": int(((r.y == 0) & (r.w < 1.0)).sum()),
                         "known_density_mean": float(np.nanmean(np.asarray(raw["known_density"])[ev])),
                         "g02c_mean": float(np.nanmean(np.asarray(raw["g02c"])[ev])) if "g02c" in raw else np.nan})
            th_t = orientation_hist(axial_orientation(truth, int(rc["orientation_window_px"])), int(rc["orientation_bins"]))
            pk = ST.permitted_known(self.p6.ctx, u)[u.sl]
            for nm, q in self._diag_maps(u, N):
                m = np.asarray(q) > 0
                dt = distance_km(m, truth, px)
                th_p = orientation_hist(axial_orientation(m, int(rc["orientation_window_px"])), int(rc["orientation_bins"]))
                tol = float(rc["truth_tolerance_km"])
                geom.append({"task": u.task, "zone": int(u.zone), "map": nm,
                             **line_stats(m, px, int(gc["short_segment_px"])),
                             "orientation_tv_vs_truth": hist_distance(th_p, th_t),
                             "share_within_tol": float((dt <= tol).mean()) if dt.size else np.nan,
                             **quantiles(dt, prefix="dist_truth_q"),
                             **quantiles(distance_km(m, pk, px), prefix="dist_known_q")})
            u.release()
        for (task, f), zs in samples.items():
            for z, v in zs.items():
                rest = np.concatenate([w for k, w in zs.items() if k != z]) if len(zs) > 1 else np.zeros(0)
                ks = ks_2samp(v, rest).statistic if v.size and rest.size else np.nan
                for row in shift:
                    if row["task"] == task and row["zone"] == z and row["feature"] == f:
                        row["ks_vs_other_zones"] = float(ks)
        tabs = {"zone_ranking": pd.DataFrame(rank_r), "feature_shift": pd.DataFrame(shift),
                "known_distance": pd.DataFrame(known), "composition": pd.DataFrame(comp),
                "prediction_geometry": pd.DataFrame(geom)}
        cov = self.load_table("system_coverage", "benchmark")
        tabs["systems"] = cov[cov.kind == "nested_choice"] if len(cov) else cov
        for n, t in tabs.items():
            self.save_table(t, n, "robustness")
        self._fig_zones(tabs)

    def _diag_maps(self, u, N: pd.DataFrame):
        """Held-out maps of the W2 nested choice and of the published fallback on this unit."""
        from ..m7.baseline import heldout_map

        n = N[(N.space == "w2_space|blend+rep+frac") & (N.task == u.task) & (N.zone == int(u.zone))] if len(N) else N
        if len(n):
            r = n.iloc[0]
            lam = float(r.blend)
            s = ((1 - lam) * self.fusion(u) + lam * self.stack_rank(u, self.w2_map(u))).astype(np.float32)
            yield "w2_nested_choice", self._apply(u, s, str(r.rep), float(r.frac))
        qf = heldout_map(self.p6.view, u.task, self.p6.cfg["probes"]["champion"], u.zone)
        if qf is not None:
            yield "fallback_published", qf

    def _fig_zones(self, tabs: dict) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
        long = []
        zr = tabs["zone_ranking"]
        zr = zr[zr.task == self.prim] if len(zr) else zr
        for nm, g in zr.groupby("ranker") if len(zr) else []:
            ax[0][0].plot(g.zone, g.truth_mean_rank, marker="o", label=nm)
            long += [{"panel": "truth_mean_rank", "series": nm, "zone": z, "value": v}
                     for z, v in zip(g.zone, g.truth_mean_rank)]
        ax[0][0].set(title="truth mean rank (0.5 = random)", xlabel="zone")
        ax[0][0].legend(fontsize=7)
        fs = tabs["feature_shift"]
        fs = fs[fs.task == self.prim] if len(fs) else fs
        if len(fs) and "ks_vs_other_zones" in fs:
            pv = fs.pivot_table(index="feature", columns="zone", values="ks_vs_other_zones")
            pv.plot.barh(ax=ax[0][1])
            long += [{"panel": "ks_shift", "series": f, "zone": z, "value": v}
                     for f, r in pv.iterrows() for z, v in r.items()]
        ax[0][1].set(title="KS distance: zone vs other zones", xlabel="KS")
        kd = tabs["known_distance"]
        kd = kd[kd.task == self.prim] if len(kd) else kd
        for w, g in kd.groupby("where") if len(kd) else []:
            ax[1][0].plot(g.zone, g.q50, marker="o", label=f"median at {w}")
            long += [{"panel": "known_dist_median_km", "series": w, "zone": z, "value": v} for z, v in zip(g.zone, g.q50)]
        ax[1][0].set(title="distance to permitted known faults (km)", xlabel="zone")
        ax[1][0].legend(fontsize=7)
        pg = tabs["prediction_geometry"]
        pg = pg[pg.task == self.prim] if len(pg) else pg
        for nm, g in pg.groupby("map") if len(pg) else []:
            ax[1][1].plot(g.zone, g.short_share, marker="o", label=f"{nm}: short share")
            ax[1][1].plot(g.zone, g.share_within_tol, marker="s", ls="--", label=f"{nm}: within 300 m")
            long += [{"panel": "short_share", "series": nm, "zone": z, "value": v} for z, v in zip(g.zone, g.short_share)]
            long += [{"panel": "share_within_tol", "series": nm, "zone": z, "value": v}
                     for z, v in zip(g.zone, g.share_within_tol)]
        ax[1][1].set(title="fragmentation / precision proxy", xlabel="zone")
        ax[1][1].legend(fontsize=7)
        fig.suptitle(f"{self.prim}: weak-zone diagnostics (held-out labels used for diagnosis only)")
        self.p6.ws.save_fig(fig, "zone_diagnostics", pd.DataFrame(long), sub=self.sub("robustness"))

    # =========================================================================== D
    def d_hybrid(self) -> dict:
        return self.section("hybrid", self._d)

    def _d(self) -> dict:
        hc, gc = self.cfg["hybrid"], self.cfg["grid"]
        spec = self.base_spec("hybrid")
        fp = self.fp("hybrid", hc, gc, spec, self.w2_inputs())
        if not self._done("hybrid", fp):
            t0 = self._begin("hybrid", fp)
            bm = self.base_maps(spec)
            rows, d3, seg = [], [], []
            for u in self.units():
                r, c3, sg = self._d_unit(u, bm[u.key])
                rows += r
                d3 += c3
                seg += sg
                u.release()
            G = pd.DataFrame(rows).replace({"candidate": {"base": spec["name"]}})
            self.save_table(G, "grid", "hybrid")
            M = matched_comparison(G)
            self.save_table(M, "matched_comparison", "hybrid")
            self.save_table(pd.DataFrame(d3), "d3_geological_context", "hybrid")
            self.save_table(pd.DataFrame(seg), "segment_selection", "hybrid")
            summ, pers = [], []
            for name in [spec["name"]] + list(hc["variants"]):
                s, P = self.summarize(G[G.candidate == name], name, ["blend", "rep", "frac"])
                s.update(status="RUN", features=spec["names"], kind="hybrid" if name != spec["name"] else spec["kind"])
                if name != spec["name"] and len(P):
                    s["fp_increase"] = _fp_increase(G, P[P.task == self.prim], name, spec["name"])
                summ.append(s)
                pers.append(P)
            out = {"grid": G, "summary": summ, "per_zone": pd.concat([p for p in pers if len(p)], ignore_index=True)
                   if any(len(p) for p in pers) else pd.DataFrame(),
                   "audit": pd.DataFrame(), "importance": pd.DataFrame(),
                   "manifests": {n: {**{k: v for k, v in spec.items() if k != "name"}, "kind": "hybrid", "op": n,
                                     "base": spec, "hybrid": hc} for n in hc["variants"]}}
            out["manifests"][spec["name"]] = spec
            self._save_variant_tables("hybrid", out)
            dec = self._decide("hybrid", out, spec["name"], hc["accept"])
            self._finish("hybrid", fp, t0, {"base": spec, "w2": self.w2_inputs()},
                         {"chosen": dec["chosen"], "accepted": dec["accepted"]})
            self.p6.push()
        return {"ablation": self.load_table("ablation", "hybrid"), "decision": self.read_json("hybrid", "decision"),
                "d3": self.load_table("d3_geological_context", "hybrid")}

    def _d_layers(self, u, F: np.ndarray, raw: dict, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, str]:
        """Label-free segment evidence: fallback-only ordering (fusion, CNN disagreement, G02-C when present) and
        the geology layer of D2 / D3 (G02-C, else magnetic-gravity strike agreement)."""
        from ..gems3 import representation as R

        g = R.rank01(raw["g02c"], ev) if "g02c" in raw else None
        ctx = (g, "g02c") if g is not None else (R.rank01(raw["mg_orient_agree"], ev) if "mg_orient_agree" in raw
                                                 else None, "mg_orient_agree")
        lay = [F, R.rank01(np.nan_to_num(raw["cnn_disagree"], nan=0.0), ev)] + ([g] if g is not None else [])
        return np.mean(np.stack(lay), 0).astype(np.float32), ctx[0], ctx[1]

    def _d_unit(self, u, smap: np.ndarray) -> tuple[list, list, list]:
        from ..postprocess.topology import _disk

        hc, gc = self.cfg["hybrid"], self.cfg["grid"]
        raw = self.raw(u)
        F, S = self.fusion(u), self.stack_rank(u, smap)
        fbe, geo, geo_name = self._d_layers(u, F, raw, u.ev)
        crest_f = self._crest(u, F)
        rows, d3, seg = [], [], []
        beta = float(hc["union_fallback_share"])
        for lam in gc["blends"]:
            s = ((1.0 - float(lam)) * F + float(lam) * S).astype(np.float32)
            crest = self._crest(u, s)
            for rn in gc["reps"]:
                for f in gc["area_fracs"]:
                    tags = {"blend": float(lam), "rep": rn, "frac": float(f)}
                    full = self._apply(u, s, rn, f, crest) > 0
                    k = int(full.sum())
                    rows.append(self._eval_q(u, full, candidate="base", **tags))
                    if "D1" in hc["variants"]:
                        part = self._apply(u, s, rn, f * (1.0 - beta), crest) > 0
                        fb = self._apply(u, F, rn, f, crest_f) > 0
                        q1 = controlled_union(part, full, fb, fbe, s, k, int(hc["union_tolerance_px"]))
                        rows.append(self._eval_q(u, q1, candidate="D1", **tags))
                    if "D2" in hc["variants"]:
                        pool = self._apply(u, s, rn, f * float(hc["pool_multiplier"]), crest) > 0
                        q2, T = connectivity_select(pool, s, k, geo if geo_name == "g02c" else None,
                                                    weights=hc["weights"], short_px=int(hc["short_px"]),
                                                    short_penalty=float(hc["short_penalty"]),
                                                    ep_radius=int(hc["endpoint_radius_px"]))
                        rows.append(self._eval_q(u, q2, candidate="D2", **tags))
                        seg.append({"unit": u.key, **tags, "pool_segments": len(T),
                                    "kept_segments": int(T.kept.sum()) if len(T) else 0,
                                    "kept_short_share": float((T[T.kept].n_px < int(hc["short_px"])).mean())
                                    if len(T) and T.kept.any() else np.nan, "geology_layer": geo_name if
                                    geo_name == "g02c" else "absent"})
                    c3 = hc["d3"]
                    if u.is_disc and rn == c3["rep"] and float(f) == float(c3["frac"]) and float(lam) == float(c3["blend"]) \
                            and geo is not None:
                        pool = self._apply(u, s, rn, f * float(hc["pool_multiplier"]), crest) > 0
                        lab = branches(pool)
                        T = seg_table(lab, s, {"ctx": geo}, pool, int(hc["endpoint_radius_px"]))
                        if len(T):
                            near = ndimage.binary_dilation(u.eval_truth(), structure=_disk(3))
                            prec = np.bincount(lab[near & (lab > 0)], minlength=int(lab.max()) + 1)[1:] / \
                                np.maximum(T.n_px.to_numpy(), 1)
                            d3 += context_table(T, prec, "mean_ctx").assign(unit=u.key, layer=geo_name).to_dict("records")
        return rows, d3, seg

    # =========================================================================== E
    def e_select(self, export: bool | None = None) -> dict:
        return self.section("selection", lambda: self._e(export))

    def _e(self, export: bool | None) -> dict:
        from . import boost_audit as BA

        sc = self.cfg["selection"]
        fp = self.fp("selection", sc, self.w2_inputs(), [(self.read_json(s, "manifest") or {}).get("fingerprint")
                                                         for s in ("benchmark", "geology", "robustness", "hybrid")])
        t0 = self._begin("selection", fp)
        parity = BA.parity_by_feature(self.read_json("preflight", "train_inference_parity"))
        champ = self.champion()
        bn = self.load_table("nested_selection", "benchmark")
        w2m = bn[bn.space == "w2_space|blend+rep+frac"] if len(bn) else bn
        if len(w2m):
            champ["disc_dti_matched"] = float(w2m.disc_dti.iloc[0])
        rows = [{"name": "fallback_fusion_thin", "role": "fallback", **_pick(bn, "fallback_published")},
                {"name": "p6_stack (W2)", "role": "champion", **{k: champ.get(k) for k in ("disc_dti", "disc_worst",
                                                                                          "geo_dti")},
                 "disc_dti_matched": champ.get("disc_dti_matched"), "source": champ["source"]}]
        best = None
        for st in ("geology", "robustness", "hybrid"):
            d = self.read_json(st, "decision") or {}
            if d.get("chosen") and d["chosen"] != d.get("base"):
                s = d["chosen_summary"]
                spec = d.get("spec") or {}
                names = list((spec.get("base", spec) if spec.get("kind") == "hybrid" else spec).get("features") or [])
                par_ok, par_why = BA.parity_ok(parity, names)
                c = {"name": d["chosen"], "role": "candidate", "stage": st, **{k: s.get(k) for k in
                                                                              ("disc_dti", "disc_worst", "geo_dti",
                                                                               "geo_worst")},
                     "params_all": s.get("params_all"), "spec": spec,
                     "provenance_ok": self._provenance_ok(st) and par_ok and all(
                         self._provenance_ok(b) for b in [(spec.get("base") or {}).get("stage")]
                         if b in ("geology", "robustness")), "parity": par_why,
                     "systems_gained": s.get("systems_gained"), "systems_lost": s.get("systems_lost")}
                ok, why, slot = boost_gate(c, champ, sc)
                c.update(gate="PASS" if ok else "FAIL", gate_reasons="; ".join(why), slot_worthy=slot)
                rows.append(c)
                if ok and (best is None or c["disc_dti"] > best["disc_dti"]):
                    best = c
        dec = {"champion": champ, "chosen": best["name"] if best else "p6_stack (W2)",
               "replace_champion": best is not None, "slot_worthy": bool(best and best["slot_worthy"]),
               "rules": {"disc_gt": "max(recorded, matched W2)", "worst_ge": champ.get("disc_worst"),
                         "geo_floor": sc["geo_floor"], "slot_min_gain": sc["slot_min_gain"]},
               "note": "no upload is automatic; record hashes and public scores in reports/p6/leaderboard_log.csv"}
        want = bool(sc["export"] if export is None else export)
        if best is not None and want:
            efp = self.fp("export", best.get("spec"), best.get("params_all"), self.w2_inputs(),
                          self.g02_ident(list((best.get("spec") or {}).get("features") or [])))
            log = self.read_json("selection", "exports") or {}
            prev = log.get(efp) or {}
            if prev.get("submission") and (self.p6.ws.root / "submissions" / prev["submission"]).exists():
                self.say(f"export: candidate {best['name']} already exported as {prev['submission']} -> reused "
                         "(no refit, no new file)")
                info = prev
            else:
                info = self.export(best)
                log[efp] = {k: info.get(k) for k in ("submission", "validation", "parameters", "written")}
                self.save_json(log, "selection", "exports")
            dec.update(export_fp=efp, submission=info.get("submission"), validation=info.get("validation"))
            if (info.get("validation") or {}).get("checks_failed"):
                dec.update(replace_champion=False, slot_worthy=False, chosen="p6_stack (W2)",
                           reason="submission validation failed")
        elif best is not None:
            dec["export"] = "not requested: rerun with BOOST_RUN['export'] = True (e_select(export=True))"
        self.save_table(pd.DataFrame([_flat(r) for r in rows]), "candidates", "selection")
        self.save_table(self.leaderboard_compare(dec), "leaderboard_compare", "selection")
        self.save_json(dec, "selection", "decision")
        self._finish("selection", fp, t0, {"w2": self.w2_inputs(), "parity_checked": bool(parity)},
                     {"chosen": dec["chosen"], "replace_champion": dec["replace_champion"],
                      "submission": dec.get("submission")})
        self.p6.push()
        return dec

    def _provenance_ok(self, stage: str) -> bool:
        """Stage finished under its fingerprint, every unit cross-fitted or reused, no BLOCKED feature in the choice."""
        m = self.read_json(stage, "manifest") or {}
        if m.get("status") != "done" or missing_outputs(self.root / stage, stage):
            return False
        w = (m.get("inputs") or {}).get("w2") or {}
        if w and w.get("maps_sha256") != self.w2_inputs()["maps_sha256"]:
            return False                                         # decided on W2 maps that are no longer on disk
        a = self.load_table("crossfit_audit", stage)
        if len(a) and a.status.astype(str).str.startswith("too few").any():
            return False
        return True

    def full_raw(self, names: list[str], view: str | None) -> dict:
        """Full-region features (label-free inputs, every mapped fault visible), as ``stack.full_region``."""
        p6 = self.p6
        data, sc = p6.ctx.data, p6.cfg["stack"]
        valid = np.asarray(data.valid, bool)
        maps, _ = p6.view.full_maps()
        raw = {"cnn_b0": maps[0], "cnn_b1": maps[1],
               "cnn_disagree": np.abs(ST.rank_in(maps[0], valid) - ST.rank_in(maps[1], valid))}
        want = {k: v for k, v in self.cfg["geology"]["keys"].items() if k in names}
        if want:
            self.say(f"full-region G02 maps {sorted(want)}: resumable tiles under stack/features/g02_full "
                     "(the heaviest step; G02-A dominates)")
            self.g02()
            raw.update(ST.g02_full(p6, want))
        raw.update(ST.pf_full(p6))
        raw.update(ST.known_features(np.asarray(data.labels, bool), valid, 0.1, float(sc["known_cap_km"]),
                                     float(sc["known_density_radius_km"]), float(sc["endpoint_max_km"])))
        raw = self.view_raw(None, view, raw, valid)
        miss = [n for n in names if n not in raw]
        if miss:
            raise FileNotFoundError(f"full-region features missing: {miss} (export refused; never zero-filled)")
        return {**raw, "_maps": maps}

    def export(self, cand: dict) -> dict:
        """Refit on every primary-task row, score the full grid, apply the nested (blend, rep, frac) of the primary
        task (``area_frac_source: matched``), D op when the candidate is a hybrid, validated GeoTIFF (never
        overwritten), log row with sha256 (public score typed by hand)."""
        import joblib

        from ..gems3 import representation as R
        from ..gems3 import submission as SUB
        from . import probes

        p6, sc = self.p6, p6_stack_cfg(self.p6)
        spec = cand["spec"]
        lg = spec.get("base", spec) if spec.get("kind") == "hybrid" else spec
        names, view, drop = list(lg["features"]), lg.get("view"), lg.get("dropout")
        data = p6.ctx.data
        valid = np.asarray(data.valid, bool)
        raw = self.full_raw(names, view)
        maps = raw.pop("_maps")
        T, _ = self.rows(names, view)
        tr = (T.task == self.prim).to_numpy()
        X = T.loc[tr, names].to_numpy(np.float32)
        if drop:
            cols = [names.index(c) for c in self.cfg["robustness"]["known_group"] if c in names]
            X = group_dropout(X, cols, float(drop["p"]), np.random.default_rng(int(p6.cfg["seed"])))
        m = ST.make_model(sc["lgbm"], int(p6.cfg["seed"]))
        m.fit(X, T.y.to_numpy()[tr], sample_weight=T.w.to_numpy()[tr])
        mp = self.root / "selection" / "models" / f"{cand['name']}_full.joblib"
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(m, mp)
        score = ST.predict_unit(m, raw, valid, names)
        p6.ws.save_raster({"score": score}, f"{self.sub('selection')}/maps/{cand['name']}_full_score.tif", data.grid)
        pa = (cand.get("params_all") or {}).get(self.prim) or {}
        if self.cfg["selection"]["area_frac_source"] != "matched":
            raise ValueError("only area_frac_source = matched is implemented (rep and frac of one nested choice)")
        lam, rep, frac = float(pa["blend"]), str(pa["rep"]), float(pa["frac"])
        w = float(self.sel[self.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
        F = R.fuse(R.rank01(maps[0], valid), R.rank01(maps[1], valid), w)
        s = ((1 - lam) * F + lam * R.rank01(np.nan_to_num(score, nan=0.0), valid)).astype(np.float32)
        el = SUB.eligible_mask(data, 0, "eligible")
        reps, sigma = p6.m9.sh["baseline"]["representations"], float(p6.m9.sh["evaluation"]["ridge_sigma_px"])

        def ap(x, f):
            return R.apply(reps[rep], x, valid, f, known=data.labels, sigma_px=sigma, eligible=el) > 0

        full = ap(s, frac)
        if spec.get("kind") == "hybrid":
            hc = self.cfg["hybrid"]
            fbe, geo, gname = self._d_layers(None, F, raw, valid)
            if spec["op"] == "D1":
                q = controlled_union(ap(s, frac * (1 - float(hc["union_fallback_share"]))), full, ap(F, frac), fbe, s,
                                     int(full.sum()), int(hc["union_tolerance_px"]))
            else:
                q, _ = connectivity_select(ap(s, frac * float(hc["pool_multiplier"])), s, int(full.sum()),
                                           geo if gname == "g02c" else None, weights=hc["weights"],
                                           short_px=int(hc["short_px"]), short_penalty=float(hc["short_penalty"]),
                                           ep_radius=int(hc["endpoint_radius_px"]))
        else:
            q = full
        q = q.astype(np.float32)
        led = [SUB.area_ledger(q, data, 0, f"boost {cand['name']} {rep} blend {lam} frac {frac:g}")]
        # r3: the single export contract (plan -> file -> audit)
        from . import export_parity as EP
        from .boost_audit import file_hash

        rec = EP.build_recipe(
            candidate=f"boost_{cand['name']}", kind="boost_candidate",
            policy={"selection_id": f"{cand.get('stage')}:{cand['name']}|nested {self.prim}|all_zones",
                    "space": "blend+rep+frac", "objective": f"nested {self.prim} (boost protocol)", "rep": rep,
                    "frac": frac, "blend": lam},
            model={"kind": spec.get("kind", "lgbm"), "features": names, "view": view, "dropout": drop,
                   "model_file": mp.name, "refit": True, **({"op": spec.get("op")} if spec.get("kind") == "hybrid"
                                                            else {})},
            sources={"w2_rows": self.w2_inputs()["rows_sha256"], "model": file_hash(mp, 1e9),
                     "full_score": file_hash(self.root / "selection" / "maps" / f"{cand['name']}_full_score.tif", 1e9)},
            feature_schema=names,
            preprocessing={"ranking": "rank01 inside data.valid", "fusion_w": w, "sigma_px": sigma,
                           "representation": reps[rep], "eligibility": "gems3.submission.eligible_mask(data, 0, "
                           "'eligible')", "known": "data.labels passed to R.apply", "ring_px": 0, "view": view})
        EP.save_plan(p6, rec, f"{self.sub('selection')}/export_plans")
        info = EP.write_validated(p6, rec, q, led, float(self.cfg["preflight"]["budget_tolerance"]))
        if (info.get("audit") or {}).get("status") != "PASS":
            info.setdefault("validation", {}).setdefault("checks_failed", []).extend(
                (info.get("audit") or {}).get("failed") or ["export audit not PASS"])
            return info
        path = p6.ws.root / "submissions" / info["submission"]
        if not info.get("reused"):
            row = probes._row(p6, "BOOST", path, info, self.sel, f"boost {cand['name']} (stage {cand.get('stage')})")
            row.update(offline_disc_dti=cand.get("disc_dti"), offline_geo_dti=cand.get("geo_dti"))
            probes.update_logs(p6, pd.DataFrame([row]))
        return info

    def leaderboard_compare(self, dec: dict) -> pd.DataFrame:
        """Fallback, the W2 submission and the boosted file side by side (sha256, offline numbers, public score)."""
        from ..artifacts.atomic import sha256_file
        from . import probes

        log = probes.read_log(self.p6)
        pub = dict(zip(log.sha256.astype(str), pd.to_numeric(log.public_score, errors="coerce"))) if len(log) else {}
        M = self.load_table("matched_comparison", "benchmark")
        bn = self.load_table("nested_selection", "benchmark")

        def cell(cand, rep, frac, task):
            m = M[(M.candidate == cand) & (M.rep == rep) & (np.isclose(M.frac, frac)) & (M.task == task)] if len(M) else M
            return float(m.dti.iloc[0]) if len(m) else np.nan

        w2c = self.w2_inputs().get("candidate") or {}
        pa = (w2c.get("params_all") or {})
        rows = []
        fbp = self.p6.m9_path(f"submissions/{self.p6.cfg['probes']['fallback_file']}")
        rows.append({"role": "fallback", "file": fbp.name, "path": fbp, **_pick(bn, "fallback_published"),
                     "rep_frac": "published per-zone choice"})
        sf = w2c.get("submission_file")
        w2p = self.p6.ws.root / "submissions" / sf if sf else None
        rep, frac = (pa.get(self.prim) or {}).get("rep"), (pa.get("geo") or {}).get("frac")
        lam = float((pa.get(self.prim) or {}).get("blend", 1.0))
        cw = "w2" if lam == 1.0 else ("hybrid" if lam == float(self.cfg["benchmark"]["hybrid_blend"]) else None)
        ok = frac is not None and cw is not None
        rows.append({"role": "w2_champion", "file": sf, "path": w2p, "disc_dti": w2c.get("disc_dti"),
                     "geo_dti": w2c.get("geo_dti"),
                     "rep_frac": f"{rep} @ {frac}, blend {lam} (rep from disc, frac from geo: W2 full_region)",
                     "disc_dti_exported_cell": cell(cw, rep, float(frac), self.prim) if ok else np.nan,
                     "geo_dti_exported_cell": cell(cw, rep, float(frac), "geo") if ok else np.nan})
        if dec.get("submission"):
            c = dec["chosen"]
            pa_b = next((r for r in [self.read_json(s, "decision") for s in ("geology", "robustness", "hybrid")]
                         if r and r.get("chosen") == c), {}).get("chosen_summary", {}).get("params_all", {})
            rows.append({"role": "boosted", "file": dec["submission"], "path": self.p6.ws.root / "submissions" /
                         dec["submission"], "rep_frac": json.dumps(pa_b.get(self.prim))})
        for r in rows:
            p = r.pop("path", None)
            r["sha256"] = sha256_file(p) if p is not None and Path(p).exists() else None
            r["public_score"] = pub.get(str(r["sha256"]), np.nan)
            r["exists"] = bool(p is not None and Path(p).exists())
        return pd.DataFrame(rows)


# =============================================================================== small helpers
_METHOD = {"benchmark": "a_benchmark", "geology": "b_geology", "robustness": "c_robustness", "hybrid": "d_hybrid",
           "selection": "e_select"}


def p6_stack_cfg(p6) -> dict:
    return p6.cfg["stack"]


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    if isinstance(o, (Path, np.ndarray)):
        return str(o) if isinstance(o, Path) else o.tolist()
    return str(o)


def _flat(d: dict) -> dict:
    """One CSV row: nested dicts / lists as JSON strings."""
    return {k: (json.dumps(v, default=_js) if isinstance(v, (dict, list, tuple)) else v) for k, v in d.items()}


def _pick(S: pd.DataFrame, cand: str) -> dict:
    r = S[S.candidate == cand] if len(S) else S
    if not len(r):
        return {}
    r = r.iloc[0]
    return {k: (float(r[k]) if k in r and pd.notna(r[k]) else None) for k in ("disc_dti", "disc_worst", "geo_dti",
                                                                              "geo_worst")}


def _fp_increase(G: pd.DataFrame, per: pd.DataFrame, name: str, base: str) -> float:
    """Relative change of the false-positive weight of ``name`` vs ``base`` at ``name``'s held-out choices."""
    fa = fb = 0.0
    for r in per.itertuples():
        m = (G.task == r.task) & (G.zone == r.zone) & np.isclose(G.blend.astype(float), float(r.blend)) & \
            (G.rep == r.rep) & np.isclose(G.frac.astype(float), float(r.frac))
        fa += float(G.loc[m & (G.candidate == name), "fp_w"].sum())
        fb += float(G.loc[m & (G.candidate == base), "fp_w"].sum())
    return fa / fb - 1.0 if fb > 0 else np.nan
