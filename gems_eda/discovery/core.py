"""Core of the exhaustive geological discovery EDA: configuration, figure registry with computed interpretation
records, and dependence-aware statistics.

Every figure is saved as PNG + SVG + plotting data CSV + an interpretation JSON whose conclusion is DERIVED from the
measured statistics passed by the figure function (``Result``), never written in advance. Evidence statuses:
SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE for tests, DESCRIPTIVE for maps and inventories.

Statistics: fault-vs-matched-control effects are Cliff's delta (2 AUC - 1) with a CLUSTER bootstrap over fault
systems (mapped-fault connected components): a system's stations and their matched controls are resampled together,
so neighbouring pixels are never treated as independent. NOT_SUPPORTED requires the whole CI inside the equivalence
band +-``min_effect`` (a weak, wide result is INCONCLUSIVE, never "no relevance").
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE, DESCRIPTIVE = "SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "DESCRIPTIVE"
NOT_AVAILABLE = "NOT_AVAILABLE"
PALETTE = {"magnetic": "#B03A2E", "gravity": "#1F618D", "terrain": "#4DAA50", "basement": "#6C3483",
           "conductivity": "#117A65", "strain": "#CA6F1E", "seismicity": "#7F8C8D", "faults": "#202124",
           "w2": "#E8B33F", "control": "#9E9E9E", "other": "#555555"}
PROFILES = {"QUICK_LOOK": {"max_fault_stations": 600, "n_boot": 100, "scales_km": [0.5, 2, 10]},
            "FULL_EDA": {"max_fault_stations": 5000, "n_boot": None, "scales_km": None}}
DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "main", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "local_root": "/content/gems_work", "mode": "FULL_EDA",
    "stages": {"inventory": True, "global_geology": True, "modality_analysis": True, "fault_geometry": True,
               "fault_profiles": True, "spatial_scales": True, "crossmodal_evidence": True, "label_bias": True,
               "regional_shift": True, "model_forensics": True, "unmapped_candidates": True, "dem_pilot": False,
               "scientific_synthesis": True},
    "resolution": "native", "pixel_size_m": 100, "scales_km": [0.2, 0.5, 1, 2, 5, 10, 20],
    "sampling": {"seed": 2026, "max_fault_systems": None, "matched_controls_per_fault": 5,
                 "profile_half_width_km": 10, "along_strike_spacing_m": 500, "n_regional_examples": 48,
                 "n_model_error_examples": 48, "control_distance_km": [2.0, 10.0],
                 "control_sensitivity_km": [[1.0, 10.0], [5.0, 20.0]]},
    "figures": {"enabled": True, "format": ["png", "svg"], "dpi": 220, "save_plot_data": True,
                "max_signals_per_panel": 4, "create_comparison_mosaics": True, "create_summary_dashboards": True},
    "statistics": {"bootstrap_units": "fault_system", "n_bootstrap": 500, "spatial_controls": True,
                   "confidence_interval": 0.95, "min_effect": 0.10, "min_region_stations": 40},
    "models": {"load_frozen_B0": True, "load_frozen_B1": True, "load_frozen_W2": True, "train_models": False},
    "dem": {"enabled": False, "cached_only": True, "pilot_only": True},
    "reuse_cached": True, "save_all_artifacts": True, "sync_to_drive": True, "run_synthetic_tests": True,
}
MODES = ("QUICK_LOOK", "FULL_EDA", "RESUME")


class EDAConfigError(ValueError):
    pass


def load_colab_config(path) -> dict:
    """configs/eda/colab_full.yaml (or another YAML) as a notebook CONFIG dict (validated by eda_config later)."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def eda_config(CONFIG: dict | None) -> dict:
    C = CONFIG or {}
    c = {**DEFAULTS, **C}
    for k in ("stages", "sampling", "figures", "statistics", "models", "dem"):
        c[k] = {**DEFAULTS[k], **(C.get(k) or {})}
    c["mode"] = str(c["mode"]).upper()
    errs = []
    if c["mode"] not in MODES:
        errs.append(f"mode {c['mode']!r} not in {MODES}")
    if c["models"]["train_models"]:
        errs.append("models.train_models=True is refused: the EDA reuses frozen predictions only")
    if c["resolution"] != "native" or int(c["pixel_size_m"]) != 100:
        errs.append("the EDA runs on the native 100 m grid (resolution='native', pixel_size_m=100)")
    if c["dem"]["enabled"] and not c["dem"]["pilot_only"]:
        errs.append("dem: only the pilot is allowed (pilot_only=True)")
    if errs:
        raise EDAConfigError("; ".join(errs))
    prof = PROFILES["QUICK_LOOK" if c["mode"] == "QUICK_LOOK" else "FULL_EDA"]
    c["max_fault_stations"] = prof["max_fault_stations"]
    c["n_boot"] = prof["n_boot"] or int(c["statistics"]["n_bootstrap"])
    c["scales_used_km"] = [float(s) for s in (prof["scales_km"] or c["scales_km"])]
    if c["mode"] == "RESUME":
        c["reuse_cached"] = True
    return c


# =============================================================================== statistics
def auc(pos, neg) -> float:
    from scipy.stats import rankdata

    p, n = np.asarray(pos, float), np.asarray(neg, float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if not len(p) or not len(n):
        return np.nan
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def cliffs_delta(pos, neg) -> float:
    a = auc(pos, neg)
    return 2 * a - 1 if np.isfinite(a) else np.nan


def cluster_effect(f_val, f_grp, c_val, c_grp, n_boot: int, seed: int, ci: float = 0.95) -> dict:
    """Cliff's delta (fault vs control) with a cluster bootstrap over groups (fault systems). Controls carry the group
    of the fault station they are matched to, so pairs are resampled together."""
    f_val, c_val = np.asarray(f_val, float), np.asarray(c_val, float)
    f_grp, c_grp = np.asarray(f_grp), np.asarray(c_grp)
    est = cliffs_delta(f_val, c_val)
    groups = np.unique(f_grp)
    out = {"effect": est, "ci_lo": np.nan, "ci_hi": np.nan, "n_groups": int(len(groups)),
           "n_fault": int(np.isfinite(f_val).sum()), "n_control": int(np.isfinite(c_val).sum())}
    if len(groups) < 5 or not np.isfinite(est):
        return out
    rng = np.random.default_rng(seed)
    fi = {g: np.flatnonzero(f_grp == g) for g in groups}
    ci_ = {g: np.flatnonzero(c_grp == g) for g in groups}
    draws = []
    for _ in range(int(n_boot)):
        gs = rng.choice(groups, len(groups), replace=True)
        a = np.concatenate([fi[g] for g in gs])
        b = np.concatenate([ci_.get(g, np.zeros(0, int)) for g in gs])
        draws.append(cliffs_delta(f_val[a], c_val[b]))
    d = np.asarray(draws)
    d = d[np.isfinite(d)]
    if d.size:
        lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
        out.update(ci_lo=float(lo), ci_hi=float(hi))
    return out


def cluster_ci(values, groups, n_boot: int, seed: int, stat=np.nanmedian, ci: float = 0.95) -> dict:
    """Statistic of one population with a cluster-bootstrap CI over groups."""
    v, g = np.asarray(values, float), np.asarray(groups)
    est = float(stat(v)) if np.isfinite(v).any() else np.nan
    gs = np.unique(g)
    out = {"estimate": est, "ci_lo": np.nan, "ci_hi": np.nan, "n": int(np.isfinite(v).sum()), "n_groups": len(gs)}
    if len(gs) < 5:
        return out
    rng = np.random.default_rng(seed)
    idx = {k: np.flatnonzero(g == k) for k in gs}
    d = [stat(v[np.concatenate([idx[k] for k in rng.choice(gs, len(gs))])]) for _ in range(int(n_boot))]
    lo, hi = np.nanquantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    out.update(ci_lo=float(lo), ci_hi=float(hi))
    return out


def status_of(effect: float, lo: float, hi: float, min_effect: float) -> str:
    """SUPPORTED: CI excludes the equivalence band on one side; NOT_SUPPORTED: CI inside +-min_effect;
    INCONCLUSIVE otherwise (including a missing CI)."""
    if not all(np.isfinite([effect, lo, hi])):
        return INCONCLUSIVE
    if lo > min_effect or hi < -min_effect:
        return SUPPORTED
    if lo > -min_effect and hi < min_effect:
        return NOT_SUPPORTED
    return INCONCLUSIVE


def robustness(per_region: pd.DataFrame, overall: float, min_effect: float) -> dict:
    """Same-sign share of regional effects (regions with enough stations) and a stability label."""
    r = per_region.dropna(subset=["effect"]) if len(per_region) else per_region
    if not len(r) or not np.isfinite(overall):
        return {"n_regions": 0, "same_sign_share": np.nan, "stability": "NOT_EVALUATED"}
    s = np.sign(overall)
    share = float(((np.sign(r.effect) == s) & (r.effect.abs() >= min_effect / 2)).mean())
    lab = "STABLE" if share >= 2 / 3 else "MIXED" if share >= 1 / 3 else "UNSTABLE"
    return {"n_regions": int(len(r)), "same_sign_share": share, "stability": lab}


def axial_mean_deg(az, w=None) -> tuple[float, float]:
    a = np.radians(2 * np.asarray(az, float) % 360)
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    C, S = (w * np.cos(a)).sum() / max(w.sum(), 1e-12), (w * np.sin(a)).sum() / max(w.sum(), 1e-12)
    return float(np.degrees(np.arctan2(S, C)) / 2 % 180), float(np.hypot(C, S))


def axial_diff_deg(a, b) -> np.ndarray:
    d = np.abs((np.asarray(a, float) - np.asarray(b, float)) % 180)
    return np.minimum(d, 180 - d)


# =============================================================================== figure registry
@dataclass
class Spec:
    """What a figure compares and how its result is read. ``outcomes`` / ``implications`` are keyed by the evidence
    status the measured ``Result`` produces (or DESCRIPTIVE)."""
    fid: str
    name: str
    title: str
    part: str
    question: str
    comparison: str
    why: str
    measurement: str
    outcomes: dict
    implications: dict
    limitations: str
    priority: int = 2                    # 1 = core to the modeling decision (executive-report candidate)


@dataclass
class Result:
    status: str
    summary: str                         # numbers-based sentence built from the computation
    stats: dict = field(default_factory=dict)
    n: int | None = None


class Registry:
    def __init__(self, E):
        self.E = E
        self.rows: list[dict] = []
        p = E.ws.path("figures/FIGURE_CATALOG.csv")
        if p.exists() and p.stat().st_size:
            self.rows = pd.read_csv(p).to_dict("records")

    def save(self, spec: Spec, fig, data: pd.DataFrame | None, result: Result, extra: dict | None = None) -> dict:
        import matplotlib.pyplot as plt

        from gems_bench.artifacts.atomic import atomic_save

        E = self.E
        stem = f"{spec.fid}_{spec.name}"
        dash = spec.fid.startswith("D")
        sub = "figures/dashboards" if dash else "figures"
        files = {}
        if fig is not None and E.c["figures"]["enabled"]:
            for fmt in E.c["figures"]["format"]:
                p = E.ws.path(f"{sub}/{stem}.{fmt}")
                atomic_save(p, lambda t, f=fig, fm=fmt: f.savefig(t, dpi=int(E.c["figures"]["dpi"]),
                                                                 bbox_inches="tight", format=fm))
                files[fmt] = p.relative_to(E.ws.root).as_posix()
        if fig is not None:
            plt.close(fig)
        if data is not None and E.c["figures"]["save_plot_data"]:
            rel = f"figures/data/{stem}.csv"
            E.ws.save_table(data, rel)
            files["data"] = rel
        interp = {"figure_id": spec.fid, "title": spec.title, "part": spec.part, "question": spec.question,
                  "comparison": spec.comparison, "why_geologically_meaningful": spec.why,
                  "measurement": spec.measurement, "possible_findings": spec.outcomes,
                  "measured": {"status": result.status, "n": result.n, **_jsonable(result.stats)},
                  "interpretation": f"{spec.outcomes.get(result.status, '')} Measured: {result.summary}".strip(),
                  "modeling_implication": spec.implications.get(result.status, spec.implications.get("ANY", "")),
                  "limitations": spec.limitations, "artifacts": files, **(extra or {}),
                  "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
        E.ws.save_json(interp, f"figures/interpretations/{spec.fid}.json")
        row = {"figure_id": spec.fid, "name": spec.name, "title": spec.title, "part": spec.part,
               "status": result.status, "priority": spec.priority, "n": result.n, "summary": result.summary[:300],
               "png": files.get("png"), "svg": files.get("svg"), "data": files.get("data"),
               "interpretation": f"figures/interpretations/{spec.fid}.json", "stage": E.current_stage}
        self.rows = [r for r in self.rows if r["figure_id"] != spec.fid] + [row]
        E.ws.save_table(pd.DataFrame(self.rows).sort_values("figure_id"), "figures/FIGURE_CATALOG.csv")
        return row

    def skip(self, spec: Spec, reason: str) -> dict:
        """A figure that could not be produced: recorded (catalog + interpretation), no image, no placeholder data."""
        return self.save(spec, None, None, Result(NOT_AVAILABLE, reason))


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else
                                 str(o)))


def effect_result(eff: dict, min_effect: float, rob: dict | None = None, label: str = "",
                  groups: str = "fault systems", items: tuple = ("fault", "control stations")) -> Result:
    st = status_of(eff.get("effect", np.nan), eff.get("ci_lo", np.nan), eff.get("ci_hi", np.nan), min_effect)
    s = (f"{label} Cliff's delta {eff.get('effect', np.nan):+.3f} [{eff.get('ci_lo', np.nan):+.3f}, "
         f"{eff.get('ci_hi', np.nan):+.3f}] over {eff.get('n_groups')} {groups} "
         f"({eff.get('n_fault')} {items[0]} / {eff.get('n_control')} {items[1]})")
    if rob:
        s += f"; regional same-sign share {rob.get('same_sign_share')} ({rob.get('stability')})"
    return Result(st, s.strip(), {**eff, **(rob or {})}, eff.get("n_fault"))
