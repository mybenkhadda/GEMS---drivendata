"""M6.1 / M6.2: read-only artifact audit and paired geographic / pseudo-discovery comparison of FULL experiments.

Audit (nothing is trained, written into experiment folders or staged here)::

    artifact_inventory(root, specs, ...)   one row per model: experiment folder, maps, units, provenance, status
    provenance_table / compatibility_table / missing_table
    require_complete(inv, models)          raises M6ArtifactError naming every missing OOF / discovery map

Status: ``COMPLETE`` (FULL run, every unit, OOF + every discovery map on the reference grid and protocol),
``PARTIAL`` (usable OOF, something missing), ``INCOMPATIBLE`` (other grid / folds / splits / protocol),
``MISSING`` (no FULL folder or no OOF), ``INVALID`` (unreadable manifest or raster).

Paired evaluation (on one consolidated scorecard, so every model shares folds, splits, budgets, policies)::

    fold_metrics(sc) / discovery_metrics(sc)      held-out (cross-fitted area) components per fold / split
    paired_deltas(tab, ref, unit_col)             per-unit differences, direction, effect size (descriptive)
    block_bootstrap(...)                          paired spatial-block bootstrap of the held-out DTI difference

With three folds and three splits nothing here is a population-level significance test. The bootstrap resamples
the folds' own spatial blocks (the independent sampling unit of the geographic design) within each held-out
zone at the cross-fitted operating point: it describes how much the difference depends on which blocks were
scored, conditional on the protocol. Pixels are never treated as independent observations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from ..metric import DTI_ALPHA, DTI_BETA, DTI_EPS, DTI_RADIUS_M

STATUSES = ("COMPLETE", "PARTIAL", "INCOMPATIBLE", "MISSING", "INVALID")
_USABLE = ("COMPLETE", "PARTIAL")


class M6ArtifactError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    stem: str            # experiment stem (configs/experiments/<stem>.yaml) or baseline name
    model: str           # map name in the zoo (model_name)
    kind: str = "experiment"   # experiment | baseline


# =============================================================================== small readers
def _json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _yaml(p: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def file_fp(p: Path) -> str:
    """Size + sha256 of the first MiB (``evaluation.protocol`` convention): cheap on Drive FUSE."""
    from .protocol import _file_fp

    return _file_fp(Path(p))


def raster_header(p: Path) -> dict:
    """Grid of a GeoTIFF without reading pixels; ``{"error": ...}`` when unreadable."""
    try:
        import rasterio

        with rasterio.open(p) as src:
            return {"height": src.height, "width": src.width, "count": src.count, "dtype": src.dtypes[0],
                    "crs": src.crs.to_string() if src.crs else "", "transform": tuple(round(v, 9) for v in src.transform)[:6]}
    except Exception as e:                                          # rasterio raises its own error types
        return {"error": f"{type(e).__name__}: {e}"}


def grid_signature(grid) -> dict:
    return {"height": int(grid.height), "width": int(grid.width),
            "crs": grid.crs.to_string() if hasattr(grid.crs, "to_string") else str(grid.crs or ""),
            "transform": tuple(round(v, 9) for v in grid.transform)[:6]}


def same_grid(header: dict, ref: dict) -> bool:
    if "error" in header:
        return False
    return (header["height"], header["width"]) == (ref["height"], ref["width"]) and \
        np.allclose(header["transform"], ref["transform"], atol=1e-6) and \
        (not ref.get("crs") or not header.get("crs") or header["crs"] == ref["crs"])


class RegistryView:
    """Read-only view of ``model_registry/registry.csv`` (``ModelRegistry`` creates folders; the audit must not)."""

    def __init__(self, root: Path):
        p = Path(root) / "registry.csv"
        self.table = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["name", "status"])

    def status(self, name: str) -> str | None:
        r = self.table[self.table.name == name] if "name" in self.table else self.table.iloc[0:0]
        return None if r.empty else str(r.iloc[-1]["status"])


def feature_version(run: Path) -> str:
    """sha12 of the sorted input channels in ``reports/spec.json`` (``""`` when the spec is absent)."""
    spec = _json(run / "reports" / "spec.json")
    if not spec or "channels" not in spec:
        return ""
    return hashlib.sha256(json.dumps(sorted(spec["channels"])).encode()).hexdigest()[:12]


# =============================================================================== experiment folders
def resolve_model_name(stem: str, code_root: Path | None = None) -> str:
    """``model_name`` of a committed experiment YAML (leaf file; children always set it), else the stem."""
    from ..paths import config_path

    p = (Path(code_root) / "configs" / "experiments" / f"{stem}.yaml") if code_root else \
        config_path("experiments", f"{stem}.yaml")
    return str(_yaml(p).get("model_name") or stem)


def full_experiment_dirs(exp_root: Path, stem: str) -> list[Path]:
    """FULL folders of ``stem``: ``<stem>__<hash8>`` (fidelity folders carry ``__screen__`` / ``__validate__``)."""
    out = []
    for d in sorted(Path(exp_root).glob(f"{stem}__*")):
        parts = d.name.split("__")
        if len(parts) == 2 and (d / "manifest.json").exists():
            out.append(d)
    return out


def _units_done(m: dict, prefix: str) -> list[int]:
    out = []
    for u, v in (m.get("units") or {}).items():
        if u.startswith(prefix) and u[len(prefix):].isdigit() and (v or {}).get("status") == "done":
            out.append(int(u[len(prefix):]))
    return sorted(out)


def _pick_run(dirs: list[Path], model: str) -> tuple[Path | None, list[str]]:
    """Folder whose manifest maps ``model``, preferring completed evaluation, then most units, then newest."""
    best, key_best, notes = None, None, []
    for d in dirs:
        m = _json(d / "manifest.json") or {}
        has = model in (m.get("maps") or {})
        ev = (m.get("stages", {}).get("evaluate") or {}).get("status") == "done"
        key = (has, ev, len(m.get("units") or {}), m.get("updated", ""))
        notes.append(f"{d.name}: maps={has} evaluate={ev} units={len(m.get('units') or {})}")
        if key_best is None or key > key_best:
            best, key_best = d, key
    return best, notes


def experiment_row(exp_root: Path, spec: ModelSpec, *, ref_grid: dict | None, n_folds: int, n_splits: int,
                   current_protocol: dict | None = None, registry=None) -> tuple[dict, list[dict]]:
    """(inventory row, missing-artifact rows) for one Phase-5 experiment model; read-only."""
    row = {"stem": spec.stem, "model": spec.model, "kind": spec.kind, "status": "MISSING", "reasons": ""}
    miss: list[dict] = []
    dirs = full_experiment_dirs(exp_root, spec.stem)
    if not dirs:
        miss.append({"model": spec.model, "artifact": "FULL experiment folder", "path": f"experiments/{spec.stem}__*",
                     "required": True})
        row["reasons"] = "no FULL experiment folder"
        return row, miss
    d, notes = _pick_run(dirs, spec.model)
    m = _json(d / "manifest.json")
    row.update(experiment_id=d.name, run_dir=str(d), candidates=" | ".join(notes))
    if m is None:
        row.update(status="INVALID", reasons="unreadable manifest.json")
        return row, miss
    cfg = _yaml(d / "config.yaml")
    prof = str((cfg.get("fidelity") or {}).get("profile") or (m.get("fidelity") or {}).get("profile") or "FULL").upper()
    proto = m.get("protocol") or {}
    maps = (m.get("maps") or {}).get(spec.model) or {}
    oof_rel = maps.get("oof")
    disc = {int(k): v for k, v in (maps.get("disc") or {}).items()}
    oof = d / oof_rel if oof_rel else None
    folds_done, splits_done = _units_done(m, "cv_fold"), _units_done(m, "disc")
    ck = d / "checkpoints"
    finals = sorted(ck.glob("*_model.pt")) if ck.exists() else []
    stages = {s: (m.get("stages", {}).get(s) or {}).get("status", "pending") for s in ("prepare", "train", "evaluate", "export")}
    row.update(profile=prof, config_hash=m.get("config_hash", ""), git_commit=str(m.get("git_commit", "")),
               package_version=m.get("package_version", ""), seed=m.get("seed"), feature_version=feature_version(d),
               folds_dir=str(cfg.get("validation", {}).get("folds_dir") or "default"),
               discovery_dir=str(cfg.get("validation", {}).get("discovery_dir") or "default"),
               folds_fp=proto.get("folds_fp", ""), discovery_fp=proto.get("discovery_fp", ""),
               protocol_hash=proto.get("protocol_hash", ""), match_frac=proto.get("match_frac"),
               area_selection=proto.get("selection", ""), policies=",".join(proto.get("policies", []) or []),
               calibration="raw probabilities; binary area budget cross-fitted at evaluation (calibration.area_policy)",
               dataset_features_bytes=((m.get("dataset") or {}).get("features") or {}).get("bytes"),
               folds_done=",".join(map(str, folds_done)), splits_done=",".join(map(str, splits_done)),
               n_checkpoints=len(finals), checkpoint_mb=round(sum(p.stat().st_size for p in finals) / 2 ** 20, 1),
               **{f"stage_{k}": v for k, v in stages.items()},
               oof=str(oof) if oof else "", n_disc_maps=sum(1 for p in disc.values() if (d / p).exists()),
               scorecard=(d / "metrics" / "scorecard.csv").exists(), scoreboard=(d / "metrics" / "scoreboard.csv").exists(),
               fold_metrics=(d / "metrics" / "fold_metrics.csv").exists(),
               disc_per_split=(d / "metrics" / "pseudo_discovery_per_split.csv").exists(),
               registry_status=(registry.status(spec.model) if registry is not None else None) or "")
    for k in range(n_splits):
        row[f"disc{k}"] = str(d / disc[k]) if k in disc else ""
    reasons, status = [], "COMPLETE"
    if prof != "FULL":
        status, reasons = "INCOMPATIBLE", [f"profile {prof} (only FULL runs are eligible)"]
    if oof is None or not oof.exists():
        miss.append({"model": spec.model, "artifact": "OOF prediction", "path": str(oof or d / "predictions/oof.tif"),
                     "required": True})
        row.update(status="MISSING", reasons="; ".join(reasons + ["OOF map absent"]))
        return row, miss
    hdr = raster_header(oof)
    row["oof_fp"] = file_fp(oof)
    if "error" in hdr:
        row.update(status="INVALID", reasons=f"OOF unreadable: {hdr['error']}")
        return row, miss
    if ref_grid is not None and not same_grid(hdr, ref_grid):
        status = "INCOMPATIBLE"
        reasons.append(f"OOF grid {hdr['height']}x{hdr['width']} differs from the native grid")
    for k in range(n_splits):
        p = d / disc[k] if k in disc else None
        if p is None or not p.exists():
            miss.append({"model": spec.model, "artifact": f"discovery map split {k}", "path": str(p or ""),
                         "required": True})
            reasons.append(f"discovery split {k} missing")
            status = "PARTIAL" if status == "COMPLETE" else status
            continue
        h = raster_header(p)
        if "error" in h:
            status = "INVALID"
            reasons.append(f"discovery split {k} unreadable")
        elif ref_grid is not None and not same_grid(h, ref_grid):
            status = "INCOMPATIBLE"
            reasons.append(f"discovery split {k} grid differs")
    if len(folds_done) < n_folds:
        reasons.append(f"CV units done {folds_done} of {n_folds}")
        status = "PARTIAL" if status == "COMPLETE" else status
    if stages["evaluate"] != "done":
        reasons.append("evaluate stage not done (no persisted section-17 scorecard)")
        status = "PARTIAL" if status == "COMPLETE" else status
    if current_protocol and proto:
        for key in ("folds_fp", "discovery_fp"):
            if proto.get(key) and current_protocol.get(key) and proto[key] != current_protocol[key]:
                status = "INCOMPATIBLE"
                reasons.append(f"{key} {proto[key]} != current {current_protocol[key]}")
    row.update(status=status, reasons="; ".join(reasons))
    return row, miss


def baseline_row(root: Path, name: str, *, grid: dict | None, n_splits: int, folds_sha=None, splits_sha=None) -> tuple[dict, list[dict]]:
    """Inventory row of an established baseline (``pipeline.baselines`` status, never retrained here)."""
    from ..pipeline import baselines as B

    st = B.baseline_status(root, name, grid=grid, n_splits=n_splits, folds_sha=folds_sha, splits_sha=splits_sha)
    status = {B.AVAILABLE: "COMPLETE", B.CV_ONLY: "PARTIAL", B.INCOMPATIBLE: "INCOMPATIBLE",
              B.INTERRUPTED: "PARTIAL", B.NOT_AVAILABLE: "MISSING"}.get(st.get("status"), "INVALID")
    if name == "B0_lineament" and status == "PARTIAL":
        status = "COMPLETE"                           # label-free: the same map serves every discovery split
    row = {"stem": name, "model": name, "kind": "baseline", "status": status, "profile": "FULL",
           "reasons": str(st.get("detail", st.get("status", ""))), "run_dir": str(root / st.get("cv_dir", "")),
           "oof": str(root / st.get("cv_dir", "") / "oof.tif") if st.get("cv_dir") else ""}
    miss = [] if status in _USABLE else [{"model": name, "artifact": "baseline maps", "path": row["run_dir"],
                                          "required": False}]
    return row, miss


def artifact_inventory(root: Path, specs: list[ModelSpec], *, ref_grid: dict | None, n_folds: int = 3,
                       n_splits: int = 3, current_protocol: dict | None = None, registry=None,
                       folds_sha=None, splits_sha=None, baseline_grid: dict | None = None) -> dict[str, pd.DataFrame]:
    """{"inventory", "provenance", "compatibility", "missing"} tables for ``specs`` under ``root``; read-only."""
    rows, miss = [], []
    for s in specs:
        if s.kind == "baseline":
            r, m = baseline_row(root, s.stem, grid=baseline_grid, n_splits=n_splits, folds_sha=folds_sha,
                                splits_sha=splits_sha)
        else:
            r, m = experiment_row(root / "experiments", s, ref_grid=ref_grid, n_folds=n_folds, n_splits=n_splits,
                                  current_protocol=current_protocol, registry=registry)
        rows.append(r)
        miss += m
    inv = pd.DataFrame(rows)
    return {"inventory": inv, "provenance": provenance_table(inv), "compatibility": compatibility_table(inv),
            "missing": pd.DataFrame(miss, columns=["model", "artifact", "path", "required"])}


PROVENANCE_FIELDS = ("experiment_id", "profile", "config_hash", "git_commit", "package_version", "seed",
                     "feature_version", "folds_dir", "discovery_dir", "folds_fp", "discovery_fp", "protocol_hash",
                     "match_frac", "area_selection", "policies", "calibration", "dataset_features_bytes", "oof_fp",
                     "registry_status")


def provenance_table(inv: pd.DataFrame) -> pd.DataFrame:
    """Long table (model, field, value); one place to read every model's provenance."""
    rows = []
    for r in inv.to_dict("records"):
        for f in PROVENANCE_FIELDS:
            v = r.get(f, "")
            rows.append({"model": r["model"], "field": f, "value": "" if v is None or (isinstance(v, float) and np.isnan(v)) else v})
    return pd.DataFrame(rows, columns=["model", "field", "value"])


# fields that must agree for two models' numbers to be comparable (feature sets may legitimately differ)
BLOCKING = ("folds_fp", "discovery_fp", "protocol_hash")


def compatibility_table(inv: pd.DataFrame) -> pd.DataFrame:
    """Pairwise: same folds / splits / protocol (blocking) and same features / commit (informative)."""
    rows = []
    recs = inv.to_dict("records")
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            r = {"model_a": a["model"], "model_b": b["model"], "status_a": a["status"], "status_b": b["status"]}
            bad = []
            for f in BLOCKING + ("feature_version", "git_commit", "seed"):
                va, vb = a.get(f, ""), b.get(f, "")
                known = va not in ("", None) and vb not in ("", None) and not (isinstance(va, float) and np.isnan(va))
                r[f"same_{f}"] = (va == vb) if known else None
                if f in BLOCKING and known and va != vb:
                    bad.append(f)
            r["comparable"] = a["status"] in _USABLE and b["status"] in _USABLE and not bad
            r["blocking"] = ",".join(bad)
            rows.append(r)
    return pd.DataFrame(rows)


def require_complete(inv: pd.DataFrame, models: list[str], n_splits: int | None = None,
                     need_discovery: bool = True) -> None:
    """Fail clearly (one message listing everything) when a required map is absent or incompatible."""
    t = inv.set_index("model")
    n = n_splits if n_splits is not None else sum(1 for c in t.columns if c.startswith("disc") and c[4:].isdigit())
    bad = []
    for m in models:
        if m not in t.index:
            bad.append(f"{m}: not inventoried")
            continue
        r = t.loc[m]
        nd = r.get("n_disc_maps", 0)
        nd = int(nd) if pd.notna(nd) else 0
        if r.status not in _USABLE:
            bad.append(f"{m}: {r.status} ({r.reasons})")
        elif need_discovery and r.kind == "experiment" and nd < n:
            bad.append(f"{m}: discovery maps incomplete, {nd} of {n} ({r.reasons})")
    if bad:
        raise M6ArtifactError("M6 needs complete FULL OOF / discovery predictions:\n  " + "\n  ".join(bad) +
                              "\nRecovery: run the missing FULL units (pipe.plan_experiments(profile='FULL')) or "
                              "remove the model from M6_CONFIG['models']; M6 never trains.")


# =============================================================================== persisted vs recomputed scores
PERSISTED_COLUMNS = ("geo_dti_area", "geo_fold_min", "disc_dti_excl", "disc_dti_retained_fp", "short_recall_w",
                     "geo_far_share_at_match", "protocol_hash")


def persisted_scores(inv: pd.DataFrame) -> pd.DataFrame:
    """The section-17 row each experiment persisted for its own model (``metrics/scoreboard.csv`` + scorecard)."""
    rows = []
    for r in inv.to_dict("records"):
        d = Path(r.get("run_dir") or "")
        sb, sc = d / "metrics" / "scoreboard.csv", d / "metrics" / "scorecard.csv"
        if r.get("kind") != "experiment" or not sb.exists():
            continue
        t = pd.read_csv(sb)
        if sc.exists():
            s = pd.read_csv(sc)
            t = t.merge(s[["model"] + [c for c in s.columns if c not in t.columns]], on="model", how="left")
        row = t[t.model == r["model"]]
        if not len(row):
            continue
        rec = {"model": r["model"], "source": f"{r.get('experiment_id')}/metrics/scoreboard.csv"}
        for c in PERSISTED_COLUMNS + tuple(c for c in t.columns if c.startswith("dti_fold")):
            rec[c] = row.iloc[0].get(c, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


METRIC_ALIASES = {"geo_dti": "geo_dti_area", "worst_fold": "geo_fold_min", "discovery_dti": "disc_dti_excl",
                  "retained_fp_dti": "disc_dti_retained_fp", "short_recall": "short_recall_w"}


def reference_discrepancies(observations: dict, persisted: pd.DataFrame, recomputed: pd.DataFrame | None,
                            tol: float = 5e-5) -> pd.DataFrame:
    """Compare the brief's reference observations with persisted and recomputed numbers; never reconcile."""
    rows = []
    P = persisted.set_index("model") if len(persisted) else pd.DataFrame()
    R = recomputed.set_index("model") if recomputed is not None and len(recomputed) else pd.DataFrame()
    for model, obs in (observations or {}).items():
        for key, v in obs.items():
            col = METRIC_ALIASES.get(key, key)
            p = float(P.loc[model, col]) if model in P.index and col in P.columns and pd.notna(P.loc[model, col]) else np.nan
            q = float(R.loc[model, col]) if model in R.index and col in R.columns and pd.notna(R.loc[model, col]) else np.nan
            if np.isnan(p) and np.isnan(q):
                st = "NOT_AVAILABLE"
            elif all(np.isnan(x) or abs(x - float(v)) <= tol for x in (p, q)):
                st = "MATCH"
            else:
                st = "DIFFERENT"
            rows.append({"model": model, "metric": col, "reference_observation": float(v), "persisted": p,
                         "recomputed": q, "persisted_minus_reference": p - float(v),
                         "recomputed_minus_persisted": q - p, "status": st})
    return pd.DataFrame(rows)


# =============================================================================== consolidated per-unit tables
def fold_metrics(ctx, sc: dict) -> pd.DataFrame:
    """Held-out geographic components per (model, fold) at the area fraction chosen on the other folds."""
    from ..p3 import budget

    curves = sc["geo_curves"]
    cf = budget.crossfit_area(curves)
    cur = curves[curves["mode"] == "per_zone"]
    rows = []
    for r in cf.to_dict("records"):
        for k in ctx.fold_ids:
            f = r.get(f"frac_fold{k}")
            if f is None or (isinstance(f, float) and np.isnan(f)):
                continue
            g = cur[(cur.model == r["model"]) & (cur.zone == k) & np.isclose(cur.target_frac, f)]
            if not len(g):
                continue
            x = g.iloc[0]
            rows.append({"model": r["model"], "fold": k, "frac_selected_on_other_folds": f,
                         "selection_zones": ",".join(str(z) for z in ctx.fold_ids if z != k),
                         "dti_heldout": r[f"dti_fold{k}"], "area_frac": x.area_frac, "area_km2": x.area_km2,
                         "recall_w": x.recall_w, "recall@300m": x.get("recall@300m", np.nan),
                         "far_share": x.get("far_area_km2", np.nan) / max(x.area_km2, 1e-9),
                         "tp_w": x.tp_w, "fp_w": x.fp_w, "fn_w": x.fn_w, "n_truth": x.n_truth})
    out = pd.DataFrame(rows)
    if len(out):
        out["rank_in_fold"] = out.groupby("fold").dti_heldout.rank(ascending=False, method="min").astype(int)
    return out


def short_recall_by_fold(ctx, sc: dict, models: list[str]) -> pd.DataFrame:
    """Weighted recall of < 2 km faults per fold at the matched budget (``sc['support']``)."""
    from ..phase4_eval import label_pixel_table

    lpt = label_pixel_table(ctx)
    if "short_fault" not in lpt:
        return pd.DataFrame()
    rows = []
    for m in models:
        if m not in sc["support"]:
            continue
        t = lpt.assign(k=sc["support"][m])
        for (k, s), g in t.groupby(["fold", "short_fault"]):
            rows.append({"model": m, "fold": int(k), "stratum": s, "n_px": len(g), "weighted_recall": float(g.k.mean())})
    return pd.DataFrame(rows)


def discovery_metrics(sc: dict) -> pd.DataFrame:
    """Held-out pseudo-discovery components per (model, policy, split) at the budget chosen on the other splits."""
    ps = sc.get("disc_per_split")
    if ps is None or not len(ps):
        return pd.DataFrame()
    dc = sc["disc_curves"]
    dc = dc[dc.variant == "model"] if "variant" in dc else dc
    rows = []
    for r in ps.to_dict("records"):
        g = dc[(dc.model == r["model"]) & (dc.policy == r["policy"]) & (dc.zone == r["zone"]) &
               np.isclose(dc.target_frac, r["target_frac"])]
        x = g.iloc[0] if len(g) else {}
        rows.append({"model": r["model"], "policy": r["policy"], "split": int(r["zone"]),
                     "frac_selected_on_other_splits": r["target_frac"], "dti_heldout": r["dti_heldout"],
                     "area_frac": x.get("area_frac", np.nan), "recall_w": x.get("recall_w", np.nan),
                     "recall@300m": x.get("recall@300m", np.nan),
                     "far_share": x.get("far_area_km2", np.nan) / max(x.get("area_km2", np.nan), 1e-9) if len(g) else np.nan,
                     "tp_w": r["tp_w"], "fp_w": r["fp_w"], "fn_w": r["fn_w"]})
    out = pd.DataFrame(rows)
    out["rank_in_split"] = out.groupby(["policy", "split"]).dti_heldout.rank(ascending=False, method="min").astype(int)
    return out


def paired_deltas(tab: pd.DataFrame, ref: str, unit_col: str, metric: str = "dti_heldout",
                  by: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-unit deltas, summary) of every model minus ``ref`` on matching units.

    Summary: mean / median / min / max difference, units improved, directional consistency (share of units with
    the sign of the mean), d_z = mean / sd of the paired differences (descriptive; n = 3 units)."""
    if not len(tab) or ref not in set(tab.model):
        return pd.DataFrame(), pd.DataFrame()
    keys = list(by) + [unit_col]
    base = tab[tab.model == ref][keys + [metric]].rename(columns={metric: "ref_value"})
    per = tab[tab.model != ref].merge(base, on=keys, how="inner")
    per["delta"] = per[metric] - per.ref_value
    per = per[["model"] + keys + [metric, "ref_value", "delta"]].rename(columns={metric: "value"})
    per.insert(1, "reference", ref)
    rows = []
    for key, g in per.groupby(["model"] + list(by)):
        key = key if isinstance(key, tuple) else (key,)
        d = g.delta.to_numpy(float)
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
        mean = float(d.mean())
        rows.append({**dict(zip(["model"] + list(by), key)), "reference": ref, "n_units": len(d),
                     "mean_delta": mean, "median_delta": float(np.median(d)), "min_delta": float(d.min()),
                     "max_delta": float(d.max()), "units_improved": int((d > 0).sum()),
                     "directional_consistency": float(np.mean(np.sign(d) == np.sign(mean))) if mean != 0 else 0.0,
                     "effect_size_dz": mean / sd if sd and sd > 0 else np.nan,
                     "sign_pattern": "".join("+" if x > 0 else "-" if x < 0 else "0" for x in d),
                     "note": f"descriptive: {len(d)} {unit_col}s, no population-level test"})
    return per, pd.DataFrame(rows)


# =============================================================================== spatial block bootstrap
def block_ids(shape: tuple[int, int], block_px: int) -> np.ndarray:
    """Regular square blocks (the geographic folds' sampling units: ``folds.make_geo_folds`` uses the same lattice)."""
    rr, cc = np.indices(shape)
    nbc = int(np.ceil(shape[1] / block_px))
    return ((rr // block_px) * nbc + cc // block_px).astype(np.int32)


def block_components(pred: np.ndarray, truth: np.ndarray, ev: np.ndarray, threshold: float, blocks: np.ndarray,
                     exclude: np.ndarray | None = None, pixel_m=(100.0, 100.0), radius_m: float = DTI_RADIUS_M) -> pd.DataFrame:
    """Exact DTI components of the mask {pred >= threshold} split by block (sums equal ``BinaryCurve.at``).

    TP / FN are attributed to the block of the truth pixel, FP to the block of the predicted pixel."""
    from ..p3.curves import BinaryCurve

    c = BinaryCurve(pred, truth, ev, exclude, radius_m=radius_m, pixel_m=pixel_m)
    nb = int(blocks.max()) + 1
    tp = np.zeros(nb)
    n = np.zeros(nb)
    if c.n:
        rc = c.truth_rc
        b = blocks[rc[:, 0], rc[:, 1]]
        tp = np.bincount(b, c.per_truth_tp(threshold), nb)
        n = np.bincount(b, minlength=nb).astype(float)
    sl = c.sl
    evs = np.asarray(ev, bool)[sl]
    ex = np.zeros_like(evs) if exclude is None else np.asarray(exclude, bool)[sl]
    G = np.asarray(truth, bool)[sl] & evs & ~ex
    kd = np.clip(1.0 - ndimage.distance_transform_edt(~G, sampling=pixel_m) / radius_m, 0.0, None) if G.any() \
        else np.zeros(G.shape)
    p = np.asarray(pred, np.float64)[sl]
    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 0.0)
    m = evs & ~ex & (p >= threshold)
    fp = np.bincount(blocks[sl][m], (1.0 - kd)[m], nb)
    area = np.bincount(blocks[sl][evs & (p >= threshold)], minlength=nb).astype(float)
    keep = np.flatnonzero((n > 0) | (fp > 0) | (np.bincount(blocks[sl][evs], minlength=nb) > 0))
    return pd.DataFrame({"block": keep, "tp_w": tp[keep], "fp_w": fp[keep], "fn_w": (n - tp)[keep],
                         "n_truth": n[keep], "area_px": area[keep]})


def _dti(tp, fp, fn):
    return tp / (tp + DTI_ALPHA * fp + DTI_BETA * fn + DTI_EPS)


def block_bootstrap(comp: pd.DataFrame, a: str, b: str, *, strata: str = "zone", n_boot: int = 1000,
                    seed: int = 2026) -> dict:
    """Paired bootstrap of pooled DTI(a) - DTI(b): blocks resampled with replacement inside each stratum, the same
    draw for both models. ``comp``: block_components rows with columns model, zone, block."""
    rng = np.random.default_rng(seed)
    A = comp[comp.model == a].set_index([strata, "block"])[["tp_w", "fp_w", "fn_w"]]
    B = comp[comp.model == b].set_index([strata, "block"])[["tp_w", "fp_w", "fn_w"]]
    idx = A.index.union(B.index)
    A, B = A.reindex(idx, fill_value=0.0), B.reindex(idx, fill_value=0.0)
    groups = [np.flatnonzero(idx.get_level_values(0) == z) for z in idx.get_level_values(0).unique()]
    Av, Bv = A.to_numpy(), B.to_numpy()
    obs = float(_dti(*Av.sum(0)) - _dti(*Bv.sum(0)))
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups if len(g)])
        deltas[i] = _dti(*Av[pick].sum(0)) - _dti(*Bv[pick].sum(0))
    lo, med, hi = np.quantile(deltas, [0.05, 0.5, 0.95])
    return {"model": a, "reference": b, "observed_delta": obs, "boot_median": float(med), "boot_q05": float(lo),
            "boot_q95": float(hi), "p_delta_positive": float((deltas > 0).mean()), "n_boot": n_boot,
            "n_blocks": len(idx), "n_strata": len(groups),
            "note": "spatial-block bootstrap at the cross-fitted operating point; descriptive, conditional on the protocol"}


def geo_block_table(ctx, maps: dict[str, np.ndarray], fold_tab: pd.DataFrame, block_px: int) -> pd.DataFrame:
    """Block components of every model on every held-out fold at its cross-fitted budget."""
    from ..p3.curves import BinaryCurve

    blocks = block_ids(ctx.data.shape, block_px)
    rows = []
    for r in fold_tab.to_dict("records"):
        ev = ctx.data.valid & (ctx.folds.fold_map == r["fold"])
        p = maps[r["model"]]
        thr = float(BinaryCurve(p, ctx.data.labels, ev).threshold_for_area([r["frac_selected_on_other_folds"]])[0])
        rows.append(block_components(p, ctx.data.labels, ev, thr, blocks).assign(model=r["model"], zone=r["fold"],
                                                                                   threshold=thr))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def discovery_block_table(ctx, disc_tab: pd.DataFrame, block_px: int, policy: str = "excl") -> pd.DataFrame:
    """Block components per split under ``policy`` at the budget chosen on the other splits."""
    from ..p3 import discovery3
    from ..p3.curves import BinaryCurve

    blocks = block_ids(ctx.data.shape, block_px)
    rows = []
    t = disc_tab[disc_tab.policy == policy]
    for d in sorted(t.split.unique()):
        m = discovery3.split_masks(ctx, int(d))
        truth, ex = discovery3.policy_truth_exclude(m, policy)
        for r in t[t.split == d].to_dict("records"):
            p = ctx.discovery_pred(r["model"], int(d))
            if p is None:
                continue
            thr = float(BinaryCurve(p, truth, m["ez"], ex).threshold_for_area([r["frac_selected_on_other_splits"]])[0])
            rows.append(block_components(p, truth, m["ez"], thr, blocks, exclude=ex).assign(
                model=r["model"], zone=int(d), policy=policy, threshold=thr))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================== figures (each returns fig, data)
def _colors(names):
    from ..phase4_eval import colors_for

    return colors_for(list(names))


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _grouped_bars(ax, tab: pd.DataFrame, unit: str, metric: str, models: list[str], col: dict):
    units = sorted(tab[unit].unique())
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        g = tab[tab.model == m].set_index(unit).reindex(units)
        ax.bar(np.arange(len(units)) + (i - (len(models) - 1) / 2) * w, g[metric], width=w, color=col[m], label=m)
    ax.set_xticks(np.arange(len(units)), [f"{unit} {u}" for u in units], fontsize=8)


def fig_geo_dti_per_fold(fold_tab: pd.DataFrame, models: list[str]):
    import matplotlib.pyplot as plt

    col = _colors(models)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    _grouped_bars(ax, fold_tab, "fold", "dti_heldout", models, col)
    _style(ax, "", "held-out DTI (budget chosen on the other folds)", "Geographic DTI per held-out fold")
    ax.legend(fontsize=7, frameon=False)
    return fig, fold_tab[["model", "fold", "dti_heldout", "frac_selected_on_other_folds", "area_frac"]]


def fig_paired_differences(per: pd.DataFrame, boot: pd.DataFrame | None, ref: str, unit_col: str = "fold"):
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(per.model))
    col = _colors(models + [ref])
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(models) + 1.6), constrained_layout=True)
    for i, m in enumerate(models):
        g = per[per.model == m]
        ax.plot(g.delta, np.full(len(g), i), "o", color=col[m], ms=6)
        for x, u in zip(g.delta, g[unit_col]):
            ax.annotate(str(u), (x, i), xytext=(0, 6), textcoords="offset points", fontsize=6.5, ha="center")
        if boot is not None and len(boot) and m in set(boot.model):
            b = boot[boot.model == m].iloc[0]
            ax.plot([b.boot_q05, b.boot_q95], [i - 0.18, i - 0.18], color="#3a3a38", lw=1.4)
            ax.plot(b.observed_delta, i - 0.18, "|", color="#3a3a38", ms=9)
    ax.axvline(0, color="#8a8985", lw=0.8, ls="--")
    ax.set_yticks(range(len(models)), models, fontsize=8)
    _style(ax, f"held-out DTI difference vs {ref}", "", f"Paired differences vs {ref} (dots = folds / splits; "
                                                        "bar = 90 % block-bootstrap interval of the pooled difference)")
    data = per if boot is None or not len(boot) else per.merge(boot, on=["model", "reference"], how="left")
    return fig, data


def fig_discovery_per_split(disc_tab: pd.DataFrame, models: list[str], policies=("excl", "retained_fp")):
    import matplotlib.pyplot as plt

    col = _colors(models)
    pols = [p for p in policies if p in set(disc_tab.policy)]
    fig, axs = plt.subplots(1, max(len(pols), 1), figsize=(6.0 * max(len(pols), 1), 3.8), squeeze=False,
                            constrained_layout=True)
    for ax, pol in zip(axs[0], pols):
        _grouped_bars(ax, disc_tab[disc_tab.policy == pol], "split", "dti_heldout", models, col)
        _style(ax, "", "held-out discovery DTI", f"Pseudo-discovery DTI per split ({pol})")
    axs[0, 0].legend(fontsize=7, frameon=False)
    return fig, disc_tab[["model", "policy", "split", "dti_heldout", "frac_selected_on_other_splits"]]


def fig_worst_fold(fold_tab: pd.DataFrame, disc_tab: pd.DataFrame, models: list[str], policy: str = "excl"):
    import matplotlib.pyplot as plt

    g = fold_tab.groupby("model").dti_heldout.agg(geo_worst="min", geo_best="max")
    d = disc_tab[disc_tab.policy == policy].groupby("model").dti_heldout.agg(disc_worst="min", disc_best="max") \
        if len(disc_tab) else pd.DataFrame()
    data = g.join(d, how="left").reindex(models).reset_index()
    col = _colors(models)
    fig, axs = plt.subplots(1, 2, figsize=(10, 0.4 * len(models) + 1.8), constrained_layout=True)
    y = np.arange(len(models))
    for ax, (lo, hi, t) in zip(axs, (("geo_worst", "geo_best", "Geographic folds"),
                                     ("disc_worst", "disc_best", f"Discovery splits ({policy})"))):
        if lo not in data:
            ax.axis("off")
            continue
        ax.hlines(y, data[lo], data[hi], color="#c9c8c2", lw=3)
        ax.scatter(data[lo], y, color=[col[m] for m in models], zorder=3, s=36, label="worst")
        ax.scatter(data[hi], y, facecolors="white", edgecolors=[col[m] for m in models], zorder=3, s=30, label="best")
        ax.set_yticks(y, models, fontsize=8)
        _style(ax, "held-out DTI", "", f"Worst (filled) and best (open) unit: {t}")
    return fig, data


def fig_short_fault_recall(score: pd.DataFrame, short_fold: pd.DataFrame, models: list[str]):
    import matplotlib.pyplot as plt

    col = _colors(models)
    lb = [c for c in ("recall_w_len_<1", "recall_w_len_1-2", "recall_w_len_2-5", "recall_w_len_5-10",
                      "recall_w_len_>10") if c in score.columns]
    s = score.set_index("model").reindex(models)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    y = np.arange(len(models))
    if "short_recall_w" in s:
        axs[0].barh(y, s.short_recall_w, color=[col[m] for m in models], height=0.6)
    if len(short_fold):
        sf = short_fold[short_fold.stratum == "<2 km"]
        for i, m in enumerate(models):
            v = sf[sf.model == m].weighted_recall
            axs[0].plot(v, np.full(len(v), i), "o", color="#3a3a38", ms=3)
    axs[0].set_yticks(y, models, fontsize=8)
    _style(axs[0], "weighted recall of < 2 km faults (matched budget)", "", "Short-fault recall (dots = folds)")
    for m in models:
        if lb:
            axs[1].plot(range(len(lb)), s.loc[m, lb].astype(float), marker="o", color=col[m], label=m, lw=1.4)
    axs[1].set_xticks(range(len(lb)), [c.replace("recall_w_len_", "") + " km" for c in lb], fontsize=8)
    _style(axs[1], "mapped segment length", "weighted recall (matched budget)", "Recall by fault length")
    axs[1].legend(fontsize=7, frameon=False)
    data = s[[c for c in ["short_recall_w"] + lb if c in s]].reset_index()
    return fig, data


def fig_score_sensitivity(sc: dict, models: list[str], policy: str = "excl"):
    """Pooled DTI vs area budget (geographic and discovery): how sensitive each ranking is to the budget."""
    import matplotlib.pyplot as plt

    from ..p3 import budget
    from ..p3.curves import pooled

    col = _colors(models)
    geo = budget.pool_models(sc["geo_curves"])
    geo = geo[geo.model.isin(models) & (geo["mode"] == "per_zone")].assign(task="geographic")
    dc = sc.get("disc_curves", pd.DataFrame())
    disc = pd.DataFrame()
    if len(dc):
        dc = dc[(dc.policy == policy) & (dc.variant == "model") & dc.model.isin(models)]
        disc = pooled(dc, ["model", "target_frac"]).assign(task=f"discovery_{policy}")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, t, title in ((axs[0], geo, "Geographic"), (axs[1], disc, f"Pseudo-discovery ({policy})")):
        for m in models:
            g = t[t.model == m] if len(t) else t
            if len(g):
                ax.plot(g.target_frac * 100, g.dti, marker="o", ms=3, color=col[m], label=m, lw=1.4)
        ax.set_xscale("log")
        _style(ax, "predicted area (% of evaluated footprint)", "pooled DTI", f"{title}: DTI vs area budget")
    axs[0].legend(fontsize=7, frameon=False)
    keep = ["task", "model", "target_frac", "dti", "recall_w", "area_frac"]
    data = pd.concat([x[[c for c in keep if c in x]] for x in (geo, disc) if len(x)], ignore_index=True)
    return fig, data
