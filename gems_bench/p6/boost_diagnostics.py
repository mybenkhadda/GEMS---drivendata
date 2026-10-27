"""P6.1 recovery: restartable stage runner over the frozen P6 boost workspace (docs/p6/P6_RECOVERY_HANDOFF.md).

    rc = p6.recovery(RECOVERY_CONFIG, BOOST_CONFIG)
    rc.preflight()           # inputs, dependencies, RAM, G02 / G01 channel registry -> recovery/preflight
    rc.export_parity()       # X1 deployable W2 policies, parity, export plan        (export_parity.py)
    rc.candidate_recall()    # X2 recall ceiling, oracle (diagnostic), taxonomy        (candidate_recall.py)
    rc.segment_recovery()    # X3 H0 / H1 / H2 at matched area                         (segment_recovery.py)
    rc.geology_audit()       # X4 G01 / G02 dependency chain + repair plan (no compute)
    rc.g02_backfill()        # X4 explicit zone backfill (boost.backfill), then rc.geology_s1()
    rc.robustness()          # X5 weak-zone identity per policy (+ boost C1 / C2 when enabled)
    rc.select()              # promotion gate over every candidate
    rc.export(candidate_fp)  # validated GeoTIFF of the selected candidate only (never uploaded)
    rc.report()              # reports/recovery_report.md + registry

Every stage writes ``recovery/<stage>/{manifest,status,decision}.json``: fingerprint (config + input hashes), source
commit, feature schema, selection provenance, reload instructions. A stage reloads only when its fingerprint matches
and every required output exists; a missing output is never reported COMPLETE. Boost outputs and W2 are read-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import boost as B

RECOVERY_VERSION = "p6r.1"                  # enters every stage fingerprint
RECOVERY_CODE_REV = "p6r.1-r1"              # checked by the notebook start-up cell (not fingerprinted)
DEFAULT_RECOVERY_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "p6" / "recovery.yaml"
STAGE_DIRS = ("preflight", "export_parity", "candidate_recall", "segments", "geology", "robustness", "selection",
              "reports")
OUTPUTS = {
    "preflight": ("preflight.json", "dependency_check.csv", "decision.json"),
    "export_parity": ("policy_comparison.csv", "nested_policy_selection.csv", "per_zone_metrics.csv",
                      "train_inference_parity.csv", "feature_distribution_shift.csv", "export_policy.json",
                      "export_audit.json", "policy_frontier.png", "policy_frontier_data.csv", "decision.json"),
    "candidate_recall": ("coverage_by_zone.csv", "coverage_by_budget.csv", "system_inventory.parquet",
                         "candidate_source_attribution.csv", "missed_system_taxonomy.csv", "oracle_upper_bound.csv",
                         "candidate_redundancy.csv", "coverage_curves.png", "coverage_curves_data.csv",
                         "missed_system_examples.png", "decision.json"),
    "segments": ("segments.parquet", "segment_features.parquet", "segment_selection.csv", "graph_diagnostics.csv",
                 "hybrid_comparison.csv", "system_recovery.csv", "per_zone_metrics.csv", "recovery_examples.png",
                 "decision.json"),
    "geology": ("dependency_graph.json", "compatibility_report.csv", "feature_provenance.json", "repair_plan.json",
                "decision.json"),
    "robustness": ("weak_zone_identity.csv", "decision.json"),
    "selection": ("candidates.csv", "decision.json"),
    "reports": ("recovery_report.md", "artifact_registry.csv", "experiment_status.csv"),
}
STATUSES = B.STATUSES


def load_recovery_config(config: dict | None = None, path: Path | None = None) -> dict:
    from .run import _merge

    with open(path or DEFAULT_RECOVERY_CONFIG, encoding="utf-8") as f:
        cfg = _merge(yaml.safe_load(f), config or {})
    if len(cfg["segments"]["strategies"]) > 3 or not set(cfg["segments"]["strategies"]) <= {"H0", "H1", "H2"}:
        raise ValueError("segment recovery is bounded: at most the predeclared H0, H1, H2")
    if cfg["export_parity"]["objective"] not in ("disc", "disc_geo_floor", "joint"):
        raise ValueError(f"deployment objective {cfg['export_parity']['objective']!r} is not a single policy")
    return cfg


def missing_outputs(stage_dir: Path, stage: str) -> list[str]:
    d = Path(stage_dir)
    return [n for n in OUTPUTS.get(stage, ()) if not (d / n).exists() or (d / n).stat().st_size == 0]


# =============================================================================== pure: X4 classification
CHANNEL_CLASSES = ("OK", "REGISTRY_MISMATCH", "SOURCE_INCOMPATIBILITY", "MISSING_SOURCE")


def channel_class(kind: str, has_now: bool, in_registry: bool, fp_old, fp_now) -> str:
    """One recorded G02 input channel. ``REGISTRY_MISMATCH``: a virtual channel (computed on demand from a raw band,
    e.g. ``g01:edge`` / ``g01:surface``) absent from the cached-layer listing ``M9Source.names()`` but available
    through ``M9Source.has()``: NOT a missing input."""
    if not has_now:
        return "MISSING_SOURCE"
    if fp_old is not None and fp_now is not None and str(fp_old) != str(fp_now):
        return "SOURCE_INCOMPATIBILITY"
    if kind == "virtual_channel" and not in_registry:
        return "REGISTRY_MISMATCH"
    return "OK"


def blocker_kinds(attached: bool, channels: pd.DataFrame, units: pd.DataFrame) -> dict:
    """Blockers by kind: missing attachment, missing saved artifact (unit rasters), feature registry mismatch,
    source incompatibility, true feature-generation dependency (a recorded source absent now)."""
    c = channels.cls.value_counts().to_dict() if len(channels) else {}
    miss_u = list(units[~units.raster_exists].unit) if len(units) else []
    return {"missing_attachment": not attached,
            "missing_saved_artifact": miss_u,
            "feature_registry_mismatch": list(channels[channels.cls == "REGISTRY_MISMATCH"].name) if c else [],
            "source_incompatibility": list(channels[channels.cls == "SOURCE_INCOMPATIBILITY"].name) if c else [],
            "true_dependency_missing": list(channels[channels.cls == "MISSING_SOURCE"].name) if c else []}


def repair_status(kinds: dict) -> str:
    """READY_FOR_BACKFILL only when the sole real blocker is missing unit rasters (registry mismatches are not
    blockers); BLOCKED for an absent attachment, a changed or missing source; NOTHING_TO_DO otherwise."""
    if kinds["missing_attachment"] or kinds["source_incompatibility"] or kinds["true_dependency_missing"]:
        return "BLOCKED"
    return "READY_FOR_BACKFILL" if kinds["missing_saved_artifact"] else "NOTHING_TO_DO"


# =============================================================================== pure: selection gate
def recovery_gate(c: dict, gates: dict, matched_w2: float | None) -> tuple[bool, list[str], bool]:
    """Promotion rules of the brief: disc > max(validated W2 reference, matched W2), worst >= floor, geo >= floor,
    deployable single policy, no target leakage (provenance), parity without BLOCKED rows, export recipe planned.
    Third value: slot-worthy (gain >= ``slot_min_gain``). Oracle, post-hoc fixed cells and split policies fail."""
    why = []
    d, w, g = (c.get(k) for k in ("disc_dti", "disc_worst", "geo_dti"))
    if d is None or not np.isfinite(d):
        return False, ["no discovery DTI"], False
    ref = max(float(gates["w2_recorded_disc"]), float(matched_w2) if matched_w2 is not None and
              np.isfinite(matched_w2) else -np.inf)
    if not d > ref:
        why.append(f"disc {d:.4f} <= reference {ref:.4f}")
    if w is None or not np.isfinite(w) or w < float(gates["worst_floor"]) - 1e-12:
        why.append(f"worst zone {w} < {float(gates['worst_floor']):.6f}")
    if g is None or not np.isfinite(g) or g < float(gates["geo_floor"]):
        why.append(f"geo {g} < {float(gates['geo_floor']):.4f}")
    for flag, msg in (("deployable", "not a single deployable policy"), ("provenance_ok", "provenance not verified"),
                      ("parity_ok", "parity BLOCKED or not run"), ("recipe_ok", "export recipe not planned")):
        if not c.get(flag):
            why.append(msg)
    if str(c.get("role", "")).startswith(("oracle", "historical", "reference", "diagnostic")):
        why.append(f"role {c.get('role')} is never promotable")
    ok = not why
    return ok, why, bool(ok and d - ref >= float(gates["slot_min_gain"]) - 1e-12)


def next_direction(x2: dict | None, x1: dict | None, geo: dict | None) -> str:
    """Section 11 of the brief from saved decisions only."""
    if x1 and x1.get("parity_blocked"):
        return "export parity fails: fix full-region preprocessing / calibration before further modelling"
    if x2 and x2.get("verdict") not in (None, "INCONCLUSIVE", "NO_SYSTEMS", "NO_MISSED_SYSTEMS"):
        msg = x2.get("next_direction") or "see candidate_recall/decision.json"
        if x2.get("verdict") == "MIXED":
            msg += f"; also {x2.get('next_if_mixed')}"
        if geo and geo.get("s1_decision") == "ACCEPT":
            msg += "; G02-C adds value: prioritise geological feature integration + full-region g02c"
        return msg
    return "PENDING: run X1 and X2 (no new architecture before these answers)"


# =============================================================================== runner
class Recovery:
    """P6.1 stages under ``<p6 output_dir>/recovery``; boost outputs, W2, G02 run and submissions are read-only
    (except the explicit zone backfill, which writes new rasters into the G02 run and never rewrites existing ones,
    and the explicit export, which writes a new, never-overwritten submission file)."""

    def __init__(self, bst, config: dict | None = None, *, config_path: Path | None = None):
        self.bst, self.p6 = bst, bst.p6
        self.cfg = load_recovery_config(config, config_path)
        self.root = self.p6.ws.root / self.cfg["subdir"]

    # ------------------------------------------------------------------------------- plumbing
    def say(self, msg: str) -> None:
        print(f"[recovery] {msg}")

    def sub(self, stage: str) -> str:
        return f"{self.cfg['subdir']}/{stage}"

    def gates(self) -> dict:
        g = dict(self.cfg["gates"])
        g["w2_recorded_disc"] = float(self.bst.champion().get("disc_dti", self.cfg["references"]["w2_recorded"]
                                                              ["disc_dti"]))
        return g

    def save_table(self, df: pd.DataFrame, name: str, stage: str):
        return self.p6.ws.save_table(df if df is not None else pd.DataFrame(), name, self.sub(stage))

    def save_parquet(self, df: pd.DataFrame, name: str, stage: str) -> Path:
        from ..artifacts.atomic import atomic_save
        from ..pipeline.m6 import _parquet_safe

        p = self.root / stage / f"{name}.parquet"
        d = _parquet_safe(df if df is not None and len(df) else pd.DataFrame({"_empty": []}))
        atomic_save(p, lambda t: d.to_parquet(t, index=False))
        return p

    def save_json(self, obj, stage: str, name: str):
        return self.p6.ws.save_json(json.loads(json.dumps(obj, default=B._js)), f"{self.sub(stage)}/{name}.json")

    def save_fig(self, fig, name: str, data, stage: str):
        return self.p6.ws.save_fig(fig, name, data if data is not None else pd.DataFrame(), sub=self.sub(stage))

    def load_table(self, name: str, stage: str) -> pd.DataFrame:
        return self.p6.ws.load_table(name, self.sub(stage))

    def read_json(self, stage: str, name: str) -> dict | None:
        p = self.root / stage / f"{name}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.say(f"WARNING {p} unreadable ({type(e).__name__}): treated as absent")
            return None

    def fp(self, *parts) -> str:
        return self.p6.fp(RECOVERY_VERSION, *parts)

    def _done(self, stage: str, fp: str) -> bool:
        m = self.read_json(stage, "manifest") or {}
        if m.get("fingerprint") != fp or m.get("status") != "done":
            return False
        miss = missing_outputs(self.root / stage, stage)
        if miss:
            self.say(f"{stage}: fingerprint matches but {miss} missing -> recomputed")
            return False
        self.say(f"{stage}: fingerprint {fp[:10]} unchanged -> reloaded (nothing recomputed)")
        return True

    def set_status(self, stage: str, status: str, **info) -> None:
        if status not in STATUSES:
            raise ValueError(f"status {status!r} not in {STATUSES}")
        self.save_json({"stage": stage, "status": status, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "code_rev": RECOVERY_CODE_REV, **info}, stage, "status")

    def _begin(self, stage: str, fp: str) -> float:
        self.set_status(stage, "RUNNING", fingerprint=fp)
        return time.perf_counter()

    def _finish(self, stage: str, fp: str, t0: float, inputs: dict, decisions: dict, *, status: str = "COMPLETE",
                reason: str = "", schema: list | None = None, selection: dict | None = None) -> None:
        from .._version import provenance

        miss = missing_outputs(self.root / stage, stage)
        if miss and status == "COMPLETE":                    # never COMPLETE with an absent output
            status, reason = "FAILED", f"required outputs missing: {miss}"
        self.save_json({"stage": stage, "status": "done" if status == "COMPLETE" else status.lower(),
                        "stage_status": status, "fingerprint": fp, "recovery_version": RECOVERY_VERSION,
                        "code_rev": RECOVERY_CODE_REV, "config": self.cfg, "boost_config": self.bst.cfg,
                        "p6_config_fp": self.p6.fp(self.p6.cfg), "inputs": inputs, "feature_schema": schema,
                        "selection_provenance": selection, "decisions": decisions, "code": provenance(),
                        "seconds": round(time.perf_counter() - t0, 1), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "outputs": sorted(p.name for p in (self.root / stage).glob("*") if p.is_file()),
                        "reload": f"p6.recovery(RECOVERY_CONFIG, BOOST_CONFIG).{_METHOD.get(stage, stage)}() reloads "
                                  "while the fingerprint holds and every required output exists"}, stage, "manifest")
        self.set_status(stage, status, fingerprint=fp, reason=reason, source_commit=provenance()["git_commit"])
        self.p6.push()

    def section(self, stage: str, fn):
        """Budgeted step on the boost CPU cap (``wb``); failures leave ``status.json`` = FAILED."""
        from .run import BudgetExceeded

        try:
            return self.p6.section(self.cfg["workstream"], "cpu", fn, what=f"recovery_{stage}")
        except BudgetExceeded as e:
            self.set_status(stage, "BLOCKED", reason=f"BudgetExceeded: {e}")
            raise
        except Exception as e:
            self.set_status(stage, "FAILED", reason=f"{type(e).__name__}: {e}")
            raise

    def load(self, stage: str) -> dict:
        d = self.root / stage
        out = {"decision": self.read_json(stage, "decision"), "status": self.read_json(stage, "status")}
        for p in sorted(d.glob("*.csv")) if d.exists() else []:
            if not p.name.endswith("_data.csv"):
                out[p.stem] = self.load_table(p.stem, stage)
        for p in sorted(d.glob("*.parquet")) if d.exists() else []:
            out[p.stem] = pd.read_parquet(p)
        return out

    def benchmark_grid(self) -> pd.DataFrame:
        """The boost stage-A grid (official components of every fixed cell); refused unless stage A finished."""
        bst = self.bst
        p = bst.root / "benchmark" / "grid.csv"
        m = bst.read_json("benchmark", "manifest") or {}
        if not p.exists() or m.get("status") != "done":
            raise FileNotFoundError(f"{p}: boost benchmark (stage A) not complete; run bst.a_benchmark() first "
                                    "(it reloads when already done)")
        G = pd.read_csv(p, dtype={"systems": str})
        return G[G.candidate != "fallback_published"].reset_index(drop=True)

    def input_hashes(self) -> dict:
        from .boost_audit import file_hash

        bst = self.bst
        root = self.p6.ws.root
        return {"benchmark_grid": file_hash(bst.root / "benchmark" / "grid.csv"),
                "benchmark_fp": (bst.read_json("benchmark", "manifest") or {}).get("fingerprint"),
                "w2": {"maps": bst.w2_inputs()["maps_sha256"], "rows": bst.w2_inputs()["rows_sha256"],
                       "full_score": file_hash(root / "stack" / "maps" / "full_score.tif", 1e9)},
                "selection_champion": json.dumps((bst.sel.get(bst.prim) or {}).get(self.p6.cfg["probes"]["champion"],
                                                                                    {}).get("params_all"), default=str)}

    # ------------------------------------------------------------------------------- figures
    def fig_frontier(self, F: pd.DataFrame, gates: dict, stage: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        if len(F):
            fx = F[F.kind.str.startswith("fixed")]
            ax.scatter(fx.geo_dti, fx.disc_dti, s=10, c="#bbbbbb", label="fixed cells (unselected, in-sample)")
            nz = F[F.kind == "nested_policy"]
            for (obj, dep), g in nz.groupby(["objective", "deployable"]):
                ax.scatter(g.geo_dti, g.disc_dti, s=36, marker="o" if dep else "x",
                           label=f"nested: {obj}{'' if dep else ' (not one deployable policy)'}")
                for r in g.itertuples():
                    ax.annotate(str(r.policy)[:16], (r.geo_dti, r.disc_dti), fontsize=6)
        ax.axvline(float(gates["geo_floor"]), ls="--", c="k", lw=0.8, label="geo floor")
        ax.axhline(float(gates["w2_recorded_disc"]), ls=":", c="k", lw=0.8, label="W2 recorded disc")
        ax.set(xlabel="geographic DTI (pooled, held-out)", ylabel="discovery DTI (pooled, held-out)",
               title="Deployable policy frontier (offline nested; not leaderboard)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        self.save_fig(fig, "policy_frontier", F, stage)

    def fig_coverage(self, D: pd.DataFrame, stage: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        if len(D) and "system_recall" in D:
            for (s, rp), g in D[D.system_recall.notna()].groupby(["source", "rep"]):
                g = g.sort_values("frac")
                ls = "-" if rp == "crest_1px" else "--"
                ax[0].plot(100 * g.frac, g.system_recall, ls, marker="o", ms=3, label=f"{s} / {rp}")
                ax[1].plot(100 * g.frac, g.length_weighted_recall, ls, marker="o", ms=3, label=f"{s} / {rp}")
        if len(D):
            o = D[D.source == "oracle_inventory_constrained"].sort_values("frac")
            if len(o):
                a2 = ax[0].twinx()
                a2.plot(100 * o.frac, o.dti, "k:", label="oracle DTI (diagnostic, labels)")
                a2.set_ylabel("oracle DTI (diagnostic)")
                a2.legend(fontsize=6, loc="lower right")
        for a, t in zip(ax, ("system recall", "length-weighted recall")):
            a.set(xlabel="area budget (% of eligible)", ylabel=t)
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=6, loc="upper left")
        fig.suptitle("Candidate-pool recall by budget (held-out discovery systems; diagnostic)")
        self.save_fig(fig, "coverage_curves", D, stage)

    def fig_segments(self, stage: str, name: str, reps: list, fracs: list, use_g: bool) -> None:
        """``recovery_examples.png``: the disc zone / cell where ``name`` adds the most pixels: W2 core, additions,
        evaluation-only truth."""
        import matplotlib.pyplot as plt

        from . import segment_recovery as SR

        S = self.load_table("segment_selection", stage)
        cfg = self.cfg["segments"]
        S = S[(S.strategy == name) & S.unit.str.startswith(self.bst.prim)] if len(S) else S
        fig, ax = plt.subplots(figsize=(8, 6))
        rows = []
        if len(S) and S.added_px.max() > 0:
            r = S.sort_values("added_px", ascending=False).iloc[0]
            u = next(x for x in self.bst.disc_units() if x.key == r.unit)
            inp = SR.unit_inputs(self.bst, u, use_g)

            def ap(x, f):
                return self.bst.binary(u, x, r.rep, f)

            q0 = ap(inp["w2"], float(r.frac))
            q, _, _ = SR.strategy_map(name, inp, ap, float(r.frac), cfg)
            truth = u.eval_truth()
            ax.imshow(inp["w2"], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            for m, col, lab in ((q0 & q, "#ff7f0e", "kept W2 (H0 and strategy)"), (q & ~q0, "#2ca02c", f"{name} additions"),
                                (q0 & ~q, "#1f77b4", "W2 removed to keep the area")):
                rr, cc = np.nonzero(m)
                ax.scatter(cc, rr, s=1, c=col, label=lab, linewidths=0)
            rr, cc = np.nonzero(truth)
            ax.scatter(cc, rr, s=1, c="magenta", label="EVALUATION-ONLY reference", linewidths=0, alpha=0.5)
            ax.legend(fontsize=6, markerscale=6)
            ax.set_title(f"{name} vs H0, {r.unit} {r.rep} @ {float(r.frac):g} (added {int(r.added_px)} px)")
            rows.append({"unit": r.unit, "rep": r.rep, "frac": float(r.frac), "strategy": name,
                         "added_px": int(r.added_px), "kept_px": int((q0 & q).sum()), "removed_px": int((q0 & ~q).sum()),
                         "truth_use": "evaluation-only overlay"})
            u.release()
        else:
            ax.text(0.5, 0.5, f"{name}: no additions in any disc cell", ha="center")
        ax.axis("off")
        self.save_fig(fig, "recovery_examples", pd.DataFrame(rows), stage)

    # =========================================================================== stages
    def preflight(self) -> dict:
        return self.section("preflight", self._preflight)

    def _preflight(self) -> dict:
        import importlib.util

        from .._version import provenance
        from .boost_audit import file_hash

        stage = "preflight"
        t0 = self._begin(stage, "always")
        bst, root = self.bst, self.p6.ws.root
        dep = []
        for mod in ("numpy", "scipy", "pandas", "pyarrow", "rasterio", "matplotlib", "lightgbm", "joblib", "skimage",
                    "psutil"):
            dep.append({"kind": "python_module", "name": mod, "ok": importlib.util.find_spec(mod) is not None})
        need = {"stack/candidate.json": root / "stack" / "candidate.json",
                "stack/features/rows.parquet": root / "stack" / "features" / "rows.parquet",
                "stack/maps/full_score.tif": root / "stack" / "maps" / "full_score.tif",
                "boost/benchmark/grid.csv": bst.root / "benchmark" / "grid.csv",
                "boost/benchmark/nested_selection.csv": bst.root / "benchmark" / "nested_selection.csv",
                "boost/preflight/manifest.json": bst.root / "preflight" / "manifest.json"}
        for k, p in need.items():
            dep.append({"kind": "artifact", "name": k, "ok": p.exists(), "sha256": file_hash(p, 1024) if p.exists()
                        else None})
        n_maps = len(list((root / "stack" / "maps").glob("*.npy"))) if (root / "stack" / "maps").exists() else 0
        dep.append({"kind": "artifact", "name": "stack/maps/<task>_<zone>.npy", "ok": n_maps == 6, "detail": n_maps})
        bm = bst.read_json("benchmark", "manifest") or {}
        dep.append({"kind": "stage", "name": "boost benchmark (A) done", "ok": bm.get("status") == "done",
                    "detail": bm.get("fingerprint")})
        sf = (bst.w2_inputs().get("candidate") or {}).get("submission_file")
        hist = root / "submissions" / sf if sf else None
        dep.append({"kind": "artifact", "name": "historical W2 submission (read-only)", "ok": bool(hist and hist.exists()),
                    "detail": sf, "sha256": file_hash(hist, 1e9) if hist and hist.exists() else None})
        try:
            import psutil

            ram = psutil.virtual_memory()
            dep.append({"kind": "runtime", "name": "RAM available GB", "ok": ram.available / 1e9 >=
                        float(self.cfg["preflight"]["min_ram_gb"]), "detail": round(ram.available / 1e9, 1)})
        except Exception as e:
            dep.append({"kind": "runtime", "name": "RAM", "ok": False, "detail": f"{type(e).__name__}: {e}"})
        dep.append({"kind": "budget", "name": f"{self.cfg['workstream']} CPU h left",
                    "ok": self.p6.budget.left_h(self.cfg["workstream"], "cpu") > 0,
                    "detail": round(self.p6.budget.left_h(self.cfg["workstream"], "cpu"), 3)})
        g02c_complete = None
        try:
            att = bst.g02()
            have = set(att.raster_units()) if att is not None else set()
            g02c_complete = bool(att is not None and all(f"{u.task}_{u.zone}" in have for u in bst.units()))
            dep.append({"kind": "g02", "name": "G02 rasters for every unit", "ok": g02c_complete,
                        "detail": sorted(have)})
        except Exception as e:
            dep.append({"kind": "g02", "name": "G02 attach", "ok": False, "detail": f"{type(e).__name__}: {e}"})
        D = pd.DataFrame(dep)
        self.save_table(D, "dependency_check", stage)
        ok = lambda n: bool(D[D.name == n].ok.all()) if (D.name == n).any() else False   # noqa: E731
        core = all(ok(n) for n in ("stack/candidate.json", "stack/features/rows.parquet", "boost/benchmark/grid.csv",
                                   "boost benchmark (A) done")) and ok("stack/maps/<task>_<zone>.npy")
        ready = {"export_parity": core and ok("stack/maps/full_score.tif"),
                 "export_parity_tables_only": core,
                 "candidate_recall": core, "segment_recovery": core and ok("stack/maps/full_score.tif"),
                 "geology_audit": True, "g02_backfill": False, "export": False}
        dec = {"stage": stage, "ready": ready, "g02c_complete": g02c_complete, "code": provenance(),
               "failed_checks": D[~D.ok.astype(bool)].name.tolist(),
               "note": "export_parity needs stack/maps/full_score.tif for parity + plan (tables run without it); "
                       "g02_backfill readiness comes from rc.geology_audit()"}
        self.save_json(dec, stage, "decision")
        self.save_json({"version": RECOVERY_VERSION, "code_rev": RECOVERY_CODE_REV, "config_fp": self.fp(self.cfg)[:12],
                        "boost_code_rev": B.BOOST_CODE_REV, "artifact_root": str(self.root), "checks": dep,
                        "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, stage, "preflight")
        self._finish(stage, "always", t0, {"checks": len(dep)}, {"ready": ready},
                     status="COMPLETE" if core else "BLOCKED", reason="" if core else f"failed: {dec['failed_checks']}")
        return dec

    def export_parity(self) -> dict:
        from . import export_parity as EP

        return self.section("export_parity", lambda: EP.run(self))

    def candidate_recall(self) -> dict:
        from . import candidate_recall as CR

        return self.section("candidate_recall", lambda: CR.run(self))

    def segment_recovery(self) -> dict:
        from . import segment_recovery as SR

        return self.section("segments", lambda: SR.run(self))

    # ------------------------------------------------------------------------------- X4 geology
    def geology_audit(self) -> dict:
        return self.section("geology", self._geology_audit)

    def _geology_audit(self) -> dict:
        """Dependency chain G01 / M9 cache / raw bands -> G02 run -> unit rasters -> W2 / boost features, with every
        blocker classified. Computes no map, never backfills."""
        stage = "geology"
        bst, p6 = self.bst, self.p6
        t0 = self._begin(stage, "audit")
        g = bst.g02()
        attached = g is not None
        ch, un, prov = [], [], {"requested_run_id": self.cfg["geology"]["g02_run_id"],
                                "attach_error": getattr(bst, "g02_error", None)}
        graph = {"nodes": [], "edges": []}

        def node(i, kind, **kw):
            graph["nodes"].append({"id": i, "kind": kind, **kw})

        def edge(a, b, why):
            graph["edges"].append({"from": a, "to": b, "relation": why})

        if attached:
            st = g.store
            inp = st.load_json("config/inputs.json") or {}
            rec = inp.get("input_fingerprints") or {}
            cached = set(g.src.names())
            run = g.attach_info.get("run_id")
            node(run, "g02_run", mode=g.attach_info.get("mode"), fingerprint=g.attach_info.get("fingerprint"))
            for name, old in rec.items():
                if name.startswith("band:"):
                    kind, has = "raw_band", (old in getattr(p6.ctx.data, "band_names", []))
                    now = old
                elif name == "m9pred":
                    from ..m9.features import file_fp

                    kind = "m9_prediction"
                    pth = Path(g.cfg["baselines"].get("m9_prediction_npy") or "")
                    has = pth.exists()
                    now = file_fp(pth) if has else None
                else:
                    kind = "virtual_channel" if name in ("g01:edge", "g01:surface") else "cached_layer"
                    has = bool(g.src.has(name))
                    now = g.src.fingerprint(name) if has else None
                cls = channel_class(kind, has, name in cached, old, now)
                ch.append({"name": name, "kind": kind, "available_now": has, "in_cache_registry": name in cached,
                           "fp_recorded": str(old), "fp_now": None if now is None else str(now), "cls": cls,
                           "note": ("computed on demand from raw band "
                                    f"{g.src._band(name) if kind == 'virtual_channel' else ''}: never a cached .npy "
                                    "(M9Source.names lists cached stems only)") if kind == "virtual_channel" else ""})
                node(name, kind, cls=cls)
                edge(name, run, "input fingerprint recorded in config/inputs.json")
                if kind == "virtual_channel":
                    b = g.src._band(name)
                    node(f"band:{b}", "raw_band")
                    edge(f"band:{b}", name, "THDR at g01.baselines.edge_sigma_m (label-free)")
            have = set(g.raster_units())
            for u in bst.units():
                k = f"{u.task}_{u.zone}"
                rel = f"rasters/units/{k}.npz"
                keys = set(st.raster_keys(rel)) if k in have else set()
                un.append({"unit": u.key, "raster": rel, "raster_exists": k in have,
                           **{f"key:{rk}": rk in keys for rk in bst.cfg["geology"]["keys"].values()},
                           "refs": (st.load_json(f"processed/refs_{k}.json") or None) is not None})
                node(rel, "unit_raster", exists=k in have)
                edge(run, rel, "G02Experiment.evaluate_unit / unit_raster (backfill)")
                for f, rk in bst.cfg["geology"]["keys"].items():
                    edge(rel, f"feature:{f}", f"raster key {rk} -> stack.g02_unit")
            from ..m9.research.g02_experiment import g01_run_root

            g01root = g01_run_root(p6.m9, g.g01cfg)
            g01runs = sorted(x.name for x in g01root.glob("g01_*")) if g01root.exists() else []
            prov.update({"resolved_run_id": run, "resolved_by": g.attach_info.get("resolved_by"),
                         "mode": g.attach_info.get("mode"), "fingerprint": g.attach_info.get("fingerprint"),
                         "manifest_sha256": g.attach_info.get("manifest_sha256"),
                         "g01_wavelengths": inp.get("g01_wavelengths"), "g02_wavelengths": inp.get("g02_wavelengths"),
                         "xmod": inp.get("xmod"), "g01_store_root": str(g01root), "g01_runs_on_disk": g01runs,
                         "g01_review_run": (g.cfg.get("g01") or {}).get("review_run"),
                         "g01_dependency": "G02 recomputes the G01 maps it needs from the M9 layer cache + virtual "
                                           "channels; a persisted G01 run is used only by the G01 review (not by any "
                                           "raster)",
                         "feature_definitions": "every unit raster (incl. backfill) uses the run's own config; "
                                                "normalisation references are fitted per unit on its training "
                                                "geography (identical definition, unit-specific reference)",
                         "full_region": "label-free stack.g02_full (whole footprint as reference); tiles present: "
                                        f"{(p6.ws.root / 'stack' / 'features' / 'g02_full').exists()}"})
            try:
                prov["input_drift"] = g.input_drift()
            except Exception as e:
                prov["input_drift"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        C, U = pd.DataFrame(ch), pd.DataFrame(un)
        kinds = blocker_kinds(attached, C, U)
        if attached and not (prov.get("input_drift") or {}).get("ok", False):
            kinds["source_incompatibility"] = kinds["source_incompatibility"] or ["input_drift: not ok"]
        status = repair_status(kinds)
        plan = {"status": status, "blockers": kinds, "steps": [
            {"step": "attach G02 run", "status": "DONE" if attached else "BLOCKED"},
            {"step": "g01:edge / g01:surface", "status": "RESOLVED_IN_CODE (registry mismatch, not a missing input)"
             if kinds["feature_registry_mismatch"] else "OK"},
            {"step": "input drift", "status": "OK" if (prov.get("input_drift") or {}).get("ok") else "BLOCKED"},
            {"step": f"backfill {kinds['missing_saved_artifact']}", "status": status if status != "NOTHING_TO_DO"
             else "DONE", "switch": "RECOVERY_RUN['g02_backfill'] (explicit; raster-only, run config, resumable)"},
            {"step": "S1 = W2 + G02-C", "status": "AFTER_BACKFILL" if kinds["missing_saved_artifact"] else "READY",
             "switch": "RECOVERY_RUN['geology']"},
            {"step": "S2 / S3", "status": "ONLY_IF_S1_ACCEPT_OR_DECLARED_DIAGNOSTIC"}],
            "schema_rule": "the original G02 run's input schema is never changed; a backfill refuses any drift"}
        self.save_json(graph, stage, "dependency_graph")
        self.save_table(C, "compatibility_report", stage)
        self.save_table(U, "unit_rasters", stage)
        self.save_json(prov, stage, "feature_provenance")
        self.save_json(plan, stage, "repair_plan")
        s1 = self.read_json(stage, "s1_decision") or {}
        dec = {"stage": stage, "repair_status": status, "blockers": kinds, "backfill_allowed": status ==
               "READY_FOR_BACKFILL", "s1_decision": s1.get("decision"), "s1": s1 or None}
        self.save_json(dec, stage, "decision")
        self._finish(stage, "audit", t0, {"g02": prov}, {"repair_status": status},
                     status="BLOCKED" if status == "BLOCKED" else "COMPLETE", reason=json.dumps(kinds, default=str))
        return dec

    def g02_backfill(self) -> list[str]:
        """Explicit zone backfill (boost.backfill: run config, drift refused, existing rasters never rewritten)."""
        dec = self.read_json("geology", "decision") or {}
        if not dec.get("backfill_allowed"):
            raise RuntimeError(f"backfill not allowed: run rc.geology_audit() first ({dec.get('repair_status')}, "
                               f"{dec.get('blockers')})")
        out = self.section("geology", lambda: self.bst.backfill(int(self.cfg["geology"]["backfill_max_units"])))
        self.say(f"backfilled {out}; rerun rc.geology_audit() then rc.geology_s1()")
        return out

    def geology_s1(self) -> dict:
        return self.section("geology", self._geology_s1)

    def _geology_s1(self) -> dict:
        """S1 (W2 + G02-C) under the boost protocol; S2 / S3 only after S1 ACCEPT or with a declared diagnostic
        reason. Cross-fitted maps share the boost geology cache (same fingerprint parts)."""
        bst = self.bst
        gcf = bst.cfg["geology"]
        info, AU = bst.g02_audit()
        present = {u.key: set(AU[(AU.unit == u.key) & AU.key_present.astype(bool)].feature) if len(AU) else set()
                   for u in bst.units()}
        base = bst.w2_features()
        prev = self.read_json("geology", "s1_decision") or {}
        why = str(self.cfg["geology"].get("s2_s3_diagnostic_reason") or "")
        names = ["S0", "S1"] + (["S2", "S3"] if prev.get("decision") == "ACCEPT" or why else [])
        specs = []
        for n in names:
            add = list(gcf["variants"][n])
            st, miss = B.variant_status(present, add)
            specs.append({"name": n, "names": base + [a for a in add if a not in base], "status": st,
                          "reason": f"missing {miss}" if miss else ""})
        out = bst.run_variants("geology", specs, "S0", {u.key: bst.w2_map(u) for u in bst.units()})
        S = {s["candidate"]: s for s in out["summary"]}
        ok, why_rej = B.accept(S.get("S1", {}), S.get("S0", {}), gcf["accept"], float(self.cfg["gates"]["geo_floor"]))
        d = {"decision": "BLOCKED" if S.get("S1", {}).get("status") == "BLOCKED" else ("ACCEPT" if ok else "REJECT"),
             "reasons": why_rej, "summary": {k: B._flat(v) for k, v in S.items()}, "variants_run": names,
             "s2_s3_reason": why or ("S1 ACCEPT" if "S2" in names else "not run (S1 not accepted, no declared reason)"),
             "spec": out["manifests"].get("S1")}
        self.save_table(pd.DataFrame([B._flat(s) for s in out["summary"]]), "s1_ablation", "geology")
        self.save_json(d, "geology", "s1_decision")
        g = self.read_json("geology", "decision") or {}
        self.save_json({**g, "s1_decision": d["decision"], "s1": d}, "geology", "decision")
        return d

    # ------------------------------------------------------------------------------- X5 robustness
    def robustness(self) -> dict:
        return self.section("robustness", self._robustness)

    def _robustness(self) -> dict:
        """Weak-zone identity per evaluated policy (never assumed), candidate recall by zone, optional boost C1 / C2
        (predeclared; pooled gain must not hide a worst-zone loss)."""
        stage = "robustness"
        t0 = self._begin(stage, "x5")
        rows = []
        for st, nm in (("export_parity", "per_zone_metrics"), ("segments", "per_zone_metrics")):
            P = self.load_table(nm, st)
            if not len(P):
                continue
            for (pol, obj, task), g in P.dropna(subset=["dti"]).groupby(["policy", "objective", "task"]):
                w = g.loc[g.dti.idxmin()]
                rows.append({"source_stage": st, "policy": pol, "objective": obj, "task": task,
                             "weakest_zone": int(w.zone), "weakest_dti": float(w.dti),
                             "weakest_cell": f"{w.rep}@{w.frac}", "zones": json.dumps({int(r.zone): float(r.dti)
                                                                                       for r in g.itertuples()})})
        W = pd.DataFrame(rows)
        self.save_table(W, "weak_zone_identity", stage)
        cz = self.load_table("coverage_by_zone", "candidate_recall")
        if len(cz):
            self.save_table(cz[cz.source.isin(["w2", "fallback_fusion", "w2_fallback_union"])], "recall_by_zone", stage)
        c = None
        if self.cfg["robustness"].get("run_boost_c"):
            c = self.bst.c_robustness()
            A = c.get("ablation") if isinstance(c, dict) else None
            if A is not None and len(A):
                self.save_table(A, "boost_c_ablation", stage)
        d = self.bst.read_json("robustness", "decision") or {}
        dec = {"stage": stage, "boost_c_chosen": d.get("chosen"), "boost_c_accepted": d.get("accepted"),
               "weak_zone_by_policy": W.to_dict("records") if len(W) else [],
               "rule": "C1 / C2 predeclared; accept only with pooled gain AND worst zone not down (boost robustness."
                       "accept); no zone-specific tuning", "boost_c_run": c is not None}
        self.save_json(dec, stage, "decision")
        self._finish(stage, "x5", t0, {"boost_c": bool(c)}, {"chosen": d.get("chosen")})
        return dec

    # ------------------------------------------------------------------------------- selection / export
    def select(self) -> dict:
        return self.section("selection", self._select)

    def _candidates(self) -> list[dict]:
        out = []
        par = self.load_table("train_inference_parity", "export_parity")
        par_ok = bool(len(par)) and "NOT_RUN" not in set(par.status) and not (par.status == "BLOCKED").any()
        if not self.cfg["selection"]["require_numeric_parity"]:
            par_ok = bool(len(par)) and not (par.status == "BLOCKED").any()
        x1 = self.read_json("export_parity", "decision") or {}
        pol = self.read_json("export_parity", "export_policy") or {}
        C = self.load_table("policy_comparison", "export_parity")
        if len(C):
            for r in C[C.objective == x1.get("objective")].to_dict("records"):
                out.append({"name": f"x1:{r['policy']}", "stage": "export_parity", "role": r.get("role"),
                            "disc_dti": r.get("disc_dti"), "disc_worst": r.get("disc_worst"),
                            "geo_dti": r.get("geo_dti"), "deployable": bool(r.get("deployable")),
                            "provenance_ok": True, "parity_ok": par_ok,
                            "recipe_ok": r["policy"] == x1.get("chosen_policy") and (pol.get("plan_audit") or {})
                            .get("status") == "PASS", "deploy_cell": r.get("deploy_cell"),
                            "candidate_fp": ((pol.get("recipe") or {}).get("fingerprint") if r["policy"] ==
                                             x1.get("chosen_policy") else None)})
        x3 = self.read_json("segments", "decision") or {}
        H = self.load_table("hybrid_comparison", "segments")
        for r in H.to_dict("records") if len(H) else []:
            if r["candidate"] == "H0":
                continue
            out.append({"name": f"x3:{r['candidate']}", "stage": "segments", "role": "deploy_candidate",
                        "disc_dti": r.get("disc_dti"), "disc_worst": r.get("disc_worst"), "geo_dti": r.get("geo_dti"),
                        "deployable": bool(r.get("deployable")), "provenance_ok": r.get("decision") == "ACCEPT",
                        "parity_ok": par_ok, "recipe_ok": r.get("decision") == "ACCEPT" and x3.get("chosen") ==
                        r["candidate"], "deploy_cell": r.get("deploy_cell"), "candidate_fp": None})
        s1 = (self.read_json("geology", "decision") or {}).get("s1") or {}
        if s1.get("decision") == "ACCEPT":
            s = json.loads(s1["summary"]["S1"]["params_all"]) if isinstance(s1["summary"]["S1"].get("params_all"),
                                                                             str) else {}
            out.append({"name": "x4:S1", "stage": "geology", "role": "deploy_candidate",
                        "disc_dti": float(s1["summary"]["S1"].get("disc_dti")),
                        "disc_worst": float(s1["summary"]["S1"].get("disc_worst")),
                        "geo_dti": float(s1["summary"]["S1"].get("geo_dti")),
                        "deployable": False, "provenance_ok": True, "parity_ok": False, "recipe_ok": False,
                        "note": "S1 uses separate per-task nested choices (boost protocol): re-evaluate as one policy "
                                "and audit full-region g02c parity before promotion", "params_all": s})
        return out

    def _select(self) -> dict:
        stage = "selection"
        t0 = self._begin(stage, "select")
        gates = self.gates()
        bn = self.bst.load_table("nested_selection", "benchmark")
        w2m = bn[bn.space == "w2_space|blend+rep+frac"] if len(bn) else bn
        matched = float(w2m.disc_dti.iloc[0]) if len(w2m) else None
        best_bench = {k: v for k, v in self.cfg["references"]["boost_benchmark"].items()}
        rows, best = [], None
        for c in self._candidates():
            ok, why, slot = recovery_gate(c, gates, matched)
            c.update(gate="PASS" if ok else "FAIL", gate_reasons="; ".join(why), slot_worthy=slot,
                     delta_vs_w2_recorded=(c["disc_dti"] - gates["w2_recorded_disc"]) if c.get("disc_dti") is not None
                     and np.isfinite(c["disc_dti"]) else None,
                     delta_vs_best_matched_w2_benchmark=(c["disc_dti"] - max(v["disc_dti"] for k, v in
                                                                            best_bench.items() if k.startswith("w2")))
                     if c.get("disc_dti") is not None and np.isfinite(c["disc_dti"]) else None)
            rows.append(c)
            if ok and (best is None or c["disc_dti"] > best["disc_dti"]):
                best = c
        dec = {"stage": stage, "chosen": best["name"] if best else None, "replace_champion": best is not None,
               "slot_worthy": bool(best and best["slot_worthy"]), "candidate_fp": (best or {}).get("candidate_fp"),
               "gates": gates, "matched_w2_benchmark": matched,
               "benchmark_note": "the stronger matched W2 benchmark cells (boost) are an additional diagnostic only",
               "export": "rc.export(candidate_fp) with RECOVERY_RUN['export'] = True after PASS; never uploaded"}
        self.save_table(pd.DataFrame([B._flat(r) for r in rows]), "candidates", stage)
        self.save_json(dec, stage, "decision")
        self._finish(stage, "select", t0, {"candidates": len(rows)}, {"chosen": dec["chosen"]})
        return dec

    def export(self, candidate_fp: str | None) -> dict:
        return self.section("selection", lambda: self._export(candidate_fp))

    def _export(self, candidate_fp: str | None) -> dict:
        """Validated GeoTIFF of the selected candidate ONLY: selection PASS, the caller's fingerprint equals the
        selected recipe's, the plan audit passed. Re-runs reuse the file of the same recipe."""
        from ..gems3 import submission as SUB
        from . import export_parity as EP
        from . import segment_recovery as SR

        dec = self.read_json("selection", "decision") or {}
        if not dec.get("replace_champion"):
            raise RuntimeError("export refused: selection did not PASS (rc.select())")
        fp = dec.get("candidate_fp")
        if not fp or not candidate_fp or len(str(candidate_fp)) < 8 or not str(fp).startswith(str(candidate_fp)):
            raise RuntimeError(f"export refused: pass the selected candidate fingerprint ({fp}); got {candidate_fp!r}")
        pol = self.read_json("export_parity", "export_policy") or {}
        rec = pol.get("recipe") if (pol.get("recipe") or {}).get("fingerprint") == fp else None
        if rec is None:
            raise RuntimeError("export refused: no planned recipe with that fingerprint")
        if (pol.get("plan_audit") or {}).get("status") != "PASS":
            raise RuntimeError(f"export refused: plan audit {pol.get('plan_audit')}")
        q = EP.w2_policy_map(self.p6, rec) if rec["kind"] == "w2_policy" else SR.full_region_map(self, rec)
        led = [SUB.area_ledger(q, self.p6.ctx.data, 0, f"{rec['candidate']} {rec['policy']['rep']} "
                               f"@ {rec['policy']['frac']:g} blend {rec['policy']['blend']:g}")]
        info = EP.write_validated(self.p6, rec, q, led, float(self.cfg["gates"]["budget_tolerance"]))
        aud = info.get("audit") or {}
        self.save_json({**aud, "submission": info.get("submission"), "reused": info.get("reused"),
                        "uploaded": False, "recipe_fp": fp}, "export_parity", "export_audit")
        if aud.get("status") == "PASS" and not info.get("reused"):
            from . import probes

            path = self.p6.ws.root / "submissions" / info["submission"]
            row = probes._row(self.p6, "RECOVERY", path, info, self.bst.sel, f"recovery {rec['candidate']}")
            row.update(offline_disc_dti=(self.read_json("export_parity", "decision") or {}).get(
                "chosen_summary", {}).get("disc_dti"))
            probes.update_logs(self.p6, pd.DataFrame([row]))
        self.p6.push()
        return {"submission": info.get("submission"), "audit_status": aud.get("status"), "sha256": aud.get("sha256"),
                "failed": aud.get("failed"), "uploaded": False}

    # ------------------------------------------------------------------------------- reports
    def report(self) -> dict:
        return self.section("reports", self._report)

    def stage_table(self) -> pd.DataFrame:
        rows = []
        for st in STAGE_DIRS[:-1]:
            s = self.read_json(st, "status") or {}
            m = self.read_json(st, "manifest") or {}
            rows.append({"stage": st, "status": s.get("status", "PENDING"), "reason": s.get("reason", ""),
                         "fingerprint": (m.get("fingerprint") or "")[:12], "missing_outputs":
                         ";".join(missing_outputs(self.root / st, st)), "updated": s.get("updated"),
                         "source_commit": s.get("source_commit")})
        return pd.DataFrame(rows)

    def registry(self) -> pd.DataFrame:
        from .boost_audit import _KIND, file_hash

        rows = []
        for p in sorted(self.root.rglob("*")) if self.root.exists() else []:
            if p.is_file():
                rows.append({"stage": p.relative_to(self.root).parts[0], "path": p.relative_to(self.p6.ws.root)
                             .as_posix(), "kind": _KIND.get(p.suffix, "other"), "bytes": p.stat().st_size,
                             "sha256": file_hash(p, 64)})
        return pd.DataFrame(rows)

    def _report(self) -> dict:
        from .boost_audit import md_table

        stage = "reports"
        t0 = self._begin(stage, "report")
        ST_ = self.stage_table()
        self.save_table(ST_, "experiment_status", stage)
        x1, x2, x3 = (self.read_json(s, "decision") for s in ("export_parity", "candidate_recall", "segments"))
        geo, x5, sel = (self.read_json(s, "decision") for s in ("geology", "robustness", "selection"))
        C = self.load_table("policy_comparison", "export_parity")
        U = self.load_table("oracle_upper_bound", "candidate_recall")
        H = self.load_table("hybrid_comparison", "segments")
        P = self.load_table("train_inference_parity", "export_parity")
        pend = "_PENDING (stage not run or outputs missing)_\n"
        refs = pd.DataFrame([{"candidate": k, **v} for k, v in self.cfg["references"]["boost_benchmark"].items()])
        L = ["# GEMS P6.1 recovery report", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')}; code {RECOVERY_CODE_REV}. Offline nested numbers only; "
             "nothing here is a leaderboard score. Absent stages are PENDING, never estimated.", "",
             "## Stage status", md_table(ST_, ["stage", "status", "missing_outputs", "reason"]),
             "## Best validated candidate", (f"`{sel.get('chosen')}` (slot-worthy: {sel.get('slot_worthy')}, "
                                             f"fingerprint `{sel.get('candidate_fp')}`)\n" if sel and sel.get("chosen")
                                             else ("No candidate passed the gate.\n" if sel else pend)),
             "## Matched benchmark (boost, recorded)", md_table(refs),
             "## Discovery / geographic trade-off (deployable policies, X1)",
             md_table(C[C.objective == (x1 or {}).get("objective")], ["policy", "role", "deployable", "disc_dti",
                                                                      "disc_worst", "disc_worst_zone", "geo_dti",
                                                                      "systems_gained", "systems_lost"])
             if len(C) and x1 else pend,
             (f"Chosen policy `{x1.get('chosen_policy')}` ({x1.get('objective')}), verdict **{x1.get('verdict')}**; "
              f"META (choosing the policy, nested): {json.dumps(x1.get('meta_nested'), default=str)}\n" if x1 else ""),
             "## Worst-zone results (X5)", md_table(pd.DataFrame((x5 or {}).get("weak_zone_by_policy") or []),
                                                    ["source_stage", "policy", "objective", "task", "weakest_zone",
                                                     "weakest_dti"]) if x5 else pend,
             "## System-level recovery (X3)", md_table(H, ["candidate", "decision", "disc_dti", "disc_worst", "geo_dti",
                                                           "systems_gained", "systems_lost", "fp_increase", "reasons"])
             if len(H) else pend,
             "## Candidate-recall ceiling (X2; oracle = diagnostic)", md_table(U) if len(U) else pend,
             "## Failure taxonomy (X2)", (f"Verdict **{x2.get('verdict')}**, dominant `{x2.get('dominant_group')}`; "
                                          f"shares by length {json.dumps(x2.get('share_by_length'), default=str)}; "
                                          f"sensitivity rows {len(x2.get('sensitivity') or [])}\n" if x2 else pend),
             "## Parity findings (X1.3)", md_table(P.groupby(["check", "status"]).size().reset_index(name="rows"))
             if len(P) else pend,
             "## Remaining dependencies (X4)", (f"Repair status **{geo.get('repair_status')}**; blockers "
                                                f"{json.dumps(geo.get('blockers'), default=str)}; S1: "
                                                f"{geo.get('s1_decision')}\n" if geo else pend),
             "## Next experiment", next_direction(x2, x1, geo) + "\n"]
        txt = "\n".join(L)
        self.p6.ws.save_text(txt, f"{self.sub(stage)}/recovery_report.md")
        self.save_table(self.registry(), "artifact_registry", stage)
        self._finish(stage, "report", t0, {}, {"next": next_direction(x2, x1, geo)})
        return {"status": ST_, "markdown": txt}


_METHOD = {"preflight": "preflight", "export_parity": "export_parity", "candidate_recall": "candidate_recall",
           "segments": "segment_recovery", "geology": "geology_audit", "robustness": "robustness",
           "selection": "select", "reports": "report"}


def startup_check(requested_ref: str, head: str, expected: dict, boost_config: dict | None = None,
                  recovery_config: dict | None = None) -> dict:
    """Boost start-up check + the recovery code revision and modules (raises on any mismatch)."""
    import importlib

    from ..pipeline.m6 import fingerprint
    from .boost_audit import startup_check as boost_check

    rep = boost_check(requested_ref, head, {k: v for k, v in expected.items() if k != "recovery_code_rev"},
                      boost_config)
    bad = []
    if expected.get("recovery_code_rev") and RECOVERY_CODE_REV != expected["recovery_code_rev"]:
        bad.append(f"recovery code rev {RECOVERY_CODE_REV} != expected {expected['recovery_code_rev']}")
    for m in ("export_parity", "candidate_recall", "segment_recovery", "boost_diagnostics"):
        try:
            importlib.import_module(f"gems_bench.p6.{m}")
        except Exception as e:
            bad.append(f"gems_bench.p6.{m} import failed: {type(e).__name__}: {e}")
    cfg = load_recovery_config(recovery_config)
    rep.update(recovery_code_rev=RECOVERY_CODE_REV, recovery_config_fp=fingerprint(RECOVERY_VERSION, cfg)[:12])
    if bad:
        raise RuntimeError("recovery start-up check failed:\n  " + "\n  ".join(bad))
    return rep
