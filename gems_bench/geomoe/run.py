"""Colab orchestrator of the discovery-first GeoMoE experiment (notebooks/colab/GEMS_DISCOVERY_GEOMOE.ipynb).

    F = Factory(FACTORY_CONFIG, HEAD)                  # the existing submission factory = the frozen engine
    G = DiscoveryGeoMoE(CONFIG, HEAD, factory=F)
    G.run()                                            # every stage in order (FAST_DIAGNOSTIC stops after [06])

Stages and their dependencies (``decide.DEPENDS``). A stage runs only when every upstream stage is COMPLETE,
REUSED_VALID_CACHE or SKIPPED_BY_VALID_DECISION; otherwise it is recorded as BLOCKED_UPSTREAM (compute stages) or
NOT_EVALUATED (decision stages) and nothing is concluded from it. A cached result is reused only when its fingerprint
matches AND every required output exists, is readable and has its required columns (REUSED_VALID_CACHE); a FAILED or
blocked manifest is never reused.
    [00] tests        synthetic self-tests + the pytest file (a failure = TESTS_FAILED: modeling and export blocked)
    [01] attach       factory self-tests + preflight + attach (frozen B0 / B1 / W2 engine on the CPU, read-only)
    [02] audit        reuse matrix, source compatibility, recorded protocols, optional limited G02 backfill
    [03] scenarios    V1 / V2 / V3 units, V2 withheld events, static leakage checks
    [04] candidates   per unit, validated cache: layers -> pools -> inventory (saved) -> coverage parts (saved)
    [05] oracle       frozen submitted-W2 baseline + policy parity + metric equivalence, coverage, kernel oracle,
                      bottleneck reconciliation, V2 coverage by subtype
    [06] decision     per-scenario generation / ranking verdicts (gate of [10])
    [07] experts      expert scores, availability, cross-fitted normalisation, consistent-availability rule
    [08] static       A_seg, B (+ ablations B_noE4, B_ctxW2, B_E3_avail)
    [09] rules        C_all + one ablation per rule
    [10] learned      D when [06] allows it
    [10b] compare     fixed-policy evaluation of every model, paired bootstrap, effects, V2 subtypes, selection
    [11] deployment   zone-stitched vs fold-average vs footprint-rank (V3, cross-fitted)
    [12] export       validated GeoTIFF of the selected model (or EXPLORATORY), factory / P6 writer
    [14] integrity    audit/stage_integrity.csv (which conclusions are allowed)
    [13] reports      figures + reports from persisted tables only
"""
from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import GEOMOE_CODE_REV, GEOMOE_VERSION
from . import candidates as CA
from . import decide as DE
from . import evaluate as EV
from . import experts as EX
from . import gating as GA
from . import scenarios as SC
from .decide import (BLOCKED, BLOCKED_UPSTREAM, COMPLETE, FAILED, NOT_EVALUATED, PRODUCED, REUSED, SKIPPED_OK,
                     TESTS_FAILED, VALID)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "geomoe" / "discovery_geomoe.yaml"
UNIT_SCHEMA = "dgm.2-unit"          # enters the stage-04 fingerprint: bump only when the unit artifacts change
STAGES = ("00_tests", "01_attach", "02_audit", "03_scenarios", "04_candidates", "05_oracle", "06_decision",
          "07_experts", "08_static", "09_rules", "10_learned", "10b_compare", "11_deployment", "12_export",
          "14_integrity", "13_reports")
LABELS = {"00_tests": "Synthetic tests", "01_attach": "Frozen artifacts", "02_audit": "Validation audit",
          "03_scenarios": "Scenarios", "04_candidates": "Candidate generation", "05_oracle": "Oracle analysis",
          "06_decision": "Bottleneck diagnosis", "07_experts": "Expert evidence", "08_static": "Static fusion",
          "09_rules": "Geological rules", "10_learned": "Learned GeoMoE", "10b_compare": "Model comparison",
          "11_deployment": "Deployment", "12_export": "Submission", "14_integrity": "Stage integrity",
          "13_reports": "Reports"}
NB_DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "main", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "local_root": "/content/gems_work", "mode": "FAST_DIAGNOSTIC", "reuse_cached": True, "device": "cpu",
    "train_neural_models": False, "backfill_missing_g02": False, "dem": None, "run_synthetic_tests": True,
    "strict_dependency_checks": True, "export_validated_candidate": None, "allow_exploratory_export": None,
    "save_all_outputs": True,
    "experts": {"cnn": True, "w2": True, "magnetic_gravity": True, "g01_g02": True, "dem": "cached_only",
                "network": True},
    "experiments": {"validation_audit": True, "candidate_ceiling": True, "static_fusion": True, "rule_gate": True,
                    "learned_gate": "conditional", "deployment_comparison": True},
    "export": {"enabled": True, "only_validated": True, "exploratory": False, "manual_upload": True},
    "save_png": True, "sync_to_drive": True, "overrides": {},
}
MODES = ("FAST_DIAGNOSTIC", "FULL", "RESUME", "DIAGNOSTIC", "REPORT_ONLY")
FAST_STAGES = ("00_tests", "01_attach", "02_audit", "03_scenarios", "04_candidates", "05_oracle", "06_decision")
EXPERT_SWITCH = {"cnn": ("cnn_b0", "cnn_b1", "fusion"), "w2": ("w2",), "g01_g02": ("g02c", "g02a", "g01_persistence"),
                 "magnetic_gravity": ("mag_hgrad", "grav_hgrad", "mg_cross", "pf_persistence"),
                 "network": ("network", "network_w2ctx")}
REF, DIAG = "A_submitted_W2", "A_W2_diag"
MAIN_MODELS = (REF, "A_W2_policy_reselected", "A_seg", "B", "C_all", "D")
ABLATIONS = {"B_noE4": "noE4", "B_ctxW2": "ctxW2", "B_E3_avail": "E3_where_available"}
# required outputs per stage: (relative path, required columns) - validated before any reuse
OUTPUTS = {
    "00_tests": [("audit/selftests.csv", ("test", "status")), ("audit/pytest_output.txt", None)],
    "02_audit": [("audit/component_inventory.csv", ("component",)), ("audit/source_compatibility.csv", None)],
    "03_scenarios": [("validation/scenarios.csv", ("scenario", "zone")), ("validation/v2_events.csv", None)],
    "04_candidates": [("candidates/candidate_inventory.parquet", ("candidate_id", "scenario", "zone")),
                      ("experts/expert_features.parquet", ("scenario", "zone", "unit", "label", "n_px")),
                      ("candidates/coverage_parts.parquet", ("scenario", "unit", "system_id", "pool", "covered")),
                      ("candidates/source_provenance.csv", ("unit", "source", "status"))],
    "05_oracle": [("gating/grid_A_W2.csv", ("scenario", "model", "rep", "frac", "tp_w")),
                  ("validation/frozen_w2_baseline.csv", ("scenario", "zone", "dti")),
                  ("candidates/oracle_coverage.csv", ("scenario", "n_systems", "cov_inventory")),
                  ("candidates/kernel_weighted_oracle.csv", ("scenario", "order", "frac", "tp_w"))],
    "06_decision": [("candidates/coverage_decision.json", None)],
    "07_experts": [("experts/expert_scores.parquet", ("scenario", "zone", "label")),
                   ("experts/source_normalization.json", None)],
    "08_static": [("gating/grid_B.csv", ("scenario", "model", "tp_w")), ("gating/scores_B.parquet", ("unit", "score"))],
    "09_rules": [("gating/grid_C.csv", ("scenario", "model")), ("gating/rule_gate_results.csv", None)],
    "10_learned": [("gating/grid_D.csv", ("scenario", "model")), ("gating/scores_D.parquet", ("unit", "score"))],
    "10b_compare": [("gating/selection.json", None), ("gating/model_summary.csv", ("scenario", "model", "dti"))],
    "11_deployment": [("gating/deployment_comparison.csv", ("construction", "dti"))],
    "12_export": [("submissions/submission_manifest.json", None)],
    "14_integrity": [("audit/stage_integrity.csv", ("stage", "actual_status"))],
    "13_reports": [("reports/EXECUTION_STATUS.md", None)],
}


class ConfigError(ValueError):
    pass


def notebook_config(CONFIG: dict | None) -> dict:
    """Notebook CONFIG with defaults. New top-level keys (dgm.2) map onto the dgm.1 nested ones, so a dgm.1 CONFIG
    still works: ``dem`` -> experts.dem, ``export_validated_candidate`` -> export.enabled,
    ``allow_exploratory_export`` -> export.exploratory."""
    C = CONFIG or {}
    c = {**NB_DEFAULTS, **C}
    for k in ("experts", "experiments", "export"):
        c[k] = {**NB_DEFAULTS[k], **(C.get(k) or {})}
    if c.get("dem") is not None:
        c["experts"]["dem"] = c["dem"]
    if c.get("export_validated_candidate") is not None:
        c["export"]["enabled"] = bool(c["export_validated_candidate"])
    if c.get("allow_exploratory_export") is not None:
        c["export"]["exploratory"] = bool(c["allow_exploratory_export"])
    c["mode"] = str(c["mode"]).upper()
    errs = []
    if c["mode"] not in MODES:
        errs.append(f"mode {c['mode']!r} not in {MODES}")
    if str(c["device"]).lower() != "cpu":
        errs.append("device must be 'cpu': no neural network is trained or re-run here")
    if c["train_neural_models"]:
        errs.append("train_neural_models=True is refused: this experiment reuses frozen CNN / W2 predictions only")
    if not c["export"]["manual_upload"]:
        errs.append("export.manual_upload must stay True: nothing is uploaded automatically")
    if c["experts"]["dem"] not in ("cached_only", "off", False):
        errs.append("dem: 'cached_only' or 'off' (a full DEM download is never triggered here)")
    if c["experiments"]["learned_gate"] not in ("conditional", True, False):
        errs.append("experiments.learned_gate: 'conditional' | True | False")
    off = [k for k in ("validation_audit", "candidate_ceiling", "static_fusion") if not c["experiments"][k]]
    if off:
        errs.append(f"experiments {off} cannot be disabled: the audit, the coverage ceiling and static fusion are the "
                    "baselines every other result is judged against")
    if errs:
        raise ConfigError("; ".join(errs))
    if c["mode"] == "RESUME":
        c["reuse_cached"] = True
    return c


def load_config(overrides: dict | None = None) -> dict:
    import yaml

    from ..p6.run import _merge

    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        return _merge(yaml.safe_load(f), overrides or {})


def factory_config(c: dict) -> dict:
    """The factory CONFIG used only as the frozen-engine loader (no candidate is exported by the factory here)."""
    return {**{k: c[k] for k in ("repo", "ref", "drive_root", "local_root", "reuse_cached")}, "run_mode": "DIAGNOSTIC",
            "candidates": ["w2_champion", "b0_b1_fallback"], "save_png": False, "sync_to_drive": c["sync_to_drive"]}


def _key(su_key: str) -> str:
    return su_key.replace(":", "_")


def pytest_failures(text: str) -> list[str]:
    """Failing test ids from pytest output (``FAILED path::test - reason`` and ``ERROR`` lines)."""
    return [m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR) (\S+)", text or "", re.M)]


def tests_verdict(selftests: pd.DataFrame | None, returncode: int | None, output: str) -> tuple[str, list[str]]:
    """TESTS_FAILED on any failed critical self-test, a nonzero pytest exit code (incl. 'no tests collected' or a
    missing pytest) or an unreadable result; COMPLETE otherwise."""
    why = []
    if selftests is None or not len(selftests):
        why.append("self-tests did not run")
    else:
        bad = selftests[(selftests.status != "PASS") & selftests.critical.astype(bool)]
        why += [f"self-test {t}" for t in bad.test]
    if returncode is not None and returncode != 0:
        f = pytest_failures(output)
        why += [f"pytest {x}" for x in f] or [f"pytest exit code {returncode}: {(output or '')[-300:]}"]
    return (TESTS_FAILED if why else COMPLETE), why


def affected_file(msg: str) -> str | None:
    m = re.search(r"([\w./\\-]+\.(?:npz|parquet|csv|json|npy|tif|yaml|py))", msg or "")
    return m.group(1) if m else None


class DiscoveryGeoMoE:
    # stage tables as class attributes: a subclass (gems_bench.reset) reuses the same machinery with its own stages
    STAGES, OUTPUTS, LABELS = STAGES, OUTPUTS, LABELS
    DEPENDS, DECISION_STAGES = DE.DEPENDS, DE.DECISION_STAGES
    REPORT_STAGES = ("13_reports", "14_integrity")
    STAGE_IN = ("manifests", "audit", "validation", "candidates", "experts", "gating", "submissions")

    def __init__(self, CONFIG: dict, head: str, factory=None):
        from ..artifacts.sync import DriveSync
        from ..leaderboard_focus.session import Workspace

        self.nb = notebook_config(CONFIG)
        self.cfg = load_config(self.nb.get("overrides"))
        self.head, self.F = head, factory
        self.local, self.drive = Path(self.nb["local_root"]), Path(self.nb["drive_root"])
        self.out = self.cfg["output_dir"]
        self.sync = DriveSync(self.local, self.drive) if self.nb["sync_to_drive"] else None
        self.ws = Workspace(self.local / self.out, self.sync, self.local, save_figures=bool(self.nb["save_png"]),
                            save_parquet=True)
        self.status: dict[str, str] = {s: "NOT_RUN" for s in self.STAGES}
        self.fps: dict[str, str] = {}
        self.errors: list[dict] = []
        self.fg = self.bst = self.p6 = self.ctx = self.m7 = self.sh = None
        self.units: list | None = None
        self._dem = self._pf2 = None
        self._zm: dict = {}
        if self.sync is not None:                          # earlier small outputs only (never predictions / tifs)
            subs = [f"{self.out}/{s}" for s in self.STAGE_IN if (self.drive / self.out / s).exists()]
            if subs:
                self.sync.stage_in(subs)

    # =========================================================================== plumbing
    def say(self, msg: str) -> None:
        print(f"[geomoe] {msg}")

    def fp(self, *parts) -> str:
        from ..pipeline.m6 import fingerprint

        return fingerprint(GEOMOE_VERSION, *parts)

    def manifest(self, stage: str) -> dict:
        return self.ws.read_json(f"manifests/stages/{stage}.json") or {}

    def save_parquet(self, df: pd.DataFrame, rel: str) -> None:
        from ..artifacts.atomic import atomic_save
        from ..pipeline.m6 import _parquet_safe

        p = self.ws.path(rel)
        atomic_save(p, lambda t: _parquet_safe(df if df is not None else pd.DataFrame()).to_parquet(t, index=False))

    def load_parquet(self, rel: str) -> pd.DataFrame:
        p = self.ws.path(rel)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def save_npz(self, rel: str, **arrays) -> None:
        from ..artifacts.atomic import atomic_save

        atomic_save(self.ws.path(rel), lambda t: np.savez_compressed(t, **arrays))

    def check_outputs(self, specs) -> tuple[list[str], list[str]]:
        """(valid outputs, problems): exists, readable, required columns present, non-empty tables."""
        ok, bad = [], []
        for rel, cols in specs:
            p = self.ws.path(rel)
            if not p.exists():
                bad.append(f"{rel}: missing")
                continue
            try:
                if p.suffix == ".csv":
                    T = pd.read_csv(p, nrows=5) if p.stat().st_size else pd.DataFrame()
                elif p.suffix == ".parquet":
                    T = pd.read_parquet(p)
                elif p.suffix == ".json":
                    json.loads(p.read_text(encoding="utf-8"))
                    T = None
                else:
                    T = None
            except Exception as e:
                bad.append(f"{rel}: unreadable ({type(e).__name__})")
                continue
            miss = [c for c in (cols or ()) if T is None or c not in T]
            if miss:
                bad.append(f"{rel}: columns {miss} absent")
            else:
                ok.append(rel)
        return ok, bad

    def upstream_ok(self, stage: str) -> tuple[str | None, list[str]]:
        return DE.dependency_status(stage, self.status, bool(self.nb["strict_dependency_checks"]), self.DEPENDS,
                                    self.DECISION_STAGES)

    def run_stage(self, stage: str, parts: list, fn, *, force: bool = False) -> dict:
        """Dependency check -> validated cache -> isolated run. ``fn() -> (status, info)``."""
        from .._version import provenance

        block, bad = self.upstream_ok(stage)
        if block is not None:
            info = {"upstream_invalid": {s: self.status.get(s) for s in bad},
                    "reason": f"upstream {[f'{s}={self.status.get(s)}' for s in bad]}"}
            return self.record(stage, block, None, info, 0.0, provenance())
        fp = self.fp(stage, *parts)
        self.fps[stage] = fp
        m = self.manifest(stage)
        specs = self.OUTPUTS.get(stage, [])
        if not force and self.nb["reuse_cached"] and m.get("fingerprint") == fp and m.get("status") in (*PRODUCED,
                                                                                                         SKIPPED_OK):
            ok, probs = self.check_outputs(specs)
            if not probs:
                st = SKIPPED_OK if m["status"] == SKIPPED_OK else REUSED
                self.status[stage] = st
                self.progress(stage, st, f"fingerprint {fp[:10]}")
                return {**m, "status": st}
            self.say(f"[{stage}] cache rejected: {probs[:3]} -> recomputed")
        t0 = time.perf_counter()
        try:
            st, info = fn()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            info = {"error": msg, "affected_file": affected_file(msg), "traceback": traceback.format_exc()[-4000:]}
            st = FAILED
            self.errors.append({"stage": stage, **info, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if st in PRODUCED:
            ok, probs = self.check_outputs(specs)
            if probs:
                st, info = FAILED, {**(info or {}), "error": f"required outputs invalid: {probs}",
                                    "affected_file": probs[0].split(":")[0]}
        return self.record(stage, st, fp, info or {}, round(time.perf_counter() - t0, 1), provenance())

    def record(self, stage: str, st: str, fp, info: dict, seconds: float, prov: dict) -> dict:
        m = {"stage": stage, "status": st, "fingerprint": fp, "code_rev": GEOMOE_CODE_REV, "version": GEOMOE_VERSION,
             "commit": self.head, "code": prov, "seconds": seconds,
             "required_outputs": [r for r, _ in self.OUTPUTS.get(stage, [])],
             "depends": list(self.DEPENDS.get(stage, ())),
             "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
        self.ws.save_json(m, f"manifests/stages/{stage}.json")
        self.status[stage] = st
        self.save_run_manifest()
        self.ws.push()
        note = info.get("note") or info.get("reason") or info.get("error") or ""
        self.progress(stage, st, note)
        if st == FAILED:
            print(f"      exception: {info.get('error')}\n      affected file: {info.get('affected_file')}\n"
                  f"      next: fix the cause above, re-run this cell (finished units / stages are reused)")
        return m

    def progress(self, stage: str, st: str, note: str = "") -> None:
        lab = f"[{stage[:3].rstrip('_')}] {self.LABELS.get(stage, stage)} "
        print(f"{lab}{'.' * max(2, 32 - len(lab))} {st}" + (f" | {str(note)[:220]}" if note else ""))

    def ok(self, *stages) -> bool:
        return all(self.status.get(s) in VALID for s in stages)

    def save_run_manifest(self) -> None:
        self.ws.save_json({"code_rev": GEOMOE_CODE_REV, "version": GEOMOE_VERSION, "commit": self.head,
                           "notebook_config": self.nb, "config": self.cfg, "stages": self.status,
                           "fingerprints": self.fps, "n_errors": len(self.errors),
                           "drive_dir": str(self.drive / self.out), "written": time.strftime("%Y-%m-%dT%H:%M:%S")},
                          "manifests/run.json")
        self.ws.save_json(self.errors, "manifests/runtime_errors.json")

    def leak(self, stage: str, rows: list[dict]) -> None:
        """Leakage-audit rows of one stage (status PASS / FAIL / UNVERIFIED); audit/label_leakage_audit.csv is the
        union of every stage's part."""
        P = pd.DataFrame(rows).assign(stage=stage)
        self.ws.save_table(P, f"audit/leakage_parts/{stage}.csv")
        parts = sorted(self.ws.path("audit/leakage_parts").glob("*.csv"))
        self.ws.save_table(pd.concat([pd.read_csv(p) for p in parts if p.stat().st_size], ignore_index=True),
                           "audit/label_leakage_audit.csv")

    # =========================================================================== engine helpers
    @property
    def sys_support(self) -> float:
        return float(self.sh["evaluation"]["system_recovered_support"])

    @property
    def sys_frac(self) -> float:
        return float(self.sh["evaluation"]["system_recovered_frac"])

    def elig(self, u) -> np.ndarray:
        return np.asarray(self.m7.elig(u, 0), bool)

    def score(self, u, q: np.ndarray) -> dict:
        from ..m7.protocol import score_row

        r = score_row(u, q, self.elig(u), self.m7.pkey(u))
        r["dti"] = EV.dti(r["tp_w"], r["fp_w"], r["fn_w"])
        return r

    def descriptor_layers(self) -> tuple:
        return EX.DESCRIPTOR_LAYERS

    def engine_identity(self) -> dict:
        """Frozen inputs every stage depends on: data fingerprint, W2 map hashes, G02 run identity + units."""
        if not hasattr(self, "_ident"):
            w2 = self.bst.w2_inputs() if self.bst is not None else {}
            g = self.bst.g02() if self.bst is not None else None
            self._ident = {"data": self.fg.data_fingerprint() if self.fg is not None else None,
                           "w2_maps": w2.get("maps_sha256"), "w2_rows": w2.get("rows_sha256"),
                           "g02": self.bst.g02_ident() if self.bst is not None else None,
                           "g02_units": sorted(g.raster_units()) if g is not None else [],
                           "experts_switch": self.nb["experts"]}
        return self._ident

    def pf_scale2(self) -> dict:
        """Potential-field horizontal gradients at the second scale (label-free, full grid, cached)."""
        if self._pf2 is None:
            rel = "experts/pf_scale2.npz"
            p = self.ws.path(rel)
            if p.exists():
                with np.load(p) as z:
                    self._pf2 = {k: z[k] for k in z.files}
            else:
                from ..m8 import potential_fields as PFm

                d = self.ctx.data
                out = {}
                for role, b in self.p6.cfg["stack"]["bands"].items():
                    if b in d.band_names:
                        a = np.where(d.valid, np.asarray(d.read_band(b), np.float32), np.nan)
                        out[f"{role}_hgrad"] = np.asarray(PFm.gradient_parts(a, float(
                            self.cfg["pf_multiscale_sigma_px"]))["thdr"], np.float32)
                self.save_npz(rel, **out)
                self._pf2 = out
        return self._pf2

    def dem_mosaic(self):
        """The cached M7 100 m DEM mosaic (staged from Drive when it exists there), else None. Never a download."""
        if self._dem is None:
            self._dem = False
            if self.nb["experts"]["dem"] in ("off", False) or self.cfg["dem"]["mode"] == "off":
                return None
            from ..m7.dem_morph import DemMosaic, mosaic_path

            p = mosaic_path(self.m7)
            if not p.exists():
                rel = p.relative_to(Path(self.m7.local)).as_posix()
                if (self.drive / rel).exists():
                    self.m7.stage_in([rel])
            if p.exists():
                self._dem = DemMosaic(p)
        return self._dem or None

    def dem_layers(self, b) -> dict | None:
        from ..p6.stack import rank_in

        mos = self.dem_mosaic()
        if mos is None:
            return None
        dc = self.cfg["dem"]
        cov = mos.coverage(float(dc["min_coverage"]))[b.sl] & b.ev
        bands = [x for x in dc["bands"] if mos.has(x)]
        if not bands or not cov.any():
            return None
        rk = [rank_in(np.where(cov, mos.band(x)[b.sl], np.nan), b.ev & cov) for x in bands]
        with np.errstate(invalid="ignore"):
            dem = np.nanmean(np.stack(rk), 0).astype(np.float32)
        return {"dem": np.where(cov, dem, np.nan), "observed": np.where(b.ev, cov.astype(np.float32), np.nan),
                "source": f"M7 mosaic {mos.path.name} bands {bands} (coverage >= {dc['min_coverage']})"}

    def zone_mask(self, su) -> np.ndarray:
        """Held-out evaluation zone of ``su`` dilated by the training buffer (full grid)."""
        from ..m7.ranking import _zone_mask

        k = su.base.key
        if k not in self._zm:
            if len(self._zm) > 3:
                self._zm.clear()
            self._zm[k] = _zone_mask(su.unit, self.ctx.data.shape, float(self.cfg["train_buffer_km"]))
        return self._zm[k]

    def release(self, b) -> None:
        for d in (getattr(self.bst, "_raw", {}), getattr(self.bst, "_cnn", {})):
            d.pop(b.key, None)
        b.release()
        gc.collect()

    def enabled_layers(self, L: dict) -> dict:
        off = {x for k, v in self.nb["experts"].items() if k in EXPERT_SWITCH and not v for x in EXPERT_SWITCH[k]}
        return {k: v for k, v in L.items() if k not in off}

    # =========================================================================== [00] tests
    def s00_tests(self) -> dict:
        """Synthetic self-tests + pytest (enforced: TESTS_FAILED blocks candidates, modeling, selection, export)."""
        from . import selftest as STT

        def fn():
            T = STT.run_selftests()
            self.ws.save_table(T, "audit/selftests.csv")
            rc, out = None, "pytest not run (selftests only)"
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", "tests/test_discovery_geomoe.py"],
                               cwd=str(REPO), capture_output=True, text=True)
            rc, out = r.returncode, r.stdout + "\n" + r.stderr
            self.ws.save_text(f"exit code {rc}\n\n{out}", "audit/pytest_output.txt")
            st, why = tests_verdict(T, rc, out)
            print(f"      self-tests {int((T.status == 'PASS').sum())}/{len(T)} PASS; pytest exit code {rc}")
            for w in why:
                print(f"      FAILING: {w}")
            return st, {"failing": why, "pytest_returncode": rc, "note": "; ".join(why)[:300] or "all tests passed",
                        "error": ("tests failed: " + "; ".join(why)[:600]) if why else None,
                        "affected_file": "audit/pytest_output.txt" if why else None}
        if not self.nb["run_synthetic_tests"]:
            self.ws.save_text("not run: run_synthetic_tests=False\n", "audit/pytest_output.txt")
            return self.run_stage("00_tests", ["disabled"], lambda: (SKIPPED_OK, {
                "reason": "run_synthetic_tests=False (recorded: conclusions carry 'tests not run')"}), force=True)
        return self.run_stage("00_tests", [time.strftime("%Y-%m-%dT%H:%M")], fn, force=True)

    # =========================================================================== [01] attach
    def s01_attach(self) -> dict:
        def fn():
            F = self.F
            if F is None:
                from ..leaderboard_focus.factory import Factory

                F = self.F = Factory(factory_config(self.nb), self.head)
            if F.selftests is None:
                F.selftest()
            if F.sess is None:
                F.preflight()
                if not F.attach():
                    return BLOCKED, {"reasons": [str(v) for vs in F.block.values() for v in vs][:6],
                                     "error": "the frozen engine did not attach (factory preflight)",
                                     "affected_file": "outputs/leaderboard_focus/submission_factory/reports/"
                                                      "PREFLIGHT.md"}
            self._bind(F)
            return COMPLETE, {"factory_selftests_ok": F.selftest_ok(), "note": "engine attached"}
        return self.run_stage("01_attach", [self.head], fn, force=True)

    def _bind(self, F) -> None:
        self.fg = F.sess.fg
        self.bst, self.p6 = self.fg.bst, self.fg.p6
        self.ctx, self.m7, self.sh = self.p6.ctx, self.p6.m9.m7, self.p6.m9.sh

    # =========================================================================== [02] audit (+ G02 backfill)
    def g02_backfill(self) -> dict:
        """Opt-in limited backfill of the MISSING per-unit G02 rasters only, with the attached run's frozen config
        (``Boost.backfill``: refused on input drift, existing rasters never rewritten), then pushed to Drive."""
        g = self.bst.g02()
        if g is None:
            return {"status": "UNAVAILABLE", "detail": getattr(self.bst, "g02_error", "no G02 run attached")}
        have = set(g.raster_units())
        need = [f"{u.task}_{u.zone}" for u in self.bst.units() if f"{u.task}_{u.zone}" not in have]
        if not need:
            return {"status": "COMPLETE_ALREADY", "missing": []}
        if not self.nb["backfill_missing_g02"]:
            return {"status": "MISSING_NOT_BACKFILLED", "missing": need,
                    "detail": "set backfill_missing_g02=True to compute only these units with the frozen G02 config"}
        t0 = time.perf_counter()
        try:
            done = self.bst.backfill(len(need))
            if hasattr(g, "push"):
                g.push()
        except Exception as e:
            return {"status": "BACKFILL_FAILED", "missing": need, "detail": f"{type(e).__name__}: {e}"}
        if hasattr(self, "_ident"):
            del self._ident
        return {"status": "BACKFILLED", "units": done, "missing_after": sorted(set(need) - set(done)),
                "seconds": round(time.perf_counter() - t0, 1)}

    def s02_audit(self) -> dict:
        def fn():
            from ..leaderboard_focus.factory import CANDIDATES

            bf = self.g02_backfill()
            self.ws.save_json(bf, "audit/g02_backfill.json")
            ident = self.engine_identity()
            g02_units = ident["g02_units"]
            mos = self.dem_mosaic()
            units = self.bst.units()
            w2_ok = {u.key: (self.p6.ws.root / "stack" / "maps" / f"{u.task}_{u.zone}.npy").exists() for u in units}
            rows = [
                ("B0 / B1 honest CNN maps", True, "yes", "yes", "OOF on geo folds; discovery maps on disc splits",
                 "E1 sources"),
                ("W2 cross-fitted held-out maps", all(w2_ok.values()), "yes", "yes", "per-unit ranks only",
                 "E1 source, A baselines"),
                ("W2 unit features", ident.get("w2_rows") is not None, "yes", "yes", "one PF scale", "expert layers"),
                ("G01 / G02 rasters", bool(g02_units), "yes (FAST_RESEARCH)", "yes",
                 f"units with rasters {g02_units}; backfill {bf.get('status')}", "E3 (consistent-availability rule)"),
                ("Potential fields", True, "yes", "yes", "second scale computed here", "E2"),
                ("Cached DEM mosaic", mos is not None, "pilot", "cached only", "never downloaded", "E4 dem"),
                ("Known-network geometry", True, "yes", "yes", "permitted known per scenario", "E4 network + tips"),
                ("Submission factory writer / validator", True, "yes", "yes", "zone-stitched only", "stage 12"),
                ("Official scorer (FastScorer)", True, "yes", "yes", "OFFICIAL_EXCLUDE_SUPPORTS_TP: repo reading",
                 "every score")]
            self.ws.save_table(pd.DataFrame(rows, columns=["component", "available", "executed", "reusable",
                                                           "limitation", "action"]), "audit/component_inventory.csv")
            sc = self.F.ws.load_table("diagnostics/source_compatibility.csv")
            extra = pd.DataFrame([{"source": "geomoe", "check": f"W2 map {k}", "status": "PASS" if v else "ABSENT"}
                                  for k, v in w2_ok.items()] +
                                 [{"source": "geomoe", "check": f"G02 raster {u.task}_{u.zone}",
                                   "status": "PASS" if f"{u.task}_{u.zone}" in g02_units else "ABSENT"} for u in units])
            self.ws.save_table(pd.concat([sc, extra], ignore_index=True), "audit/source_compatibility.csv")
            hist = [{"candidate": cid, "protocol": s["historical"].get("protocol"),
                     "disc_dti": s["historical"].get("disc_dti"),
                     "geo_dti": s["historical"].get("geo_dti", s["historical"].get("geo_dti_separate")),
                     "source": s["historical"].get("source"), "note": s["historical"].get("note", ""),
                     "comparable_to_this_run": "no (recorded, other protocol)"} for cid, s in CANDIDATES.items()]
            hist += [{"candidate": "w2_champion", "protocol": "public leaderboard", "public_dti": 0.0942,
                      "source": "user-recorded", "note": "external reference, never a tuning target"},
                     {"candidate": "M7 generator", "protocol": "label-free candidate coverage",
                      "note": "~12.5 % of systems", "comparable_to_this_run": "re-measured: audit/"
                      "candidate_bottleneck_reconciliation.csv"},
                     {"candidate": "P6.1 X2", "protocol": "missed-length taxonomy (learned pool <= 5 %)",
                      "note": "82.9 % ranking-limited", "comparable_to_this_run": "re-measured: audit/"
                      "candidate_bottleneck_reconciliation.csv"}]
            self.ws.save_table(pd.DataFrame(hist), "audit/protocol_comparison.csv")
            return COMPLETE, {"g02_units": g02_units, "g02_backfill": bf, "dem": mos is not None,
                              "note": f"G02 units {g02_units} (backfill {bf.get('status')}); DEM "
                                      f"{'present' if mos else 'ABSENT (cached_only)'}"}
        return self.run_stage("02_audit", [self.head, self.nb["backfill_missing_g02"]], fn, force=True)

    # =========================================================================== [03] scenarios
    def s03_scenarios(self) -> dict:
        from ..m7.workspace import mask_hash

        if self.ok("01_attach"):
            self.units = SC.build(self)                 # deterministic (seeded); rebuilt every session
        th = {su.key: mask_hash(su.unit.eval_truth()) for su in (self.units or [])}

        def fn():
            lab = np.asarray(self.ctx.data.labels, bool)
            self.ws.save_table(SC.scenario_table(self.units, self.cfg), "validation/scenarios.csv")
            self.ws.save_table(pd.DataFrame([{"unit": su.key, "scenario": su.scenario, "zone": su.zone,
                                              "row0": su.unit.sl[0].start, "row1": su.unit.sl[0].stop,
                                              "col0": su.unit.sl[1].start, "col1": su.unit.sl[1].stop,
                                              "truth_hash": th[su.key], "base": su.base.key} for su in self.units]),
                               "validation/folds.csv")
            E = pd.concat([su.events for su in self.units if su.scenario == "V2" and len(su.events)] or
                          [pd.DataFrame(columns=["event_id", "class", "px", "length_km", "zone"])], ignore_index=True)
            self.ws.save_table(E, "validation/v2_events.csv")
            rows = []
            for su in self.units:
                u = su.unit
                for c in SC.leakage_checks(su.key, u.eval_truth(), su.known & u.ev, u.exclude,
                                           lab[u.sl] if su.scenario == "V2" else None):
                    rows.append({**c, "scenario": su.scenario, "status": "PASS" if c["ok"] else "FAIL",
                                 "method": "explicit pixel test"})
            rows.append({"unit": "*", "scenario": "*", "check": "V2 classes exclude lateral shifts", "ok":
                         "lateral_shift" not in self.cfg["v2"]["classes"], "status": "PASS" if "lateral_shift" not in
                         self.cfg["v2"]["classes"] else "FAIL", "method": "config", "detail": str(
                             self.cfg["v2"]["classes"])})
            self.leak("03_scenarios", rows)
            bad = [r for r in rows if r["status"] == "FAIL"]
            return (COMPLETE if not bad else FAILED), {
                "truth_hashes": th, "leakage_failures": bad, "error": f"leakage checks failed: {bad[:3]}" if bad
                else None, "note": f"{len(self.units)} units; {len(rows) - len(bad)}/{len(rows)} leakage checks PASS"}
        return self.run_stage("03_scenarios", [self.engine_identity() if self.bst is not None else None,
                                               self.cfg["scenarios"], self.cfg["v2"], th], fn)

    # =========================================================================== [04] candidates
    def cand_dir(self) -> str:
        return f"candidates/units/{self.fps['04_candidates'][:10]}"

    def unit_files(self, su) -> dict:
        k = f"{self.cand_dir()}/{_key(su.key)}"
        return {"npz": f"{k}.npz", "segments": f"{k}_segments.parquet", "prov": f"{k}_prov.parquet",
                "dedup": f"{k}_dedup.parquet", "parts": f"{k}_parts.json", "coverage": f"{k}_coverage.parquet",
                "attrs": f"{k}_attrs.parquet", "unique": f"{k}_unique.parquet", "bands": f"{k}_bands.parquet",
                "oracle": f"{k}_oracle.parquet"}

    def unit_fp(self, su) -> str:
        return self.fp("unit", self.fps["04_candidates"], su.key, (self.manifest("03_scenarios").get("truth_hashes")
                                                                    or {}).get(su.key))

    def unit_valid(self, su, part: str) -> tuple[bool, str]:
        """Is the cached inventory (``part='inventory'``) or coverage (``'coverage'``) of one unit reusable?"""
        f = self.unit_files(su)
        meta = self.ws.read_json(f["parts"]) or {}
        if meta.get("unit_fp") != self.unit_fp(su):
            return False, "unit fingerprint differs or parts.json absent"
        if part == "inventory":
            try:
                with np.load(self.ws.path(f["npz"])) as z:
                    if z["lab"].shape != su.unit.ev.shape or z["known_dist_km"].shape != su.unit.ev.shape:
                        return False, "label raster shape differs from the unit crop"
                    if "shape" not in z.files:
                        return False, "pools absent"
            except Exception as e:
                return False, f"npz unreadable: {type(e).__name__}"
            need = {"segments": ("label", "n_px", "unit", "row_full"), "prov": ("unit", "source", "status")}
        else:
            if meta.get("coverage") != "COMPLETE":
                return False, "coverage not completed"
            need = {"coverage": ("system_id", "pool", "covered"), "attrs": ("system_id",), "unique": ("system_id",),
                    "bands": ("pool", "band"), "oracle": ("order", "frac")}
        for nm, cols in need.items():
            p = self.ws.path(f[nm])
            if not p.exists():
                return False, f"{nm} missing"
            try:
                T = pd.read_parquet(p)
            except Exception as e:
                return False, f"{nm} unreadable: {type(e).__name__}"
            if len(T) and [c for c in cols if c not in T]:
                return False, f"{nm} lacks columns"
        return True, "valid"

    def lab(self, su) -> np.ndarray:
        with np.load(self.ws.path(self.unit_files(su)["npz"])) as z:
            return z["lab"]

    def dist_km(self, su) -> np.ndarray:
        with np.load(self.ws.path(self.unit_files(su)["npz"])) as z:
            return z["known_dist_km"].astype(np.float32)

    def pools(self, su) -> dict:
        with np.load(self.ws.path(self.unit_files(su)["npz"])) as z:
            return CA.unpack_pools(z)

    def unit_leakage(self, su, L: dict, prov: dict) -> list[dict]:
        """Explicit per-unit checks from the actual loaded layers / provenance (UNVERIFIED where nothing can be
        inspected from here)."""
        u = su.unit
        exp = "oof" if su.base.task == "geo" else f"discovery_pred:{su.base.zone}"
        rows = []
        for k in ("cnn_b0", "cnn_b1"):
            v = prov.get(k)
            rows.append({"unit": su.key, "scenario": su.scenario, "check": f"{k} map provenance tag == {exp}",
                         "status": "PASS" if v == exp else ("UNVERIFIED" if v is None else "FAIL"), "detail": str(v),
                         "method": "p6.stack.unit_raw provenance (asserted by assert_provenance)"})
        rows.append({"unit": su.key, "scenario": su.scenario, "check": "CNN training labels excluded the held-out "
                     "truth", "status": "UNVERIFIED", "method": "not inspectable here (frozen experiments)",
                     "detail": "relies on the B0 / B1 experiment protocol (OOF / discovery splits)"})
        t = u.eval_truth()
        kd = L.get("known_dist_km")
        if kd is not None and t.any():
            mn = float(np.nanmin(kd[t]))
            rows.append({"unit": su.key, "scenario": su.scenario, "check": "known-distance feature is > 0 on every "
                         "scored truth pixel (truth not in the known context)", "status": "PASS" if mn > 0 else "FAIL",
                         "detail": f"min {mn:.3f} km", "method": "explicit layer test"})
        for k in ("g02c", "g02a", "g01_persistence"):
            v = prov.get(k)
            if v and v != "absent":
                ok = v == f"g02_unit:{su.base.task}_{su.base.zone}"
                rows.append({"unit": su.key, "scenario": su.scenario, "check": f"{k} raster is the base unit's own "
                             "label-free raster", "status": "PASS" if ok else "FAIL", "detail": v,
                             "method": "G02 raster metadata (task, zone, window, labels_stored) checked at load"})
        return rows

    def s04_candidates(self) -> dict:
        parts = [self.fps.get("03_scenarios"), UNIT_SCHEMA, self.cfg["candidates"], self.cfg["experts"],
                 self.cfg["dem"], self.cfg["network_scale_km"], self.cfg["pf_multiscale_sigma_px"],
                 self.cfg["distance_bands_km"], self.cfg["evaluation"]["area_fracs"],
                 self.engine_identity().get("g02_units") if self.bst is not None else None]
        self.fps["04_candidates"] = self.fp("04_candidates", *parts)

        def fn():
            order = sorted(self.units, key=lambda s: (s.base.task, s.base.zone, s.scenario))
            last, leak, n_new, n_reused = None, [], 0, 0
            for su in order:
                f = self.unit_files(su)
                if last is not None and last.key != su.base.key:
                    self.release(last)
                last = su.base
                inv_ok, why = self.unit_valid(su, "inventory") if self.nb["reuse_cached"] else (False, "no reuse")
                if not inv_ok:
                    t0 = time.perf_counter()
                    L, prov = EX.unit_layers(self, su)
                    L = self.enabled_layers(L)
                    out = CA.build_unit(self, su, L)
                    dk = L.get("known_dist_km")
                    dk = np.nan_to_num(dk, nan=99.0).astype(np.float16) if dk is not None else \
                        np.full(out["lab"].shape, 99.0, np.float16)
                    self.save_npz(f["npz"], lab=out["lab"], known_dist_km=dk, shape=np.asarray(out["lab"].shape),
                                  **CA.pack_pools(out["pools"]))
                    self.save_parquet(out["T"], f["segments"])
                    self.save_parquet(out["prov"], f["prov"])
                    self.save_parquet(out["dedup"], f["dedup"])
                    lk = self.unit_leakage(su, L, prov)
                    self.ws.save_json({"unit": su.key, "unit_fp": self.unit_fp(su), "layers": sorted(L),
                                       "provenance": prov, "pool_px": out["pool_px"], "eligible_px": out["eligible_px"],
                                       "n_candidates": int(out["lab"].max()), "leakage": lk, "coverage": "PENDING",
                                       "seconds": round(time.perf_counter() - t0, 1)}, f["parts"])
                    self.say(f"  {su.key}: {int(out['lab'].max())} candidates, {len(out['pool_px'])} pools "
                             f"({round(time.perf_counter() - t0, 1)} s) [inventory saved: {why}]")
                    del L, out
                    gc.collect()
                    self.ws.push()
                    n_new += 1
                else:
                    n_reused += 1
                meta = self.ws.read_json(f["parts"]) or {}
                leak += meta.get("leakage") or []
                cov_ok, _ = self.unit_valid(su, "coverage")
                if not cov_ok:
                    T = self.load_parquet(f["segments"])
                    cv = CA.unit_coverage(self, su, self.lab(su), T, self.pools(su), self.dist_km(su))
                    for nm in ("coverage", "attrs", "unique", "bands", "oracle"):
                        self.save_parquet(cv[nm], f[nm])
                    self.ws.save_json({**meta, "coverage": "COMPLETE"}, f["parts"])
                    su.unit.release()
                    self.ws.push()
            if last is not None:
                self.release(last)
            self.leak("04_candidates", leak)
            self.merge_candidates()
            return COMPLETE, {"n_units_computed": n_new, "n_units_reused": n_reused,
                              "note": f"{n_new} unit(s) computed, {n_reused} reused (validated)"}
        return self.run_stage("04_candidates", parts, fn)

    def merge_candidates(self) -> None:
        cat = (lambda x: pd.concat([a for a in x if len(a)], ignore_index=True) if any(len(a) for a in x)
               else pd.DataFrame())
        F = [self.unit_files(su) for su in self.units]
        T = cat([self.load_parquet(f["segments"]) for f in F])
        if not len(T):
            raise RuntimeError("no candidate segment in any unit: the inventory is empty (nothing to evaluate)")
        self.save_parquet(CA.inventory_view(T), "candidates/candidate_inventory.parquet")
        self.save_parquet(T, "experts/expert_features.parquet")
        for nm, part in (("coverage_parts", "coverage"), ("system_attributes", "attrs"), ("unique_parts", "unique"),
                         ("band_coverage_parts", "bands"), ("oracle_curve_parts", "oracle")):
            self.save_parquet(cat([self.load_parquet(f[part]) for f in F]), f"candidates/{nm}.parquet")
        self.ws.save_table(cat([self.load_parquet(f["prov"]) for f in F]), "candidates/source_provenance.csv")
        self.ws.save_table(self.agreement(T), "candidates/candidate_source_agreement.csv")
        D = cat([self.load_parquet(f["dedup"]) for f in F])
        self.ws.save_table(D.groupby(["unit", "source"]).agg(n=("duplicate", "size"), duplicates=("duplicate", "sum"))
                           .reset_index() if len(D) else pd.DataFrame(), "candidates/source_overlap.csv")
        self.geometries(T)

    @staticmethod
    def agreement(T: pd.DataFrame) -> pd.DataFrame:
        on = [c for c in T.columns if c.startswith("on_")]
        rows = []
        for sc, g in T.groupby("scenario"):
            B = g[on].to_numpy() >= 0.5
            for i, a in enumerate(on):
                na = int(B[:, i].sum())
                for j, b in enumerate(on):
                    rows.append({"scenario": sc, "source_a": a[3:], "source_b": b[3:], "n_a": na,
                                 "share_a_also_b": float((B[:, i] & B[:, j]).sum() / max(na, 1))})
        return pd.DataFrame(rows)

    def geometries(self, T: pd.DataFrame) -> None:
        """candidates/candidate_geometries.gpkg (optional: geopandas; a failure is recorded, never fatal)."""
        try:
            import geopandas as gpd

            t = self.ctx.data.template_profile
            parts = []
            for su in self.units:
                g = T[T.unit == su.key]
                if len(g):
                    gd = CA.to_geodataframe(self.lab(su), g, su.unit.sl, t["transform"], t["crs"])
                    gd["candidate_id"] = su.key + "#" + gd.label.astype(str)
                    parts.append(gd)
            if parts:
                p = self.ws.path("candidates/candidate_geometries.gpkg")
                p.parent.mkdir(parents=True, exist_ok=True)
                gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=t["crs"]).to_file(p, driver="GPKG")
        except Exception as e:
            self.errors.append({"stage": "04_candidates", "error": f"geometry export skipped: {type(e).__name__}: {e}"})
            self.say(f"  candidate_geometries.gpkg skipped ({type(e).__name__}: {e})")

    # =========================================================================== frozen submitted W2
    def frozen_policy(self) -> dict:
        """The submitted W2 operating policy: the factory's evaluated champion row (deploy cell = rep, frac, blend,
        grid candidate; deploy kind), recipe id / sha from the factory manifest when present. Configured fallback
        only when the factory table is absent (recorded as such)."""
        if not hasattr(self, "_frozen"):
            from ..leaderboard_focus.factory import CANDIDATES

            src = CANDIDATES["w2_champion"]["source"]
            C = getattr(self.F, "C", None)
            r = C[C.candidate_id == src] if C is not None and len(C) else pd.DataFrame()
            man = self.F.ws.read_json("manifests/candidates/w2_champion.json") or {} if self.F is not None else {}
            if len(r) and isinstance(r.deploy_cell.iloc[0], str):
                cell, kind = json.loads(r.deploy_cell.iloc[0]), str(r.deploy.iloc[0])
                source = f"factory selection/champion_candidates.csv row {src}"
            else:
                fb = self.cfg["frozen_w2"]
                cell, kind = {"candidate": fb["grid_candidate"], "blend": fb["blend"], "rep": fb["rep"],
                              "frac": fb["frac"]}, fb["deploy"]
                source = "configured fallback configs/geomoe frozen_w2 (factory table absent)"
            self._frozen = {"cell": cell, "kind": kind, "source": source, "recipe_id": man.get("recipe_id"),
                            "file": man.get("file"), "sha256": man.get("sha256"),
                            "recorded": CANDIDATES["w2_champion"]["historical"]}
        return self._frozen

    def frozen_smap(self, su) -> np.ndarray | None:
        """Score map of the submitted policy on one unit, built exactly as the factory evaluated it
        (``p6.final_submission.zone_score`` with the per-zone champion fusion)."""
        from ..p6 import final_submission as FS

        fz = self.frozen_policy()
        b = su.base
        try:
            src = {"fusion": self.bst.fusion(b), "stack": self.bst.stack_rank(b, self.bst.w2_map(b))}
            b0, b1 = self.bst.cnn(b)
            from ..gems3 import representation as R

            src.update(cnn_b0=R.rank01(b0, b.ev), cnn_b1=R.rank01(b1, b.ev))
            return np.where(b.ev, FS.zone_score(fz["kind"], fz["cell"], src), np.nan).astype(np.float32)
        except (FileNotFoundError, ValueError, KeyError):
            return None

    def w2_smap(self, su) -> np.ndarray | None:
        try:
            return np.where(su.base.ev, self.bst.stack_rank(su.base, self.bst.w2_map(su.base)), np.nan).astype(
                np.float32)
        except (FileNotFoundError, ValueError):
            return None

    # =========================================================================== [05] oracle + baselines
    def s05_oracle(self) -> dict:
        def fn():
            fz = self.frozen_policy()
            cell = (str(fz["cell"]["rep"]), float(fz["cell"]["frac"]))
            rows, parity = [], []
            for su in self.units:
                s = self.frozen_smap(su)
                if s is not None:
                    r = EV.eval_grid(self, su, s, REF, cells=[cell])
                    rows += r
                    parity += self.policy_parity(su, s, cell, r[0])
                w = self.w2_smap(su)
                if w is not None:
                    rows += EV.eval_grid(self, su, w, DIAG)
                su.unit.release()
                gc.collect()
            G = pd.DataFrame(rows)
            self.ws.save_table(G, "gating/grid_A_W2.csv")
            fb = G[G.model == REF].assign(policy=json.dumps(fz["cell"]), deploy_kind=fz["kind"],
                                          policy_source=fz["source"], recipe_id=fz.get("recipe_id"),
                                          applied_mask="m7 eligible domain of the unit (V1 / V2: retained faults "
                                                       "masked pixel-exactly)",
                                          construction="factory evaluation construction (zone_score + per-zone "
                                                       "champion fusion), frozen cell")
            self.ws.save_table(fb, "validation/frozen_w2_baseline.csv")
            self.ws.save_table(pd.DataFrame(parity), "audit/baseline_policy_parity.csv")
            self.ws.save_table(self.metric_equivalence(G[G.model == DIAG]), "audit/metric_equivalence.csv")
            self.coverage_tables(G)
            ok = len(fb) == len(self.units)
            return (COMPLETE if ok else FAILED), {
                "frozen_policy": fz, "error": None if ok else f"frozen W2 evaluated on {len(fb)}/{len(self.units)} units",
                "note": f"frozen W2 {fz['cell']} on {len(fb)} units; coverage measured"}
        return self.run_stage("05_oracle", [self.fps.get("04_candidates"), self.cfg["evaluation"],
                                            self.cfg.get("frozen_w2")], fn)

    def policy_parity(self, su, s: np.ndarray, cell, row: dict) -> list[dict]:
        """B3: the frozen-policy map of this unit vs the factory evaluation construction (``Boost.binary``) and vs the
        frozen boost-A grid components of the same cell; W2 map checksum and recipe identity recorded."""
        fz = self.frozen_policy()
        q = EV.policy_q(self, su.unit, s, cell[0], cell[1])
        qe = self.bst.binary(su.base, np.nan_to_num(s, nan=0.0), cell[0], cell[1])
        from ..p6 import export_parity as EP

        ma = EP.map_agreement(np.asarray(q) > 0, qe) if su.scenario != "V2" else {"jaccard": None}
        sha = (self.bst.w2_inputs().get("maps_sha256") or {}).get(f"{su.base.task}_{su.base.zone}.npy")
        out = {"unit": su.key, "scenario": su.scenario, "rep": cell[0], "frac": cell[1],
               "blend": fz["cell"].get("blend"), "grid_candidate": fz["cell"].get("candidate"),
               "area_frac_eligible": row.get("area_frac_eligible"), "area_px": row.get("area_px"),
               "jaccard_vs_factory_evaluation_map": ma.get("jaccard"), "w2_map_sha256": sha,
               "recipe_id": fz.get("recipe_id"), "policy_source": fz["source"]}
        B = self.bst.load_table("grid", "benchmark")
        if su.scenario in ("V1", "V3") and len(B):
            task = "disc_official" if su.scenario == "V1" else "geo"
            cand = str(fz["cell"].get("candidate", "")).partition(":")[2] or "w2"
            g = B[(B.task == task) & (B.zone == su.zone) & (B.candidate.astype(str) == cand) & (B.rep == cell[0]) &
                  (np.isclose(B.frac.astype(float), cell[1]))]
            if "blend" in g and fz["cell"].get("blend") is not None:
                g = g[np.isclose(g.blend.astype(float).fillna(-1), float(fz["cell"]["blend"]))]
            if len(g):
                d = max(abs(row[c] - float(g[c].iloc[0])) for c in ("tp_w", "fp_w", "fn_w"))
                out.update(grid_max_abs_diff=d, grid_status="PASS" if d <= 1e-3 * max(1.0, row["tp_w"] + row["fn_w"])
                           else "FAIL")
            else:
                out.update(grid_status="UNVERIFIED (no frozen grid cell)")
        else:
            out.update(grid_status="NOT_APPLICABLE (V2 is new)")
        jac = ma.get("jaccard")
        if out["grid_status"] == "FAIL" or (jac is not None and jac < 0.999):
            out["status"] = "FAIL"
        elif out["grid_status"] == "PASS":
            out["status"] = "PASS"
        else:
            out["status"] = "UNVERIFIED" if jac is None else "PASS"
        return [out]

    def metric_equivalence(self, G: pd.DataFrame) -> pd.DataFrame:
        """W2 pixel-map components vs the frozen boost-A grid (candidate w2) on shared cells."""
        B = self.bst.load_table("grid", "benchmark")
        if not len(B) or not len(G):
            return pd.DataFrame([{"check": "boost-A grid available", "status": "UNAVAILABLE"}])
        B = B[(B.candidate.astype(str) == "w2")]
        g = G[G.scenario.isin(["V1", "V3"])].copy()
        g["task"] = np.where(g.scenario == "V1", "disc_official", "geo")
        m = g.merge(B, on=["task", "zone", "rep", "frac"], suffixes=("", "_frozen"))
        if not len(m):
            return pd.DataFrame([{"check": "shared cells", "status": "UNAVAILABLE", "detail": "no common cell"}])
        return pd.DataFrame([{"task": r.task, "zone": r.zone, "rep": r.rep, "frac": r.frac,
                              "max_abs_diff": max(abs(r.tp_w - r.tp_w_frozen), abs(r.fp_w - r.fp_w_frozen),
                                                  abs(r.fn_w - r.fn_w_frozen)),
                              "status": "PASS" if max(abs(r.tp_w - r.tp_w_frozen), abs(r.fp_w - r.fp_w_frozen),
                                                      abs(r.fn_w - r.fn_w_frozen)) <= 1e-3 * max(
                                  1.0, r.tp_w_frozen + r.fn_w_frozen) else "FAIL"} for r in m.itertuples()])

    def recovered_systems(self, G: pd.DataFrame, model: str) -> pd.DataFrame:
        """Systems recovered by ``model`` at its frozen cell (one row per grid row of that model)."""
        rows = []
        for r in G[G.model == model].itertuples():
            ids = {int(float(x)) for x in str(r.systems).split(";") if x not in ("", "nan")}
            su = next(s for s in self.units if s.key == r.unit)
            rows += [{"scenario": su.scenario, "unit": su.key, "system_id": int(i), "recovered": int(i) in ids}
                     for i in np.unique(su.sysmap[su.sysmap >= 0])]
        return pd.DataFrame(rows, columns=["scenario", "unit", "system_id", "recovered"])

    def coverage_tables(self, G: pd.DataFrame) -> None:
        C = self.load_parquet("candidates/coverage_parts.parquet")
        U = self.load_parquet("candidates/unique_parts.parquet")
        A = self.load_parquet("candidates/system_attributes.parquet")
        O = self.load_parquet("candidates/oracle_curve_parts.parquet")
        Bd = self.load_parquet("candidates/band_coverage_parts.parquet")
        for nm, T in (("coverage_parts", C), ("unique_parts", U), ("system_attributes", A)):
            if not len(T) or "scenario" not in T:
                raise RuntimeError(f"candidates/{nm}.parquet is empty or lacks 'scenario': stage 04 did not produce "
                                   "coverage (no default is substituted)")
        av = EX.availability_table(self.expert_table(), self.cfg["experts"], ["scenario"])
        w2s = self.recovered_systems(G, REF)
        S = DE.coverage_summary(C, U, w2s, av)
        self.ws.save_table(S, "candidates/oracle_coverage.csv")
        self.ws.save_table(O, "candidates/oracle_curves.csv")
        self.ws.save_table(O[O.order == "oracle"].assign(metric="official components (linear 300 m kernel, "
                                                                "pixel-exact mask), DIAGNOSTIC upper bound"),
                           "candidates/kernel_weighted_oracle.csv")
        if len(Bd):
            g = Bd.groupby(["scenario", "pool", "band"])[["truth_px", "covered_px"]].sum().reset_index()
            g["covered_share"] = g.covered_px / g.truth_px.clip(lower=1)
            self.ws.save_table(g, "candidates/coverage_by_distance.csv")
        inv = C[C.pool == "inventory"][["scenario", "unit", "system_id", "near_share", "covered"]]
        M = inv.merge(A.drop(columns=["zone"], errors="ignore"), on=["scenario", "unit", "system_id"], how="left")
        M = M.merge(w2s[["unit", "system_id", "recovered"]].rename(columns={"recovered": "w2_recovered"}),
                    on=["unit", "system_id"], how="left")
        M["w2_recovered"] = M.w2_recovered.astype("boolean").fillna(False).astype(bool)
        M = M.merge(U.drop(columns=["scenario", "zone"], errors="ignore"), on=["unit", "system_id"], how="left")
        M["covered"] = M.covered.astype(bool)
        self.ws.save_table(M[~M.covered], "candidates/missed_systems.csv")
        self.ws.save_table(M[M.covered & ~M.w2_recovered], "candidates/unique_systems.csv")
        self.ws.save_table(self.by_type(M), "candidates/coverage_by_fault_type.csv")
        self.ws.save_table(self.reconciliation(C, w2s, A), "audit/candidate_bottleneck_reconciliation.csv")
        self.ws.save_table(self.v2_subtype_coverage(M, C), "validation/v2_coverage_by_subtype.csv")

    @staticmethod
    def by_type(M: pd.DataFrame) -> pd.DataFrame:
        if not len(M):
            return M
        M = M.assign(cl=M.length_km * M.covered)
        g = M.groupby(["scenario", "subtype", "length_bin"]).agg(
            n_systems=("system_id", "size"), length_km=("length_km", "sum"), covered=("covered", "mean"),
            w2_recovered=("w2_recovered", "mean"), cl=("cl", "sum")).reset_index()
        g["covered_length_share"] = g.cl / g.length_km.clip(lower=1e-9)
        return g.drop(columns="cl")

    @staticmethod
    def v2_subtype_coverage(M: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
        V = M[M.scenario == "V2"]
        if not len(V):
            return pd.DataFrame()
        fam = C[(C.scenario == "V2") & C.pool.astype(str).str.startswith("fam:")]
        F = fam.pivot_table(index=["unit", "system_id"], columns="pool", values="covered", aggfunc="max",
                            fill_value=False).reset_index()
        V = V.merge(F, on=["unit", "system_id"], how="left")
        agg = {"n_events": ("system_id", "size"), "truth_length_km": ("length_km", "sum"),
               "inventory_coverage": ("covered", "mean"), "w2_recovered_share": ("w2_recovered", "mean")}
        for c in [c for c in V.columns if str(c).startswith("fam:")]:
            V[c] = V[c].astype("boolean").fillna(False).astype(bool)
            agg[f"coverage_{c[4:]}"] = (c, "mean")
        return V.groupby("subtype").agg(**agg).reset_index()

    def reconciliation(self, C: pd.DataFrame, w2s: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
        """M7 vs P6 X2 under ONE definition (truth, zones, mask, 300 m, covered share, per-source pool budget,
        system ids). For each pool: system coverage share AND covered length share; X2-sense ranking-limited =
        missed by the frozen W2 but covered by the pool, as a share of missed length."""
        cc = self.cfg["candidates"]
        defs = {"label_free": "M7-like: E2 + E3 lines only", "fam:E1_learned": "X2-like: CNN / W2 / fusion lines",
                "inventory": "unified pool (all families)"}
        rows = []
        for sc, g in C.groupby("scenario"):
            w = w2s[w2s.scenario == sc][["unit", "system_id", "recovered"]]
            a = A[A.scenario == sc][["unit", "system_id", "length_km"]]
            for pool, lab in defs.items():
                h = g[g.pool == pool].merge(w, on=["unit", "system_id"], how="left").merge(
                    a, on=["unit", "system_id"], how="left")
                if not len(h):
                    rows.append({"scenario": sc, "pool": pool, "definition": lab, "status": NOT_EVALUATED})
                    continue
                h["recovered"] = h.recovered.astype("boolean").fillna(False).astype(bool)
                miss = ~h.recovered
                ml = float(h.length_km[miss].sum())
                rows.append({"scenario": sc, "pool": pool, "definition": lab, "n_systems": int(len(h)),
                             "system_coverage": float(h.covered.mean()),
                             "length_coverage": float((h.near_share * h.length_km).sum() / max(h.length_km.sum(), 1e-9)),
                             "missed_by_w2_systems": int(miss.sum()),
                             "ranking_limited_length_share": float(h.length_km[miss & h.covered].sum() / ml) if ml
                             else np.nan,
                             "generation_limited_length_share": float(h.length_km[miss & ~h.covered].sum() / ml) if ml
                             else np.nan, "tolerance_m": 100 * float(cc["tol_px"]),
                             "covered_rule": f">= {cc['covered_share']} of scored pixels within tolerance",
                             "budget": f"{cc['pool_frac']} of eligible per source", "status": "MEASURED"})
        R = pd.DataFrame(rows)
        R["reading"] = ("A discrepancy between rows of one scenario is due to the candidate SOURCE (same truth, "
                        "tolerance, budget, systems); M7's original number also used a density cap and X2 measured "
                        "LENGTH against learned maps at up to 5 % area")
        return R

    # =========================================================================== [06] decision
    def s06_decision(self) -> dict:
        def fn():
            S = self.ws.load_table("candidates/oracle_coverage.csv")
            d = DE.coverage_decision(S, self.cfg["candidates"]["gate"])
            self.ws.save_json(d, "candidates/coverage_decision.json")
            if d["overall"] == NOT_EVALUATED:
                return FAILED, {"decision": d, "error": "coverage table holds no measured scenario",
                                "affected_file": "candidates/oracle_coverage.csv"}
            return COMPLETE, {"decision": d, "note": f"overall {d['overall']}; per scenario "
                              f"{ {k: v['verdict'] for k, v in d['per_scenario'].items()} }; geology "
                              f"{d['geology_status']} unique={d['geology_unique']}"}
        return self.run_stage("06_decision", [self.fps.get("05_oracle"), self.cfg["candidates"]["gate"]], fn)

    # =========================================================================== [07] experts / views
    def expert_table(self) -> pd.DataFrame:
        if getattr(self, "_X_fp", None) != self.fps.get("04_candidates"):
            T = self.load_parquet("experts/expert_features.parquet")
            if not len(T) or "scenario" not in T:
                raise RuntimeError("experts/expert_features.parquet is empty or lacks 'scenario': stage 04 did not "
                                   "complete")
            self._X = pd.concat([T, EX.expert_raw(T, self.cfg["experts"])], axis=1)
            self._X_fp = self.fps.get("04_candidates")
        return self._X

    RULE_DESC = ("mean_mag_hgrad", "mean_grav_hgrad", "mean_g02a", "mean_g02c", "mean_dem")

    def e3_consistent(self, T: pd.DataFrame) -> tuple[bool, list[int]]:
        """Is E3 available (>= half the candidates) in EVERY zone of this scenario? (zones lacking it)"""
        if "avail_E3_structural" not in T:
            return False, sorted(T.zone.unique().tolist())
        z = T.groupby("zone").avail_E3_structural.mean()
        miss = sorted(int(k) for k, v in z.items() if v < 0.5)
        return not miss, miss

    def views(self, su, X: pd.DataFrame | None = None, only_zone: int | None = None, variant: str | None = None) -> dict:
        """Cross-fitted view of one held-out scenario unit: training mask (other zones, outside the buffer),
        normaliser fitted on training rows only, normalised expert matrix S, rule descriptors, PU targets.
        Consistent availability: when E3 is missing in some zone of the scenario it is dropped for EVERY row of that
        scenario (no gate learns from inconsistent availability) unless ``variant='E3_where_available'``.
        Variants: ``noE4`` (E4 removed), ``ctxW2`` (E4 from W2's own known context; tips-only candidates unscored)."""
        X = self.expert_table() if X is None else X
        T = X[X.scenario == su.scenario].reset_index(drop=True)
        drop_rows = np.zeros(len(T), bool)
        if variant == "ctxW2":
            T = T.copy()
            e4 = EX.expert_raw(T, {"E4_terrain_network": ["network_w2ctx", "dem"]})
            T["raw_E4_terrain_network"] = e4["raw_E4_terrain_network"].to_numpy()
            drop_rows = (T.proposed_by.astype(str) == "tips").to_numpy() if "proposed_by" in T else drop_rows
        tr = GA.training_rows(T, su.zone, self.zone_mask(su)) & (T.eligible_share.to_numpy() > 0) & ~drop_rows
        if only_zone is not None:
            tr &= (T.zone.to_numpy() == int(only_zone))
        GA.assert_disjoint(T.zone[tr].unique(), su.zone)
        held = (T.zone == su.zone).to_numpy()
        kn = int(self.cfg["normalization"]["knots"])
        raw = [f"raw_{f}" for f in EX.FAMILIES]
        ne = EX.fit_normalizer(T[tr], raw, kn)
        nd = EX.fit_normalizer(T[tr], [c for c in self.RULE_DESC if c in T], kn)
        S = EX.normalize(T, ne).to_numpy(float)
        e3_ok, e3_miss = self.e3_consistent(T)
        e3_mode = "consistent"
        if not e3_ok and variant != "E3_where_available":
            S[:, EX.FAMILIES.index("E3_structural")] = np.nan
            e3_mode = f"dropped (unavailable in zones {e3_miss})"
        if variant == "noE4":
            S[:, EX.FAMILIES.index("E4_terrain_network")] = np.nan
        D = pd.concat([T, EX.normalize(T, nd).add_prefix("q_")], axis=1)
        y, w = GA.pu_targets(T, self.cfg["static"]["positive_share"], self.cfg["static"]["pu_unlabeled_weight"])
        return {"T": T, "train": tr, "held": held, "S": S, "D": D, "y": y, "w": w, "drop_rows": drop_rows,
                "norm": {"experts": EX.norm_summary(ne), "descriptors": EX.norm_summary(nd)}, "e3_mode": e3_mode,
                "train_zones": sorted(int(z) for z in T.zone[tr].unique()), "n_train": int(tr.sum()),
                "n_pos": int((y[tr] == 1).sum()),
                "n_buffer_dropped": int(((T.zone != su.zone) & ~GA.training_rows(T, su.zone, self.zone_mask(su))).sum())}

    def gate_matrix(self, v: dict) -> tuple[np.ndarray, list[str]]:
        T = v["T"]
        cols = [c for c in GA.GATE_DESCRIPTORS if c in T]
        if any(c.startswith("y_") for c in cols):
            raise GA.LeakageError("a truth column reached the gate features")
        agree, disp = GA.expert_agreement(v["S"])
        Z = np.column_stack([T[cols].to_numpy(float), v["S"], agree, disp])
        return Z, cols + [f"s_{f}" for f in EX.FAMILIES] + ["expert_agreement", "expert_dispersion"]

    def fit_score(self, model: str, v: dict) -> tuple[np.ndarray, dict]:
        """(held-out candidate scores, fit record) of ``model`` from one view. The ONE fitting routine of B / C / D
        (stages 08-10 and the deployment simulation)."""
        tr, h, T = v["train"], v["held"], v["T"]
        info = {"train_zones": v["train_zones"], "n_train": v["n_train"], "n_positive": v["n_pos"],
                "n_buffer_dropped": v["n_buffer_dropped"], "e3_mode": v["e3_mode"]}
        if model == "A_seg":
            sc = T.mean_w2.to_numpy(float)[h] if "mean_w2" in T else np.full(int(h.sum()), np.nan)
            return sc, {**info, "status": "W2 mean on the pool (no fit)"}
        if v["n_train"] >= 10 and v["n_pos"] >= 1:
            fit = GA.fit_static(v["S"][tr], v["y"][tr], v["w"][tr], float(self.cfg["static"]["grid_step"]))
            w, st = np.asarray(fit["weights"]), "FITTED"
            info["train_ap"] = fit["ap"]
        else:
            w, st = np.full(len(EX.FAMILIES), 1.0 / len(EX.FAMILIES)), "FALLBACK_UNIFORM (too few training rows)"
        info["weights"] = dict(zip(EX.FAMILIES, w.tolist()))
        if model == "B":
            sc = GA.combine(v["S"][h], w)[0]
        elif model.startswith("C_"):
            rc = self.cfg["rules"]
            masks = GA.rule_masks(v["D"][h].reset_index(drop=True), rc)
            sc = GA.rule_gate(v["S"][h], w, masks, GA.rule_variants()[model], rc)[0]
            info["rule_firing"] = {r: float(m.mean()) for r, m in masks.items()}
        elif model == "D":
            lc = self.cfg["learned"]
            if v["n_train"] < int(lc["min_train_candidates"]) or v["n_pos"] < int(lc["min_train_positives"]):
                sc, st = GA.combine(v["S"][h], w)[0], "FALLBACK_STATIC (too few training candidates / positives)"
            else:
                Z, cols = self.gate_matrix(v)
                stz = GA.standardize_fit(Z[tr])
                Zs = GA.standardize_apply(Z, stz)
                g = GA.fit_gate(Zs[tr], v["S"][tr], v["y"][tr], v["w"][tr], w, lc)
                sc, gw = GA.gate_predict(g, Zs[h], v["S"][h])
                info.update({k: g[k] for k in ("converged", "n_iter", "objective", "message", "theta", "prior")},
                            features=cols, standardize=stz, collapse=GA.collapse(gw, v["S"][h], lc["collapse"]),
                            gate_weights=gw)
                st = "FITTED"
        else:
            raise KeyError(model)
        sc = np.where(v["drop_rows"][h], np.nan, sc)
        return sc, {**info, "status": st}

    def s07_experts(self) -> dict:
        def fn():
            X = self.expert_table()
            av = EX.availability_table(X, self.cfg["experts"], ["scenario", "zone"])
            self.ws.save_table(av, "experts/source_availability.csv")
            norm, rows, leak = {}, [], []
            for su in self.units:
                v = self.views(su)
                norm[su.key] = {**v["norm"], "train_zones": v["train_zones"], "n_train": v["n_train"],
                                "n_positive": v["n_pos"], "n_buffer_dropped": v["n_buffer_dropped"],
                                "e3_mode": v["e3_mode"], "fitted_on": "training-zone candidates of the same scenario"}
                h = v["held"]
                rows.append(pd.DataFrame(v["S"][h], columns=[f"s_{f}" for f in EX.FAMILIES]).assign(
                    scenario=su.scenario, zone=su.zone, unit=su.key, label=v["T"].label.to_numpy()[h]))
                ok = su.zone not in v["train_zones"]
                leak.append({"unit": su.key, "scenario": su.scenario, "check": "normalisation, targets and fits use "
                             "training zones only", "status": "PASS" if ok else "FAIL",
                             "detail": f"train zones {v['train_zones']}, held {su.zone}, buffer dropped "
                                       f"{v['n_buffer_dropped']}", "method": "gating.assert_disjoint (executed)"})
            self.leak("07_experts", leak)
            self.ws.save_json(norm, "experts/source_normalization.json")
            self.save_parquet(pd.concat(rows, ignore_index=True), "experts/expert_scores.parquet")
            return COMPLETE, {"note": "cross-fitted ECDF normalisation; E3 modes "
                              f"{sorted({n['e3_mode'] for n in norm.values()})}"}
        return self.run_stage("07_experts", [self.fps.get("04_candidates"), self.cfg["normalization"],
                                             self.cfg["static"], self.cfg["train_buffer_km"]], fn)

    # =========================================================================== shared model fitting / grids
    def fit_models(self, models: list[str], variants: dict | None = None) -> tuple[dict, dict]:
        """{model: scores frame}, {model|unit: fit record} for every scenario unit."""
        variants = variants or {}
        scores, reg = {m: [] for m in models}, {}
        for su in self.units:
            cache = {}
            for m in models:
                var = variants.get(m)
                if var not in cache:
                    cache[var] = self.views(su, variant=var)
                v = cache[var]
                sc, info = self.fit_score("B" if m in ABLATIONS else m, v)
                base = v["T"][v["held"]][["unit", "label", "n_px"]].reset_index(drop=True)
                scores[m].append(base.assign(score=sc, model=m))
                reg[f"{m}|{su.key}"] = {k: val for k, val in info.items() if k != "gate_weights"}
                if "gate_weights" in info:
                    reg[f"{m}|{su.key}"]["_g"] = info["gate_weights"]
        return {m: pd.concat(x, ignore_index=True) for m, x in scores.items()}, reg

    def evaluate_models(self, scores: dict, name: str) -> pd.DataFrame:
        rows = []
        for su in self.units:
            lab = None
            for m, S in scores.items():
                s = S[(S.unit == su.key)].dropna(subset=["score"])
                if not len(s):
                    continue
                lab = self.lab(su) if lab is None else lab
                rows += EV.eval_grid(self, su, EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(),
                                                            s.n_px.to_numpy()), m)
            su.unit.release()
            gc.collect()
        G = pd.DataFrame(rows)
        self.ws.save_table(G, f"gating/grid_{name}.csv")
        return G

    def save_scores(self, scores: dict) -> None:
        for m, s in scores.items():
            self.save_parquet(s, f"gating/scores_{m}.parquet")

    @staticmethod
    def json_reg(reg: dict) -> dict:
        return {k: {a: b for a, b in v.items() if a != "_g"} for k, v in reg.items()}

    # =========================================================================== [08] static fusion
    def s08_static(self) -> dict:
        def fn():
            models = ["A_seg", "B", *ABLATIONS]
            scores, reg = self.fit_models(models, ABLATIONS)
            self.save_scores(scores)
            G = self.evaluate_models(scores, "B")
            S, _ = EV.summarize(G)
            self.ws.save_table(S.assign(protocol="scenario_selected_diagnostic"), "gating/static_fusion_results.csv")
            self.ws.save_table(pd.DataFrame([{"key": k, **{f"w_{a}": b for a, b in (v.get("weights") or {}).items()},
                                              **{a: b for a, b in v.items() if a not in ("weights",)}}
                                             for k, v in reg.items() if k.startswith("B|")]),
                               "gating/static_weights.csv")
            self.ws.save_json(self.json_reg(reg), "gating/registry_B.json")
            return COMPLETE, {"note": f"models {models}"}
        return self.run_stage("08_static", [self.fps.get("07_experts"), self.cfg["static"], self.cfg["evaluation"],
                                            sorted(ABLATIONS)], fn)

    # =========================================================================== [09] rules
    def s09_rules(self) -> dict:
        def fn():
            variants = list(GA.rule_variants())
            scores, reg = self.fit_models(variants)
            self.save_scores(scores)
            G = self.evaluate_models(scores, "C")
            S, _ = EV.summarize(G)
            self.ws.save_table(S.assign(protocol="scenario_selected_diagnostic"), "gating/rule_gate_results.csv")
            self.ws.save_table(pd.DataFrame([{"key": k, **(v.get("rule_firing") or {})} for k, v in reg.items()
                                             if k.startswith("C_all|")]), "gating/rule_firing_rates.csv")
            self.ws.save_json(self.json_reg(reg), "gating/registry_C.json")
            return COMPLETE, {"note": f"{len(variants)} variants"}
        if not self.nb["experiments"]["rule_gate"]:
            return self.run_stage("09_rules", ["disabled"], lambda: (SKIPPED_OK, {"reason": "disabled in CONFIG"}))
        return self.run_stage("09_rules", [self.fps.get("08_static"), self.cfg["rules"], self.cfg["evaluation"]], fn)

    # =========================================================================== [10] learned gate
    def learned_allowed(self) -> tuple[bool, str]:
        lg = self.nb["experiments"]["learned_gate"]
        if lg is False:
            return False, "disabled in CONFIG"
        d = self.ws.read_json("candidates/coverage_decision.json") or {}
        if lg is True:
            return True, "forced in CONFIG (recorded)"
        if not d.get("learned_gate_allowed"):
            return False, f"coverage verdicts {[v['verdict'] for v in (d.get('per_scenario') or {}).values()]}: " \
                          "no scenario is ranking-limited, candidate generation first"
        return True, f"coverage verdict {d.get('overall')}"

    def s10_learned(self) -> dict:
        allowed, why = (self.learned_allowed() if self.ok("06_decision") else (False, "06 not valid"))

        def fn():
            scores, reg = self.fit_models(["D"])
            self.save_scores(scores)
            G = self.evaluate_models(scores, "D")
            S, _ = EV.summarize(G)
            self.ws.save_table(S.assign(protocol="scenario_selected_diagnostic"), "gating/learned_gate_results.csv")
            imps, wts = [], []
            for k, r in reg.items():
                unit = k.split("|", 1)[1]
                if r.get("status") != "FITTED":
                    continue
                d = len(r["features"])
                E = len(EX.FAMILIES)
                V = np.asarray(r["theta"][:E * d]).reshape(E, d)
                imps.append(pd.DataFrame({"unit": unit, "feature": r["features"],
                                          **{f"coef_{f}": V[i] for i, f in enumerate(EX.FAMILIES)}}))
                s = scores["D"][scores["D"].unit == unit]
                wts.append(pd.DataFrame(r["_g"], columns=[f"g_{f}" for f in EX.FAMILIES]).assign(
                    unit=unit, label=s.label.to_numpy()))
            I = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
            if len(I):
                I["abs_mean"] = I[[c for c in I if c.startswith("coef_")]].abs().mean(axis=1)
            W = pd.concat(wts, ignore_index=True) if wts else pd.DataFrame()
            if len(W):
                X = self.expert_table()
                X = X[[c for c in ("unit", "label", "scenario", "zone", "mean_known_dist_km", "mean_dem_observed")
                       if c in X]]
                W = W.merge(X, on=["unit", "label"], how="left").rename(columns={"mean_known_dist_km": "known_dist_km",
                                                                                 "mean_dem_observed": "dem_observed"})
            self.ws.save_table(I, "gating/feature_importance.csv")
            self.save_parquet(W, "gating/gating_weights.parquet")
            self.ws.save_json(self.json_reg(reg), "gating/registry_D.json")
            self.leak("10_learned", [{"unit": k.split("|", 1)[1], "check": "gate features contain no truth column",
                                      "status": "PASS", "method": "gate_matrix raises on y_ columns (executed)"}
                                     for k in reg])
            n_col = sum(1 for r in reg.values() if (r.get("collapse") or {}).get("collapsed"))
            return COMPLETE, {"collapsed_units": n_col, "note": f"{len(reg)} units; collapsed {n_col}"}
        if self.ok("06_decision", "08_static") and not allowed:
            return self.run_stage("10_learned", ["skipped", why], lambda: (SKIPPED_OK, {"reason": why}))
        return self.run_stage("10_learned", [self.fps.get("08_static"), self.fps.get("06_decision"),
                                             self.cfg["learned"], self.cfg["evaluation"]], fn)

    # =========================================================================== [10b] comparison + selection
    def model_smap(self, m: str, su, lab: np.ndarray) -> np.ndarray | None:
        p = self.ws.path(f"gating/scores_{m}.parquet")
        if not p.exists():
            return None
        S = pd.read_parquet(p)
        s = S[S.unit == su.key].dropna(subset=["score"])
        return EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(), s.n_px.to_numpy()) if len(s) else None

    def s10b_compare(self) -> dict:
        def fn():
            src = {"A_W2": "05_oracle", "B": "08_static", "C": "09_rules", "D": "10_learned"}
            # only grids produced (or validly reused) in THIS run: a stale grid of an earlier run never enters
            grids = [self.ws.load_table(f"gating/grid_{n}.csv") for n, s in src.items()
                     if self.status.get(s) in PRODUCED]
            G = pd.concat([g for g in grids if len(g)], ignore_index=True)
            EV.guard_oracle(G)
            G = EV.budget_rows(G, float(self.fg.cfg["submission"]["budget_tolerance"]))
            dev, rule = EV.dev_units(self.units, float(self.cfg["selection"]["policy_max_overlap"]))
            self.ws.save_table(pd.DataFrame([{"unit": k, "rule": rule[k], "dev_units": ";".join(v)}
                                             for k, v in dev.items()]), "validation/policy_development_units.csv")
            Sf, Pf = EV.frozen_rows(G, REF)
            Gx = G[G.model != REF].copy()
            Gx.loc[Gx.model == DIAG, "model"] = "A_W2_policy_reselected"
            Sx, Px, deploy = EV.fixed_policy(Gx, dev)
            if len(Px):
                Px["dev_rule"] = Px.unit.map(rule)
            S = pd.concat([Sf, Sx], ignore_index=True)
            P = pd.concat([Pf, Px], ignore_index=True)
            self.ws.save_table(S, "gating/model_summary.csv")
            self.ws.save_table(P, "validation/per_zone_metrics.csv")
            Sd, Pd = EV.summarize(G[G.model != REF].replace({"model": {DIAG: "A_W2_scenario_selected_diagnostic"}}))
            self.ws.save_table(Sd.assign(protocol="scenario_selected_diagnostic (not for selection)"),
                               "gating/model_summary_scenario_selected.csv")
            self.ws.save_json(deploy, "gating/deployable_policies.json")
            self.ws.save_table(Sf.merge(pd.DataFrame([self.frozen_policy()["cell"]]).assign(model=REF), on="model"),
                               "validation/frozen_w2_summary.csv")
            # chosen held-out maps -> blocks, bands, systems, V2 subtypes
            A = self.load_parquet("candidates/system_attributes.parquet")
            short = {(r.unit, int(r.system_id)) for r in A.itertuples() if r.length_km < float(self.cfg["short_fault_km"])}
            ev = pd.read_csv(self.ws.path("validation/v2_events.csv")) if self.ws.path("validation/v2_events.csv"
                                                                                        ).stat().st_size else None
            models = [m for m in MAIN_MODELS + tuple(ABLATIONS) if m in set(P.model)]
            blocks, bands, systems, subt = [], [], [], []
            for su in self.units:
                lab, dk = self.lab(su), self.dist_km(su)
                sid = {i for (k, i) in short if k == su.key}
                for m in models:
                    pz = P[(P.unit == su.key) & (P.model == m)]
                    if not len(pz):
                        continue
                    sm = (self.frozen_smap(su) if m == REF else self.w2_smap(su) if m == "A_W2_policy_reselected"
                          else self.model_smap(m, su, lab))
                    if sm is None:
                        continue
                    r = pz.iloc[0]
                    d = EV.chosen_details(self, su, sm, m, str(r.rep), float(r.frac), sid, dk)
                    blocks.append(d["blocks"])
                    bands.append(d["bands"])
                    systems.append(d["systems"].assign(unit=su.key))
                    if su.scenario == "V2" and ev is not None:
                        e = ev[ev.zone == su.zone] if "zone" in ev else ev
                        st = EV.subtype_components(su.unit, d["q"].astype(np.float32), su.sysmap, e)
                        subt.append(st.assign(model=m, unit=su.key, zone=su.zone))
                    self.save_npz(f"predictions/heldout/{_key(su.key)}_{m}.npz", q=np.packbits(d["q"]),
                                  shape=np.asarray(d["q"].shape), rep=str(r.rep), frac=float(r.frac))
                su.unit.release()
                gc.collect()
            Bk = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
            pairs = sorted({(m, REF) for m in models if m != REF} | {(m, "B") for m in ("C_all", "D", *ABLATIONS)
                                                                     if m in models})
            ec = self.cfg["evaluation"]
            PC = (EV.paired(Bk, pairs, int(ec["n_boot"]), int(self.cfg["v2"]["seed"])) if len(Bk) else pd.DataFrame())
            self.ws.save_table(PC, "gating/paired_comparison.csv")
            self.ws.save_table(EV.band_table(pd.concat(bands, ignore_index=True)) if bands else pd.DataFrame(),
                               "validation/near_known_metrics.csv")
            SY = pd.concat(systems, ignore_index=True) if systems else pd.DataFrame()
            self.save_parquet(SY, "validation/system_recovery.parquet")
            self.ws.save_table(self.fault_type_metrics(SY, A), "validation/per_fault_type_metrics.csv")
            self.ws.save_table(self.v2_by_subtype(subt, SY, A), "validation/v2_by_subtype.csv")
            self.ws.save_table(DE.effect_decomposition(S), "gating/effect_decomposition.csv")
            hard = self.hard_checks(G)
            reg_d = self.ws.read_json("gating/registry_D.json") or {}
            collapsed = (sum(1 for r in reg_d.values() if (r.get("collapse") or {}).get("collapsed")) > len(reg_d) / 2
                         if reg_d else False)
            sel = DE.select_model(S, PC, hard, collapsed, self.cfg["selection"])
            sel.update(hard_checks=hard, collapsed_majority=collapsed, reference=REF, frozen_policy=self.frozen_policy(),
                       protocol="fixed policy (one rule for every scenario, development units only) vs frozen "
                                "submitted W2", dev_rules=dict(pd.Series(rule).value_counts()),
                       stage_06_status=self.status.get("06_decision"),
                       stage_10_status=self.status.get("10_learned"))
            self.ws.save_json(sel, "gating/selection.json")
            self.ws.save_json({"B": self.ws.read_json("gating/registry_B.json"), "C": self.ws.read_json(
                "gating/registry_C.json"), "D": reg_d, "rules": self.cfg["rules"], "selection": sel["selected"],
                "deployable_policies": deploy, "normalization": "experts/source_normalization.json"},
                "gating/model_registry.json")
            return COMPLETE, {"selected": sel["selected"], "selection_status": sel["status"], "hard": hard,
                              "note": f"{sel['status']}: {'; '.join(sel['reasons'])[:200]}"}
        return self.run_stage("10b_compare", [self.fps.get("05_oracle"), self.fps.get("08_static"),
                                              self.fps.get("09_rules"), self.fps.get("10_learned"),
                                              self.status.get("10_learned"), self.cfg["selection"],
                                              self.cfg["evaluation"]], fn)

    def v2_by_subtype(self, subt: list, SY: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
        """V2 per subtype: events, truth length, candidate coverage, recovered share per model, gain vs the frozen
        W2, official TP / FP / FN and DTI under the subtype's own mask."""
        C = self.ws.load_table("validation/v2_coverage_by_subtype.csv")
        K = pd.concat(subt, ignore_index=True) if subt else pd.DataFrame()
        if not len(K):
            return C
        K = K.groupby(["subtype", "model"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
        K["dti"] = [EV.dti(a, b, c) for a, b, c in zip(K.tp_w, K.fp_w, K.fn_w)]
        V = SY[SY.scenario == "V2"].merge(A[["unit", "system_id", "subtype"]], on=["unit", "system_id"], how="left")
        R = V.groupby(["subtype", "model"]).recovered.mean().rename("recovered_share").reset_index()
        K = K.merge(R, on=["subtype", "model"], how="left")
        ref = K[K.model == REF].set_index("subtype")
        K["recovery_gain_vs_ref"] = K.recovered_share - K.subtype.map(ref.recovered_share)
        K["dti_gain_vs_ref"] = K.dti - K.subtype.map(ref.dti)
        return K.merge(C, on="subtype", how="left") if len(C) else K

    def hard_checks(self, G: pd.DataFrame) -> dict:
        L = self.ws.load_table("audit/label_leakage_audit.csv")
        E = self.ws.load_table("audit/metric_equivalence.csv")
        Pp = self.ws.load_table("audit/baseline_policy_parity.csv")
        return {"no_leakage_failure": bool(len(L)) and not (L.status == "FAIL").any(),
                "tests_passed_or_disabled": self.status.get("00_tests") in VALID,
                "official_metric_equivalence": bool(len(E)) and "status" in E and bool(E.status.eq("PASS").all()),
                "frozen_w2_policy_parity": bool(len(Pp)) and not (Pp.status == "FAIL").any() and
                bool((Pp.status == "PASS").any()),
                "fair_budgets": bool(G.budget_ok.all()) if "budget_ok" in G else False,
                "w2_reference_evaluated": bool((G.model == REF).any())}

    @staticmethod
    def fault_type_metrics(SY: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
        if not len(SY):
            return SY
        M = SY.merge(A[["unit", "system_id", "subtype", "length_bin", "length_km"]], on=["unit", "system_id"],
                     how="left")
        return M.groupby(["scenario", "model", "subtype", "length_bin"]).agg(
            n_systems=("system_id", "size"), recovered_share=("recovered", "mean"),
            short_systems=("short", "sum")).reset_index()

    # =========================================================================== [11] deployment, [12] export
    def s11_deployment(self) -> dict:
        from . import deployment as DP

        if not self.nb["experiments"]["deployment_comparison"]:
            return self.run_stage("11_deployment", ["disabled"], lambda: (SKIPPED_OK, {"reason": "disabled"}))
        return self.run_stage("11_deployment", [self.fps.get("10b_compare"), self.cfg["deployment"]],
                              lambda: DP.run(self))

    def s12_export(self) -> dict:
        from . import export as XP

        sel = self.ws.read_json("gating/selection.json") or {}
        ex = self.nb["export"]
        if not ex["enabled"] or self.nb["mode"] == "DIAGNOSTIC":
            return self.run_stage("12_export", ["disabled"], lambda: (SKIPPED_OK, {"reason": "export disabled / "
                                                                                            "DIAGNOSTIC mode"}))
        return self.run_stage("12_export", [self.fps.get("10b_compare"), self.fps.get("11_deployment"),
                                            self.status.get("11_deployment"), sel.get("selected"), ex,
                                            self.fg.cfg["submission"] if self.fg is not None else None],
                              lambda: XP.run(self, sel))

    # =========================================================================== [14] integrity, [13] reports
    def integrity(self) -> pd.DataFrame:
        """audit/stage_integrity.csv: per stage dependencies, status, outputs, failure, whether a scientific
        conclusion may be drawn from it."""
        def eff(s):
            st = self.status.get(s, "NOT_RUN")
            return st if st != "NOT_RUN" else (self.manifest(s).get("status") or "NOT_RUN")

        rows = []
        for s in self.STAGES:
            if s in self.REPORT_STAGES:
                continue
            st = eff(s)
            m = self.manifest(s)
            specs = self.OUTPUTS.get(s, [])
            ok, bad = self.check_outputs(specs)
            deps = self.DEPENDS.get(s, ())
            up_ok = all(eff(d) in VALID for d in deps)
            allowed = st in (*PRODUCED, SKIPPED_OK) and up_ok and not (st in PRODUCED and bad)
            rows.append({"stage": s, "label": self.LABELS.get(s), "required_dependencies": ";".join(deps),
                         "actual_status": st, "required_outputs": ";".join(r for r, _ in specs),
                         "missing_outputs": ";".join(bad), "valid_outputs": ";".join(ok),
                         "failure_reason": m.get("error") or m.get("reason") or "",
                         "affected_file": m.get("affected_file") or "",
                         "scientific_conclusion_allowed": bool(allowed),
                         "tests_status": self.status.get("00_tests")})
        return pd.DataFrame(rows)

    def s14_integrity(self) -> dict:
        def fn():
            self.ws.save_table(self.integrity(), "audit/stage_integrity.csv")
            return COMPLETE, {"note": "audit/stage_integrity.csv"}
        return self.run_stage("14_integrity", [time.time()], fn, force=True)

    def s13_reports(self) -> dict:
        from . import figures as FI
        from . import report as RP

        def fn():
            miss = FI.write_all(self)
            RP.write_all(self)
            return COMPLETE, {"note": f"reports written; figures missing: {len(miss)}"}
        return self.run_stage("13_reports", [time.time()], fn, force=True)

    def failure_summary(self) -> pd.DataFrame:
        rows = [{"stage": s, "status": v, "error": self.manifest(s).get("error") or self.manifest(s).get("reason"),
                 "affected_file": self.manifest(s).get("affected_file")} for s, v in self.status.items()
                if v in (FAILED, TESTS_FAILED, BLOCKED, BLOCKED_UPSTREAM, NOT_EVALUATED)]
        return pd.DataFrame(rows)

    # =========================================================================== RUN_ALL
    def run(self) -> dict:
        t0 = time.perf_counter()
        steps = [("00_tests", self.s00_tests), ("01_attach", self.s01_attach), ("02_audit", self.s02_audit),
                 ("03_scenarios", self.s03_scenarios), ("04_candidates", self.s04_candidates),
                 ("05_oracle", self.s05_oracle), ("06_decision", self.s06_decision), ("07_experts", self.s07_experts),
                 ("08_static", self.s08_static), ("09_rules", self.s09_rules), ("10_learned", self.s10_learned),
                 ("10b_compare", self.s10b_compare), ("11_deployment", self.s11_deployment),
                 ("12_export", self.s12_export)]
        if self.nb["mode"] == "FAST_DIAGNOSTIC":
            steps = [s for s in steps if s[0] in FAST_STAGES]
        elif self.nb["mode"] == "REPORT_ONLY":
            steps = steps[:3]
        for name, fn in steps:
            fn()
        self.s14_integrity()
        self.s13_reports()
        self.say(f"done in {round(time.perf_counter() - t0, 1)} s")
        self.final_message()
        return self.status

    def final_message(self) -> None:
        F = self.failure_summary()
        if len(F):
            print("\nNot completed:")
            for r in F.itertuples():
                print(f"  {r.stage}: {r.status} | {str(r.error)[:200]} | file: {r.affected_file}")
        d = self.ws.read_json("candidates/coverage_decision.json") or {}
        print(f"\nCoverage verdict: {d.get('overall', NOT_EVALUATED)} "
              f"{ {k: v['verdict'] for k, v in (d.get('per_scenario') or {}).items()} } | geology "
              f"{d.get('geology_status', NOT_EVALUATED)} unique={d.get('geology_unique')}")
        print("Inspect: audit/stage_integrity.csv, candidates/oracle_coverage.csv, "
              "audit/candidate_bottleneck_reconciliation.csv, validation/frozen_w2_baseline.csv, "
              "validation/v2_coverage_by_subtype.csv, reports/EXECUTION_STATUS.md")
        if self.nb["mode"] != "FAST_DIAGNOSTIC":
            sel = self.ws.read_json("gating/selection.json") or {}
            print(f"Selection: {sel.get('status', NOT_EVALUATED)} -> {sel.get('selected')}")
            sub = self.ws.read_json("submissions/submission_manifest.json") or {}
            for f in sub.get("files") or []:
                print(f"  {f.get('status')}: {f.get('path_drive')} sha256 {f.get('sha256')}")
        print(f"Drive: {self.drive / self.out}")
