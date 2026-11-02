"""Colab orchestrator of G05.1 (notebooks/colab/GEMS_G05_1_SYSTEM_RECOVERY_AND_RANKING.ipynb).

    R = G051(CONFIG, RUN, HEAD, factory=F)
    R.run()

``G051`` IS a ``geomoe.run.DiscoveryGeoMoE`` with its own stage tables: it reuses unchanged the frozen-engine attach
(submission factory), the V1 / V2 / V3 units, the stage-04 candidate inventory, the official scorer, the canonical
system recovery, the frozen submitted-W2 reconstruction, the fixed-policy selection, validated stage caching and
dependency enforcement. Output root: ``outputs/g05_1_system_recovery/<run_id>`` (deterministic run id). A stage whose
earlier SUCCESSFUL output would be replaced by a run with a different fingerprint is first archived to
``history/<stage>/<old fingerprint>/`` (nothing is silently overwritten). Every stage also writes
``stages/<stage>/{status,manifest,config_snapshot,input_fingerprints,decision}.json``.

Stages (``DEPENDS``):
    00_tests       synthetic self-tests + tests/test_g05_1_*.py
    01_attach      frozen B0 / B1 / W2 engine (factory)
    A_inventory    preflight: dependency inventory, grid contract, exact bands, G05 run by manifest policy (fails closed)
    02_audit .. 05_oracle   GeoMoE audit, scenarios, candidate inventory, frozen-W2 baseline + parity
    B_atlas        failure atlas of the 8 G05 investigation systems + matched controls       (RUN['atlas'])
    C_recall       equal-budget candidate recall of experts / unions                           (RUN['recall'])
    D_registry     fault-hypothesis registry (D1-D7)                                           (RUN['registry'])
    E_displacement displacement-style profile correspondence + negative controls             (RUN['displacement'])
    F_graph        candidate graph features                                                    (RUN['graph'])
    G_rankers      G0-G5 + S1, cross-fitted (SCREENING_ONLY for learned models)               (RUN['training'])
    H_nesting      strict-nesting audit, STRICT_NESTING_BLOCKED + retraining plan             (RUN['strict'])
    I_selection    reps x budgets, fixed policy, geographic guardrail, promotion gates        (RUN['selection'])
    J_ablation     family ablations + TreeSHAP contributions                                   (RUN['ablations'])
    K_reconstruct  research_score.tif; validated candidate only if PROMOTE + RUN['export']    (RUN['reconstruction'])
    Y_integrity, Z_report
"""
from __future__ import annotations

import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..geomoe import candidates as CA
from ..geomoe import evaluate as EV
from ..geomoe import experts as EX
from ..geomoe.decide import COMPLETE, FAILED, PRODUCED, SKIPPED_OK, VALID
from ..geomoe.run import REF, DiscoveryGeoMoE, tests_verdict
from ..geomoe.scenarios import system_recovery
from . import G051_CODE_REV, G051_VERSION
from . import atlas as AT
from . import bands as BD
from . import config as CF
from . import crossfit as XF
from . import displacement as DP
from . import explain as XP
from . import geometry as GM
from . import graph as GR
from . import inventory as INV
from . import rankers as RK
from . import recall as RC
from . import registry as RG
from . import selection as SL

REPO = Path(__file__).resolve().parents[2]
NOT_REQUESTED = "NOT_REQUESTED"
STAGE_DIR = {"A_inventory": "preflight", "B_atlas": "atlas", "C_recall": "recall", "D_registry": "registry",
             "E_displacement": "displacement", "F_graph": "graph", "G_rankers": "rankers", "H_nesting": "nesting",
             "I_selection": "selection", "J_ablation": "ablation", "K_reconstruct": "reconstruction"}
XC_BANDS = ("mag_anom", "rtp", "tmi_hg", "tmi_vg", "iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg")


class G051(DiscoveryGeoMoE):
    STAGES = ("00_tests", "01_attach", "A_inventory", "02_audit", "03_scenarios", "04_candidates", "05_oracle",
              "B_atlas", "C_recall", "D_registry", "E_displacement", "F_graph", "G_rankers", "H_nesting",
              "I_selection", "J_ablation", "K_reconstruct", "Y_integrity", "Z_report")
    DEPENDS = {"00_tests": (), "01_attach": (), "A_inventory": (), "02_audit": ("01_attach", "A_inventory"),
               "03_scenarios": ("01_attach", "A_inventory"), "04_candidates": ("00_tests", "03_scenarios"),
               "05_oracle": ("04_candidates",), "B_atlas": ("05_oracle",), "C_recall": ("05_oracle",),
               "D_registry": ("04_candidates",), "E_displacement": ("D_registry",), "F_graph": ("D_registry",),
               "G_rankers": ("D_registry",), "H_nesting": ("G_rankers",), "I_selection": ("05_oracle", "G_rankers"),
               "J_ablation": ("I_selection",), "K_reconstruct": ("I_selection",), "Y_integrity": (),
               "Z_report": ()}
    DECISION_STAGES = ("I_selection", "K_reconstruct")
    REPORT_STAGES = ("Y_integrity", "Z_report")
    STAGE_IN = ("manifests", "audit", "validation", "candidates", "experts", "gating", "stages", "history",
                *STAGE_DIR.values(), "submissions")
    LABELS = {"00_tests": "Synthetic tests", "01_attach": "Frozen artifacts", "A_inventory": "Preflight inventory",
              "02_audit": "Validation audit", "03_scenarios": "Scenarios", "04_candidates": "Candidate inventory",
              "05_oracle": "Frozen W2 baseline", "B_atlas": "Failure atlas", "C_recall": "Candidate recall",
              "D_registry": "Hypothesis registry", "E_displacement": "Displacement evidence", "F_graph": "Graph",
              "G_rankers": "Residual rankers", "H_nesting": "Strict nesting", "I_selection": "Selection",
              "J_ablation": "Ablations", "K_reconstruct": "Reconstruction", "Y_integrity": "Stage integrity",
              "Z_report": "Report"}
    OUTPUTS = {
        "00_tests": [("audit/selftests.csv", ("test", "status")), ("audit/pytest_output.txt", None)],
        "A_inventory": [("preflight/dependency_inventory.csv", ("component", "required", "status")),
                        ("preflight/g05_run_selection.json", None)],
        "02_audit": DiscoveryGeoMoE.OUTPUTS["02_audit"],
        "03_scenarios": DiscoveryGeoMoE.OUTPUTS["03_scenarios"],
        "04_candidates": DiscoveryGeoMoE.OUTPUTS["04_candidates"],
        "05_oracle": DiscoveryGeoMoE.OUTPUTS["05_oracle"],
        "B_atlas": [("atlas/failure_atlas.csv", ("unit", "system_id", "role", "category")),
                    ("atlas/case_browser.html", None)],
        "C_recall": [("recall/recall_rows.csv", ("unit", "construction", "frac", "role")),
                     ("recall/recall_summary.csv", ("scenario", "construction", "frac", "dti")),
                     ("recall/recovery_matrix.parquet", ("unit", "system_id", "construction", "recovered"))],
        "D_registry": [("registry/fault_hypothesis_registry.parquet", ("candidate_id", "unit", "label")),
                       ("registry/feature_dictionary.csv", ("feature", "family")),
                       ("registry/feature_provenance.csv", ("feature", "source"))],
        "E_displacement": [("displacement/displacement_evidence.parquet", ("unit", "label")),
                           ("displacement/control_summary.csv", ("field",))],
        "F_graph": [("graph/graph_features.parquet", ("unit", "label", "g_support")),
                    ("graph/graph_summary.csv", ("unit",))],
        "G_rankers": [("rankers/model_registry.json", None), ("rankers/scores_G0.parquet", ("unit", "label", "score"))],
        "H_nesting": [("nesting/nesting_audit.csv", ("model", "unit", "check", "status")),
                      ("nesting/strictness.json", None), ("nesting/retraining_plan.csv", None)],
        "I_selection": [("selection/grid.csv", ("unit", "model", "rep", "frac")),
                        ("selection/model_summary.csv", ("scenario", "model", "dti")),
                        ("selection/decision.json", None)],
        "J_ablation": [("ablation/ablation_summary.csv", None), ("ablation/feature_importance.csv", None)],
        "K_reconstruct": [("reconstruction/reconstruction_manifest.json", None),
                          ("submissions/submission_manifest.json", None)],
        "Y_integrity": [("audit/stage_integrity.csv", ("stage", "actual_status"))],
        "Z_report": [("reports/G051_REPORT.md", None)],
    }

    def __init__(self, CONFIG: dict, RUN: dict, head: str, factory=None):
        self.nb51, self.run_flags = CF.notebook(CONFIG, RUN)
        self.cfg51 = CF.load_yaml(self.nb51["overrides"])
        self.run_id = self.nb51["run_id"] or CF.run_id(self.cfg51, self.nb51["geomoe_overrides"], self.nb51["ref"])
        self.code_rev = G051_CODE_REV
        out = f"{self.cfg51['output_root']}/{self.run_id}"
        gm = {k: self.nb51[k] for k in ("repo", "ref", "drive_root", "local_root", "reuse_cached", "sync_to_drive",
                                       "run_synthetic_tests", "backfill_missing_g02", "save_png")}
        gm.update(mode="FULL", strict_dependency_checks=True,
                  overrides=CF.merge(self.nb51["geomoe_overrides"], {"output_dir": out}))
        super().__init__(gm, head, factory)
        self._dom = None
        print(f"[g05.1] run id {self.run_id} -> {self.local / self.out} (Drive {self.drive / self.out})")

    # =========================================================================== plumbing overrides
    def say(self, msg: str) -> None:
        print(f"[g05.1] {msg}")

    def fp(self, *parts) -> str:
        from ..pipeline.m6 import fingerprint

        return fingerprint(G051_VERSION, *parts)

    def archive(self, stage: str, m: dict) -> None:
        """Copy an earlier SUCCESSFUL stage output to history/<stage>/<fingerprint[:10]>/ before it is replaced."""
        dst = f"history/{stage}/{str(m.get('fingerprint'))[:10]}"
        if self.ws.path(dst).exists():
            return
        copied = []
        for rel, _ in self.OUTPUTS.get(stage, []):
            p = self.ws.path(rel)
            if p.exists():
                q = self.ws.path(f"{dst}/{rel}")
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
                copied.append(rel)
        d = STAGE_DIR.get(stage)
        if d and self.ws.path(d).exists():
            shutil.copytree(self.ws.path(d), self.ws.path(f"{dst}/{d}"), dirs_exist_ok=True)
            copied.append(f"{d}/")
        for nm in ("status", "manifest", "decision"):
            p = self.ws.path(f"stages/{stage}/{nm}.json")
            if p.exists():
                q = self.ws.path(f"{dst}/stages/{stage}/{nm}.json")
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
        idx = self.ws.load_table("history/index.csv")
        row = pd.DataFrame([{"stage": stage, "fingerprint": m.get("fingerprint"), "status": m.get("status"),
                             "written": m.get("written"), "archived": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "path": dst, "files": ";".join(copied)}])
        self.ws.save_table(pd.concat([idx, row], ignore_index=True) if len(idx) else row, "history/index.csv")
        self.say(f"[{stage}] earlier successful output archived to {dst}")

    def run_stage(self, stage: str, parts: list, fn, *, force: bool = False) -> dict:
        if not force:
            m = self.manifest(stage)
            if m.get("status") in PRODUCED and m.get("fingerprint") and m["fingerprint"] != self.fp(stage, *parts):
                self.archive(stage, m)
        return super().run_stage(stage, parts, fn, force=force)

    def record(self, stage: str, st: str, fp, info: dict, seconds: float, prov: dict) -> dict:
        m = super().record(stage, st, fp, info, seconds, prov)
        base = f"stages/{stage}"
        self.ws.save_json({"stage": stage, "status": st, "written": m["written"], "run_id": self.run_id,
                           "code_rev": self.code_rev, "seconds": seconds}, f"{base}/status.json")
        self.ws.save_json({**m, "run_id": self.run_id, "g051_code_rev": self.code_rev,
                           "g051_version": G051_VERSION}, f"{base}/manifest.json")
        self.ws.save_json({"run_flags": self.run_flags, "notebook": self.nb51, "g051": self.cfg51,
                           "geomoe": self.cfg}, f"{base}/config_snapshot.json")
        self.ws.save_json({"stage_fingerprint": fp, "upstream": {d: self.fps.get(d) for d in self.DEPENDS.get(
            stage, ())}, "engine": self._ident if hasattr(self, "_ident") else None,
            "head": self.head}, f"{base}/input_fingerprints.json")
        if "decision" in info:
            self.ws.save_json(info["decision"], f"{base}/decision.json")
        return m

    def not_requested(self, stage: str, why: str) -> None:
        self.status[stage] = NOT_REQUESTED
        self.progress(stage, NOT_REQUESTED, why)

    def produced(self, stage: str) -> bool:
        return self.status.get(stage) in PRODUCED

    def units_of(self, scenario: str | None = None) -> list:
        return [su for su in self.units if scenario is None or su.scenario == scenario]

    def segments(self, su) -> pd.DataFrame:
        return self.load_parquet(self.unit_files(su)["segments"])

    def frozen_cell(self) -> tuple[str, float]:
        c = self.frozen_policy()["cell"]
        return str(c["rep"]), float(c["frac"])

    # =========================================================================== [00] tests
    def s00_tests(self) -> dict:
        from . import selftest as STT

        def fn():
            T = STT.run_selftests()
            self.ws.save_table(T, "audit/selftests.csv")
            tests = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_g05_1_*.py"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", *tests], cwd=str(REPO),
                               capture_output=True, text=True)
            out = r.stdout + "\n" + r.stderr
            self.ws.save_text(f"exit code {r.returncode}\n\n{out}", "audit/pytest_output.txt")
            st, why = tests_verdict(T, r.returncode, out)
            for w in why:
                print(f"      FAILING: {w}")
            return st, {"failing": why, "note": "; ".join(why)[:300] or f"{len(T)} self-tests + {len(tests)} files",
                        "error": ("tests failed: " + "; ".join(why)[:600]) if why else None,
                        "affected_file": "audit/pytest_output.txt" if why else None}
        if not self.nb["run_synthetic_tests"]:
            self.ws.save_text("not run: run_synthetic_tests=False\n", "audit/pytest_output.txt")
            return self.run_stage("00_tests", ["disabled"], lambda: (SKIPPED_OK, {"reason": "disabled in CONFIG"}),
                                  force=True)
        return self.run_stage("00_tests", [time.strftime("%Y-%m-%dT%H:%M")], fn, force=True)

    # =========================================================================== [A] preflight
    def g05_roots(self) -> list[Path]:
        g = self.cfg51["inputs"]["g05_root"]
        return [self.local / g, self.drive / g]

    def sA_inventory(self) -> dict:
        def fn():
            rows = INV.modules()
            rows.append(INV.row("frozen engine attached (01_attach)", True, self.ok("01_attach"),
                                self.status.get("01_attach")))
            rows.append(INV.row("configs/g05_1/recovery.yaml", True, CF.DEFAULT_YAML.exists(), CF.DEFAULT_YAML))
            if self.ok("01_attach"):
                try:
                    rows += INV.engine_rows(self, self.cfg51)
                    rows.append(INV.row("features GeoTIFF", True, Path(self.ctx.data.paths.raw("features")).exists(),
                                        "", self.ctx.data.paths.raw("features")))
                    bt = pd.DataFrame(BD.resolution_table(self.cfg51["inputs"]["raw_bands"],
                                                          self.ctx.data.band_names))
                    self.ws.save_table(bt, "preflight/band_resolution.csv")
                except Exception as e:                          # noqa: BLE001 - recorded, fails closed below
                    rows.append(INV.row("engine inventory", True, False, f"{type(e).__name__}: {e}"))
            dom = self.domains_path()
            rows.append(INV.row("EDA spatial domains (G4 region-aware ranker)", False, dom is not None, "", dom or
                                self.cfg51["inputs"]["domains_npz"]))
            g = INV.discover_g05(self.g05_roots(), self.cfg51["inputs"]["g05_run_policy"],
                                 self.nb51["g05_run_id"] or self.cfg51["inputs"]["g05_run_id"])
            sel = g.get("selected")
            rows.append(INV.row("G05 run (optional, by manifest policy)", False, sel is not None,
                                f"{g['status']}; {g['n_runs_seen']} run dir(s) seen", (sel or {}).get("path", ""),
                                warn=True))
            if sel is not None:
                self.ws.save_table(INV.g05_schema(Path(sel["path"])), "preflight/g05_schema.csv")
            self.ws.save_json(g, "preflight/g05_run_selection.json")
            self.ws.save_table(pd.DataFrame(rows), "preflight/dependency_inventory.csv")
            bad = INV.fail_closed(rows)
            if bad:
                return FAILED, {"error": "preflight failed closed: " + "; ".join(bad)[:600],
                                "affected_file": "preflight/dependency_inventory.csv"}
            return COMPLETE, {"note": f"{len(rows)} dependencies, 0 required failures; G05 {g['status']}"}
        return self.run_stage("A_inventory", [time.strftime("%Y-%m-%dT%H:%M")], fn, force=True)

    def domains_path(self) -> str | None:
        rel = self.cfg51["inputs"]["domains_npz"]
        for root in (self.local, self.drive):
            if (root / rel).exists():
                return str(root / rel)
        return None

    def domains(self) -> np.ndarray | None:
        """Full-grid integer domain raster from the EDA domains.npz (first 2-D integer array of grid shape)."""
        if self._dom is None:
            self._dom = False
            p = self.domains_path()
            if p:
                with np.load(p, allow_pickle=False) as z:
                    keys = sorted(z.files, key=lambda k: (k not in ("domain", "domains", "labels"), k))
                    for k in keys:
                        a = z[k]
                        if a.ndim == 2 and a.shape == tuple(self.ctx.data.shape) and np.issubdtype(a.dtype, np.integer):
                            self._dom = a
                            break
        return self._dom if self._dom is not False else None

    # =========================================================================== [B] atlas
    def sB_atlas(self) -> dict:
        tg = {int(k): [int(x) for x in v] for k, v in self.cfg51["target_systems"].items()}

        def fn():
            GA = self.ws.load_table("gating/grid_A_W2.csv")
            w2s = self.recovered_systems(GA, REF)
            C = self.load_parquet("candidates/coverage_parts.parquet")
            A = self.load_parquet("candidates/system_attributes.parquet")
            rep, frac = self.frozen_cell()
            tol = float(self.cfg["candidates"]["tol_px"])
            rows, panels = [], {}
            for su in self.units_of("V1"):
                if su.zone not in tg:
                    continue
                L, _ = EX.unit_layers(self, su)
                lab, T = self.lab(su), self.segments(su)
                s = self.frozen_smap(su)
                q = EV.policy_q(self, su.unit, s, rep, frac) if s is not None else np.zeros(su.unit.ev.shape)
                hs = system_recovery(su.unit, q, su.sysmap, self.sys_support, self.sys_frac)
                for sid in tg[su.zone]:
                    todo = [(sid, "target")] + [(c, f"control_for_{sid}") for c in
                                                AT.matched_controls(A, w2s, su.key, sid, 2)]
                    for k, role in todo:
                        m = AT.system_metrics(self, su, k, L, lab, T, q, C, hs, tol)
                        m.update(role=role, observation=AT.observation(m), hypothesis=AT.hypothesis(m))
                        rows.append(m)
                        if self.nb["save_png"]:
                            panels[(su.key, k)] = AT.write_panels(self, su, k, L, lab, T, q, role, 30,
                                                                  "atlas/panels")
                del L
                self.release(su.base)
                gc.collect()
            M = pd.DataFrame(rows)
            missing = [(z, s) for z, ids in tg.items() for s in ids
                       if not len(M) or not ((M.role == "target") & (M.system_id == s)).any()]
            for z, s in missing:
                M = pd.concat([M, pd.DataFrame([{"unit": f"V1:{z}", "zone": z, "system_id": s, "role": "target",
                                                 "category": "NOT_FOUND_IN_UNIT",
                                                 "observation": "no V1 unit for this zone in the attached engine",
                                                 "hypothesis": "HYPOTHESIS: none"}])], ignore_index=True)
            self.ws.save_table(M, "atlas/failure_atlas.csv")
            self.ws.save_table(M[M.role == "target"], "atlas/target_systems.csv")
            self.ws.save_text(AT.browser_html(M, panels), "atlas/case_browser.html")
            self.ws.save_json({f"{k[0]}|{k[1]}": v for k, v in panels.items()}, "atlas/panel_index.json")
            cats = M[M.role == "target"].category.value_counts().to_dict()
            lines = ["# G05.1 failure atlas", "", "Measured observations per system; HYPOTHESIS lines are readings, "
                     "not findings. Target systems come from G05's union pool (not a matched budget).", "",
                     f"Target categories: {cats}", ""]
            for r in M[M.role == "target"].itertuples():
                lines += [f"## {r.unit} system {int(r.system_id)}: {r.category}", "", f"* observation: {r.observation}",
                          f"* {r.hypothesis}", ""]
            self.ws.save_text("\n".join(lines), "atlas/ATLAS_SUMMARY.md")
            return COMPLETE, {"note": f"{int((M.role == 'target').sum())} targets, "
                                      f"{int(M.role.str.startswith('control').sum())} controls; {cats}"}
        return self.run_stage("B_atlas", [self.fps.get("05_oracle"), tg, self.code_rev], fn)

    # =========================================================================== [C] recall
    def sC_recall(self) -> dict:
        rc = self.cfg51["recall"]

        def fn():
            sh_reps = self.sh["baseline"]["representations"]
            rep = sh_reps.get(rc["rep"]) or self.cfg51["selection"]["reps"][rc["rep"]]
            fracs = [float(f) for f in rc["final_budgets"]] + [float(f) for f in rc["diagnostic_pools"]]
            roles = {**{float(f): "final" for f in rc["final_budgets"]},
                     **{float(f): "diagnostic" for f in rc["diagnostic_pools"]}}
            cell = self.frozen_cell()
            rows, us = [], {}
            for su in self.units:
                lab, T = self.lab(su), self.segments(su)
                us[su.key] = sorted(int(x) for x in np.unique(su.sysmap[su.sysmap >= 0]))
                rows += RC.unit_rows(self, su, lab, T, rc, rep, fracs, roles, self.w2_smap(su), cell,
                                     self.frozen_smap(su))
                su.unit.release()
                self.release(su.base)
                gc.collect()
            Rw = pd.DataFrame(rows)
            self.ws.save_table(Rw, "recall/recall_rows.csv")
            M = Rw[Rw.get("status", pd.Series("MEASURED", index=Rw.index)) == "MEASURED"]
            P = RC.pooled(M)
            self.ws.save_table(P, "recall/recall_summary.csv")
            self.ws.save_table(RC.new_lost(M), "recall/new_lost_vs_w2.csv")
            self.ws.save_table(RC.marginal(P), "recall/leave_one_expert_out.csv")
            Mx = RC.recovery_matrix(M, us, self.cfg51["target_systems"])
            self.save_parquet(Mx, "recall/recovery_matrix.parquet")
            self.ws.save_table(Mx[Mx.target], "recall/target_system_recovery.csv")
            O = self.load_parquet("candidates/oracle_curve_parts.parquet")
            self.ws.save_table(O.assign(note="DIAGNOSTIC ceiling (truth-ordered); never a deployable result"),
                               "recall/oracle_ceiling.csv")
            self.ws.save_json({"dti": "tp / (tp + 0.2 fp + 0.8 fn), official FastScorer components, 300 m linear kernel, "
                               "pooled by summing components (never averaging DTIs)",
                               "system_recovered": f"support >= {self.sys_support} on >= {self.sys_frac} of the "
                               "system's scored pixels (geomoe.scenarios.system_recovery)",
                               "budget": "share of the unit's eligible pixels, same for every construction",
                               "representation": rep, "union_rule": rc["union_rule"], "quota_rule": rc["quota_rule"],
                               "roles": {"final": rc["final_budgets"], "diagnostic": rc["diagnostic_pools"]}},
                              "recall/metric_definition.json")
            self.leak("C_recall", [{"unit": "*", "check": "constructions use label-free segment means only",
                                    "status": "PASS", "method": "recall.expert_scores reads mean_<layer> columns"}])
            self.recall_figure(P)
            return COMPLETE, {"note": f"{len(Rw)} rows; {Rw.construction.nunique()} constructions x {len(fracs)} budgets"}
        return self.run_stage("C_recall", [self.fps.get("05_oracle"), rc, self.cfg51["target_systems"],
                                           self.code_rev], fn)

    def recall_figure(self, P: pd.DataFrame) -> None:
        if not len(P) or not self.nb["save_png"]:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = P[(P.scenario == "V1") & (P.role == "final")]
        if not len(d):
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        piv = d.pivot_table(index="construction", columns="frac", values="n_recovered", aggfunc="sum")
        piv.plot.barh(ax=ax)
        ax.set_xlabel("V1 systems recovered (pooled over zones)")
        ax.set_title("Equal-budget recall by construction (V1)")
        fig.tight_layout()
        self.ws.save_fig(fig, "recall/recall_by_construction.png", piv.reset_index())
        plt.close(fig)

    # =========================================================================== [D] registry
    def unit_registry(self, su, bands_ok: list[str], dom) -> pd.DataFrame:
        rcfg = self.cfg51["registry"]
        u = su.unit
        lab, T = self.lab(su), self.segments(su)
        if not len(T):
            return pd.DataFrame()
        r0, r1, c0, c1 = u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop
        d = self.ctx.data
        raw = {b: np.where(u.ev, d.read_window(b, r0, r1, c0, c1), np.nan) for b in bands_ok}
        xc = {b: DP.prepare(raw[b]) for b in XC_BANDS if b in raw}
        D = RG.segment_pass(lab, T.label.to_numpy(), {}, xc, self.cfg51["displacement"], set(), rcfg)
        layers = {}
        w2 = self.w2_smap(su)
        el = self.elig(u)
        if w2 is not None:
            layers["w2_local_contrast"] = RG.local_contrast(w2, u.ev, int(round(float(rcfg["ring_km"]) / 0.1)))
        if "tmi_hg" in raw and "tmi_vg" in raw:
            layers["tilt_deg"] = np.degrees(np.arctan2(raw["tmi_vg"], np.abs(raw["tmi_hg"]))).astype(np.float32)
        Lm = CA.layer_means(lab, layers) if layers else pd.DataFrame({"label": T.label})
        Lm = Lm.rename(columns={"mean_w2_local_contrast": "w2_local_contrast", "mean_tilt_deg": "mean_tilt_deg"})
        if "mean_tilt_deg" in Lm:
            Lm["abs_tilt_deg"] = Lm.mean_tilt_deg.abs()
        Lm = Lm[[c for c in Lm.columns if not c.startswith("fin_")]]
        out = T.merge(D, on="label", how="left", suffixes=("", "_geom")).merge(Lm, on="label", how="left")
        out["w2_unit_rank"] = RC.unit_rank(out.mean_w2.to_numpy(float)) if "mean_w2" in out else np.nan
        if w2 is not None:
            pc = RG.persistence_count(lab, w2, el, rcfg["persistence_budgets"])
            out["w2_persistence"] = pc[out.label.to_numpy() - 1]
        E = RC.expert_scores(out, self.cfg51["recall"]["experts"])
        out["expert_rank_std"] = np.nanstd(E.to_numpy(float), 1) if E.shape[1] else np.nan
        out["n_experts_available"] = np.isfinite(E.to_numpy(float)).sum(1)
        for k in E.columns:
            out[f"expert_rank_{k}"] = E[k].to_numpy()
        out["domain"] = RG.domain_of(dom, out.row_full, out.col_full)
        out["candidate_id"] = out.unit.astype(str) + "#" + out.label.astype(str)
        return out

    def sD_registry(self) -> dict:
        def fn():
            bands_ok = list(BD.resolve(self.cfg51["inputs"]["raw_bands"], self.ctx.data.band_names))
            dom = self.domains()
            parts = []
            cache = f"registry/units/{self.fps['D_registry'][:10]}"
            for su in self.units:
                rel = f"{cache}/{su.key.replace(':', '_')}.parquet"
                if self.nb["reuse_cached"] and self.ws.path(rel).exists():
                    parts.append(self.load_parquet(rel))
                    continue
                t0 = time.perf_counter()
                Ru = self.unit_registry(su, bands_ok, dom)
                self.save_parquet(Ru, rel)
                parts.append(Ru)
                self.say(f"  registry {su.key}: {len(Ru)} candidates ({round(time.perf_counter() - t0, 1)} s)")
                su.unit.release()
                self.release(su.base)
                gc.collect()
                self.ws.push()
            R = pd.concat([p for p in parts if len(p)], ignore_index=True)
            RG.assert_label_free(RG.feature_columns(R))
            self.save_parquet(R, "registry/fault_hypothesis_registry.parquet")
            self.ws.save_table(RG.dictionary(), "registry/feature_dictionary.csv")
            self.ws.save_table(RG.provenance(), "registry/feature_provenance.csv")
            av = pd.DataFrame([{"feature": c, "family": RG.FAMILY[c], "present": c in R,
                                "finite_share": float(np.isfinite(pd.to_numeric(R[c], errors="coerce")).mean())
                                if c in R else 0.0} for c in RG.FAMILY])
            self.ws.save_table(av, "registry/feature_availability.csv")
            self.registry_geometry(R)
            self.leak("D_registry", [{"unit": "*", "check": "registry features contain no truth column",
                                      "status": "PASS", "method": "registry.assert_label_free (executed)"}])
            return COMPLETE, {"note": f"{len(R)} candidates x {len(RG.feature_columns(R))} features; domains "
                                      f"{'present' if dom is not None else 'ABSENT'}"}
        self.fps["D_registry"] = self.fp("D_registry", self.fps.get("04_candidates"), self.cfg51["registry"],
                                         self.cfg51["inputs"]["raw_bands"], self.cfg51["recall"]["experts"],
                                         self.domains_path() is not None, self.code_rev)
        return self.run_stage("D_registry", [self.fps.get("04_candidates"), self.cfg51["registry"],
                                             self.cfg51["inputs"]["raw_bands"], self.cfg51["recall"]["experts"],
                                             self.domains_path() is not None, self.code_rev], fn)

    def registry_geometry(self, R: pd.DataFrame) -> None:
        """registry/hypothesis_geometry.gpkg + .parquet (GeoParquet); optional, a failure is recorded."""
        try:
            import geopandas as gpd

            t = self.ctx.data.template_profile
            keep = ["candidate_id", "scenario", "zone", "unit", "label", "w2_unit_rank", "chain_length_km",
                    "strike_deg", "n_proposers"]
            parts = []
            for su in self.units:
                g = R[R.unit == su.key]
                if len(g):
                    gd = CA.to_geodataframe(self.lab(su), g[[c for c in keep if c in g]], su.unit.sl, t["transform"],
                                            t["crs"])
                    parts.append(gd)
            if parts:
                G = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=t["crs"])
                p = self.ws.path("registry/hypothesis_geometry.gpkg")
                p.parent.mkdir(parents=True, exist_ok=True)
                G.to_file(p, driver="GPKG")
                G.to_parquet(self.ws.path("registry/hypothesis_geometry.parquet"))
        except Exception as e:                                  # noqa: BLE001 - optional format
            self.errors.append({"stage": "D_registry", "error": f"geometry export skipped: {type(e).__name__}: {e}"})

    def registry(self) -> pd.DataFrame:
        if getattr(self, "_R_fp", None) != (self.fps.get("D_registry"), self.fps.get("E_displacement"),
                                            self.fps.get("F_graph")):
            R = self.load_parquet("registry/fault_hypothesis_registry.parquet")
            if self.produced("E_displacement"):
                E = self.load_parquet("displacement/displacement_evidence.parquet")
                R = R.merge(E, on=["unit", "label"], how="left", suffixes=("", "_e"))
            if self.produced("F_graph"):
                F = self.load_parquet("graph/graph_features.parquet")
                R = R.drop(columns=[c for c in F.columns if c in R and c not in ("unit", "label")]).merge(
                    F, on=["unit", "label"], how="left")
            self._R = R
            self._R_fp = (self.fps.get("D_registry"), self.fps.get("E_displacement"), self.fps.get("F_graph"))
        return self._R

    # =========================================================================== [E] displacement
    def sE_displacement(self) -> dict:
        dc = self.cfg51["displacement"]

        def fn():
            fields = list(BD.resolve(dc["enabled_fields"], self.ctx.data.band_names))
            R = self.load_parquet("registry/fault_hypothesis_registry.parquet")
            out = []
            for su in self.units:
                g = R[R.unit == su.key]
                if not len(g):
                    continue
                u = su.unit
                top = set(g.sort_values("w2_unit_rank", ascending=False).label.head(int(dc["max_segments_per_unit"]))
                          .astype(int))
                r0, r1, c0, c1 = u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop
                fs = {f: DP.smoothed(np.where(u.ev, self.ctx.data.read_window(f, r0, r1, c0, c1), np.nan),
                                     dc["smoothing_sigma_px"]) for f in fields}
                P = GM.segment_pixels(self.lab(su))
                for k in sorted(top):
                    rr, cc = P.get(k, (np.zeros(0, int), np.zeros(0, int)))
                    row = {"unit": su.key, "label": int(k)}
                    order = GM.path_order(rr, cc) if len(rr) else np.zeros(0, int)
                    for f in fields:
                        e = DP.segment_evidence(fs[f], rr, cc, dc, order)
                        row.update({f"disp_{f}_{a}": e.get(a, np.nan) for a in ("status", "shift_px", "gain",
                                                                                 "control_gain", "excess")})
                        row[f"disp_{f}_consistent"] = float(e["consistent"]) if "consistent" in e else np.nan
                    out.append(row)
                self.say(f"  displacement {su.key}: {len(top)} segments")
                su.unit.release()
                gc.collect()
            E = pd.DataFrame(out)
            self.save_parquet(E, "displacement/displacement_evidence.parquet")
            rows = []
            for f in fields:
                st = E.get(f"disp_{f}_status", pd.Series(dtype=str))
                m = st == "MEASURED"
                g, cg = E.get(f"disp_{f}_gain"), E.get(f"disp_{f}_control_gain")
                rows.append({"field": f, "n_segments": int(len(E)), "n_measured": int(m.sum()),
                             "n_unresolvable": int(st.astype(str).str.startswith("UNRESOLVABLE").sum()),
                             "share_consistent": float(E[f"disp_{f}_consistent"][m].mean()) if m.any() else np.nan,
                             "mean_gain": float(g[m].mean()) if m.any() else np.nan,
                             "mean_control_gain": float(cg[m].mean()) if m.any() else np.nan,
                             "share_gain_above_control": float((g[m] > cg[m]).mean()) if m.any() else np.nan,
                             "reading": "evidence only where gain exceeds the same-side control; a share near 0.5 "
                                        "means the shift search finds structure anywhere (no displacement signal)"})
            self.ws.save_table(pd.DataFrame(rows), "displacement/control_summary.csv")
            self.ws.save_text("offset_proxy (G05) is a CURVATURE measure; it is kept as `curvature_deg_per_km` in the "
                              "registry and never presented as displacement.\n", "displacement/OFFSET_PROXY_NOTE.md")
            return COMPLETE, {"note": f"{len(E)} segments x {fields}"}
        return self.run_stage("E_displacement", [self.fps.get("D_registry"), dc, self.code_rev], fn)

    # =========================================================================== [F] graph
    def sF_graph(self) -> dict:
        gc_ = self.cfg51["graph"]

        def fn():
            R = self.load_parquet("registry/fault_hypothesis_registry.parquet")
            feats, eds, summ = [], [], []
            for su in self.units:
                g = R[R.unit == su.key].reset_index(drop=True)
                if not len(g):
                    continue
                P = GM.segment_pixels(self.lab(su))
                Ed = GR.edges(g, P, gc_)
                conf = dict(zip(g.label.astype(int), g.w2_unit_rank.astype(float)))
                Fu = GR.node_features(g, Ed, conf, gc_, GR.known_tips(su.known)).assign(unit=su.key)
                feats.append(Fu)
                eds.append(Ed.assign(unit=su.key))
                summ.append({"unit": su.key, "n_nodes": len(g), "n_edges": len(Ed),
                             "n_continuity": int((Ed.kind == "continuity").sum()) if len(Ed) else 0,
                             "n_branch": int((Ed.kind == "branch").sum()) if len(Ed) else 0,
                             "isolated_share": float(Fu.g_isolated.mean()),
                             "max_component_size": int(Fu.g_component_size.max())})
            F = pd.concat(feats, ignore_index=True)
            self.save_parquet(F, "graph/graph_features.parquet")
            self.save_parquet(pd.concat(eds, ignore_index=True) if eds else pd.DataFrame(), "graph/edges.parquet")
            self.ws.save_table(pd.DataFrame(summ), "graph/graph_summary.csv")
            return COMPLETE, {"note": f"{len(F)} nodes; edges {sum(s['n_edges'] for s in summ)}"}
        return self.run_stage("F_graph", [self.fps.get("D_registry"), gc_, self.code_rev], fn)

    # =========================================================================== [G] rankers
    def ranker_models(self, R: pd.DataFrame) -> tuple[list[str], dict]:
        nog = RG.feature_columns(R, include_graph=False)
        gr = RG.feature_columns(R) if self.produced("F_graph") else nog
        models = ["G0", "S1"] + [m for m in self.cfg51["rankers"]["models"] if m != "G0"]
        if self.produced("F_graph"):
            models.append("G5")
        return models, {"nograph": [c for c in nog if c != "domain"], "graph": [c for c in gr if c != "domain"]}

    def sG_rankers(self) -> dict:
        rk, lc = self.cfg51["rankers"], self.cfg51["labels"]

        def fn():
            R = self.registry()
            models, feats = self.ranker_models(R)
            scores, reg, leak, labs = {m: [] for m in models}, {}, [], []
            for su in self.units:
                out, rec, lk = XF.crossfit_unit(R, su.key, su.scenario, su.zone, self.zone_mask(su), models, rk, lc,
                                                feats)
                for m, s in out.items():
                    scores[m].append(s)
                for m, r in rec.items():
                    reg[f"{m}|{su.key}"] = r
                leak += [{**x, "scenario": su.scenario} for x in lk]
                T = R[R.scenario == su.scenario]
                tr = XF.training_mask(T.reset_index(drop=True), su.zone, self.zone_mask(su))
                L = RK.make_labels(T.reset_index(drop=True)[tr], lc)
                labs.append(L.cls.value_counts().rename_axis("cls").reset_index(name="n").assign(unit=su.key))
                gc.collect()
            for m, parts in scores.items():
                if parts:
                    self.save_parquet(pd.concat(parts, ignore_index=True), f"rankers/scores_{m}.parquet")
            self.ws.save_json(reg, "rankers/model_registry.json")
            self.ws.save_table(pd.concat(labs, ignore_index=True), "rankers/label_summary.csv")
            self.ws.save_json({"nograph": feats["nograph"], "graph": feats["graph"]}, "rankers/feature_lists.json")
            self.leak("G_rankers", leak)
            fitted = sorted({k.split("|")[0] for k, v in reg.items() if v.get("status") == "FITTED"})
            return COMPLETE, {"note": f"models {models}; fitted {fitted}; learned results are SCREENING_ONLY"}
        return self.run_stage("G_rankers", [self.fps.get("D_registry"), self.fps.get("E_displacement"),
                                            self.fps.get("F_graph"), rk, lc, self.cfg["train_buffer_km"],
                                            self.code_rev], fn)

    # =========================================================================== [H] nesting
    def sH_nesting(self) -> dict:
        def fn():
            reg = self.ws.read_json("rankers/model_registry.json") or {}
            models = sorted({k.split("|")[0] for k in reg})
            us = [(su.key, su.scenario, su.zone) for su in self.units]
            A = XF.nesting_audit(us, models)
            self.ws.save_table(A, "nesting/nesting_audit.csv")
            self.ws.save_table(XF.retraining_plan(sorted({su.zone for su in self.units_of("V1")}),
                                                  sorted({su.zone for su in self.units_of("V3")})),
                               "nesting/retraining_plan.csv")
            st = {m: XF.STRICTNESS.get(m, XF.SCREENING) for m in models}
            self.ws.save_json(st, "nesting/strictness.json")
            d = {"strictness": st, "strict_nesting_learned": XF.BLOCKED,
                 "reason": "learned rankers see training-row W2 / CNN values and known context built with the held-out "
                           "zone's truth; strict nesting needs per-pair refits (nesting/retraining_plan.csv)"}
            return COMPLETE, {"decision": d, "note": f"learned -> {XF.SCREENING}; S1 / G0 -> {XF.STRICT_FREE}; "
                                                     f"strict learned nesting {XF.BLOCKED}"}
        return self.run_stage("H_nesting", [self.fps.get("G_rankers"), self.code_rev], fn)

    # =========================================================================== [I] selection
    def verification(self) -> dict:
        L = self.ws.load_table("audit/label_leakage_audit.csv")
        Pp = self.ws.load_table("audit/baseline_policy_parity.csv")
        parity = bool(len(Pp)) and not (Pp.status == "FAIL").any() and bool((Pp.status == "PASS").any())
        leak_ok = bool(len(L)) and not (L.status == "FAIL").any()
        return {"provenance_verified": parity and self.ok("03_scenarios"),
                "upstream_honesty_verified": leak_ok and self.produced("H_nesting")}

    def sI_selection(self) -> dict:
        sc = self.cfg51["selection"]

        def fn():
            reg = self.ws.read_json("rankers/model_registry.json") or {}
            models = [m for m in dict.fromkeys(k.split("|")[0] for k in reg)
                      if self.ws.path(f"rankers/scores_{m}.parquet").exists()]
            S_ = {m: self.load_parquet(f"rankers/scores_{m}.parquet") for m in models}
            strict = self.ws.read_json("nesting/strictness.json") if self.produced("H_nesting") else {}
            cell = self.frozen_cell()
            rows, exact = [], []
            for su in self.units:
                lab, w2 = self.lab(su), self.w2_smap(su)
                for m in models:
                    s = S_[m][S_[m].unit == su.key].dropna(subset=["score"])
                    if not len(s):
                        continue
                    sm = EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(), s.n_px.to_numpy())
                    sm = SL.partial_map(sm, w2) if sc["partial_segments"] else sm
                    st = (strict or {}).get(m, "UNVERIFIED (Stage H not run)")
                    rows += SL.grid_rows(self, su, sm, m, sc["reps"], sc["budgets"], {"strictness": st})
                    exact.append({**SL.exact_policy_row(self, su, sm, m, cell), "strictness": st})
                su.unit.release()
                self.release(su.base)
                gc.collect()
            Gg = EV.budget_rows(pd.DataFrame(rows), float(sc["budget_tolerance"]))
            EV.guard_oracle(Gg)
            self.ws.save_table(Gg, "selection/grid.csv")
            X = pd.DataFrame(exact)
            self.ws.save_table(X, "selection/exact_frozen_cell.csv")
            dev, rule = EV.dev_units(self.units, float(self.cfg["selection"]["policy_max_overlap"]))
            S, P, deploy = EV.fixed_policy(Gg, dev)
            if not len(S) or "model" not in S:
                raise RuntimeError("fixed-policy selection produced no summary row (no development unit?)")
            P = SL.attach_systems(P, Gg)
            GA = self.ws.load_table("gating/grid_A_W2.csv")
            Sf, Pf = EV.frozen_rows(GA, REF)
            self.ws.save_table(pd.concat([Sf, S], ignore_index=True), "selection/model_summary.csv")
            self.ws.save_table(pd.concat([Pf, P], ignore_index=True), "selection/per_unit.csv")
            self.ws.save_json(deploy, "selection/deployable_policies.json")
            geo = {m: g for m, g in X[X.scenario == "V3"].groupby("model")}
            geo_m = {m: {"dti": SL.dti(g.tp_w.sum(), g.fp_w.sum(), g.fn_w.sum())} for m, g in geo.items()}
            fv3 = Sf[Sf.scenario == "V3"]
            geo_ref = {"dti": float(fv3.dti.iloc[0])} if len(fv3) else None
            checks = self.verification()
            decs = []
            for m in models:
                if m == "G0":
                    continue
                decs.append(SL.promotion(m, (strict or {}).get(m, "UNVERIFIED"), S[S.model == m], P[P.model == m],
                                         Sf, Pf, geo_m.get(m), geo_ref, self.cfg51["promotion"], checks))
            g0 = [SL.promotion("G0", "STRICT_LABEL_FREE", S[S.model == "G0"], P[P.model == "G0"], Sf, Pf,
                               geo_m.get("G0"), geo_ref, self.cfg51["promotion"], checks)]
            D = SL.decision_table(decs + g0)
            self.ws.save_table(D, "selection/promotion_decisions.csv")
            self.ws.save_table(self.vs_g0(S), "selection/challenger_vs_g0.csv")
            prom = sorted([d for d in decs if d["status"] == SL.PROMOTE], key=lambda d: -d["disc_gain"])
            prs = sorted([d for d in decs if d["status"] == SL.PROMISING], key=lambda d: -d["disc_gain"])
            measured = [d for d in decs if d.get("disc_dti") is not None]
            research = max(measured, key=lambda d: d["disc_dti"])["model"] if measured else None
            if prom:
                status, sel = SL.PROMOTE, prom[0]["model"]
            elif prs:
                status, sel = SL.PROMISING, None
            else:
                status, sel = (SL.RETAIN if measured else SL.NOT_EVAL), None
            dec = {"status": status, "selected": sel, "research_model": research,
                   "promising_screening_only": [d["model"] for d in prs], "checks": checks,
                   "reference": f"{REF} frozen cell {cell}", "deployable_policies": deploy,
                   "dev_rules": dict(pd.Series(rule).value_counts()),
                   "protocol": "fixed policy (development units only) vs frozen submitted W2; learned models are "
                               "SCREENING_ONLY and can never be PROMOTE_CANDIDATE",
                   "decisions": decs}
            self.ws.save_json(dec, "selection/decision.json")
            return COMPLETE, {"decision": {k: v for k, v in dec.items() if k != "decisions"},
                              "note": f"{status}: selected {sel}; promising (screening) {dec['promising_screening_only']}"}
        return self.run_stage("I_selection", [self.fps.get("G_rankers"), self.fps.get("H_nesting"),
                                              self.status.get("H_nesting"), self.fps.get("05_oracle"), sc,
                                              self.cfg51["promotion"], self.code_rev], fn)

    @staticmethod
    def vs_g0(S: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sc_, g in S.groupby("scenario"):
            b = g[g.model == "G0"]
            if not len(b):
                continue
            for r in g[g.model != "G0"].itertuples():
                rows.append({"scenario": sc_, "model": r.model, "dti": r.dti, "dti_g0": float(b.dti.iloc[0]),
                             "delta_vs_g0": r.dti - float(b.dti.iloc[0]), "worst_zone_dti": r.worst_zone_dti})
        return pd.DataFrame(rows)

    # =========================================================================== [J] ablation
    def sJ_ablation(self) -> dict:
        rk, lc = self.cfg51["rankers"], self.cfg51["labels"]

        def fn():
            R = self.registry()
            _, feats = self.ranker_models(R)
            base = "G5" if self.produced("F_graph") else "G1"
            P = self.ws.load_table("selection/per_unit.csv")
            Pb = P[P.model == base]
            if not len(Pb):
                self.ws.save_table(pd.DataFrame([{"status": f"{base} has no fixed-policy rows"}]),
                                   "ablation/ablation_summary.csv")
                self.ws.save_table(pd.DataFrame(), "ablation/feature_importance.csv")
                return COMPLETE, {"note": f"{base} not evaluated: nothing to ablate"}
            reps = self.cfg51["selection"]["reps"]
            rows, contrib, tsys = [], [], []
            tg = {int(k): [int(x) for x in v] for k, v in self.cfg51["target_systems"].items()}
            for su in self.units:
                pr = Pb[Pb.unit == su.key]
                if not len(pr):
                    continue
                rep, frac = str(pr.rep.iloc[0]), float(pr.frac.iloc[0])
                lab, w2 = self.lab(su), self.w2_smap(su)
                variants = {base: feats}
                for fam in self.cfg51["ablation_families"]:
                    drop = set(RG.ABLATION_MAP.get(fam, RG.FAMILY_LAYERS.get(fam, ())))
                    variants[f"{base}-minus-{fam}"] = {k: [c for c in v if c not in drop] for k, v in feats.items()}
                for name, fv in variants.items():
                    out, rec, _ = XF.crossfit_unit(R, su.key, su.scenario, su.zone, self.zone_mask(su), [base], rk,
                                                   lc, fv)
                    if base not in out:
                        continue
                    s = out[base]
                    sm = SL.partial_map(EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(), s.n_px.to_numpy()),
                                        w2)
                    r = SL.grid_rows(self, su, sm, name, {rep: reps[rep]}, [frac])[0]
                    rows.append({**r, "zone": su.zone})
                # TreeSHAP of the full base model on the held-out rows
                T = R[R.scenario == su.scenario].reset_index(drop=True)
                tr = XF.training_mask(T, su.zone, self.zone_mask(su))
                Ttr, Th = T[tr].reset_index(drop=True), T[T.unit == su.key].reset_index(drop=True)
                L = RK.make_labels(Ttr, lc)
                spec = RK.model_spec(base, R, feats["nograph"], feats["graph"])
                if spec is not None and (L.cls == "positive").sum() >= 5:
                    mdl, _ = RK._fit_predict(spec, Ttr, L, Th, rk, int(rk["seed"]))
                    Cn = XP.contributions(mdl, Th[spec["features"]].astype(float))
                    contrib.append(XP.importance(Cn).assign(unit=su.key))
                    if su.scenario == "V1" and su.zone in tg:
                        from scipy import ndimage

                        for sid in tg[su.zone]:
                            near = ndimage.binary_dilation(su.sysmap == sid, iterations=3)
                            ls = set(np.unique(lab[near])) - {0}
                            ix = Th.label.isin(ls).to_numpy()
                            if ix.any():
                                tsys.append(Cn[ix].mean().rename("mean_contrib").reset_index().rename(
                                    columns={"index": "feature"}).assign(unit=su.key, system_id=sid,
                                                                         n_segments=int(ix.sum())))
                su.unit.release()
                self.release(su.base)
                gc.collect()
            A = pd.DataFrame(rows)
            S = []
            for (sc_, m), g in A.groupby(["scenario", "model"]):
                S.append({"scenario": sc_, "model": m, "dti": SL.dti(g.tp_w.sum(), g.fp_w.sum(), g.fn_w.sum()),
                          "n_recovered": int(g.n_recovered.sum())})
            S = pd.DataFrame(S)
            self.ws.save_table(A, "ablation/ablation_rows.csv")
            self.ws.save_table(XP.ablation_deltas(S, base) if len(S) else S, "ablation/ablation_summary.csv")
            I = pd.concat(contrib, ignore_index=True) if contrib else pd.DataFrame()
            if len(I):
                agg = I.groupby(["feature", "family"]).mean_abs_contrib.mean().reset_index().sort_values(
                    "mean_abs_contrib", ascending=False)
                self.ws.save_table(agg, "ablation/feature_importance.csv")
                self.ws.save_table(XP.family_importance(agg), "ablation/family_importance.csv")
            else:
                self.ws.save_table(pd.DataFrame(), "ablation/feature_importance.csv")
            self.ws.save_table(pd.concat(tsys, ignore_index=True) if tsys else pd.DataFrame(),
                               "ablation/target_system_contributions.csv")
            return COMPLETE, {"note": f"{base}: {len(self.cfg51['ablation_families'])} family ablations at the "
                                      "fixed-policy cell (descriptive, single run)"}
        return self.run_stage("J_ablation", [self.fps.get("I_selection"), self.cfg51["ablation_families"],
                                             self.code_rev], fn)

    # =========================================================================== [K] reconstruction / export
    def sK_reconstruct(self) -> dict:
        from . import export as KX

        def fn():
            dec = self.ws.read_json("selection/decision.json") or {}
            info = {"research": None, "validated": None}
            rm = dec.get("research_model") or "G0"
            info["research"] = KX.research_score(self, rm)
            files = []
            if self.run_flags["export"] and dec.get("status") == SL.PROMOTE and dec.get("selected"):
                d = next(x for x in dec["decisions"] if x["model"] == dec["selected"])
                cell = dec["deployable_policies"][dec["selected"]]
                st, row = KX.validated_candidate(self, dec["selected"], d, (cell["rep"], cell["frac"]))
                info["validated"] = {"status": st, **row}
                files.append(row)
                dec["export_parity_verified"] = bool(row.get("export_parity_verified"))
                dec["final_status"] = "VALIDATED_FOR_MANUAL_UPLOAD" if row.get("status") == KX.READY else \
                    "PROMOTED_BUT_EXPORT_FAILED"
            else:
                why = ("RUN['export'] is False" if not self.run_flags["export"] else
                       f"Stage I status {dec.get('status')} (no PROMOTE_CANDIDATE)")
                files.append({"file": None, "status": KX.NONE, "reason": why})
                dec["final_status"] = KX.NONE
            self.ws.save_json({"files": files, "uploaded": False, "upload": "manual only",
                               "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "submissions/submission_manifest.json")
            self.ws.save_json({**info, "research_model": rm, "role": "RESEARCH_NOT_A_SUBMISSION"},
                              "reconstruction/reconstruction_manifest.json")
            self.ws.save_json(dec, "selection/decision.json")
            return COMPLETE, {"decision": {"final_status": dec["final_status"], "research_model": rm},
                              "note": f"research_score_{rm}.tif; submission: {dec['final_status']}"}
        return self.run_stage("K_reconstruct", [self.fps.get("I_selection"), self.run_flags["export"],
                                                self.code_rev], fn)

    # =========================================================================== integrity / report
    def sY_integrity(self) -> dict:
        def fn():
            self.ws.save_table(self.integrity(), "audit/stage_integrity.csv")
            return COMPLETE, {"note": "audit/stage_integrity.csv"}
        return self.run_stage("Y_integrity", [time.time()], fn, force=True)

    def sZ_report(self) -> dict:
        from . import report as RP

        def fn():
            RP.write(self)
            return COMPLETE, {"note": "reports/G051_REPORT.md"}
        return self.run_stage("Z_report", [time.time()], fn, force=True)

    # =========================================================================== RUN
    def run(self) -> dict:
        t0 = time.perf_counter()
        for fn in (self.s00_tests, self.s01_attach, self.sA_inventory, self.s02_audit, self.s03_scenarios,
                   self.s04_candidates, self.s05_oracle):
            fn()
        opt = [("B_atlas", "atlas", self.sB_atlas), ("C_recall", "recall", self.sC_recall),
               ("D_registry", "registry", self.sD_registry), ("E_displacement", "displacement", self.sE_displacement),
               ("F_graph", "graph", self.sF_graph), ("G_rankers", "training", self.sG_rankers),
               ("H_nesting", "strict", self.sH_nesting), ("I_selection", "selection", self.sI_selection),
               ("J_ablation", "ablations", self.sJ_ablation), ("K_reconstruct", "reconstruction",
                                                               self.sK_reconstruct)]
        for stage, flag, fn in opt:
            if self.run_flags.get(flag):
                fn()
            else:
                self.not_requested(stage, f"RUN['{flag}'] is False")
        self.sY_integrity()
        self.sZ_report()
        self.say(f"done in {round(time.perf_counter() - t0, 1)} s")
        self.final_message()
        return self.status

    def final_message(self) -> None:
        F = self.failure_summary()
        if len(F):
            print("\nNot completed:")
            for r in F.itertuples():
                print(f"  {r.stage}: {r.status} | {str(r.error)[:200]} | file: {r.affected_file}")
        dec = self.ws.read_json("selection/decision.json") or {}
        print(f"\nSelection: {dec.get('status', 'NOT_EVALUATED')} -> {dec.get('selected')} | final "
              f"{dec.get('final_status', '-')}")
        print(f"Stage statuses: { {k: v for k, v in self.status.items() if v not in VALID} or 'all valid' }")
        print(f"Run dir: {self.local / self.out}\nDrive:   {self.drive / self.out}\nReport:  reports/G051_REPORT.md")
