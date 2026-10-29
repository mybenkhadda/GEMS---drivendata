"""Colab orchestrator of the fault-discovery reset (notebooks/colab/GEMS_FAULT_DISCOVERY_RESET.ipynb).

    F = Factory(...); R = FaultDiscoveryReset(CONFIG, HEAD, factory=F); R.run()

``FaultDiscoveryReset`` IS a ``geomoe.run.DiscoveryGeoMoE`` with its own stage tables, so it reuses, unchanged: the
frozen-engine attach (submission factory), the V1 / V2 / V3 scenario units, the official scorer and eligible domain,
the frozen submitted-W2 reconstruction (``frozen_policy`` / ``frozen_smap``), the fixed-policy selection, the paired
block bootstrap, validated stage caching (fingerprint + required outputs), dependency enforcement
(BLOCKED_UPSTREAM / NOT_EVALUATED), stage integrity and Drive sync. Outputs: ``outputs/fault_discovery_reset``.

Stages: [00] tests, [01] attach, [02] audit, [03] scenarios, [04] atlas, [05] physics, [06] benchmark, [07] terrain
(opt-in), [08] challenger decision (+ the selected path), [09] export (factory writer), [11] integrity, [12] reports.
Modes: ATLAS (00-05), BENCHMARK (00-06), DEM_PILOT (00-03, 07), CHALLENGER (00-08), FULL / RESUME (all; DEM and
downloads only when explicitly enabled; no neural network is ever trained here).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..geomoe.decide import COMPLETE, SKIPPED_OK
from ..geomoe.run import DiscoveryGeoMoE, tests_verdict
from . import RESET_CODE_REV

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "configs" / "reset" / "fault_discovery_reset.yaml"
MODES = {"ATLAS": ("00_tests", "01_attach", "02_reset_audit", "03_scenarios", "04_atlas", "05_physics"),
         "BENCHMARK": ("00_tests", "01_attach", "02_reset_audit", "03_scenarios", "04_atlas", "05_physics",
                       "06_benchmark"),
         "DEM_PILOT": ("00_tests", "01_attach", "02_reset_audit", "03_scenarios", "07_terrain"),
         "CHALLENGER": ("00_tests", "01_attach", "02_reset_audit", "03_scenarios", "04_atlas", "05_physics",
                        "06_benchmark", "07_terrain", "08_challenger")}
MODES["FULL"] = MODES["CHALLENGER"] + ("09_export",)
MODES["RESUME"] = MODES["FULL"]
NB_DEFAULTS = {"repo": "mybenkhadda/GEMS---drivendata", "ref": "main", "drive_root": "/content/drive/MyDrive/GEMS_2",
               "local_root": "/content/gems_work", "mode": "ATLAS", "reuse_cached": True, "native_resolution_m": 100,
               "atlas": {}, "dem": {"enabled": False, "cached_only": True, "pilot_only": True,
                                    "process_cached_tiles": False, "allow_download": False},
               "benchmark": {}, "challenger": {"enabled": False, "train_large_neural_model": False},
               "export": {"enabled": True, "manual_upload_only": True, "exploratory": False},
               "run_synthetic_tests": True, "save_all_artifacts": True, "sync_to_drive": True, "overrides": {}}


class ResetConfigError(ValueError):
    pass


def reset_config(CONFIG: dict | None) -> dict:
    C = CONFIG or {}
    c = {**NB_DEFAULTS, **C}
    for k in ("dem", "challenger", "export"):
        c[k] = {**NB_DEFAULTS[k], **(C.get(k) or {})}
    c["mode"] = str(c["mode"]).upper()
    errs = []
    if c["mode"] not in MODES:
        errs.append(f"mode {c['mode']!r} not in {list(MODES)}")
    if c["challenger"].get("train_large_neural_model"):
        errs.append("challenger.train_large_neural_model=True is refused in this notebook (no neural training)")
    if not c["export"].get("manual_upload_only", True):
        errs.append("export.manual_upload_only must stay True")
    if int(c["native_resolution_m"]) != 100:
        errs.append("native_resolution_m must be 100 (the official grid)")
    if c["dem"].get("allow_download") and not c["dem"].get("enabled"):
        errs.append("dem.allow_download requires dem.enabled")
    if c["dem"].get("allow_download") and not c["dem"].get("pilot_only", True):
        errs.append("dem.allow_download is limited to the pilot (pilot_only=True)")
    if errs:
        raise ResetConfigError("; ".join(errs))
    if c["mode"] == "RESUME":
        c["reuse_cached"] = True
    return c


def load_reset_config(CONFIG: dict) -> dict:
    import yaml

    from ..p6.run import _merge

    with open(DEFAULT, encoding="utf-8") as f:
        rc = _merge(yaml.safe_load(f), CONFIG.get("overrides") or {})
    for k in ("atlas", "benchmark"):                      # notebook CONFIG sections override the YAML ones
        rc[k] = _merge(rc[k], {a: b for a, b in (CONFIG.get(k) or {}).items() if a in rc[k] or k == "atlas"})
    return rc


class FaultDiscoveryReset(DiscoveryGeoMoE):
    STAGES = ("00_tests", "01_attach", "02_reset_audit", "03_scenarios", "04_atlas", "05_physics", "06_benchmark",
              "07_terrain", "08_challenger", "09_export", "11_integrity", "12_reports")
    DEPENDS = {"00_tests": (), "01_attach": (), "02_reset_audit": ("01_attach",), "03_scenarios": ("01_attach",),
               "04_atlas": ("00_tests", "03_scenarios"), "05_physics": ("04_atlas",),
               "06_benchmark": ("00_tests", "03_scenarios", "04_atlas"), "07_terrain": ("00_tests", "03_scenarios"),
               "08_challenger": ("06_benchmark",), "09_export": ("08_challenger",), "11_integrity": (),
               "12_reports": ()}
    DECISION_STAGES = ("08_challenger", "09_export")
    REPORT_STAGES = ("11_integrity", "12_reports")
    STAGE_IN = ("manifests", "audit", "atlas", "physics", "benchmark", "terrain", "challenger", "submissions",
                "validation")
    LABELS = {"00_tests": "Synthetic tests", "01_attach": "Frozen artifacts", "02_reset_audit": "Scientific audit",
              "03_scenarios": "Validation scenarios", "04_atlas": "Evidence atlas", "05_physics": "Physical evidence",
              "06_benchmark": "Benchmark", "07_terrain": "DEM pilot", "08_challenger": "Challenger decision",
              "09_export": "Submission", "11_integrity": "Stage integrity", "12_reports": "Reports"}
    OUTPUTS = {
        "00_tests": [("audit/selftests.csv", ("test", "status")), ("audit/pytest_output.txt", None)],
        "02_reset_audit": [("audit/source_inventory.csv", ("kind", "present")),
                           ("audit/prior_results_audit.csv", ("finding_id", "classification")),
                           ("audit/validation_contract.csv", None)],
        "03_scenarios": [("validation/scenarios.csv", ("scenario", "zone")), ("validation/v2_events.csv", None)],
        "04_atlas": [("atlas/atlas_index.csv", ("candidate_id", "category", "x_utm", "y_utm")),
                     ("atlas/candidate_catalog.parquet", ("candidate_id",)),
                     ("atlas/annotation_template.csv", ("candidate_id", "interpretation")),
                     ("atlas/atlas_overview.html", None)],
        "05_physics": [("physics/evidence_by_modality.csv", ("candidate_id", "modality", "status")),
                       ("physics/structural_evidence.parquet", ("candidate_id",)),
                       ("physics/localization_error.csv", ("source",))],
        "06_benchmark": [("benchmark/baseline_metrics.csv", ("scenario", "model", "dti")),
                         ("benchmark/representation_grid.csv", ("model", "rep", "frac")),
                         ("benchmark/selected_policy.json", None)],
        "07_terrain": [("terrain/pilot_decision.json", None)],
        "08_challenger": [("challenger/selected_challenger.json", None)],
        "09_export": [("submissions/submission_manifest.json", None)],
        "11_integrity": [("audit/stage_integrity.csv", ("stage", "actual_status"))],
        "12_reports": [("reports/SCIENTIFIC_AUDIT.md", None)],
    }

    def __init__(self, CONFIG: dict, head: str, factory=None):
        self.rnb = reset_config(CONFIG)
        self.rc = load_reset_config(self.rnb)
        gm = {k: self.rnb[k] for k in ("repo", "ref", "drive_root", "local_root", "reuse_cached", "sync_to_drive",
                                       "run_synthetic_tests")}
        gm.update(mode="FULL", strict_dependency_checks=True, save_png=True,
                  overrides={"output_dir": self.rc["output_dir"]})
        super().__init__(gm, head, factory)
        self.gm = self                                   # the GeoMoE helpers are methods of this object
        self._geom = None
        self._ref = None

    # =========================================================================== reviews / geometry helpers
    def candidate_pixels(self, cid: str):
        if self._geom is None:
            p = self.ws.path("atlas/cache/geometry.npz")
            if not p.exists():
                return None
            with np.load(p, allow_pickle=False) as z:
                ids, off, rr, cc = list(z["ids"]), z["offsets"], z["rr"], z["cc"]
            self._geom = {i: (rr[off[k]:off[k + 1]], cc[off[k]:off[k + 1]]) for k, i in enumerate(ids)}
        return self._geom.get(cid)

    def load_reviews(self) -> pd.DataFrame:
        from .atlas import REVIEW_FIELDS

        T = self.ws.load_table("atlas/reviewed_annotations.csv")
        return T if len(T) else pd.DataFrame(columns=list(REVIEW_FIELDS))

    def record_review(self, review: dict | list) -> dict:
        """Append review event(s) (validated; earlier rows are never edited). Returns {added, errors}."""
        from .atlas import append_reviews

        C = self.ws.load_table("atlas/atlas_index.csv")
        old = self.load_reviews()
        new, errs = append_reviews(old, review if isinstance(review, list) else [review], C,
                                   tuple(self.rc["challenger"]["hard_negative"]))
        if len(new) != len(old):
            self.ws.save_table(new, "atlas/reviewed_annotations.csv")
            self.ws.push()
        return {"added": int(len(new) - len(old)), "errors": errs}

    def import_reviews(self, path) -> dict:
        """Ingest a filled annotation template (rows with interpretation != NOT_REVIEWED)."""
        T = pd.read_csv(path)
        rows = [r for r in T.to_dict("records") if str(r.get("interpretation", "NOT_REVIEWED")) != "NOT_REVIEWED"]
        return self.record_review(rows) if rows else {"added": 0, "errors": ["no reviewed row in the template"]}

    def reference_crop(self, u):
        ref = self.rc["benchmark"].get("reference_prediction")
        if not ref:
            return None
        if self._ref is None:
            import rasterio

            p = self.drive / ref
            if not p.exists():
                return None
            with rasterio.open(p) as s:
                a = s.read(1).astype(np.float32)
            if a.shape != self.ctx.data.shape:
                raise ValueError(f"reference prediction {p} shape {a.shape} != grid {self.ctx.data.shape}")
            self._ref = np.nan_to_num(a, nan=0.0)
        return self._ref[u.sl]

    # =========================================================================== [00] tests
    def s00_tests(self) -> dict:
        from . import selftest as RST

        def fn():
            T = RST.run_selftests()
            self.ws.save_table(T, "audit/selftests.csv")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", "tests/test_fault_discovery_reset.py"],
                               cwd=str(REPO), capture_output=True, text=True)
            out = r.stdout + "\n" + r.stderr
            self.ws.save_text(f"exit code {r.returncode}\n\n{out}", "audit/pytest_output.txt")
            st, why = tests_verdict(T, r.returncode, out)
            for w in why:
                print(f"      FAILING: {w}")
            return st, {"failing": why, "note": "; ".join(why)[:300] or "all tests passed",
                        "error": ("tests failed: " + "; ".join(why)[:600]) if why else None,
                        "affected_file": "audit/pytest_output.txt" if why else None}
        if not self.rnb["run_synthetic_tests"]:
            self.ws.save_text("not run: run_synthetic_tests=False\n", "audit/pytest_output.txt")
            return self.run_stage("00_tests", ["disabled"], lambda: (SKIPPED_OK, {"reason": "disabled in CONFIG"}),
                                  force=True)
        return self.run_stage("00_tests", [time.strftime("%Y-%m-%dT%H:%M")], fn, force=True)

    # =========================================================================== [02] audit
    def s02_reset_audit(self) -> dict:
        from . import audit as AU

        def fn():
            if self.units is None:
                from ..geomoe import scenarios as SC

                self.units = SC.build(self)
            self.ws.save_table(AU.source_inventory(self), "audit/source_inventory.csv")
            self.ws.save_table(AU.prior_findings(self), "audit/prior_results_audit.csv")
            self.ws.save_table(AU.validation_contract(), "audit/validation_contract.csv")
            self.ws.save_text(AU.LIMITATIONS, "audit/known_limitations.md")
            return COMPLETE, {"note": "audit tables written"}
        return self.run_stage("02_reset_audit", [self.head, self.rc["bands"]], fn, force=True)

    # =========================================================================== [04] atlas
    def s04_atlas(self) -> dict:
        from . import atlas as AT
        from . import physics as PH

        parts = [self.fps.get("03_scenarios"), self.rc["atlas"], self.rc["physics"]["sigmas_px"],
                 self.frozen_policy()["cell"] if self.F is not None and self.fg is not None else None]

        def fn():
            L = PH.full_layers(self)
            M = AT.honest_maps(self)
            C, meta = AT.build_catalog(self, L, M)
            tag = self.fps["04_atlas"][:6]
            C["candidate_id"] = tag + "-" + C.candidate_id
            geom = {tag + "-" + k: v for k, v in meta["geometry"].items()}
            ids = list(geom)
            off = np.cumsum([0] + [len(geom[i][0]) for i in ids])
            self.save_npz("atlas/cache/geometry.npz", ids=np.asarray(ids), offsets=off,
                          rr=np.concatenate([geom[i][0] for i in ids]).astype(np.int32) if ids else np.zeros(0),
                          cc=np.concatenate([geom[i][1] for i in ids]).astype(np.int32) if ids else np.zeros(0))
            self._geom = None
            P = AT.write_panels(self, C, L, M, geom)
            self.ws.save_table(C.drop(columns=["geometry_wkt"]), "atlas/atlas_index.csv")
            self.save_parquet(C, "atlas/candidate_catalog.parquet")
            self.save_parquet(C, f"atlas/catalog_history/{tag}.parquet")     # earlier catalogs stay reloadable
            self.gpkg(C)
            tpl = C[[c for c in AT.CATALOG_FIELDS if c in C]].copy()
            for k in ("reviewer", "review_mode", "evidence_sources", "data_license"):
                tpl[k] = ""
            self.ws.save_table(tpl, "atlas/annotation_template.csv")
            if not self.ws.path("atlas/reviewed_annotations.csv").exists():  # never overwritten
                self.ws.save_table(pd.DataFrame(columns=list(AT.REVIEW_FIELDS)), "atlas/reviewed_annotations.csv")
            self.ws.save_text(AT.overview_html(C, P, self.load_reviews()), "atlas/atlas_overview.html")
            self.case_studies(C)
            short = {k: v for k, v in meta["category_counts"].items() if v < AT.quotas(
                self.rc["atlas"]["categories"], int(self.rc["atlas"]["n_examples"])).get(k, 0)}
            return COMPLETE, {"category_counts": meta["category_counts"], "pool_sizes": meta["pool_sizes"],
                              "categories_below_quota": short, "note": f"{len(C)} candidates; below quota {short}"}
        return self.run_stage("04_atlas", parts, fn)

    def gpkg(self, C: pd.DataFrame) -> None:
        try:
            import geopandas as gpd
            from shapely import wkt

            g = gpd.GeoDataFrame(C.drop(columns=["geometry_wkt"]), geometry=[wkt.loads(x) for x in C.geometry_wkt],
                                 crs=self.ctx.data.template_profile["crs"])
            p = self.ws.path("atlas/candidate_catalog.gpkg")
            p.parent.mkdir(parents=True, exist_ok=True)
            g.to_file(p, driver="GPKG")
        except Exception as e:                          # optional format, recorded
            self.errors.append({"stage": "04_atlas", "error": f"gpkg skipped: {type(e).__name__}: {e}"})

    def case_studies(self, C: pd.DataFrame, F: pd.DataFrame | None = None) -> None:
        lines = ["# Atlas case studies", "", "One section per category; evidence observations are filled by stage 05. "
                 "Interpretations come only from reviews (reviewed_annotations.csv).", ""]
        for k, g in C.groupby("category"):
            lines += [f"## {k}", ""]
            for r in g.head(3).itertuples():
                fl = F[F.candidate_id == r.candidate_id].iloc[0].to_dict() if F is not None and len(F) and \
                    (F.candidate_id == r.candidate_id).any() else {}
                obs = ", ".join(x for x in ("GEOLOGICAL_CONTRAST", "GEOLOGICAL_DISRUPTION", "STRUCTURAL_CONTINUITY",
                                            "MULTI_SUPPORT") if fl.get(x)) or ("pending stage 05" if F is None else
                                                                                "none observed")
                lines += [f"* `{r.candidate_id}` UTM ({r.x_utm:.0f}, {r.y_utm:.0f}), {r.distance_to_known_fault_km:.2f} "
                          f"km to a mapped fault; observations: {obs}"
                          + (f"; alternative: {fl.get('alternative_explanation')}" if fl.get(
                              "alternative_explanation") else ""),
                          f"  panels: `panels/{r.candidate_id}_evidence_only.png`, "
                          f"`panels/{r.candidate_id}_diagnostic.png`"]
            lines.append("")
        self.ws.save_text("\n".join(lines), "atlas/case_studies.md")

    # =========================================================================== [05] physics
    def s05_physics(self) -> dict:
        from . import physics as PH

        def fn():
            C = self.load_parquet("atlas/candidate_catalog.parquet")
            E, Fl, Pr = PH.candidate_evidence(self, C)
            self.ws.save_table(E, "physics/evidence_by_modality.csv")
            self.save_parquet(Fl, "physics/structural_evidence.parquet")
            self.ws.save_table(Pr, "physics/scale_profiles.csv")
            num = [c for c in E.columns if c not in ("candidate_id", "modality", "probe", "status") and
                   pd.api.types.is_numeric_dtype(E[c])]
            W = E.pivot_table(index="candidate_id", columns="modality", values=num, aggfunc="first")
            W.columns = [f"{m}_{c}" for c, m in W.columns]
            self.save_parquet(W.reset_index(), "physics/magnetic_gravity_features.parquet")
            self.ws.save_table(PH.scale_localization(self), "physics/localization_error.csv")
            self.case_studies(C, Fl)
            return COMPLETE, {"note": f"{len(C)} candidates x {E.modality.nunique()} modalities; status counts "
                                      f"{E.status.value_counts().to_dict()}"}
        return self.run_stage("05_physics", [self.fps.get("04_atlas"), self.rc["physics"]], fn)

    # =========================================================================== [06] benchmark, [07] terrain
    def s06_benchmark(self) -> dict:
        from . import benchmark as BM
        from . import physics as PH

        def fn():
            res = BM.run(self, PH.full_layers(self))
            loc = self.ws.path("physics/localization_error.csv")
            if loc.exists():
                self.ws.save_table(pd.read_csv(loc), "benchmark/localization_error.csv")
            return COMPLETE, res
        return self.run_stage("06_benchmark", [self.fps.get("04_atlas"), self.fps.get("03_scenarios"),
                                               self.rc["benchmark"], self.cfg["selection"],
                                               self.rc["challenger"]["selection"], len(self.load_reviews())], fn)

    def s07_terrain(self) -> dict:
        from . import terrain as TE

        def fn():
            r = TE.run(self)
            return r.pop("status"), r
        return self.run_stage("07_terrain", [self.fps.get("03_scenarios"), self.rnb["dem"], self.rc["terrain"]], fn)

    # =========================================================================== [08] challenger
    def s08_challenger(self) -> dict:
        from . import challenger as CH

        def fn():
            r = CH.run(self)
            return COMPLETE, r
        return self.run_stage("08_challenger", [self.fps.get("06_benchmark"), self.status.get("07_terrain"),
                                                self.fps.get("07_terrain"), len(self.load_reviews()),
                                                self.rc["challenger"]], fn)

    # =========================================================================== [09] export
    def s09_export(self) -> dict:
        def fn():
            return self.export()
        if not self.rnb["export"].get("enabled"):
            return self.run_stage("09_export", ["disabled"], lambda: (SKIPPED_OK, {"reason": "export disabled"}))
        return self.run_stage("09_export", [self.fps.get("08_challenger"), self.rnb["export"],
                                            self.rc["benchmark"].get("exportable_representations")], fn)

    def export(self) -> tuple[str, dict]:
        """Zone-stitched frozen W2 score with the selected representation, through the factory / P6 path
        (``final_submission.build_map / recipe / write_file / validate`` + ``factory.readback_checks``)."""
        from ..artifacts.atomic import sha256_file
        from ..gems3 import submission as SUB
        from ..geomoe.export import chk, write_index
        from ..leaderboard_focus import factory as FA
        from ..leaderboard_focus import submission as SU
        from ..leaderboard_focus import validation as VA
        from ..p6 import export_parity as EP
        from ..p6 import final_submission as FS

        ch = self.ws.read_json("challenger/selected_challenger.json") or {}
        sp = self.ws.read_json("benchmark/selected_policy.json") or {}
        dep = sp.get("deployable_cell") or {}
        exportable = set(self.rc["benchmark"].get("exportable_representations") or [])
        role = None
        if ch.get("selected_case") == "CASE_1_ALLOCATION" and sp.get("credible"):
            role = "SELECTED"
        elif self.rnb["export"].get("exploratory") and dep:
            role = "EXPLORATORY"
        if role is None:
            note = (f"No new submission: challenger {ch.get('selected_case')}, representation decision "
                    f"{sp.get('status')}. W2 stays the reference (factory w2_champion file).")
            write_index(self, [{"file": None, "status": "NOT_SELECTED", "role": "none", "model": None}], note)
            return SKIPPED_OK, {"reason": note}
        rep, frac = str(dep.get("rep")), float(dep.get("frac"))
        if rep not in exportable:
            note = (f"{rep} is not exportable: the factory validator enforces realised area <= frac (soft / dilated "
                    "representations need a validator-contract change; a user decision)")
            write_index(self, [{"file": None, "status": "NOT_EXPORTABLE", "role": role, "model": f"W2|{rep}"}], note)
            return SKIPPED_OK, {"reason": note}
        gates = [chk("gate", "self-tests passed", self.status.get("00_tests") == COMPLETE),
                 chk("gate", "factory runtime self-tests", self.F.selftest_ok())]
        L = self.ws.load_table("audit/label_leakage_audit.csv")
        gates.append(chk("gate", "no leakage failure", bool(len(L)) and not (L.status == "FAIL").any()))
        bad = [g for g in gates if not g["ok"]]
        if bad:
            write_index(self, [{"file": None, "status": "BLOCKED", "role": role, "model": f"W2|{rep}",
                                "reasons": "; ".join(g["check"] for g in bad)}], "export blocked by a hard gate")
            return "BLOCKED", {"error": f"export gates {[g['check'] for g in bad]}"}
        from .benchmark import install_representations

        install_representations(self)                    # a cached benchmark stage did not install them
        fg, data = self.fg, self.ctx.data
        fz = self.frozen_policy()
        cell = {**fz["cell"], "rep": rep, "frac": frac}
        brow = {"name": f"reset_w2_{rep}", "deploy": fz["kind"], "deploy_cell": cell, "space": ""}
        built = FS.build_map(fg, brow, fusion_w="all_zones")
        rec = FS.recipe(fg, brow, built, "fixed_policy (fault-discovery reset)")
        pa = EP.plan_audit(rec)
        pc = fg.cfg["submission"]
        Z = built["zones"]
        tol = float(pc["budget_tolerance"])
        tech = [chk("export", "recipe plan audit", pa["status"] == "PASS", json.dumps(pa["checks"])),
                chk("budget", "per-zone area <= frac x (1 + tol), non-empty", bool(((Z.frac_of_eligible <= frac * (
                    1 + tol) + 1e-9) & (Z.positive_px > 0)).all())),
                chk("mask", "no positive pixel on a mapped fault", int(Z.positive_on_known_px.sum()) == 0),
                chk("parity", "no BLOCKED zone parity", not (built["parity"].status == "BLOCKED").any())]
        arr = SUB.finalise(built["q"], data, pc["outside_valid"])
        vsha = VA.array_sha(arr)
        rid = rec["fingerprint"][:10]
        name = f"{'exploratory_' if role == 'EXPLORATORY' else ''}submission_reset_w2_{rep}_{rid}"
        row = {"model": f"W2|{rep}@{frac:g}", "role": role, "recipe_id": rid, "rep": rep, "frac": frac,
               "validated_array_sha256": vsha}
        if any(not c["ok"] for c in tech):
            row.update(status="INVALID", reasons="; ".join(c["check"] for c in tech if not c["ok"]))
            write_index(self, [row], "technical validation failed before writing")
            return "BLOCKED", {"error": row["reasons"]}
        ns = SimpleNamespace(p6=fg.p6, cfg=fg.cfg, root=self.ws.root)
        ledger = [SUB.area_ledger(built["q"], data, 0, f"reset W2 {rep} @ {frac:g}")]
        w = FS.write_file(ns, name, built["q"], rec, ledger)
        path = w["path"]
        v = FS.validate(ns, path, rec, built)
        checks = tech + [chk(f"official:{g}", c["check"], c["ok"], c.get("detail", ""), c.get("hard", True))
                         for g, cs in v["groups"].items() for c in cs]
        facts = SU.raster_facts(path)
        side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        fsha = sha256_file(path)
        mc = self.F.cfg["submission"]
        checks += FA.readback_checks(facts, data.template_profile, mc["expected"], path.stat().st_size / 2 ** 20,
                                     float(mc["max_size_mb"]), vsha, VA.array_sha(facts["array"]), side.get("recipe_fp"),
                                     rec["fingerprint"], fsha)
        status = "READY_FOR_UPLOAD" if FA.verdict(checks) in ("PASS", "WARNING") else "INVALID"
        self.ws.save_json({"checks": checks, "recipe": rec}, f"submissions/{path.stem}_validation.json")
        row.update(file=path.name, status=status, sha256=fsha, path_drive=str(self.drive / self.out / "submissions" /
                                                                              path.name))
        write_index(self, [row], f"{role}: frozen W2 score, representation {rep} @ {frac:g} (zone-stitched)")
        return (COMPLETE if status == "READY_FOR_UPLOAD" else "FAILED"), {"file": path.name, "status": status}

    # =========================================================================== [11] integrity, [12] reports
    def s11_integrity(self) -> dict:
        def fn():
            self.ws.save_table(self.integrity(), "audit/stage_integrity.csv")
            return COMPLETE, {"note": "audit/stage_integrity.csv"}
        return self.run_stage("11_integrity", [time.time()], fn, force=True)

    def s12_reports(self) -> dict:
        from . import report as RP

        def fn():
            miss = RP.figures(self)
            r = RP.reports(self)
            return COMPLETE, {"note": f"figures missing {len(miss)}; next: {r['next_action'][:160]}"}
        return self.run_stage("12_reports", [time.time()], fn, force=True)

    # =========================================================================== run
    def run(self) -> dict:
        fns = {"00_tests": self.s00_tests, "01_attach": self.s01_attach, "02_reset_audit": self.s02_reset_audit,
               "03_scenarios": self.s03_scenarios, "04_atlas": self.s04_atlas, "05_physics": self.s05_physics,
               "06_benchmark": self.s06_benchmark, "07_terrain": self.s07_terrain,
               "08_challenger": self.s08_challenger, "09_export": self.s09_export}
        for s in MODES[self.rnb["mode"]]:
            fns[s]()
        self.s11_integrity()
        self.s12_reports()
        ran = {k: v for k, v in self.status.items() if v != "NOT_RUN"}
        print(f"\ncode rev {RESET_CODE_REV} | mode {self.rnb['mode']} | stages {ran}")
        p = self.ws.path("reports/NEXT_ACTION.md")
        print("Next:", p.read_text(encoding="utf-8")[:400] if p.exists() else "reports not written")
        print(f"Drive: {self.drive / self.out}")
        return self.status

    # GeoMoE stages that do not exist in this experiment are not callable by mistake
    def s04_candidates(self):                            # noqa: D102
        raise RuntimeError("not part of the fault-discovery reset")

    s05_oracle = s06_decision = s07_experts = s08_static = s09_rules = s10_learned = s10b_compare = s04_candidates
