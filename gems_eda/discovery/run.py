"""Colab orchestrator of the exhaustive geological discovery EDA (notebooks/colab/GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb).

    E = ExhaustiveEDA(CONFIG, HEAD, factory=F); E.run()

``ExhaustiveEDA`` is a ``gems_bench.reset.run.FaultDiscoveryReset`` (itself a ``geomoe.run.DiscoveryGeoMoE``) with
its own stage tables, so validated caching, dependency blocking (BLOCKED_UPSTREAM / NOT_EVALUATED), stage integrity,
Drive sync, the frozen B0 / B1 / W2 engine (only for the model-forensics and candidate stages), honest zone-stitched
maps and append-only reviews are reused unchanged. The data-only stages read the three competition rasters at native
100 m through ``gems_eda`` and never need the engine.

Stages: 00 tests | e01 inventory | e02 landscape (A) | e03 geometry (B) | e04 profiles (measurements) |
e05 signatures (C) | e06 scales (D) | e07 cross-modal (E) | e08 label bias (F) | e09 regional (G) |
01 attach + 03 scenarios (frozen engine) | e11 model forensics (H) | e12 candidates (I) | e13 DEM (opt-in) |
e14 integrity | e15 synthesis (dashboards, findings, hypotheses, decision, reports).
"""
from __future__ import annotations

import gc
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from gems_bench.geomoe.decide import COMPLETE, FAILED, SKIPPED_OK
from gems_bench.geomoe.run import tests_verdict
from gems_bench.reset.run import FaultDiscoveryReset

from . import EDA_CODE_REV
from . import core as K
from . import data as DA

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = "outputs/geological_discovery_eda"
STAGE_SWITCH = {"e01_inventory": "inventory", "e02_landscape": "global_geology", "e03_geometry": "fault_geometry",
                "e04_profiles": "fault_profiles", "e05_signatures": "modality_analysis", "e06_scales": "spatial_scales",
                "e07_crossmodal": "crossmodal_evidence", "e08_label_bias": "label_bias",
                "e09_regional": "regional_shift", "e11_forensics": "model_forensics",
                "e12_candidates": "unmapped_candidates", "e13_dem": "dem_pilot", "e15_synthesis": "scientific_synthesis"}


class ExhaustiveEDA(FaultDiscoveryReset):
    STAGES = ("00_tests", "e01_inventory", "e02_landscape", "e03_geometry", "e04_profiles", "e05_signatures",
              "e06_scales", "e07_crossmodal", "e08_label_bias", "e09_regional", "01_attach", "03_scenarios",
              "e11_forensics", "e12_candidates", "e13_dem", "e14_integrity", "e15_synthesis")
    DEPENDS = {"00_tests": (), "e01_inventory": ("00_tests",), "e02_landscape": ("e01_inventory",),
               "e03_geometry": ("e01_inventory",), "e04_profiles": ("e02_landscape", "e03_geometry"),
               "e05_signatures": ("e04_profiles",), "e06_scales": ("e04_profiles",),
               "e07_crossmodal": ("e04_profiles", "e06_scales"), "e08_label_bias": ("e07_crossmodal",),
               "e09_regional": ("e05_signatures", "e06_scales"), "01_attach": (), "03_scenarios": ("01_attach",),
               "e11_forensics": ("03_scenarios", "e04_profiles"), "e12_candidates": ("e07_crossmodal",),
               "e13_dem": ("e04_profiles",), "e14_integrity": (), "e15_synthesis": ()}
    DECISION_STAGES = ()
    REPORT_STAGES = ("e14_integrity", "e15_synthesis")
    STAGE_IN = ("manifests", "audit", "datasets", "tables", "figures", "atlas", "reports", "rasters")
    LABELS = {"00_tests": "Synthetic tests", "e01_inventory": "Data inventory", "e02_landscape": "Geological landscape",
              "e03_geometry": "Fault geometry", "e04_profiles": "Fault profiles + controls",
              "e05_signatures": "Fault signatures", "e06_scales": "Scale and localisation",
              "e07_crossmodal": "Cross-modal evidence", "e08_label_bias": "Label bias", "e09_regional": "Regional",
              "01_attach": "Frozen engine", "03_scenarios": "Validation units", "e11_forensics": "Model forensics",
              "e12_candidates": "Discovery candidates", "e13_dem": "DEM pilot", "e14_integrity": "Stage integrity",
              "e15_synthesis": "Scientific synthesis"}
    # Output contract: every entry has a producer in the stage body (audit: docs/GEMS_EXHAUSTIVE_EDA_HANDOFF.md,
    # "Stage contract audit"). Column tuples are the required schema; None = existence + readability only. Raster /
    # array caches (.npz) that a LATER stage reads are part of the contract, so a reused stage always provides them.
    OUTPUTS = {
        "00_tests": [("audit/selftests.csv", ("test", "status")), ("audit/pytest_output.txt", None)],
        "e01_inventory": [("audit/raw_data_inventory.csv", ("file", "exists")),
                          ("audit/band_metadata.csv", ("band", "modality", "provenance_class")),
                          ("audit/previous_eda_audit.csv", ("finding_id", "classification"))],
        "e02_landscape": [("datasets/physical_regions.parquet", ("domain",)),
                          ("rasters/spatial_domains/domains.npz", ("domain",)),
                          ("tables/band_characterization.csv", ("band", "modality", "provenance_class", "units",
                                                                "valid_frac_footprint", "p50", "robust_scale_mad",
                                                                "local_regional_std_ratio", "limitations"))],
        "e03_geometry": [("datasets/fault_segments.parquet", ("seg_id", "length_km")),
                         ("tables/fault_geometry.csv", ("seg_id", "system_id", "length_km", "strike_deg",
                                                        "sinuosity", "n_junction_ends", "system_kind", "region",
                                                        "x_mid_utm", "y_mid_utm"))],
        "e04_profiles": [("datasets/matched_controls.parquet", ("sample_id", "kind", "system")),
                         ("datasets/fault_profiles.parquet", ("sample_id", "signal", "abs_step")),
                         ("datasets/station_structure.parquet", ("sample_id", "field")),
                         ("datasets/fault_profile_cube.npz", ("cube", "offsets", "signals"))],
        "e05_signatures": [("tables/matched_control_tests.csv", ("signal", "metric", "effect"))],
        "e06_scales": [("tables/scale_localization.csv", ("modality", "scale_km")),
                       ("tables/scale_regional.csv", None),
                       ("datasets/station_scale_values.parquet", ("sample_id", "modality", "scale_km")),
                       ("rasters/evidence_maps/crests_1km.npz", ("shape",))],
        "e07_crossmodal": [("datasets/structural_lineaments.parquet", ("lineament_id", "status")),
                           ("rasters/evidence_maps/lineaments.npz", ("labels", "family_count")),
                           ("tables/crossmodal_complementarity.csv", ("figure", "status"))],
        "e08_label_bias": [("tables/label_selection_bias.csv", ("label", "effect", "ci_lo", "ci_hi"))],
        "e09_regional": [("tables/regional_robustness.csv", ("held_out_region", "signal", "metric", "effect",
                                                             "supported_same_direction"))],
        "03_scenarios": [("validation/scenarios.csv", ("scenario", "zone"))],
        "e11_forensics": [("tables/model_failure_taxonomy.csv", ("side", "error_class", "n", "length_km")),
                          ("datasets/model_error_cases.parquet", ("sample_id", "side", "error_class"))],
        "e12_candidates": [("atlas/candidate_catalog.parquet", ("candidate_id",)),
                           ("atlas/atlas_index.csv", ("candidate_id", "category", "review_split")),
                           ("atlas/reviewed_annotations.csv", None)],
        "e13_dem": [("tables/dem_pilot.csv", None)],
        "e14_integrity": [("audit/stage_integrity.csv", ("stage", "actual_status"))],
        "e15_synthesis": [("tables/evidence_synthesis.csv", None), ("reports/MODELING_DECISION.md", None)],
    }
    # bump a stage's schema version when its producer changes the content of its outputs: only that stage's cache
    # (and, through the dependency chain at run time, nothing else) is invalidated
    STAGE_SCHEMA = {"e01_inventory": "2", "e02_landscape": "2", "e03_geometry": "2", "e06_scales": "2",
                    "e07_crossmodal": "2",
                    "e08_label_bias": "2", "e09_regional": "2", "e11_forensics": "2", "e12_candidates": "2"}
    # mandatory products written BEFORE the figures (core) and after them (post); "module:function"
    CORE = {"e05_signatures": ("parts_cd:effects", None), "e06_scales": ("parts_cd:scale_data", None),
            "e07_crossmodal": ("parts_efg:lin_data", "parts_efg:crossmodal_table"),
            "e08_label_bias": ("parts_efg:label_bias_table", None), "e09_regional": ("parts_efg:lro_table", None),
            "e11_forensics": ("parts_hi:failure_taxonomy", None), "e12_candidates": ("parts_hi:write_atlas", None)}
    # session caches (E.D.cache keys) each stage's producers fill: dropped when the stage (re)computes, so a rerun
    # after a config change never reuses the previous in-memory result
    SESSION_CACHE = {"e05_signatures": ("effects",), "e06_scales": ("scale", "crests1", "persistence"),
                     "e07_crossmodal": ("LIN", "LINLAB", "FAMCOUNT", "FAMST"), "e08_label_bias": ("LABELBIAS",),
                     "e09_regional": ("LRO",), "e11_forensics": ("SEGERR", "FPERR", "OFFICIAL", "TAXONOMY"),
                     "e12_candidates": ("ATLAS", "ATLAS_WRITTEN")}
    ATLAS_STAGE = "e12_candidates"                     # read by gems_bench.reset.atlas.review_readiness

    def __init__(self, CONFIG: dict, head: str, factory=None):
        self.c = K.eda_config(CONFIG)
        rc = {k: self.c[k] for k in ("repo", "ref", "drive_root", "local_root", "reuse_cached", "sync_to_drive",
                                     "run_synthetic_tests")}
        rc.update(mode="ATLAS", overrides={"output_dir": OUT_DIR},
                  dem={"enabled": bool(self.c["dem"]["enabled"]), "cached_only": True, "pilot_only": True,
                       "process_cached_tiles": False, "allow_download": False})
        super().__init__(rc, head, factory)
        self.reg = K.Registry(self)
        self.current_stage = None
        self._D = None
        self.root = Path(self.c["drive_root"])
        self.min_effect = float(self.c["statistics"]["min_effect"])
        self.n_boot = int(self.c["n_boot"])
        self.seed = int(self.c["sampling"]["seed"])

    # =========================================================================== data access
    @property
    def D(self) -> DA.EDAData:
        if self._D is None:
            print("[eda] loading the native 100 m stack, mapped faults, template (gems_eda) ...")
            self._D = DA.load_data(self.root, (3, 3))
        return self._D

    def table(self, rel: str) -> pd.DataFrame:
        p = self.ws.path(rel)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p) if p.suffix == ".parquet" else self.ws.load_table(rel)

    def domains(self) -> np.ndarray:
        if "domains" not in self.D.cache:
            p = self.ws.path("rasters/spatial_domains/domains.npz")
            with np.load(p) as z:
                self.D.cache["domains"] = z["domain"]
        return self.D.cache["domains"]

    def stations_data(self):
        """(stations, metrics, cube, offsets, signal names, structure metrics) from the persisted e04 outputs."""
        if "S" not in self.D.cache:
            S = self.table("datasets/matched_controls.parquet")
            M = self.table("datasets/fault_profiles.parquet")
            ST = self.table("datasets/station_structure.parquet")
            with np.load(self.ws.path("datasets/fault_profile_cube.npz")) as z:
                cube, off, sigs = z["cube"].astype(np.float32), z["offsets"], [str(s) for s in z["signals"]]
            self.D.cache.update(S=S, M=M, cube=cube, off=off, sigs=sigs, ST=ST)
        d = self.D.cache
        return d["S"], d["M"], d["cube"], d["off"], d["sigs"], d["ST"]

    def run_figures(self, funcs: list) -> dict:
        ok, bad = [], []
        for fn in funcs:
            try:
                fn(self)
                ok.append(fn.__name__)
            except Exception as e:                         # one figure never kills its stage; recorded
                bad.append({"figure": fn.__name__, "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-1500:]})
                print(f"      {fn.__name__} FAILED: {type(e).__name__}: {str(e)[:200]}")
            gc.collect()
        self.ws.save_json(bad, f"manifests/figure_failures/{self.current_stage}.json")
        st = COMPLETE if len(bad) <= len(funcs) // 2 else FAILED
        return {"status": st, "figures_ok": len(ok), "figure_failures": [b["figure"] for b in bad],
                "error": f"{len(bad)} of {len(funcs)} figures failed" if st == FAILED else None,
                "note": f"{len(ok)}/{len(funcs)} figures; failures {[b['figure'] for b in bad]}"}

    def stage_parts(self, name: str, parts: list) -> list:
        """Cache-fingerprint parts: code version, mode / sampling / statistics / scales, the stage's OUTPUT CONTRACT
        and schema version (a changed contract or producer invalidates exactly that stage), plus stage parts."""
        return [EDA_CODE_REV.split("-")[0], self.c["mode"], self.c["sampling"], self.c["statistics"],
                self.c["scales_used_km"], {"contract": self.OUTPUTS.get(name, []),
                                           "schema": self.STAGE_SCHEMA.get(name, "1")}, *parts]

    def stage(self, name: str, parts: list, body) -> dict:
        """Stage wrapper: CONFIG switch, current stage name for the registry, session-cache reset, run_stage caching.
        The status is decided by ``run_stage``: COMPLETE only when ``body`` succeeded AND every contracted output
        exists and validates (``check_outputs``); figure counts never override a missing product."""
        sw = STAGE_SWITCH.get(name)
        if sw and not self.c["stages"].get(sw, True):
            return self.run_stage(name, ["disabled"], lambda: (SKIPPED_OK, {"reason": f"stages.{sw}=False"}))

        def fn():
            self.current_stage = name
            for k in self.SESSION_CACHE.get(name, ()):
                self.D.cache.pop(k, None)
            r = body()
            return r.pop("status", COMPLETE), r
        return self.run_stage(name, self.stage_parts(name, parts), fn)

    def check_outputs(self, specs) -> tuple[list[str], list[str]]:
        """Base checks (exists, readable, required columns) + zero-byte files are invalid + ``.npz`` caches must load
        and hold their required arrays. File existence alone never validates an output."""
        ok, bad = super().check_outputs([(r, c) for r, c in specs if not r.endswith(".npz")])
        for rel, keys in specs:
            p = self.ws.path(rel)
            if rel in ok and p.exists() and not p.stat().st_size:
                ok.remove(rel)
                bad.append(f"{rel}: empty file (0 bytes)")
            if not rel.endswith(".npz"):
                continue
            if not p.exists() or not p.stat().st_size:
                bad.append(f"{rel}: missing")
                continue
            try:
                with np.load(p, allow_pickle=False) as z:
                    miss = [k for k in (keys or ()) if k not in z.files]
            except Exception as e:
                bad.append(f"{rel}: unreadable ({type(e).__name__})")
                continue
            (bad.append(f"{rel}: arrays {miss} absent") if miss else ok.append(rel))
        return ok, bad

    def run_core(self, name: str, which: int) -> None:
        """Run a stage's mandatory producer (0 = before the figures, 1 = after). Exceptions propagate: a failed
        mandatory product FAILS the stage whatever the figures did."""
        import importlib

        ref = self.CORE.get(name, (None, None))[which]
        if ref:
            mod, fn = ref.split(":")
            getattr(importlib.import_module(f"gems_eda.discovery.{mod}"), fn)(self)

    def health(self, stages=None) -> pd.DataFrame:
        """Low-overhead diagnostic: per stage, status of this session, manifest status, missing / invalid contracted
        outputs, upstream stages that are not valid, root error, figure failures and a recovery hint."""
        from gems_bench.geomoe.decide import PRODUCED, VALID

        rows = []
        for s in stages or self.STAGES:
            m = self.manifest(s)
            st = self.status.get(s, "NOT_RUN")
            # a validly skipped stage (disabled switch / decision) has no contracted outputs to check
            _, probs = ([], []) if st == SKIPPED_OK else self.check_outputs(self.OUTPUTS.get(s, []))
            up = [f"{d}={self.status.get(d, 'NOT_RUN')}" for d in self.DEPENDS.get(s, ())
                  if self.status.get(d) not in VALID]
            ff = self.ws.read_json(f"manifests/figure_failures/{s}.json") or []
            if st == SKIPPED_OK or (st in PRODUCED and not probs):
                hint = ""
            elif up:
                hint = f"run / fix upstream first: {up}"
            elif probs:
                hint = f"re-run {s} (contracted outputs invalid)"
            elif st == "NOT_RUN":
                hint = f"not run in this session (last manifest: {m.get('status', 'none')})"
            else:
                hint = f"see manifests/stages/{s}.json and runtime_errors.json"
            rows.append({"stage": s, "status": st, "manifest_status": m.get("status"),
                         "missing_or_invalid": "; ".join(probs), "upstream_not_valid": "; ".join(up),
                         "error": str(m.get("error") or "")[:300], "affected_file": m.get("affected_file"),
                         "figure_failures": ";".join(f.get("figure", "") for f in ff), "recovery": hint})
        return pd.DataFrame(rows)

    # =========================================================================== stages
    def s00_tests(self) -> dict:
        from . import selftest as ST

        def fn():
            T = ST.run_selftests()
            self.ws.save_table(T, "audit/selftests.csv")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", "tests/test_geological_eda.py"],
                               cwd=str(REPO), capture_output=True, text=True)
            out = r.stdout + "\n" + r.stderr
            self.ws.save_text(f"exit code {r.returncode}\n\n{out}", "audit/pytest_output.txt")
            st, why = tests_verdict(T, r.returncode, out)
            return st, {"failing": why, "note": "; ".join(why)[:300] or "all tests passed",
                        "error": ("tests failed: " + "; ".join(why)[:600]) if why else None}
        if not self.c["run_synthetic_tests"]:
            self.ws.save_text("not run\n", "audit/pytest_output.txt")
            return self.run_stage("00_tests", ["disabled"], lambda: (SKIPPED_OK, {"reason": "disabled"}), force=True)
        return self.run_stage("00_tests", [time.strftime("%Y-%m-%dT%H:%M")], fn, force=True)

    def s_inventory(self):
        from gems_bench.reset import audit as AU

        def body():
            F, B = DA.raster_inventory(self.root)
            self.ws.save_table(F, "audit/raw_data_inventory.csv")
            self.ws.save_table(B, "audit/band_metadata.csv")
            rows = []
            for fid, claim, src, cls, why, chk in AU.PRIOR:
                if not src.startswith("outputs/eda"):
                    continue
                res, det = AU.verify(None, chk) if chk in ("header_grid", "bound_method_repr", "saturated_evidence") \
                    else ("NOT_RUN", "re-measured in e03 at native 100 m" if chk == "segment_lengths" else "")
                rows.append({"finding_id": fid, "claim": claim, "classification": cls, "reason": why,
                             "check_result": res, "check_detail": det})
            self.ws.save_table(pd.DataFrame(rows), "audit/previous_eda_audit.csv")
            dep = [{"stage": s, "requires": ";".join(self.DEPENDS.get(s, ())), "engine_needed": s in (
                "01_attach", "03_scenarios", "e11_forensics"), "outputs": ";".join(r for r, _ in self.OUTPUTS.get(
                    s, []))} for s in self.STAGES]
            self.ws.save_table(pd.DataFrame(dep), "audit/source_dependencies.csv")
            self.ws.save_json({"commit": self.head, "code_rev": EDA_CODE_REV, "config": self.c,
                               "inputs": F.to_dict("records")}, "audit/experiment_provenance.json")
            bad = [c for c in ("check_epsg", "check_res", "check_shape", "check_bands")
                   if c in F and not F[c].astype("boolean").fillna(True).all()]
            detail = {r.file: r.shape_detail for r in F.itertuples() if isinstance(getattr(r, "shape_detail", None),
                                                                               str)}
            off = list(F.file[~F["check_same_grid_as_features"].astype("boolean").fillna(True)]) \
                if "check_same_grid_as_features" in F else []
            if off:                                      # labels / template off the feature grid would be resampled
                return {"status": FAILED, "error": f"rasters not on the features grid: {off} ({detail})",
                        "affected_file": "audit/raw_data_inventory.csv", "mismatches": bad}
            return {"note": f"files {list(F.file)}; recorded-property mismatches {bad or 'none'}; grids {detail}",
                    "mismatches": bad, "shape_detail": detail}
        return self.stage("e01_inventory", [], body)

    def s_landscape(self):
        from . import parts_ab as AB

        from . import products as PR

        def body():
            D = self.D
            dom, B = DA.domains(D, 6, 50, self.seed)
            self.save_npz("rasters/spatial_domains/domains.npz", domain=dom)
            self.save_parquet(B, "datasets/physical_regions.parquet")
            D.cache["domains"] = dom
            T = PR.write_band_characterization(self, PR.band_characterization(self))   # core, before any figure
            r = self.run_figures(AB.PART_A)
            T = PR.write_band_characterization(self, PR.enrich_band_characterization(self, T))  # + F005-F008 data
            r["note"] += f"; band_characterization {len(T)} bands ({T.measured_in.iloc[0] if len(T) else ''})"
            return r
        return self.stage("e02_landscape", [], body)

    def s_geometry(self):
        from . import parts_ab as AB
        from . import products as PR

        def body():
            seg = self.D.net.segments
            self.save_parquet(seg, "datasets/fault_segments.parquet")
            T = PR.write_fault_geometry(self, PR.fault_geometry(self))                     # core, before any figure
            r = self.run_figures(AB.PART_B)
            r["note"] += f"; fault_geometry {len(T)} branches on {T.system_id.nunique()} systems"
            return r
        return self.stage("e03_geometry", [], body)

    def s_profiles(self):
        def body():
            D, c = self.D, self.c
            rng = np.random.default_rng(self.seed)
            strata = DA.context_strata(D)
            dom = self.domains()
            S = DA.stations(D, c, rng, strata, dom)
            sig, scale = DA.profile_signals(D)
            off = DA.offsets(float(c["sampling"]["profile_half_width_km"]))
            from gems_eda.profiles import extract_profiles

            ps = extract_profiles(sig, S, off)
            M = DA.station_metrics(ps.cube, S, list(sig), off)
            fields = {m: D.band(DA.PRIMARY[m]) for m in ("magnetic", "gravity", "terrain") if D.has(DA.PRIMARY[m])}
            ST = DA.structure_metrics(D, S, fields)
            alt = []
            half_px = int(np.ceil(c["sampling"]["profile_half_width_km"] / D.grid.pixel_km)) + 2
            fs = S[S.kind == "fault"].reset_index(drop=True)
            for lo, hi in c["sampling"]["control_sensitivity_km"]:
                A = DA.matched_controls(D, fs, strata, int(c["sampling"]["matched_controls_per_fault"]), lo, hi,
                                        half_px, np.random.default_rng(self.seed + int(10 * lo)),
                                        kind=f"control_{lo:g}_{hi:g}km")
                A["domain"] = dom[A.r, A.c]
                A.insert(0, "sample_id", np.arange(len(A)) + 10_000_000 * (len(alt) + 1))
                pa = extract_profiles({k: sig[k] for k in sig if not k.endswith("thdr1km")}, A, off)
                alt.append(DA.station_metrics(pa.cube, A, [k for k in sig if not k.endswith("thdr1km")], off))
            self.save_parquet(S, "datasets/matched_controls.parquet")
            self.save_parquet(S[S.kind == "fault"], "datasets/fault_systems.parquet")
            self.save_parquet(M, "datasets/fault_profiles.parquet")
            self.save_parquet(ST, "datasets/station_structure.parquet")
            self.save_parquet(pd.concat(alt, ignore_index=True) if alt else pd.DataFrame(),
                              "datasets/control_sensitivity.parquet")
            self.save_npz("datasets/fault_profile_cube.npz", cube=ps.cube.astype(np.float16), offsets=off,
                          signals=np.asarray(list(sig)))
            self.ws.save_table(scale, "tables/profile_signal_scaling.csv")
            D.cache.update(S=S, M=M, cube=ps.cube, off=off, sigs=list(sig), ST=ST)
            lv = S[S.kind == "control"].match_level.value_counts().to_dict()
            return {"note": f"{int((S.kind == 'fault').sum())} fault stations on {S[S.kind == 'fault'].system.nunique()} "
                            f"systems; controls by match level {lv}"}
        return self.stage("e04_profiles", [], body)

    def s_part(self, name: str, module: str, attr: str, extra=None):
        import importlib

        def body():
            m = importlib.import_module(f"gems_eda.discovery.{module}")
            if extra:
                extra()
            self.run_core(name, 0)                       # mandatory products first: never coupled to a figure
            r = self.run_figures(getattr(m, attr))
            self.run_core(name, 1)
            return r
        return self.stage(name, [], body)

    def s_forensics(self):
        def prep():
            from gems_bench.reset.atlas import honest_maps

            self.D.cache["maps"] = honest_maps(self)
        return self.s_part("e11_forensics", "parts_hi", "PART_H", prep)

    def s_candidates(self):
        def prep():
            p = self.ws.path("atlas/cache/honest_maps.npz")
            if p.exists() and "maps" not in self.D.cache:
                from gems_bench.reset.atlas import honest_maps

                self.D.cache["maps"] = honest_maps(self)
        return self.s_part("e12_candidates", "parts_hi", "PART_I", prep)

    def s_dem(self):
        if not self.c["dem"]["enabled"]:
            return self.stage("e13_dem", ["disabled"], lambda: {"status": SKIPPED_OK,
                                                                "reason": "dem.enabled=False (DEM coverage not "
                                                                          "evaluated; never evidence of absence)"})
        return self.s_part("e13_dem", "parts_hi", "PART_DEM")

    def s_integrity(self):
        def fn():
            self.ws.save_table(self.integrity(), "audit/stage_integrity.csv")
            self.ws.save_table(pd.DataFrame([{"stage": k, "status": v} for k, v in self.status.items()]),
                               "manifests/stage_status.csv")
            self.ws.save_json(self.fps, "manifests/cache_fingerprints.json")
            return COMPLETE, {"note": "audit/stage_integrity.csv"}
        return self.run_stage("e14_integrity", [time.time()], fn, force=True)

    def s_synthesis(self):
        from . import synthesis as SY

        def fn():
            self.current_stage = "e15_synthesis"
            return COMPLETE, SY.run(self)
        return self.run_stage("e15_synthesis", [time.time()], fn, force=True)

    # =========================================================================== run
    def run(self) -> dict:
        self.s00_tests()
        self.s_inventory()
        self.s_landscape()
        self.s_geometry()
        self.s_profiles()
        self.s_part("e05_signatures", "parts_cd", "PART_C")
        self.s_part("e06_scales", "parts_cd", "PART_D")
        self.s_part("e07_crossmodal", "parts_efg", "PART_E")
        self.s_part("e08_label_bias", "parts_efg", "PART_F")
        self.s_part("e09_regional", "parts_efg", "PART_G")
        if self.c["stages"]["model_forensics"] or self.c["stages"]["unmapped_candidates"]:
            if self.F is not None:
                self.s01_attach()
                self.s03_scenarios()
            else:
                print("[eda] no factory passed: model forensics BLOCKED (frozen engine not attached)")
        self.s_forensics()
        self.s_candidates()
        self.s_dem()
        self.s_integrity()
        self.s_synthesis()
        ran = {k: v for k, v in self.status.items() if v != "NOT_RUN"}
        print(f"\ncode rev {EDA_CODE_REV} | mode {self.c['mode']} | {ran}")
        print(f"Figures: {self.ws.path('figures/FIGURE_CATALOG.csv')} | Drive: {self.drive / self.out}")
        return self.status
