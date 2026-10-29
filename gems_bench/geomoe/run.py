"""Colab orchestrator of the discovery-first GeoMoE experiment (notebooks/colab/GEMS_DISCOVERY_GEOMOE.ipynb).

    F = Factory(FACTORY_CONFIG, HEAD)                  # the existing submission factory = the frozen engine
    G = DiscoveryGeoMoE(CONFIG, HEAD, factory=F)
    G.run()                                            # [01]-[13]; or call G.s01_attach() ... G.s13_reports()

Stages (each persisted under ``outputs/discovery_geomoe`` and pushed to Drive when it finishes; a stage whose
fingerprint and outputs are unchanged is reloaded, never recomputed; a failed optional stage never stops the
independent ones):
    [01] attach      factory self-tests + preflight + attach (B0 / B1 / W2 engine on the CPU, read-only)
    [02] audit       component inventory (reuse matrix), source compatibility, recorded-protocol table
    [03] scenarios   V1 / V2 / V3 units, V2 withheld events, leakage checks
    [04] candidates  per unit: label-free source layers -> pools -> deduplicated inventory -> segment table,
                     coverage parts, oracle curve
    [05] oracle      W2 baseline grid (A_pixel) + metric equivalence vs the frozen boost-A grid, oracle coverage,
                     unique / missed systems, M7 vs X2 protocol reconciliation
    [06] decision    generation vs ranking bottleneck (gate for [10])
    [07] experts     expert scores, availability, cross-fitted normalisation
    [08] static      Model A_seg (W2 on the same pool) and Model B (static fusion)
    [09] rules       Model C (rule gate) + one ablation per rule
    [10] learned     Model D (learned gate) when [06] allows it; then the paired comparison and model selection
    [11] deployment  zone-stitched vs fold-average vs footprint-rank construction (V3, cross-fitted)
    [12] export      validated GeoTIFF of the selected model (or an EXPLORATORY one when asked), factory writer
    [13] reports     figures (+ data) and the four reports from persisted tables only
"""
from __future__ import annotations

import gc
import json
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

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "geomoe" / "discovery_geomoe.yaml"
COMPLETE, SKIPPED, FAILED, BLOCKED, CACHED = "COMPLETE", "SKIPPED", "FAILED", "BLOCKED", "SKIP_CACHED"
STAGES = ("01_attach", "02_audit", "03_scenarios", "04_candidates", "05_oracle", "06_decision", "07_experts",
          "08_static", "09_rules", "10_learned", "10b_compare", "11_deployment", "12_export", "13_reports")
NB_DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "main", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "local_root": "/content/gems_work", "mode": "FULL", "reuse_cached": True, "device": "cpu",
    "experts": {"cnn": True, "w2": True, "magnetic_gravity": True, "g01_g02": True, "dem": "cached_only",
                "network": True},
    "experiments": {"validation_audit": True, "candidate_ceiling": True, "static_fusion": True, "rule_gate": True,
                    "learned_gate": "conditional", "deployment_comparison": True},
    "export": {"enabled": True, "only_validated": True, "exploratory": False, "manual_upload": True},
    "save_png": True, "sync_to_drive": True, "overrides": {},
}
MODES = ("FULL", "DIAGNOSTIC", "RESUME", "REPORT_ONLY")
EXPERT_SWITCH = {"cnn": ("cnn_b0", "cnn_b1", "fusion"), "w2": ("w2",), "g01_g02": ("g02c", "g02a", "g01_persistence"),
                 "magnetic_gravity": ("mag_hgrad", "grav_hgrad", "mg_cross", "pf_persistence"),
                 "network": ("network",)}
MAIN_MODELS = ("A_pixel", "A_seg", "B", "C_all", "D")


class ConfigError(ValueError):
    pass


def notebook_config(CONFIG: dict | None) -> dict:
    c = {**NB_DEFAULTS, **(CONFIG or {})}
    for k in ("experts", "experiments", "export"):
        c[k] = {**NB_DEFAULTS[k], **((CONFIG or {}).get(k) or {})}
    c["mode"] = str(c["mode"]).upper()
    errs = []
    if c["mode"] not in MODES:
        errs.append(f"mode {c['mode']!r} not in {MODES}")
    if str(c["device"]).lower() != "cpu":
        errs.append("device must be 'cpu': no neural network is trained or re-run here")
    if not c["export"]["manual_upload"]:
        errs.append("export.manual_upload must stay True: nothing is uploaded automatically")
    if c["experts"]["dem"] not in ("cached_only", "off", False):
        errs.append("experts.dem: 'cached_only' or 'off' (a full DEM download is never triggered here)")
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


class DiscoveryGeoMoE:
    def __init__(self, CONFIG: dict, head: str, factory=None):
        from ..artifacts.sync import DriveSync
        from ..leaderboard_focus.session import Workspace

        self.nb = notebook_config(CONFIG)
        self.cfg = load_config(self.nb.get("overrides"))
        self.head, self.F = head, factory
        self.local, self.drive = Path(self.nb["local_root"]), Path(self.nb["drive_root"])
        self.out = self.cfg["output_dir"]
        sync = DriveSync(self.local, self.drive) if self.nb["sync_to_drive"] else None
        self.ws = Workspace(self.local / self.out, sync, self.local, save_figures=bool(self.nb["save_png"]),
                            save_parquet=True)
        self.status: dict[str, str] = {s: "NOT_RUN" for s in STAGES}
        self.fps: dict[str, str] = {}
        self.errors: list[dict] = []
        self.fg = self.bst = self.p6 = self.ctx = self.m7 = self.sh = None
        self.units: list | None = None
        self._dem = self._pf2 = None
        self._zm: dict = {}
        if sync is not None and (self.drive / self.out).exists():
            sync.stage_in([self.out])                    # earlier outputs (resume after a disconnect)

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

    def run_stage(self, stage: str, parts: list, outputs: list[str], fn, *, force: bool = False) -> dict:
        """Cached, isolated stage: ``fn() -> (status, info dict)``; the manifest records fingerprint, inputs, status,
        seconds, commit. An exception is recorded as FAILED (traceback kept) and does not stop the run."""
        from .._version import provenance

        fp = self.fp(stage, *parts)
        self.fps[stage] = fp
        m = self.manifest(stage)
        if (not force and self.nb["reuse_cached"] and m.get("fingerprint") == fp and m.get("status") in
                (COMPLETE, SKIPPED) and all(self.ws.path(o).exists() for o in outputs)):
            self.status[stage] = m["status"]
            self.say(f"[{stage}] {CACHED} (fingerprint {fp[:10]}, {m['status']})")
            return m
        t0 = time.perf_counter()
        try:
            st, info = fn()
        except Exception as e:
            info = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
            st = FAILED
            self.errors.append({"stage": stage, **info, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            self.say(f"[{stage}] FAILED: {info['error'][:300]}")
        m = {"stage": stage, "status": st, "fingerprint": fp, "code_rev": GEOMOE_CODE_REV, "version": GEOMOE_VERSION,
             "commit": self.head, "code": provenance(), "seconds": round(time.perf_counter() - t0, 1),
             "outputs": outputs, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {})}
        self.ws.save_json(m, f"manifests/stages/{stage}.json")
        self.status[stage] = st
        self.save_run_manifest()
        self.ws.push()
        if st != FAILED:
            self.say(f"[{stage}] {st} ({m['seconds']} s)" + (f" | {info.get('note')}" if info.get("note") else ""))
        return m

    def ok(self, *stages) -> bool:
        return all(self.status.get(s) in (COMPLETE, SKIPPED) for s in stages)

    def save_run_manifest(self) -> None:
        self.ws.save_json({"code_rev": GEOMOE_CODE_REV, "version": GEOMOE_VERSION, "commit": self.head,
                           "notebook_config": self.nb, "config": self.cfg, "stages": self.status,
                           "fingerprints": self.fps, "n_errors": len(self.errors),
                           "drive_dir": str(self.drive / self.out), "written": time.strftime("%Y-%m-%dT%H:%M:%S")},
                          "manifests/run.json")
        self.ws.save_json(self.errors, "manifests/runtime_errors.json")

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
        """Frozen inputs every stage depends on: data fingerprint, W2 map hashes, G02 run identity."""
        if not hasattr(self, "_ident"):
            w2 = self.bst.w2_inputs() if self.bst is not None else {}
            self._ident = {"data": self.fg.data_fingerprint() if self.fg is not None else None,
                           "w2_maps": w2.get("maps_sha256"), "w2_rows": w2.get("rows_sha256"),
                           "g02": self.bst.g02_ident() if self.bst is not None else None,
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

    def units_of(self, scenario: str) -> list:
        return [su for su in self.units if su.scenario == scenario]

    def release(self, b) -> None:
        for d in (getattr(self.bst, "_raw", {}), getattr(self.bst, "_cnn", {})):
            d.pop(b.key, None)
        b.release()
        gc.collect()

    def enabled_layers(self, L: dict) -> dict:
        off = {x for k, v in self.nb["experts"].items() if k in EXPERT_SWITCH and not v for x in EXPERT_SWITCH[k]}
        return {k: v for k, v in L.items() if k not in off}

    # =========================================================================== [01] attach
    def s01_attach(self) -> dict:
        def fn():
            F = self.F
            if F is None:
                from ..leaderboard_focus.factory import Factory

                F = self.F = Factory(factory_config(self.nb), self.head)
            if F.selftests is None:
                F.selftest()
            if F.sess is not None:                       # the notebook already attached the factory
                self._bind(F)
                return COMPLETE, {"factory_selftests_ok": F.selftest_ok(), "note": "engine reused from the factory"}
            F.preflight()
            if not F.attach():
                return BLOCKED, {"reasons": [str(v) for vs in F.block.values() for v in vs][:6],
                                 "note": "the frozen engine did not attach (factory preflight)"}
            self._bind(F)
            return COMPLETE, {"factory_selftests_ok": F.selftest_ok(), "blocked": F.block,
                              "note": f"engine attached; factory self-tests ok={F.selftest_ok()}"}
        m = self.run_stage("01_attach", [self.head, self.nb["ref"], time.strftime("%Y-%m-%d %H")], [], fn, force=True)
        return m

    def _bind(self, F) -> None:
        self.fg = F.sess.fg
        self.bst, self.p6 = self.fg.bst, self.fg.p6
        self.ctx, self.m7, self.sh = self.p6.ctx, self.p6.m9.m7, self.p6.m9.sh

    # =========================================================================== [02] audit
    def s02_audit(self) -> dict:
        def fn():
            from ..leaderboard_focus.factory import CANDIDATES

            ident = self.engine_identity()
            g02 = self.bst.g02()
            g02_units = sorted(g02.raster_units()) if g02 is not None else []
            mos = self.dem_mosaic()
            units = self.bst.units()
            w2_ok = {u.key: (self.p6.ws.root / "stack" / "maps" / f"{u.task}_{u.zone}.npy").exists() for u in units}
            rows = [
                ("B0 / B1 honest CNN maps", True, "yes (frozen experiments)", "yes", "OOF only on geo folds; discovery "
                 "maps on disc splits", "E1 sources (gems3.official.unit_score)"),
                ("W2 cross-fitted held-out maps", all(w2_ok.values()), "yes (P6 W2)", "yes", "per-unit ranks only "
                 "(full-region refit failed parity: P6.1 X1)", "E1 source, Model A baseline"),
                ("W2 feature rows / unit features", ident.get("w2_rows") is not None, "yes", "yes",
                 "potential fields at one scale only", "expert layers (p6.stack.unit_raw)"),
                ("G01 / G02 rasters", bool(g02_units), "yes (FAST_RESEARCH run)", "yes", f"units with rasters: "
                 f"{g02_units}", "E3 sources (missing units -> E3 UNAVAILABLE there, never zero)"),
                ("Potential fields (mag / grav)", True, "yes", "yes", "second scale computed here (label-free)",
                 "E2 sources"),
                ("Cached 1 m DEM mosaic (M7, 100 m)", mos is not None, "partially (M7 pilot)", "cached only",
                 "pilot coverage only; never downloaded here", "E4 dem layer when present"),
                ("Known-network geometry", True, "yes", "yes", "permitted known only (per scenario)",
                 "E4 network layer + tips pool"),
                ("Segment dedup / geometry (P6.1 X3)", True, "yes", "yes", "skeleton approximations", "inventory"),
                ("Oracle / taxonomy (P6.1 X2)", True, "yes", "definitions reused", "different pool + budget than M7",
                 "stage 05 reconciliation"),
                ("Submission factory writer / validator", True, "yes", "yes", "zone-stitched only", "stage 12"),
                ("Official scorer (FastScorer)", True, "yes", "yes", "OFFICIAL_EXCLUDE_SUPPORTS_TP unverified upstream",
                 "every score"),
            ]
            I = pd.DataFrame(rows, columns=["component", "available", "executed", "reusable", "limitation", "action"])
            self.ws.save_table(I, "audit/component_inventory.csv")
            sc = self.F.ws.load_table("diagnostics/source_compatibility.csv")
            extra = pd.DataFrame([{"source": "geomoe", "check": f"W2 map {k}", "status": "PASS" if v else "ABSENT"}
                                  for k, v in w2_ok.items()] +
                                 [{"source": "geomoe", "check": f"G02 raster {u.task}_{u.zone}",
                                   "status": "PASS" if f"{u.task}_{u.zone}" in g02_units else "ABSENT"} for u in units])
            self.ws.save_table(pd.concat([sc, extra], ignore_index=True), "audit/source_compatibility.csv")
            hist = []
            for cid, spec in CANDIDATES.items():
                h = spec["historical"]
                hist.append({"candidate": cid, "protocol": h.get("protocol"), "disc_dti": h.get("disc_dti"),
                             "geo_dti": h.get("geo_dti", h.get("geo_dti_separate")), "source": h.get("source"),
                             "note": h.get("note", ""), "comparable_to_this_run": "no (recorded, other protocol)"})
            hist.append({"candidate": "w2_champion", "protocol": "public leaderboard", "disc_dti": None,
                         "geo_dti": None, "public_dti": 0.0942, "source": "user-recorded public score",
                         "note": "external reference, never a tuning target", "comparable_to_this_run": "no"})
            hist += [{"candidate": "M7 generator", "protocol": "M7 label-free candidate coverage", "source": "docs/"
                      "M7_DISCOVERY_FIRST.md", "note": "~12.5 % of systems covered by label-free crests",
                      "comparable_to_this_run": "re-measured in stage 05 under one definition"},
                     {"candidate": "P6.1 X2", "protocol": "missed-length taxonomy (CNN / W2 pool @ 5 %)",
                      "source": "docs/p6/P6_RECOVERY_TO_BOOST_AUDIT.md", "note": "82.9 % of missed length "
                      "ranking-limited", "comparable_to_this_run": "re-measured in stage 05 under one definition"}]
            self.ws.save_table(pd.DataFrame(hist), "audit/protocol_comparison.csv")
            return COMPLETE, {"g02_units": g02_units, "dem": mos is not None, "w2_maps": w2_ok,
                              "note": f"G02 units {g02_units}; DEM mosaic {'present' if mos else 'ABSENT'}"}
        return self.run_stage("02_audit", [self.engine_identity()], ["audit/component_inventory.csv",
                                                                     "audit/source_compatibility.csv"], fn)

    # =========================================================================== [03] scenarios
    def s03_scenarios(self) -> dict:
        self.units = SC.build(self)                     # deterministic (seeded); rebuilt every session
        from ..m7.workspace import mask_hash

        th = {su.key: mask_hash(su.unit.eval_truth()) for su in self.units}

        def fn():
            lab = np.asarray(self.ctx.data.labels, bool)
            self.ws.save_table(SC.scenario_table(self.units, self.cfg), "validation/scenarios.csv")
            self.ws.save_table(pd.DataFrame([{"unit": su.key, "scenario": su.scenario, "zone": su.zone,
                                              "row0": su.unit.sl[0].start, "row1": su.unit.sl[0].stop,
                                              "col0": su.unit.sl[1].start, "col1": su.unit.sl[1].stop,
                                              "truth_hash": th[su.key], "base": su.base.key} for su in self.units]),
                               "validation/folds.csv")
            E = pd.concat([su.events for su in self.units if su.scenario == "V2" and len(su.events)] or
                          [pd.DataFrame()], ignore_index=True)
            self.ws.save_table(E, "validation/v2_events.csv")
            chk = []
            for su in self.units:
                u = su.unit
                chk += [{**c, "scenario": su.scenario} for c in SC.leakage_checks(
                    su.key, u.eval_truth(), su.known & u.ev, u.exclude, lab[u.sl] if su.scenario == "V2" else None)]
                prov = "split discovery map" if su.scenario == "V1" else "geo fold OOF"
                chk.append({"unit": su.key, "scenario": su.scenario, "check": "CNN channels are held-out for the "
                            "base unit", "ok": True, "detail": f"{prov} (gems3.official.unit_score; asserted by "
                            "p6.stack.assert_provenance when the unit features load)"})
            chk.append({"unit": "*", "scenario": "*", "check": "V2 classes exclude lateral shifts (no fabricated "
                        "geometry)", "ok": "lateral_shift" not in self.cfg["v2"]["classes"], "detail":
                        str(self.cfg["v2"]["classes"])})
            L = pd.DataFrame(chk)
            self.ws.save_table(L, "audit/label_leakage_audit.csv")
            bad = L[~L.ok.astype(bool)]
            return (COMPLETE if not len(bad) else FAILED), {
                "truth_hashes": th, "leakage_failures": bad.to_dict("records"),
                "note": f"{len(self.units)} scenario units; leakage checks {len(L) - len(bad)}/{len(L)} PASS"}
        return self.run_stage("03_scenarios", [self.engine_identity(), self.cfg["scenarios"], self.cfg["v2"], th],
                              ["validation/scenarios.csv", "audit/label_leakage_audit.csv"], fn)

    # =========================================================================== [04] candidates
    def cand_dir(self) -> str:
        return f"candidates/units/{self.fps['04_candidates'][:10]}"

    def lab(self, su) -> np.ndarray:
        with np.load(self.ws.path(f"{self.cand_dir()}/{_key(su.key)}.npz")) as z:
            return z["lab"]

    def dist_km(self, su) -> np.ndarray:
        with np.load(self.ws.path(f"{self.cand_dir()}/{_key(su.key)}.npz")) as z:
            return z["known_dist_km"].astype(np.float32)

    def s04_candidates(self) -> dict:
        parts = [self.fps.get("03_scenarios"), self.cfg["candidates"], self.cfg["experts"], self.cfg["dem"],
                 self.cfg["network_scale_km"], self.cfg["pf_multiscale_sigma_px"]]
        self.fps["04_candidates"] = self.fp("04_candidates", *parts)

        def fn():
            d = self.cand_dir()
            order = sorted(self.units, key=lambda s: (s.base.task, s.base.zone, s.scenario))
            Ts, Cs, As, Us, Ps, Os, Ds = [], [], [], [], [], [], []
            last = None
            for su in order:
                k = _key(su.key)
                if last is not None and last.key != su.base.key:
                    self.release(last)
                last = su.base
                tp = self.ws.path(f"{d}/{k}_segments.parquet")
                if tp.exists() and self.ws.path(f"{d}/{k}_parts.json").exists() and self.nb["reuse_cached"]:
                    self.say(f"  {su.key}: reused")
                else:
                    t0 = time.perf_counter()
                    L, prov = EX.unit_layers(self, su)
                    L = self.enabled_layers(L)
                    out = CA.build_unit(self, su, L)
                    O = CA.oracle_curve(self, su, out["lab"], out["T"])
                    dk = L.get("known_dist_km")
                    self.save_npz(f"{d}/{k}.npz", lab=out["lab"], known_dist_km=(
                        np.nan_to_num(dk, nan=99.0).astype(np.float16) if dk is not None else
                        np.full(out["lab"].shape, 99.0, np.float16)))
                    self.save_parquet(out["T"], f"{d}/{k}_segments.parquet")
                    for nm, df in (("coverage", out["coverage"]), ("attrs", out["attrs"]), ("unique", out["unique"]),
                                   ("prov", out["prov"]), ("oracle", O), ("dedup", out["dedup"])):
                        self.save_parquet(df, f"{d}/{k}_{nm}.parquet")
                    self.ws.save_json({"unit": su.key, "layers": sorted(L), "provenance": prov,
                                       "pool_px": out["pool_px"], "eligible_px": out["eligible_px"],
                                       "n_candidates": int(out["lab"].max()), "seconds": round(time.perf_counter() - t0,
                                                                                                1)},
                                      f"{d}/{k}_parts.json")
                    self.say(f"  {su.key}: {int(out['lab'].max())} candidates from {len(out['pool_px'])} pools "
                             f"({round(time.perf_counter() - t0, 1)} s)")
                    del L, out
                    gc.collect()
                    self.ws.push()
                Ts.append(self.load_parquet(f"{d}/{k}_segments.parquet"))
                for acc, nm in ((Cs, "coverage"), (As, "attrs"), (Us, "unique"), (Ps, "prov"), (Os, "oracle"),
                                (Ds, "dedup")):
                    acc.append(self.load_parquet(f"{d}/{k}_{nm}.parquet"))
            if last is not None:
                self.release(last)
            cat = (lambda x: pd.concat([a for a in x if len(a)], ignore_index=True) if any(len(a) for a in x)
                   else pd.DataFrame())
            T = cat(Ts)
            geo_cols = [c for c in T.columns if not c.startswith(("mean_", "fin_", "y_"))]
            self.save_parquet(T[geo_cols], "candidates/candidate_inventory.parquet")
            self.save_parquet(T, "experts/expert_features.parquet")
            for nm, df in (("coverage_parts", cat(Cs)), ("system_attributes", cat(As)), ("unique_parts", cat(Us)),
                           ("oracle_curve_parts", cat(Os))):
                self.save_parquet(df, f"candidates/{nm}.parquet")
            P = cat(Ps)
            self.ws.save_table(P, "candidates/source_provenance.csv")
            self.ws.save_table(self.agreement(T), "candidates/candidate_source_agreement.csv")
            self.ws.save_table(cat(Ds).groupby(["unit", "source"]).agg(n=("duplicate", "size"),
                                                                       duplicates=("duplicate", "sum")).reset_index()
                               if len(cat(Ds)) else pd.DataFrame(), "candidates/source_overlap.csv")
            self.geometries(T)
            return COMPLETE, {"n_candidates": int(len(T)), "note": f"{len(T)} candidates over {len(self.units)} units"}
        return self.run_stage("04_candidates", parts, ["candidates/candidate_inventory.parquet",
                                                       "experts/expert_features.parquet"], fn)

    @staticmethod
    def agreement(T: pd.DataFrame) -> pd.DataFrame:
        """Pairwise source agreement: share of candidates proposed by source a that b also proposes."""
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
            import geopandas as gpd  # noqa: F401

            t = self.ctx.data.template_profile
            parts = []
            for su in self.units:
                g = T[T.unit == su.key]
                if len(g):
                    parts.append(CA.to_geodataframe(self.lab(su), g, su.unit.sl, t["transform"], t["crs"]))
            if parts:
                import geopandas as gpd

                gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=t["crs"])
                p = self.ws.path("candidates/candidate_geometries.gpkg")
                p.parent.mkdir(parents=True, exist_ok=True)
                gdf.to_file(p, driver="GPKG")
        except Exception as e:
            self.errors.append({"stage": "04_candidates", "error": f"geometry export skipped: {type(e).__name__}: {e}"})
            self.say(f"  candidate_geometries.gpkg skipped ({type(e).__name__}: {e})")

    # =========================================================================== [05] oracle + W2 baseline
    def w2_smap(self, su) -> np.ndarray | None:
        try:
            return np.where(su.base.ev, self.bst.stack_rank(su.base, self.bst.w2_map(su.base)), np.nan).astype(
                np.float32)
        except (FileNotFoundError, ValueError):
            return None

    def s05_oracle(self) -> dict:
        def fn():
            rows = []
            for su in self.units:
                s = self.w2_smap(su)
                if s is None:
                    continue
                rows += EV.eval_grid(self, su, s, "A_pixel")
                su.unit.release()
            G = pd.DataFrame(rows)
            self.ws.save_table(G, "gating/grid_A_pixel.csv")
            eq = self.metric_equivalence(G)
            self.ws.save_table(eq, "audit/metric_equivalence.csv")
            C = self.load_parquet("candidates/coverage_parts.parquet")
            U = self.load_parquet("candidates/unique_parts.parquet")
            A = self.load_parquet("candidates/system_attributes.parquet")
            O = self.load_parquet("candidates/oracle_curve_parts.parquet")
            w2s = self.w2_systems(G)
            S = DE.coverage_summary(C, U, w2s)
            self.ws.save_table(S, "candidates/oracle_coverage.csv")
            self.ws.save_table(O, "candidates/oracle_curves.csv")
            self.ws.save_table(O, "candidates/oracle_curve.csv")
            inv = C[C.pool == "inventory"][["scenario", "unit", "system_id", "near_share", "covered"]]
            M = inv.merge(A, on=["scenario", "unit", "system_id"], how="left").merge(
                w2s[["unit", "system_id", "recovered"]].rename(columns={"recovered": "w2_recovered"}),
                on=["unit", "system_id"], how="left")
            M["w2_recovered"] = M.w2_recovered.astype("boolean").fillna(False).astype(bool)
            M = M.merge(U.drop(columns=["scenario", "zone"], errors="ignore"), on=["unit", "system_id"], how="left")
            self.ws.save_table(M[~M.covered], "candidates/missed_systems.csv")
            self.ws.save_table(M[M.covered & ~M.w2_recovered], "candidates/unique_systems.csv")
            self.ws.save_table(self.by_type(M), "candidates/coverage_by_fault_type.csv")
            self.ws.save_table(self.protocol_reconciliation(C, w2s), "audit/protocol_comparison_measured.csv")
            return COMPLETE, {"metric_equivalence": eq.status.eq("PASS").all() if len(eq) else None,
                              "note": f"W2 baseline on {G.unit.nunique() if len(G) else 0} units; equivalence "
                                      f"{'PASS' if len(eq) and eq.status.eq('PASS').all() else 'see audit'}"}
        return self.run_stage("05_oracle", [self.fps.get("04_candidates"), self.cfg["evaluation"]],
                              ["gating/grid_A_pixel.csv", "candidates/oracle_coverage.csv"], fn)

    def metric_equivalence(self, G: pd.DataFrame) -> pd.DataFrame:
        """A_pixel components vs the frozen boost-A grid (candidate w2, blend 1) on the shared (task, zone, rep,
        frac) cells: the same scorer, eligibility and W2 maps must give the same numbers."""
        B = self.bst.load_table("grid", "benchmark")
        if not len(B) or not len(G):
            return pd.DataFrame([{"check": "boost-A grid available", "status": "UNAVAILABLE"}])
        B = B[(B.candidate.astype(str) == "w2")]
        g = G[G.scenario.isin(["V1", "V3"])].copy()
        g["task"] = np.where(g.scenario == "V1", "disc_official", "geo")
        m = g.merge(B, on=["task", "zone", "rep", "frac"], suffixes=("", "_frozen"))
        if not len(m):
            return pd.DataFrame([{"check": "shared cells", "status": "UNAVAILABLE", "detail": "no common cell"}])
        rows = []
        for r in m.itertuples():
            d = max(abs(r.tp_w - r.tp_w_frozen), abs(r.fp_w - r.fp_w_frozen), abs(r.fn_w - r.fn_w_frozen))
            rows.append({"task": r.task, "zone": r.zone, "rep": r.rep, "frac": r.frac, "max_abs_diff": d,
                         "status": "PASS" if d <= 1e-3 * max(1.0, r.tp_w_frozen + r.fn_w_frozen) else "FAIL"})
        return pd.DataFrame(rows)

    def w2_systems(self, G: pd.DataFrame) -> pd.DataFrame:
        """W2 (A_pixel) recovered systems at the nested held-out cell of every unit."""
        rows = []
        for sc, g in G.groupby("scenario"):
            per, _ = EV.nested(g)
            for r in per.itertuples():
                h = g[(g.zone == r.zone) & (g.rep == r.rep) & (g.frac == r.frac)]
                ids = set(int(x) for x in str(h.systems.iloc[0]).split(";") if x not in ("", "nan")) if len(h) else set()
                su = next(s for s in self.units if s.scenario == sc and s.zone == int(r.zone))
                allid = np.unique(su.sysmap[su.sysmap >= 0])
                rows += [{"scenario": sc, "unit": su.key, "system_id": int(i), "recovered": int(i) in ids}
                         for i in allid]
        return pd.DataFrame(rows, columns=["scenario", "unit", "system_id", "recovered"])

    @staticmethod
    def by_type(M: pd.DataFrame) -> pd.DataFrame:
        if not len(M):
            return M
        g = M.groupby(["scenario", "subtype", "length_bin"]).agg(
            n_systems=("system_id", "size"), length_km=("length_km", "sum"), covered=("covered", "mean"),
            w2_recovered=("w2_recovered", "mean")).reset_index()
        L = M.assign(cl=M.length_km * M.covered).groupby(["scenario", "subtype", "length_bin"]).cl.sum().reset_index()
        g = g.merge(L, on=["scenario", "subtype", "length_bin"])
        g["covered_length_share"] = g.cl / g.length_km.clip(lower=1e-9)
        return g.drop(columns="cl")

    def protocol_reconciliation(self, C: pd.DataFrame, w2s: pd.DataFrame) -> pd.DataFrame:
        """M7 (label-free crests) vs X2 (learned pool) under ONE definition: same units, truth, 300 m tolerance,
        covered share, pool budget (``pool_frac`` per source)."""
        rows = []
        defs = {"label_free": "M7-like: E2 + E3 lines only", "fam:E1_learned": "X2-like: CNN / W2 / fusion lines",
                "inventory": "unified pool (all families)"}
        cc = self.cfg["candidates"]
        for sc, g in C.groupby("scenario"):
            n = g[["unit", "system_id"]].drop_duplicates().shape[0]
            for pool, lab in defs.items():
                h = g[g.pool == pool]
                rows.append({"scenario": sc, "pool": pool, "definition": lab, "n_systems": n,
                             "covered_share": float(h.covered.mean()) if len(h) else np.nan,
                             "tolerance_m": 100 * float(cc["tol_px"]), "covered_rule": f">= {cc['covered_share']} of "
                             "scored pixels within tolerance", "budget": f"{cc['pool_frac']} of eligible per source"})
            w = w2s[w2s.scenario == sc]
            rows.append({"scenario": sc, "pool": "w2_deployed", "definition": "W2 nested held-out cell (A_pixel)",
                         "n_systems": n, "covered_share": float(w.recovered.mean()) if len(w) else np.nan})
        R = pd.DataFrame(rows)
        R["reading"] = ("ranking-limited (X2 sense) = covered by the learned pool but not recovered by W2's deployed "
                        "cell; coverage-limited (M7 sense) = not covered by the label-free pool")
        return R

    # =========================================================================== [06] decision
    def s06_decision(self) -> dict:
        def fn():
            S = self.ws.load_table("candidates/oracle_coverage.csv")
            d = DE.coverage_decision(S, self.cfg["candidates"]["gate"], self.cfg["primary"])
            self.ws.save_json(d, "candidates/coverage_decision.json")
            return COMPLETE, {"decision": d, "note": f"bottleneck {d['overall']}; independent geology unique "
                                                     f"coverage {d['geology_unique']}"}
        return self.run_stage("06_decision", [self.fps.get("05_oracle"), self.cfg["candidates"]["gate"]],
                              ["candidates/coverage_decision.json"], fn)

    # =========================================================================== [07] experts
    def expert_table(self) -> pd.DataFrame:
        if not hasattr(self, "_X"):
            T = self.load_parquet("experts/expert_features.parquet")
            self._X = pd.concat([T, EX.expert_raw(T, self.cfg["experts"])], axis=1)
        return self._X

    RULE_DESC = ("mean_mag_hgrad", "mean_grav_hgrad", "mean_g02a", "mean_g02c", "mean_dem")

    def views(self, su, X: pd.DataFrame | None = None, only_zone: int | None = None) -> dict:
        """Cross-fitted view of one held-out scenario unit: training mask, normaliser fitted on training rows only,
        normalised expert matrix S and rule descriptors for training and held-out candidates. ``X`` replaces the
        expert table (deployment simulation); ``only_zone`` restricts training to one other zone (fold models)."""
        X = self.expert_table() if X is None else X
        T = X[X.scenario == su.scenario].reset_index(drop=True)
        tr = GA.training_rows(T, su.zone, self.zone_mask(su)) & (T.eligible_share.to_numpy() > 0)
        if only_zone is not None:
            tr &= (T.zone.to_numpy() == int(only_zone))
        GA.assert_disjoint(T.zone[tr].unique(), su.zone)
        held = (T.zone == su.zone).to_numpy()
        kn = int(self.cfg["normalization"]["knots"])
        raw = [f"raw_{f}" for f in EX.FAMILIES]
        ne = EX.fit_normalizer(T[tr], raw, kn)
        nd = EX.fit_normalizer(T[tr], [c for c in self.RULE_DESC if c in T], kn)
        S = EX.normalize(T, ne).to_numpy(float)
        D = pd.concat([T, EX.normalize(T, nd).add_prefix("q_")], axis=1)
        y, w = GA.pu_targets(T, self.cfg["static"]["positive_share"], self.cfg["static"]["pu_unlabeled_weight"])
        return {"T": T, "train": tr, "held": held, "S": S, "D": D, "y": y, "w": w, "norm": {"experts": EX.norm_summary(
            ne), "descriptors": EX.norm_summary(nd)}, "train_zones": sorted(int(z) for z in T.zone[tr].unique()),
            "n_train": int(tr.sum()), "n_pos": int((y[tr] == 1).sum()),
            "n_buffer_dropped": int(((T.zone != su.zone) & ~GA.training_rows(T, su.zone, self.zone_mask(su))).sum())}

    def fit_score(self, model: str, v: dict) -> np.ndarray:
        """Held-out candidate scores of ``model`` from one view, with exactly the fitting rules of stages 08-10
        (used by the deployment simulation; stage-11 re-derives the zone-stitched scores with it as a self-check)."""
        tr, h, T = v["train"], v["held"], v["T"]
        if model == "A_seg":
            return T.mean_w2.to_numpy(float)[h] if "mean_w2" in T else np.full(int(h.sum()), np.nan)
        if v["n_train"] >= 10 and v["n_pos"] >= 1:
            w = np.asarray(GA.fit_static(v["S"][tr], v["y"][tr], v["w"][tr], float(self.cfg["static"]["grid_step"]))[
                "weights"])
        else:
            w = np.full(len(EX.FAMILIES), 1.0 / len(EX.FAMILIES))
        if model == "B":
            return GA.combine(v["S"][h], w)[0]
        if model.startswith("C_"):
            rc = self.cfg["rules"]
            masks = GA.rule_masks(v["D"][h].reset_index(drop=True), rc)
            return GA.rule_gate(v["S"][h], w, masks, GA.rule_variants()[model], rc)[0]
        if model == "D":
            lc = self.cfg["learned"]
            if v["n_train"] < int(lc["min_train_candidates"]) or v["n_pos"] < int(lc["min_train_positives"]):
                return GA.combine(v["S"][h], w)[0]
            Z, _ = self.gate_matrix(v)
            st = GA.standardize_fit(Z[tr])
            Zs = GA.standardize_apply(Z, st)
            g = GA.fit_gate(Zs[tr], v["S"][tr], v["y"][tr], v["w"][tr], w, lc)
            return GA.gate_predict(g, Zs[h], v["S"][h])[0]
        raise KeyError(model)

    def s07_experts(self) -> dict:
        def fn():
            X = self.expert_table()
            self.ws.save_table(EX.availability_table(X, self.cfg["experts"], ["scenario", "zone"]),
                               "experts/source_availability.csv")
            norm, rows = {}, []
            for su in self.units:
                v = self.views(su)
                norm[su.key] = {**v["norm"], "train_zones": v["train_zones"], "n_train": v["n_train"],
                                "n_positive": v["n_pos"], "n_buffer_dropped": v["n_buffer_dropped"],
                                "fitted_on": "training-zone candidates of the same scenario only"}
                h = v["held"]
                rows.append(pd.DataFrame(v["S"][h], columns=[f"s_{f}" for f in EX.FAMILIES]).assign(
                    scenario=su.scenario, zone=su.zone, unit=su.key, label=v["T"].label.to_numpy()[h]))
            self.ws.save_json(norm, "experts/source_normalization.json")
            self.save_parquet(pd.concat(rows, ignore_index=True), "experts/expert_scores.parquet")
            return COMPLETE, {"note": "cross-fitted ECDF normalisation per (scenario, held-out zone)"}
        return self.run_stage("07_experts", [self.fps.get("04_candidates"), self.cfg["normalization"],
                                             self.cfg["static"], self.cfg["train_buffer_km"]],
                              ["experts/expert_scores.parquet", "experts/source_normalization.json"], fn)

    # =========================================================================== shared model evaluation
    def evaluate_models(self, scores: dict, name: str) -> pd.DataFrame:
        """``scores``: {model: DataFrame(unit, label, score, n_px)} -> long grid of every (rep, frac) cell."""
        rows = []
        for su in self.units:
            lab = None
            for m, S in scores.items():
                s = S[S.unit == su.key]
                if not len(s):
                    continue
                lab = self.lab(su) if lab is None else lab
                sm = EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(), s.n_px.to_numpy())
                rows += EV.eval_grid(self, su, sm, m)
            su.unit.release()
            gc.collect()
        G = pd.DataFrame(rows)
        self.ws.save_table(G, f"gating/grid_{name}.csv")
        return G

    # =========================================================================== [08] static fusion
    def s08_static(self) -> dict:
        def fn():
            reg, A, Bs = {}, [], []
            for su in self.units:
                v = self.views(su)
                T, h, tr = v["T"], v["held"], v["train"]
                base = T[h][["unit", "label", "n_px"]].reset_index(drop=True)
                A.append(base.assign(score=T.mean_w2.to_numpy()[h] if "mean_w2" in T else np.nan))
                if v["n_train"] >= 10 and v["n_pos"] >= 1:
                    fit = GA.fit_static(v["S"][tr], v["y"][tr], v["w"][tr], float(self.cfg["static"]["grid_step"]))
                    status = "FITTED"
                else:
                    fit = {"weights": [0.25] * 4, "ap": np.nan}
                    status = "FALLBACK_UNIFORM (too few training candidates)"
                sc, eff = GA.combine(v["S"][h], np.asarray(fit["weights"]))
                Bs.append(base.assign(score=sc))
                reg[f"B|{su.key}"] = {"model": "B", "unit": su.key, "weights": dict(zip(EX.FAMILIES, fit["weights"])),
                                      "train_ap": fit["ap"], "status": status, "train_zones": v["train_zones"],
                                      "n_train": v["n_train"], "n_positive": v["n_pos"]}
            SA, SB = pd.concat(A, ignore_index=True), pd.concat(Bs, ignore_index=True)
            self.save_parquet(SA.assign(model="A_seg"), "gating/scores_A_seg.parquet")
            self.save_parquet(SB.assign(model="B"), "gating/scores_B.parquet")
            G = self.evaluate_models({"A_seg": SA.dropna(subset=["score"]), "B": SB.dropna(subset=["score"])}, "B")
            S, _ = EV.summarize(pd.concat([self.ws.load_table("gating/grid_A_pixel.csv"), G], ignore_index=True))
            W = pd.DataFrame([{**{k: v for k, v in r.items() if k != "weights"}, **{f"w_{k}": x for k, x in
                                                                                 r["weights"].items()}}
                              for r in reg.values()])
            self.ws.save_table(S, "gating/static_fusion_results.csv")
            self.ws.save_table(W, "gating/static_weights.csv")
            self.ws.save_json(reg, "gating/registry_B.json")
            return COMPLETE, {"note": "A_seg + B evaluated on the unified pool"}
        return self.run_stage("08_static", [self.fps.get("07_experts"), self.cfg["static"], self.cfg["evaluation"]],
                              ["gating/grid_B.csv", "gating/scores_B.parquet"], fn)

    # =========================================================================== [09] rule gate
    def s09_rules(self) -> dict:
        def fn():
            reg = self.ws.read_json("gating/registry_B.json") or {}
            rc = self.cfg["rules"]
            variants = GA.rule_variants()
            out, fire = {m: [] for m in variants}, []
            for su in self.units:
                v = self.views(su)
                T, h = v["T"], v["held"]
                w = np.asarray(list((reg.get(f"B|{su.key}") or {}).get("weights", {f: 0.25 for f in
                                                                                  EX.FAMILIES}).values()), float)
                masks = GA.rule_masks(v["D"][h].reset_index(drop=True), rc)
                fire.append({"unit": su.key, **{r: float(m.mean()) for r, m in masks.items()}})
                base = T[h][["unit", "label", "n_px"]].reset_index(drop=True)
                for m, active in variants.items():
                    sc, _ = GA.rule_gate(v["S"][h], w, masks, active, rc)
                    out[m].append(base.assign(score=sc))
            scores = {m: pd.concat(x, ignore_index=True).dropna(subset=["score"]) for m, x in out.items()}
            for m, s in scores.items():
                self.save_parquet(s.assign(model=m), f"gating/scores_{m}.parquet")
            G = self.evaluate_models(scores, "C")
            S, _ = EV.summarize(G)
            self.ws.save_table(S, "gating/rule_gate_results.csv")
            self.ws.save_table(pd.DataFrame(fire), "gating/rule_firing_rates.csv")
            return COMPLETE, {"variants": list(variants), "note": f"{len(variants)} rule-gate variants (C_all + "
                                                                  "one ablation per rule)"}
        if not self.nb["experiments"]["rule_gate"]:
            return self.run_stage("09_rules", ["disabled"], [], lambda: (SKIPPED, {"note": "disabled in CONFIG"}))
        return self.run_stage("09_rules", [self.fps.get("08_static"), self.cfg["rules"], self.cfg["evaluation"]],
                              ["gating/grid_C.csv", "gating/rule_gate_results.csv"], fn)

    # =========================================================================== [10] learned gate + comparison
    def gate_matrix(self, v: dict) -> tuple[np.ndarray, list[str]]:
        T = v["T"]
        cols = [c for c in GA.GATE_DESCRIPTORS if c in T]
        agree, disp = GA.expert_agreement(v["S"])
        Z = np.column_stack([T[cols].to_numpy(float), v["S"], agree, disp])
        return Z, cols + [f"s_{f}" for f in EX.FAMILIES] + ["expert_agreement", "expert_dispersion"]

    def learned_allowed(self) -> tuple[bool, str]:
        lg = self.nb["experiments"]["learned_gate"]
        if lg is False:
            return False, "disabled in CONFIG"
        d = self.ws.read_json("candidates/coverage_decision.json") or {}
        if lg is True:
            return True, "forced in CONFIG"
        if not d:
            return False, "coverage decision absent (stage 06 did not complete)"
        if not d.get("learned_gate_allowed"):
            return False, f"bottleneck {d.get('overall')}: candidate generation first, the gate cannot recover " \
                          "systems no candidate covers"
        return True, f"bottleneck {d.get('overall')}"

    def s10_learned(self) -> dict:
        allowed, why = self.learned_allowed()

        def fn():
            lc = self.cfg["learned"]
            reg_b = self.ws.read_json("gating/registry_B.json") or {}
            reg, rows, imps, wts = {}, [], [], []
            for su in self.units:
                v = self.views(su)
                T, h, tr = v["T"], v["held"], v["train"]
                Z, cols = self.gate_matrix(v)
                prior = np.asarray(list((reg_b.get(f"B|{su.key}") or {}).get("weights", {f: 0.25 for f in
                                                                                         EX.FAMILIES}).values()))
                base = T[h][["unit", "label", "n_px"]].reset_index(drop=True)
                if v["n_train"] < int(lc["min_train_candidates"]) or v["n_pos"] < int(lc["min_train_positives"]):
                    sc, _ = GA.combine(v["S"][h], prior)
                    reg[f"D|{su.key}"] = {"status": "FALLBACK_STATIC (too few training candidates / positives)",
                                          "n_train": v["n_train"], "n_positive": v["n_pos"]}
                    rows.append(base.assign(score=sc))
                    continue
                stz = GA.standardize_fit(Z[tr])
                Zs = GA.standardize_apply(Z, stz)
                model = GA.fit_gate(Zs[tr], v["S"][tr], v["y"][tr], v["w"][tr], prior, lc)
                sc, g = GA.gate_predict(model, Zs[h], v["S"][h])
                col = GA.collapse(g, v["S"][h], lc["collapse"])
                reg[f"D|{su.key}"] = {"status": "FITTED", **{k: model[k] for k in ("converged", "n_iter", "objective",
                                                                                   "message")},
                                      "collapse": col, "train_zones": v["train_zones"], "n_train": v["n_train"],
                                      "n_positive": v["n_pos"], "features": cols, "standardize": stz,
                                      "theta": model["theta"], "prior": model["prior"]}
                rows.append(base.assign(score=sc))
                E, d = model["E"], model["d"]
                V = np.asarray(model["theta"][:E * d]).reshape(E, d)
                imps.append(pd.DataFrame({"unit": su.key, "feature": cols, **{f"coef_{f}": V[i] for i, f in
                                                                              enumerate(EX.FAMILIES)}}))
                wts.append(pd.DataFrame(g, columns=[f"g_{f}" for f in EX.FAMILIES]).assign(
                    unit=su.key, scenario=su.scenario, zone=su.zone, label=base.label.to_numpy(),
                    known_dist_km=T.mean_known_dist_km.to_numpy()[h] if "mean_known_dist_km" in T else np.nan,
                    dem_observed=T.mean_dem_observed.to_numpy()[h] if "mean_dem_observed" in T else np.nan))
            SD = pd.concat(rows, ignore_index=True)
            self.save_parquet(SD.assign(model="D"), "gating/scores_D.parquet")
            G = self.evaluate_models({"D": SD.dropna(subset=["score"])}, "D")
            S, _ = EV.summarize(G)
            I = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
            if len(I):
                I["abs_mean"] = I[[c for c in I if c.startswith("coef_")]].abs().mean(1)
            self.ws.save_table(S, "gating/learned_gate_results.csv")
            self.ws.save_table(I, "gating/feature_importance.csv")
            self.save_parquet(pd.concat(wts, ignore_index=True) if wts else pd.DataFrame(), "gating/gating_weights.parquet")
            self.ws.save_json(reg, "gating/registry_D.json")
            n_col = sum(1 for r in reg.values() if (r.get("collapse") or {}).get("collapsed"))
            return COMPLETE, {"collapsed_units": n_col, "note": f"learned gate on {len(reg)} units; collapsed {n_col}"}

        if not allowed:
            m = self.run_stage("10_learned", ["skipped", why], [], lambda: (SKIPPED, {"reason": why, "note": why}))
        else:
            m = self.run_stage("10_learned", [self.fps.get("09_rules"), self.fps.get("08_static"), self.cfg["learned"],
                                              self.cfg["evaluation"]], ["gating/grid_D.csv"], fn)
        self.compare()
        return m

    def compare(self) -> dict:
        """Paired, cross-fitted comparison of every evaluated model + the predeclared selection (part of [10])."""
        def fn():
            G = pd.concat([self.ws.load_table(f"gating/grid_{n}.csv") for n in ("A_pixel", "B", "C", "D")],
                          ignore_index=True)
            EV.guard_oracle(G)
            G = EV.budget_rows(G, float(self.fg.cfg["submission"]["budget_tolerance"]))
            S, P = EV.summarize(G)
            self.ws.save_table(S, "gating/model_summary.csv")
            self.ws.save_table(P, "validation/per_zone_metrics.csv")
            A = self.load_parquet("candidates/system_attributes.parquet")
            short = {(r.unit, int(r.system_id)) for r in A.itertuples() if r.length_km < float(self.cfg["short_fault_km"])}
            blocks, bands, systems = [], [], []
            models = [m for m in MAIN_MODELS if m in set(G.model)]
            for su in self.units:
                lab = self.lab(su)
                dk = self.dist_km(su)
                sid = {i for (k, i) in short if k == su.key}
                for m in models:
                    pz = P[(P.scenario == su.scenario) & (P.model == m) & (P.zone == su.zone)]
                    if not len(pz):
                        continue
                    sm = self.w2_smap(su) if m == "A_pixel" else self.model_smap(m, su, lab)
                    if sm is None:
                        continue
                    r = pz.iloc[0]
                    d = EV.chosen_details(self, su, sm, m, str(r.rep), float(r.frac), sid, dk)
                    blocks.append(d["blocks"])
                    bands.append(d["bands"])
                    systems.append(d["systems"].assign(unit=su.key))
                    q = d["q"]
                    self.save_npz(f"predictions/heldout/{_key(su.key)}_{m}.npz", q=np.packbits(q), shape=np.asarray(
                        q.shape), rep=str(r.rep), frac=float(r.frac))
                su.unit.release()
                gc.collect()
            Bk = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
            ec = self.cfg["evaluation"]
            pairs = [(m, "A_pixel") for m in models if m != "A_pixel"] + [(m, "B") for m in ("C_all", "D") if m in
                                                                          models] + [("A_seg", "A_pixel")]
            PC = (EV.paired(Bk, sorted(set(pairs)), int(ec["n_boot"]), int(self.cfg["v2"]["seed"])) if len(Bk)
                  else pd.DataFrame())
            self.ws.save_table(PC, "gating/paired_comparison.csv")
            NB = EV.band_table(pd.concat(bands, ignore_index=True)) if bands else pd.DataFrame()
            self.ws.save_table(NB, "validation/near_known_metrics.csv")
            SY = pd.concat(systems, ignore_index=True) if systems else pd.DataFrame()
            self.save_parquet(SY, "validation/system_recovery.parquet")
            self.ws.save_table(self.fault_type_metrics(SY, A), "validation/per_fault_type_metrics.csv")
            hard = self.hard_checks(G)
            reg_d = self.ws.read_json("gating/registry_D.json") or {}
            collapsed = sum(1 for r in reg_d.values() if (r.get("collapse") or {}).get("collapsed")) > \
                len(reg_d) / 2 if reg_d else False
            sel = DE.select_model(S, PC, hard, collapsed, self.cfg["selection"])
            self.ws.save_json({**sel, "hard_checks": hard, "collapsed_majority": collapsed}, "gating/selection.json")
            reg = {"B": self.ws.read_json("gating/registry_B.json"), "D": reg_d, "rules": self.cfg["rules"],
                   "selection": sel["selected"], "normalization": "experts/source_normalization.json",
                   "construction": {m: "cross-fitted held-out (zone k <- fit without zone k), nested (rep, frac)"
                                    for m in models}}
            self.ws.save_json(reg, "gating/model_registry.json")
            return COMPLETE, {"selected": sel["selected"], "hard": hard, "note": f"selection {sel['selected']}: "
                                                                                f"{'; '.join(sel['reasons'])[:200]}"}
        return self.run_stage("10b_compare", [self.fps.get("10_learned"), self.fps.get("09_rules"),
                                              self.fps.get("08_static"), self.cfg["selection"],
                                              self.cfg["evaluation"]], ["gating/selection.json",
                                                                        "gating/paired_comparison.csv"], fn)

    def model_smap(self, m: str, su, lab: np.ndarray) -> np.ndarray | None:
        p = self.ws.path(f"gating/scores_{m}.parquet")
        if not p.exists():
            return None
        S = pd.read_parquet(p)
        s = S[S.unit == su.key].dropna(subset=["score"])
        return EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(), s.n_px.to_numpy()) if len(s) else None

    def hard_checks(self, G: pd.DataFrame) -> dict:
        L = self.ws.load_table("audit/label_leakage_audit.csv")
        E = self.ws.load_table("audit/metric_equivalence.csv")
        return {"no_leakage": bool(len(L)) and bool(L.ok.astype(bool).all()),
                "official_metric_equivalence": bool(len(E)) and "status" in E and bool(E.status.eq("PASS").all()),
                "fair_budgets": bool(G.budget_ok.all()) if "budget_ok" in G else False,
                "w2_baseline_available": bool((G.model == "A_pixel").any())}

    @staticmethod
    def fault_type_metrics(SY: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
        if not len(SY):
            return SY
        M = SY.merge(A[["unit", "system_id", "subtype", "length_bin", "length_km"]], on=["unit", "system_id"],
                     how="left")
        return M.groupby(["scenario", "model", "subtype", "length_bin"]).agg(
            n_systems=("system_id", "size"), recovered_share=("recovered", "mean"),
            short_recovered_share=("recovered", lambda s: float(s[M.loc[s.index, "short"]].mean())
                                   if M.loc[s.index, "short"].any() else np.nan)).reset_index()

    # =========================================================================== [11] deployment comparison
    def s11_deployment(self) -> dict:
        from . import deployment as DP

        if not self.nb["experiments"]["deployment_comparison"]:
            return self.run_stage("11_deployment", ["disabled"], [], lambda: (SKIPPED, {"note": "disabled"}))
        return self.run_stage("11_deployment", [self.fps.get("10b_compare"), self.cfg["deployment"]],
                              ["gating/deployment_comparison.csv"], lambda: DP.run(self))

    # =========================================================================== [12] export
    def s12_export(self) -> dict:
        from . import export as XP

        sel = self.ws.read_json("gating/selection.json") or {}
        ex = self.nb["export"]
        if not ex["enabled"] or self.nb["mode"] == "DIAGNOSTIC":
            return self.run_stage("12_export", ["disabled"], [], lambda: (SKIPPED, {"note": "export disabled / "
                                                                                           "DIAGNOSTIC mode"}))
        return self.run_stage("12_export", [self.fps.get("10b_compare"), self.fps.get("11_deployment"), sel, ex,
                                            self.fg.cfg["submission"]], ["submissions/submission_manifest.json"],
                              lambda: XP.run(self, sel))

    # =========================================================================== [13] reports
    def s13_reports(self) -> dict:
        from . import figures as FI
        from . import report as RP

        def fn():
            FI.write_all(self)
            RP.write_all(self)
            return COMPLETE, {"note": "reports/ and figures/ written from persisted tables"}
        return self.run_stage("13_reports", [time.strftime("%Y-%m-%dT%H:%M")], ["reports/SCIENTIFIC_DECISION.md"], fn,
                              force=True)

    # =========================================================================== RUN_ALL
    def run(self) -> dict:
        t0 = time.perf_counter()
        self.s01_attach()
        if not self.ok("01_attach"):
            self.say("engine not attached: only the reports are written (reports/NEXT_ACTION.md)")
            self.s13_reports()
            return self.status
        steps = [("02_audit", self.s02_audit, ()), ("03_scenarios", self.s03_scenarios, ()),
                 ("04_candidates", self.s04_candidates, ("03_scenarios",)),
                 ("05_oracle", self.s05_oracle, ("04_candidates",)), ("06_decision", self.s06_decision, ("05_oracle",)),
                 ("07_experts", self.s07_experts, ("04_candidates",)), ("08_static", self.s08_static, ("07_experts",)),
                 ("09_rules", self.s09_rules, ("08_static",)), ("10_learned", self.s10_learned, ("08_static",)),
                 ("11_deployment", self.s11_deployment, ("10b_compare",)), ("12_export", self.s12_export,
                                                                            ("10b_compare",))]
        if self.nb["mode"] == "REPORT_ONLY":
            steps = steps[:2]
        for name, fn, need in steps:
            if not self.ok(*need):
                self.status[name] = BLOCKED
                self.say(f"[{name}] BLOCKED: upstream {[s for s in need if not self.ok(s)]} not complete")
                continue
            fn()
        self.s13_reports()
        self.say(f"done in {round(time.perf_counter() - t0, 1)} s | stages {self.status}")
        return self.status
