"""G14.1 orchestrator: stage runner with resumability, artifact reuse and Drive synchronisation (Phases B-T).

    pipe = G141Pipeline(CONFIG, g14_overrides=G14_OVERRIDES)
    pipe.plan()                      # what will run and why the rest is skipped
    pipe.run("FAST")                 # R00-R05 (+R12, R17 when the mode allows) from existing artifacts
    pipe.report()                    # decision report + figures from whatever stages produced outputs

Each stage writes CSV (+ Parquet when pyarrow is present) and a JSON summary under
``<local_root>/g14_1/<subdir>/`` and mirrors them to ``<drive>/outputs/G14_1_CRITICAL_RECOVERY/<subdir>/``.
The stage ledger (``manifests/stages.json``) records config hash, code revision, input digest, outputs and timings;
a completed stage with the same digest is reused, not recomputed (``resume=True``, ``force_rebuild=False``).

UNTESTED: written without execution.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import G14_1_VERSION, OUTPUT_DIRS
from .config import G141Config

PROFILE_REPS = {
    "FAST": (("raw", "power2", "top", "crest_w1", "crest_w0", "sparse_soft"), ("pixel_exact", "mask_before")),
    "STANDARD": (("raw", "power2", "power4", "top", "crest_w1", "crest_w0", "nms", "skeleton", "soft_skeleton",
                  "sparse_soft", "hysteresis", "segment_rank"), ("pixel_exact", "mask_before", "mask_after", "ring1")),
    "FULL": (None, ("pixel_exact", "mask_before", "mask_after", "mask_both", "ring1")),
}
CONFIG_REP_NAMES = {"power": ("power2", "power4")}


class G141Pipeline:
    def __init__(self, config: dict | G141Config, g14_overrides: dict | None = None, repo_dir: str = "/content/GEMS",
                 log=print):
        from .artifacts import StageLedger, code_revision

        self.cfg = config if isinstance(config, G141Config) else G141Config.from_dict(config)
        self.g14_overrides = dict(g14_overrides or {})
        self.log = log
        self.local = Path(self.cfg.local_root) / "g14_1"
        self.drive = Path(self.cfg.out_root)
        for d in OUTPUT_DIRS:
            (self.local / d).mkdir(parents=True, exist_ok=True)
        self.code_rev = code_revision(repo_dir)
        self.ledger = StageLedger(self.local / "manifests" / "stages.json", self.cfg.semantic_hash(), self.code_rev)
        self._g14 = self._inv = None
        self.results: dict = {}

    # ------------------------------------------------------------------ shared objects
    @property
    def g14(self):
        """G14 pipeline object of the source experiment (data, splits, stack, experiment directory)."""
        if self._g14 is None:
            from ..g14.pipeline import G14Pipeline
            from .leakage import g14_config_for

            g = g14_config_for(self.cfg.source_experiment, {**self.g14_overrides, "local_root": self.cfg.local_root,
                                                             "drive_root": self.cfg.drive_root})
            self._g14 = G14Pipeline(g, log=self.log)
            if self._g14.exp_dir.name != self.cfg.source_experiment:
                raise RuntimeError(f"G14 experiment dir {self._g14.exp_dir.name} != {self.cfg.source_experiment}")
        return self._g14

    @property
    def data(self):
        return self.g14.data

    @property
    def system(self):
        return self.g14.system

    @property
    def pixel_m(self) -> float:
        from .geometry import pixel_size_m

        py, px = pixel_size_m(self.data.transform)
        if abs(py - px) > 1e-6:
            raise ValueError(f"non-square pixels {py} x {px}: kernels need per-axis handling")
        return px

    @property
    def roots(self):
        from .artifacts import Roots

        return Roots(Path(self.cfg.local_root), Path(self.cfg.drive_root), self.cfg.source_experiment,
                     Path(self.cfg.g14_drive_root))

    @property
    def inv(self) -> pd.DataFrame:
        if self._inv is None:
            p = self.local / "inventory" / "inventory.csv"
            if p.exists() and self.cfg.reuse_artifacts and not self.cfg.force_rebuild:
                self._inv = pd.read_csv(p)
            else:
                from .artifacts import discover

                shape = None
                try:
                    shape = self.data.shape
                except Exception as e:  # noqa: BLE001
                    self.log(f"[g14.1] official inputs not loaded ({e}); inventory without grid checks")
                self._inv = discover(self.roots, shape)
        return self._inv

    # ------------------------------------------------------------------ io helpers
    def _df(self, df: pd.DataFrame, sub: str, name: str) -> Path:
        p = self.local / sub / f"{name}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        try:
            df.to_parquet(p.with_suffix(".parquet"), index=False)
        except Exception:  # noqa: BLE001  (pyarrow absent or mixed object columns)
            pass
        return p

    def _json(self, obj, sub: str, name: str) -> Path:
        p = self.local / sub / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, default=_jsonable))
        return p

    def sync(self, subs=None) -> int:
        """Mirror local G14.1 outputs to Drive (size / mtime check; no partial files)."""
        if not Path(self.cfg.drive_root).exists():
            return 0
        n = 0
        for sub in subs or OUTPUT_DIRS:
            src = self.local / sub
            for f in src.rglob("*") if src.exists() else []:
                if not f.is_file() or f.suffix == ".partial":
                    continue
                dst = self.drive / f.relative_to(self.local)
                if dst.exists() and dst.stat().st_size == f.stat().st_size and dst.stat().st_mtime >= f.stat().st_mtime:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
        return n

    def g14_unit_maps(self) -> dict:
        from .artifacts import unit_maps
        from .champion import unit_maps_from_paths

        return unit_maps_from_paths(unit_maps(self.inv, "g14_unit_maps"))

    def full_map(self) -> np.ndarray | None:
        from .artifacts import best_copy

        p = best_copy(self.inv, "g14_full_map")
        return None if p is None else np.asarray(np.load(p, mmap_mode="r"), np.float32)

    def eval_units(self):
        return self.system.eval_units(self.cfg.tasks)

    def _reps(self):
        reps, orders = PROFILE_REPS[self.cfg.profile]
        if reps is None:
            reps = []
            for r in self.cfg.sweep.representations:
                reps += list(CONFIG_REP_NAMES.get(r, (r,)))
            from .representations import ALL

            reps = [r for r in reps if r in ALL]
        return tuple(reps), tuple(o for o in orders if o in self.cfg.sweep.masking_orders or o == "pixel_exact")

    # ------------------------------------------------------------------ stage runner
    def plan(self) -> pd.DataFrame:
        from .experiments import plan

        return plan(self.cfg)

    def _stage(self, sid: str, fn, inputs_sig=()):
        from .experiments import STAGE

        st = STAGE[sid]
        if not self.cfg.stage_enabled(st.switch):
            self.log(f"[g14.1] {sid} skipped (switch '{st.switch}' off or not allowed in mode {self.cfg.mode})")
            return None
        digest = self.ledger.digest(sid, self.cfg.semantic_hash(), self.cfg.source_experiment, list(inputs_sig))
        summ_p = self.local / "reports" / f"{sid}.json"
        if self.cfg.resume and not self.cfg.force_rebuild and self.ledger.fresh(sid, digest) and summ_p.exists():
            self.log(f"[g14.1] {sid} reused (same inputs, outputs present)")
            self.results[sid] = json.loads(summ_p.read_text())
            return self.results[sid]
        self.log(f"[g14.1] {sid} {st.title} ...")
        t0 = self.ledger.start(sid, digest)
        try:
            outputs, summary = fn()
        except Exception as e:  # noqa: BLE001
            self.ledger.fail(sid, t0, traceback.format_exc())
            self.log(f"[g14.1] {sid} FAILED: {type(e).__name__}: {e}")
            raise
        summary = {"stage": sid, "title": st.title, "version": G14_1_VERSION, "cfg_hash": self.cfg.semantic_hash(),
                   "code_rev": self.code_rev, "source_experiment": self.cfg.source_experiment,
                   "seconds": round(time.perf_counter() - t0, 1), **summary}
        p = self._json(summary, "reports", sid)
        self.ledger.finish(sid, t0, list(outputs) + [p])
        self.results[sid] = summary
        self.sync()
        return summary

    def run(self, profile: str | None = None, stages: tuple | None = None) -> dict:
        """Run the planned stages in order. ``stages`` restricts to a subset (e.g. ('R00', 'R01'))."""
        if profile and profile != self.cfg.profile:
            self.cfg = self.cfg.updated({"profile": profile})
        order = [("R00", self.r00_inventory), ("R01", self.r01_geometry), ("R02", self.r02_diagnostics),
                 ("R03", self.r03_sweep), ("R04", self.r04_decomposition), ("R05", self.r05_champion),
                 ("R05b", self.r05b_atlas), ("R06", self.r06_counterfactuals), ("R07", self.r07_discovery_splits),
                 ("R08", lambda: self.train_stage("R08")), ("R09", lambda: self.train_stage("R09")),
                 ("R10", lambda: self.train_stage("R10")), ("R11", lambda: self.train_stage("R11")),
                 ("R12", self.r12_residual), ("R13", lambda: self.train_stage("R13")),
                 ("R14", lambda: self.train_stage("R14")), ("R15", lambda: self.train_stage("R15")),
                 ("R16", self.r16_ensemble)]
        pl = self.plan().set_index("stage")
        for sid, fn in order:
            if stages and sid not in stages:
                continue
            if pl.loc[sid, "action"] != "RUN" and not stages:
                continue
            self._stage(sid, lambda f=fn: f())
        return self.results

    # ================================================================== R00
    def r00_inventory(self):
        from .artifacts import experiment_hash_check, feature_provenance, preflight

        self._inv = None
        inv = self.inv
        out = [self._df(inv, "inventory", "inventory")]
        man = [Path(p) for p in inv[(inv.artifact == "g14_train_manifests") & (inv.location == "local")].path] or \
            [Path(p) for p in inv[inv.artifact == "g14_train_manifests"].path]
        hc = experiment_hash_check(self.cfg.source_experiment, man)
        fp = feature_provenance(man) if man else pd.DataFrame()
        out += [self._df(hc, "inventory", "experiment_hash_check"), self._df(fp, "inventory", "feature_provenance")]
        try:
            paths = {k: v for k, v in self.g14.paths.items() if k != "notes"}
            shape = self.data.shape
        except Exception as e:  # noqa: BLE001
            paths, shape = None, None
            self.log(f"[g14.1] R00: official inputs unavailable: {e}")
        pf = preflight(inv, paths, shape)
        out.append(self._df(pf, "inventory", "preflight"))
        summ = {"n_artifacts": int(len(inv)), "by_artifact": inv.artifact.value_counts().to_dict() if len(inv) else {},
                "hash_mismatch": int((hc.status != "OK").sum()) if len(hc) else 0,
                "target_derived_channels": int(fp.target_derived.sum()) if len(fp) else None,
                "channels_per_unit": fp.groupby("unit").size().to_dict() if len(fp) else {},
                "preflight": pf.set_index("stage").status.to_dict(), "official_inputs": paths, "grid": shape}
        return out, summ

    # ================================================================== R01
    def r01_geometry(self):
        from .geometry import geometry_report, ridge_offset_to_known, round_trip, submission_parity

        inv, data = self.inv, self.data
        subs = inv[inv.artifact == "g14_submissions"]
        extra = {f"submission:{Path(p).name}": p for p in subs.path}
        rep = geometry_report(data, extra)
        out = [self._df(rep["grid"], "geometry_audit", "grid_table"),
               self._df(rep["assumptions"], "geometry_audit", "hard_coded_assumptions"),
               self._json({k: rep[k] for k in ("kernel", "footprint", "array_axes")}, "geometry_audit", "kernel_footprint")]
        full = self.full_map()
        parities = []
        for p in subs.path:
            p = Path(p)
            op = _submission_operating_point(p)
            expected = None
            if full is not None and op is not None:
                from ..g14.submission import operating_point_map

                expected = operating_point_map(full, data, op["rep"], op["frac"])
            r = submission_parity(p, data, expected)
            r.update(operating_point=op)
            parities.append(r)
            if expected is not None:
                rt = round_trip(expected, data, self.local / "geometry_audit")
                parities.append({**rt, "file": f"round_trip:{p.name}"})
        if parities:
            out.append(self._df(pd.DataFrame(parities), "geometry_audit", "submission_parity"))
        ro = ridge_offset_to_known(full, data.labels, data.valid) if full is not None else {}
        out.append(self._json(ro, "geometry_audit", "ridge_offset_to_known"))
        summ = {"kernel": rep["kernel"], "footprint": rep["footprint"],
                "grid_status": rep["grid"].set_index("raster").status.to_dict() if "status" in rep["grid"] else {},
                "parity": parities[0] if parities else None, "ridge_offset": ro,
                "inconsistent_assumptions": int((rep["assumptions"].status == "INCONSISTENT").sum())}
        return out, summ

    # ================================================================== R02
    def r02_diagnostics(self):
        from . import leakage as LK
        from . import probability as P
        from .discovery import protocol_units

        data, out, summ = self.data, [], {}
        full = self.full_map()
        regimes = self._regimes()
        if full is not None:
            s = P.summary(full, data.valid, data.labels, data.transform)
            out += [self._json(s, "probability_diagnostics", "full_summary"),
                    self._df(P.histogram(full, data.valid, data.labels), "probability_diagnostics", "full_histogram"),
                    self._df(P.distance_curve(full, data.valid, data.labels, data.transform),
                             "probability_diagnostics", "full_distance_curve"),
                    self._df(pd.DataFrame([P.component_stats(full, data.valid, data.labels, data.transform, f)
                                           for f in (0.005, 0.02, 0.08)]), "probability_diagnostics", "full_components"),
                    self._json(P.spatial_concentration(full, data.valid), "probability_diagnostics", "full_concentration")]
            if regimes is not None:
                out.append(self._df(P.regional_calibration(full, data.valid, regimes), "probability_diagnostics",
                                    "full_regional_calibration"))
            summ["full_map"] = s
        maps = self.g14_unit_maps()
        rk = []
        for u in self.eval_units():
            k = (u.task, u.zone)
            if k in maps:
                rk.append({"task": u.task, "zone": u.zone, **P.truth_ranking(np.asarray(u.crop(maps[k]), np.float32),
                                                                         u.eval_truth(), u.scored_mask())})
                u.release()
        if rk:
            rk = pd.DataFrame(rk)
            out.append(self._df(rk, "probability_diagnostics", "heldout_ranking"))
            summ["ranking"] = {"auc_mean": float(rk.auc.mean()), "ap_mean": float(rk.ap.mean())}
        ov, sep, cp = [], [], []
        for u in protocol_units(self.system, self.cfg.validation.protocol):
            ov.append(LK.target_context_overlap(u))
            sep.append(LK.known_only_separability(u, data))
            cp.append(LK.copy_baseline(u, data, (self.pixel_m, self.pixel_m)))
        ov, sep, cp = pd.DataFrame(ov), pd.DataFrame(sep), pd.concat(cp, ignore_index=True) if cp else pd.DataFrame()
        out += [self._df(ov, "leakage_audit", "target_context_overlap"), self._df(sep, "leakage_audit", "separability"),
                self._df(cp, "leakage_audit", "copy_baseline")]
        ck = cp[(cp.region == "inner_val") & (cp.predictor == "copy_kernel")] if len(cp) else cp
        summ["leakage"] = {"train_pos_in_context_mean": float(ov.train_pos_in_context.mean()) if len(ov) else None,
                           "units_with_shortcut": int((ov.status == "SHORTCUT").sum()) if len(ov) else None,
                           "copy_kernel_inner_dti_mean": float(ck.dti.mean()) if len(ck) else None,
                           "auc_train_mean": float(sep.auc_train.mean()) if len(sep) else None,
                           "auc_heldout_mean": float(sep.auc_heldout.mean()) if len(sep) else None}
        return out, summ

    # ================================================================== R03
    def r03_sweep(self):
        from . import calibration as C
        from .representations import best_per_rep, crossfit_reps, sweep_unit

        maps = self.g14_unit_maps()
        if not maps:
            raise FileNotFoundError("no G14 held-out unit maps (maps/geo_*.npy): restore them from Drive or run "
                                    "pipe.g14.predict_heldout() in a G14 EVALUATE session")
        reps, orders = self._reps()
        fracs = self.cfg.sweep.area_fracs
        units = [u for u in self.eval_units() if (u.task, u.zone) in maps]
        tab = pd.concat([sweep_unit(u, maps[(u.task, u.zone)], reps, fracs, orders) for u in units], ignore_index=True)
        out = [self._df(tab, "representation_sweep", "g14_sweep")]
        cf = crossfit_reps(tab)
        best = best_per_rep(tab)
        out.append(self._df(best, "representation_sweep", "g14_best_per_representation"))
        sel, summ = {}, {"tasks": {}}
        for task, r in cf.items():
            out.append(self._df(r["per_zone"], "representation_sweep", f"g14_crossfit_{task}"))
            s = r["summary"]
            sel[task] = {"rep": s["params_all"]["rep"], "order": s["params_all"]["order"],
                         "frac": float(s["params_all"]["target_frac"]), "calibration": "identity",
                         "crossfit_dti": s["dti_crossfit"], "zone_min": s["zone_min"], "optimism": s["optimism"]}
            orig = tab[(tab.task == task) & (tab.rep == "crest_w1") & (tab.order == "pixel_exact")
                       & np.isclose(tab.target_frac, 0.08)]
            from ..g13.evaluation import pooled_dti

            summ["tasks"][task] = {**sel[task], "g14_original_choice_dti": pooled_dti(orig) if len(orig) else None,
                                   "crest_w1_8pct_dilation_factor": float(orig.dilation_factor.mean()) if len(orig) else None,
                                   "crest_w1_8pct_final_frac": float(orig.final_frac.mean()) if len(orig) else None}
        if self.cfg.stage_enabled("calibration"):
            kinds = [k for k in self.cfg.sweep.calibrations if k != "regime" or self._regimes() is not None]
            ct = C.crossfit_calibration(maps, units, kinds, ("raw", "sparse_soft"), (0.0025, 0.005, 0.01, 0.02, 0.05),
                                        self._regimes())
            out.append(self._df(ct, "calibration", "g14_calibration_sweep"))
            from ..g13.evaluation import crossfit

            for task, t in ct.groupby("task"):
                per, s = crossfit(t, ["calibration", "rep", "target_frac"])
                out.append(self._df(per, "calibration", f"g14_calibration_crossfit_{task}"))
                if s["dti_crossfit"] > sel.get(task, {}).get("crossfit_dti", -1):
                    kind = s["params_all"]["calibration"]
                    prm = {}
                    if kind in ("temperature", "power", "identity"):
                        crops = [(u, np.asarray(u.crop(maps[(u.task, u.zone)]), np.float32)) for u in units
                                 if u.task == task]
                        _, prm = C.fit(kind, crops)
                        sel[task] = {"rep": s["params_all"]["rep"], "order": "pixel_exact",
                                     "frac": float(s["params_all"]["target_frac"]), "calibration": kind,
                                     "calibration_params": {"kind": kind, **prm}, "crossfit_dti": s["dti_crossfit"]}
        self._merge_selection("g14", sel)
        out.append(self.local / "calibration" / "selected_operating_points.json")
        g = summ["tasks"].get("geo") or next(iter(summ["tasks"].values()), {})
        summ["sweep"] = {"best_crossfit_dti": g.get("crossfit_dti"), "g14_original_choice_dti": g.get("g14_original_choice_dti"),
                         "task": "geo" if "geo" in summ["tasks"] else next(iter(summ["tasks"]), None)}
        summ["selected"] = sel
        summ["area_optimum"] = {t: v["frac"] for t, v in sel.items()}
        return out, summ

    def _merge_selection(self, model: str, sel: dict) -> None:
        p = self.local / "calibration" / "selected_operating_points.json"
        j = json.loads(p.read_text()) if p.exists() else {}
        j[model] = {**j.get(model, {}), **sel}
        p.write_text(json.dumps(j, indent=1, default=_jsonable))

    # ================================================================== R04
    def r04_decomposition(self):
        from scipy import ndimage

        from . import metric_audit as M
        from .representations import apply_rep

        maps = self.g14_unit_maps()
        sel_p = self.local / "representation_sweep"
        rows, fails, regime_rows, kd = [], [], [], []
        r02 = self.results.get("R02") or _read(self.local / "reports" / "R02.json")
        r01 = self.results.get("R01") or _read(self.local / "reports" / "R01.json")
        r03 = self.results.get("R03") or _read(self.local / "reports" / "R03.json")
        near = ((r02 or {}).get("full_map") or {}).get("top0.08_eligible_share_within_3px_known")
        regimes = self._regimes()
        edges = self._edge_rasters()
        km = self.pixel_m / 1000
        for u in self.eval_units():
            k = (u.task, u.zone)
            if k not in maps:
                continue
            cfp = sel_p / f"g14_crossfit_{u.task}.csv"
            per = pd.read_csv(cfp) if cfp.exists() else None
            if per is not None and u.zone in set(per.zone):
                r = per[per.zone == u.zone].iloc[0]
                rep, order, fr = str(r.rep), str(r.order), float(r.target_frac)
            else:
                rep, order, fr = "crest_w1", "pixel_exact", 0.08
            s = np.asarray(u.crop(maps[k]), np.float32)
            q, acc = apply_rep(rep, s, u.ev, u.exclude, u.scored_mask(), None if fr < 0 else fr, order)
            row = {**M.decompose_unit(u, q, km), "rep": rep, "order": order, "frac": fr,
                   **{f"acc_{a}": b for a, b in acc.items() if a not in ("rep", "order")}}
            rows.append(row)
            rr = None
            if regimes is not None:
                rr = _regime_components(u, q, u.crop(regimes))
                regime_rows.append(rr)
            truth = u.eval_truth()
            far = (ndimage.distance_transform_edt(~truth) > 300.0 / self.pixel_m) if truth.any() \
                else np.ones(truth.shape, bool)
            eo = M.edge_overlap((q > 0) & far, {kk: u.crop(v) for kk, v in edges.items()}, u.ev) if edges else None
            ctx = self._heldout_context(u)
            kd.append(M.recall_by_known_distance(u, q, u.crop(ctx), km))
            best_rep = ((r03 or {}).get("selected") or {}).get(u.task, {}).get("crossfit_dti")
            rk = None
            rkp = self.local / "probability_diagnostics" / "heldout_ranking.csv"
            if rkp.exists():
                t = pd.read_csv(rkp)
                t = t[(t.task == u.task) & (t.zone == u.zone)]
                rk = t.iloc[0].to_dict() if len(t) else None
            for c in M.classify(row, {"top_share_near_known": near} if near is not None else None,
                                (r01 or {}).get("parity"), rk, best_rep, rr, eo):
                fails.append({"task": u.task, "zone": u.zone, **c})
        dec = pd.DataFrame(rows)
        out = [self._df(dec, "metric_decomposition", "g14_decomposition"),
               self._df(pd.DataFrame(fails), "metric_decomposition", "failure_classes")]
        if regime_rows:
            out.append(self._df(pd.concat(regime_rows, ignore_index=True), "metric_decomposition", "by_regime"))
        if kd:
            kdd = pd.concat(kd, ignore_index=True).assign(model="g14")
            out.append(self._df(kdd, "metric_decomposition", "recall_by_known_distance"))
        fired = pd.DataFrame(fails)
        summ = {"fired": fired[fired.status == "FIRED"].groupby("class").size().to_dict() if len(fired) else {},
                "denominator_share_alpha_fp": float((dec.alpha_fp / (dec.alpha_fp + dec.beta_fn)).mean()) if len(dec) else None,
                "pooled": {"tp": float(dec.tp_w.sum()), "fp": float(dec.fp_w.sum()), "fn": float(dec.fn_w.sum())}
                if len(dec) else {}}
        return out, summ

    # ================================================================== R05 / R12
    def _champions(self) -> dict:
        from .artifacts import best_copy, g13_models
        from .champion import g13_champions

        if self.system.backend != "g13":
            return {}
        sb = best_copy(self.inv, "g13_scoreboard")
        return g13_champions(self.system, g13_models(self.inv), sb, top=3)

    def r05_champion(self):
        from . import champion as CH
        from .artifacts import best_copy, g13_models
        from .representations import best_per_rep, crossfit_reps, sweep_unit

        data, out, summ = self.data, [], {}
        full = self.full_map()
        lf = []
        if full is not None:
            for m, files in g13_models(self.inv).items():
                if "full" in files:
                    lf.append({"other": f"G13_{m}", **CH.label_free(full, CH.read_map(files["full"]), data.valid, data.labels)})
            for art, name in (("p6_w2_full", "P6_W2"),):
                p = best_copy(self.inv, art)
                if p is not None:
                    a = CH.read_map(p)
                    if a.shape[-2:] == data.shape:
                        lf.append({"other": name, **CH.label_free(full, np.asarray(a), data.valid, data.labels)})
            for p in self.inv[self.inv.artifact.isin(["p7_maps", "g07_maps"])].path:
                if "full" in Path(p).name:
                    a = CH.read_map(p)
                    lf.append({"other": Path(p).name, **(CH.label_free(full, np.asarray(a), data.valid, data.labels)
                                                         if a.shape[-2:] == data.shape else {"note": "grid differs"})})
        if lf:
            out.append(self._df(pd.DataFrame(lf), "champion_comparison", "label_free_full_maps"))
        champs = self._champions()
        g14m = self.g14_unit_maps()
        units = self.eval_units()
        pcs, sweeps = [], []
        for name, cm in champs.items():
            us = [u for u in units if (u.task, u.zone) in cm]
            if not us:
                continue
            t = pd.concat([sweep_unit(u, cm[(u.task, u.zone)], ("top", "sparse_soft"), self.cfg.sweep.area_fracs,
                                      ("pixel_exact",)) for u in us], ignore_index=True).assign(model=name)
            sweeps.append(t)
            cf = crossfit_reps(t)
            sel = {task: {"rep": r["summary"]["params_all"]["rep"], "order": "pixel_exact",
                          "frac": float(r["summary"]["params_all"]["target_frac"]), "calibration": "identity",
                          "crossfit_dti": r["summary"]["dti_crossfit"]} for task, r in cf.items()}
            self._merge_selection(f"g13:{name[4:]}", sel)
            summ.setdefault("champions", {})[name] = sel
            for u in us:
                k = (u.task, u.zone)
                if k in g14m:
                    fr = sel.get(u.task, {}).get("frac", 0.02)
                    pcs.append({"champion": name, **CH.pixel_classes(u, np.asarray(u.crop(cm[k]), np.float32),
                                                                     np.asarray(u.crop(g14m[k]), np.float32), fr,
                                                                     self.pixel_m)})
        if sweeps:
            sw = pd.concat(sweeps, ignore_index=True)
            out += [self._df(sw, "champion_comparison", "champion_sweeps"),
                    self._df(best_per_rep(sw.drop(columns="model")), "champion_comparison", "champion_best_per_rep")]
            ao = {}
            for task, t in sw[sw.rep == "top"].groupby("task"):
                from ..g13.evaluation import crossfit

                ao[task] = float(crossfit(t, ["target_frac"])[1]["params_all"]["target_frac"])
            summ["area_optimum_champion"] = ao
        if pcs:
            out.append(self._df(pd.DataFrame(pcs), "complementarity", "pixel_classes"))
        first = next(iter(summ.get("champions", {}).values()), {})
        summ["champion"] = {"disc_dti": first.get("disc_official", {}).get("crossfit_dti"),
                            "geo_dti": first.get("geo", {}).get("crossfit_dti")}
        return out, summ

    def r12_residual(self):
        from . import champion as CH

        champs, g14m = self._champions(), self.g14_unit_maps()
        if not champs or not g14m:
            return [], {"status": "BLOCKED: needs champion and G14 maps on the same units"}
        name, cm = next(iter(champs.items()))
        units = self.eval_units()
        bt = CH.blend_sweep(cm, g14m, units, self.cfg.sweep.g14_weights,
                            fracs=(0.0025, 0.005, 0.01, 0.02, 0.05) if self.cfg.profile == "FAST" else
                            tuple(f for f in self.cfg.sweep.area_fracs if f <= 0.1))
        out = [self._df(bt, "complementarity", "blend_sweep")]
        res = CH.crossfit_blend(bt)
        for task, r in res.items():
            out.append(self._df(r["per_zone"], "complementarity", f"blend_crossfit_{task}"))
        regimes = self._regimes()
        if regimes is not None and self.cfg.profile != "FAST":
            cb = CH.conditional_blend(cm, g14m, units, regimes, self.cfg.sweep.g14_weights, 0.02)
            out.append(self._df(cb, "complementarity", "regime_conditional_blend"))
        summ = {"champion": name, "blend": {t: {"delta_vs_champion": r["delta_vs_champion"],
                                                "champion_crossfit": r["champion_crossfit"],
                                                "blend_crossfit": r["summary"]["dti_crossfit"]} for t, r in res.items()}}
        first = next(iter(summ["blend"].values()), {})
        summ["delta_vs_champion"] = first.get("delta_vs_champion")
        return out, summ

    # ================================================================== R05b atlas
    def r05b_atlas(self):
        from . import atlas as AT
        from .metric_audit import coverage_boundary
        from .representations import apply_rep

        maps, data = self.g14_unit_maps(), self.data
        champs = self._champions()
        cm = next(iter(champs.values()), {})
        regimes = self._regimes()
        bands = {b: data.band_valid(b) for b in ("rtp", "iso_grav_anom", "det_elev", "cond_surf") if data.has(b)}
        edge = coverage_boundary(data.valid, bands)
        sel = _read(self.local / "calibration" / "selected_operating_points.json") or {}
        items = []
        for u in [u for u in self.eval_units() if (u.task, u.zone) in maps][: (2 if self.cfg.profile == "STANDARD" else 99)]:
            k = (u.task, u.zone)
            op = sel.get("g14", {}).get(u.task, {"rep": "crest_w1", "frac": 0.08, "order": "pixel_exact"})
            s = np.asarray(u.crop(maps[k]), np.float32)
            q, _ = apply_rep(op["rep"], s, u.ev, u.exclude, u.scored_mask(), op["frac"] if op["frac"] > 0 else None,
                             op.get("order", "pixel_exact"))
            cq = None
            if k in cm:
                cq, _ = apply_rep("top", np.asarray(u.crop(cm[k]), np.float32), u.ev, u.exclude, u.scored_mask(), 0.02)
            ts = AT.tile_stats(u, q, s, cq, u.crop(regimes) if regimes is not None else None, u.crop(edge),
                               pixel_m=self.pixel_m)
            full_q = np.full(data.shape, np.nan, np.float32)
            full_q[u.sl] = q
            full_c = None
            if cq is not None:
                full_c = np.full(data.shape, np.nan, np.float32)
                full_c[u.sl] = cq
            truth = np.zeros(data.shape, bool)
            truth[u.sl] = u.eval_truth()
            layers = AT.layer_stack(data, u, {"g14_raw": maps[k], "g14_representation": full_q, "champion": full_c},
                                    regimes, self._structural(), self._heldout_context(u), truth)
            items.append((u, layers, ts))
        idx = AT.build_atlas(data, items, self.local / "failure_atlas")
        return [idx], {"index": str(idx), "units": len(items)}

    # ================================================================== R06 counterfactuals
    def r06_counterfactuals(self):
        from .artifacts import best_copy, restore
        from .leakage import counterfactuals

        g = self.g14
        for art in ("g14_checkpoints", "g14_train_manifests"):
            for rel in sorted(set(self.inv[self.inv.artifact == art].rel)):
                src = best_copy(self.inv, art, rel)
                restore(src, g.exp_dir / rel)
        if g.stack is None:
            self.log("[g14.1] R06: channel stack absent locally; rebuilding it with the G14 code (label-free, "
                     "deterministic; the checkpoint schema hash is verified on load)")
            g.validate_data()
        rows, curves = [], []
        units = g.units()
        for u in units[: (1 if self.cfg.profile == "STANDARD" else len(units))]:
            if not (g.exp_dir / "checkpoints" / f"{u.key}_best.pt").exists():
                continue
            df = counterfactuals(g.cfg, g.stack, g.data, u, g.exp_dir, pixel_m=(self.pixel_m, self.pixel_m),
                                 transform=g.data.transform, log=self.log)
            rows.append(df)
            curves.append(df.attrs.get("curves", pd.DataFrame()))
        if not rows:
            raise FileNotFoundError("no G14 unit checkpoint found (checkpoints/<unit>_best.pt)")
        cf = pd.concat(rows, ignore_index=True)
        out = [self._df(cf, "leakage_audit", "counterfactuals")]
        cv = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
        if len(cv):
            out.append(self._df(cv, "leakage_audit", "counterfactual_distance_curves"))
        inner = cf[(cf.region == "inner_val") & (cf.scoring == "all_truth")].set_index(["unit", "intervention"]).dti
        drop = [inner[(u, "original")] - inner[(u, "target_removed")] for u in inner.index.get_level_values(0).unique()
                if (u, "original") in inner and (u, "target_removed") in inner]
        man = _read(g.exp_dir / "manifests" / f"{units[0].key}_train.json") or {}
        summ = {"counterfactual": {"inner_drop_target_removed": float(np.mean(drop)) if drop else None,
                                   "inner_region_dti_original": float(inner.xs("original", level=1).mean())
                                   if "original" in inner.index.get_level_values(1) else None},
                "inner": {"enriched_surrogate": man.get("best_inner_score"),
                          "full_region_dti": float(inner.xs("original", level=1).mean())
                          if "original" in inner.index.get_level_values(1) else None}}
        return out, summ

    # ================================================================== R07
    def r07_discovery_splits(self):
        from . import discovery as D

        sysm, data, v = self.system, self.data, self.cfg.validation
        audit = D.discovery_split_audit(sysm, v.protocol)
        inv = self.inv
        maps = {"geo": int(inv.rel.str.contains("maps/geo_").sum()), "disc_official": int(inv.rel.str.contains(
            "maps/disc_official_").sum())}
        chk = D.check_protocol(v.protocol, v.primary_task, maps, {"final": int(inv.rel.str.contains("final_0").sum())},
                               submission=True)
        draws, leaks = [], []
        for u in D.protocol_units(sysm, v.protocol):
            try:
                cu = D.ConditionalDiscoveryUnit.build(u, v.fault_link_m, v.fault_link_angle_deg, self.pixel_m,
                                                      self.cfg.training.context_fraction)
                for s in range(3):
                    _, _, rec = cu.draw(s)
                    draws.append({"unit": u.key, "n_systems": int(cu.systems.size), **rec})
            except ValueError as e:
                draws.append({"unit": u.key, "error": str(e)})
            region, tr, _ = D.separate_targets(u, u.labels, 300.0 / self.pixel_m, 300.0, self.pixel_m)
            leaks.append(D.target_leak_check(u, region, tr, 300.0 / self.pixel_m))
        out = [self._df(audit, "discovery_splits", "split_audit"), self._df(chk, "discovery_splits", "protocol_check"),
               self._df(pd.DataFrame(draws), "discovery_splits", "conditional_draws"),
               self._df(pd.DataFrame(leaks), "discovery_splits", "separated_target_checks")]
        summ = {"protocol_issues": chk[chk.level.isin(["ERROR", "PROVISIONAL"])].issue.tolist(),
                "units": int(len(audit)), "shortcut_units_in_original_g14": int((audit.status == "SHORTCUT").sum())}
        return out, summ

    # ================================================================== R08-R15 training
    def train_stage(self, sid: str):
        from . import discovery as D
        from .experiments import specs_for
        from .representations import crossfit_reps, sweep_unit
        from .training import fit_corrected, predict_heldout

        g, tc, rs = self.g14, self.cfg.training, self.cfg.resources
        if g.stack is None:
            g.validate_data()
        specs = specs_for(sid, self.cfg.profile, tuple(tc.experiments) if tc.experiments else ())
        if not specs:
            return [], {"status": f"no spec selected for {sid} (training.experiments filter / profile)"}
        root = self.local / "corrected_models"
        units = D.protocol_units(self.system, self.cfg.validation.protocol)
        cards, out = [], []
        for spec in specs:
            maps, inner = {}, []
            for u in units:
                champ = None
                if spec.residual_champion:
                    p = root / "champion_nested" / f"{u.key}.npy"
                    if not p.exists():
                        self.log(f"[g14.1] {spec.name}/{u.key}: no nested champion map ({p}); skipped")
                        continue
                    champ = np.load(p)
                meta = fit_corrected(spec, g.cfg, g.stack, g.data, u, root, tc, pixel_m=self.pixel_m,
                                     device_pref=rs.device, amp=rs.mixed_precision, num_workers=rs.num_workers,
                                     link_m=self.cfg.validation.fault_link_m,
                                     angle_deg=self.cfg.validation.fault_link_angle_deg, champion_rank=champ,
                                     max_minutes=rs.max_minutes_per_unit, log=self.log, resume=self.cfg.resume)
                maps[(u.task, u.zone)] = predict_heldout(spec, g.cfg, g.stack, g.data, u, root, self.pixel_m, self.log)
                best = next((h for h in meta["history"] if h["epoch"] == meta["best_epoch"]), {})
                inner.append({"unit": u.key, **{k: best.get(k) for k in ("inner_full_region_dti", "inner_tiled_dti",
                                                                        "enriched_surrogate", "inner_loss")}})
            if not maps:
                continue
            us = [u for u in self.eval_units() if (u.task, u.zone) in maps]
            t = pd.concat([sweep_unit(u, maps[(u.task, u.zone)], ("top", "sparse_soft"),
                                      (0.0025, 0.005, 0.01, 0.02, 0.05, 0.08), ("pixel_exact",)) for u in us],
                          ignore_index=True)
            out.append(self._df(t.assign(spec=spec.name), "corrected_models", f"sweep_{spec.name}"))
            cf = crossfit_reps(t)
            ii = pd.DataFrame(inner)
            card = {"stage": sid, "spec": spec.name, **{f"{('disc' if k.startswith('disc') else k)}_dti":
                                                        v["summary"]["dti_crossfit"] for k, v in cf.items()},
                    **{f"{('disc' if k.startswith('disc') else k)}_zone_min": v["summary"]["zone_min"] for k, v in cf.items()},
                    **{f"mean_{c}": float(ii[c].astype(float).mean()) for c in ii.columns if c != "unit"}}
            cards.append(card)
            self._merge_selection(f"g14_1:{spec.name}", {task: {"rep": v["summary"]["params_all"]["rep"],
                                                               "order": "pixel_exact", "calibration": "identity",
                                                               "frac": float(v["summary"]["params_all"]["target_frac"]),
                                                               "crossfit_dti": v["summary"]["dti_crossfit"]}
                                                        for task, v in cf.items()})
        sc = pd.DataFrame(cards)
        p = root / "scorecard.csv"
        allc = pd.concat([pd.read_csv(p), sc], ignore_index=True).drop_duplicates("spec", keep="last") if p.exists() else sc
        out.append(self._df(allc, "corrected_models", "scorecard"))
        return out, {"scorecards": cards}

    def train_final(self, spec_name: str, include_test: bool = False) -> Path:
        """Final model of one spec + its full-region map (context = every mapped fault). Mode TRAIN."""
        from ..g14.inference import predict_full
        from . import discovery as D
        from .experiments import REGISTRY
        from .training import fit_corrected, load_trained

        if not self.cfg.allow_training:
            raise PermissionError("train_final needs mode='TRAIN'")
        spec = REGISTRY[spec_name][0]
        g = self.g14
        if g.stack is None:
            g.validate_data()
        u = self.system.final_unit()
        if include_test:                     # official file only: the locked TEST fold is then no longer clean
            u = D.with_context(u, u.permitted_known, self.data.labels)
            u.train = self.data.valid & ~u.inner_val
        ctx = D.corrected_final_context(self.data)
        root = self.local / "corrected_models"
        fit_corrected(spec, g.cfg, g.stack, g.data, D.with_context(u, ctx, u.labels), root, self.cfg.training,
                      pixel_m=self.pixel_m, device_pref=self.cfg.resources.device, log=self.log)
        model, view, icfg = load_trained(spec, g.cfg, g.stack, g.data, D.with_context(u, ctx, u.labels), root,
                                         None if spec.supervision == "geophysics_only" else ctx, pixel_m=self.pixel_m)
        r = predict_full(model, view, icfg, self.data.valid, mc_samples=0, tta=False, log=self.log)
        p = root / "maps" / f"{spec.name}__final_0.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, r["prob"])
        self.sync(["corrected_models"])
        return p

    def r16_ensemble(self):
        sc = self.local / "corrected_models" / "scorecard.csv"
        if not sc.exists():
            return [], {"status": "BLOCKED: no corrected model scorecard (run R09-R15 first)"}
        t = pd.read_csv(sc)
        key = "disc_dti" if "disc_dti" in t else "geo_dti"
        best = t.sort_values(key, ascending=False).iloc[0].spec
        from . import champion as CH

        champs = self._champions()
        if not champs:
            return [], {"status": "BLOCKED: no champion maps on the same units", "best_spec": best}
        maps = {}
        for f in (self.local / "corrected_models" / "maps").glob(f"{best}__*.npy"):
            stem = f.stem.split("__")[-1]
            if stem.startswith("final"):
                continue
            task, zone = stem.rsplit("_", 1)
            maps[(task, int(zone))] = np.load(f)
        name, cm = next(iter(champs.items()))
        bt = CH.blend_sweep(cm, maps, self.eval_units(), self.cfg.sweep.g14_weights)
        res = CH.crossfit_blend(bt)
        out = [self._df(bt, "complementarity", f"ensemble_sweep_{best}")]
        return out, {"best_spec": best, "champion": name,
                     "ensemble": {t: {"delta_vs_champion": r["delta_vs_champion"], "selected": r["selected_weights"]}
                                  for t, r in res.items()}}

    # ================================================================== R17 submission
    def submission(self, model: str = "g14", **kw) -> dict:
        from .submission import generate_submission

        return generate_submission(self.cfg, self.data, self.inv, model=model, log=self.log, **kw)

    # ================================================================== report / figures
    def collect(self) -> dict:
        R = {}
        for sid in ("R00", "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R12", "R16"):
            s = self.results.get(sid) or _read(self.local / "reports" / f"{sid}.json")
            if s:
                R[sid] = s
        flat = {}
        if "R01" in R:
            flat["parity"] = R["R01"].get("parity")
        if "R02" in R:
            flat["full_map"] = R["R02"].get("full_map")
            flat["ranking"] = R["R02"].get("ranking")
            flat["leakage"] = R["R02"].get("leakage")
        if "R03" in R:
            flat["sweep"] = R["R03"].get("sweep")
            flat["area_optimum"] = R["R03"].get("area_optimum")
        if "R05" in R:
            flat["champion"] = R["R05"].get("champion")
            if R["R05"].get("area_optimum_champion"):
                flat["area_optimum"] = R["R05"]["area_optimum_champion"]
        if "R06" in R:
            flat["counterfactual"] = R["R06"].get("counterfactual")
            flat["inner"] = R["R06"].get("inner")
        if "R12" in R:
            flat["blend"] = {"delta_vs_champion": R["R12"].get("delta_vs_champion")}
        sc = self.local / "corrected_models" / "scorecard.csv"
        if sc.exists():
            t = pd.read_csv(sc)
            flat["models"] = {r.spec: {k: r[k] for k in t.columns if k != "spec"} for _, r in t.iterrows()}
            loc = t[t.spec == "M1_geophysics_only"]
            ctx = t[t.spec.str.startswith(("N1_", "N2_", "N3_", "N4_"))]
            if len(loc) and len(ctx) and "disc_dti" in t:
                flat["architecture"] = {"best_context_minus_local": float(ctx.disc_dti.max() - loc.disc_dti.iloc[0]),
                                        "ci_low": None}
            from .metric_audit import ranking_consistency

            cols = [c for c in ("mean_inner_full_region_dti", "mean_enriched_surrogate") if c in t]
            if cols and "disc_dti" in t:
                rc = ranking_consistency(t, cols, ["disc_dti"])
                self._df(rc, "reports", "ranking_consistency")
                r = rc[rc.inner == cols[0]]
                flat["ranking_consistency"] = {"spearman_inner_vs_disc": float(r.spearman.iloc[0]) if len(r) else None}
        return {**flat, "stages": R}

    def report(self) -> Path:
        from .decision import evaluate, write_report

        R = self.collect()
        table = evaluate(R)
        p = write_report(table, R, self.local / "reports" / "DECISION_REPORT.md")
        self.figures()
        self.sync(["reports"])
        return p

    def figures(self) -> list[Path]:
        """Every figure whose source table exists (silently skips the others; listed in figures_index.csv)."""
        from . import figures as F

        root, fig = self.local, self.local / "reports" / "figures"
        made, idx = [], []

        def rd(sub, name):
            p = root / sub / f"{name}.csv"
            return pd.read_csv(p) if p.exists() else None

        jobs = [(4, rd("probability_diagnostics", "full_histogram"), F.f04_probability_histograms),
                (5, rd("representation_sweep", "g14_sweep"), F.f05_area_vs_dti),
                (6, rd("representation_sweep", "g14_sweep"), F.f06_tp_fp_fn),
                (7, rd("representation_sweep", "g14_best_per_representation"), F.f07_representations),
                (8, rd("leakage_audit", "counterfactual_distance_curves"), F.f08_context_dependence),
                (9, rd("metric_decomposition", "recall_by_known_distance"),
                 lambda d, o: F.f09_perf_vs_known_distance(d.groupby(["model", "dist_bin_km"])[["tp_w", "n_truth"]]
                                                           .sum().assign(recall_w=lambda x: x.tp_w / x.n_truth.clip(lower=1))
                                                           .reset_index(), o)),
                (17, rd("complementarity", "pixel_classes"), F.f17_unique_recovery),
                (18, rd("complementarity", "blend_sweep"), F.f18_ensemble_weight),
                (19, rd("metric_decomposition", "g14_decomposition"), F.f19_fragmentation)]
        sc = rd("corrected_models", "scorecard")
        if sc is not None and "disc_dti" in sc:
            jobs += [(10, sc[sc.spec.str.startswith("K_")], F.f10_feature_ablation),
                     (24, sc[sc.spec.str.startswith(("M1_", "N"))], F.f24_architecture)]
        rc = rd("reports", "ranking_consistency")
        if rc is not None:
            jobs.append((3, rc, F.f03_rank_correlation))
        for n, df, fn in jobs:
            if df is None or not len(df):
                idx.append({"figure": n, "status": "no source table"})
                continue
            try:
                made.append(fn(df, fig / f"fig{n:02d}"))
                idx.append({"figure": n, "status": "ok", "path": str(made[-1])})
            except Exception as e:  # noqa: BLE001
                idx.append({"figure": n, "status": f"error: {e}"})
        pd.DataFrame(idx).to_csv(fig.parent / "figures_index.csv", index=False)
        return made

    # ------------------------------------------------------------------ small helpers
    def _regimes(self):
        try:
            return self.g14._regimes()
        except Exception:  # noqa: BLE001
            return None

    def _structural(self):
        st = self.g14.stack
        if st is None:
            return None
        names = [c.name for c in st.channels]
        for k in ("g14:msdi", "g14:tli", "g14:gsdi"):
            if k in names:
                return np.asarray(st.level(1)[names.index(k)], np.float32)
        return None

    def _edge_rasters(self) -> dict:
        from .metric_audit import coverage_boundary

        d = self.data
        bands = {b: d.band_valid(b) for b in ("rtp", "iso_grav_anom", "det_elev", "cond_surf") if d.has(b)}
        out = {"coverage_boundary": coverage_boundary(d.valid, bands)}
        if d.has("det_elev_slope"):
            out["terrain_slope"] = d.band("det_elev_slope")
        return out

    def _heldout_context(self, u) -> np.ndarray:
        """Permitted known map of the G14 training unit that predicted evaluation unit ``u`` (same task / zone)."""
        s = self.system
        if u.task == "geo":
            return s.geo_unit(u.zone).permitted_known
        if u.task.startswith("disc"):
            return s.disc_unit(u.zone).permitted_known
        return np.zeros(self.data.shape, bool)


# =============================================================================== module helpers
def _read(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.DataFrame,)):
        return o.to_dict("records")
    return str(o)


def _submission_operating_point(tif: Path) -> dict | None:
    """(rep, frac) recorded by the G14 / G13 exporters next to a submitted GeoTIFF."""
    for j in (tif.with_suffix(".json"), tif.with_name(tif.stem + "_g14.json")):
        if j.exists():
            try:
                r = json.loads(j.read_text())
            except Exception:  # noqa: BLE001
                continue
            s = str(r.get("representation", ""))
            if "|frac=" in s:
                rep, fr = s.split("|frac=")
                return {"rep": rep, "frac": float(fr), "record": str(j)}
    return None


def _regime_components(u, q_crop: np.ndarray, reg_crop: np.ndarray) -> pd.DataFrame:
    from ..metric import dti_components

    rows = []
    for g in np.unique(reg_crop[u.ev]):
        ev = u.ev & (reg_crop == g)
        c = dti_components(np.asarray(q_crop, np.float64), u.truth, eval_mask=ev, exclude_mask=u.exclude, crop=False,
                           exclude_supports_tp=False)
        rows.append({"task": u.task, "zone": u.zone, "regime": int(g), "dti": c.dti, "tp_w": c.tp, "fp_w": c.fp,
                     "fn_w": c.fn, "n_truth": c.n_truth})
    return pd.DataFrame(rows)
