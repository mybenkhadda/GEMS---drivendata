"""G13 orchestration: stages, fingerprints, resume, modes (section 01) and the CLI.

    from gems_bench.g13.engine import Engine
    eng = Engine.from_mode("FAST_BASELINE", drive_root="/content/drive/MyDrive/GEMS_2")
    eng.run()                       # every stage of the mode; verified completed stages are reused
    eng.status()                    # stage table

Each stage function returns (outputs, summary). Its manifest stores a fingerprint of the config keys it depends on,
the fingerprints of its upstream stages and the G13 version; a changed fingerprint recomputes the stage and every
stage after it. Optional research stages fail gracefully (logged, marked ``failed``) without stopping the run.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import G13_VERSION, viz
from . import contracts as C
from .cache import Workspace, atomic_save_npy, atomic_save_table, atomic_write_json, combine, load_table
from .config import MODE_STAGES, STAGE_DEPS, STAGE_KEYS, G13Config
from .report import FigureIndex, finding, findings_table, runtime_table, write_report

CORE = ("inventory", "splits", "features", "baseline", "dti_opt", "submission")
SUBMISSION_FILES = {"A": "submission_g13_baseline", "B": "submission_g13_physics", "C": "submission_g13_champion"}


class Engine:
    def __init__(self, cfg: G13Config, log=print):
        self.cfg = cfg
        self.ws = Workspace(cfg.work_dir, cfg.out_dir or None)
        self.fx = FigureIndex(self.ws)
        self.log = log
        self.rng = np.random.default_rng(cfg.seed)
        self._data = self._sp = self._store = None
        self._fps: dict[str, str] = {}
        self.findings: list = []
        # resume on a fresh runtime: bring back persisted small artifacts and selected predictions
        for sub in (
            "manifests",
            "inventory",
            "splits",
            "tables",
            "regimes",
            "dti_opt",
            "predictions",
            "models",
            "quality",
            "eda",
            "submissions",
        ):
            self.ws.restore_tree(sub)

    @classmethod
    def from_mode(cls, mode: str, log=print, **kw) -> "Engine":
        return cls(G13Config.from_mode(mode, **kw), log)

    # ------------------------------------------------------------------ shared state (lazy)
    @property
    def paths(self) -> dict:
        p = self.ws.root / "inventory" / "paths.json"
        if p.exists():
            return json.loads(p.read_text())
        paths = C.discover_inputs(
            self.cfg.drive_root, {"raw": self.cfg.raw_path, "known": self.cfg.known_path, "template": self.cfg.template_path}
        )
        atomic_write_json(p, paths)
        return paths

    @property
    def data(self) -> C.GeoData:
        if self._data is None:
            self._data = C.load(self.paths, self.ws.root / "cache")
        return self._data

    @property
    def sp(self):
        if self._sp is None:
            from .splits import Splits

            self._sp = Splits.load(self.ws.root / "splits", self.data.valid, self.data.labels)
        return self._sp

    @property
    def store(self):
        if self._store is None:
            from .features import FeatureStore

            self._store = FeatureStore(self.ws.root / "features", self.data)
        return self._store

    # ------------------------------------------------------------------ stage control
    def fingerprint(self, stage: str) -> str:
        if stage not in self._fps:
            up = [self.fingerprint(d) for d in STAGE_DEPS[stage]]
            self._fps[stage] = combine(
                G13_VERSION,
                stage,
                self.cfg.fingerprint(*STAGE_KEYS[stage]) if STAGE_KEYS[stage] else "",
                up,
                self.cfg.mode if stage in ("submission", "report") else "",
            )
        return self._fps[stage]

    def status(self) -> pd.DataFrame:
        rows = []
        for s in self.cfg.active_stages():
            ok, why = self.ws.stage_complete(s, self.fingerprint(s))
            m = self.ws.read_manifest(s) or {}
            rows.append(
                {
                    "stage": s,
                    "complete": ok,
                    "detail": why,
                    "seconds": m.get("seconds"),
                    "summary": json.dumps(m.get("summary", {}), default=str)[:160],
                }
            )
        return pd.DataFrame(rows)

    def run(self, stages=None) -> pd.DataFrame:
        todo = tuple(stages) if stages else self.cfg.active_stages()
        for s in todo:
            self.run_stage(s)
        return self.status()

    def run_stage(self, stage: str) -> dict | None:
        fp = self.fingerprint(stage)
        ok, why = self.ws.stage_complete(stage, fp)
        if ok and stage not in self.cfg.force_stages:
            self.log(f"[{stage}] reused ({why})")
            return self.ws.read_manifest(stage)
        self.log(f"[{stage}] running ({why})")
        t0 = time.perf_counter()
        try:
            outs, summary = getattr(self, f"stage_{stage}")()
        except Exception as e:
            self.ws.log(stage, "failed", error=repr(e), tb=traceback.format_exc()[-4000:])
            self.ws.write_manifest(stage, fp, [], {"error": repr(e)}, status="failed", seconds=time.perf_counter() - t0)
            if stage in CORE:
                raise
            self.log(f"[{stage}] FAILED (optional stage, continuing): {e!r}")
            return None
        rec = self.ws.write_manifest(stage, fp, outs, summary, seconds=time.perf_counter() - t0)
        if stage in ("baseline", "ablation", "pu_learning", "deep", "fusion", "dti_opt", "submission"):
            self._persist_top_predictions()
        n = self.ws.persist()
        self.log(f"[{stage}] complete in {rec['seconds']:.1f}s; {len(outs)} outputs; {n} files persisted")
        return rec

    # ================================================================== stages
    def stage_inventory(self):
        d = self.data
        res = C.run_inventory(d, self.paths, self.ws, self.cfg.dem_dir)
        hw = C.hardware_report()
        outs = res["outputs"] + [atomic_write_json(self.ws.path("inventory", "hardware.json"), hw)]
        if self.cfg.figures:
            outs += [
                viz.coverage_maps(self.fx, d, res["band_inventory"]),
                viz.coverage_matrix(self.fx, res["coverage"]),
                viz.raw_atlas(self.fx, d),
                viz.valid_by_modality(self.fx, res["band_inventory"]),
            ]
        return [o for o in outs if o], {**res["summary"], "hardware": hw}

    def stage_splits(self):
        from .splits import assert_valid, audit, build_splits

        sp = build_splits(self.data, self.cfg)
        assert_valid(sp)
        outs = sp.save(self.ws.root / "splits")
        self._sp = sp
        a = audit(sp)
        outs.append(atomic_save_table(self.ws.path("splits", "audit.csv"), a))
        if self.cfg.figures:
            outs.append(viz.partitions(self.fx, self.data, sp))
        return outs, {
            "test_fold": sp.test_fold,
            "cv_folds": sp.cv_folds,
            "n_splits": sp.n_splits,
            "test_fault_px": int((sp.test() & sp.labels).sum()),
        }

    def stage_features(self):
        from .features import build_store

        store = build_store(self.data, self.cfg, self.ws.root / "features", log=self.log)
        self._store = store
        outs = [store.reg_path] if store.reg_path.exists() else []
        atlas = [
            "magnetic:rtp:thd",
            "magnetic:rtp:tilt",
            "magnetic:rtp:phase_congruency",
            "gravity:iso_grav_anom:thd",
            "terrain:det_elev:slope_break_s400",
            "structural:rtp:orient_persistence",
            "cross_modal:all:agreement_frac",
            "directional:det_elev:gabor800_aniso",
        ]
        if self.cfg.figures and len(store.registry):
            outs.append(viz.feature_atlas(self.fx, self.data, store, atlas))
        fams = store.registry.family.value_counts().to_dict() if len(store.registry) else {}
        return [o for o in outs if o], {
            "n_features": len(store.registry),
            "by_family": fams,
            "storage_mb": float(store.registry.bytes.sum() / 2**20) if len(store.registry) else 0,
        }

    def stage_quality(self):
        from . import quality as Q

        st, d, sp = self.store, self.data, self.sp
        names = st.names()
        q = Q.quality_table(st, d, sp, self.rng, names=names)
        eng = [n for n in names if not n.startswith("raw:")]
        q = q.merge(
            Q.smoothing_sensitivity(st, eng, d, self.rng).rename_axis("feature").reset_index(), on="feature", how="left"
        )
        clusters = Q.redundancy_clusters(st, names, d, self.rng)
        util = None
        ds = self.ws.root / "eda" / "discriminative_summary.csv"
        if ds.exists():
            util = pd.read_csv(ds).set_index("feature").cd_random.abs()
        q = Q.apply_rejection(q, clusters, util)
        reg = (
            st.registry.merge(q, left_on="name", right_on="feature", how="left", suffixes=("", "_q"))
            if len(st.registry)
            else q
        )
        outs = [
            atomic_save_table(self.ws.path("quality", "feature_quality.csv"), q),
            atomic_save_table(self.ws.path("quality", "geological_feature_registry.csv"), reg),
        ]
        return outs, {
            "n": len(q),
            "kept": int(q.kept.sum()),
            "rejected": q.loc[~q.kept, "reject_reason"].value_counts().to_dict(),
        }

    def stage_eda(self):
        from . import eda as E

        st, d, sp = self.store, self.data, self.sp
        pops = E.populations(d, st, sp, self.rng)
        names = st.names()
        ct = E.contrast_table(st, names, pops, sp, self.rng, n_boot=60)
        summ = E.discriminative_summary(ct)
        bands = [b for b in ("rtp", "iso_grav_anom", "det_elev", "cond_surf", "depth_to_base_surf") if d.has(b)]
        ss = E.scale_space_auc(d, sp, bands, (100, 200, 400, 800, 1600, 3200, 6400), self.rng)
        vg = {b: E.variogram(d.band(b), d.valid & np.isfinite(d.band(b)), self.rng) for b in bands}
        cl = pd.concat([E.correlation_length(v).assign(band=b) for b, v in vg.items()]) if vg else pd.DataFrame()
        spec = {b: E.radial_power_spectrum(d.band(b), d.valid & np.isfinite(d.band(b))) for b in bands}
        prof = E.cross_fault_profiles(d, sp, bands, self.rng)
        fam, famsel = E.profile_families(prof["profiles"], self.rng)
        if len(prof["meta"]):
            prof["meta"]["family"] = fam
        rose = E.orientation_rose(d, sp, bands)
        ids = np.concatenate([v for v in pops.values()])
        corr = E.modality_correlation(st, [f"raw:{b}" for b in d.names], ids[:20000])
        outs = [
            atomic_save_table(self.ws.path("eda", "contrast_table.csv"), ct),
            atomic_save_table(self.ws.path("eda", "discriminative_summary.csv"), summ),
            atomic_save_table(self.ws.path("eda", "scale_space_auc.csv"), ss),
            atomic_save_table(
                self.ws.path("eda", "variograms.csv"),
                pd.concat([v.assign(band=b) for b, v in vg.items()]) if vg else pd.DataFrame(),
            ),
            atomic_save_table(self.ws.path("eda", "correlation_lengths.csv"), cl),
            atomic_save_table(self.ws.path("eda", "profile_stations.csv"), prof["meta"]),
            atomic_save_table(self.ws.path("eda", "profile_family_selection.csv"), famsel),
            atomic_save_npy(self.ws.path("eda", "profiles.npy"), prof["profiles"]),
            atomic_save_table(self.ws.path("eda", "orientation_rose.csv"), rose),
            atomic_save_table(self.ws.path("eda", "raw_band_spearman.csv"), corr.reset_index()),
        ]
        if self.cfg.figures:
            outs += [
                viz.scale_space(self.fx, ss),
                viz.variograms(self.fx, vg),
                viz.spectra(self.fx, spec),
                viz.contrasts(self.fx, ct, summ),
                viz.profiles(self.fx, prof, fam),
                viz.roses(self.fx, rose),
            ]
        n_disc = int((summ.verdict == "discriminative").sum()) if len(summ) else 0
        return [o for o in outs if o], {
            "populations": {k: len(v) for k, v in pops.items()},
            "discriminative_features": n_disc,
            "top_features": summ.head(10).feature.tolist() if len(summ) else [],
        }

    def stage_regimes(self):
        from . import regimes as Rg
        from .evaluation import per_truth_support

        st, d, sp = self.store, self.data, self.sp
        model = Rg.fit_regimes(st, d, sp, self.rng)
        reg = Rg.assign_regimes(model, st, d)
        atlas = Rg.regime_atlas(reg, d, sp, st)
        ev = Rg.evidence_ranks(st, d)
        arch = Rg.archetypes(ev, d.labels & sp.cv_region())
        support = {}
        board = self._scoreboard()
        if len(board):
            best = board.sort_values("disc_proxy_dti", ascending=False).iloc[0]
            maps = self._maps(best.model)
            if maps.get("oof") is not None:
                for u in sp.units(("geo",)):
                    support[f"geo{u.zone}"] = per_truth_support(maps["oof"], u, float(best.get("geo_frac", 0.02)))
        at = Rg.archetype_table(arch, d.labels & sp.cv_region(), support or None)
        outs = [
            atomic_save_npy(self.ws.path("regimes", "regimes.npy"), reg),
            atomic_save_npy(self.ws.path("regimes", "archetypes.npy"), arch),
            atomic_save_table(self.ws.path("regimes", "regime_selection.csv"), model["selection"]),
            atomic_save_table(self.ws.path("regimes", "regime_atlas.csv"), atlas),
            atomic_save_table(self.ws.path("regimes", "archetypes.csv"), at),
            atomic_write_json(
                self.ws.path("regimes", "regime_model.json"),
                {
                    "k": model["k"],
                    "features": model["names"],
                    "comparison": model["comparison"],
                    "explained_variance": model["explained_variance"],
                },
            ),
        ]
        if self.cfg.figures:
            outs.append(viz.regime_map(self.fx, d, reg, model["selection"], model["k"]))
        return outs, {"k": model["k"], **model["comparison"]}

    # ------------------------------------------------------------------ modelling
    def _scoreboard(self) -> pd.DataFrame:
        p = self.ws.root / "tables" / "scoreboard.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _maps(self, name: str) -> dict:
        from .models import load_maps

        return load_maps(self.ws.root / "predictions" / name, self.sp)

    def _hard(self):
        from .models import hard_negative_categories

        return hard_negative_categories(self.store, self.data)

    def _score_experiment(self, name: str, maps: dict, family: str, kind: str, strategy: str, n_feat: int) -> dict:
        from .evaluation import block_components, bootstrap_dti, model_scorecard

        row, tabs = model_scorecard(name, maps["oof"], maps["disc"], self.sp, self.cfg.area_fracs)
        row.update(family=family, kind=kind, strategy=strategy, n_features=n_feat)
        # spatial-block bootstrap of the disc proxy at the cross-fitted area per split
        if "disc_choice" in tabs:
            blocks = []
            for u in self.sp.units(("disc_official",)):
                if u.zone in maps["disc"]:
                    ch = tabs["disc_choice"]
                    fr = float(ch.loc[ch.zone == u.zone, "target_frac"].iloc[0])
                    blocks.append(block_components(maps["disc"][u.zone], u, fr))
                    u.release()
            if blocks:
                bl = pd.concat(blocks)
                atomic_save_table(self.ws.path("tables", "blocks", f"{name}__disc.csv"), bl)
                b = bootstrap_dti(bl, 300, self.cfg.seed)
                row.update(disc_proxy_dti_ci_lo=b["ci_lo"], disc_proxy_dti_ci_hi=b["ci_hi"])
        if maps.get("full") is not None:
            tu = self.sp.test_unit()
            row["test_reserved"] = True  # TEST is scored once, in the dti_opt stage, for the champion
            del tu
        for k, t in tabs.items():
            atomic_save_table(self.ws.path("tables", "experiments", f"{name}__{k}.csv"), t)
        stamp = json.loads((self.ws.root / "predictions" / name / "experiment.json").read_text())
        row.update({k: v for k, v in stamp.get("timings", {}).items() if k.endswith("predict_s")})
        board = self._scoreboard()
        board = pd.concat([board[board.model != name] if len(board) else board, pd.DataFrame([row])], ignore_index=True)
        atomic_save_table(self.ws.path("tables", "scoreboard.csv"), board)
        return row

    def _experiment(self, name, features, kind="xgboost", strategy="standard", disc=True, full=True, family=""):
        from .models import Experiment, run_experiment

        exp = Experiment(name, list(features), kind, strategy, disc, full)
        hard = self._hard() if strategy == "hard_negative" else None
        maps = run_experiment(exp, self.store, self.data, self.sp, self.cfg, self.ws, hard=hard, log=self.log)
        return self._score_experiment(name, maps, family, kind, strategy, len(features))

    def _quick_submission(self, model: str, file_stem: str, label: str) -> dict:
        """Milestone export: the model's full map at its cross-fitted disc (else geo) area budget."""
        from .dti_opt import render

        board = self._scoreboard().set_index("model")
        r = board.loc[model]
        frac = float(r.disc_frac) if "disc_frac" in r and np.isfinite(r.disc_frac) else float(r.geo_frac)
        full = np.asarray(self._maps(model)["full"], np.float32)
        spec = {"rep": {"select": "top"}, "frac": frac, "ring_px": 0}
        q = render(spec, full, self.data.valid, self.data.labels, eligible=self.data.valid)
        from .submission import export_candidate

        rec = export_candidate(
            file_stem,
            q,
            self.data,
            self.ws,
            {
                "model": model,
                "representation": f"top|frac={frac}",
                "geo_proxy_dti": r.get("geo_proxy_dti"),
                "disc_proxy_dti": r.get("disc_proxy_dti"),
                "milestone": label,
            },
            self.cfg.submission_known_policy,
        )
        self.log(
            f"[milestone {label}] {rec['validation']['path']} valid={rec['validation']['valid']} "
            f"checks_failed={rec['validation']['checks_failed']}"
        )
        return rec

    def stage_baseline(self):
        st = self.store
        raw = st.names(families=[], include_raw=True)
        rows = []
        for kind in self.cfg.tabular_models:
            rows.append(self._experiment(f"S0_raw_{kind}", raw, kind, family="A0_raw"))
        best_raw = max(rows, key=lambda r: r.get("disc_proxy_dti", r.get("geo_proxy_dti", 0)))
        outs = []
        s0 = self._quick_submission(best_raw["model"], SUBMISSION_FILES["A"], "S0")
        outs.append(Path(s0["validation"]["path"]))
        phys = [n for n in st.names() if not n.startswith("raw:") and st.family_of(n) != "validity"]
        if phys:
            r = self._experiment("S1_physics_all_xgboost", st.names(), "xgboost", family="A9_all")
            rows.append(r)
            s1 = self._quick_submission(r["model"], SUBMISSION_FILES["B"], "S1")
            outs.append(Path(s1["validation"]["path"]))
        # snapshot (the live scoreboard keeps growing in later stages and must not invalidate this manifest)
        outs.append(atomic_save_table(self.ws.path("tables", "scoreboard_after_baseline.csv"), self._scoreboard()))
        return outs, {
            "models": [r["model"] for r in rows],
            "disc_proxy": {r["model"]: r.get("disc_proxy_dti") for r in rows},
            "geo_proxy": {r["model"]: r.get("geo_proxy_dti") for r in rows},
        }

    def stage_ablation(self):
        from . import quality as Q
        from .evaluation import holm, paired_bootstrap
        from .features import family_feature_sets

        st = self.store
        sets = family_feature_sets(st)
        rows = []
        for name, feats in sets.items():
            if name != "A0_raw" and len(feats) == len(sets["A0_raw"]):
                continue  # family not built
            rows.append(
                self._experiment(f"{name}_xgboost", feats, "xgboost", disc=True, full=name in ("A9_all",), family=name)
            )
        # A10 compact selection from A9 importances (fold models) + quality rejection
        imp_p = self.ws.root / "predictions" / "A9_all_xgboost" / "importance.csv"
        qp = self.ws.root / "quality" / "feature_quality.csv"
        if imp_p.exists():
            q = (
                pd.read_csv(qp)
                if qp.exists()
                else pd.DataFrame({"feature": st.names(), "family": [st.family_of(n) for n in st.names()], "kept": True})
            )
            a10 = Q.compact_selection(pd.read_csv(imp_p), q)
            rows.append(self._experiment("A10_compact_xgboost", a10, "xgboost", full=True, family="A10_compact"))
            atomic_write_json(self.ws.path("ablation", "A10_features.json"), a10)
        board = self._scoreboard()
        abl = board[board.model.str.match(r"^A\d")].assign(experiment=lambda b: b.family)
        comp = Q.complementary_families(abl, "disc_proxy_dti")
        if comp:
            feats = sorted(set(sum((sets[c] for c in comp if c in sets), [])))
            rows.append(
                self._experiment("A11_complementary_xgboost", feats, "xgboost", full=True, family="A11_complementary")
            )
        # targeted interactions
        pairs = [
            ("A1_magnetic", "A2_gravity"),
            ("A1_magnetic", "A4_terrain"),
            ("A2_gravity", "A5_depth"),
            ("A3_conductivity", "A4_terrain"),
            ("A6_structural", "A8_cross_modal"),
            ("A6_structural", "A7_directional"),
        ]
        inter = []
        for a, b in pairs:
            if a not in sets or b not in sets or len(sets[a]) == len(sets["A0_raw"]) or len(sets[b]) == len(sets["A0_raw"]):
                continue
            nm = f"I_{a.split('_')[1]}_x_{b.split('_')[1]}_xgboost"
            r = self._experiment(nm, sorted(set(sets[a]) | set(sets[b])), "xgboost", full=False, family=f"{a}+{b}")
            bd = self._scoreboard().set_index("model")
            val = lambda m: float(bd.loc[m, "disc_proxy_dti"]) if m in bd.index else np.nan  # noqa: E731
            inter.append(
                {
                    "pair": f"{a} x {b}",
                    "dti_pair": r.get("disc_proxy_dti"),
                    "dti_a": val(f"{a}_xgboost"),
                    "dti_b": val(f"{b}_xgboost"),
                    "dti_raw": val("A0_raw_xgboost"),
                }
            )
        it = pd.DataFrame(inter)
        if len(it):
            it["interaction"] = it.dti_pair - it.dti_a - it.dti_b + it.dti_raw
        # tier-1 model benchmark: LightGBM on the best feature set
        if "lightgbm" in self.cfg.tabular_models and "A9_all" in sets:
            rows.append(self._experiment("A9_all_lightgbm", sets["A9_all"], "lightgbm", full=True, family="A9_all"))
        # paired spatial bootstrap vs raw-only on identical disc blocks; Holm-adjusted
        base = self.ws.root / "tables" / "blocks" / "A0_raw_xgboost__disc.csv"
        pairs_out, pv = [], {}
        if base.exists():
            ba = pd.read_csv(base)
            for r in rows:
                p = self.ws.root / "tables" / "blocks" / f"{r['model']}__disc.csv"
                if p.exists() and r["model"] != "A0_raw_xgboost":
                    res = paired_bootstrap(ba, pd.read_csv(p), 300, self.cfg.seed)
                    pairs_out.append({"model": r["model"], **res})
                    pv[r["model"]] = res["p_le_0"]
        pt = pd.DataFrame(pairs_out)
        if len(pt):
            adj = holm(pv)
            pt["p_holm"] = pt.model.map(adj)
        outs = [
            atomic_save_table(self.ws.path("ablation", "interactions.csv"), it),
            atomic_save_table(self.ws.path("ablation", "paired_vs_raw.csv"), pt),
            atomic_save_table(
                self.ws.path("ablation", "ablation_rankings.csv"),
                self._scoreboard().sort_values("disc_proxy_dti", ascending=False),
            ),
        ]
        if self.cfg.figures:
            outs.append(viz.ablation_bars(self.fx, self._scoreboard(), "disc_proxy_dti"))
            outs.append(viz.ablation_bars(self.fx, self._scoreboard(), "geo_proxy_dti"))
        for r in pairs_out:
            self.findings.append(
                finding(
                    "21",
                    f"{r['model']} vs raw-only (disc proxy)",
                    "paired block bootstrap",
                    r["delta"],
                    r["ci_lo"],
                    r["ci_hi"],
                    int(r["ci_lo"] > 0),
                    1,
                    "confirm on TEST only for the selected champion",
                )
            )
        return [o for o in outs if o], {"experiments": [r["model"] for r in rows], "complementary": comp}

    def _best_features(self) -> tuple[str, list]:
        board = self._scoreboard()
        cand = board[board.kind == "xgboost"].sort_values("disc_proxy_dti", ascending=False)
        for m in cand.model:
            p = self.ws.root / "predictions" / m / "experiment.json"
            if p.exists():
                return m, json.loads(p.read_text())["features"]
        return "S0_raw_xgboost", self.store.names(families=[])

    def stage_pu_learning(self):
        from .evaluation import paired_bootstrap

        base_model, feats = self._best_features()
        rows = []
        for s in self.cfg.pu_strategies:
            nm = f"PU_{s}_xgboost"
            rows.append(self._experiment(nm, feats, "xgboost", s, full=s != "standard", family=f"PU:{base_model}"))
        out = []
        std = self.ws.root / "tables" / "blocks" / "PU_standard_xgboost__disc.csv"
        if std.exists():
            for r in rows:
                p = self.ws.root / "tables" / "blocks" / f"{r['model']}__disc.csv"
                if p.exists() and r["model"] != "PU_standard_xgboost":
                    res = paired_bootstrap(pd.read_csv(std), pd.read_csv(p), 300, self.cfg.seed)
                    out.append({"strategy": r["strategy"], **res})
                    self.findings.append(
                        finding(
                            "23",
                            f"{r['strategy']} vs standard sampling",
                            "paired block bootstrap",
                            res["delta"],
                            res["ci_lo"],
                            res["ci_hi"],
                            int(res["ci_lo"] > 0),
                            1,
                        )
                    )
        t = pd.DataFrame(out)
        return [atomic_save_table(self.ws.path("ablation", "pu_vs_standard.csv"), t)], {
            "base_features_from": base_model,
            "strategies": list(self.cfg.pu_strategies),
        }

    def stage_deep(self):
        from . import deep as D

        if not self.cfg.run_deep:
            return [], {"skipped": "run_deep=False"}
        if not D.available():
            return [], {"skipped": "torch not installed"}
        _, feats = self._best_features()
        feats = feats[:48]
        stack = np.stack([self.store.get(n) for n in feats])
        sp, d = self.sp, self.data
        outs = []
        for multitask in (True, False):
            name = f"DL_unet_{'multitask' if multitask else 'single'}"
            root = self.ws.root / "predictions" / name
            root.mkdir(parents=True, exist_ok=True)
            oof = np.full(d.shape, np.nan, np.float32)
            for k in sp.cv_folds:
                X, _ = D.standardise(stack.copy(), sp.train(k))
                net = D.train_unit(X, sp.labels, sp.train(k), self.cfg, self.cfg.seed + k, multitask=multitask, log=self.log)
                pk = D.predict(net, X, sp.val(k) & d.valid, self.cfg.deep_tile)
                oof[np.isfinite(pk)] = pk[np.isfinite(pk)]
            atomic_save_npy(root / "oof.npy", oof)
            for dd in range(sp.n_splits):
                X, _ = D.standardise(stack.copy(), sp.disc_train(dd) & ~sp.ignore(dd))
                net = D.train_unit(
                    X,
                    sp.disc_labels(dd),
                    sp.disc_train(dd) & ~sp.ignore(dd),
                    self.cfg,
                    self.cfg.seed + 10 + dd,
                    multitask=multitask,
                    log=self.log,
                )
                atomic_save_npy(root / f"disc{dd}.npy", D.predict(net, X, sp.eval_zone(dd) & d.valid, self.cfg.deep_tile))
            X, _ = D.standardise(stack.copy(), sp.train_final())
            net = D.train_unit(
                X, sp.labels, sp.train_final(), self.cfg, self.cfg.seed + 99, multitask=multitask, log=self.log
            )
            atomic_save_npy(root / "full.npy", D.predict(net, X, d.valid, self.cfg.deep_tile))
            atomic_write_json(
                root / "experiment.json", {"features": feats, "kind": "torch_unet", "strategy": "standard", "timings": {}}
            )
            self._score_experiment(name, self._maps(name), "deep", "torch_unet", "standard", len(feats))
            outs.append(root / "full.npy")
        return outs, {"features": len(feats)}

    def stage_fusion(self):
        from . import fusion as F
        from .evaluation import model_scorecard

        st, d, sp = self.store, self.data, self.sp
        # modality experts (section 27)
        for mod, fams in (("magnetic", ("magnetic",)), ("gravity", ("gravity",)), ("terrain", ("terrain",))):
            raw = [f"raw:{b}" for b in d.names if C.MODALITY.get(b) == mod]
            feats = raw + st.names(families=list(fams), include_raw=False)
            if len(feats) > len(raw) or raw:
                self._experiment(f"X_{mod}_expert_xgboost", feats, "xgboost", full=True, family=f"expert:{mod}")
        board = self._scoreboard()
        cand = board[board.model.map(lambda m: (self.ws.root / "predictions" / m / "full.npy").exists())]
        cand = cand.sort_values("disc_proxy_dti", ascending=False)
        experts = {}
        for m in cand.model:
            if len(experts) >= 3:
                break
            fam = cand.set_index("model").loc[m, "family"]
            if any(cand.set_index("model").loc[e, "family"] == fam for e in experts):
                continue
            experts[m] = self._maps(m)
        if len(experts) < 2:
            return [], {"skipped": "fewer than 2 experts with full maps"}
        ranks = F.rank_maps(experts, sp, d)
        names = list(experts)
        dis = F.disagreement(ranks, "full")
        outs = [atomic_save_npy(self.ws.path("fusion", "disagreement.npy"), dis)]
        dc = F.disagreement_candidates(dis, d)
        outs.append(atomic_save_table(self.ws.path("fusion", "disagreement_candidates.csv"), dc))
        rows = []
        # rank mean
        mean_maps = {
            "oof": F.weighted(ranks, names, tuple([1 / len(names)] * len(names)), "oof"),
            "disc": {
                u: F.weighted(ranks, names, tuple([1 / len(names)] * len(names)), "disc", u)
                for u in range(sp.n_splits)
                if all(ranks[n]["disc"].get(u) is not None for n in names)
            },
            "full": F.weighted(ranks, names, tuple([1 / len(names)] * len(names)), "full"),
        }
        for key, maps in (("F_rank_mean", mean_maps),):
            self._register_fusion(key, maps)
            r, _ = model_scorecard(key, maps["oof"], maps["disc"], sp, self.cfg.area_fracs)
            rows.append(r)
        fw = F.fixed_weight_search(ranks, names, sp, self.cfg.area_fracs)
        if fw:
            for task, res in fw.items():
                atomic_save_table(self.ws.path("fusion", f"fixed_weights_{task}.csv"), res["per_zone"])
                rows.append(
                    {
                        "model": f"F_fixed_weights[{task}]",
                        f"{'disc' if task != 'geo' else 'geo'}_proxy_dti": res["summary"]["dti_crossfit"],
                        "weights_all": res["summary"]["params_all"]["weights"],
                    }
                )
        ctx = F.context_layers(st, d, self._regimes())
        g = F.gated_maps(ranks, names, ctx, sp, d, self.cfg, self.rng)
        self._register_fusion("F_gated_moe", g)
        r, _ = model_scorecard("F_gated_moe", g["oof"], g["disc"], sp, self.cfg.area_fracs)
        rows.append(r)
        t = pd.DataFrame(rows)
        simple_best = t.loc[t.model != "F_gated_moe", "disc_proxy_dti"].max()
        keep_gate = bool(t.loc[t.model == "F_gated_moe", "disc_proxy_dti"].iloc[0] > simple_best)
        t["kept"] = t.model.eq("F_gated_moe") & keep_gate | t.model.ne("F_gated_moe")
        outs.append(atomic_save_table(self.ws.path("fusion", "fusion_scoreboard.csv"), t))
        for m in ("F_rank_mean", "F_gated_moe"):
            if m != "F_gated_moe" or keep_gate:
                self._score_experiment(m, self._maps(m), "fusion", "fusion", "standard", len(names))
        self.findings.append(
            finding(
                "28",
                "gated mixture-of-experts beats the best simple fusion",
                "disc proxy (cross-fitted)",
                float(t.loc[t.model == "F_gated_moe", "disc_proxy_dti"].iloc[0] - simple_best),
                follow_up="keep only if positive",
            )
        )
        return outs, {"experts": names, "gate_kept": keep_gate}

    def _persist_top_predictions(self, n: int = 6) -> int:
        """Mirror the prediction folders of the best models (and every submission candidate) to Drive."""
        board = self._scoreboard()
        if board.empty or self.ws.out is None:
            return 0
        keep = set(board.sort_values("disc_proxy_dti", ascending=False).model.head(n))
        keep |= set(board[board.family.isin(["A0_raw"])].model)
        cc = self.ws.root / "submissions" / "candidate_comparison.csv"
        if cc.exists():
            keep |= set(pd.read_csv(cc).model)
        return self.ws.persist(
            [f"predictions/{m}" for m in sorted(keep) if (self.ws.root / "predictions" / m).exists()], exclude=()
        )

    def _register_fusion(self, name: str, maps: dict):
        root = self.ws.root / "predictions" / name
        root.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(root / "oof.npy", maps["oof"])
        for d, a in maps["disc"].items():
            atomic_save_npy(root / f"disc{d}.npy", a)
        if maps.get("full") is not None:
            atomic_save_npy(root / "full.npy", maps["full"])
        atomic_write_json(
            root / "experiment.json", {"features": [], "kind": "fusion", "strategy": "standard", "timings": {}}
        )

    def _regimes(self):
        p = self.ws.root / "regimes" / "regimes.npy"
        return np.load(p) if p.exists() else None

    def _champion(self) -> str:
        board = self._scoreboard()
        if board.empty:
            raise RuntimeError("no scored model: run the baseline stage")
        metric = "disc_proxy_dti" if self.cfg.primary_task == "disc_official" else "geo_proxy_dti"
        has_full = board.model.map(lambda m: (self.ws.root / "predictions" / m / "full.npy").exists())
        b = board[has_full].sort_values([metric, "geo_proxy_dti"], ascending=False)
        return str(b.model.iloc[0])

    def stage_topology(self):
        from . import topology as T
        from .regimes import evidence_ranks

        d, st, sp = self.data, self.store, self.sp
        champ = self._champion()
        full = np.asarray(self._maps(champ)["full"], np.float32)
        board = self._scoreboard().set_index("model")
        frac = float(board.loc[champ, "disc_frac"]) if np.isfinite(board.loc[champ].get("disc_frac", np.nan)) else 0.02
        ev = {}
        for nm, key in (
            ("cross_modal:all:agreement_frac", "cross_modal_agreement"),
            ("structural:rtp:orient_persistence", "magnetic_persistence"),
            ("validity:all:boundary_dist_km", "dem_reliability_km"),
        ):
            if st.has(nm):
                ev[key] = st.physical(nm)
        for k, v in evidence_ranks(st, d).items():
            ev[f"{k}_edge_rank"] = v
        region = np.full(d.shape, np.nan, np.float32)
        geo_choice = self.ws.root / "tables" / "experiments" / f"{champ}__geo_choice.csv"
        if geo_choice.exists():
            gc = pd.read_csv(geo_choice)
            for z, v in zip(gc.zone, gc.dti_heldout):
                region[sp.fold_map == z] = v
        cands, seg_r = T.candidate_table(full, d, st, frac=frac, evidence=ev, region_confidence=region)
        summ = T.network_summary(cands)
        outs = [
            atomic_save_table(self.ws.path("topology", "candidates.csv"), cands),
            atomic_save_table(self.ws.path("topology", "network_summary.csv"), summ),
            atomic_save_npy(self.ws.path("topology", "segments.npy"), seg_r),
            atomic_write_json(self.ws.path("topology", "candidates.geojson"), _geojson(cands)),
        ]
        if self.cfg.figures:
            outs.append(viz.candidate_atlas(self.fx, d, seg_r, cands))
        return [o for o in outs if o], {
            "champion": champ,
            "frac": frac,
            "n_segments": len(cands),
            "by_class": cands["class"].value_counts().to_dict() if len(cands) else {},
        }

    def stage_dti_opt(self):
        from . import dti_opt as O

        sp = self.sp
        syn = O.synthetic_sensitivity()
        champ = self._champion()
        maps = self._maps(champ)
        units = []
        score_by_unit = {}
        for u in sp.units(("geo", "disc_official")):
            if u.task == "geo" and maps["oof"] is not None:
                score_by_unit[u.key] = maps["oof"]
                units.append(u)
            elif u.task == "disc_official" and u.zone in maps["disc"]:
                score_by_unit[u.key] = maps["disc"][u.zone]
                units.append(u)
        reg = self._regimes()
        res = O.tournament(score_by_unit, units, self.cfg, reg, log=self.log)
        task = self.cfg.primary_task if self.cfg.primary_task in res["by_task"] else "geo"
        key = res["by_task"][task]["summary"]["params_all"]["key"]
        spec = O.spec_by_key(self.cfg, key)
        mc = O.mass_curves(score_by_unit, units)
        frac_for_wd = spec["frac"] or 0.02
        wd = O.width_displacement_real(score_by_unit, units, frac_for_wd)
        fam = O.family_summary(res["table"])
        # ---- once-only TEST evaluation of the selected (model, representation)
        tu = sp.test_unit()
        full = np.asarray(maps["full"], np.float32)
        q = O.render(spec, tu.crop(full), tu.ev, None, tu.crop(reg) if reg is not None else None, tu.scored_mask())
        tc = tu.components(q)
        test = {
            "model": champ,
            "key": key,
            "test_proxy_dti": tc.dti,
            "tp_w": tc.tp,
            "fp_w": tc.fp,
            "fn_w": tc.fn,
            "note": "geographic TEST fold, all mapped faults as truth (rediscovery proxy); evaluated once",
        }
        sel = {
            "champion_model": champ,
            "task": task,
            "key": key,
            "spec": {k: v for k, v in spec.items() if k != "rep"},
            "rep": spec["rep"],
            "crossfit": {t: v["summary"] for t, v in res["by_task"].items()},
            "test": test,
        }
        outs = [
            atomic_save_table(self.ws.path("dti_opt", "synthetic_sensitivity.csv"), syn),
            atomic_save_table(self.ws.path("dti_opt", "tournament_table.csv"), res["table"]),
            atomic_save_table(self.ws.path("dti_opt", "representation_families.csv"), fam),
            atomic_save_table(self.ws.path("dti_opt", "mass_curves.csv"), mc),
            atomic_save_table(self.ws.path("dti_opt", "width_displacement.csv"), wd),
            atomic_write_json(self.ws.path("dti_opt", "selection.json"), sel),
        ]
        for t, v in res["by_task"].items():
            outs.append(atomic_save_table(self.ws.path("dti_opt", f"choice_{t}.csv"), v["per_zone"]))
        if self.cfg.figures:
            outs += [viz.dti_sensitivity(self.fx, syn), viz.mass_curve(self.fx, mc)]
        return [o for o in outs if o], {
            "champion": champ,
            "key": key,
            "crossfit_dti": res["by_task"][task]["summary"]["dti_crossfit"],
            "test_proxy_dti": tc.dti,
        }

    def stage_failure_modes(self):
        from . import dti_opt as O
        from . import failure as Fm

        sel = json.loads((self.ws.root / "dti_opt" / "selection.json").read_text())
        spec = O.spec_by_key(self.cfg, sel["key"])
        maps = self._maps(sel["champion_model"])
        reg = self._regimes()
        units, q_by = [], {}
        for u in self.sp.units(("geo", "disc_official")):
            s = maps["oof"] if u.task == "geo" else maps["disc"].get(u.zone)
            if s is None:
                continue
            q_by[u.key] = O.render(spec, u.crop(s), u.ev, u.known, u.crop(reg) if reg is not None else None, u.scored_mask())
            units.append(u)
        arch_p = self.ws.root / "regimes" / "archetypes.npy"
        per = Fm.taxonomy(units, q_by, np.load(arch_p) if arch_p.exists() else None, self._hard())
        summ = Fm.summary(per)
        wins = Fm.example_windows(units, q_by)
        outs = [
            atomic_save_table(self.ws.path("failure_modes", "per_unit.csv"), per),
            atomic_save_table(self.ws.path("failure_modes", "taxonomy.csv"), summ),
            atomic_save_table(self.ws.path("failure_modes", "example_windows.csv"), pd.DataFrame(wins)),
        ]
        if self.cfg.figures:
            full = np.asarray(maps["full"], np.float32)
            outs.append(viz.error_panels(self.fx, self.data, wins, full))
        return [o for o in outs if o], {"categories": len(summ)}

    def stage_submission(self):
        from . import dti_opt as O
        from .submission import COMPETITION_URL, export_candidate

        d = self.data
        board = self._scoreboard()
        sel = json.loads((self.ws.root / "dti_opt" / "selection.json").read_text())
        has_full = board.model.map(lambda m: (self.ws.root / "predictions" / m / "full.npy").exists())
        b = board[has_full]
        metric = "disc_proxy_dti" if self.cfg.primary_task == "disc_official" else "geo_proxy_dti"
        reg = self._regimes()
        cands = {}
        raw = b[b.family == "A0_raw"].sort_values(metric, ascending=False)
        if len(raw):
            cands["A"] = (
                raw.model.iloc[0],
                {
                    "rep": {"select": "top"},
                    "frac": float(raw[metric.replace("proxy_dti", "frac")].iloc[0]),
                    "ring_px": 0,
                    "key": "top|crossfit",
                },
            )
        phys = b[~b.family.isin(["A0_raw", "fusion"]) & ~b.model.str.startswith("F_")].sort_values(metric, ascending=False)
        if len(phys):
            cands["B"] = (
                phys.model.iloc[0],
                {
                    "rep": {"select": "top"},
                    "frac": float(phys[metric.replace("proxy_dti", "frac")].iloc[0]),
                    "ring_px": 0,
                    "key": "top|crossfit",
                },
            )
        cands["C"] = (sel["champion_model"], O.spec_by_key(self.cfg, sel["key"]))
        recs, rows = {}, []
        for c, (model, spec) in cands.items():
            full = np.asarray(self._maps(model)["full"], np.float32)
            q = O.render(spec, full, d.valid, d.labels, reg, d.valid)
            r = board.set_index("model").loc[model]
            stem = {
                "A": "submission_g13_candA_safe_baseline",
                "B": "submission_g13_candB_physics",
                "C": "submission_g13_candC_discovery",
            }[c]
            rec = export_candidate(
                stem,
                q,
                d,
                self.ws,
                {
                    "model": model,
                    "representation": spec.get("key", ""),
                    "geo_proxy_dti": r.get("geo_proxy_dti"),
                    "disc_proxy_dti": r.get("disc_proxy_dti"),
                    "test_proxy_dti": sel["test"]["test_proxy_dti"] if c == "C" else None,
                    "selection_metric": metric,
                },
                self.cfg.submission_known_policy,
            )
            recs[c] = rec
            v = rec["validation"]
            # same metric for every candidate (cross-fitted top-a proxy of its model); C's tournament score had a wider
            # representation search and would be optimistically biased against A / B
            crossfit_val = r.get(metric)
            rows.append(
                {
                    "candidate": c,
                    "file": v["file"],
                    "model": model,
                    "representation": spec.get("key", ""),
                    "selection_score": crossfit_val,
                    "geo_proxy_dti": r.get("geo_proxy_dti"),
                    "disc_proxy_dti": r.get("disc_proxy_dti"),
                    "valid": v["valid"],
                    "positive_px": v["positive_px"],
                    "pred_mass": v["pred_mass"],
                    "sha256": v["sha256"][:16],
                }
            )
        comp = pd.DataFrame(rows)
        best = comp[comp.valid].sort_values("selection_score", ascending=False).iloc[0]
        src = self.ws.root / "submissions" / best.file
        outs = []
        for stem in (SUBMISSION_FILES["C"], "predicted_faults"):
            p = self.ws.path("submissions", f"{stem}.tif")
            p.write_bytes(src.read_bytes())
            outs.append(p)
        champ_meta = {
            "chosen_candidate": best.candidate,
            "file": best.file,
            "model": best.model,
            "selection_score": float(best.selection_score),
            "selection_rule": f"highest cross-fitted {metric} among valid candidates (evidence, not complexity)",
            "upload": f"manual: {COMPETITION_URL} -> Submit -> upload predicted_faults.tif",
            "leaderboard_score": "not available until uploaded and logged with Engine.log_leaderboard()",
        }
        outs += [
            atomic_save_table(self.ws.path("submissions", "candidate_comparison.csv"), comp),
            atomic_write_json(self.ws.path("submissions", "champion.json"), champ_meta),
        ]
        outs += [Path(r["validation"]["path"]) for r in recs.values()]
        return outs, champ_meta

    def stage_report(self):
        T = lambda rel: load_table(self.ws.root / rel) if (self.ws.root / rel).exists() else None  # noqa: E731
        inv = self.ws.root / "inventory" / "inventory_summary.json"
        invs = json.loads(inv.read_text()) if inv.exists() else {}
        sel_p = self.ws.root / "dti_opt" / "selection.json"
        sel = json.loads(sel_p.read_text()) if sel_p.exists() else {}
        champ_p = self.ws.root / "submissions" / "champion.json"
        champ = json.loads(champ_p.read_text()) if champ_p.exists() else {}
        board = self._scoreboard()
        cols = [
            c
            for c in (
                "model",
                "family",
                "kind",
                "strategy",
                "n_features",
                "geo_proxy_dti",
                "geo_zone_min",
                "disc_proxy_dti",
                "disc_proxy_dti_ci_lo",
                "disc_proxy_dti_ci_hi",
                "disc_zone_min",
                "disc_frac",
            )
            if c in board
        ]
        exec_lines = [
            f"Mode `{self.cfg.mode}`; G13 {G13_VERSION}.",
            f"Champion: `{champ.get('model', 'n/a')}` (candidate {champ.get('chosen_candidate', 'n/a')}), "
            f"selection score {champ.get('selection_score', float('nan')):.4f} "
            f"(cross-fitted {self.cfg.primary_task} proxy).",
            f"Once-only TEST proxy DTI (rediscovery of mapped faults in the held-out fold): "
            f"{sel.get('test', {}).get('test_proxy_dti', float('nan')):.4f}.",
            "No official leaderboard score is reported: upload manually and log it with "
            "`Engine.log_leaderboard(file, score)`.",
        ]
        hyp = [
            "H1: faults leave coincident edges in independent modalities (tested: cross-modal family, A8).",
            "H2: buried faults are visible in potential fields without terrain expression (archetype "
            "`buried_no_terrain`, magnetic/gravity experts).",
            "H3: multiscale persistence separates structures from noise (structural family, A6).",
            "H4: incomplete labels bias standard sampling; PU-aware sampling improves discovery (section 23).",
            "Each hypothesis is judged only by held-out proxies; the findings table states the evidence.",
        ]
        content = {
            "Executive summary": exec_lines,
            "Dataset characterization": [
                f"Grid {invs.get('shape')} at 100 m, {invs.get('crs')}; "
                f"{invs.get('fault_px', 'n/a')} mapped fault px "
                f"({invs.get('fault_frac_valid', float('nan')):.3%} of valid).",
                T("inventory/band_inventory.csv"),
            ],
            "Geological hypotheses": hyp,
            "Feature engineering": [T("quality/feature_quality.csv")],
            "Key exploratory results": [T("eda/scale_space_auc.csv"), T("eda/correlation_lengths.csv")],
            "Best-performing physical indicators": [T("eda/discriminative_summary.csv")],
            "Geological regimes": [T("regimes/regime_atlas.csv"), T("regimes/archetypes.csv")],
            "Fault discovery candidates": [
                "Candidates are model predictions, not confirmed faults.",
                T("topology/network_summary.csv"),
                T("topology/candidates.csv"),
            ],
            "Model comparisons": [
                board[cols].sort_values("disc_proxy_dti", ascending=False) if len(board) else None,
                T("ablation/paired_vs_raw.csv"),
                T("ablation/interactions.csv"),
                T("fusion/fusion_scoreboard.csv"),
            ],
            "Validation methodology": [
                T("splits/audit.csv"),
                "Geographic CV folds with a 13 km embargo; one untouched TEST fold; "
                "whole fault-system discovery splits scored with the official pixel-exact "
                "known-fault mask (disc_official). All selections are cross-fitted.",
            ],
            "DTI analysis": [
                T("dti_opt/representation_families.csv"),
                T("dti_opt/width_displacement.csv"),
                T("dti_opt/synthetic_sensitivity.csv"),
            ],
            "Failure categories": [T("failure_modes/taxonomy.csv")],
            "Final submission selection": [
                T("submissions/candidate_comparison.csv"),
                f"Chosen: `{champ.get('file', 'n/a')}`; {champ.get('selection_rule', '')}",
            ],
            "Uncertainty and limitations": [
                "Offline DTI uses withheld *known* faults; the hidden test contains *new* faults only.",
                "Discovery splits withhold whole systems, under-representing new faults near known traces.",
                "Confidence intervals are spatial-block bootstraps; with few zones they are descriptive.",
                findings_table(self.findings) if self.findings else "No quantified findings in this run.",
            ],
            "Recommended next experiments": self._next_steps(board),
        }
        content["Executive summary"].append(runtime_table(self.ws))
        md, html = write_report(self.ws, content)
        ft = atomic_save_table(self.ws.path("reports", "findings.csv"), findings_table(self.findings))
        return [md, html, ft], {"report": str(md)}

    def _next_steps(self, board: pd.DataFrame) -> list:
        out = []
        if board.empty:
            return ["Run the baseline stage."]
        if "disc_proxy_dti" in board and board.disc_proxy_dti.notna().any():
            b = board.sort_values("disc_proxy_dti", ascending=False).iloc[0]
            out.append(
                f"Best disc proxy: `{b.model}` ({b.disc_proxy_dti:.4f}). Upload its candidate and log the public "
                "score to calibrate the proxy."
            )
        if not (self.ws.root / "predictions" / "DL_unet_multitask").exists():
            out.append("Tier-2 network not run: set run_deep=True on a GPU runtime and compare under the same units.")
        out.append("Add DEM-derived 1 m features (G07 dem_1m_100m) through `dem_dir` and test on DEM-covered pixels.")
        return out

    # ------------------------------------------------------------------ leaderboard
    def log_leaderboard(self, submission_file: str, public_score: float, notes: str = "") -> pd.DataFrame:
        from .submission import leaderboard_feedback, log_leaderboard

        log_leaderboard(self.ws, submission_file, public_score, notes)
        self.ws.persist(["submissions"])
        return leaderboard_feedback(self.ws)


def _geojson(c: pd.DataFrame) -> dict:
    feats = []
    for r in c.itertuples():
        props = {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in r._asdict().items()
            if k not in ("Index",)
        }
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r.x_utm), float(r.y_utm)]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32611"}}, "features": feats}


def main(argv=None):
    ap = argparse.ArgumentParser(description="GEMS G13 GeoDiscovery engine")
    ap.add_argument("--mode", default="FAST_BASELINE", choices=list(MODE_STAGES))
    ap.add_argument("--config", default="", help="YAML overrides (configs/g13/*.yaml)")
    ap.add_argument("--drive-root", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--raw", default="")
    ap.add_argument("--known", default="")
    ap.add_argument("--template", default="")
    ap.add_argument("--stages", default="", help="comma-separated subset")
    ap.add_argument("--force", default="", help="comma-separated stages to recompute")
    a = ap.parse_args(argv)
    kw = {
        k: v
        for k, v in {
            "drive_root": a.drive_root,
            "work_dir": a.work_dir,
            "out_dir": a.out_dir,
            "raw_path": a.raw,
            "known_path": a.known,
            "template_path": a.template,
        }.items()
        if v
    }
    if a.stages:
        kw["stages"] = tuple(s.strip() for s in a.stages.split(","))
    if a.force:
        kw["force_stages"] = tuple(s.strip() for s in a.force.split(","))
    cfg = G13Config.from_yaml(a.config, mode=a.mode, **kw) if a.config else G13Config.from_mode(a.mode, **kw)
    eng = Engine(cfg)
    print(eng.run().to_string())


if __name__ == "__main__":
    main()
