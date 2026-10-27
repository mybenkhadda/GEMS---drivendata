"""G01 orchestration: ``G01Experiment(m9, G01_CONFIG)`` -> prepare / evaluate / exploratory / metrics / figures / report.

    g01 = G01Experiment(m9, G01_CONFIG)       # m9 = pipe.m9(M9_CONFIG); no M9 section is run, no model is built
    g01.prepare()                             # cache audit, wavelength alignment, fingerprinted run folder
    g01.evaluate()                            # held-out units (resumable per unit)
    g01.exploratory()                         # label-free full-region maps (optional)
    g01.metrics(); g01.figures(); g01.report(); g01.push()
    # later session, no recomputation:
    g01 = G01Experiment.attach(m9)            # newest run (LATEST.json) or attach(m9, run_id="g01_...")
    g01.results()

Hypothesis contrasts (``CONTRASTS``) are evaluated with a paired block bootstrap at the primary budget / distance and
classified SUPPORTED / REFUTED / INCONCLUSIVE by ``decision`` (configuration), never by inspection of the maps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import geology_figures as FG
from . import geology_metrics as GM
from .geology_io import (G01_VERSION, G01Store, M9Source, build_missing_layers, channel_audit, fingerprint_config,
                         latest_run, list_runs, load_g01_config, run_root, set_latest, xmod_names)
from .geology_wavelet import (BASELINES, CANDIDATES, KINDS, ROLES, Wavelength, build_evidence, channel, halo_m,
                              is_candidate_key, meters_to_px, tolerance_table)

TASK_CODE = {"geo": 0, "disc_official": 1, "disc_retained_fp": 2}
CONTRASTS = [  # (hypothesis, candidate, reference, question)
    ("G01-A", "G01-A", "edge", "magnetic wavelet energy beyond a simple magnetic gradient edge"),
    ("G01-A", "G01-A", "random", "magnetic wavelet energy beyond random placement"),
    ("G01-B", "G01-B", "edge", "gravity wavelet energy beyond a simple magnetic gradient edge"),
    ("G01-B", "G01-B", "random", "gravity wavelet energy beyond random placement"),
    ("G01-C", "G01-C", "G01-A", "joint evidence beyond magnetic alone"),
    ("G01-C", "G01-C", "G01-B", "joint evidence beyond gravity alone"),
    ("G01-D", "G01-D", "G01-C", "strike compatibility beyond joint evidence"),
    ("G01-E", "G01-E", "G01-D", "multiscale persistence beyond joint + orientation"),
    ("G01-F", "G01-E", "G01-F", "G01 physical fusion beyond M9 engineered cross-modal layers"),
    ("G01-F", "G01-C", "G01-F", "simple joint wavelet evidence beyond M9 engineered cross-modal layers"),
    ("G01-F", "G01-F", "edge", "M9 engineered cross-modal layers beyond a simple edge"),
]
COMPONENT_KEYS = ("comp:M", "comp:G", "comp:J", "comp:J_D", "comp:P", "comp:P_mag", "comp:P_grav", "comp:P_cross",
                  "comp:orient_agreement", "comp:orient_confidence", "comp:orient_gate", "comp:S")


class G01Experiment:
    def __init__(self, m9, config: dict | None = None, *, _cfg: dict | None = None):
        self.m9 = m9
        self.cfg = _cfg if _cfg is not None else load_g01_config(config)
        self.src = M9Source(m9, self.cfg)
        self.parent = run_root(m9, self.cfg)
        self.store: G01Store | None = None
        self.lams: list[Wavelength] = []
        self.xmod: list[str] = []
        self._refs_all: dict | None = None
        self.warnings: list[str] = []

    def say(self, msg: str) -> None:
        print(f"[g01] {msg}")

    # =========================================================================== inputs
    def _rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(Path(self.m9.local).resolve()).as_posix()

    def stage_inputs(self) -> None:
        """Bring the M9 layer cache and earlier G01 runs from Drive when they exist there (never regenerated)."""
        rels = [f"{self.m9.cfg['cache_dir']}/layers"]
        try:
            rels.append(self._rel(self.parent))
        except ValueError:
            pass
        self.m9.stage_in(rels)

    def push(self) -> None:
        if self.store is None or not self.cfg["save"].get("push_to_drive", True):
            return
        try:
            self.m9.push([self._rel(self.store.root)])
        except ValueError as e:
            self.say(f"push skipped: {e}")

    def push_cache(self) -> None:
        """Push the M9 layer cache (only after ``build_missing`` created layers): later sessions reuse them."""
        if self.m9.pc is not None and self.cfg["save"].get("push_to_drive", True):
            try:
                self.m9.pc.push([f"{self.m9.cfg['cache_dir']}/layers"])
            except Exception as e:
                self.say(f"cache push skipped: {type(e).__name__}: {e}")

    def channels(self) -> list[str]:
        fam = self.cfg["fields"]
        out = [channel(fam[r], k, w.tag(r)) for w in self.lams for r in ROLES for k in KINDS]
        return [n for n in out if self.src.has(n)] + list(self.xmod) + \
            [v for v in ("g01:edge", "g01:surface") if self.src.has(v)]

    def percentile_channels(self) -> list[str]:
        fam = self.cfg["fields"]
        out = [channel(fam[r], "energy", w.tag(r)) for w in self.lams for r in ROLES]
        return [n for n in out if self.src.has(n)] + list(self.xmod) + \
            [v for v in ("g01:edge", "g01:surface") if self.src.has(v)]

    def grid_meta(self) -> dict:
        g = self.src.data.grid
        crs = g.crs.to_epsg() if hasattr(g.crs, "to_epsg") else str(g.crs)
        return {"shape": list(self.src.shape), "pixel_m_rows": self.src.pixel_m[0], "pixel_m_cols": self.src.pixel_m[1],
                "crs_epsg": crs, "transform": [float(x) for x in tuple(g.transform)[:6]],
                "valid_px": int(self.src.valid.sum())}

    def prepare(self) -> dict:
        """Audit the cache, align wavelengths, fingerprint the inputs, open (or reuse) the versioned run folder."""
        self.m9.load()
        self.stage_inputs()
        self.warnings = []
        cfg = self.cfg
        names = self.src.names()
        audit, lams = channel_audit(names, cfg["fields"], cfg["wavelengths_km"], float(cfg["wavelength_rel_tol"]))
        need_build = (not lams) or (len(audit) and (audit.status == "missing").any())
        if need_build and any(cfg["build_missing"].values()):
            built = build_missing_layers(self.m9, cfg, audit, self.say)
            if built:
                self.push_cache()
            names = self.src.names()
            audit, lams = channel_audit(names, cfg["fields"], cfg["wavelengths_km"], float(cfg["wavelength_rel_tol"]))
        self.xmod = xmod_names(names, cfg["existing"]["patterns"])
        if not self.xmod and "G01-F" in cfg["candidates"] and cfg["build_missing"].get("crossmodal"):
            build_missing_layers(self.m9, {**cfg, "build_missing": {"wavelets": False, "crossmodal": True}}, audit,
                                 self.say)
            self.push_cache()
            names = self.src.names()
            self.xmod = xmod_names(names, cfg["existing"]["patterns"])
        if not lams:
            raise RuntimeError("G01: no wavelength is cached for BOTH fields in "
                               f"{self.m9.layers.dir}. Run M9 section 03 once (m9.s03_spectral()) or set "
                               "G01_CONFIG['build_missing'] = {'wavelets': True} (computes only the log-Gabor layers).")
        self.lams = lams
        if not self.xmod:
            self.warnings.append("G01-F unavailable: no cached m9:xmod:mag_grav_* layer (run M9 section 02 or set "
                                 "build_missing.crossmodal)")
        gm = self.grid_meta()
        exp = float(cfg["expected_pixel_m"])
        if max(abs(gm["pixel_m_rows"] - exp), abs(gm["pixel_m_cols"] - exp)) > 0.01 * exp:
            self.warnings.append(f"grid pixel ({gm['pixel_m_rows']:.1f} x {gm['pixel_m_cols']:.1f} m) differs from "
                                 f"the {exp:g} m assumed by M9 wavelet wavelength labels: wavelengths are mislabelled")
        inputs = self.channels()
        fps = {n: self.src.fingerprint(n) for n in inputs}
        from ...pipeline.m6 import fingerprint

        fp = fingerprint(G01_VERSION, fingerprint_config(cfg), fps, gm, [(w.lam_km, dict(w.tags)) for w in lams])
        run_id = f"g01_{fp[:10]}"
        self.store = G01Store(self.parent / run_id, overwrite=bool(cfg["overwrite"]), fingerprint=fp, push=self.push)
        st = self.store
        st.save_json(json.loads(json.dumps(cfg, default=str)), "config/g01_config.json")
        st.save_json({"run_id": run_id, "fingerprint": fp, "g01_version": G01_VERSION, "grid": gm,
                      "wavelengths": [{"lam_km": w.lam_km, "tags": dict(w.tags)} for w in lams],
                      "xmod": self.xmod, "input_fingerprints": fps, "warnings": self.warnings,
                      "m9_layer_dir": str(self.m9.layers.dir), "m9_spectral_wavelet": self.m9.cfg["spectral"]["wavelet"],
                      "m9_crossmodal": self.m9.cfg["features"]["crossmodal"],
                      "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "config/inputs.json", sources=inputs)
        st.save_table(audit, "channel_audit", "tables", sources=inputs)
        st.save_table(pd.DataFrame(tolerance_table(lams, cfg, self.src.pixel_m)), "wavelength_tolerances", "tables",
                      params=cfg["tolerance"])
        set_latest(self.parent, run_id, fp)
        st.mark_done("inputs", {"n_wavelengths": len(lams), "n_xmod": len(self.xmod)})
        st.flush_manifest({"run_id": run_id, "grid": gm})
        for w_ in self.warnings:
            self.say(f"WARNING {w_}")
        self.say(f"run {run_id}: wavelengths {[w.label for w in lams]}, {len(self.xmod)} M9 cross-modal layers -> {st.root}")
        return {"run_id": run_id, "audit": audit, "wavelengths": lams, "grid": gm, "warnings": self.warnings}

    @classmethod
    def attach(cls, m9, run_id: str | None = None, config: dict | None = None) -> "G01Experiment":
        """Open an existing run (newest by default) without recomputing anything."""
        tmp = cls(m9, config)
        tmp.stage_inputs()
        rid = run_id or latest_run(tmp.parent)
        if not rid or not (tmp.parent / rid).exists():
            runs = list_runs(tmp.parent) if tmp.parent.exists() else pd.DataFrame()
            known = list(runs.run_id) if len(runs) else []
            raise FileNotFoundError(f"no G01 run under {tmp.parent} (known runs: {known})")
        root = tmp.parent / rid
        cfg = json.loads((root / "config" / "g01_config.json").read_text(encoding="utf-8"))
        inp = json.loads((root / "config" / "inputs.json").read_text(encoding="utf-8"))
        g = cls(m9, _cfg=cfg)
        g.store = G01Store(root, overwrite=False, fingerprint=inp["fingerprint"], push=g.push)
        g.lams = [Wavelength(float(x["lam_km"]), tuple(sorted(x["tags"].items()))) for x in inp["wavelengths"]]
        g.xmod = list(inp.get("xmod") or [])
        g.warnings = list(inp.get("warnings") or [])
        return g

    def _need(self) -> G01Store:
        if self.store is None:
            raise RuntimeError("call g01.prepare() (or G01Experiment.attach) first")
        return self.store

    # =========================================================================== windows / references
    def fit_refs(self, ref_mask: np.ndarray, seed: int) -> dict:
        from .geology_wavelet import QuantileRef

        nc = self.cfg["normalization"]
        idx = np.flatnonzero(np.asarray(ref_mask, bool).ravel())
        if idx.size > int(nc["sample_px"]):
            idx = np.sort(np.random.default_rng(seed).choice(idx, int(nc["sample_px"]), replace=False))
        refs = {}
        for n in self.percentile_channels():
            v = np.asarray(self.src.get(n)).reshape(-1)[idx]
            refs[n] = QuantileRef.fit(v, int(nc["n_quantiles"]))
        self.src.release()
        return refs

    def halo_window(self, sl) -> tuple:
        H, W = self.src.shape
        hy, hx = meters_to_px(halo_m(self.lams, self.cfg), self.src.pixel_m)
        r0, r1 = max((sl[0].start or 0) - hy, 0), min((sl[0].stop or H) + hy, H)
        c0, c1 = max((sl[1].start or 0) - hx, 0), min((sl[1].stop or W) + hx, W)
        hs = (slice(r0, r1), slice(c0, c1))
        inner = (slice((sl[0].start or 0) - r0, (sl[0].stop or H) - r0),
                 slice((sl[1].start or 0) - c0, (sl[1].stop or W) - c0))
        return hs, inner, (r0 == 0, r1 == H, c0 == 0, c1 == W)

    def read_window(self, hs) -> dict:
        valid = self.src.valid[hs]
        out = {}
        for n in self.channels():
            a = np.asarray(self.src.get(n)[hs], np.float32)
            out[n] = np.where(valid, a, np.nan).astype(np.float32)
        return out

    def evidence_on(self, sl, refs: dict, *, keep_per_scale: bool = False,
                    ablations: bool = True) -> tuple[dict, np.ndarray]:
        """(evidence trimmed to ``sl``, interior mask on ``sl``) computed on a halo window."""
        hs, inner, touches = self.halo_window(sl)
        win = self.read_window(hs)
        ev = build_evidence(win, refs, self.lams, self.cfg, self.src.pixel_m, keep_per_scale=keep_per_scale,
                            ablations=ablations)
        interior = GM.interior_mask(self.src.valid[hs], float(self.cfg["boundary_margin_m"]), self.src.pixel_m, touches)
        self.src.release()
        return {k: v[inner] for k, v in ev.items()}, interior[inner]

    # =========================================================================== held-out evaluation
    def units(self, task: str) -> list:
        return self.m9.m7.units_of(task) if self.cfg["evaluation"]["zones"] == "all" else self.m9.units_of(task)

    def main_names(self) -> list[str]:
        out = list(self.cfg["candidates"]) + ["edge"]
        return out + (["random"] if self.cfg["baselines"].get("random", True) else [])

    def evaluate(self, tasks: list[str] | None = None) -> pd.DataFrame:
        st = self._need()
        rows = []
        for task in tasks or self.cfg["evaluation"]["tasks"]:
            for u in self.units(task):
                rows.append(self.evaluate_unit(u))
        T = pd.DataFrame(rows)
        st.save_table(T, "units", "tables")
        st.flush_manifest()
        self.push()
        return T

    def evaluate_unit(self, u) -> dict:
        st = self._need()
        cfg, ev_c = self.cfg, self.cfg["evaluation"]
        key = f"{u.task}_{u.zone}"
        if st.is_done(f"unit_{key}"):
            self.say(f"{key}: reused")
            return (st.load_json(f"processed/unit_{key}.json") or {"unit": key, "status": "reused"})
        t0 = time.perf_counter()
        H, W = self.src.shape
        seed = int(cfg["seed"]) + 7919 * int(u.zone) + 104729 * TASK_CODE.get(u.task, 9)
        ev_full = GM.full_mask((H, W), u.sl, u.ev)
        if cfg["normalization"]["reference"] == "training_geography":
            ref = GM.reference_mask(self.m9.ctx, u.task, u.zone)
            GM.assert_disjoint(ref, ev_full, f"{u.key} normalization reference")
        else:
            ref = self.src.valid.copy()                       # explicit ablation: label-free but not held-out clean
        refs = self.fit_refs(ref, seed)
        st.save_json({n: r.to_dict() for n, r in refs.items()}, f"processed/refs_{key}.json",
                      params={"reference": cfg["normalization"]["reference"], "ref_px": int(ref.sum())})
        evd, interior = self.evidence_on(u.sl, refs)
        el = np.asarray(self.m9.elig(u, 0), bool)
        domain = el & interior
        truth = np.asarray(u.eval_truth(), bool) & interior
        scores = {k: v for k, v in evd.items() if is_candidate_key(k)}
        if cfg["baselines"].get("random", True):
            scores["random"] = np.where(domain, np.random.default_rng(seed).random(domain.shape), np.nan).astype(np.float32)
        main = [n for n in self.main_names() if n in scores]
        fr = {n: [float(f) for f in ev_c["area_fracs"]] for n in main}
        fr.update({n: [float(f) for f in ev_c["ablation_area_fracs"]] for n in scores if n not in fr})
        py, px = self.src.pixel_m
        bpx = max(int(round(float(ev_c["block_km"]) * 1000.0 / (0.5 * (py + px)))), 8)
        blk, ids = GM.block_index(u.sl, domain.shape, bpx)
        B = GM.unit_block_counts(scores, domain, truth, self.src.pixel_m, fr, ev_c["distance_tol_m"], blk, ids,
                                 tags={"task": u.task, "zone": int(u.zone), "unit": key})
        st.save_table(B, f"blocks_{key}", "processed", params={"block_px": bpx})
        pkey = self.m9.pkey(u)
        if ev_c.get("official_style", True):
            masks = {(n, f): GM.equal_area_mask(scores[n], domain, f) for n in main for f in fr[n]}
            O = pd.DataFrame(GM.official_style_rows(u, masks, el, pkey)).assign(selection_domain="eligible & interior")
            st.save_table(O, f"official_{key}", "processed")
        keep = {k: v for k, v in evd.items() if k in set(main) or k in COMPONENT_KEYS}
        keep["domain"] = domain
        keep["interior"] = interior
        r0, r1, c0, c1 = u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop
        st.save_raster(keep, f"rasters/units/{key}.npz",
                       {"task": u.task, "zone": int(u.zone), "window": [r0, r1, c0, c1], "grid": self.grid_meta(),
                        "protocol_key": pkey, "normalization": cfg["normalization"]["reference"],
                        "selection_domain": "m9.elig(u, 0) & interior(boundary_margin_m)",
                        "labels_stored": False, "note": "truth is re-derived from the M7 unit (labels never stored)"},
                       dtype=cfg["save"]["raster_dtype"], sources=self.channels())
        info = {"unit": key, "task": u.task, "zone": int(u.zone), "status": "computed",
                "eval_px": int(u.ev.sum()), "eligible_px": int(el.sum()), "domain_px": int(domain.sum()),
                "interior_loss_share": float(1 - domain.sum() / max(int(el.sum()), 1)),
                "truth_px": int(truth.sum()), "reference_px": int(ref.sum()), "block_px": bpx,
                "n_candidates": len(scores), "seconds": round(time.perf_counter() - t0, 1), "protocol_key": pkey}
        st.save_json(info, f"processed/unit_{key}.json")
        st.mark_done(f"unit_{key}", info)
        st.flush_manifest()
        u.release()
        self.say(f"{key}: {info['domain_px']} domain px, {info['truth_px']} truth px, {info['seconds']} s")
        return info

    # =========================================================================== exploratory full region
    def refs_all(self) -> dict:
        if self._refs_all is None:
            st = self._need()
            d = st.load_json("processed/refs_all_valid.json")
            if d:
                from .geology_wavelet import QuantileRef

                self._refs_all = {n: QuantileRef.from_dict(v) for n, v in d.items()}
            else:
                self._refs_all = self.fit_refs(self.src.valid, int(self.cfg["seed"]))
                st.save_json({n: r.to_dict() for n, r in self._refs_all.items()}, "processed/refs_all_valid.json",
                             params={"reference": "all_valid (exploratory, label-free)"})
        return self._refs_all

    def exploratory(self) -> dict | None:
        """Label-free full-region maps (reference = every valid pixel). EXPLORATORY: not a validated result."""
        st = self._need()
        if not self.cfg["exploratory"].get("full_region", True):
            return None
        rel = "rasters/exploratory_full_region.npz"
        if st.is_done("exploratory") and st.exists(rel):
            self.say("exploratory maps: reused")
            return st.load_raster(rel)[0]
        refs = self.refs_all()
        H, W = self.src.shape
        tp = int(self.cfg["exploratory"]["tile_px"])
        keys = [k for k in list(self.cfg["candidates"]) + ["edge"]] + list(COMPONENT_KEYS)
        out: dict = {}
        interior = np.zeros((H, W), bool)
        for r in range(0, H, tp):
            for c in range(0, W, tp):
                sl = (slice(r, min(r + tp, H)), slice(c, min(c + tp, W)))
                if not self.src.valid[sl].any():
                    continue
                ev, inter = self.evidence_on(sl, refs, ablations=False)
                interior[sl] = inter
                for k in keys:
                    if k in ev:
                        if k not in out:
                            out[k] = np.full((H, W), np.nan, np.float16 if self.cfg["save"]["raster_dtype"] == "float16"
                                             else np.float32)
                        out[k][sl] = ev[k]
        out["interior"] = interior
        st.save_raster(out, rel, {"kind": "exploratory", "reference": "all_valid", "grid": self.grid_meta(),
                                  "window": [0, H, 0, W], "warning": "label-free evidence; not verified fault discovery"},
                       dtype=self.cfg["save"]["raster_dtype"], sources=self.channels())
        if self.cfg["save"].get("geotiff"):
            for k in self.cfg["candidates"]:
                if k in out:
                    st.save_geotiff(out[k], f"rasters/exploratory_{k}.tif", self.src.data.grid, (0, H, 0, W))
        st.mark_done("exploratory", {"keys": sorted(out)})
        st.flush_manifest()
        self.push()
        return out

    # =========================================================================== metrics
    def blocks(self) -> pd.DataFrame:
        st = self._need()
        parts = [st.load_table(p.stem, "processed") for p in sorted(st.path("processed").glob("blocks_*.csv"))
                 if p.stem != "blocks_all"]                    # per-unit tables only (never the pooled copy)
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def official(self) -> pd.DataFrame:
        st = self._need()
        parts = [st.load_table(p.stem, "processed") for p in sorted(st.path("processed").glob("official_*.csv"))]
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def metrics(self) -> dict:
        st = self._need()
        cfg, ev_c = self.cfg, self.cfg["evaluation"]
        B = self.blocks()
        if not len(B):
            raise RuntimeError("no evaluated unit: run g01.evaluate() first")
        f0, d0 = float(ev_c["primary_area_frac"]), float(ev_c["primary_distance_m"])
        fa = float(ev_c["ablation_area_fracs"][0]) if f0 not in [float(x) for x in ev_c["ablation_area_fracs"]] else f0
        seed, nb = int(cfg["seed"]), int(ev_c["n_boot"])
        main = [n for n in self.main_names() if n in set(B.candidate)]
        keys = ["task", "area_frac", "tol_m"]
        S = GM.versus_edge(GM.summarize(B, ["task", "candidate", "area_frac", "tol_m"]), keys)
        Z = GM.versus_edge(GM.summarize(B, ["task", "zone", "candidate", "area_frac", "tol_m"]),
                           ["task", "zone", "area_frac", "tol_m"])
        nbc = min(nb, 200)
        Bm = B[B.candidate.isin(main)]
        ci = []
        for m in ("precision", "recall", "enrichment"):
            a = GM.per_block_uncertainty(Bm[Bm.tol_m == d0], ["task", "candidate", "area_frac", "tol_m"], m, nbc, seed)
            b = GM.per_block_uncertainty(Bm[(Bm.area_frac == f0) & (Bm.tol_m != d0)],
                                         ["task", "candidate", "area_frac", "tol_m"], m, nbc, seed)
            u = pd.concat([a, b], ignore_index=True)
            if len(u):
                ci.append(u.rename(columns={"ci_lo": f"{m}_lo", "ci_hi": f"{m}_hi"}).drop(columns=["metric", "value"]))
        for c in ci:
            S = S.merge(c, on=["task", "candidate", "area_frac", "tol_m"], how="left")
        boot = GM.bootstrap_table(B, main, list(ev_c["references"]), f0, d0, nb, seed)
        H = self.contrasts(B, Z, f0, d0)
        abl = S[S.candidate.str.contains("@|\\|tol=", regex=True) & (S.area_frac == fa)]
        fams = {}
        for base in ("G01-A", "G01-B", "G01-C", "G01-D"):
            v = sorted(n for n in set(B.candidate) if n.startswith(f"{base}@"))
            if len(v) > 1:
                fams[f"{base}:wavelength"] = v
        tv = sorted(n for n in set(B.candidate) if n.startswith("G01-C|tol="))
        if len(tv) > 1:
            fams["G01-C:tolerance"] = tv
        Zf = Z[(Z.area_frac == fa) & (Z.tol_m == d0)]
        nested = GM.nested_choice(Zf, fams, cfg["decision"]["metric"]) if fams else pd.DataFrame()
        O = self.official()
        Op = GM.pooled_official(O) if len(O) else pd.DataFrame()
        for name, df in (("summary_pooled", S), ("summary_per_zone", Z), ("bootstrap_vs_references", boot),
                         ("hypothesis_contrasts", H), ("ablations", abl), ("nested_choice", nested),
                         ("official_style_per_zone", O), ("official_style_pooled", Op)):
            st.save_table(df, name, "tables")
        st.save_table(B, "blocks_all", "tables")
        summ = self.summary_json(S, H, f0, d0)
        st.save_json(summ, "reports/summary.json")
        st.mark_done("metrics", {"n_blocks": int(len(B))})
        st.flush_manifest()
        self.push()
        return {"pooled": S, "per_zone": Z, "bootstrap": boot, "contrasts": H, "ablations": abl, "nested": nested,
                "official_pooled": Op, "summary": summ}

    def contrasts(self, B: pd.DataFrame, Z: pd.DataFrame, f0: float, d0: float) -> pd.DataFrame:
        dc, ev_c = self.cfg["decision"], self.cfg["evaluation"]
        m = dc["metric"]
        have = set(B.candidate)
        rows = []
        Bs = B[(B.area_frac == f0) & (B.tol_m == d0)]
        Zs = Z[(Z.area_frac == f0) & (Z.tol_m == d0)]
        for hyp, c, r, q in CONTRASTS:
            for task, g in Bs.groupby("task"):
                if c not in have or r not in have:
                    rows.append({"hypothesis": hyp, "task": task, "candidate": c, "reference": r, "question": q,
                                 "verdict": "UNAVAILABLE"})
                    continue
                bt = GM.paired_block_bootstrap(g, c, r, m, int(ev_c["n_boot"]), int(self.cfg["seed"]))
                zc = Zs[(Zs.task == task) & (Zs.candidate == c)].set_index("zone")[m]
                zr = Zs[(Zs.task == task) & (Zs.candidate == r)].set_index("zone")[m]
                j = pd.concat([zc, zr], axis=1, keys=["c", "r"]).dropna()
                win = float((j.c > j.r).mean()) if len(j) else np.nan
                p = bt["p_delta_positive"]
                if np.isfinite(p) and bt["delta"] > 0 and p >= float(dc["min_p_positive"]) and \
                        np.isfinite(win) and win >= float(dc["min_zone_win_share"]):
                    v = "SUPPORTED"
                elif np.isfinite(p) and bt["delta"] < 0 and p <= 1 - float(dc["min_p_positive"]):
                    v = "REFUTED"
                else:
                    v = "INCONCLUSIVE"
                rows.append({"hypothesis": hyp, "task": task, "question": q, **bt, "zone_win_share": win,
                             "n_zones": int(len(j)), "area_frac": f0, "tol_m": d0, "verdict": v})
        return pd.DataFrame(rows)

    def summary_json(self, S: pd.DataFrame, H: pd.DataFrame, f0: float, d0: float) -> dict:
        s = S[(S.area_frac == f0) & (S.tol_m == d0) & S.candidate.isin(self.main_names())]
        cols = ["precision", "recall", "f1", "enrichment", "precision_vs_edge"]
        return {"run_id": self.store.root.name, "fingerprint": self.store.fp, "primary_area_frac": f0,
                "primary_distance_m": d0, "decision": self.cfg["decision"], "warnings": self.warnings,
                "status": "validated on held-out zones (these zones were inspected in earlier GEMS work: not an "
                          "untouched test set)",
                "pooled": {t: {r.candidate: {c: (None if pd.isna(getattr(r, c)) else float(getattr(r, c))) for c in cols}
                               for r in g.itertuples()} for t, g in s.groupby("task")},
                "verdicts": (H[[c for c in ("hypothesis", "task", "candidate", "reference", "verdict") if c in H]]
                             .to_dict("records") if len(H) else [])}

    def results(self) -> dict:
        """Every saved table of the run (no recomputation)."""
        st = self._need()
        names = ("units", "channel_audit", "wavelength_tolerances", "summary_pooled", "summary_per_zone",
                 "bootstrap_vs_references", "hypothesis_contrasts", "ablations", "nested_choice",
                 "official_style_pooled")
        out = {n: st.load_table(n, "tables") for n in names}
        out["summary"] = st.load_json("reports/summary.json")
        out["manifest"] = st.load_json("manifest.json")
        return out

    # =========================================================================== figures
    def figure_center(self) -> tuple[int, int]:
        for task in self.cfg["evaluation"]["tasks"]:
            us = self.units(task)
            if us:
                return FG.unit_center(us[0])
        H, W = self.src.shape
        return H // 2, W // 2

    def figures(self, center=None) -> list[str]:
        st = self._need()
        fc, ev_c = self.cfg["figures"], self.cfg["evaluation"]
        grid = self.src.data.grid
        half = int(fc["window_half_px"])
        w = FG.window(self.src.shape, center or self.figure_center(), half)
        sl = (slice(w[0], w[1]), slice(w[2], w[3]))
        ev, _ = self.evidence_on(sl, self.refs_all(), keep_per_scale=True, ablations=False)
        mid = self.lams[len(self.lams) // 2]
        note = "EXPLORATORY: label-free, percentile reference = every valid pixel"
        figs = {"01_raw_signals": FG.raw_signals(self.src, w),
                "02_magnetic_energy_scales": FG.energy_scales(grid, w, ev, self.lams, "mag"),
                "03_gravity_energy_scales": FG.energy_scales(grid, w, ev, self.lams, "grav"),
                "04_strike_coherence": FG.strike_coherence(self.src, w, self.cfg["fields"], mid),
                "05_orientation_agreement": FG.orientation_agreement(grid, w, ev, self.lams),
                "06_single_vs_joint": FG.single_vs_joint(grid, w, ev),
                "07_persistence": FG.persistence_maps(grid, w, ev),
                "08_six_way": FG.six_way(grid, w, ev, note)}
        self.src.release()
        S = st.load_table("summary_pooled", "tables")
        main = self.main_names()
        if len(S):
            figs["10_performance_vs_area"] = FG.perf_vs_area(S, float(ev_c["primary_distance_m"]), main)
            figs["11_performance_vs_distance"] = FG.perf_vs_distance(S, float(ev_c["primary_area_frac"]), main)
        o9 = self._overlay_figure()
        if o9 is not None:
            figs["09_label_overlays"] = o9
        o12 = self._cases_figure(S)
        if o12 is not None:
            figs["12_cases"] = o12
        written = []
        for name, out in figs.items():
            fig, data = out if isinstance(out, tuple) else (out, None)
            p = st.save_fig(fig, name, data, dpi=int(fc["dpi"]), pdf=bool(self.cfg["save"].get("pdf", True)))
            if p is not None:
                written.append(str(p))
        st.mark_done("figures", {"n": len(written)})
        st.flush_manifest()
        self.push()
        return written

    def _unit_arrays(self, task: str, zone: int):
        st = self._need()
        rel = f"rasters/units/{task}_{zone}.npz"
        if not st.exists(rel):
            return None
        arrs, meta = st.load_raster(rel)
        u = self.m9.m7.unit(f"{task}:{zone}")
        if u is None:
            return None
        truth = np.asarray(u.eval_truth(), bool) & np.asarray(arrs["interior"], bool)
        known = np.asarray(u.known, bool) if u.known is not None else None
        return u, arrs, meta, truth, known

    def _overlay_figure(self):
        ev_c = self.cfg["evaluation"]
        f0 = float(ev_c["primary_area_frac"])
        for task in ev_c["tasks"]:
            for u in self.units(task):
                got = self._unit_arrays(u.task, u.zone)
                if got is None or not got[3].any():
                    continue
                u, arrs, meta, truth, known = got
                dom = np.asarray(arrs["domain"], bool)
                names = [n for n in self.main_names() if n in arrs and n != "random"]
                sels = {n: GM.equal_area_mask(np.asarray(arrs[n], np.float32), dom, f0) for n in names}
                rr, cc = np.nonzero(truth)
                half = int(self.cfg["figures"]["window_half_px"])
                wc = FG.window(dom.shape, (int(np.median(rr)), int(np.median(cc))), half)
                R0, C0 = u.sl[0].start, u.sl[1].start
                wf = (wc[0] + R0, wc[1] + R0, wc[2] + C0, wc[3] + C0)
                crop = lambda a: None if a is None else np.asarray(a)[wc[0]:wc[1], wc[2]:wc[3]]   # noqa: E731
                return FG.label_overlays(self.src.data.grid, wf, {n: crop(arrs[n]) for n in names},
                                         {n: crop(s) for n, s in sels.items()}, crop(truth), crop(known), f0)
        return None

    def _cases_figure(self, S: pd.DataFrame):
        """Best (highest recall) and worst (most truth, lowest recall) blocks of the strongest G01 candidate."""
        st = self._need()
        ev_c = self.cfg["evaluation"]
        f0, d0 = float(ev_c["primary_area_frac"]), float(ev_c["primary_distance_m"])
        B = st.load_table("blocks_all", "tables")
        if not len(B) or not len(S):
            return None
        s = S[(S.area_frac == f0) & (S.tol_m == d0) & S.candidate.isin(self.cfg["candidates"])]
        if not len(s) or s.f1.dropna().empty:
            return None
        cand = str(s.groupby("candidate").f1.mean().idxmax())
        b = B[(B.candidate == cand) & (B.area_frac == f0) & (B.tol_m == d0) & (B.n_truth >= 20)].copy()
        if not len(b):
            return None
        b["recall"] = b.n_truth_hit / b.n_truth
        n = int(self.cfg["figures"]["n_cases"])
        pick = [("success", r) for r in b.sort_values(["recall", "n_truth"], ascending=[False, False]).head(n).itertuples()]
        pick += [("failure", r) for r in b.sort_values(["recall", "n_truth"], ascending=[True, False]).head(n).itertuples()]
        py, px = self.src.pixel_m
        bpx = max(int(round(float(ev_c["block_km"]) * 1000.0 / (0.5 * (py + px)))), 8)
        items = []
        for lab, r in pick:
            got = self._unit_arrays(r.task, int(r.zone))
            if got is None:
                continue
            u, arrs, meta, truth, known = got
            br, bc = int(r.block) // 100000, int(r.block) % 100000
            R0, C0 = u.sl[0].start, u.sl[1].start
            wf = (br * bpx, (br + 1) * bpx, bc * bpx, (bc + 1) * bpx)
            wc = (max(wf[0] - R0, 0), min(wf[1] - R0, truth.shape[0]), max(wf[2] - C0, 0), min(wf[3] - C0, truth.shape[1]))
            if wc[1] <= wc[0] or wc[3] <= wc[2]:
                continue
            crop = lambda a: None if a is None else np.asarray(a, np.float32)[wc[0]:wc[1], wc[2]:wc[3]]   # noqa: E731
            dom = np.asarray(arrs["domain"], bool)
            sel = GM.equal_area_mask(np.asarray(arrs[cand], np.float32), dom, f0)
            items.append({"label": f"{lab} {r.task}:{r.zone} block {r.block}",
                          "window": (wc[0] + R0, wc[1] + R0, wc[2] + C0, wc[3] + C0), "score": crop(arrs[cand]),
                          "sel": crop(sel) > 0, "truth": crop(truth) > 0,
                          "known": (crop(known) > 0) if known is not None else None,
                          "M": crop(arrs.get("comp:M")), "G": crop(arrs.get("comp:G")),
                          "stats": {"recall": float(r.recall), "n_truth": int(r.n_truth), "n_pred": int(r.n_pred)}})
        return FG.cases(self.src.data.grid, items, cand)

    # =========================================================================== report
    def report(self) -> Path:
        st = self._need()
        res = self.results()
        ev_c = self.cfg["evaluation"]
        f0, d0 = float(ev_c["primary_area_frac"]), float(ev_c["primary_distance_m"])
        S = res["summary_pooled"]
        lines = [f"# G01 physics-informed wavelet fault discovery: run `{st.root.name}`", "",
                 f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `gems_bench.m9.research` ({G01_VERSION}). "
                 "Template and interpretation guide: `docs/m9/G01_GEOLOGICAL_WAVELET.md`.", "",
                 "> Numbers below are produced by this run. The held-out zones were inspected in earlier GEMS work: "
                 "they are not an untouched test set. Evidence scores are not calibrated probabilities. This "
                 "evaluation is separate from the official competition score.", ""]
        if self.warnings:
            lines += ["## Warnings", ""] + [f"* {w}" for w in self.warnings] + [""]
        lines += ["## Inputs", "", f"* wavelengths: {', '.join(w.label for w in self.lams)}",
                  f"* M9 cross-modal layers (G01-F): {', '.join(self.xmod) or 'none'}",
                  f"* primary budget {100 * f0:g} % of the domain, primary distance {d0:g} m", ""]
        H = res["hypothesis_contrasts"]
        if len(H):
            cols = [c for c in ("hypothesis", "task", "candidate", "reference", "metric", "delta", "ci_lo", "ci_hi",
                                "p_delta_positive", "zone_win_share", "verdict") if c in H]
            lines += ["## Hypothesis contrasts (paired spatial-block bootstrap)", "", _md(H[cols]), ""]
        if len(S):
            s = S[(S.area_frac == f0) & (S.tol_m == d0) & S.candidate.isin(self.main_names())]
            cols = [c for c in ("task", "candidate", "precision", "recall", "f1", "enrichment", "precision_vs_edge",
                                "f1_lo", "f1_hi") if c in s]
            lines += ["## Pooled held-out results at the primary budget / distance", "", _md(s[cols]), ""]
        N = res["nested_choice"]
        if len(N):
            lines += ["## Nested (leave-one-zone-out) choices among ablation variants", "", _md(N), ""]
        Op = res["official_style_pooled"]
        if len(Op):
            lines += ["## Official-style DTI of the same binary masks (orientation only)", "", _md(Op), ""]
        lines += ["## Interpretation (to be written after reading the tables and figures 9 and 12)", "",
                  "* Which contrasts are SUPPORTED / REFUTED, on which task, at which distances?",
                  "* Do gains survive at 0-100 m or only at >= 500 m (proximity vs placement)?",
                  "* Where do failures concentrate (lithological contacts, intrusions, data edges)?", ""]
        p = st.path("reports/G01_REPORT.md")
        from ...artifacts.atomic import atomic_write_text

        atomic_write_text(p, "\n".join(lines))
        st.record("reports/G01_REPORT.md", "report_md", ["tables/*"], {})
        st.mark_done("report")
        st.flush_manifest()
        self.push()
        return p

    def run(self) -> dict:
        out = {"prepare": self.prepare()}
        out["units"] = self.evaluate()
        out["exploratory"] = bool(self.exploratory() is not None)
        out["metrics"] = self.metrics()
        out["figures"] = self.figures()
        out["report"] = self.report()
        return out


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    """Plain Markdown table (no tabulate dependency)."""
    if not len(df):
        return "_(empty)_"
    cols = list(df.columns)

    def f(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.{digits}g}"
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(out)


__all__ = ["G01Experiment", "CONTRASTS", "CANDIDATES", "BASELINES"]
