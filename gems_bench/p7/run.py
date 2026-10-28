"""P7 orchestration: ``P7GeoFaultFormer(p6)`` -> one method per stage, each restartable from persisted outputs.

    m9 = pipe.m9(M9_CONFIG); m9.load(); p6 = P6FastTrack(m9, P6_CONFIG)
    p7 = P7GeoFaultFormer(p6, P7_CONFIG)
    p7.preflight(); p7.prepare_features(); p7.train("T1"); p7.evaluate(["T1"]); p7.candidate_recall(["T1"])
    p7.tracer("T4"); p7.w2_integration(); p7.full_inference("T1"); p7.selection(); p7.export(); p7.report()

Composition, not duplication: M9 provides the context, units, cube, scalers and targets; P6 boost provides the
official scoring calls, the B0 / B1 / fusion / W2 references and the eligible budget; P6 export_parity provides the
single validated GeoTIFF writer. Everything P7 writes lands under ``outputs/p7_geofaultformer`` (local, then Drive);
nothing under M9 / P6 is rewritten. Experiment directories are named by their configuration fingerprint
(``p7_<name>_<fp10>``), so a changed configuration never overwrites an earlier experiment.

Stage status files: ``manifests/status/<stage>.json`` with PENDING / BLOCKED / RUNNING / COMPLETE / INTERRUPTED /
FAILED, the fingerprint, inputs, source commit and the reason when not COMPLETE.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..m7.workspace import M7Workspace
from ..pipeline.m6 import fingerprint
from .config import (P7_CODE_REV, P7_VERSION, experiment_id, gpu_preset, load_p7_config, micro_split,
                     resolve_experiment, semantic_view, trained_experiments)

HYPOTHESES = {"H1": ("T1", "T1_cnn"), "H2": ("T2", "T1"), "H2-ablation": ("T2c", "T1"), "H3": ("T3", "T2")}
REFS = ("cnn_b0", "cnn_b1", "fallback_fusion", "w2")


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:                                   # tabulate absent
        return "```\n" + df.to_string(index=False) + "\n```"


class P7Workspace(M7Workspace):
    SUBDIRS = ("manifests", "manifests/units", "manifests/status", "environment", "folds", "features", "experiments",
               "comparisons", "candidate_recall", "stacker", "selection", "submissions", "reports", "figures")


class _Shim:
    """The attributes ``export_parity.write_validated`` / ``M7View`` read, served by P7 (writes land in P7)."""

    def __init__(self, p7):
        from ..m8.run import M7View
        from ..p6.run import _P6Like

        self.m9, self.ws, self.ctx = p7.m9, p7.ws, p7.ctx
        self._p7 = p7
        self.view = M7View(_P6Like(self))

    def push(self):
        return self._p7.push()


class P7GeoFaultFormer:
    def __init__(self, p6, config: dict | None = None, *, experiments: dict | None = None, gpu: dict | None = None,
                 boost_config: dict | None = None):
        self.p6, self.m9 = p6, p6.m9
        self.cfg = load_p7_config(config, experiments=experiments, gpu=gpu)
        self.ws = P7Workspace(Path(self.m9.local) / self.cfg["output_dir"], self.cfg)
        self._boost_config = boost_config
        self._bst = None
        self._memo: dict = {}

    # ------------------------------------------------------------------------------- context
    @property
    def ctx(self):
        return self.m9.ctx

    @property
    def prim(self) -> str:
        return self.m9.prim

    @property
    def bst(self):
        if self._bst is None:
            self._bst = self.p6.boost(self._boost_config)
        return self._bst

    def say(self, msg: str) -> None:
        print(f"[p7] {msg}")

    def fp(self, *parts) -> str:
        return fingerprint(P7_VERSION, *parts)

    def read_json(self, rel: str) -> dict | None:
        p = self.ws.root / rel
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def save_json(self, obj, rel: str):
        from ..p6.boost import _js

        return self.ws.save_json(json.loads(json.dumps(obj, default=_js)), rel)

    def stage_in(self, paths: list) -> None:
        rels = []
        for p in paths:
            try:
                rels.append(Path(p).resolve().relative_to(Path(self.m9.local).resolve()).as_posix())
            except ValueError:
                continue
        self.m9.stage_in(rels)

    def push(self) -> None:
        if not self.cfg.get("push_to_drive", True) or self.m9.pc is None:
            return
        try:
            self.m9.pc.push([Path(self.ws.root).resolve().relative_to(Path(self.m9.local).resolve()).as_posix()])
        except Exception as e:
            self.say(f"push skipped: {type(e).__name__}: {e}")

    def attach(self) -> dict:
        """Stage earlier P7 outputs and the P6 / M9 inputs P7 reads (read-only) from Drive."""
        self.m9.stage_in([self.cfg["output_dir"],
                          f"{self.cfg['p6_output_dir']}/stack", f"{self.cfg['p6_output_dir']}/boost",
                          f"{self.cfg['m9_output_dir']}/features", f"{self.cfg['m9_output_dir']}/spectral",
                          f"{self.cfg['m9_output_dir']}/baseline", f"{self.cfg['m9_output_dir']}/datasets/cv"])
        st = {p.stem: json.loads(p.read_text(encoding="utf-8")).get("status")
              for p in sorted((self.ws.root / "manifests" / "status").glob("*.json"))}
        return {"root": str(self.ws.root), "stages": st}

    # ------------------------------------------------------------------------------- experiments
    def exp_dir(self, exp_id: str) -> Path:
        return self.ws.root / "experiments" / exp_id

    @staticmethod
    def exp_sub(exp_id: str, kind: str) -> str:
        return f"experiments/{exp_id}/{kind}"

    def experiment_fp(self, name: str) -> str:
        ecfg = resolve_experiment(self.cfg, name)
        inp = self.inputs()
        from .train import P7_TRAIN_CONTRACT

        return self.fp(P7_TRAIN_CONTRACT, semantic_view(ecfg), inp["meta"]["key"], inp["extra_key"], self.cfg["scope"])

    def experiment_id(self, name: str) -> str:
        return experiment_id(name, self.experiment_fp(name))

    def registry(self) -> pd.DataFrame:
        return self.ws.load_table("registry", "experiments")

    def _register(self, name: str, exp_id: str, status: str, **info) -> None:
        R = self.registry()
        row = {"experiment": name, "experiment_id": exp_id, "status": status, "code_rev": P7_CODE_REV,
               "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: json.dumps(v, default=str) if isinstance(v, (dict, list))
                                                                 else v for k, v in info.items()}}
        R = pd.concat([R[R.experiment_id != exp_id] if len(R) else R, pd.DataFrame([row])], ignore_index=True)
        self.ws.save_table(R, "registry", "experiments")

    def latest_id(self, name: str, status: str | None = "COMPLETE") -> str | None:
        """Experiment id of ``name`` under the CURRENT configuration (fingerprint), if registered with ``status``."""
        try:
            eid = self.experiment_id(name)
        except Exception:
            return None
        R = self.registry()
        if not len(R):
            return None
        r = R[(R.experiment_id == eid) & ((R.status == status) if status else True)]
        return eid if len(r) else None

    # ------------------------------------------------------------------------------- stage plumbing
    def set_status(self, stage: str, status: str, **info) -> None:
        from .._version import provenance

        self.save_json({"stage": stage, "status": status, "p7_version": P7_VERSION, "code_rev": P7_CODE_REV,
                        "source_commit": provenance().get("git_commit"),
                        "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}, f"manifests/status/{stage}.json")

    def _stage(self, stage: str, fn, **info):
        from ..training.trainer import TrainingInterrupted

        self.set_status(stage, "RUNNING", **info)
        t0 = time.perf_counter()
        try:
            out = fn()
        except TrainingInterrupted as e:
            self.set_status(stage, "INTERRUPTED", reason=str(e), seconds=round(time.perf_counter() - t0, 1), **info)
            self.push()
            self.say(f"{stage} INTERRUPTED (resumable): {e}")
            return None
        except Exception as e:
            self.set_status(stage, "FAILED", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:],
                            seconds=round(time.perf_counter() - t0, 1), **info)
            self.push()
            raise
        st = out.get("status", "COMPLETE") if isinstance(out, dict) else "COMPLETE"
        self.set_status(stage, st, seconds=round(time.perf_counter() - t0, 1), reason=(out or {}).get("reason", "")
                        if isinstance(out, dict) else "", **info)
        self.push()
        return out

    def blocked(self, stage: str, reason: str) -> dict:
        self.set_status(stage, "BLOCKED", reason=reason)
        self.say(f"{stage} BLOCKED: {reason}")
        return {"status": "BLOCKED", "reason": reason}

    # ------------------------------------------------------------------------------- runtime / inputs
    def runtime(self, ecfg: dict | None = None, require: bool | None = None) -> dict:
        import torch

        from ..p6 import gpu_runtime as GR
        from .train import amp_dtype

        g = self.cfg["gpu"]
        info = GR.detect()
        req = bool(g["require_gpu"]) if require is None else bool(require)
        dev_s = GR.require_gpu(info, {"required": req, "min_free_gb": float(g["min_free_gb"])})
        device = torch.device(dev_s or "cpu")
        name, pre = gpu_preset(g, info)
        eb = int((ecfg or self.cfg)["training"]["effective_batch"])
        micro, acc = micro_split(eb, int(pre["micro_batch"]))
        return {"device": device, "preset": name, "micro_batch": micro, "grad_accum": acc,
                "infer_batch": int(pre["infer_batch"]), "amp_dtype": amp_dtype(pre.get("amp_dtype"), device),
                "amp_dtype_name": pre.get("amp_dtype"), "oom_retries": int(g["oom_retries"]),
                "ckpt_sync_min": float(g["ckpt_sync_min"]), "stride_frac": float(self.cfg["inference"]["stride_frac"]),
                "gpu": info}

    def deadline(self) -> float | None:
        m = float(self.cfg["gpu"].get("max_runtime_min") or 0)
        return time.monotonic() + 60.0 * m if m > 0 else None

    def inputs(self) -> dict:
        """Cube (M9 variant's channel groups) + optional extra layers + channel registry (contract asserted)."""
        if "inputs" in self._memo:
            return self._memo["inputs"]
        import hashlib

        from .data import assert_contract, build_extra, channel_registry, extra_specs, modality_groups

        self.m9.load()
        vcfg = self.m9.variant_cfg(self.cfg["inputs"]["m9_variant"])
        meta, cube, _ = self.m9.cube_for(vcfg)
        specs = extra_specs(self.cfg)
        extra = build_extra(self, specs)
        reg = channel_registry(list(meta["local"]), specs, self.cfg["inputs"]["modality_patterns"])
        assert_contract(reg)
        self._memo["inputs"] = {"meta": meta, "cube": cube, "extra": extra, "registry": reg,
                                "groups": modality_groups(reg), "extra_names": [s["name"] for s in specs],
                                "extra_key": hashlib.sha1(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:12]
                                if specs else None}
        return self._memo["inputs"]

    # =========================================================================== stages
    def preflight(self, profile_memory: bool = True) -> dict:
        """Environment, input availability, fold / leakage audits, channel contract, memory profile (synthetic)."""
        def run():
            from .._version import provenance
            from ..p6 import gpu_runtime as GR
            from .data import fold_provenance, overlap_audit, p7_units

            info = GR.detect()
            name, pre = gpu_preset(self.cfg["gpu"], info)
            env = {"gpu": info, "preset": name, "preset_cfg": pre, "dependencies": GR.dependency_record(),
                   "memory": GR.memory_report(), "code": provenance(), "p7_version": P7_VERSION,
                   "code_rev": P7_CODE_REV}
            inp = self.inputs()
            self.ws.save_table(inp["registry"], "channel_registry", "features")
            units = p7_units(self)
            OV = overlap_audit(self, units)
            self.ws.save_table(OV, "leakage_overlap_audit", "folds")
            self.ws.save_table(pd.DataFrame([fold_provenance(t) for t in units]), "fold_provenance", "folds")
            avail = []
            for u in self.bst.units():
                row = {"unit": u.key}
                try:
                    self.bst.cnn(u)
                    row["b0_b1_honest"] = True
                except FileNotFoundError as e:
                    row["b0_b1_honest"], row["note"] = False, str(e)
                row["w2_map"] = (self.p6.ws.root / "stack" / "maps" / f"{u.task}_{u.zone}.npy").exists()
                avail.append(row)
            AV = pd.DataFrame(avail)
            self.ws.save_table(AV, "reference_availability", "environment")
            prof = []
            if profile_memory and info.get("cuda_available"):
                import torch

                from . import model as MD
                from .train import amp_dtype

                dev = torch.device(info["device"])
                ch = list(inp["meta"]["local"]) + inp["extra_names"]
                for n in trained_experiments(self.cfg):
                    e = resolve_experiment(self.cfg, n)
                    micro, _ = micro_split(int(e["training"]["effective_batch"]), int(pre["micro_batch"]))
                    try:
                        spec = MD.net_spec(ch, inp["groups"], e, inp["meta"]["key"])
                        prof.append({"experiment": n,
                                     **MD.profile_memory(spec, micro, dev, amp_dtype(pre.get("amp_dtype"), dev))})
                    except Exception as ex:                       # OOM or kernel error: recorded, not hidden
                        prof.append({"experiment": n, "error": f"{type(ex).__name__}: {ex}"})
                        torch.cuda.empty_cache()
            PR = pd.DataFrame(prof)
            self.ws.save_table(PR, "memory_profile", "environment")
            blockers = []
            if not OV.ok.all():
                blockers.append(f"leakage overlap audit failed for {list(OV[~OV.ok].unit)}")
            if len(AV) and not AV.b0_b1_honest.all():
                blockers.append("honest B0/B1 maps missing for some units (references incomplete)")
            if len(AV) and not AV.w2_map.all():
                blockers.append("frozen W2 held-out maps missing (W2 reference / T5 blocked)")
            env.update(blockers=blockers, n_channels=int(len(inp["registry"])),
                       modalities=inp["registry"].modality.value_counts().to_dict(), n_units=len(units))
            self.save_json(env, "environment/preflight.json")
            return {"status": "COMPLETE" if not blockers else "COMPLETE_WITH_BLOCKERS", "blockers": blockers,
                    "env": env, "overlap": OV, "availability": AV, "memory": PR, "reason": "; ".join(blockers)}

        return self._stage("preflight", run)

    def prepare_features(self) -> dict:
        """Cube + extra layers, feature registry snapshot, target audit, label-free geographic shift per geo unit."""
        def run():
            from .data import geographic_shift, p7_units
            from .targets import audit_targets, build_p7_targets

            inp = self.inputs()
            self.ws.save_table(inp["registry"], "channel_registry", "features")
            self.save_json({"cube": {k: v for k, v in inp["meta"].items() if k not in ("cube", "pooled")},
                            "extra": inp["extra_names"], "extra_key": inp["extra_key"], "groups": inp["groups"]},
                           "features/feature_contract.json")
            units = p7_units(self)
            base = resolve_experiment(self.cfg, "T1")
            TA, SH = [], []
            for tu in units:
                t = build_p7_targets(self.m9, tu, base)
                G = self.ctx.splits.withheld(tu.zone) if tu.kind == "disc" else None
                TA.append(audit_targets(tu, t, G))
                if tu.kind == "cv":
                    s = geographic_shift(inp["cube"], list(inp["meta"]["local"]), tu.allowed, tu.predict,
                                         int(self.cfg["evaluation"]["shift_sample_px"]), int(self.cfg["seed"]))
                    SH.append(s.assign(unit=tu.name))
            TA = pd.DataFrame(TA)
            self.ws.save_table(TA, "target_audit", "folds")
            self.ws.save_table(pd.concat(SH, ignore_index=True) if SH else pd.DataFrame(), "geographic_shift",
                               "features")
            bad = list(TA[~TA.ok].unit) if len(TA) else []
            return {"status": "COMPLETE" if not bad else "BLOCKED", "reason": f"target audit failed: {bad}" if bad else "",
                    "target_audit": TA}

        return self._stage("prepare_features", run)

    def train(self, name: str) -> dict | None:
        """Train every unit of one rung (geo units first; discovery units with the predeclared fixed epochs)."""
        ecfg = resolve_experiment(self.cfg, name)
        if ecfg["experiment"]["kind"] != "train":
            raise ValueError(f"{name} is a {ecfg['experiment']['kind']} rung, not trainable")
        ov = (self.cfg["ladder"]["experiments"][name].get("overrides") or {})
        if "inputs" in ov:
            raise ValueError(f"{name}: inputs are shared by the ladder (first controlled comparison); not overridable")
        exp_id = self.experiment_id(name)

        def run():
            from ..p6.gpu_runtime import seed_everything
            from .data import p7_units
            from .model import build, net_spec, param_breakdown
            from .train import discovery_epochs, train_unit

            inp = self.inputs()
            rt = self.runtime(ecfg)
            seeds = seed_everything(int(ecfg["seed"]), bool(ecfg["training"]["deterministic"]))
            ch = list(inp["meta"]["local"]) + inp["extra_names"]
            spec = net_spec(ch, inp["groups"], ecfg, inp["meta"]["key"])
            man = {"experiment": name, "experiment_id": exp_id, "config": ecfg, "semantic": semantic_view(ecfg),
                   "net_spec": spec, "params": param_breakdown(build(spec)), "runtime": {k: str(v) for k, v in rt.items()
                                                                                         if k != "gpu"},
                   "gpu": rt["gpu"], "seeds": seeds, "registry_sha": inp["meta"]["key"], "extra_key": inp["extra_key"]}
            self.save_json(man, f"{self.exp_sub(exp_id, 'manifests')}/experiment.json")
            self._register(name, exp_id, "RUNNING", hypothesis=ecfg["experiment"]["hypothesis"])
            units = p7_units(self)
            dl = self.deadline()
            recs = [train_unit(self, exp_id, ecfg, tu, inp, rt, deadline=dl) for tu in units if tu.kind == "cv"]
            ep = discovery_epochs(recs, int(ecfg["training"]["min_epochs"]), int(ecfg["training"]["epochs"]))
            recs += [train_unit(self, exp_id, ecfg, tu, inp, rt, fixed_epochs=ep, deadline=dl)
                     for tu in units if tu.kind == "disc"]
            R = pd.DataFrame(recs)
            self.ws.save_table(R, "units", self.exp_sub(exp_id, "manifests"))
            n_scope = len(units)
            status = "COMPLETE" if len(R) == n_scope else "INCOMPLETE"
            self._register(name, exp_id, status, n_units=len(R), disc_epochs=ep,
                           n_params=int(R.n_params.iloc[0]) if len(R) else None)
            return {"status": status, "units": R, "experiment_id": exp_id, "disc_epochs": ep}

        out = self._stage(f"train_{name}", run, experiment_id=exp_id)
        if out is None:
            self._register(name, exp_id, "INTERRUPTED")
        return out

    def evaluate(self, names: list[str] | None = None) -> dict:
        """Held-out maps -> official grid -> nested summaries, paired deltas, hypothesis verdicts (vs references)."""
        def run():
            from . import evaluate as EV

            names_ = names or [n for n in trained_experiments(self.cfg) if self.latest_id(n)]
            R = EV.reference_grid(self)
            SR, PR = EV.summarize(self, R, list(REFS))
            self.ws.save_table(SR, "reference_summary", "comparisons")
            self.ws.save_table(PR, "reference_per_zone", "comparisons")
            S_all, P_all, D_all, M_all = [SR], [PR], [], []
            for n in names_:
                eid = self.latest_id(n)
                if eid is None:
                    S_all.append(pd.DataFrame([{"policy": f"p7_{n}", "status": "NOT_TRAINED"}]))
                    continue
                from .data import p7_units

                ecfg = resolve_experiment(self.cfg, n)
                rt = self.runtime(ecfg, require=False)
                EV.heldout(self, eid, ecfg, p7_units(self), rt)
                G = EV.grid(self, eid, n)
                cands = sorted(set(G.candidate)) if len(G) else []
                S, P = EV.summarize(self, G, cands)
                S = S.assign(experiment=n, experiment_id=eid)
                self.ws.save_table(S, "summary", self.exp_sub(eid, "metrics"))
                self.ws.save_table(P, "per_zone", self.exp_sub(eid, "metrics"))
                self.ws.save_table(EV.unit_mass(self, eid, n), "prediction_mass", self.exp_sub(eid, "metrics"))
                GG = pd.concat([G, R], ignore_index=True)
                for c in cands:
                    pc = P[P.policy == c]
                    for ref in REFS:
                        D_all.append(EV.per_zone_deltas(pc, PR[PR.policy == ref], c, ref))
                        M_all.append(EV.matched_deltas(GG, c, ref, 1.0 if ref == "w2" else None))
                S_all.append(S)
                P_all.append(P)
            S = pd.concat(S_all, ignore_index=True)
            P = pd.concat(P_all, ignore_index=True)
            D = pd.concat(D_all, ignore_index=True) if D_all else pd.DataFrame()
            M = pd.concat(M_all, ignore_index=True) if M_all else pd.DataFrame()
            self.ws.save_table(S, "summary", "comparisons")
            self.ws.save_table(P, "per_zone", "comparisons")
            self.ws.save_table(D, "paired_zone_deltas", "comparisons")
            if len(M):
                self.ws.save_table(M.groupby(["candidate", "reference", "task"]).agg(
                    mean_delta=("delta", "mean"), win_share=("delta", lambda x: float((x > 0).mean())),
                    n_cells=("delta", "size")).reset_index(), "matched_cell_deltas", "comparisons")
            H = self._hypotheses(S, D)
            self.ws.save_table(H, "hypotheses", "comparisons")
            return {"summary": S, "per_zone": P, "deltas": D, "hypotheses": H}

        return self._stage("evaluate", run)

    def _hypotheses(self, S: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
        from .evaluate import hypothesis_verdict

        by = {r["policy"]: r for r in S.to_dict("records")} if len(S) else {}
        rows = []
        for h, (a, b) in HYPOTHESES.items():
            ca, cb = by.get(f"p7_{a}"), by.get(f"p7_{b}")
            zd = None
            if len(D) and ca is not None and cb is not None:
                pa = D[(D.candidate == f"p7_{a}") & (D.reference == "w2")][["task", "zone", "dti_cand"]]
                pb = D[(D.candidate == f"p7_{b}") & (D.reference == "w2")][["task", "zone", "dti_cand"]]
                zd = pa.merge(pb, on=["task", "zone"], suffixes=("_a", "_b"))
                zd = zd.assign(delta=zd.dti_cand_a - zd.dti_cand_b)
            v = hypothesis_verdict(ca if ca and ca.get("status") == "COMPLETE" else None,
                                   cb if cb and cb.get("status") == "COMPLETE" else None, zone_deltas=zd)
            rows.append({"hypothesis": h, "candidate": f"p7_{a}", "control": f"p7_{b}", **v})
        return pd.DataFrame(rows)

    def candidate_recall(self, names: list[str] | None = None) -> dict:
        """Recall at the configured area budgets + systems new vs each reference (H5 complementarity)."""
        def run():
            from .evaluate import recall_table, reference_grid

            ev = self.cfg["evaluation"]
            R = reference_grid(self)
            out, H5 = [], []
            for n in names or [n for n in trained_experiments(self.cfg) if self.latest_id(n)]:
                eid = self.latest_id(n)
                G = self.ws.load_table("grid", self.exp_sub(eid, "metrics")) if eid else pd.DataFrame()
                if not len(G):
                    continue
                cands = sorted(c for c in set(G.candidate) if "+fusion" not in c)
                T = recall_table(pd.concat([G, R], ignore_index=True), cands, list(ev["recall_budgets"]), self.prim,
                                 {"w2": 1.0, "fallback_fusion": None, "cnn_b0": None, "cnn_b1": None})
                self.ws.save_table(T.assign(experiment_id=eid), f"recall_{eid}", "candidate_recall")
                b = float(self.cfg["promotion"]["complementarity"]["budget"])
                tb = T[(T.candidate == f"p7_{n}") & np.isclose(T.frac, b)] if len(T) else T
                new = int(tb.new_vs_w2.max()) if len(tb) and "new_vs_w2" in tb else None
                H5.append({"experiment": n, "experiment_id": eid, "budget": b, "new_systems_vs_w2": new,
                           "verdict": "INCOMPLETE" if new is None else ("COMPLEMENTARY" if new >= int(
                               self.cfg["promotion"]["complementarity"]["min_new_systems"]) else "NOT_COMPLEMENTARY")})
                out.append(T)
            H = pd.DataFrame(H5)
            self.ws.save_table(H, "h5_complementarity", "candidate_recall")
            return {"tables": out, "h5": H}

        return self._stage("candidate_recall", run)

    def tracer(self, name: str = "T4") -> dict:
        """T4: fit FaultTraceGRU per held-out disc unit (cross-fitted), roll out from the parent's policy-map
        endpoints, compare with geometric linking and the matched-area base map."""
        e = self.cfg["ladder"]["experiments"][name]
        parent = e.get("parent") or "T2"
        pid = self.latest_id(parent)
        if pid is None:
            return self.blocked(f"tracer_{name}", f"parent {parent} not trained / evaluated under this configuration")

        def run():
            from . import tracer as TR
            from .evaluate import load_pred

            tc = self.cfg["tracer"]
            rt = self.runtime(require=False)
            dev = rt["device"]
            import torch

            P = self.ws.load_table("per_zone", self.exp_sub(pid, "metrics"))
            cand = f"p7_{parent}"
            units = [u for u in self.bst.disc_units()]
            ev_of, traj_of = {}, {}
            from ..geology.skeleton import skeletonize
            from ..p6.stack import pf_full, rank_in

            pf = pf_full(self.p6)
            for u in units:
                pr = load_pred(self, pid, u)
                if pr is None:
                    continue
                cen = pr.get("center", pr["seg"])
                unc = pr.get("seg_std", 1.0 - np.hypot(pr.get("ocos", 0 * cen), pr.get("osin", 0 * cen)))
                layers = [cen, pr["seg"], rank_in(np.asarray(pf["mag_hgrad"])[u.sl], u.ev),
                          rank_in(np.asarray(pf["grav_hgrad"])[u.sl], u.ev), np.zeros_like(cen), np.asarray(unc)]
                ev_of[u.key] = np.nan_to_num(np.stack(layers).astype(np.float32))
                traj_of[u.key] = TR.skeleton_paths(skeletonize(u.eval_truth()))
            rows, hist_all, figs = [], [], {}
            for u in units:
                if u.key not in ev_of:
                    continue
                zrow = P[(P.policy == cand) & (P.task == u.task) & (P.zone == u.zone)]
                if not len(zrow):
                    continue
                cell = zrow.iloc[0]
                s = load_pred(self, pid, u)["seg"]
                from ..gems3 import representation as R

                rk = R.rank01(np.nan_to_num(s), u.ev)
                base = self.bst.binary(u, rk, cell.rep, float(cell.frac))
                ev = ev_of[u.key].copy()
                ev[4] = base.astype(np.float32)
                full_ex = np.zeros(self.ctx.data.shape, bool)
                full_ex[u.sl] = u.ev
                from scipy import ndimage

                buf = ndimage.distance_transform_edt(~full_ex) * 0.1 <= float(tc["train"]["exclude_buffer_km"])
                withheld_u = set(np.unique(np.asarray(self.ctx.splits.system_map)[self.ctx.splits.withheld(u.zone)]))
                data = []
                for v in units:
                    if v.key == u.key or v.key not in ev_of:
                        continue
                    sysm = np.asarray(self.ctx.splits.system_map)[v.sl]
                    drop = buf[v.sl] | np.isin(sysm, list(withheld_u - {-1}))
                    trajs = TR.trajectory_set(traj_of[v.key], float(tc["step_px"]), int(tc["train"]["min_traj_px"]), drop)
                    if trajs:
                        evv = ev_of[v.key].copy()
                        evv[4] = 0.0
                        data.append((torch.from_numpy(evv).to(dev), trajs))
                if not data:
                    rows.append({"unit": u.key, "status": "no training trajectories"})
                    continue
                model = TR.make_tracer(ev.shape[0], int(tc["embed_px"]), int(tc["hidden"])).to(dev)
                hist = TR.fit(model, data, tc, dev, self.say)
                hist_all += [{**h, "unit": u.key} for h in hist]
                model.eval()
                evt = torch.from_numpy(ev).to(dev)
                support = np.maximum(ev[0], ev[1])
                paths, reasons, unsup = [], [], 0.0
                for pos, hd in TR.endpoint_seeds(base):
                    p, why, us = TR.rollout(TR.model_step_fn(model, evt, dev, tc), pos, hd, support, u.ev, u.known, tc)
                    if len(p) > 1:
                        paths.append(p)
                        reasons.append(why)
                        unsup += us
                added = np.zeros(base.shape, bool)
                for p in paths:
                    added |= TR.rasterize(p, base.shape) & u.ev
                added &= ~base
                link = TR.geometric_link(base, u.ev, float(tc["linking"]["max_gap_px"]),
                                         float(tc["linking"]["max_angle_deg"]))
                el = self.bst.elig(u)
                fr_m = float(((base | added) & el).sum() / max(int(el.sum()), 1))
                matched = self.bst.binary(u, rk, cell.rep, fr_m)
                for kind, q in (("base", base), ("traced", base | added), ("geometric_link", base | link),
                                ("matched_area_base", matched)):
                    r = self.bst._eval_q(u, q.astype(np.float32), candidate=f"{cand}:{kind}", rep=cell.rep,
                                         frac=float(cell.frac), blend=1.0)
                    r.update(kind=kind, added_px=int((q & ~base).sum()),
                             unsupported_px=float(unsup) if kind == "traced" else np.nan,
                             n_paths=len(paths) if kind == "traced" else np.nan,
                             stop_reasons=json.dumps(pd.Series(reasons).value_counts().to_dict()) if kind == "traced"
                             else "")
                    if kind != "base" and u.eval_truth().any():
                        dt = ndimage.distance_transform_edt(~u.eval_truth())
                        a = q & ~base
                        r["added_mean_dist_to_truth_px"] = float(dt[a].mean()) if a.any() else np.nan
                    rows.append(r)
                figs[u.key] = (ev[0], base, paths, reasons, u.ev)
                from ..artifacts.atomic import atomic_torch_save

                atomic_torch_save(model.state_dict(), self.exp_dir(pid) / "checkpoints" / f"tracer_{u.task}_{u.zone}.pt")
                u.release()
            T = pd.DataFrame(rows)
            sub = self.exp_sub(pid, "tracer")
            self.ws.save_table(T, "tracer_grid", sub)
            self.ws.save_table(pd.DataFrame(hist_all), "tracer_history", sub)
            self._memo["tracer_figs"] = figs
            verdict = self._tracer_verdict(T)
            self.save_json(verdict, f"{sub}/decision.json")
            return {"status": "COMPLETE", "grid": T, "decision": verdict}

        return self._stage(f"tracer_{name}", run, parent_experiment_id=pid)

    def _tracer_verdict(self, T: pd.DataFrame) -> dict:
        from ..p6.boost import dti

        if not len(T) or "kind" not in T:
            return {"verdict": "INCOMPLETE"}
        g = T.groupby("kind")[["tp_w", "fp_w", "fn_w"]].sum()
        d = {k: float(dti(g.loc[k, "tp_w"], g.loc[k, "fp_w"], g.loc[k, "fn_w"])) for k in g.index}
        acc = self.cfg["tracer"]["accept"]
        gain = d.get("traced", np.nan) - max(d.get("matched_area_base", -np.inf), d.get("geometric_link", -np.inf))
        fp_inc = (g.loc["traced", "fp_w"] - g.loc["base", "fp_w"]) / max(g.loc["base", "fp_w"], 1e-9) \
            if {"traced", "base"} <= set(g.index) else np.nan
        ok = np.isfinite(gain) and gain >= float(acc["min_disc_gain"]) and fp_inc <= float(acc["max_fp_increase"])
        return {"pooled_dti": d, "gain_vs_best_control": gain, "fp_increase": fp_inc,
                "verdict": "ACCEPT" if ok else "REJECT",
                "note": "pooled over disc units with each unit's nested P7 cell (cells chosen on other zones)"}

    def w2_integration(self, name: str | None = None) -> dict:
        """T5: P7-enhanced W2 (schema w2_p7.v1) + matched W2 control; requires every unit's P7 held-out map."""
        from .evaluate import pred_rel

        best = name or self._best_trained()
        eid = self.latest_id(best) if best else None
        if eid is None:
            return self.blocked("w2_integration", "no completed, evaluated P7 rung under this configuration")
        miss = [u.key for u in self.bst.units() if not (self.ws.root / pred_rel(self, eid, u)).exists()]
        if miss:
            return self.blocked("w2_integration", f"P7 held-out maps missing for {miss} (no valid OOF provenance)")

        def run():
            from . import w2

            out = w2.run(self, eid, best)
            return {"status": "COMPLETE", **out}

        return self._stage("w2_integration", run, experiment_id=eid)

    def _best_trained(self) -> str | None:
        S = self.ws.load_table("summary", "comparisons")
        if not len(S) or "disc_dti" not in S:
            return None
        s = S[S.policy.astype(str).str.match(r"^p7_[A-Za-z0-9_]+$") & (S.status == "COMPLETE")
              & ~S.policy.astype(str).str.endswith("_center")]
        if not len(s):
            return None
        return str(s.sort_values("disc_dti", ascending=False).policy.iloc[0])[3:]

    def full_inference(self, name: str) -> dict:
        """Full-region P7 maps: zone_stitched (each geo zone from the model that held it out = the evaluated maps;
        pixels outside every fold from the fold average, flagged) or fold_average (diagnostic, never deployed)."""
        eid = self.latest_id(name)
        if eid is None:
            return self.blocked(f"full_inference_{name}", f"{name} not trained under this configuration")

        def run():
            from scipy.stats import spearmanr

            from .evaluate import load_pred
            from .infer import full_region, predict_mask
            from .train import load_unit

            con = self.cfg["inference"]["full_region"]["construction"]
            ecfg = resolve_experiment(self.cfg, name)
            rt = self.runtime(ecfg, require=False)
            inp = self.inputs()
            valid = np.asarray(self.ctx.data.valid, bool)
            fm = np.asarray(self.ctx.folds.fold_map)
            H, W = valid.shape
            out, par = {}, []
            P = int(ecfg["data"]["patch_px"])
            kw = dict(P=P, stride=max(int(P * rt["stride_frac"]), 1), device=rt["device"], amp=bool(ecfg["training"]["amp"]),
                      amp_dtype=rt["amp_dtype"], batch=int(rt["infer_batch"]))
            models = [load_unit(self, eid, f"geo_f{k}", rt["device"]) for k in self.ctx.fold_ids]
            models = [(m, s, e) for x in models if x is not None for (m, s, e) in [x[:3]]]
            if con == "zone_stitched":
                for u in self.m9.m7.units_of("geo"):
                    p = load_pred(self, eid, u)
                    if p is None:
                        raise FileNotFoundError(f"held-out map of {u.key} missing: run evaluate first")
                    for h, a in p.items():
                        o = out.setdefault(h, np.full((H, W), np.nan, np.float32))
                        sub = o[u.sl]
                        sub[u.ev] = a[u.ev]
                    s_full = out["seg"][u.sl][u.ev]
                    rho = float(spearmanr(s_full, p["seg"][u.ev]).statistic) if u.ev.sum() > 2 else np.nan
                    par.append({"zone": int(u.zone), "spearman_vs_eval": rho, "ok": bool(rho >= 0.999)})
                unc = valid & ~np.isfinite(out["seg"])
                if unc.any() and models:
                    fill = predict_mask(models, inp["cube"], inp["extra"], unc, **kw)
                    for h, a in fill.items():
                        if h in out:
                            out[h][unc] = a[unc]
                cover = {"uncovered_px": int(unc.sum()), "uncovered_frac": float(unc.sum() / max(valid.sum(), 1)),
                         "fill": "fold_average" if unc.any() else "none",
                         "fold_map_negative_px": int((valid & (fm < 0)).sum())}
            else:
                out = full_region(models, inp["cube"], inp["extra"], valid, self.exp_dir(eid) / "processed" / "tiles",
                                  tile_px=int(self.cfg["inference"]["tile_px"]),
                                  halo_px=int(self.cfg["inference"]["halo_px"]), log=self.say, **kw)
                cover = {"construction": "fold_average", "deployable": False}
            if "center" in out:
                from .infer import centerline_reconcile

                out["center_crest"] = centerline_reconcile(out["center"], valid).astype(np.float32)
            sub = self.exp_sub(eid, "predictions")
            for h, a in out.items():
                self.ws.save_raster({f"full_{h}": a}, f"{sub}/full_region_{h}.tif", self.ctx.data.grid)
            PT = pd.DataFrame(par)
            self.ws.save_table(PT, "full_region_parity", self.exp_sub(eid, "metrics"))
            man = {"experiment_id": eid, "construction": con, "coverage": cover,
                   "parity_ok": bool(len(PT) and PT.ok.all()) if con == "zone_stitched" else False,
                   "deployable": con == "zone_stitched"}
            self.save_json(man, f"{sub}/full_region_manifest.json")
            return {"status": "COMPLETE", **man}

        return self._stage(f"full_inference_{name}", run, experiment_id=eid)

    def selection(self) -> dict:
        """Predeclared promotion rules over every evaluated P7 candidate (and T5); W2 stays the champion otherwise."""
        def run():
            from .evaluate import promotion_verdict

            S = self.ws.load_table("summary", "comparisons")
            if not len(S):
                return {"status": "BLOCKED", "reason": "no comparisons/summary.csv: run evaluate first"}
            by = {r["policy"]: r for r in S.to_dict("records")}
            w2 = by.get("w2")
            H5 = self.ws.load_table("h5_complementarity", "candidate_recall")
            OV = self.ws.load_table("leakage_overlap_audit", "folds")
            TA = self.ws.load_table("target_audit", "folds")
            no_leak = bool(len(OV) and OV.ok.all() and len(TA) and TA.ok.all())
            rows = []
            for pol, r in by.items():
                if not str(pol).startswith("p7_") or r.get("status") != "COMPLETE":
                    continue
                name = str(pol)[3:].split("+")[0].replace("_center", "")
                eid = self.latest_id(name)
                man = self.read_json(f"{self.exp_sub(eid, 'predictions')}/full_region_manifest.json") if eid else None
                new = None
                if len(H5) and "experiment" in H5:
                    h = H5[H5.experiment == name]
                    new = int(h.new_systems_vs_w2.iloc[0]) if len(h) and pd.notna(h.new_systems_vs_w2.iloc[0]) else None
                dep = r.get("deployable")
                checks = {"no_leakage": no_leak, "schema_compatible": True, "parity": bool(man and man.get("parity_ok")),
                          "deployable_policy": str(dep).lower() == "true",
                          "export_valid": True}                    # validated by export(); pending here
                v = promotion_verdict(r, w2, self.cfg["promotion"], new, checks)
                if v["decision"] == "PROMOTE":
                    v["decision"] = "PROMOTE_PENDING_EXPORT"
                rows.append({"candidate": pol, "experiment_id": eid, "disc_dti": r.get("disc_dti"),
                             "disc_worst": r.get("disc_worst"), "geo_dti": r.get("geo_dti"), **v,
                             "reasons": "; ".join(v["reasons"]), "deploy_cell": r.get("deploy_cell")})
            T = pd.DataFrame(rows)
            self.ws.save_table(T, "candidates", "selection")
            win = T[T.decision == "PROMOTE_PENDING_EXPORT"].sort_values("disc_dti", ascending=False) if len(T) else T
            dec = {"winner": win.candidate.iloc[0] if len(win) else None,
                   "experiment_id": win.experiment_id.iloc[0] if len(win) else None,
                   "deploy_cell": win.deploy_cell.iloc[0] if len(win) else None,
                   "champion_if_none": "existing W2 (unchanged)", "auxiliary": list(T[T.decision == "AUXILIARY"].candidate)
                   if len(T) else [], "w2_reference": {k: (w2 or {}).get(k) for k in ("disc_dti", "disc_worst", "geo_dti")},
                   "rule": "configs/p7/geofaultformer.yaml promotion (predeclared)"}
            self.save_json(dec, "selection/decision.json")
            return {"status": "COMPLETE", "candidates": T, "decision": dec}

        return self._stage("selection", run)

    def export(self) -> dict:
        """Validated zone-stitched GeoTIFF of the selected candidate through ``export_parity.write_validated``."""
        dec = self.read_json("selection/decision.json") or {}
        if not dec.get("winner"):
            return self.blocked("export", "no candidate passed selection (W2 remains the champion)")

        def run():
            from ..artifacts.atomic import sha256_file
            from ..gems3 import representation as R
            from ..gems3 import submission as SUB
            from ..p6 import export_parity as EP
            from .evaluate import load_pred

            cell = dec["deploy_cell"]
            cell = json.loads(cell) if isinstance(cell, str) else cell
            eid, cand = dec["experiment_id"], dec["winner"]
            data = self.ctx.data
            el_full = SUB.eligible_mask(data, 0, "eligible")
            labels = np.asarray(data.labels, bool)
            sh = self.m9.sh
            rep = sh["baseline"]["representations"][cell["rep"]]
            q = np.zeros(data.shape, np.float32)
            srcs = {}
            for u in self.m9.m7.units_of("geo"):
                p = load_pred(self, eid, u)
                head = "center" if cand.endswith("_center") else "seg"
                s = R.rank01(np.nan_to_num(p[head]), u.ev)
                if "+fusion" in cand:
                    lam = float(cell["blend"])
                    s = ((1.0 - lam) * self.bst.fusion(u) + lam * s).astype(np.float32)
                el = np.asarray(el_full, bool)[u.sl] & u.ev
                qz = R.apply(rep, s, u.ev, float(cell["frac"]), known=labels[u.sl], sigma_px=float(
                    sh["evaluation"]["ridge_sigma_px"]), eligible=el)
                sub = q[u.sl]
                sub[u.ev] = np.maximum(sub[u.ev], qz[u.ev])
                srcs[u.key] = sha256_file(self.ws.root / f"{self.exp_sub(eid, 'predictions')}/units/{u.task}_{u.zone}.npz")
            rec = EP.build_recipe(candidate=cand, kind="p7_zone_stitched",
                                  policy={"selection_id": "p7_selection", "space": cand, "objective":
                                          self.cfg["evaluation"]["objective"], "rep": cell["rep"], "frac": cell["frac"],
                                          "blend": max(float(cell["blend"]), 0.0)},
                                  model={"experiment_id": eid, "features": list(self.inputs()["registry"].name)},
                                  sources=srcs, feature_schema=list(self.inputs()["registry"].name),
                                  preprocessing={"eligibility": "gems3.submission.eligible_mask(eligible), ring 0",
                                                 "known": "masked pixel-exactly (no positive on mapped faults)",
                                                 "construction": "zone_stitched per geographic fold"})
            pa = EP.save_plan(_Shim(self), rec, "submissions")
            if pa["status"] != "PASS":
                return {"status": "BLOCKED", "reason": f"export plan audit: {pa}"}
            info = EP.write_validated(_Shim(self), rec, q, [], 0.02)
            ok = (info.get("audit") or {}).get("status") == "PASS"
            final = {**dec, "export": {k: info.get(k) for k in ("submission", "reused")}, "audit": info.get("audit"),
                     "final_decision": "PROMOTE" if ok else "BLOCKED_EXPORT", "uploaded": False}
            self.save_json(final, "selection/final_decision.json")
            return {"status": "COMPLETE" if ok else "BLOCKED", "reason": "" if ok else "export audit failed", **final}

        return self._stage("export", run)

    def figures(self, name: str) -> dict:
        """Every figure of one experiment that its persisted outputs support (PNG + SVG + data CSV)."""
        from . import figures as FG
        from .evaluate import load_pred

        eid = self.latest_id(name)
        if eid is None:
            return {"status": "BLOCKED", "reason": f"{name} not trained"}
        sub = self.exp_sub(eid, "figures")
        made = {}

        def save(fname, res):
            fig, data = res if isinstance(res, tuple) else (res, None)
            made[fname] = "ABSENT" if fig is None else "SAVED"
            if fig is not None:
                self.ws.save_fig(fig, fname, data, sub)

        hs = []
        for p in sorted((self.ws.root / self.exp_sub(eid, "metrics")).glob("history_*.csv")):
            hs.append(pd.read_csv(p).assign(unit=p.stem.replace("history_", ""), experiment_id=eid))
        save("training_curves", FG.training_curves(pd.concat(hs, ignore_index=True) if hs else None))
        Pz = self.ws.load_table("per_zone", "comparisons")
        keep = Pz[Pz.policy.isin([f"p7_{name}", "w2", "fallback_fusion", "cnn_b0", "cnn_b1"])] if len(Pz) else Pz
        save("fold_dti", FG.fold_dti(keep))
        S = self.ws.load_table("summary", "comparisons")
        save("pareto", FG.pareto(S))
        rt = self.ws.load_table(f"recall_{eid}", "candidate_recall")
        save("recall_curves", FG.recall_curves(rt))
        save("coverage_by_budget", FG.coverage_by_budget(rt))
        from ..gems3 import representation as R

        dom = self.m9.domains()
        regime = []
        for u in self.bst.disc_units():
            p = load_pred(self, eid, u)
            if p is None:
                continue
            maps = {k: p[k] for k in ("seg", "center") if k in p}
            save(f"probability_maps_{u.task}_{u.zone}", FG.probability_maps(maps, u.ev, f"{eid} {u.key}"))
            save(f"centerline_comparison_{u.zone}", FG.centerline_comparison(p["seg"], p.get("center"), u.eval_truth(),
                                                                              u.ev))
            save(f"orientation_field_{u.zone}", FG.orientation_field(p.get("ocos"), p.get("osin"), p["seg"], u.ev))
            z = Pz[(Pz.policy == f"p7_{name}") & (Pz.task == u.task) & (Pz.zone == u.zone)] if len(Pz) else Pz
            if len(z) and pd.notna(z.rep.iloc[0]):
                rk = R.rank01(np.nan_to_num(p["seg"]), u.ev)
                q = self.bst.binary(u, rk, z.rep.iloc[0], float(z.frac.iloc[0]))
                save(f"heldout_examples_{u.zone}", FG.heldout_examples(q, u.eval_truth(), u.known, u.ev, u.key))
                save(f"error_maps_{u.zone}", FG.error_maps(q.astype(np.float32), u.eval_truth(), u.ev))
                w2r = self.bst.stack_rank(u, self.bst.w2_map(u))
                save(f"disagreement_w2_{u.zone}", FG.disagreement(rk, w2r, u.ev, (f"p7_{name}", "w2")))
                if dom is not None:
                    from ..gems3.official import FastScorer

                    dd = np.asarray(dom)[u.sl]
                    for d in sorted(set(np.unique(dd[u.ev])) - {-1}):
                        m = u.ev & (dd == d)
                        fs = FastScorer(u.truth & m, m, u.exclude, exclude_supports_tp=u.supports_tp)
                        c = fs.components(q.astype(np.float64))
                        regime.append({"candidate": f"p7_{name}", "unit": u.key, "domain": int(d), "dti": float(c.dti),
                                       "n_truth": int(fs.n)})
            u.release()
        save("regime_errors", FG.regime_errors(pd.DataFrame(regime)))
        tf = self._memo.get("tracer_figs") or {}
        for k, (evd, base, paths, reasons, evm) in tf.items():
            save(f"tracer_trajectories_{k.replace(':', '_')}", FG.tracer_trajectories(evd, base, paths, reasons, evm))
        save_internal = self._internal_figures(eid, name)
        for k, v in save_internal.items():
            save(k, v)
        self.save_json(made, f"{sub}/figure_index.json")
        self.push()
        return {"status": "COMPLETE", "figures": made}

    def _internal_figures(self, eid: str, name: str) -> dict:
        """Attention weights + saliency proxy and ConvGRU states on one held-out geo patch (needs torch)."""
        from . import figures as FG
        from .train import load_unit

        out = {}
        try:
            import torch

            rt = self.runtime(require=False)
            u = self.m9.m7.units_of("geo")[0]
            lu = load_unit(self, eid, f"geo_f{u.zone}", rt["device"])
            if lu is None:
                return out
            model, sc, esc, obj = lu
            inp = self.inputs()
            P = int(obj["net_spec"]["patch_px"])
            r, c = np.nonzero(u.ev)
            r0, c0 = int(np.median(r)) + u.sl[0].start - P // 2, int(np.median(c)) + u.sl[1].start - P // 2
            sl = (slice(max(r0, 0), max(r0, 0) + P), slice(max(c0, 0), max(c0, 0) + P))
            x = sc.apply(inp["cube"][:, sl[0], sl[1]])
            if inp["extra"] is not None:
                x = np.concatenate([x, esc.apply(inp["extra"][:, sl[0], sl[1]])], 0)
            xt = torch.from_numpy(np.ascontiguousarray(x[None])).to(rt["device"]).requires_grad_(True)
            model.set_capture(True)
            o = model(xt)
            o["seg"][0, 0, P // 2, P // 2].backward()
            sal = xt.grad.abs().mean(1)[0].cpu().numpy()
            attn, ws = None, None
            if len(model.attn):
                st = model.attn[sorted(model.attn)[-1]]
                w = st.blocks[0].attn.last_weights
                ws = st.blocks[0].ws
                if w is not None:
                    attn = w[len(w) // 2].numpy()
            out["attention_context"] = FG.attention_context(x[0], attn, sal, ws or 1, ((ws or 1) // 2, (ws or 1) // 2))
            if model.gru is not None:
                out["gru_refinement"] = FG.gru_refinement([s[0].numpy() for s in model.gru.states])
            model.set_capture(False)
        except Exception as e:
            self.say(f"internal figures skipped: {type(e).__name__}: {e}")
        return out

    def report(self) -> dict:
        """Markdown report of what exists (statuses, nested numbers, verdicts); absent stages are listed as such."""
        st = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in
              sorted((self.ws.root / "manifests" / "status").glob("*.json"))}
        S = self.ws.load_table("summary", "comparisons")
        H = self.ws.load_table("hypotheses", "comparisons")
        H5 = self.ws.load_table("h5_complementarity", "candidate_recall")
        dec = self.read_json("selection/final_decision.json") or self.read_json("selection/decision.json") or {}
        lines = ["# P7 GeoFaultFormer report", "",
                 "Every number below was produced by the Colab run that wrote these files. Inner checkpoint scores, "
                 "nested outer DTI and the public leaderboard are different quantities and are never mixed.", "",
                 "## Stage status", "", "| stage | status | reason |", "|---|---|---|"]
        lines += [f"| {k} | {v.get('status')} | {str(v.get('reason', ''))[:120]} |" for k, v in st.items()]
        cols = [c for c in ("policy", "status", "disc_dti", "disc_worst", "geo_dti", "system_recall", "disc_fp_w")
                if c in S]
        lines += ["", "## Nested summaries (official protocol)", ""]
        lines += [_md(S[cols]) if len(S) else "ABSENT (evaluate not run)"]
        lines += ["", "## Hypotheses", "", _md(H) if len(H) else "ABSENT"]
        lines += ["", "## H5 complementarity", "", _md(H5) if len(H5) else "ABSENT"]
        lines += ["", "## Selection", "", "```json", json.dumps(dec, indent=2, default=str)[:3000], "```"]
        md = "\n".join(lines)
        self.ws.save_text(md, "reports/p7_report.md")
        self.push()
        return {"markdown": md, "status": pd.DataFrame([{"stage": k, "status": v.get("status")} for k, v in st.items()])}
