"""G14 notebook orchestrator (Phase 14). Every side effect is gated by the run mode and an explicit flag.

    pipe = G14Pipeline(CONFIG)          # CONFIG: notebook dict, validated into G14Config
    pipe.inventory()                    # AUDIT_ONLY: read-only discovery of inputs and existing artifacts
    pipe.validate_data()                # VALIDATE_DATA+: build the label-free channel stack + contract checks
    pipe.build_splits()                 # VALIDATE_DATA+: units, audits, fold manifest
    pipe.train()                        # TRAIN + train_enabled
    pipe.predict_heldout()              # inference_enabled: honest held-out maps of every trained unit
    pipe.evaluate()                     # EVALUATE+: official components, nested selection, diagnostics
    pipe.submission(confirm=...)        # SUBMIT + submission_enabled + confirm token; manual upload only

Artifacts: ``<drive_root>/outputs/G14_GEOFAULTFORMER/experiments/<experiment>__<cfg_hash>/`` (checkpoints, manifests,
maps, evaluation) and ``.../shared/`` (channel stack manifest, fold manifests, inventory).

UNTESTED: written without execution.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import G14_VERSION
from .config import G14Config
from .experiments import REGISTRY, resolve

MODE_ORDER = {"AUDIT_ONLY": 0, "VALIDATE_DATA": 1, "TRAIN": 2, "EVALUATE": 2, "SUBMIT": 3}


class G14Pipeline:
    def __init__(self, config: dict | G14Config, log=print):
        base = config if isinstance(config, G14Config) else G14Config.from_dict(_flatten_notebook(config))
        self.cfg = resolve(base.experiment, base) if base.experiment in REGISTRY else base
        self.log = log
        self.local = Path(self.cfg.local_root)
        self.out = Path(self.cfg.out_root)
        self.exp_dir = self.local / "experiments" / f"{self.cfg.experiment}__{self.cfg.semantic_hash()}"
        self.shared = self.local / "shared"
        for p in (self.exp_dir, self.shared):
            p.mkdir(parents=True, exist_ok=True)
        self._data = self._paths = self._store = self._stack = self._system = None

    # ------------------------------------------------------------------ guards
    def _need(self, mode: str, flag: str | None = None):
        if MODE_ORDER[self.cfg.run_mode] < MODE_ORDER[mode] and self.cfg.run_mode != mode:
            raise PermissionError(f"needs run_mode >= {mode} (current {self.cfg.run_mode})")
        if flag and not getattr(self.cfg, flag):
            raise PermissionError(f"needs {flag}=True")

    # ------------------------------------------------------------------ inputs
    @property
    def paths(self) -> dict:
        if self._paths is None:
            from ..g13.contracts import discover_inputs

            self._paths = discover_inputs(self.cfg.drive_root)
        return self._paths

    @property
    def data(self):
        if self._data is None:
            from ..g13.contracts import load

            self._data = load(self.paths, self.local / "cache")
        return self._data

    @property
    def g13_store(self):
        if self._store is None:
            from ..g13.features import FeatureStore

            root = Path(self.cfg.data.g13_features_dir)
            if (root / "registry.csv").exists():
                self._store = FeatureStore(root, self.data)
            elif self.cfg.data.build_g13_features_if_missing:
                self._need("VALIDATE_DATA")
                from ..g13.config import G13Config
                from ..g13.features import build_store

                g13cfg = G13Config.from_mode("FEATURE_DISCOVERY", work_dir=str(root.parent), out_dir="")
                self._store = build_store(self.data, g13cfg, root, log=self.log)
        return self._store

    def g13_path(self, rel: str) -> Path | None:
        """First existing G13 artifact among ``data.g13_artifact_dirs`` (local work tree, then the Drive mirror)."""
        for base in self.cfg.data.g13_artifact_dirs:
            p = Path(base) / rel
            if p.exists():
                return p
        return None

    def inventory(self) -> pd.DataFrame:
        """Read-only: hardware, official inputs, reusable artifacts. Safe in AUDIT_ONLY."""
        from ..g13.contracts import hardware_report

        rows = [{"item": "g14_version", "status": G14_VERSION, "path": ""},
                {"item": "config_hash", "status": self.cfg.semantic_hash(), "path": str(self.exp_dir)}]
        for k, v in hardware_report().items():
            rows.append({"item": f"hw:{k}", "status": v, "path": ""})
        try:
            for role, p in self.paths.items():
                if role != "notes":
                    rows.append({"item": f"input:{role}", "status": "FOUND", "path": p})
        except Exception as e:  # noqa: BLE001
            rows.append({"item": "inputs", "status": f"MISSING: {e}", "path": self.cfg.drive_root})
        dr = Path(self.cfg.drive_root)
        g = lambda rel: self.g13_path(rel) or Path(self.cfg.data.g13_artifact_dirs[0]) / rel   # noqa: E731
        for item, rel in (("g13_features", Path(self.cfg.data.g13_features_dir) / "registry.csv"),
                          ("g13_splits", g("splits/params.json")),
                          ("g13_scoreboard", g("tables/scoreboard.csv")),
                          ("g13_regimes", g("regimes/regimes.npy")),
                          ("p6_w2_maps (drive)", dr / "outputs" / "p6_fast_track" / "stack" / "maps"),
                          ("p6_w2_candidate (drive)", dr / "outputs" / "p6_fast_track" / "stack" / "candidate.json"),
                          ("p7_outputs (drive)", dr / "outputs" / "p7_geofaultformer"),
                          ("g14_outputs (drive)", self.out)):
            p = Path(rel)
            rows.append({"item": item, "status": "FOUND" if p.exists() else "absent", "path": str(p)})
        df = pd.DataFrame(rows)
        df.to_csv(self.shared / "inventory.csv", index=False)
        return df

    # ------------------------------------------------------------------ data contract
    @property
    def stack(self):
        if self._stack is None:
            from .contract import ChannelStack

            st = ChannelStack(self.shared / "stack")
            self._stack = st if st.exists() else None
        return self._stack

    def validate_data(self) -> pd.DataFrame:
        self._need("VALIDATE_DATA")
        from .contract import build_stack, check_contract

        self._stack = build_stack(self.data, self.cfg, self.shared / "stack", self.g13_store, log=self.log)
        chk = check_contract(self._stack, self.data, self.cfg, self.paths)
        chk.to_csv(self.shared / "contract_check.csv", index=False)
        fails = chk[chk.status == "FAIL"]
        if len(fails):
            self.log(f"[g14] CONTRACT FAILED ({len(fails)} checks): see shared/contract_check.csv")
        return chk

    # ------------------------------------------------------------------ splits
    def _regimes(self):
        p = self.g13_path("regimes/regimes.npy")
        return np.load(p) if p is not None else None

    @property
    def system(self):
        if self._system is None:
            from .splits import UnitSystem

            if self.cfg.split.backend == "g13":
                from ..g13.config import G13Config
                from ..g13.splits import Splits, build_splits

                p = self.g13_path("splits/params.json")
                if p is not None:
                    sp = Splits.load(p.parent, self.data.valid, self.data.labels)
                else:
                    sp = build_splits(self.data, G13Config.from_mode("FAST_BASELINE"))
                self._system = UnitSystem("g13", self.data, g13_splits=sp, regimes=self._regimes(),
                                          embargo_m=self.cfg.split.embargo_m)
            else:
                from ..phase5_context import load_context5

                ctx = load_context5()
                self._system = UnitSystem("phase5", self.data, ctx=ctx, regimes=self._regimes(),
                                          embargo_m=self.cfg.split.embargo_m)
        return self._system

    def units(self):
        return self.system.units_for(self.cfg.split.protocol)

    def build_splits(self) -> pd.DataFrame:
        self._need("VALIDATE_DATA")
        from .splits import fault_system_groups, save_manifest, system_crossing_report

        units = self.units()
        self.system.assert_no_leakage(units)
        reach = int(self.cfg.data.regional.extent_m / 2 / self.cfg.data.local.resolution_m)
        audit = self.system.audit(units, context_radius_px=reach)
        sysmap, comps = fault_system_groups(self.data.labels, self.cfg.split.fault_link_m,
                                            self.cfg.split.fault_link_angle_deg)
        cross = system_crossing_report(sysmap, units)
        man = self.system.manifest(units, self.cfg.split.protocol, {
            "fault_grouping": {"link_m": self.cfg.split.fault_link_m, "angle_deg": self.cfg.split.fault_link_angle_deg,
                               "n_systems": int(sysmap.max() + 1)},
            "context": "label-free regional / global inputs may cover held-out geography; labels never; "
                       f"strict_context_embargo={self.cfg.split.strict_context_embargo}"})
        root = self.shared / "folds" / f"{self.cfg.split.backend}_{self.cfg.split.protocol}"
        save_manifest(man, audit, root)
        cross.to_csv(root / "system_crossing.csv", index=False)
        comps.to_csv(root / "fault_components.csv", index=False)
        np.save(root / "sysmap.npy", sysmap)
        return audit

    def preview(self, unit_index: int = 0, n: int = 4) -> list[dict]:
        """A few dataset samples of one unit (CPU, no training) for inspection."""
        self._need("VALIDATE_DATA")
        from .multiscale import MultiScaleDataset, UnitInputs, unit_targets

        u = self.units()[unit_index]
        inp = UnitInputs(self.stack, self.data, u, self.cfg)
        tg = unit_targets(u.labels & (u.train | u.inner_val), self.cfg.loss.kernel_radius_px)
        ds = MultiScaleDataset(inp, tg, u.train, u.ignore, self.cfg, n, 0, augment=True)
        return [ds[i] for i in range(n)]

    # ------------------------------------------------------------------ training / inference
    def train(self, unit_keys: tuple = (), include_final: bool = False) -> list[dict]:
        self._need("TRAIN", "train_enabled")
        from .train import fit_unit

        units = [u for u in self.units() if not unit_keys or u.key in unit_keys]
        if include_final:
            units.append(self.system.final_unit())
        out = []
        for u in units:
            out.append(fit_unit(self.cfg, self.stack, self.data, u, self.exp_dir, log=self.log))
            self.sync([self.exp_dir.relative_to(self.local)])
        return out

    def predict_heldout(self) -> dict:
        self._need("EVALUATE", "inference_enabled")
        from .inference import heldout_maps

        maps = heldout_maps(self.cfg, self.stack, self.data, self.system, self.units(), self.exp_dir, log=self.log)
        root = self.exp_dir / "maps"
        root.mkdir(exist_ok=True)
        for k, a in maps.items():
            np.save(root / f"{k}.npy", a)
        return maps

    def predict_final(self) -> np.ndarray:
        """Full-region map of the final unit's model (permitted known = every mapped fault)."""
        self._need("EVALUATE", "inference_enabled")
        from .inference import predict_full
        from .train import load_unit_model

        u = self.system.final_unit()
        model, inputs, _ = load_unit_model(self.cfg, self.stack, self.data, u, self.exp_dir)
        r = predict_full(model, inputs, self.cfg, self.data.valid, mc_samples=self.cfg.eval.mc_samples
                         if self.cfg.model.uncertainty == "mc_dropout" else 0, log=self.log)
        np.save(self.exp_dir / "maps" / "full.npy", r["prob"])
        if "mc_std" in r:
            np.save(self.exp_dir / "maps" / "full_mc_std.npy", r["mc_std"])
        return r["prob"]

    # ------------------------------------------------------------------ evaluation
    def model_maps(self) -> dict:
        from .evaluation import maps_from_units

        root = self.exp_dir / "maps"
        return maps_from_units({p.stem: np.load(p) for p in root.glob("*.npy") if p.stem.split("_")[0] in
                                ("geo", "disc")})

    def reference_maps(self) -> dict[str, dict]:
        """T0: strongest existing honest maps on the same units (only those that exist and match the backend)."""
        from .evaluation import load_g13_model, load_unit_maps

        refs = {}
        dr = Path(self.cfg.drive_root) / "outputs"
        if self.system.backend == "phase5":
            w2 = dr / "p6_fast_track" / "stack" / "maps"
            if w2.exists():
                refs["T0_P6_W2"] = load_unit_maps(w2, self.system)
        else:
            sb = self.g13_path("tables/scoreboard.csv")
            if sb is not None:
                b = pd.read_csv(sb).sort_values("disc_proxy_dti", ascending=False)
                for m in b.model.head(3):
                    p = self.g13_path(f"predictions/{m}")
                    if p is not None and (p / "oof.npy").exists():
                        refs[f"T0_G13_{m}"] = load_g13_model(p, self.system)
        return refs

    def evaluate(self, include_references: bool = True) -> pd.DataFrame:
        self._need("EVALUATE")
        from .evaluation import compare, score

        cards = []
        ev_dir = self.exp_dir / "evaluation"
        maps = self.model_maps()
        if maps:
            cards.append(score(self.cfg.experiment, maps, self.system, self.cfg, ev_dir)["row"])
        if include_references:
            for name, m in self.reference_maps().items():
                cards.append(score(name, m, self.system, self.cfg, ev_dir)["row"])
        df = compare(cards) if cards else pd.DataFrame()
        df.to_csv(ev_dir / "comparison.csv", index=False) if len(df) else None
        return df

    # ------------------------------------------------------------------ submission
    def submission(self, rep: str, frac: float, confirm: str = "") -> dict:
        self._need("SUBMIT", "submission_enabled")
        from .submission import prepare_submission

        full = np.load(self.exp_dir / "maps" / "full.npy")
        prov = {"model": self.cfg.experiment, "construction": "final unit model, every mapped fault as known context",
                "selection": "cross-fitted", "cfg_hash": self.cfg.semantic_hash(), "g14_version": G14_VERSION}
        return prepare_submission(self.cfg, self.data, full, rep, frac, self.exp_dir, name=f"g14_{self.cfg.experiment}",
                                  provenance=prov, confirm=confirm)

    # ------------------------------------------------------------------ persistence / report
    def sync(self, rels: list | None = None) -> int:
        """Copy local artifacts (except the channel stack and caches) to the Drive output tree."""
        n = 0
        items = [self.local / r for r in rels] if rels else [self.local / "experiments", self.shared]
        for src in items:
            for f in ([src] if src.is_file() else src.rglob("*")):
                if not f.is_file() or "stack_l" in f.name or f.suffix == ".partial" or "/cache/" in str(f):
                    continue
                dst = self.out / f.relative_to(self.local)
                if dst.exists() and dst.stat().st_size == f.stat().st_size and dst.stat().st_mtime >= f.stat().st_mtime:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
        return n

    def report(self) -> Path:
        """Markdown summary of what EXISTS (no number is written unless it was computed by a stage)."""
        lines = [f"# G14 experiment report: {self.cfg.experiment}", "",
                 f"Config hash `{self.cfg.semantic_hash()}`, G14 `{G14_VERSION}`, generated {time.strftime('%Y-%m-%d %H:%M')}.",
                 "Offline DTI values are proxies on held-out known faults, never leaderboard scores.", ""]
        for title, p in (("Contract check", self.shared / "contract_check.csv"),
                         ("Comparison", self.exp_dir / "evaluation" / "comparison.csv")):
            lines.append(f"## {title}\n")
            if p.exists():
                df = pd.read_csv(p)
                try:
                    lines.append(df.to_markdown(index=False))
                except ImportError:                       # tabulate missing
                    lines.append("```\n" + df.to_string(index=False) + "\n```")
            else:
                lines.append("_not run_")
            lines.append("")
        man = sorted((self.exp_dir / "manifests").glob("*_train.json")) if (self.exp_dir / "manifests").exists() else []
        lines.append("## Training\n")
        for m in man:
            j = json.loads(m.read_text())
            lines.append(f"* {j['unit']}: best epoch {j['best_epoch']}, inner score {j['best_inner_score']:.4f}, "
                         f"{j['seconds'] / 60:.1f} min, params {j['parameters']['total']:,}")
        if not man:
            lines.append("_no unit trained_")
        p = self.exp_dir / "REPORT.md"
        p.write_text("\n".join(lines))
        return p


def _flatten_notebook(c: dict) -> dict:
    """Accept the notebook's flat CONFIG keys plus nested sections."""
    return dict(c)
