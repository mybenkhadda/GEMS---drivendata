"""G02 orchestration: ``G02Experiment(m9, G02_CONFIG)`` -> prepare / evaluate / metrics / figures / report.

    g02 = G02Experiment(m9, G02_CONFIG)       # m9 = pipe.m9(M9_CONFIG): no M9 section is run, nothing is trained
    g02.prepare()                             # cache audit, G01 review, fingerprinted run folder, scale table
    g02.evaluate()                            # held-out units, tiled, resumable per unit (and per tile)
    g02.metrics(); g02.figures(); g02.report(); g02.push()
    g02 = G02Experiment.attach(m9)            # later session: newest run, nothing recomputed
    g02.results()

Per held-out unit (geographic fold or discovery split):
    1. percentile references on the unit's TRAINING geography (G01 ``fit_refs`` + raw-field quantiles);
       ``assert_disjoint`` refuses any overlap with the evaluation zone
    2. tiles with halos -> G01 evidence (B0 edge, G01-A..F, per-wavelength variants, M, G, J, P) and G02-A / B / C / E
       maps, written into per-unit memory-mapped arrays (never all maps of a region in RAM)
    3. G02-D networks on the unit window from each ``network.sources`` evidence map (+ GeoJSON / GPKG, segment and node
       tables)
    4. equal-area counts per spatial block for every candidate (seeded tie-break noise shared by all maps),
       complementarity counts, truth-length strata, label-free rank correlations, lineament-geometry metrics,
       official-style DTI (orientation only)
    5. one streamed NPZ of every map + domain + interior (labels are never stored)
B1 / B2 / G02-F are nested (leave-one-zone-out, overlapping zones excluded) choices made in ``metrics``.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from . import g02_metrics as XM
from . import geology_metrics as GM
from .experiment import TASK_CODE, G01Experiment, _md
from .g01_review import review_g01
from .g02_io import (G02_FAMILIES, G02_VERSION, G02Store, fingerprint_config, g01_config, latest_run, list_runs,
                     load_g02_config, run_root, set_latest)
from .geology_continuity import combine_fields, continuity_field
from .geology_displacement import DISP_KEYS, check_square, disp_halo_m, disp_scale, displacement_evidence, noisy_or
from .geology_hypotheses import hypotheses
from .geology_io import _safe, channel_audit, xmod_names
from .geology_io import fingerprint_config as g01_fingerprint_config
from .geology_io import latest_run as g01_latest_run
from .geology_io import run_root as g01_run_root
from .geology_network import build_network, segment_features
from .geology_truncation import TRUNC_KEYS, ridge_skeleton, truncation_evidence
from .geology_wavelet import QuantileRef, build_evidence, channel, combine, halo_m as g01_halo_m, is_candidate_key, \
    strike_from_doubled, tolerance_m

G01_KEEP = ("comp:M", "comp:G", "comp:J", "comp:P", "comp:orient_gate", "comp:S")
G01_MAIN = ("G01-A", "G01-B", "G01-C", "G01-D", "G01-E", "G01-F")
ATTACH = ("G02-A", "G02-B", "G02-C", "hyp:fault", "hyp:contact", "hyp:intrusive", "hyp:ambiguous", "G01-E", "edge")


# =============================================================================== per-unit memory-mapped maps
class MapStore:
    """Per-unit on-disk maps (``processed/tmp_<unit>/``): float32 (NaN), uint8 masks, int8 categories."""

    def __init__(self, root: Path, shape):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(int(s) for s in shape)
        self.m: dict = {}
        idx = self.root / "keys.json"
        if idx.exists():                                           # resume an interrupted unit
            for k, f in json.loads(idx.read_text(encoding="utf-8")).items():
                p = self.root / f
                if p.exists():
                    self.m[k] = np.load(p, mmap_mode="r+")

    def _index(self):
        (self.root / "keys.json").write_text(json.dumps({k: f"{_safe(k)}.npy" for k in self.m}), encoding="utf-8")

    def put(self, k: str, sl, a) -> None:
        a = np.asarray(a)
        if k not in self.m:
            dt = np.uint8 if a.dtype == bool else (np.int8 if a.dtype == np.int8 else np.float32)
            mm = np.lib.format.open_memmap(self.root / f"{_safe(k)}.npy", mode="w+", dtype=dt, shape=self.shape)
            mm[:] = np.nan if dt == np.float32 else (0 if dt == np.uint8 else -1)
            self.m[k] = mm
            self._index()
        self.m[k][sl] = a

    def __contains__(self, k) -> bool:
        return k in self.m

    def __getitem__(self, k):
        return self.m[k]

    def keys(self) -> list[str]:
        return list(self.m)

    def tiles_done(self) -> set:
        p = self.root / "tiles_done.json"
        return set(map(tuple, json.loads(p.read_text(encoding="utf-8")))) if p.exists() else set()

    def mark_tile(self, done: set) -> None:
        for mm in self.m.values():
            mm.flush()
        (self.root / "tiles_done.json").write_text(json.dumps(sorted(done)), encoding="utf-8")

    def close(self) -> None:
        for mm in self.m.values():
            mm.flush()
        self.m = {}


class LazyScores(Mapping):
    """Candidate scores computed on access (memmap + shared tie-break noise; fusions; random) -- bounded memory."""

    def __init__(self, names: list[str], get):
        self._names, self._get = list(names), get

    def __getitem__(self, k):
        if k not in self._names:
            raise KeyError(k)
        return self._get(k)

    def __contains__(self, k) -> bool:                             # never computes a map to answer membership
        return k in self._names

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


class LazyMasks(Mapping):
    def __init__(self, keys: list[tuple], get):
        self._keys, self._get = list(keys), get

    def __getitem__(self, k):
        return self._get(k)

    def __contains__(self, k) -> bool:
        return k in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


def fusion_name(base: str, fam: str, w: float) -> str:
    return f"F:{base}+{fam}@{float(w):g}"


def parse_fusion(name: str) -> tuple[str, str, float] | None:
    if not name.startswith("F:") or "+" not in name or "@" not in name:
        return None
    body = name[2:]
    base, rest = body.split("+", 1)
    fam, w = rest.rsplit("@", 1)
    return base, fam, float(w)


# =============================================================================== experiment
class G02Experiment:
    def __init__(self, m9, config: dict | None = None, *, _cfg: dict | None = None):
        self.m9 = m9
        self.cfg = _cfg if _cfg is not None else load_g02_config(config)
        self.g01cfg = g01_config(self.cfg)
        self.g01 = G01Experiment(m9, _cfg=self.g01cfg)              # G01 code only: its store is never opened
        self.src = self.g01.src
        self.parent = run_root(m9, self.cfg)
        self.store: G02Store | None = None
        self.lams: list = []
        self.warnings: list[str] = []
        self._raw: dict = {}
        self._m9pred = None
        self._refs_all = None

    def say(self, msg: str) -> None:
        print(f"[g02] {msg}")

    # ------------------------------------------------------------------------------- Drive
    def _rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(Path(self.m9.local).resolve()).as_posix()

    def stage_inputs(self) -> None:
        """Bring the M9 layer cache, G01 runs and earlier G02 runs from Drive when they exist (never regenerated)."""
        rels = [f"{self.m9.cfg['cache_dir']}/layers"]
        for p in (g01_run_root(self.m9, self.g01cfg), self.parent):
            try:
                rels.append(self._rel(p))
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

    def _need(self) -> G02Store:
        if self.store is None:
            raise RuntimeError("call g02.prepare() (or G02Experiment.attach) first")
        return self.store

    # ------------------------------------------------------------------------------- inputs
    def _select_lams(self, lams: list) -> list:
        want = self.cfg["wavelengths_km"]
        if want == "auto":
            return list(lams)
        tol = float(self.g01cfg["wavelength_rel_tol"])
        keep = [w for w in lams if any(abs(w.lam_km - float(x)) <= tol * float(x) for x in want)]
        if not keep:
            mid = len(lams) // 2
            keep = lams[max(mid - 1, 0):mid + 1]
            self.warnings.append(f"wavelengths_km {want} not cached for both fields: using {[w.label for w in keep]}")
        return keep

    def raw(self, role: str) -> np.ndarray | None:
        """Full-grid raw band of a G01 role (mag / grav / terrain), NaN off the validity mask (read once)."""
        if role not in self._raw:
            b = self.g01cfg["raw_bands"].get(role)
            if b is None or b not in self.src.data.band_names:
                self._raw[role] = None
            else:
                a = np.asarray(self.src.data.read_band(b), np.float32)
                self._raw[role] = np.where(self.src.valid, a, np.nan).astype(np.float32)
        return self._raw[role]

    def m9pred(self) -> np.ndarray | None:
        p = self.cfg["baselines"].get("m9_prediction_npy")
        if not p:
            return None
        if self._m9pred is None:
            a = np.load(p, mmap_mode="r")
            if tuple(a.shape) != tuple(self.src.shape):
                self.warnings.append(f"m9_prediction_npy shape {a.shape} != grid {self.src.shape}: ignored")
                self.cfg["baselines"]["m9_prediction_npy"] = None
                return None
            self._m9pred = a
        return self._m9pred

    def halo_m(self) -> float:
        cfg, px = self.cfg, self.src.pixel_m
        lm = [w.lam_km * 1000.0 for w in self.lams]
        tc, cc, hc = cfg["truncation"], cfg["continuity"], cfg["hypotheses"]
        trunc = max(max(float(tc["strip_half_length_frac"]) * x, float(tc["min_strip_half_length_m"]))
                    + float(tc["strip_width_frac"]) * x
                    + 3.0 * min(float(tc["background_sigma_frac"]) * x, float(tc["max_background_sigma_m"])) for x in lm)
        cont = max(max(float(cc["along_half_length_frac"]) * x, float(cc["min_along_half_length_m"]))
                   + float(cc["curvature_offset_frac"]) * x for x in lm) * np.sqrt(2.0)
        hyp = max(3.0 * float(hc["fabric_sigma_m"]), 3.0 * max(float(s) for s in hc["blob_sigma_m"])
                  + float(hc["blob_radius_m"]), (float(hc["side_offset_m"]) + float(hc["side_half_length_m"])) * np.sqrt(2))
        h = max(g01_halo_m(self.g01.lams, self.g01cfg), disp_halo_m([w.lam_km for w in self.lams], px, cfg), trunc,
                cont, hyp) + 200.0
        return float(min(h, float(cfg["tiling"]["max_halo_m"])))

    def scale_table(self) -> pd.DataFrame:
        px = self.src.pixel_m
        rows = []
        for w in self.lams:
            sc = disp_scale(w.lam_km, px, self.cfg["displacement"])
            rows.append({"wavelength_km": w.lam_km, "step_m": sc.step_m, "h1_m": sc.k1 * sc.step_m,
                         "h2_m": sc.k2 * sc.step_m, "half_length_m": sc.half * sc.step_m,
                         "max_offset_m": sc.M * sc.step_m, "offset_resolution_m": sc.a0_m,
                         "g01_tolerance_m": tolerance_m(w.lam_km, self.g01cfg["tolerance"]),
                         "band": "narrow" if w.lam_km <= float(self.cfg["continuity"]["split_km"]) else "broad"})
        return pd.DataFrame(rows)

    def prepare(self) -> dict:
        """Audit the cache, choose scales, review G01, fingerprint inputs, open (or reuse) the versioned run folder."""
        self.m9.load()
        self.stage_inputs()
        self.warnings = []
        cfg, gc = self.cfg, self.g01cfg
        check_square(self.src.pixel_m, float(cfg["max_pixel_aspect"]))
        names = self.src.names()
        audit, lams = channel_audit(names, gc["fields"], gc["wavelengths_km"], float(gc["wavelength_rel_tol"]))
        if not lams:
            raise RuntimeError(f"G02: no wavelength cached for BOTH fields in {self.m9.layers.dir}. Run M9 section 03 "
                               "once (m9.s03_spectral()) or run G01 with build_missing.wavelets = True first.")
        self.g01.lams = lams                                      # G01 baselines at ALL G01 scales (its definition)
        self.g01.xmod = xmod_names(names, gc["existing"]["patterns"])
        self.lams = self._select_lams(lams)                       # G02 families at the configured subset
        if not self.g01.xmod:
            self.warnings.append("G01-F unavailable (no cached m9:xmod:mag_grav_* layer): B2 family reduced")
        if self.raw("mag") is None:
            self.warnings.append("raw magnetic band missing: G02-A / G02-E magnetic parts unavailable")
        self.m9pred()
        gm = self.g01.grid_meta()
        from ...pipeline.m6 import fingerprint
        from ..features import file_fp

        fps = {n: self.src.fingerprint(n) for n in self.g01.channels()}
        fps.update({f"band:{r}": self.g01cfg["raw_bands"].get(r) for r in ("mag", "grav", "terrain")})
        if cfg["baselines"].get("m9_prediction_npy"):
            fps["m9pred"] = file_fp(Path(cfg["baselines"]["m9_prediction_npy"]))
        fp = fingerprint(G02_VERSION, fingerprint_config(cfg), g01_fingerprint_config(gc), fps, gm,
                         [w.lam_km for w in self.lams], [w.lam_km for w in lams])
        run_id = f"g02_{fp[:10]}"
        self.store = G02Store(self.parent / run_id, overwrite=bool(cfg["overwrite"]), fingerprint=fp, push=self.push)
        st = self.store
        st.save_json(json.loads(json.dumps(cfg, default=str)), "config/g02_config.json")
        st.save_json(json.loads(json.dumps(gc, default=str)), "config/g01_config_used.json")
        st.save_json({"run_id": run_id, "fingerprint": fp, "g02_version": G02_VERSION, "mode": cfg["mode"], "grid": gm,
                      "g01_wavelengths": [{"lam_km": w.lam_km, "tags": dict(w.tags)} for w in lams],
                      "g02_wavelengths": [{"lam_km": w.lam_km, "tags": dict(w.tags)} for w in self.lams],
                      "xmod": self.g01.xmod, "input_fingerprints": fps, "warnings": self.warnings,
                      "halo_m": self.halo_m(), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "config/inputs.json")
        st.save_table(audit, "channel_audit", "tables")
        st.save_table(self.scale_table(), "g02_scales", "tables", params=cfg["displacement"])
        rev = self.g01_review()
        set_latest(self.parent, run_id, fp, cfg["mode"])
        st.mark_done("inputs", {"n_g01_wavelengths": len(lams), "n_g02_wavelengths": len(self.lams)})
        st.flush_manifest({"run_id": run_id, "grid": gm, "mode": cfg["mode"]})
        for w_ in self.warnings:
            self.say(f"WARNING {w_}")
        self.say(f"run {run_id} ({cfg['mode']}): G01 scales {[w.label for w in lams]}, G02 scales "
                 f"{[w.label for w in self.lams]}, halo {self.halo_m():.0f} m -> {st.root}")
        return {"run_id": run_id, "audit": audit, "wavelengths": self.lams, "grid": gm, "warnings": self.warnings,
                "g01_review": rev}

    def g01_review(self) -> dict:
        """Review of the persisted G01 run named by ``g01.review_run`` (read-only); ABSENT answers when none."""
        st = self._need()
        rr = self.cfg["g01"]["review_run"]
        parent = g01_run_root(self.m9, self.g01cfg)
        root = None
        if rr == "auto":
            rid = g01_latest_run(parent)
            root = parent / rid if rid else None
        elif rr != "none":
            root = parent / rr
        if root is not None and not root.exists():
            self.warnings.append(f"G01 review: run {root.name} not found")
            root = None
        rev = review_g01(root, self.cfg["decision"]["metric"])
        st.save_table(rev["questions"], "g01_review_questions", "tables")
        st.save_table(rev["decision_table"], "g01_decision_table", "tables")
        st.save_table(rev["edge_redundancy"], "g01_edge_redundancy", "tables")
        from ...artifacts.atomic import atomic_write_text

        atomic_write_text(st.path("reports/G01_REVIEW.md"), rev["markdown"])
        st.record("reports/G01_REVIEW.md", "report_md", [str(root) if root else "none"], {})
        if not rev["available"]:
            self.warnings.append("no persisted G01 run: G01 review is ABSENT (G02 recomputes G01 maps itself)")
        return rev

    @classmethod
    def attach(cls, m9, run_id: str | None = None, config: dict | None = None) -> "G02Experiment":
        """Open an existing run (newest by default) without recomputing anything."""
        tmp = cls(m9, config)
        tmp.stage_inputs()
        rid = run_id or latest_run(tmp.parent)
        if not rid or not (tmp.parent / rid).exists():
            runs = list_runs(tmp.parent) if tmp.parent.exists() else pd.DataFrame()
            raise FileNotFoundError(f"no G02 run under {tmp.parent} (known: {list(runs.run_id) if len(runs) else []})")
        root = tmp.parent / rid
        cfg = json.loads((root / "config" / "g02_config.json").read_text(encoding="utf-8"))
        inp = json.loads((root / "config" / "inputs.json").read_text(encoding="utf-8"))
        g = cls(m9, _cfg=cfg)
        from .geology_wavelet import Wavelength

        g.store = G02Store(root, overwrite=False, fingerprint=inp["fingerprint"], push=g.push)
        g.g01.lams = [Wavelength(float(x["lam_km"]), tuple(sorted(x["tags"].items()))) for x in inp["g01_wavelengths"]]
        g.lams = [Wavelength(float(x["lam_km"]), tuple(sorted(x["tags"].items()))) for x in inp["g02_wavelengths"]]
        g.g01.xmod = list(inp.get("xmod") or [])
        g.warnings = list(inp.get("warnings") or [])
        return g

    # ------------------------------------------------------------------------------- windows
    def halo_window(self, sl) -> tuple:
        H, W = self.src.shape
        hy = int(np.ceil(self.halo_m() / self.src.pixel_m[0]))
        hx = int(np.ceil(self.halo_m() / self.src.pixel_m[1]))
        r0, r1 = max((sl[0].start or 0) - hy, 0), min((sl[0].stop or H) + hy, H)
        c0, c1 = max((sl[1].start or 0) - hx, 0), min((sl[1].stop or W) + hx, W)
        hs = (slice(r0, r1), slice(c0, c1))
        inner = (slice((sl[0].start or 0) - r0, (sl[0].stop or H) - r0),
                 slice((sl[1].start or 0) - c0, (sl[1].stop or W) - c0))
        return hs, inner

    def fit_raw_refs(self, ref_mask: np.ndarray, seed: int) -> dict:
        """Raw-field percentile tables + IQR on the reference geography (label-free; for G02-E step / blob)."""
        nc = self.g01cfg["normalization"]
        idx = np.flatnonzero(np.asarray(ref_mask, bool).ravel())
        if idx.size > int(nc["sample_px"]):
            idx = np.sort(np.random.default_rng(seed + 1).choice(idx, int(nc["sample_px"]), replace=False))
        out = {}
        for role in ("mag", "grav"):
            a = self.raw(role)
            if a is None:
                continue
            v = a.reshape(-1)[idx]
            v = v[np.isfinite(v)]
            q = QuantileRef.fit(v, int(nc["n_quantiles"]))
            iqr = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size >= 10 else np.nan
            out[role] = {"q": q, "iqr": iqr}
        return out

    def tile_maps(self, sl, refs: dict, rrefs: dict) -> dict:
        """Every G01 and G02 map on one tile (computed on its halo window, trimmed to ``sl``). Label-free."""
        cfg, gc = self.cfg, self.g01cfg
        px = self.src.pixel_m
        hs, inner = self.halo_window(sl)
        valid = self.src.valid[hs]
        win = self.g01.read_window(hs)
        ev = build_evidence(win, refs, self.g01.lams, gc, px, keep_per_scale=True, ablations=True)
        out = {k: v for k, v in ev.items() if is_candidate_key(k) or k in G01_KEEP}
        g02l = {w.label for w in self.lams}
        out.update({k: v for k, v in ev.items() if k.startswith(("scale:M@", "scale:G@")) and k.split("@")[1] in g02l})
        fam = gc["fields"]
        E, C2S2, ST = {}, {}, {}
        for role in ("mag", "grav"):
            E[role], C2S2[role], ST[role] = {}, {}, {}
            for w in self.lams:
                e = ev.get(f"scale:{'M' if role == 'mag' else 'G'}@{w.label}")
                c2 = win.get(channel(fam[role], "strike_c2", w.tag(role)))
                s2 = win.get(channel(fam[role], "strike_s2", w.tag(role)))
                if e is None or c2 is None or s2 is None:
                    continue
                E[role][w.lam_km] = e
                C2S2[role][w.lam_km] = (c2, s2)
                ST[role][w.lam_km] = np.where(np.hypot(c2, s2) > 1e-6, strike_from_doubled(c2, s2), np.nan)
        raw = {r: (self.raw(r)[hs] if self.raw(r) is not None else None) for r in ("mag", "grav", "terrain")}
        A = {}
        for role in cfg["displacement"]["fields"]:
            if raw.get(role) is None or not E[role]:
                continue
            d = displacement_evidence(raw[role], valid, E[role], sorted(E[role]), px, cfg)
            A[role] = out[f"G02-A[{role}]"] = d["score"]
            out.update({f"disp[{role}]:{k}": d[k] for k in DISP_KEYS if k != "score"})
        if A:
            out["G02-A"] = noisy_or(*A.values())
        tc = cfg["truncation"]
        skels = {r: {lam: ridge_skeleton(E[r][lam], ST[r][lam], valid, px, float(tc["ridge_q"]),
                                         float(tc["min_ridge_len_m"])) for lam in E[r]} for r in ("mag", "grav")}
        for r, d in skels.items():
            out.update({f"ridge[{r}]@{lam:g}km": s for lam, s in d.items()})
        Bt = {}
        for role in tc["fields"]:
            if not skels.get(role):
                continue
            t = truncation_evidence(skels[role], E[role], ST[role], valid, px, cfg)
            Bt[role] = out[f"G02-B[{role}]"] = t["score"]
            out.update({f"trunc[{role}]:{k}": t[k] for k in TRUNC_KEYS if k != "score"})
        if Bt:
            out["G02-B"] = noisy_or(*Bt.values())
        per = {r: continuity_field(skels[r], E[r], C2S2[r], valid, px, cfg, gc["tolerance"]) for r in skels if skels[r]}
        cont = combine_fields(per)
        if cont:
            out["G02-C"], out["G02-C[narrow]"], out["G02-C[broad]"] = cont["score"], cont["narrow"], cont["broad"]
            out.update({f"cont:{k}": cont[k] for k in ("P_ridge", "strike_stability", "along", "straightness")})
        M, G = ev.get("comp:M"), ev.get("comp:G")
        L = np.fmax(M, G) if (M is not None and G is not None) else (M if M is not None else G)
        if L is not None and cont:
            fc = []
            for r in E:
                ls = sorted(E[r])
                if len(ls) >= 2:
                    fc.append(E[r][ls[0]] - E[r][ls[-1]])
            inp = {"L": L, "disp": out.get("G02-A"), "trunc": out.get("G02-B"), "straight": cont["straightness"],
                   "persist": cont["P_ridge"],
                   "vectors": [(C2S2[r][lam][0], C2S2[r][lam][1], E[r][lam]) for r in C2S2 for lam in C2S2[r]],
                   "raw_pct": {r: rrefs[r]["q"](raw[r]) for r in rrefs if raw.get(r) is not None},
                   "raw": {r: raw[r] for r in rrefs if raw.get(r) is not None},
                   "raw_iqr": {r: rrefs[r]["iqr"] for r in rrefs}, "fine_coarse": fc,
                   "S": ev.get("comp:S"), "elev": raw.get("terrain")}
            out.update(hypotheses(inp, valid, px, cfg))
        return {k: np.asarray(v)[inner] for k, v in out.items()}

    # ------------------------------------------------------------------------------- held-out units
    def units(self, task: str) -> list:
        ev = self.cfg["evaluation"]
        us = self.m9.m7.units_of(task) if ev["zones"] == "all" else self.m9.units_of(task)
        us = sorted(us, key=lambda u: int(u.zone))
        if ev.get("units"):
            want = set(ev["units"])
            us = [u for u in us if f"{u.task}:{u.zone}" in want]
        if ev.get("max_units_per_task"):
            us = us[:int(ev["max_units_per_task"])]
        return us

    def main_names(self, present) -> list[str]:
        out = ["edge"] + [c for c in G01_MAIN if c in present] + [f for f in G02_FAMILIES if f in present]
        if self.cfg["baselines"].get("random", True):
            out.append("random")
        if self.cfg["baselines"].get("m9_prediction_npy"):
            out.append("M9pred")
        return out

    def ablation_names(self, present) -> list[str]:
        out = [k for k in present if "@" in k and k.startswith(("G01-A", "G01-B", "G01-C", "G01-D"))]
        out += [k for k in present if k.startswith(("G02-A[", "G02-B[", "G02-C["))]
        out += [k for k in present if k.startswith("net[")]
        return sorted(out)

    def fusion_names(self, present) -> list[str]:
        fc = self.cfg["fusion"]
        m9p = bool(self.cfg["baselines"].get("m9_prediction_npy"))
        bases = [b for b in dict.fromkeys(list(fc["bases"]) + (["M9pred"] if m9p else []))
                 if b in present or (b == "M9pred" and m9p)]
        fams = [f for f in fc["families"] if f in present]
        return [fusion_name(b, f, w) for b in bases for f in fams for w in fc["weights"]]

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

    def unit_maps(self, u, refs: dict, rrefs: dict, key: str) -> MapStore:
        st = self._need()
        R0, R1, C0, C1 = u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop
        ms = MapStore(st.path(f"processed/tmp_{key}"), (R1 - R0, C1 - C0))
        done = ms.tiles_done()
        tp = int(self.cfg["tiling"]["tile_px"])
        tiles = [(r, c) for r in range(0, R1 - R0, tp) for c in range(0, C1 - C0, tp)]
        for i, (r, c) in enumerate(tiles):
            if (r, c) in done:
                continue
            gs = (slice(R0 + r, min(R0 + r + tp, R1)), slice(C0 + c, min(C0 + c + tp, C1)))
            ls = (slice(r, min(r + tp, R1 - R0)), slice(c, min(c + tp, C1 - C0)))
            if self.src.valid[gs].any():
                t0 = time.perf_counter()
                for k, a in self.tile_maps(gs, refs, rrefs).items():
                    ms.put(k, ls, a)
                self.src.release()
                self.say(f"{key}: tile {i + 1}/{len(tiles)} in {time.perf_counter() - t0:.0f} s")
            done.add((r, c))
            ms.mark_tile(done)
        return ms

    def unit_networks(self, u, ms: MapStore, valid_u: np.ndarray, key: str) -> dict:
        st = self._need()
        nc = self.cfg["network"]
        px = self.src.pixel_m
        r0, c0 = u.sl[0].start, u.sl[1].start
        g = self.src.data.grid
        epsg = g.crs.to_epsg() if hasattr(g.crs, "to_epsg") else None
        out = {}
        for i, s in enumerate(nc["sources"]):
            if s not in ms:
                self.warnings.append(f"network source {s} unavailable on {key}")
                continue
            attach = {k: np.asarray(ms[k], np.float32) for k in ATTACH if k in ms and k != s}
            net = build_network(np.asarray(ms[s], np.float32), valid_u, px, self.cfg, attach)
            name = "G02-D" if i == 0 else f"net[{s}]"
            ms.put(name, (slice(None), slice(None)), net["raster"])
            S = net["segments"].assign(task=u.task, zone=int(u.zone), source=s, network=name)
            st.save_table(S, f"segments_{key}_{_safe(s)}", "lineaments", params=net["thresholds"])
            st.save_table(net["graph"]["nodes"].assign(task=u.task, zone=int(u.zone), source=s),
                          f"nodes_{key}_{_safe(s)}", "lineaments")
            if nc.get("vectors", True) and len(S):
                feats = segment_features(net["graph"], net["segments"], g.rc_to_xy, r0, c0,
                                         {"task": u.task, "zone": int(u.zone), "source": s, "network": name})
                st.save_vectors(feats, f"lineaments/{key}_{_safe(s)}", epsg, params=net["thresholds"])
            out[s] = {"name": name, "seg_lab": net["graph"]["seg_lab"], "segments": net["segments"],
                      "thresholds": net["thresholds"]}
            del attach, net
        return out

    def evaluate_unit(self, u) -> dict:
        st = self._need()
        cfg, ev_c = self.cfg, self.cfg["evaluation"]
        key = f"{u.task}_{u.zone}"
        if st.is_done(f"unit_{key}"):
            self.say(f"{key}: reused")
            return st.load_json(f"processed/unit_{key}.json") or {"unit": key, "status": "reused"}
        t0 = time.perf_counter()
        H, W = self.src.shape
        px = self.src.pixel_m
        seed = int(cfg["seed"]) + 7919 * int(u.zone) + 104729 * TASK_CODE.get(u.task, 9)
        ev_full = GM.full_mask((H, W), u.sl, u.ev)
        ref = GM.reference_mask(self.m9.ctx, u.task, u.zone)
        GM.assert_disjoint(ref, ev_full, f"{u.key} G02 normalization reference")
        refs = self.g01.fit_refs(ref, seed)
        rrefs = self.fit_raw_refs(ref, seed)
        ref_px = int(ref.sum())
        del ev_full, ref
        st.save_json({n: r.to_dict() for n, r in refs.items()}, f"processed/refs_{key}.json",
                     params={"reference": "training_geography", "ref_px": ref_px})
        ms = self.unit_maps(u, refs, rrefs, key)
        r0, r1, c0, c1 = u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop
        valid_u = self.src.valid[u.sl]
        interior = GM.interior_mask(valid_u, float(self.g01cfg["boundary_margin_m"]), px,
                                    (r0 == 0, r1 == H, c0 == 0, c1 == W))
        nets = self.unit_networks(u, ms, valid_u, key)
        el = np.asarray(self.m9.elig(u, 0), bool)
        domain = el & interior
        truth = np.asarray(u.eval_truth(), bool) & interior
        present = set(ms.keys())
        noise = XM.tie_noise(domain.shape, seed, float(cfg["baselines"]["tie_break"]))
        m9p = self.m9pred()

        def base_map(k):
            if k == "M9pred":
                return np.asarray(m9p[u.sl], np.float32)
            return np.asarray(ms[k], np.float32)

        def get(k):
            if k == "random":
                return np.where(domain, np.random.default_rng(seed).random(domain.shape, dtype=np.float32), np.nan)
            f = parse_fusion(k)
            a = combine([base_map(f[0]), base_map(f[1])], [1.0 - f[2], f[2]]) if f else base_map(k)
            return np.where(np.isfinite(a), a + noise, np.nan).astype(np.float32)

        main = [n for n in self.main_names(present) if n in present or n in ("random", "M9pred")]
        abl = self.ablation_names(present)
        fus = self.fusion_names(present)
        af, bf = [float(f) for f in ev_c["area_fracs"]], [float(f) for f in ev_c["ablation_area_fracs"]]
        fr = {n: af for n in main}
        fr.update({n: bf for n in abl})
        fr.update({n: (af if cfg["fusion"]["budgets"] == "all" else bf) for n in fus})
        fams = list(cfg["baselines"]["b1_family"]) + list(cfg["baselines"]["b2_family"])
        fr.update({n: af for n in XM.match_family(list(fr), fams)})   # B1 / B2 curves pool the SAME zones per budget
        LS = LazyScores(list(fr), get)
        bpx = max(int(round(float(ev_c["block_km"]) * 1000.0 / (0.5 * (px[0] + px[1])))), 8)
        blk, ids = GM.block_index(u.sl, domain.shape, bpx)
        tags = {"task": u.task, "zone": int(u.zone), "unit": key}
        B = GM.unit_block_counts(LS, domain, truth, px, fr, ev_c["distance_tol_m"], blk, ids, tags=tags)
        st.save_table(B, f"blocks_{key}", "processed", params={"block_px": bpx})
        f0, d0 = float(ev_c["primary_area_frac"]), float(ev_c["primary_distance_m"])
        refs_c = ["edge"] + [b for b in cfg["fusion"]["bases"] if b in LS]
        cands = [f for f in G02_FAMILIES if f in LS] + [b for b in cfg["fusion"]["bases"] if b in LS] + \
            [c for c in ("G01-A", "random") if c in LS]
        C = XM.complement_counts(LS, refs_c, cands, domain, truth, px, f0, d0, blk, ids, tags)
        st.save_table(C, f"complement_{key}", "processed")
        strata, labels = XM.truth_strata(truth, px, ev_c["length_strata_km"])
        Sx = XM.strata_counts(LS, main, domain, truth, strata, labels, px, f0, d0, tags)
        st.save_table(Sx, f"strata_{key}", "processed")
        Rc = XM.rank_correlations(LS, main, domain, int(ev_c["correlation_sample_px"]), seed, tags)
        st.save_table(Rc, f"correlations_{key}", "processed")
        st.save_table(self._lambda_usage(LS, ms, domain, truth, f0, d0, tags), f"lambda_usage_{key}", "processed")
        Lm = []
        for s, net in nets.items():
            lm = XM.lineament_metrics(net["seg_lab"], net["segments"], truth, domain, px,
                                      ev_c["lineament_budgets_km_per_100km2"], ev_c["distance_tol_m"],
                                      {**tags, "source": s, "network": net["name"]})
            Lm.append(lm)
        st.save_table(pd.concat(Lm, ignore_index=True) if Lm else pd.DataFrame(), f"lineament_{key}", "processed")
        pkey = self.m9.pkey(u)
        if ev_c.get("official_style", True):
            mk = [(n, f) for n in main for f in fr[n]]
            O = pd.DataFrame(GM.official_style_rows(u, LazyMasks(mk, lambda k: GM.equal_area_mask(LS[k[0]], domain, k[1])),
                                                    el, pkey)).assign(selection_domain="eligible & interior")
            st.save_table(O, f"official_{key}", "processed")
        fp = XM.footprint(u, int(ev_c["overlap_stride"]))
        from ...artifacts.atomic import atomic_save

        def wfp(t):
            with open(t, "wb") as f:
                np.savez_compressed(f, r0=fp["r0"], c0=fp["c0"], stride=fp["stride"], mask=fp["mask"])
        atomic_save(st.path(f"processed/footprint_{key}.npz"), wfp)
        keys = sorted(ms.keys())
        extra = {"domain": domain, "interior": interior}
        st.save_raster_stream(keys + list(extra), lambda k: extra[k] if k in extra else ms[k],
                              f"rasters/units/{key}.npz",
                              {"task": u.task, "zone": int(u.zone), "window": [r0, r1, c0, c1],
                               "grid": self.g01.grid_meta(), "protocol_key": pkey, "mode": cfg["mode"],
                               "selection_domain": "m9.elig(u, 0) & interior(G01 boundary_margin_m)",
                               "labels_stored": False, "note": "truth is re-derived from the M7 unit"},
                              dtype=cfg["save"]["raster_dtype"], sources=self.g01.channels())
        info = {"unit": key, "task": u.task, "zone": int(u.zone), "status": "computed", "eval_px": int(u.ev.sum()),
                "eligible_px": int(el.sum()), "domain_px": int(domain.sum()), "truth_px": int(truth.sum()),
                "reference_px": ref_px, "block_px": bpx, "n_candidates": len(fr), "n_maps": len(keys),
                "networks": {s: {"n_segments": int(len(n["segments"])), **n["thresholds"]} for s, n in nets.items()},
                "seconds": round(time.perf_counter() - t0, 1), "protocol_key": pkey}
        st.save_json(info, f"processed/unit_{key}.json")
        st.mark_done(f"unit_{key}", info)
        st.flush_manifest()
        ms.close()
        if not cfg["save"].get("keep_tmp", False):
            shutil.rmtree(st.path(f"processed/tmp_{key}"), ignore_errors=True)
        u.release()
        self.push()
        self.say(f"{key}: {info['domain_px']} domain px, {info['truth_px']} truth px, {info['seconds']} s")
        return info

    def _lambda_usage(self, LS, ms: MapStore, domain, truth, f0: float, d0: float, tags: dict) -> pd.DataFrame:
        """Which wavelength produced the selected G02-A / G02-B pixels, and how often they were near truth."""
        rows = []
        d_truth = GM.distance_m(truth, self.src.pixel_m)
        for fam, pre in (("G02-A", "disp"), ("G02-B", "trunc")):
            for role in ("mag", "grav"):
                c, lk = f"{fam}[{role}]", f"{pre}[{role}]:lambda_km"
                if c not in LS or lk not in ms:
                    continue
                sel = GM.equal_area_mask(LS[c], domain, f0)
                lam = np.asarray(ms[lk], np.float32)
                for v in np.unique(lam[sel & np.isfinite(lam)]):
                    m = sel & (lam == v)
                    rows.append({"candidate": c, "lambda_km": float(v), "n_selected": int(m.sum()),
                                 "n_selected_near_truth": int((m & (d_truth <= d0)).sum()), "area_frac": f0,
                                 "tol_m": d0, **tags})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------------------- metrics
    def _concat(self, prefix: str) -> pd.DataFrame:
        st = self._need()
        parts = [st.load_table(p.stem, "processed") for p in sorted(st.path("processed").glob(f"{prefix}_*.csv"))]
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def footprints(self) -> dict:
        st = self._need()
        out = {}
        for p in sorted(st.path("processed").glob("footprint_*.npz")):
            task, zone = p.stem[len("footprint_"):].rsplit("_", 1)
            with np.load(p) as z:
                out[(task, int(zone))] = {"r0": int(z["r0"]), "c0": int(z["c0"]), "stride": int(z["stride"]),
                                          "mask": np.asarray(z["mask"], bool)}
        return out

    def pairs(self, have: set) -> list[tuple]:
        fc = self.cfg["fusion"]
        P = [("baseline", "baseline", "B1", "B0", "best magnetic-only G01 map beyond a simple magnetic gradient"),
             ("baseline", "baseline", "B2", "B1", "best G01 fusion beyond the best magnetic-only G01 map"),
             ("baseline", "baseline", "B2", "B0", "best G01 fusion beyond a simple magnetic gradient"),
             ("baseline", "baseline", "G01-E", "G01-D", "G01 persistence (re-measured inside G02)")]
        for X in G02_FAMILIES:
            P += [(X, "standalone", X, "B0", f"{X} alone beyond a simple magnetic gradient"),
                  (X, "standalone", X, "random", f"{X} alone beyond random placement"),
                  (X, "standalone", X, "B2", f"{X} alone vs the best G01 fusion")]
            for b in fc["bases"]:
                P.append((X, "incremental", f"F:{b}+{X}", b, f"{X} added to {b} (weight chosen nested) beyond {b}"))
        for fam in ("G02-A", "G02-B"):
            P += [(fam, "field", f"{fam}[mag]", "random", f"{fam} magnetic only beyond random"),
                  (fam, "field", f"{fam}[grav]", "random", f"{fam} gravity only beyond random"),
                  (fam, "field", fam, f"{fam}[mag]", f"{fam}: gravity adds to magnetic")]
        P += [("G02-C", "band", "G02-C[narrow]", "G02-C[broad]", "narrow-band continuity beyond broad-band"),
              ("G02-D", "network", "G02-D", "net[edge]", "network on G01 evidence beyond network on the edge")]
        P += [("G02-F", "fusion", "G02-F", "B2", "G01 + selected complementary G02 evidence beyond the best G01 fusion"),
              ("G02-F", "fusion", "G02-F", "B0", "G01 + G02 beyond a simple magnetic gradient")]
        if "M9pred" in have:
            P += [("M9", "baseline", "M9pred", "B0", "existing M9 prediction beyond the gradient")]
            for X in G02_FAMILIES:
                P.append((X, "m9_increment", f"F:M9pred+{X}", "M9pred", f"{X} added to the M9 prediction"))
        return P

    def metrics(self) -> dict:
        st = self._need()
        cfg, ev_c, dc = self.cfg, self.cfg["evaluation"], self.cfg["decision"]
        B = self._concat("blocks")
        if not len(B):
            raise RuntimeError("no evaluated unit: run g02.evaluate() first")
        f0, d0 = float(ev_c["primary_area_frac"]), float(ev_c["primary_distance_m"])
        seed, nb, metric = int(cfg["seed"]), int(ev_c["n_boot"]), dc["metric"]
        ov = XM.unit_overlap(self.footprints())
        excl = XM.exclusions(ov)
        st.save_table(ov, "unit_overlap", "tables")
        have = set(B.candidate)
        virt, choices = [B[B.candidate == "edge"].assign(candidate="B0")], []
        for name, pats in (("B1", cfg["baselines"]["b1_family"]), ("B2", cfg["baselines"]["b2_family"])):
            rows, ch = XM.nested_virtual(B, XM.match_family(have, pats), name, metric, f0, d0, excl)
            virt.append(rows)
            choices.append(ch)
        allf = []
        for X in G02_FAMILIES:
            for b in dict.fromkeys(list(cfg["fusion"]["bases"]) + (["M9pred"] if "M9pred" in have else [])):
                v = [n for n in have if (p := parse_fusion(n)) and p[0] == b and p[1] == X]
                allf += v
                rows, ch = XM.nested_virtual(B, v, f"F:{b}+{X}", metric, f0, d0, excl)
                virt.append(rows)
                choices.append(ch)
        rows, ch = XM.nested_virtual(B, [n for n in allf if not n.startswith("F:M9pred")], "G02-F", metric, f0, d0, excl)
        virt.append(rows)
        choices.append(ch)
        Bv = pd.concat([B] + [v for v in virt if len(v)], ignore_index=True)
        CH = pd.concat([c for c in choices if len(c)], ignore_index=True) if choices else pd.DataFrame()
        S = GM.versus_edge(GM.summarize(Bv, ["task", "candidate", "area_frac", "tol_m"]), ["task", "area_frac", "tol_m"])
        Z = GM.summarize(Bv, ["task", "zone", "candidate", "area_frac", "tol_m"])
        nbc = min(nb, 200)
        heads = ["B0", "B1", "B2", "G02-F", *G02_FAMILIES, "random"]
        Bm = Bv[Bv.candidate.isin(heads)]
        ci = []
        for m in ("precision", "recall", "f1", "enrichment"):
            a = GM.per_block_uncertainty(Bm[Bm.tol_m == d0], ["task", "candidate", "area_frac", "tol_m"], m, nbc, seed)
            b = GM.per_block_uncertainty(Bm[(Bm.area_frac == f0) & (Bm.tol_m != d0)],
                                         ["task", "candidate", "area_frac", "tol_m"], m, nbc, seed)
            u = pd.concat([a, b], ignore_index=True)
            if len(u):
                ci.append(u.rename(columns={"ci_lo": f"{m}_lo", "ci_hi": f"{m}_hi"}).drop(columns=["metric", "value"]))
        for c in ci:
            S = S.merge(c, on=["task", "candidate", "area_frac", "tol_m"], how="left")
        pairs = self.pairs(set(Bv.candidate))
        H = XM.contrast_table(Bv, Z, pairs, f0, d0, dc, nb, seed)
        Hp = XM.contrast_table(Bv, Z, [p for p in pairs if p[1] == "incremental"], f0, d0, {**dc, "metric": "precision"},
                               nb, seed)
        D = XM.per_zone_deltas(Z, pairs, f0, d0, metric)
        R = XM.decide(H, cfg["mode"], dc, list(G02_FAMILIES) + ["G02-F"])
        R = self._fusion_recommendation(R, H, dc)
        C = self._concat("complement")
        Cp = XM.summarize_complement(C, ["task", "reference", "candidate"]) if len(C) else pd.DataFrame()
        Cz = XM.summarize_complement(C, ["task", "zone", "reference", "candidate"]) if len(C) else pd.DataFrame()
        Sx = self._concat("strata")
        if len(Sx):
            Sx = Sx.groupby(["task", "candidate", "stratum"])[["n_truth", "n_truth_hit"]].sum().reset_index()
            Sx["recall"] = Sx.n_truth_hit / Sx.n_truth.where(Sx.n_truth > 0)
        Rc = self._concat("correlations")
        Rcp = Rc.groupby(["a", "b"]).spearman.median().reset_index() if len(Rc) else Rc
        Lu = self._concat("lambda_usage")
        if len(Lu):
            Lu = Lu.groupby(["task", "candidate", "lambda_km"])[["n_selected", "n_selected_near_truth"]].sum().reset_index()
            Lu["precision"] = Lu.n_selected_near_truth / Lu.n_selected.where(Lu.n_selected > 0)
        Lm = self._concat("lineament")
        Lmp = Lm.groupby(["task", "source", "network", "budget_km_per_100km2", "tol_m"])[
            ["length_precision", "truth_recall", "random_precision", "orientation_agreement"]].mean().reset_index() \
            if len(Lm) else Lm
        O = self._concat("official")
        Op = GM.pooled_official(O) if len(O) else pd.DataFrame()
        abl = S[S.candidate.str.contains("@|\\[", regex=True) & (S.area_frac == f0)]
        for name, df in (("summary_pooled", S), ("summary_per_zone", Z), ("nested_choices", CH),
                         ("contrasts", H), ("contrasts_precision", Hp), ("per_zone_deltas", D),
                         ("recommendations", R), ("complement_pooled", Cp), ("complement_per_zone", Cz),
                         ("strata_recall", Sx), ("rank_correlations", Rcp), ("lambda_usage", Lu),
                         ("lineament_metrics", Lmp), ("lineament_metrics_per_unit", Lm), ("ablations", abl),
                         ("official_style_per_zone", O), ("official_style_pooled", Op)):
            st.save_table(df, name, "tables")
        st.save_table(Bv, "blocks_all", "tables")
        summ = self.summary_json(S, H, R, f0, d0)
        st.save_json(summ, "reports/summary.json")
        st.mark_done("metrics", {"n_blocks": int(len(Bv))})
        st.flush_manifest()
        self.push()
        return {"pooled": S, "per_zone": Z, "choices": CH, "contrasts": H, "contrasts_precision": Hp, "deltas": D,
                "recommendations": R, "complement": Cp, "strata": Sx, "correlations": Rcp, "lambda_usage": Lu,
                "lineament": Lmp, "official_pooled": Op, "summary": summ}

    def _fusion_recommendation(self, R: pd.DataFrame, H: pd.DataFrame, dc: dict) -> pd.DataFrame:
        """G02-F is judged on G02-F vs B2 (kind 'fusion') with the same rules as the families' increments."""
        h = H[(H.family == "G02-F") & (H.reference == "B2")].assign(kind="incremental") if len(H) else H
        r = XM.decide(h, self.cfg["mode"], dc, ["G02-F"])
        return pd.concat([R[R.family != "G02-F"], r], ignore_index=True)

    def summary_json(self, S, H, R, f0, d0) -> dict:
        heads = ["B0", "B1", "B2", *G02_FAMILIES, "G02-F", "random"]
        s = S[(S.area_frac == f0) & (S.tol_m == d0) & S.candidate.isin(heads)]
        cols = ["precision", "recall", "f1", "enrichment", "precision_vs_edge"]
        return {"run_id": self.store.root.name, "fingerprint": self.store.fp, "mode": self.cfg["mode"],
                "primary_area_frac": f0, "primary_distance_m": d0, "decision": self.cfg["decision"],
                "warnings": self.warnings,
                "status": ("EXPLORATORY (FAST_RESEARCH subset): not a held-out verdict" if self.cfg["mode"] != "FULL_EVALUATION"
                           else "validated on held-out zones (inspected in earlier GEMS work: not an untouched test set)"),
                "pooled": {t: {r.candidate: {c: (None if pd.isna(getattr(r, c, np.nan)) else float(getattr(r, c)))
                                             for c in cols} for r in g.itertuples()} for t, g in s.groupby("task")},
                "recommendations": R.to_dict("records") if len(R) else [],
                "verdicts": (H[["family", "kind", "task", "candidate", "reference", "verdict"]].to_dict("records")
                             if len(H) else [])}

    def results(self) -> dict:
        """Every saved table of the run (no recomputation)."""
        st = self._need()
        names = ("units", "channel_audit", "g02_scales", "g01_review_questions", "g01_decision_table",
                 "summary_pooled", "summary_per_zone", "nested_choices", "contrasts", "contrasts_precision",
                 "per_zone_deltas", "recommendations", "complement_pooled", "strata_recall", "rank_correlations",
                 "lambda_usage", "lineament_metrics", "ablations", "official_style_pooled", "unit_overlap")
        out = {n: st.load_table(n, "tables") for n in names}
        out["summary"] = st.load_json("reports/summary.json")
        out["manifest"] = st.load_json("manifests/manifest.json")
        return out

    # ------------------------------------------------------------------------------- figures
    def figure_unit(self) -> tuple | None:
        """(unit, raster rel path) of the evaluated unit with the most truth pixels (figures 1-11, 15)."""
        st = self._need()
        T = st.load_table("units", "tables")
        if not len(T) or "truth_px" not in T:
            return None
        for r in T.sort_values("truth_px", ascending=False).itertuples():
            rel = f"rasters/units/{r.task}_{int(r.zone)}.npz"
            if st.exists(rel):
                u = self.m9.m7.unit(f"{r.task}:{int(r.zone)}")
                if u is not None:
                    return u, rel
        return None

    def most_chosen(self, virtual: str) -> str | None:
        CH = self._need().load_table("nested_choices", "tables")
        if not len(CH) or "virtual" not in CH:
            return None
        c = CH[(CH.virtual == virtual) & CH.chosen.notna()].chosen.value_counts()
        return str(c.index[0]) if len(c) else None

    def figures(self) -> list[str]:
        from . import g02_figures as FX

        st = self._need()
        fc, ev_c = self.cfg["figures"], self.cfg["evaluation"]
        f0, d0 = float(ev_c["primary_area_frac"]), float(ev_c["primary_distance_m"])
        figs = {}
        S = st.load_table("summary_pooled", "tables")
        heads = ["B0", "B1", "B2", *G02_FAMILIES, "G02-F", "random"]
        if len(S):
            figs["12_performance_vs_distance"] = FX.perf_curves(S, "tol_m", f0, heads, "12. Performance vs distance "
                                                                f"tolerance (budget {100 * f0:g} % of the domain)")
            figs["13_performance_vs_area"] = FX.perf_curves(S, "area_frac", d0, heads, "13. Performance vs prediction "
                                                            f"area (distance {d0:g} m)")
        D = st.load_table("per_zone_deltas", "tables")
        if len(D):
            figs["14_per_fold_improvements"] = FX.per_fold(D, st.load_table("contrasts", "tables"))
        fu = self.figure_unit()
        if fu is not None:
            u, rel = fu
            keys = [k for k in st.raster_keys(rel) if not k.startswith(("G01-A@", "G01-B@", "G01-C@", "G01-D@"))]
            base, meta = st.load_raster_keys(rel, ["interior", "domain"])
            r0, _, c0, _ = meta["window"]
            truth = np.asarray(u.eval_truth(), bool) & (np.asarray(base["interior"]) > 0)
            known = np.asarray(u.known, bool) if u.known is not None else None
            half = int(fc["window_half_px"])
            rr, cc = np.nonzero(truth) if truth.any() else np.nonzero(np.asarray(base["domain"]) > 0)
            from .geology_figures import window

            wl = window(truth.shape, (int(np.median(rr)), int(np.median(cc))), half)
            wg = (wl[0] + r0, wl[1] + r0, wl[2] + c0, wl[3] + c0)
            crop = lambda a: None if a is None else np.asarray(a)[wl[0]:wl[1], wl[2]:wl[3]]   # noqa: E731
            A = {}
            for k in keys:                                          # one map at a time, cropped at once
                got = st.load_raster_keys(rel, [k])[0]
                if k in got:
                    A[k] = crop(got[k]).astype(np.float32) if np.issubdtype(got[k].dtype, np.floating) else crop(got[k])
                del got
            T, K = crop(truth), crop(known)
            del base
            grid = self.src.data.grid
            raw = {r: (None if self.raw(r) is None else self.raw(r)[wg[0]:wg[1], wg[2]:wg[3]])
                   for r in ("mag", "grav", "terrain")}
            lam_l = [w.label for w in self.lams]
            b2, gf = self.most_chosen("B2") or "G01-E", self.most_chosen("G02-F")
            figs["01_raw_fields_faults"] = FX.raw_fields(grid, wg, raw, T, K)
            figs["02_wavelet_energy"] = FX.energy(grid, wg, A, lam_l)
            figs["03_orientation_coherence"] = FX.orientation(grid, wg, A)
            pts = FX.pick_points(A, "G02-A", int(fc["n_profiles"]), int(self.cfg["seed"]))
            figs["04_cross_fault_profiles"] = FX.cross_profiles(grid, wg, raw, A, pts, self.src.pixel_m, T)
            figs["05_displacement_examples"] = FX.displacement_examples(grid, wg, raw, A, pts, self.src.pixel_m,
                                                                        self.cfg, T)
            figs["06_truncation_examples"] = FX.truncation_examples(grid, wg, A, lam_l, T, int(self.cfg["seed"]))
            figs["07_ridge_continuity"] = FX.continuity_maps(grid, wg, A, lam_l, T)
            figs["08_multiscale_persistence"] = FX.persistence_maps(grid, wg, A, T)
            key = f"{u.task}_{u.zone}"
            figs["09_candidate_network"] = FX.network_map(grid, wg, A, st, key, self.cfg["network"]["sources"][0], T, K)
            figs["10_g01_vs_g02"] = FX.g01_vs_g02(grid, wg, A, b2, gf, T, K, f0)
            figs["11_case_studies"] = FX.case_studies(grid, wg, A, b2, gf, T, f0, d0, self.src.pixel_m,
                                                      int(self.cfg["seed"]), int(fc["n_cases"]))
            figs["15_geological_explanations"] = FX.hypothesis_maps(grid, wg, A, T)
            del A, keys
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

    # ------------------------------------------------------------------------------- report
    def report(self) -> Path:
        st = self._need()
        res = self.results()
        ev_c, dc = self.cfg["evaluation"], self.cfg["decision"]
        f0, d0 = float(ev_c["primary_area_frac"]), float(ev_c["primary_distance_m"])
        mode = self.cfg["mode"]
        S, H, R = res["summary_pooled"], res["contrasts"], res["recommendations"]
        lines = [f"# G02 structural geology: run `{st.root.name}` ({mode})", "",
                 f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `gems_bench.m9.research` ({G02_VERSION}). Guide: "
                 "`docs/m9/G02_STRUCTURAL_GEOLOGY.md`. G01 review: `reports/G01_REVIEW.md`.", "",
                 "> Every number below was produced by this run. Evidence maps are hypothesis scores, not calibrated "
                 "probabilities. The held-out zones were inspected in earlier GEMS work (not an untouched test set). "
                 "Separate from the official competition score.", ""]
        if mode != "FULL_EVALUATION":
            lines += ["> **FAST_RESEARCH: exploratory subset of zones / scales. No recommendation below can be KEEP; "
                      "run FULL_EVALUATION before any integration decision.**", ""]
        if self.warnings:
            lines += ["## Warnings", ""] + [f"* {w}" for w in self.warnings] + [""]
        lines += ["## Recommendations (from incremental contrasts only)", "", _md(R) if len(R) else "_(none)_", ""]
        if len(H):
            cols = [c for c in ("family", "kind", "task", "candidate", "reference", "delta", "ci_lo", "ci_hi",
                                "p_delta_positive", "zone_win_share", "n_zones", "verdict") if c in H]
            lines += [f"## Contrasts ({dc['metric']}, budget {100 * f0:g} %, distance {d0:g} m; paired block bootstrap)",
                      "", _md(H[cols]), ""]
        if len(S):
            heads = ["B0", "B1", "B2", *G02_FAMILIES, "G02-F", "random"]
            s = S[(S.area_frac == f0) & (S.tol_m == d0) & S.candidate.isin(heads)]
            cols = [c for c in ("task", "candidate", "precision", "recall", "f1", "enrichment", "precision_vs_edge",
                                "f1_lo", "f1_hi") if c in s]
            lines += ["## Pooled held-out results", "", _md(s[cols]), ""]
        lines += ["## Decision questions (answered from the tables above)", ""] + self._answers(res, f0, d0) + [""]
        p = st.path("reports/G02_REPORT.md")
        from ...artifacts.atomic import atomic_write_text

        atomic_write_text(p, "\n".join(lines))
        st.record("reports/G02_REPORT.md", "report_md", ["tables/*"], {})
        st.mark_done("report")
        st.flush_manifest()
        self.push()
        return p

    def _answers(self, res: dict, f0: float, d0: float) -> list[str]:
        H, C, Lu, Rc = res["contrasts"], res["complement_pooled"], res["lambda_usage"], res["rank_correlations"]
        R = res["recommendations"]

        def ctr(cand, ref):
            if not len(H):
                return "ABSENT"
            h = H[(H.candidate == cand) & (H.reference == ref)]
            if not len(h):
                return "ABSENT"
            return "; ".join(f"{r.task}: {r.verdict} (delta {getattr(r, 'delta', np.nan):+.4f}, "
                             f"P>0 {getattr(r, 'p_delta_positive', np.nan):.2f})" for r in h.itertuples())

        def comp(cand, ref="edge"):
            if not len(C) or "recovery_vs_random" not in C:
                return "ABSENT"
            c = C[(C.candidate == cand) & (C.reference == ref)]
            return "; ".join(f"{r.task}: recovers {r.recovered_share:.3f} of truth missed by {ref} "
                             f"({r.recovery_vs_random:.2f}x random), new-pick precision {r.new_pick_precision:.3f}"
                             for r in c.itertuples()) or "ABSENT"

        def corr(a, b):
            if not len(Rc):
                return "ABSENT"
            r = Rc[((Rc.a == a) & (Rc.b == b)) | ((Rc.a == b) & (Rc.b == a))]
            return f"median Spearman {r.spearman.iloc[0]:.2f}" if len(r) else "ABSENT"

        lam = "ABSENT"
        if len(Lu):
            lam = "; ".join(f"{r.task} {r.candidate} {r.lambda_km:g} km: {int(r.n_selected)} px, precision "
                            f"{r.precision:.3f}" for r in Lu.itertuples())
        rec = {r.family: r.recommendation for r in R.itertuples()} if len(R) else {}
        return [
            f"1. **Does displacement find faults missed by gradient detection?** {comp('G02-A')}",
            f"2. **Does truncation provide independent information?** increment: {ctr('F:G01-E+G02-B', 'G01-E')}; "
            f"redundancy with G01-E {corr('G02-B', 'G01-E')}, with the edge {corr('G02-B', 'edge')}",
            f"3. **Does persistence improve reliability?** G01-E vs G01-D: {ctr('G01-E', 'G01-D')}; G02-C increment: "
            f"{ctr('F:G01-E+G02-C', 'G01-E')}",
            "4. **Can geological continuity reduce false positives?** see `contrasts_precision` (precision deltas of "
            "the incremental pairs) and G02-D vs net[edge]: " + ctr("G02-D", "net[edge]"),
            f"5. **Which wavelength ranges are most informative?** G02-C narrow vs broad: "
            f"{ctr('G02-C[narrow]', 'G02-C[broad]')}; G02-A/B argmax-wavelength usage: {lam}; B1 nested choices in "
            "`nested_choices`",
            f"6. **Do magnetic and gravity contribute differently?** G02-A: {ctr('G02-A', 'G02-A[mag]')}; G02-B: "
            f"{ctr('G02-B', 'G02-B[mag]')}",
            "7. **Are the physical features complementary to M9?** increments over G01-F (M9 engineered layers): "
            + "; ".join(f"{X}: {ctr(f'F:G01-F+{X}', 'G01-F')}" for X in G02_FAMILIES)
            + (" | over the M9 prediction: " + "; ".join(f"{X}: {ctr(f'F:M9pred+{X}', 'M9pred')}" for X in G02_FAMILIES)
               if self.cfg["baselines"].get("m9_prediction_npy") else " | M9 prediction not supplied"),
            "8. **Is any improvement consistent across geographic regions?** `zone_win_share` in the contrasts and "
            "figure 14 (per-fold deltas)",
            f"9. **Does any improvement justify integration?** recommendations: {rec or 'none'}. Integrate only a "
            "KEEP from FULL_EVALUATION; INVESTIGATE / UNVERIFIED never enter the submission pipeline.",
        ]

    def run(self) -> dict:
        out = {"prepare": self.prepare()}
        out["units"] = self.evaluate()
        out["metrics"] = self.metrics()
        out["figures"] = self.figures()
        out["report"] = self.report()
        return out


__all__ = ["G02Experiment", "MapStore", "LazyScores", "fusion_name", "parse_fusion"]
