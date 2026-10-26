"""Section 02: the geological feature pyramid (reuse first, add only what is missing, never concatenate blindly).

Inputs are the ~139 cached feature entries of the Phase-2 / EDA / Phase-4 / Phase-5 store (memory-mapped, never
rebuilt unless ``features.build_missing``) plus a few NEW label-free cross-modal layers that do not exist yet:

    m9:xmod:mag_grav_tol@<tol>m        magnetic edge strength x max(gravity edge strength within <tol> m)
                                       (spatially tolerant: gravity maxima need not coincide; ~2 km resolution)
    m9:xmod:terr_mag_tol@<tol>m        surface (terrain) / subsurface (magnetic) tolerant edge alignment
    m9:xmod:mag_grav_orient@<s>km      c_m c_g cos 2(theta_m - theta_g) of doubled-angle fields smoothed at the
                                       tolerance scale (axial, 180 deg periodic; magnitude-weighted)
    m9:xmod:mag_over_grav@<s>km        signed log ratio of rank-normalised magnetic / gravity edge strength
                                       (which modality dominates; never requires both to agree)
    m9:xmod:<mod>_lcn@<s>km            THDR / local RMS(THDR): localized physical contrast (weak edges in quiet areas)
    m9:raw:<band>                      robust-standardised raw band, only when the store's raw:<band> is not cached

Channel groups handed to the network (``channel_groups``): ``raw`` (local raw fields), ``struct`` (selected
structural families + new cross-modal evidence), ``spectral`` (section 03, gated), ``context`` (regional context
features and context bands; low-resolution branch only: strain / seismicity are regional descriptors, not fault
locators) and optional ``dem``.

Redundancy: per family, a greedy label-free Spearman filter on a fixed random sample of valid pixels; channels of the
preferred legacy spec (B0's P4_B_gated) are kept first. Supervised value is NEVER used to choose channels here: the
nested ``family_screen`` (pixel LightGBM trained on each zone's training geography, scored on its held-out zone with
the official scorer) is a diagnostic and the evidence of Gate 1.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

PIXEL_KM = 0.1


# =============================================================================== layer cache (new M9 layers)
class LayerCache:
    """Store-like cache of M9 layers: ``cache/m9/layers/<name>.npy`` (float32, memory-mapped on read)."""

    def __init__(self, root: Path):
        self.dir = Path(root) / "layers"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mm: dict = {}

    def path(self, name: str) -> Path:
        return self.dir / (re.sub(r"[^A-Za-z0-9_@.-]+", "_", name) + ".npy")

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def get(self, name: str) -> np.ndarray:
        if name not in self._mm:
            self._mm[name] = np.load(self.path(name), mmap_mode="r")
        return self._mm[name]

    def put(self, name: str, a: np.ndarray) -> Path:
        from ..artifacts.atomic import atomic_save

        self._mm.pop(name, None)
        return atomic_save(self.path(name), lambda t: np.save(t, np.asarray(a, np.float32)))

    def release(self) -> None:
        self._mm.clear()


def file_fp(p: Path) -> str:
    """Cheap content fingerprint of a cached array: size + sha1 of the first and last MiB."""
    p = Path(p)
    if not p.exists():
        return "absent"
    h = hashlib.sha1(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(p.stat().st_size - (1 << 20), 0))
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


# =============================================================================== reading any channel
def read(m9, name: str) -> np.ndarray:
    """(H, W) array of a channel: M9 layer cache, then the feature store, then a raw band (standardised once)."""
    if name.startswith("m9:"):
        return m9.layers.get(name)
    st = m9.ctx.store
    if name in st.specs and st.has(name):
        return st.get(name)
    if name in st.specs and m9.cfg["features"]["build_missing"]:
        st.build([name], verbose=False)
        return st.get(name)
    if name.startswith("raw:"):
        alt = f"m9:{name}"
        if not m9.layers.has(alt):
            from gems_eda.enhance import standardize_signal

            a = m9.ctx.data.read_band(name[4:])
            a[~np.asarray(m9.ctx.data.valid, bool)] = np.nan
            z, _ = standardize_signal(a)
            m9.layers.put(alt, z)
        return m9.layers.get(alt)
    raise KeyError(f"channel {name} is neither cached nor buildable (features.build_missing = false)")


def available(m9, name: str) -> bool:
    if name.startswith("m9:"):
        return m9.layers.has(name)
    st = m9.ctx.store
    if name in st.specs and (st.has(name) or m9.cfg["features"]["build_missing"]):
        return True
    return name.startswith("raw:") and name[4:] in m9.ctx.data.band_names


# =============================================================================== inventory + families
def family_of(name: str, rules: list[dict]) -> str:
    for r in rules:
        if re.search(r["pattern"], name):
            return r["family"]
    return "other"


def inventory(m9) -> pd.DataFrame:
    """Every store entry with family, provenance, cache state, resolution and a file fingerprint."""
    st = m9.ctx.store
    rules = m9.cfg["features"]["families"]
    rows = []
    for n, s in st.specs.items():
        cached = bool(st.has(n))
        p = st.path(n) if hasattr(st, "path") else None
        rows.append({"name": n, "family": "raw" if n.startswith("raw:") else family_of(n, rules),
                     "modality": getattr(s, "modality", ""), "source": getattr(s, "source", ""),
                     "transform": getattr(s, "transform", ""), "scale_km": float(getattr(s, "scale_km", 0.0) or 0.0),
                     "units": getattr(s, "units", "robust-z"), "support_km": float(getattr(s, "support_km", 0.0) or 0.0),
                     "origin": getattr(s, "origin", "native"), "kind": getattr(s, "kind", "local"),
                     "phase": ("phase5" if n.startswith(("p5:", "dem:")) else "phase4" if n.startswith("rs:")
                               else "eda" if n.startswith("eda:") else "phase2"),
                     "cached": cached, "fingerprint": file_fp(p) if (cached and p is not None) else "absent"})
    return pd.DataFrame(rows)


def _sample_idx(valid: np.ndarray, n: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(valid.ravel())
    if idx.size > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return idx


def _ranks(m9, names: list[str], idx: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    out = np.empty((len(names), idx.size), np.float32)
    for j, n in enumerate(names):
        v = np.asarray(read(m9, n)).reshape(-1)[idx].astype(np.float64)
        v = np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0.0)
        out[j] = rankdata(v).astype(np.float32)
    return out


def redundancy_select(m9, inv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy per-family pruning (|Spearman| on a label-free sample); returns (decisions, correlation pairs)."""
    fc = m9.cfg["features"]
    valid = np.asarray(m9.ctx.data.valid, bool)
    idx = _sample_idx(valid, int(fc["redundancy"]["sample_px"]), int(m9.cfg["seed"]))
    thr = float(fc["redundancy"]["max_abs_spearman"])
    prefer = set(_preferred(m9))
    raw = [f"raw:{b}" for b in fc["raw_local"] if available(m9, f"raw:{b}")]
    excl = set(fc["exclude_families"])
    pats = [re.compile(p) for p in fc.get("exclude_patterns") or []]
    bad = inv.name.map(lambda n: any(p.search(n) for p in pats))
    cand = inv[(inv.family != "raw") & ~inv.family.isin(excl | {"other"}) & ~bad]
    cand = cand[[available(m9, n) for n in cand.name]]
    names = raw + list(cand.name)
    R = _ranks(m9, names, idx)
    C = np.corrcoef(R) if len(names) > 1 else np.ones((1, 1))
    pos = {n: i for i, n in enumerate(names)}
    decisions, pairs = [], []
    for n in raw:
        decisions.append({"name": n, "family": "raw", "selected": True, "reason": "raw physical field (always kept)"})
    kept_all = list(raw)
    for fam, g in cand.groupby("family", sort=False):
        order = sorted(g.name, key=lambda n: (n not in prefer, g.set_index("name").scale_km.get(n, 0.0)))
        kept = []
        for n in order:
            if len(kept) >= int(fc["max_per_family"]):
                decisions.append({"name": n, "family": fam, "selected": False, "reason": "family cap reached"})
                continue
            rho = [(k, abs(float(C[pos[n], pos[k]]))) for k in kept + raw]
            worst = max(rho, key=lambda x: x[1]) if rho else (None, 0.0)
            if worst[1] > thr:
                decisions.append({"name": n, "family": fam, "selected": False,
                                  "reason": f"|rho|={worst[1]:.3f} with {worst[0]}"})
                pairs.append({"a": n, "b": worst[0], "abs_spearman": worst[1]})
                continue
            kept.append(n)
            decisions.append({"name": n, "family": fam, "selected": True,
                              "reason": "preferred (legacy B0 channel)" if n in prefer else "kept (not redundant)"})
        kept_all += kept
    for _, r in inv[inv.family.isin(excl) | (inv.family == "other")].iterrows():
        decisions.append({"name": r["name"], "family": r["family"], "selected": False,
                          "reason": "family excluded from local inputs (context / optional)"})
    for _, r in inv[bad & ~inv.family.isin(excl | {"other", "raw"})].iterrows():
        decisions.append({"name": r["name"], "family": r["family"], "selected": False,
                          "reason": "excluded pattern (direction-dependent: not invariant under rot180)"})
    return pd.DataFrame(decisions), pd.DataFrame(pairs)


def _preferred(m9) -> list[str]:
    name = m9.cfg["features"].get("prefer")
    if not name:
        return []
    try:
        from ..pipeline.baselines import legacy_spec

        s = legacy_spec(name) or {}
        return list(s.get("channels", []))
    except Exception:
        return []


def correlation_matrix(m9, names: list[str]) -> pd.DataFrame:
    """Label-free Spearman matrix (long format) of the given channels (figure / audit)."""
    idx = _sample_idx(np.asarray(m9.ctx.data.valid, bool), min(int(m9.cfg["features"]["redundancy"]["sample_px"]),
                                                                  100_000), int(m9.cfg["seed"]) + 1)
    names = [n for n in names if available(m9, n)]
    if len(names) < 2:
        return pd.DataFrame()
    C = np.corrcoef(_ranks(m9, names, idx))
    return pd.DataFrame([{"a": a, "b": b, "spearman": float(C[i, j])} for i, a in enumerate(names)
                         for j, b in enumerate(names)])


# =============================================================================== new cross-modal evidence
def _edge(m9, band: str, scale_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rank-normalised THDR in [0,1], strike deg, coherence) of a raw band at ``scale_km`` (label-free)."""
    from ..m8.potential_fields import coherence, fill, hgrad, restore, sigma_px_of, strike_from_gradient
    from ..rs_features import prank

    valid = np.asarray(m9.ctx.data.valid, bool)
    a = m9.ctx.data.read_band(band)
    f, m = fill(a)
    s = sigma_px_of(scale_km * 1000.0 / 2.0)
    ge, gn = hgrad(f, s)
    th = np.hypot(ge, gn)
    r = prank(restore(th, m), valid & m, seed=int(m9.cfg["seed"]))
    st = strike_from_gradient(ge, gn)
    coh = coherence(ge, gn, 2.0 * s)
    return np.where(valid & m, np.nan_to_num(r), np.nan).astype(np.float32), st, coh


def crossmodal_layers(m9) -> pd.DataFrame:
    """Compute (once) the new label-free cross-modal layers into the M9 layer cache; returns their registry."""
    from ..m8.potential_fields import fill, operator, restore

    xc = m9.cfg["features"]["crossmodal"]
    if not xc.get("enabled", True):
        return pd.DataFrame()
    bands = m9.ctx.data.band_names
    valid = np.asarray(m9.ctx.data.valid, bool)
    rows = []

    def put(name, a, note, units="dimensionless"):
        if not m9.layers.has(name) or m9.cfg.get("overwrite_existing"):
            a = np.where(valid, a, np.nan).astype(np.float32)
            m9.layers.put(name, a)
        rows.append({"name": name, "family": "crossmodal_new", "units": units, "note": note,
                     "source_resolution_m": 100, "fingerprint": file_fp(m9.layers.path(name))})

    have = {k: b for k, b in (("mag", "rtp"), ("grav", "iso_grav_anom"), ("terr", "det_elev")) if b in bands}
    for s in xc["scale_km"]:
        E = {k: _edge(m9, b, float(s)) for k, b in have.items()}
        if "mag" in E and "grav" in E:
            for tol in xc["tolerance_m"]:
                r = max(int(round(float(tol) / 100.0)), 1)
                fp = ndimage.maximum_filter(np.nan_to_num(E["grav"][0]), footprint=_disk(r))
                put(f"m9:xmod:mag_grav_tol@{int(tol)}m_{s:g}km", np.sqrt(np.nan_to_num(E["mag"][0]) * fp),
                    f"geometric mean of magnetic edge rank and the max gravity edge rank within {tol} m (scale {s} km)")
            sig = max(float(max(xc["tolerance_m"])) / 100.0 / 2.0, 1.0)
            c_m, s_m = _doubled(E["mag"][1], E["mag"][2] * np.nan_to_num(E["mag"][0]), sig)
            c_g, s_g = _doubled(E["grav"][1], E["grav"][2] * np.nan_to_num(E["grav"][0]), sig)
            put(f"m9:xmod:mag_grav_orient@{s:g}km", c_m * c_g + s_m * s_g,
                "tolerant axial orientation agreement c_m c_g cos 2(dtheta) of smoothed doubled-angle fields")
            eps = 0.02
            put(f"m9:xmod:mag_over_grav@{s:g}km",
                np.log((np.nan_to_num(E["mag"][0]) + eps) / (np.nan_to_num(E["grav"][0]) + eps)),
                "signed log ratio of rank-normalised magnetic / gravity edge strength", "log ratio")
        if "terr" in E and "mag" in E:
            tol = int(min(xc["tolerance_m"]))
            r = max(int(round(tol / 100.0)), 1)
            fm = ndimage.maximum_filter(np.nan_to_num(E["mag"][0]), footprint=_disk(r))
            put(f"m9:xmod:terr_mag_tol@{tol}m_{s:g}km", np.sqrt(np.nan_to_num(E["terr"][0]) * fm),
                f"surface / subsurface: terrain edge rank x max magnetic edge rank within {tol} m")
        del E
    win = float(xc["contrast_window_km"])
    for k, b, s in (("mag", "rtp", 1.0), ("grav", "iso_grav_anom", 2.0)):
        if b not in bands:
            continue
        a = m9.ctx.data.read_band(b)
        lay, _ = operator("lcn", a, max(s * 10.0 / 2.0, 0.7), lcn_sigma_px=win * 10.0 / 2.0)
        put(f"m9:xmod:{k}_lcn@{s:g}km", np.nan_to_num(lay), "THDR / local RMS(THDR): localized contrast")
    m9.layers.release()
    return pd.DataFrame(rows)


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _doubled(strike_deg: np.ndarray, weight: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed, weight-normalised doubled-angle vector (cos 2t, sin 2t) * resultant length."""
    t = np.radians(np.nan_to_num(strike_deg))
    w = np.nan_to_num(weight)
    c = ndimage.gaussian_filter(w * np.cos(2 * t), sigma_px)
    s = ndimage.gaussian_filter(w * np.sin(2 * t), sigma_px)
    n = ndimage.gaussian_filter(w, sigma_px) + 1e-6
    return (c / n).astype(np.float32), (s / n).astype(np.float32)


# =============================================================================== channel groups
def selected_registry(m9) -> pd.DataFrame:
    return m9.ws.load_table("selected_registry", "features")


def channel_groups(m9, vcfg: dict | None = None) -> dict[str, list[str]]:
    """Channel groups of the network for one variant configuration (reads the section-02 / 03 registries)."""
    vcfg = vcfg or m9.cfg
    fc = vcfg["features"]
    reg = selected_registry(m9)
    if not len(reg):
        raise RuntimeError("no features/selected_registry: run section 02 first")
    sel = reg[reg.selected.astype(bool)]
    fam = dict(zip(sel.name, sel.family))
    g1 = m9.ws.load_table("gate1", "spectral")
    dec = dict(zip(g1.component, g1.decision)) if len(g1) and "component" in g1 else {}
    groups = {"raw": [n for n in sel.name if fam[n] == "raw"],
              "struct": [n for n in sel.name if fam[n] != "raw"],
              "spectral": [], "context": [], "dem": []}
    if dec.get("crossmodal_new") == "EXCLUDE":                     # Gate 1: new cross-modal layers earned no place
        groups["struct"] = [n for n in groups["struct"] if not n.startswith("m9:xmod:")]
    inc = vcfg["spectral"]["include_in_model"]
    spec_ok = inc == "always" or (inc == "auto" and dec.get("spectral") == "INCLUDE")
    if spec_ok and vcfg["spectral"].get("enabled", True):
        sp = m9.ws.load_table("spectral_registry", "spectral")
        groups["spectral"] = list(sp[sp.selected.astype(bool)].name) if len(sp) and "selected" in sp else []
    inv = m9.ws.load_table("inventory", "features")
    ctxf = set(fc["context_families"])
    groups["context"] = [f"raw:{b}" for b in fc["raw_context"] if available(m9, f"raw:{b}")]
    if len(inv):
        groups["context"] += [n for n in inv[inv.family.isin(ctxf) & inv.cached.astype(bool)].name]
    groups["context"] = list(dict.fromkeys(groups["context"]))
    if vcfg["model"].get("dem_branch"):
        groups["dem"] = [n for n in inv[inv.family == "dem"].name if available(m9, n)] if len(inv) else []
    return groups


# =============================================================================== nested family screen (diagnostic)
def _train_mask(m9, u) -> np.ndarray:
    ctx = m9.ctx
    valid = np.asarray(ctx.data.valid, bool)
    if u.task == "geo":
        return ctx.folds.train(u.zone) & valid
    return valid & ~ctx.splits.ignore(u.zone)


def _xy(m9, names: list[str], mask: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = np.asarray(m9.ctx.data.labels, bool)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero((mask & lab).ravel())
    neg = np.flatnonzero((mask & ~lab).ravel())
    pos = rng.choice(pos, min(pos.size, n // 2), replace=False) if pos.size else pos
    neg = rng.choice(neg, min(neg.size, n - pos.size), replace=False) if neg.size else neg
    idx = np.sort(np.r_[pos, neg])
    X = np.stack([np.asarray(read(m9, c)).reshape(-1)[idx] for c in names], 1).astype(np.float32)
    y = lab.reshape(-1)[idx].astype(np.int8)
    w = np.where(y == 1, 1.0, float(m9.cfg["data"]["pu"]["unlabeled_weight"]))
    return X, y, w


def _predict_unit(model, m9, names: list[str], u, chunk_rows: int = 256) -> np.ndarray:
    h, w = u.ev.shape
    out = np.zeros((h, w), np.float32)
    r0, c0 = u.sl[0].start, u.sl[1].start
    for a in range(0, h, chunk_rows):
        b = min(a + chunk_rows, h)
        X = np.stack([np.asarray(read(m9, c)[r0 + a:r0 + b, c0:c0 + w]) for c in names], -1).reshape(-1, len(names))
        out[a:b] = model.predict_proba(X.astype(np.float32))[:, 1].reshape(b - a, w)
    return np.where(u.ev, out, 0.0)


def family_screen(m9, sets: dict[str, list[str]], tag: str = "s02") -> pd.DataFrame:
    """Per unit (trained zones of the profile) and feature set: pixel LightGBM on the unit's TRAINING geography,
    held-out official-style DTI (binary top-frac, eligible budget) + AUC-PR + systems recovered."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    from ..gems3.representation import top_mask
    from ..m7.protocol import score_row, system_hits

    sc = m9.cfg["features"]["family_screen"]
    p = dict(sc["lgbm"])
    seed = int(p.pop("seed", 2026))
    rows, hits = [], []
    for task in ("geo", m9.prim):
        for u in m9.units_of(task):
            tm = _train_mask(m9, u)
            for sname, names in sets.items():
                names = [n for n in names if available(m9, n)]
                if not names:
                    continue
                X, y, w = _xy(m9, names, tm, int(sc["sample_px_per_zone"]), seed + u.zone)
                mdl = lgb.LGBMClassifier(random_state=seed, verbose=-1, **p).fit(X, y, sample_weight=w)
                s = _predict_unit(mdl, m9, names, u)
                el = m9.elig(u, 0)
                q = top_mask(s, el, float(sc["frac"])).astype(np.float32)
                r = score_row(u, q, el, m9.pkey(u), method=f"screen_{sname}", feature_set=sname,
                              n_features=len(names))
                G = u.eval_truth()
                sv = s[u.ev & ~(u.exclude if u.exclude is not None else np.zeros_like(u.ev))]
                gv = G[u.ev & ~(u.exclude if u.exclude is not None else np.zeros_like(u.ev))]
                r["auc_pr"] = float(average_precision_score(gv, sv)) if gv.any() else np.nan
                h = system_hits(m9.ctx, u, q, m9.sh)
                r["system_recall"] = float(h.recovered.mean()) if len(h) else np.nan
                rows.append(r)
                if len(h):
                    hits.append(h.assign(task=task, zone=u.zone, feature_set=sname))
                m9.say(f"{tag} screen {u.key} {sname}: dti={r['dti']:.4f} auc_pr={r['auc_pr']:.4f}")
            u.release()
    out = pd.DataFrame(rows)
    out.attrs["hits"] = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame()
    return out


def pooled_screen(t: pd.DataFrame) -> pd.DataFrame:
    from ..m7.protocol import dti_of

    if not len(t):
        return t
    return (t.groupby(["task", "feature_set"])
             .apply(lambda g: pd.Series({"dti_pooled": dti_of(g), "worst_zone_dti": float(g.dti.min()),
                                         "auc_pr_mean": float(g.auc_pr.mean()), "system_recall": float(g.system_recall.mean()),
                                         "n_features": int(g.n_features.iloc[0]), "n_zones": int(len(g))}))
             .reset_index())


def unique_systems(hits: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Systems recovered by feature set ``a`` and not by ``b`` (per task)."""
    if not len(hits):
        return pd.DataFrame()
    piv = hits.pivot_table(index=["task", "zone", "system_id"], columns="feature_set", values="recovered",
                           aggfunc="max").fillna(False)
    if a not in piv or b not in piv:
        return pd.DataFrame()
    m = piv[a].astype(bool) & ~piv[b].astype(bool)
    return m.groupby(level="task").sum().rename("unique_systems").reset_index().assign(set_a=a, set_b=b)
