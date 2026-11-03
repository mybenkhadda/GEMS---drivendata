"""High-recall, label-free candidate structures with provenance.

Independent pools per unit (crop of the 100 m grid; each = the exact top-``frac`` eligible pixels of one score,
deterministic tie-breaking): ``w2`` (honest held-out W2), ``cnn`` (B0 / B1 rank fusion), ``surface`` (honest surface
model OOF or the label-free DEM scarp score), ``subsurface`` (honest G07 / G06 geology OOF), ``magnetic`` / ``gravity``
(G06 gradient magnitude), ``structural`` (G06 coherence), ``persistence`` (multiscale edge persistence).
Pools are skeleton-branched and de-duplicated (``p6.segment_recovery.dedup_inventory`` via ``g06.datasets
.generate_candidates``); every candidate keeps the share of its pixels inside every pool (``on_<pool>``) - the
source-expert identity. Candidate GENERATION never reads a label; the W2 PIXEL map is never replaced by candidate
means (the portfolio adds whole candidates on top of an exact-k W2 base).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .representation import exact_k_mask


def _pool(score, elig, frac, offset, width_full, seed) -> np.ndarray:
    m, _ = exact_k_mask(score, elig, frac, offset=offset, width_full=width_full, seed=seed)
    return m


def unit_scores(engine, store, vu, cfg: dict, model_maps: dict | None = None) -> tuple[dict, dict]:
    """{pool: score crop} + {pool: provenance} for the configured pools that are available for this unit."""
    from ..gems3.representation import rank01

    u, b = vu.unit, vu.base
    win = (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
    P = cfg.get("pools") or {}
    S, prov = {}, {}
    if P.get("w2"):
        try:
            S["w2"], prov["w2"] = engine.w2_rank(b), engine.provenance(b)["w2"]
        except (FileNotFoundError, ValueError) as e:
            prov["w2"] = f"UNAVAILABLE: {e}"
    if P.get("cnn"):
        try:
            b0, b1 = engine.cnn(b)
            S["cnn"] = 0.5 * (rank01(b0, u.ev) + rank01(b1, u.ev))
            prov["cnn"] = engine.provenance(b)["cnn_b0"]
        except (FileNotFoundError, ValueError) as e:
            prov["cnn"] = f"UNAVAILABLE: {e}"
    for name in ("magnetic", "gravity", "structural", "persistence", "surface_label_free"):
        pc = P.get(name)
        if pc and pc.get("feature") in store.names:
            S[name] = store.load([pc["feature"]], win)[0]
            prov[name] = f"store:{pc['feature']}"
    for name, m in (model_maps or {}).items():
        if P.get(name) and m is not None:
            S[name], prov[name] = m, f"honest_prediction:{name}"
    return S, prov


def generate_w2_candidates(S: dict, elig, cfg, **kw) -> dict:
    return {"w2": _pool(S["w2"], elig, float(cfg["pools"]["w2"]["frac"]), **kw)} if "w2" in S else {}


def generate_surface_candidates(S: dict, elig, cfg, **kw) -> dict:
    out = {}
    for k in ("surface", "surface_label_free"):
        if k in S:
            out[k] = _pool(S[k], elig & np.isfinite(S[k]), float(cfg["pools"][k]["frac"]), **kw)
    return out


def generate_subsurface_candidates(S: dict, elig, cfg, **kw) -> dict:
    out = {}
    for k in ("subsurface", "cnn", "magnetic", "gravity", "structural", "persistence"):
        if k in S:
            out[k] = _pool(S[k], elig, float(cfg["pools"][k]["frac"]), **kw)
    return out


def merge_candidate_inventory(pools: dict, cfg: dict) -> np.ndarray:
    """One de-duplicated, length-limited label raster from every pool (``g06.datasets.generate_candidates``)."""
    from ..g06.datasets import generate_candidates

    return generate_candidates({k: v for k, v in pools.items() if np.asarray(v).any()} or
                               {"empty": np.zeros(next(iter(pools.values())).shape, bool)}, cfg)


def deduplicate_candidates(lab: np.ndarray, pools: dict) -> pd.DataFrame:
    """Provenance of the merged inventory: per label, share of pixels inside each pool and the dominant source."""
    from ..g06.datasets import _means

    n = int(lab.max())
    if not n:
        return pd.DataFrame(columns=["label", "dominant_source", "n_sources"])
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    T = pd.DataFrame({"label": np.arange(1, n + 1)})
    for k, m in pools.items():
        T[f"on_{k}"], _ = _means(lb, np.asarray(m)[rr, cc].astype(np.float64), n)
    on = [c for c in T if c.startswith("on_")]
    T["n_sources"] = (T[on].fillna(0) >= 0.5).sum(1)
    T["dominant_source"] = T[on].fillna(0).idxmax(1).str[3:] if on else ""
    return T


def generate_unit_candidates(engine, store, vu, cfg: dict, model_maps: dict | None, seed: int,
                             width_full: int) -> dict:
    """Pools, merged inventory and provenance of one unit (label-free)."""
    u = vu.unit
    elig = u.scored_mask() & engine.template_valid()[u.sl]
    S, prov = unit_scores(engine, store, vu, cfg, model_maps)
    kw = {"offset": (u.sl[0].start, u.sl[1].start), "width_full": width_full, "seed": seed}
    pools = {**generate_w2_candidates(S, elig, cfg, **kw), **generate_surface_candidates(S, elig, cfg, **kw),
             **generate_subsurface_candidates(S, elig, cfg, **kw)}
    lab = merge_candidate_inventory(pools, cfg) if pools else np.zeros(u.ev.shape, np.int32)
    return {"scores": S, "pools": pools, "lab": lab, "provenance": prov, "dedup": deduplicate_candidates(lab, pools)}
