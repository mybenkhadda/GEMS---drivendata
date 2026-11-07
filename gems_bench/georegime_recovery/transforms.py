"""Section 14 - at most four PREDECLARED continuous-score policies (config['transforms']; no sweep).

identity    q = p
gamma_0.7   q = p^0.7                       (mild recall expansion)
gamma_1.4   q = p^1.4                       (mild concentration)
geo_support q = clip(p (a + b S), 0, 1)     a = 0.9, b = 0.2  ->  factor in [0.9, 1.1]
            S = robust 0-1 rescale (5th-95th percentile over the eligible footprint) of the mean of the
            label-free Atlas layers ``xm_evidence_mean`` and ``xm_multiscale_persistence``; where no layer is
            finite S = 0.5 (factor 1.0, i.e. unchanged) - missing evidence is never weak evidence.
No mapped-fault distance, no label-derived cluster id, no hidden truth, no floor, no binary mask.
Every policy is scored on every fold; the choice happens only in :mod:`crossfit` from OTHER folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as MX

FORBIDDEN_TOKENS = ("fault", "label", "dist", "cluster", "truth", "proximity")


def support_raster(evidence, valid: np.ndarray, pol: dict):
    for n in pol["support_features"]:
        if any(t in n for t in FORBIDDEN_TOKENS):
            raise ValueError(f"support feature {n} is not label-free")
    layers = [evidence.get(n) for n in pol["support_features"]]
    used = [n for n, a in zip(pol["support_features"], layers) if a is not None]
    layers = [a for a in layers if a is not None]
    if not layers:
        return None, {"status": "UNAVAILABLE", "reason": "no support layer in the Atlas registry"}
    with np.errstate(invalid="ignore"):
        M = np.nanmean(np.stack(layers), axis=0)
    v = M[valid & np.isfinite(M)]
    lo, hi = np.quantile(v, pol["robust_quantiles"])
    S = np.clip((M - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)
    S[~valid] = np.nan
    info = {"status": "OK", "layers": used, "atlas_features": [evidence.source.get(n) for n in used],
            "q_lo": float(lo), "q_hi": float(hi), "finite_fraction": float(np.isfinite(S[valid]).mean()),
            "label_free": True}
    return S, info


def apply(pol: dict, p: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
    fin = np.isfinite(p)
    x = np.clip(np.where(fin, p, 0.0), 0, 1).astype(np.float32)
    if pol["kind"] == "identity":
        q = x
    elif pol["kind"] == "gamma":
        q = np.power(x, float(pol["gamma"]))
    elif pol["kind"] == "support":
        if S is None:
            raise ValueError("support policy without a support raster")
        s = np.where(np.isfinite(S), S, float(pol["missing_support"]))
        q = np.clip(x * (float(pol["a"]) + float(pol["b"]) * s), 0, 1)
    else:
        raise ValueError(pol["kind"])
    return np.where(fin, q, np.nan).astype(np.float32)


def response_curves(policies) -> pd.DataFrame:
    p = np.linspace(0, 1, 201, dtype=np.float32)
    rows = []
    for pol in policies:
        for s in ([0.0, 0.5, 1.0] if pol["kind"] == "support" else [None]):
            S = None if s is None else np.full_like(p, s)
            q = apply(pol, p, S)
            rows += [{"policy": pol["name"], "support": s, "p": float(a), "q": float(b)} for a, b in zip(p, q)]
    return pd.DataFrame(rows)


def evaluate(units: dict, preds: dict, policies, S, valid, region, exclusion: str, block_px: int):
    """``preds``: unit -> full-area prob; returns (components, blocks, regions)."""
    comp, blocks, regions = [], [], []
    bid, _ = MX.block_ids(valid.shape, block_px)
    nb = int(bid.max()) + 1
    vb = np.bincount(bid.ravel(), weights=valid.ravel().astype(float), minlength=nb)
    present = np.flatnonzero(vb > 0)
    rid = np.where(region >= 0, region, region.max() + 1)
    nr = int(rid.max()) + 1
    for u, p in preds.items():
        U = units[u]
        proto = "DISCOVERY_FULL_AREA" if U.protocol == "WITHHELD_SYSTEM" else "GEOGRAPHIC_FULL_AREA"
        cm = MX.ComponentMaps(U.truth, valid, U.exclusion(exclusion))
        for pol in policies:
            q = apply(pol, p, S)
            mp = cm.maps(q)
            c = mp["components"]
            comp.append(MX.row(c, policy=pol["name"], unit=u, protocol=proto, exclusion=exclusion))
            bs = MX.block_sums(mp, bid, nb)
            blocks.append(pd.DataFrame({"policy": pol["name"], "unit": u, "protocol": proto, "block": present,
                                        **{k: v[present] for k, v in bs.items()}}))
            rs = MX.block_sums(mp, rid.astype(np.int32), nr)
            for r in range(nr):
                if rs["n_truth"][r] > 0 or rs["fp_w"][r] > 0:
                    regions.append({"policy": pol["name"], "unit": u, "protocol": proto,
                                    "region": int(r) if r < nr - 1 else -1, "tp_w": rs["tp_w"][r], "fp_w": rs["fp_w"][r],
                                    "fn_w": rs["fn_w"][r], "n_truth": rs["n_truth"][r]})
    return pd.DataFrame(comp), pd.concat(blocks, ignore_index=True), pd.DataFrame(regions)
