"""E6 evaluation: canonical DTI on geographic folds and withheld fault systems, pooled by summing components.

For every unit the final E6System predicts its scored region (cached in ``predictions/units/<unit>_eval.npy``,
keyed by the stage-E checkpoint sha256). Two fixed output policies are scored - the probabilities as they are
(``continuous``) and the submission's area-budget binary rule applied inside the scored region
(``area_budget_binary``); nothing is tuned on these numbers. Slices: all, DEM-covered (coverage >= 0.5),
partially covered (0 < coverage < 0.5), uncovered (coverage == 0, where E6 equals the frozen reference).
Short-fault recall: mean TP support of the truth segments shorter than ``evaluation.short_fault_km``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from . import metric
from .cache import atomic_save_npy, atomic_write_csv, atomic_write_json, load_npy, read_json
from .inference import bbox_of, predict, system_fn
from .paths import E6V2Paths
from .spatial_splits import DISC_UNITS, GEO_UNITS
from .submission import area_fraction, budget_select

log = logging.getLogger("e6v2.evaluator")
SLICES = ("all", "dem_full", "dem_partial", "no_dem")


def slice_mask(name: str, cov: np.ndarray) -> np.ndarray:
    if name == "all":
        return np.ones(cov.shape, bool)
    if name == "dem_full":
        return cov >= 0.5
    if name == "dem_partial":
        return (cov > 0) & (cov < 0.5)
    return cov <= 0


def unit_prediction(paths: E6V2Paths, tr, cfg: dict) -> tuple[np.ndarray, tuple, str]:
    ev = tr.masks["eval"]
    box = bbox_of(ev)
    sysm, sha = tr.load_system()
    out = paths.predictions / "units" / f"{tr.unit}_eval.npy"
    js = read_json(out.with_suffix(".json"))
    if js and js.get("checkpoint_sha256") == sha and js.get("bbox") == list(box) and out.exists():
        return load_npy(out), box, sha
    prob = predict(system_fn(sysm), tr.data, box, cfg, tr.device, tr.amp).astype(np.float32)
    atomic_save_npy(out, prob)
    atomic_write_json(out.with_suffix(".json"), {"unit": tr.unit, "checkpoint_sha256": sha, "bbox": list(box),
                                                 "dataset_fingerprint": tr.data.fingerprint})
    return prob, box, sha


def segment_recall(sup: np.ndarray, truth: np.ndarray, px_km: float, short_km: float, thr: float) -> list[dict]:
    from skimage.morphology import skeletonize

    lab, n = ndimage.label(truth, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    n_px = ndimage.sum(np.ones_like(sup), lab, idx)
    s_sum = ndimage.sum(sup, lab, idx)
    sk = ndimage.sum(skeletonize(truth).astype(float), lab, idx)
    length = np.maximum(sk, 1) * px_km * 1.1
    rec = s_sum / np.maximum(n_px, 1)
    rows = []
    for name, m in (("short", length < short_km), ("long", length >= short_km), ("all", np.ones(n, bool))):
        rows.append({"length_bin": name, "n_segments": int(m.sum()), "truth_px": int(n_px[m].sum()),
                     "support_sum": float(s_sum[m].sum()),
                     "weighted_recall": float(s_sum[m].sum() / max(n_px[m].sum(), 1)),
                     "segments_recovered": int((rec[m] >= thr).sum()),
                     "recovered_share": float((rec[m] >= thr).mean()) if m.any() else float("nan")})
    return rows


@torch.no_grad()
def gate_statistics(tr, sysm, n_patches: int = 64) -> list[dict]:
    """Mean gate weights per level and coverage bin on patches drawn from the unit's DEM-covered candidates."""
    pool = tr.sampler.with_dem if len(tr.sampler.with_dem) else tr.sampler.all
    rng = np.random.default_rng(tr.seed + 11)
    sel = pool[rng.choice(len(pool), min(n_patches, len(pool)), replace=False)]
    bins = [0.0, 1e-9, 0.25, 0.5, 0.75, 1.0001]
    acc = {}
    for i in range(0, len(sel), 8):
        items = [tr.data.inputs(int(r), int(c), tr.sampler.P, tr.sampler.ext) for r, c in sel[i:i + 8]]
        b = {k: torch.from_numpy(np.stack([it[k] for it in items]).astype(np.float32)).to(tr.device) for k in items[0]}
        out = sysm.e6(b["terrain"], b["field"], b["dem"], b["cov"], b["ctx"], aux=False)
        for lvl, w in enumerate(out["gates"]):
            m = torch.nn.functional.avg_pool2d(b["cov"], 2 ** lvl) if lvl else b["cov"]
            m, w = m.flatten().cpu().numpy(), w.float().permute(1, 0, 2, 3).reshape(3, -1).cpu().numpy()
            for j in range(len(bins) - 1):
                sel_j = (m >= bins[j]) & (m < bins[j + 1])
                if sel_j.any():
                    a = acc.setdefault((lvl, j), [0.0, 0.0, 0.0, 0])
                    for q in range(3):
                        a[q] += float(w[q, sel_j].sum())
                    a[3] += int(sel_j.sum())
    rows = []
    for (lvl, j), (wt, wf, wd, n) in sorted(acc.items()):
        rows.append({"unit": tr.unit, "level": lvl, "coverage_lo": bins[j], "coverage_hi": min(bins[j + 1], 1.0),
                     "w_terrain": wt / n, "w_field": wf / n, "w_dem": wd / n, "px": n})
    return rows


def evaluate(paths: E6V2Paths, cfg: dict, trainers: dict) -> dict:
    px = float(next(iter(trainers.values())).data.schema["grid"]["pixel_m"])
    ev_cfg = cfg["evaluation"]
    comp_rows, short_rows, gate_rows, hist = [], [], [], []
    for unit in GEO_UNITS + DISC_UNITS:
        tr = trainers[unit]
        H, W = tr.data.H, tr.data.W
        cov_full = tr.data.dem_coverage[:H, :W]
        valid = tr.data.valid[:H, :W]
        prob, (r0, r1, c0, c1), sha = unit_prediction(paths, tr, cfg)
        sl = (slice(r0, r1), slice(c0, c1))
        ev = tr.masks["eval"][sl] & valid[sl]
        truth = tr.masks["truth"][sl]
        ex = tr.masks["exclude"][sl] if tr.masks["exclude"] is not None else None
        cov = cov_full[sl]
        frac = area_fraction(cfg, tr.tg.label_density)
        elig = ev & (~ex if ex is not None else True)
        binary = budget_select(prob, elig, int(round(frac * ev.sum()))).astype(np.float32)
        protocol = "geographic" if unit in GEO_UNITS else "discovery"
        for policy, p in (("continuous", prob), ("area_budget_binary", binary)):
            for s in SLICES:
                m = ev & slice_mask(s, cov)
                c = metric.components(p, truth, m, ex, px=px)
                comp_rows.append({"protocol": protocol, "unit": unit, "policy": policy, "slice": s,
                                  "eval_px": int(m.sum()), "area_fraction": frac if policy != "continuous" else None,
                                  "checkpoint_sha256": sha, **c.row()})
            sup = metric.support(p, ev, ex, px=px)
            for r in segment_recall(sup, truth & ev, px / 1000.0, ev_cfg["short_fault_km"],
                                    ev_cfg["system_recovered_support"]):
                short_rows.append({"protocol": protocol, "unit": unit, "policy": policy, **r})
        sysm, _ = tr.load_system()
        gate_rows += gate_statistics(tr, sysm)
        hp = tr.ckpt.dir / "history.csv"
        if hp.exists():
            hist.append(pd.read_csv(hp))
        del sysm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    comp = pd.DataFrame(comp_rows)
    pooled = []
    for (proto, pol, s), g in comp.groupby(["protocol", "policy", "slice"], sort=False):
        c = metric.pooled([metric.Components(r.tp_w, r.fp_w, r.fn_w, int(r.n_truth), r.pred_mass)
                           for r in g.itertuples()])
        pooled.append({"protocol": proto, "unit": "pooled", "policy": pol, "slice": s,
                       "eval_px": int(g.eval_px.sum()), **c.row()})
    comp_all = pd.concat([comp, pd.DataFrame(pooled)], ignore_index=True)
    m = paths.metrics
    outs = {
        "dti_components": atomic_write_csv(m / "dti_components.csv", comp_all),
        "geographic_dti": atomic_write_csv(m / "geographic_dti.csv",
                                           comp_all[(comp_all.protocol == "geographic") & (comp_all.slice == "all")]),
        "discovery_dti": atomic_write_csv(m / "discovery_dti.csv",
                                          comp_all[(comp_all.protocol == "discovery") & (comp_all.slice == "all")]),
        "coverage_slices": atomic_write_csv(m / "coverage_slices.csv", comp_all[comp_all.unit == "pooled"]),
        "short_fault_recall": atomic_write_csv(m / "short_fault_recall.csv", _pool_short(pd.DataFrame(short_rows))),
        "gate_weights": atomic_write_csv(m / "gate_weights.csv", pd.DataFrame(gate_rows)),
        "training_history": atomic_write_csv(m / "training_history.csv",
                                             pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()),
    }
    head = comp_all[(comp_all.unit == "pooled") & (comp_all.slice == "all")][["protocol", "policy", "dti", "tp_w",
                                                                              "fp_w", "fn_w", "n_truth"]]
    atomic_write_json(m / "evaluation_summary.json", {"pooled": head.to_dict("records"),
                                                      "units": {u: trainers[u].identity for u in trainers}})
    return {"pooled": head, "outputs": outs}


def _pool_short(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["protocol", "policy", "length_bin"], sort=False).agg(
        n_segments=("n_segments", "sum"), truth_px=("truth_px", "sum"), support_sum=("support_sum", "sum"),
        segments_recovered=("segments_recovered", "sum")).reset_index()
    g["unit"] = "pooled"
    g["weighted_recall"] = g.support_sum / g.truth_px.clip(lower=1)
    g["recovered_share"] = g.segments_recovered / g.n_segments.clip(lower=1)
    return pd.concat([df, g], ignore_index=True)
