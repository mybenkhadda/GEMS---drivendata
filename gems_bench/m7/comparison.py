"""Section 13: independent comparison of every method on identical units, masks and budget rules.

Inputs: the held-out maps every section saved (``validation/maps/<method>/<unit>.npz``; each zone's map was built
with choices made on the OTHER zones). Each map is re-scored here with the same FastScorer, so numbers from
different sections are comparable; ``protocol.assert_comparable`` refuses to put two maps of one (task, zone) side
by side when their masks / budget domain differ.

Per method: geographic DTI per fold, ``disc_official`` per split, the retained-FP alternative (same maps, retained
faults NOT masked), worst zone, weighted short-fault recall (mean TP support over pixels of systems < 2 km), system
recall, length recall (= weighted recall), predicted density, regional DTI (3 x 3 regions, geo), sensitivity to
known-fault proximity (recall of truth and FP mass by distance to the known faults, disc), paired per-zone deltas
and a paired spatial block bootstrap against the champion baseline. Protocols are never averaged together.

Honesty note: every number is nested-cross-fitted, but these zones were inspected in earlier sessions and the M7
grids were designed after them; they are NOT an untouched test set, and with 3 folds / 3 splits a paired delta
is descriptive evidence, not a significance test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..gems3.official import FastScorer
from .protocol import (COMP, ProtocolMismatch, assert_comparable, block_components, distance_km, dti_of,
                       paired_block_bootstrap, score_row, system_hits)
from .rasterize import load_heldout, saved_methods


def _short_recall(m7, u, q) -> tuple[float, float]:
    sup = u.scorer.support(np.asarray(q, np.float64))
    if not len(sup):
        return np.nan, 0.0
    from .protocol import system_ids

    sid = system_ids(m7.ctx, u)
    tab = m7.ctx.splits.systems.set_index("system_id").trace_km
    km = tab.reindex(sid).to_numpy()
    short = km < float(m7.cfg["evaluation"]["short_system_km"])
    return (float(sup[short].mean()) if short.any() else np.nan), float(short.sum())


def _proximity(m7, u, q) -> list[dict]:
    if u.known is None or not u.known.any():
        return []
    b1, b2 = m7.cfg["evaluation"]["proximity_bands_km"]
    dk = distance_km(u.known)
    band = np.digitize(dk, [b1, b2])
    labels = [f"<{b1:g} km", f"{b1:g}-{b2:g} km", f">{b2:g} km"]
    s = u.scorer
    rows = []
    p = np.clip(np.where(u.ev, np.nan_to_num(q, nan=0.0), 0.0), 0, 1)
    p[s.zero] = 0.0
    fp = p * s.fp_w
    sup = s.support(p, cleaned=True) if s.n else np.zeros(0)
    gr, gc = s.g_rc
    for i, lab in enumerate(labels):
        m = band == i
        tb = band[gr, gc] == i if s.n else np.zeros(0, bool)
        rows.append({"distance_band": lab, "truth_px": int(tb.sum()),
                     "recall_w": float(sup[tb].mean()) if tb.any() else np.nan,
                     "fp_mass": float(fp[m].sum()), "pred_px": int(((p > 0) & m).sum())})
    return rows


def _regions(m7, u, q) -> list[dict]:
    from ..geo_eval import region_map
    from gems_eda.morphology import region_name

    reg = m7.cache_get("region_map", lambda: region_map(m7.ctx.data))[u.sl]
    rows = []
    for r in np.unique(reg[u.ev & (reg >= 0)]):
        ev = u.ev & (reg == r)
        G = u.eval_truth() & ev
        if not G.any():
            continue
        c = FastScorer(G, ev, u.exclude, exclude_supports_tp=u.supports_tp).components(np.asarray(q, np.float64))
        rows.append({"region": region_name(int(r), (3, 3)), **{k: v for k, v in c.as_dict().items() if k in COMP + ["dti"]}})
    return rows


def evaluate(m7) -> dict:
    cfg = m7.cfg
    prim = cfg["evaluation"]["primary_task"]
    block_px = max(int(round(float(cfg["evaluation"]["block_km"]) / 0.1)), 8)
    per, alt, sysr, prox, regs, blocks = [], [], [], [], [], []
    methods = saved_methods(m7.ws)
    for mth in methods:
        for task in ("geo", prim):
            for u in m7.units_of(task):
                q, stored = load_heldout(m7.ws, mth, u, with_key=True)
                if q is None:
                    continue
                if q.shape != u.ev.shape:
                    raise ProtocolMismatch(f"{mth} {u.key}: saved map shape {q.shape} != unit crop {u.ev.shape}")
                # the stored key is what the map was built under; a mismatch with the current unit is refused below
                row = score_row(u, q, m7.elig(u, 0), stored or m7.pkey(u), method=mth)
                sr, n_short = _short_recall(m7, u, q)
                h = system_hits(m7.ctx, u, q, cfg)
                if len(h):
                    km = m7.ctx.splits.systems.set_index("system_id").trace_km.reindex(h.system_id).to_numpy()
                    row["system_recall"] = float(h.recovered.mean())
                    row["system_length_recall"] = float(np.nansum(km * h.recovered) / max(np.nansum(km), 1e-9))
                    row["n_systems"] = int(len(h))
                    sysr.append(h.assign(method=mth, task=task, zone=u.zone))
                row["short_recall_w"], row["short_truth_px"] = sr, n_short
                row["density_px_per_100km2"] = float((q > 0).sum() / max(u.ev.sum() * 0.01, 1e-9) * 100.0)
                per.append(row)
                blocks.append(block_components(u, q, block_px).assign(method=mth))
                if task == "geo":
                    regs += [{**r, "method": mth, "task": task, "zone": u.zone} for r in _regions(m7, u, q)]
                else:
                    prox += [{**r, "method": mth, "task": task, "zone": u.zone} for r in _proximity(m7, u, q)]
                    ua = m7.unit(f"disc_retained_fp:{u.zone}")
                    if ua is not None:
                        qa = m7.transfer(q, u, ua)
                        alt.append(score_row(ua, qa, m7.elig(ua, 0), m7.pkey(ua), method=mth, selected_on=task))
    P = pd.DataFrame(per)
    A = pd.DataFrame(alt)
    cur = m7.protocol_keys()
    assert_comparable(P, cur)          # every saved map must have been built under the current unit protocol
    assert_comparable(A)
    B = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    pooled = pool_methods(P, A)
    ref = champion_name(m7, methods)
    paired = paired_zones(P, ref)
    boot = []
    for task in ("geo", prim):
        bt = B[B.task == task] if len(B) else B
        for mth in methods:
            if mth == ref or not len(bt):
                continue
            boot.append({**paired_block_bootstrap(bt, mth, ref, int(cfg["evaluation"]["n_boot"]), int(cfg["seed"])),
                         "task": task})
    return {"per_zone": P, "retained_fp": A, "pooled": pooled, "paired": paired, "bootstrap": pd.DataFrame(boot),
            "systems": pd.concat(sysr, ignore_index=True) if sysr else pd.DataFrame(),
            "proximity": pool_proximity(pd.DataFrame(prox)), "regions": pool_regions(pd.DataFrame(regs)),
            "champion_reference": ref, "dem_restricted": m7.ws.load_table("pooled", "dem/candidates")}


def champion_name(m7, methods: list[str]) -> str:
    want = f"s01_{m7.cfg['comparison']['champion_reference']}"
    return want if want in methods else (methods[0] if methods else want)


def pool_methods(P: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    if not len(P):
        return P
    rows = []
    for (task, mth), g in P.groupby(["task", "method"]):
        r = {"task": task, "method": mth, "dti_pooled": dti_of(g), "worst_zone_dti": float(g.dti.min()),
             "n_zones": int(len(g)), "recall_w": float(g.tp_w.sum() / max(g.n_truth.sum(), 1)),
             "short_recall_w": float(np.nansum(g.short_recall_w * g.short_truth_px) / max(g.short_truth_px.sum(), 1)),
             "density_px_per_100km2": float(g.density_px_per_100km2.mean()),
             "area_frac_eligible_mean": float(g.area_frac_eligible.mean())}
        if "system_recall" in g:
            w = g.n_systems.fillna(0)
            r["system_recall"] = float(np.nansum(g.system_recall * w) / max(w.sum(), 1))
            r["system_length_recall"] = float(g.system_length_recall.mean())
        r.update({f"dti_zone{int(z)}": float(v) for z, v in zip(g.zone, g.dti)})
        if len(A) and task.startswith("disc"):
            ga = A[A.method == mth]
            r["dti_retained_fp"] = dti_of(ga) if len(ga) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["task", "dti_pooled"], ascending=[True, False])


def paired_zones(P: pd.DataFrame, ref: str) -> pd.DataFrame:
    if not len(P) or ref not in set(P.method):
        return pd.DataFrame()
    r = P[P.method == ref].set_index(["task", "zone"]).dti
    d = P.copy()
    d["ref_dti"] = [r.get((t, z), np.nan) for t, z in zip(d.task, d.zone)]
    d["delta_vs_ref"] = d.dti - d.ref_dti
    d["reference"] = ref
    return d[["task", "zone", "method", "reference", "dti", "ref_dti", "delta_vs_ref"]]


def pool_proximity(d: pd.DataFrame) -> pd.DataFrame:
    if not len(d):
        return d
    g = d.groupby(["task", "method", "distance_band"]).apply(
        lambda x: pd.Series({"truth_px": x.truth_px.sum(),
                             "recall_w": np.nansum(x.recall_w * x.truth_px) / max(x.truth_px.sum(), 1),
                             "fp_mass": x.fp_mass.sum(), "pred_px": x.pred_px.sum()})).reset_index()
    tot = g.groupby(["task", "method"]).fp_mass.transform("sum")
    g["fp_share"] = g.fp_mass / tot.replace(0, np.nan)
    return g


def pool_regions(d: pd.DataFrame) -> pd.DataFrame:
    if not len(d):
        return d
    g = d.groupby(["task", "method", "region"])[COMP].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    return g


def figures(m7, out: dict) -> None:
    from .figures import bar_panels16, line_panels16

    ws, sub = m7.ws, "validation/figures"
    p = out["pooled"]
    if len(p):
        ws.save_fig(bar_panels16(p, "method", "dti_pooled", "task", ref_col="worst_zone_dti",
                                 title="Section 13 - held-out DTI per method (tick = worst zone)"), "method_comparison", p, sub)
        if "system_recall" in p:
            ws.save_fig(bar_panels16(p, "method", "system_recall", "task", xlabel="system recall",
                                     title="Fault-system recall per method"), "system_recall_comparison", p, sub)
    pz = out["paired"]
    if len(pz):
        ws.save_fig(line_panels16(pz.assign(z=pz.zone.astype(int)), "z", "delta_vs_ref", "method", "task",
                                  xlabel="zone", ylabel=f"DTI - {out['champion_reference']}",
                                  title="Paired per-zone differences against the champion baseline",
                                  hline={t: 0.0 for t in pz.task.unique()}), "paired_zone_deltas", pz, sub)
    pr = out["proximity"]
    if len(pr):
        ws.save_fig(bar_panels16(pr.assign(cat=pr.method + " | " + pr.distance_band), "cat", "recall_w", "task",
                                 xlabel="weighted recall of truth in each distance band to known faults",
                                 title="Sensitivity to known-fault proximity"), "known_proximity_recall", pr, sub)
    rg = out["regions"]
    if len(rg):
        ws.save_fig(bar_panels16(rg.assign(cat=rg.method + " | " + rg.region), "cat", "dti", "task",
                                 title="Regional robustness (geographic folds, 3 x 3 regions)"), "regional_dti", rg, sub)
