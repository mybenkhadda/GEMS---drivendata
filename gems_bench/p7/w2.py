"""T5 (H5): W2 + P7 features under a NEW versioned schema (``w2_p7.v1``); the frozen W2 schema is never changed.

Rows are rebuilt from scratch with ``p6.stack.unit_rows`` (same sampling, seeds and label rules as W2) for two
feature sets fitted on the SAME rows:
    w2_ctrl   W2's own frozen feature list (matched control: isolates the P7 features from refit / sampling noise)
    w2_p7     W2's list + the configured P7 features
Cross-fitting is W2's (``train_index``: same task, other zones, outside u's evaluation zone + buffer; fixed
LightGBM hyper-parameters). Held-out maps of both are scored with ``Boost.eval_variant`` (W2's own fusion blends),
i.e. exactly how the frozen ``w2`` reference is scored, so the incremental gain is measured under matched cells.

Provenance (``LeakageError``): every P7 feature of unit u comes from the P7 held-out prediction file OF u (protocol
key equal to u's, checkpoint of the unit that held u out: geo_f<k> for geo:<k>, disc_s<d> for disc:<d>); W2's own
channels keep ``stack.assert_provenance``. Old in-sample rows (``stack/features/rows.parquet``) are never mixed in.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..p6 import stack as ST
from ..p6.stack import LeakageError

SCHEMA = "w2_p7.v1"
P7_FEATURES = ("p7_fault", "p7_center", "p7_orient_coh", "p7_b0_disagree", "p7_b1_disagree", "p7_uncertainty")


def p7_unit_features(pred: dict, raw: dict, ev: np.ndarray) -> dict:
    """P7 layers of one unit (crop) from its held-out prediction; ranks inside ``ev`` like W2's CNN channels."""
    out = {"p7_fault": ST.rank_in(pred["seg"], ev)}
    if "center" in pred:
        out["p7_center"] = ST.rank_in(pred["center"], ev)
    if "ocos" in pred and "osin" in pred:
        out["p7_orient_coh"] = np.hypot(pred["ocos"], pred["osin"]).astype(np.float32)
    for k, src in (("p7_b0_disagree", "cnn_b0"), ("p7_b1_disagree", "cnn_b1")):
        if src in raw:
            out[k] = np.abs(out["p7_fault"] - ST.rank_in(raw[src], ev)).astype(np.float32)
    if "seg_std" in pred:
        out["p7_uncertainty"] = ST.rank_in(pred["seg_std"], ev)
    return out


def assert_p7_provenance(meta: dict, u, pkey: str) -> None:
    want = f"geo_f{u.zone}" if u.task == "geo" else f"disc_s{u.zone}"
    if meta.get("protocol_key") != pkey:
        raise LeakageError(f"{u.key}: P7 map protocol key {meta.get('protocol_key')} != {pkey}")
    if meta.get("unit") != want:
        raise LeakageError(f"{u.key}: P7 map made by {meta.get('unit')!r}, expected the held-out unit {want!r}")
    if meta.get("labels_stored"):
        raise LeakageError(f"{u.key}: P7 prediction file stores labels")


def feature_lists(w2_names: list[str], available: set[str], wanted: list[str]) -> dict:
    p7 = [f for f in wanted if f in available]
    return {"w2_ctrl": list(w2_names), "w2_p7": list(w2_names) + p7}


def run(p7, exp_id: str, exp: str) -> dict:
    """Cross-fitted w2_ctrl + w2_p7 on rebuilt rows; grid rows scored like the frozen W2; summaries + deltas."""
    from .evaluate import pred_rel, summarize

    bst, p6 = p7.bst, p7.p6
    sc = p6.cfg["stack"]
    seed = int(p6.cfg["seed"])
    cand = json.loads((p6.ws.root / "stack" / "candidate.json").read_text(encoding="utf-8"))
    w2_names = list(cand["features"])
    units = bst.units()
    raws, prov, avail = {}, [], None
    for u in units:
        z = np.load(p7.ws.root / pred_rel(p7, exp_id, u))
        meta = {"protocol_key": str(z["protocol_key"]), "unit": str(z["unit"]), "labels_stored": bool(z["labels_stored"])}
        assert_p7_provenance(meta, u, p7.m9.pkey(u))
        pred = {k: z[k].astype(np.float32) for k in ("seg", "center", "ocos", "osin", "seg_std") if k in z.files}
        raw, _ = ST.unit_raw(p6, u, list(p7.m9.models)[:2])     # exactly W2's features (provenance asserted)
        raw.update(p7_unit_features(pred, raw, u.ev))
        raws[u.key] = raw
        prov.append({"unit": u.key, "p7_source": meta["unit"], "protocol_key": meta["protocol_key"],
                     "experiment_id": exp_id})
        have = {k for k, v in raw.items() if isinstance(v, np.ndarray) and np.isfinite(v[u.ev]).any()}
        avail = have if avail is None else avail & have
    missing_w2 = [n for n in w2_names if n not in (avail or set())]
    if missing_w2:
        raise FileNotFoundError(f"W2 features missing in some unit: {missing_w2} (the control cannot be rebuilt)")
    sets = feature_lists(w2_names, avail or set(), list(p7.cfg["w2"]["p7_features"]))
    sub = f"stacker/{SCHEMA}/{exp_id}"
    p7.ws.save_table(pd.DataFrame(prov), "provenance", sub)
    p7.ws.save_json({"schema": SCHEMA, "sets": sets, "base_schema": "w2 (frozen, stack/candidate.json)",
                     "p7_experiment_id": exp_id}, f"{sub}/feature_schema.json")
    all_names = sets["w2_p7"]
    T = pd.concat([ST.unit_rows(p6, u, raws[u.key], all_names, seed + 101 * i) for i, u in enumerate(units)],
                  ignore_index=True)
    try:
        T.to_parquet(p7.ws.root / sub / "rows.parquet", index=False)
    except Exception as e:
        p7.say(f"rows.parquet skipped: {type(e).__name__}: {e}")
    shape = p7.ctx.data.shape
    grid, audit, imps = [], [], []
    for name, names in sets.items():
        if name == "w2_ctrl" and not p7.cfg["w2"].get("control", True):
            continue
        for u in units:
            tr = ST.train_index(T, u, shape, float(sc["exclude_buffer_km"]))
            y = T.y.to_numpy()[tr]
            a = {"set": name, "unit": u.key, "n_train": int(tr.sum()), "n_pos": int((y == 1).sum()),
                 "n_neg": int((y == 0).sum())}
            if a["n_pos"] < 50 or a["n_neg"] < 50:
                audit.append({**a, "status": "too few rows: unit skipped"})
                continue
            mp = p7.ws.root / sub / "maps" / f"{name}_{u.task}_{u.zone}.npy"
            if mp.exists():
                s = np.load(mp)
                a["status"] = "reused"
            else:
                m = ST.make_model(sc["lgbm"], seed)
                m.fit(T.loc[tr, names].to_numpy(np.float32), y, sample_weight=T.w.to_numpy()[tr])
                s = ST.predict_unit(m, raws[u.key], u.ev, names)
                mp.parent.mkdir(parents=True, exist_ok=True)
                np.save(mp, s)
                imps.append(pd.DataFrame({"set": name, "unit": u.key, "feature": names,
                                          "gain": m.booster_.feature_importance("gain").astype(float)}))
                a["status"] = "trained"
            audit.append(a)
            grid += bst.eval_variant(u, s, name)
            u.release()
    G = pd.DataFrame(grid)
    p7.ws.save_table(G, "grid", sub)
    p7.ws.save_table(pd.DataFrame(audit), "crossfit_audit", sub)
    if imps:
        p7.ws.save_table(pd.concat(imps, ignore_index=True).groupby(["set", "feature"]).gain.mean().reset_index(),
                         "feature_importance", sub)
    S, Pz = summarize(p7, G, [n for n in sets if n in set(G.candidate)] if len(G) else [])
    p7.ws.save_table(S, "summary", sub)
    p7.ws.save_table(Pz, "per_zone", sub)
    return {"summary": S, "per_zone": Pz, "grid": G, "sets": sets, "root": str(p7.ws.root / sub)}
