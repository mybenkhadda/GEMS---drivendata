"""Splits generated from the raw labels and footprint only.

Geographic CV: the footprint is tiled into ``block_km`` blocks, block centroids are clustered (k-means,
``n_folds * clusters_per_fold`` clusters, weighted by valid pixels) and clusters are assigned greedily to folds by
label + area load. For fold k: val = fold-k pixels; buffer = pixels within ``buffer_km`` of val (never touched
in training); inner_val = the training cluster with the median label load, separated from inner training by
``inner_buffer_km`` (early stopping only); train = the rest.

Discovery: mapped fault pixels linked within ``link_km`` form fault systems. Whole systems of
``min_system_km`` .. ``max_system_frac`` x total trace are withheld until ``frac_trace`` of the trace length is
withheld per split (region-stratified, seeded, disjoint across splits). Withheld pixels and an ``ignore_km`` ring
get weight 0 (never negatives); evaluation zone = ``eval_km`` around the withheld systems. Each discovery split
gets its own inner-validation cluster outside its evaluation zone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .cache import atomic_save_npz, atomic_write_csv, sha256_array

ROLE = {"invalid": -1, "train": 0, "val": 1, "buffer": 2, "inner_val": 3, "inner_buffer": 4}
DROLE = {"none": 0, "eval_zone": 1, "ignore": 2, "withheld": 3}


def within(mask: np.ndarray, km: float, px_km: float) -> np.ndarray:
    if km <= 0 or not mask.any():
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) * px_km <= km


def skeleton(labels: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return skeletonize(labels)


def geo_folds(valid: np.ndarray, labels: np.ndarray, px_km: float, g: dict, seed: int) -> dict:
    from sklearn.cluster import KMeans

    h, w = valid.shape
    bpx = max(int(round(g["block_km"] / px_km)), 8)
    rr, cc = np.indices((h, w))
    nbc = (w + bpx - 1) // bpx
    bid = ((rr // bpx) * nbc + cc // bpx).astype(np.int32)
    nb = int(bid.max()) + 1
    v_px = np.bincount(bid[valid], minlength=nb)
    l_px = np.bincount(bid[valid & labels], minlength=nb)
    r_c = np.bincount(bid[valid], weights=rr[valid], minlength=nb)
    c_c = np.bincount(bid[valid], weights=cc[valid], minlength=nb)
    keep = np.flatnonzero(v_px > 0)
    blocks = pd.DataFrame({"block": keep, "valid_px": v_px[keep], "label_px": l_px[keep],
                           "r_cent": r_c[keep] / v_px[keep], "c_cent": c_c[keep] / v_px[keep]})
    K = int(g["n_folds"])
    n_cl = min(K * int(g["clusters_per_fold"]), len(blocks))
    km = KMeans(n_clusters=n_cl, n_init=20, random_state=seed)
    blocks["cluster"] = km.fit_predict(blocks[["r_cent", "c_cent"]].to_numpy(), sample_weight=blocks.valid_px)
    cl = blocks.groupby("cluster").agg(valid_px=("valid_px", "sum"), label_px=("label_px", "sum")).reset_index()
    cl["load"] = cl.label_px / max(cl.label_px.sum(), 1) + cl.valid_px / cl.valid_px.sum()
    cl = cl.sort_values(["load", "cluster"], ascending=[False, True])
    load, assign = np.zeros(K), {}
    for row in cl.itertuples():
        f = int(np.argmin(load))
        assign[row.cluster] = f
        load[f] += row.load
    blocks["fold"] = blocks.cluster.map(assign).astype(int)
    lut = np.full(nb, -1, np.int16)
    lut[blocks.block.to_numpy()] = blocks.fold.to_numpy()
    fold_map = np.where(valid, lut[bid], -1).astype(np.int16)
    clut = np.full(nb, -1, np.int16)
    clut[blocks.block.to_numpy()] = blocks.cluster.to_numpy()
    cluster_map = np.where(valid, clut[bid], -1).astype(np.int16)
    roles = np.full((K, h, w), ROLE["invalid"], np.int8)
    for k in range(K):
        val = fold_map == k
        r = roles[k]
        r[valid] = ROLE["train"]
        r[within(val, g["buffer_km"], px_km) & valid & ~val] = ROLE["buffer"]
        r[val] = ROLE["val"]
        tr = blocks[blocks.fold != k].groupby("cluster").label_px.sum().sort_values(kind="stable")
        if len(tr) >= 2:
            ic = int(tr.index[len(tr) // 2])
            ival = (cluster_map == ic) & (r == ROLE["train"])
            ibuf = within(ival, g["inner_buffer_km"], px_km) & (r == ROLE["train"]) & ~ival
            r[ibuf] = ROLE["inner_buffer"]
            r[ival] = ROLE["inner_val"]
            blocks[f"inner_val_fold{k}"] = blocks.cluster == ic
    return {"fold_map": fold_map, "cluster_map": cluster_map, "roles": roles, "blocks": blocks, "block_px": bpx}


def fault_systems(labels: np.ndarray, px_km: float, link_km: float) -> tuple[np.ndarray, pd.DataFrame]:
    linked = within(labels, link_km / 2.0, px_km)
    comp, n = ndimage.label(linked, structure=np.ones((3, 3), bool))
    sysmap = np.where(labels, comp - 1, -1).astype(np.int32)
    skel = skeleton(labels) & labels
    ids = sysmap[labels]
    n_px = np.bincount(ids, minlength=n)
    sk = np.bincount(sysmap[skel], minlength=n)
    rr, cc = np.nonzero(labels)
    r_m = np.bincount(ids, weights=rr, minlength=n) / np.maximum(n_px, 1)
    c_m = np.bincount(ids, weights=cc, minlength=n) / np.maximum(n_px, 1)
    h, w = labels.shape
    region = (np.clip((r_m * 3 // h).astype(int), 0, 2) * 3 + np.clip((c_m * 3 // w).astype(int), 0, 2))
    df = pd.DataFrame({"system_id": np.arange(n), "n_px": n_px, "trace_km": sk * px_km * 1.1, "r_cent": r_m,
                       "c_cent": c_m, "region": region})
    return sysmap, df[df.n_px > 0].reset_index(drop=True)


def discovery_splits(labels: np.ndarray, valid: np.ndarray, cluster_map: np.ndarray, px_km: float, g: dict,
                     inner_buffer_km: float, seed: int) -> dict:
    sysmap, systems = fault_systems(labels, px_km, g["link_km"])
    total = float(systems.trace_km.sum())
    rng = np.random.default_rng(seed)
    elig = systems[(systems.trace_km >= g["min_system_km"]) & (systems.trace_km <= g["max_system_frac"] * total)].copy()
    elig["rand"] = rng.random(len(elig))
    queues = {r: list(gr.sort_values("rand").system_id) for r, gr in elig.groupby("region")}
    order = []
    while any(queues.values()):
        for r in sorted(queues, key=lambda _: rng.random()):
            if queues[r]:
                order.append(queues[r].pop(0))
    D = int(g["n_splits"])
    target = g["frac_trace"] * total
    assign, load = {}, np.zeros(D)
    tk = systems.set_index("system_id").trace_km
    for sid in order:
        d = int(np.argmin(load))
        if load[d] >= target:
            break
        assign[sid] = d
        load[d] += float(tk[sid])
    systems["discovery_split"] = systems.system_id.map(assign).fillna(-1).astype(int)
    roles = np.zeros((D, *labels.shape), np.int8)
    inner = np.zeros((D, *labels.shape), np.int8)
    inner_cluster = []
    for d in range(D):
        ids = systems.system_id[systems.discovery_split == d].to_numpy()
        wh = np.isin(sysmap, ids) & labels
        retained = labels & ~wh
        r = roles[d]
        r[within(wh, g["eval_km"], px_km) & valid] = DROLE["eval_zone"]
        r[within(wh, g["ignore_km"], px_km) & ~retained] = DROLE["ignore"]
        r[wh] = DROLE["withheld"]
        # inner validation: a geographic cluster that does not touch this split's evaluation zone
        touched = set(np.unique(cluster_map[(r > 0) & (cluster_map >= 0)]).tolist())
        loads = pd.Series({c: int((labels & (cluster_map == c)).sum()) for c in np.unique(cluster_map[cluster_map >= 0])
                           if c not in touched})
        loads = loads[loads > 0].sort_values(kind="stable")
        ic = int(loads.index[len(loads) // 2]) if len(loads) else -1
        inner_cluster.append(ic)
        if ic >= 0:
            ival = (cluster_map == ic) & valid & (r == 0)
            ibuf = within(ival, inner_buffer_km, px_km) & valid & ~ival & (r == 0)
            inner[d][ibuf] = ROLE["inner_buffer"]
            inner[d][ival] = ROLE["inner_val"]
    return {"system_map": sysmap, "systems": systems, "roles": roles, "inner": inner,
            "withheld_km": load.tolist(), "total_trace_km": total, "inner_cluster": inner_cluster}


def build_splits(valid: np.ndarray, labels: np.ndarray, pixel_m: float, cfg: dict, out_dir) -> dict:
    s = cfg["splits"]
    px_km = pixel_m / 1000.0
    labels = labels & valid
    geo = geo_folds(valid, labels, px_km, s["geographic"], s["seed"])
    disc = discovery_splits(labels, valid, geo["cluster_map"], px_km, s["discovery"],
                            s["geographic"]["inner_buffer_km"], s["seed"])
    outs = [atomic_save_npz(out_dir / "geo_folds.npz", fold_map=geo["fold_map"], cluster_map=geo["cluster_map"],
                            roles=geo["roles"]),
            atomic_write_csv(out_dir / "blocks.csv", geo["blocks"]),
            atomic_save_npz(out_dir / "discovery_splits.npz", system_map=disc["system_map"], roles=disc["roles"],
                            inner=disc["inner"]),
            atomic_write_csv(out_dir / "fault_systems.csv", disc["systems"])]
    identity = {"geo": sha256_array(geo["roles"])[:16], "discovery": sha256_array(disc["roles"])[:16],
                "discovery_inner": sha256_array(disc["inner"])[:16]}
    summary = split_summary(geo, disc, labels, valid, px_km)
    outs.append(atomic_write_csv(out_dir / "split_summary.csv", summary))
    return {"outputs": outs, "identity": identity, "withheld_km": disc["withheld_km"],
            "total_trace_km": disc["total_trace_km"], "inner_cluster_discovery": disc["inner_cluster"]}


def split_summary(geo, disc, labels, valid, px_km) -> pd.DataFrame:
    rows = []
    for k in range(geo["roles"].shape[0]):
        for name, code in ROLE.items():
            if code < 0:
                continue
            m = geo["roles"][k] == code
            rows.append({"protocol": "geographic", "unit": f"geo{k}", "role": name, "px": int(m.sum()),
                         "km2": float(m.sum() * px_km ** 2), "label_px": int((labels & m).sum())})
    for d in range(disc["roles"].shape[0]):
        for name, code in DROLE.items():
            m = disc["roles"][d] == code
            rows.append({"protocol": "discovery", "unit": f"disc{d}", "role": name, "px": int(m.sum()),
                         "km2": float(m.sum() * px_km ** 2), "label_px": int((labels & m).sum())})
        for name in ("inner_val", "inner_buffer"):
            m = disc["inner"][d] == ROLE[name]
            rows.append({"protocol": "discovery", "unit": f"disc{d}", "role": name, "px": int(m.sum()),
                         "km2": float(m.sum() * px_km ** 2), "label_px": int((labels & m).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# training units
# ---------------------------------------------------------------------------
GEO_UNITS = ("geo0", "geo1", "geo2")
DISC_UNITS = ("disc0", "disc1", "disc2")


def unit_masks(unit: str, splits: dict, labels: np.ndarray, valid: np.ndarray) -> dict:
    """Masks of one training unit.

    train:      pixels whose labels / negatives are supervised (weight 1 unless ignored)
    forbidden:  pixels no training patch may touch
    inner_val:  early-stopping region; inner_truth its truth
    eval/truth/exclude: the scored region of the unit (DTI)
    ignore:     weight-0 pixels inside training patches (discovery: withheld systems + ring)
    visible:    positives the unit may see
    """
    kind, idx = unit[:-1], int(unit[-1])
    if kind == "geo":
        r = splits["geo_roles"][idx]
        train = r == ROLE["train"]
        visible = labels & train
        return {"train": train, "forbidden": (r == ROLE["val"]) | (r == ROLE["buffer"]) | (r == ROLE["inner_val"])
                | (r == ROLE["inner_buffer"]), "inner_val": r == ROLE["inner_val"],
                "inner_truth": labels & (r == ROLE["inner_val"]), "eval": r == ROLE["val"],
                "truth": labels & (r == ROLE["val"]), "exclude": None, "ignore": np.zeros_like(train),
                "visible": visible}
    if kind == "disc":
        r = splits["disc_roles"][idx]
        inner = splits["disc_inner"][idx]
        withheld = r == DROLE["withheld"]
        ignore = r >= DROLE["ignore"]
        iv = inner == ROLE["inner_val"]
        forb = iv | (inner == ROLE["inner_buffer"])
        train = valid & ~forb
        retained = labels & ~withheld
        return {"train": train, "forbidden": forb, "inner_val": iv, "inner_truth": retained & iv,
                "eval": r >= DROLE["eval_zone"], "truth": withheld, "exclude": retained, "ignore": ignore,
                "visible": retained & train & ~ignore}
    raise KeyError(unit)
