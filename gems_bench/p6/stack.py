"""W2: stacked LightGBM ranker = honest B0 / B1 maps + complementary structural evidence (<= 25 features).

Not the pixel LightGBM that lost to the CNN fusion (0.090): the CNN ranks are the backbone, and the trees may only
re-order pixels using evidence the CNNs do not see (G02 continuity / apparent offset, potential-field gradients,
the geometry of the PERMITTED known map).

Leakage rules (enforced, ``LeakageError``):
* CNN channels come from ``gems3.official.unit_score`` only: OOF for ``geo`` folds, the split's discovery map for
  ``disc_*`` (models trained with the withheld systems hidden). Provenance is recorded and asserted per unit.
* G02 / G01 rasters are the per-unit files written by G02 (normalised on the unit's training geography); their
  metadata (task, zone, window, ``labels_stored``) must match the unit.
* Known-fault features use only the faults visible to the unit: ``geo`` fold k -> faults outside fold k;
  ``disc`` split d -> retained faults (labels minus withheld); full region -> every mapped fault.
* Withheld systems are positives of their own unit and are never negatives anywhere; the discovery ignore ring and
  every mapped fault (+ ``neg_buffer_px``) are excluded from negatives.
* Continuous features are ranked inside each unit's evaluation mask (label-free, like the champion's per-zone rank),
  so no statistic crosses from evaluation to training geography; distances stay in km.
* Held-out unit u is predicted by a model fitted on the other zones of its task, minus rows within
  ``exclude_buffer_km`` of u's evaluation zone; hyper-parameters are fixed a priori (no tuning on held-out truth).

Evaluation reuses the M7 protocol: per-unit official components (``m7.protocol.score_row``), the champion's thin
representations, nested (blend, rep, frac) choice (``gems3.metric_opt.nested_select``), system hits against the
fallback's held-out map. Gate (``stack.gate``): discovery DTI >= max(0.167, ref + 0.005), worst zone >= 0.126,
geographic DTI no worse than ref - 0.005. A failed gate stops W2: no re-tuning.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import ndimage

CNN = ("cnn_b0", "cnn_b1", "cnn_disagree")
G02 = ("g02c", "g02a", "g02f", "g01_persistence")
PF = ("mag_hgrad", "grav_hgrad", "mg_orient_agree", "mag_tilt")
KNOWN = ("known_dist_km", "known_density", "tip_dist_km", "tip_along_cos")
RANKED = set(CNN[:2]) | set(G02) | set(PF[:2]) | {"mag_tilt"}


class LeakageError(RuntimeError):
    """A prediction channel or label-derived feature would carry held-out information."""


# =============================================================================== pure feature functions
def potential_features(mag: np.ndarray | None, grav: np.ndarray | None, sigma_px: float = 1.0) -> dict:
    """Magnetic / gravity horizontal gradient, doubled-angle strike agreement in [0, 1], magnetic tilt (rad)."""
    from ..m8 import potential_fields as PFm

    out, strike = {}, {}
    for role, a in (("mag", mag), ("grav", grav)):
        if a is None:
            continue
        g = PFm.gradient_parts(np.asarray(a, np.float32), sigma_px)
        out[f"{role}_hgrad"] = g["thdr"]
        strike[role] = g["strike"]
        if role == "mag":
            f, m = PFm.fill(np.asarray(a, np.float32))
            vz = PFm.vdr_fft(PFm.smooth(f, sigma_px))
            out["mag_tilt"] = PFm.restore(np.arctan2(vz, np.hypot(g["ge"], g["gn"]) + 1e-12), m)
    if len(strike) == 2:
        d = PFm.axial_diff(strike["mag"], strike["grav"])
        out["mg_orient_agree"] = (np.cos(np.radians(2.0 * d)) * 0.5 + 0.5).astype(np.float32)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def endpoints(sk: np.ndarray) -> np.ndarray:
    """Skeleton pixels with exactly one 8-neighbour."""
    sk = np.asarray(sk, bool)
    n = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk.astype(np.uint8)
    return sk & (n == 1)


def known_features(known: np.ndarray, valid: np.ndarray, px_km: float = 0.1, cap_km: float = 20.0,
                   density_radius_km: float = 2.0, tip_max_km: float = 5.0, tail_px: int = 5) -> dict:
    """Geometry of the permitted known map: distance, local density, distance to the nearest trace tip and the
    cosine between (tip -> pixel) and the trace direction at that tip (1 = straight ahead along strike)."""
    from ..postprocess.topology import skeleton

    known = np.asarray(known, bool)
    shape = known.shape
    if not known.any():
        z = np.full(shape, cap_km, np.float32)
        return {"known_dist_km": z, "known_density": np.zeros(shape, np.float32), "tip_dist_km": z.copy(),
                "tip_along_cos": np.zeros(shape, np.float32)}
    dist = np.minimum(ndimage.distance_transform_edt(~known) * px_km, cap_km).astype(np.float32)
    r = max(int(round(density_radius_km / px_km)), 1)
    dens = ndimage.uniform_filter(known.astype(np.float32), 2 * r + 1, mode="constant")
    sk = skeleton(known)
    ep = endpoints(sk)
    if not ep.any():
        return {"known_dist_km": dist, "known_density": dens.astype(np.float32),
                "tip_dist_km": np.full(shape, cap_km, np.float32), "tip_along_cos": np.zeros(shape, np.float32)}
    w = 2 * tail_px + 1
    rr, cc = np.indices(shape, dtype=np.float32)
    m = ndimage.uniform_filter(sk.astype(np.float32), w, mode="constant")
    mr = ndimage.uniform_filter(sk * rr, w, mode="constant") / np.maximum(m, 1e-9)
    mc = ndimage.uniform_filter(sk * cc, w, mode="constant") / np.maximum(m, 1e-9)
    tr, tc = rr - mr, cc - mc                                   # tip direction: away from the trace's local centroid
    nrm = np.hypot(tr, tc)
    tr, tc = np.where(nrm > 0, tr / np.maximum(nrm, 1e-9), 0.0), np.where(nrm > 0, tc / np.maximum(nrm, 1e-9), 0.0)
    d_ep, (ir, ic) = ndimage.distance_transform_edt(~ep, return_indices=True)
    vr, vc = rr - ir, cc - ic
    vn = np.hypot(vr, vc)
    cos = np.where(vn > 0, (vr * tr[ir, ic] + vc * tc[ir, ic]) / np.maximum(vn, 1e-9), 1.0)
    tip_km = (d_ep * px_km).astype(np.float32)
    cos = np.where(tip_km <= tip_max_km, cos, 0.0).astype(np.float32)
    return {"known_dist_km": dist, "known_density": dens.astype(np.float32),
            "tip_dist_km": np.minimum(tip_km, cap_km), "tip_along_cos": cos}


def rank_in(a: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """Per-unit rank in [0, 1] inside ``ev``; NaN outside ``ev`` and where ``a`` is NaN."""
    from ..gems3.representation import rank01

    a = np.asarray(a, np.float32)
    ok = ev & np.isfinite(a)
    out = np.full(a.shape, np.nan, np.float32)
    if ok.any():
        out[ok] = rank01(a, ok)[ok]
    return out


def assemble(raw: dict, ev: np.ndarray, names: list[str]) -> np.ndarray:
    """(n_ev, n_features) float32 matrix in ``names`` order; ranked features ranked inside ``ev``."""
    cols = []
    for n in names:
        a = raw.get(n)
        if a is None:
            cols.append(np.full(int(ev.sum()), np.nan, np.float32))
            continue
        a = rank_in(a, ev) if n in RANKED else np.asarray(a, np.float32)
        cols.append(a[ev])
    return np.stack(cols, 1) if cols else np.zeros((int(ev.sum()), 0), np.float32)


def ev_position(ev: np.ndarray, flat_idx: np.ndarray) -> np.ndarray:
    """Row of ``assemble(raw, ev, ...)`` for each flat crop index (all indices must lie inside ``ev``)."""
    e = np.asarray(ev, bool).ravel()
    if not e[flat_idx].all():
        raise ValueError("sampled pixels outside the evaluation mask")
    return (np.cumsum(e) - 1)[flat_idx]


def training_masks(eval_truth: np.ndarray, scored: np.ndarray, all_labels: np.ndarray, ignore: np.ndarray | None,
                   buffer_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(positives, negatives) of one unit. Negatives never touch any mapped fault (+ buffer) or the ignore ring."""
    from ..postprocess.topology import _disk

    pos = np.asarray(eval_truth, bool) & scored
    near = ndimage.binary_dilation(np.asarray(all_labels, bool), structure=_disk(int(buffer_px))) if buffer_px > 0 \
        else np.asarray(all_labels, bool)
    neg = np.asarray(scored, bool) & ~near
    if ignore is not None:
        neg &= ~np.asarray(ignore, bool)
    return pos, neg


def sample_weights(pos: np.ndarray, neg: np.ndarray, ambiguous: np.ndarray, rng, n_neg: int, pos_cap: int,
                   neg_w: float, amb_w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(flat indices, labels, weights): every positive up to ``pos_cap``, ``n_neg`` random negatives; ambiguous
    negatives (label-free high-evidence pixels) get ``amb_w`` x ``neg_w``."""
    ip, ineg = np.flatnonzero(pos), np.flatnonzero(neg)
    if ip.size > pos_cap:
        ip = np.sort(rng.choice(ip, pos_cap, replace=False))
    if ineg.size > n_neg:
        ineg = np.sort(rng.choice(ineg, n_neg, replace=False))
    idx = np.concatenate([ip, ineg])
    y = np.concatenate([np.ones(ip.size, np.int8), np.zeros(ineg.size, np.int8)])
    w = np.where(y == 1, 1.0, neg_w * np.where(np.asarray(ambiguous).ravel()[idx], amb_w, 1.0)).astype(np.float32)
    return idx, y, w


def select_features(cfg: dict, available: set[str]) -> list[str]:
    """Priority order of the brief, capped at ``max_features``; absent families are dropped consistently."""
    order = list(CNN) + list(G02) + list(PF) + list(KNOWN)
    return [n for n in order if n in available][: int(cfg["max_features"])]


def assert_provenance(prov: dict, task: str, zone: int) -> None:
    """CNN channels: OOF on geo folds, the split's own discovery map on disc units; G02 rasters of this unit."""
    for k, v in prov.items():
        if k.startswith("cnn_b"):
            ok = v == "oof" if task == "geo" else v == f"discovery_pred:{zone}"
            if not ok:
                raise LeakageError(f"{task}:{zone} {k} provenance {v!r} is not held-out for this unit")
        if k.startswith(("g02", "g01")) and v != "absent":
            if v != f"g02_unit:{task}_{zone}":
                raise LeakageError(f"{task}:{zone} {k} provenance {v!r} is not this unit's G02 raster")


# =============================================================================== Colab side: features per unit
def permitted_known(ctx, u) -> np.ndarray:
    """Full-grid known faults visible to unit ``u`` (never its own truth)."""
    lab = np.asarray(ctx.data.labels, bool)
    if u.task == "geo":
        return lab & (ctx.folds.fold_map != u.zone)
    if u.is_disc:
        return lab & ~ctx.splits.withheld(u.zone)
    return lab


def _g02(p6):
    if "g02" not in p6.__dict__:
        from ..m9.research.g02_experiment import G02Experiment

        try:
            p6.__dict__["g02"] = G02Experiment.attach(p6.m9, p6.cfg["stack"].get("g02_run_id"))
        except FileNotFoundError as e:
            p6.say(f"W2: no G02 run ({e}); G02 features absent")
            p6.__dict__["g02"] = None
    return p6.__dict__["g02"]


def g02_keys(p6, present: list[str]) -> dict:
    """Configured G02 / G01 keys; ``null`` entries resolved by prefix among the raster's keys."""
    want = dict(p6.cfg["stack"]["g02_keys"])
    pref = {"g02f": "F:", "g01_persistence": "G01-"}
    for k, v in want.items():
        if v is None and k in pref:
            hit = [x for x in present if x.startswith(pref[k]) and ("pers" in x.lower() or k == "g02f")]
            want[k] = hit[0] if hit else None
    return {k: v for k, v in want.items() if v}


def g02_unit(p6, u) -> tuple[dict, dict]:
    g = _g02(p6)
    rel = f"rasters/units/{u.task}_{u.zone}.npz"
    if g is None or not g.store.path(rel).exists():
        return {}, {k: "absent" for k in G02}
    keys = g02_keys(p6, g.store.raster_keys(rel))
    arrs, meta = g.store.load_raster_keys(rel, list(keys.values()))
    r0, r1, c0, c1 = (int(x) for x in meta["window"])
    if (meta.get("task"), int(meta.get("zone", -1))) != (u.task, int(u.zone)) or meta.get("labels_stored"):
        raise LeakageError(f"G02 raster {rel} metadata does not match {u.key}")
    if (r0, r1, c0, c1) != (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop):
        raise LeakageError(f"G02 raster {rel} window {(r0, r1, c0, c1)} differs from the unit crop")
    out = {k: np.asarray(arrs[v], np.float32) for k, v in keys.items() if v in arrs}
    return out, {k: (f"g02_unit:{u.task}_{u.zone}" if k in out else "absent") for k in G02}


def pf_full(p6) -> dict:
    """Potential-field features on the full grid (label-free; cached under stack/features/pf_full.npz)."""
    p = p6.ws.root / "stack" / "features" / "pf_full.npz"
    if p.exists():
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    d = p6.ctx.data
    bands = p6.cfg["stack"]["bands"]
    rd = {r: (np.where(d.valid, np.asarray(d.read_band(b), np.float32), np.nan) if b in d.band_names else None)
          for r, b in bands.items()}
    out = potential_features(rd.get("mag"), rd.get("grav"))
    p6.ws.save_npz(out, "stack/features/pf_full.npz")
    return out


def unit_raw(p6, u, models: list[str]) -> tuple[dict, dict]:
    """Cropped raw features of one unit + provenance (asserted)."""
    from ..gems3.official import unit_score

    raw, prov = {}, {}
    for name, m in zip(("cnn_b0", "cnn_b1"), models):
        a = unit_score(p6.ctx, m, u)
        if a is None:
            raise FileNotFoundError(f"{u.key}: honest map of {m} missing")
        raw[name] = a
        prov[name] = "oof" if u.task == "geo" else f"discovery_pred:{u.zone}"
    r0, r1 = rank_in(raw["cnn_b0"], u.ev), rank_in(raw["cnn_b1"], u.ev)
    raw["cnn_disagree"] = np.abs(r0 - r1)
    g, gp = g02_unit(p6, u)
    raw.update(g)
    prov.update(gp)
    pf = pf_full(p6)
    raw.update({k: np.asarray(v)[u.sl] for k, v in pf.items()})
    sc = p6.cfg["stack"]
    kn = known_features(permitted_known(p6.ctx, u), p6.ctx.data.valid, 0.1, float(sc["known_cap_km"]),
                        float(sc["known_density_radius_km"]), float(sc["endpoint_max_km"]))
    raw.update({k: v[u.sl] for k, v in kn.items()})
    assert_provenance(prov, u.task, u.zone)
    return raw, prov


def unit_rows(p6, u, raw: dict, names: list[str], seed: int) -> pd.DataFrame:
    sc = p6.cfg["stack"]["sampling"]
    lab = np.asarray(p6.ctx.data.labels, bool)[u.sl]
    ign = p6.ctx.splits.ignore(u.zone)[u.sl] if u.is_disc else None
    pos, neg = training_masks(u.eval_truth(), u.scored_mask(), lab, ign, int(sc["neg_buffer_px"]))
    amb_src = np.fmax(rank_in(raw["cnn_b0"], u.ev), rank_in(raw["cnn_b1"], u.ev))
    if "g02c" in raw:
        amb_src = np.fmax(amb_src, rank_in(raw["g02c"], u.ev))
    amb = np.nan_to_num(amb_src, nan=0.0) >= float(sc["ambiguous_quantile"])
    idx, y, w = sample_weights(pos, neg, amb, np.random.default_rng(seed), int(sc["neg_per_unit"]),
                               int(sc["pos_cap_per_unit"]), float(sc["negative_weight"]), float(sc["ambiguous_weight"]))
    X = assemble(raw, u.ev, names)[ev_position(u.ev, idx)]      # ranked inside ev exactly as at prediction time
    r, c = np.unravel_index(idx, u.ev.shape)
    T = pd.DataFrame(X, columns=names)
    return T.assign(task=u.task, zone=int(u.zone), r_full=r + u.sl[0].start, c_full=c + u.sl[1].start, y=y, w=w)


# =============================================================================== Colab side: cross-fitting
def make_model(gc: dict, seed: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(n_estimators=int(gc["n_estimators"]), learning_rate=float(gc["learning_rate"]),
                              num_leaves=int(gc["num_leaves"]), max_depth=int(gc["max_depth"]),
                              min_child_samples=int(gc["min_child_samples"]), subsample=float(gc["subsample"]),
                              subsample_freq=1, colsample_bytree=float(gc["colsample"]),
                              reg_lambda=float(gc["reg_lambda"]), random_state=seed, verbose=-1)


def train_index(T: pd.DataFrame, u, shape, buffer_km: float) -> np.ndarray:
    """Rows of the same task, other zones, outside u's evaluation zone + buffer."""
    from ..m7.ranking import _zone_mask

    ez = _zone_mask(u, shape, buffer_km)
    inside = ez[T.r_full.to_numpy().astype(int), T.c_full.to_numpy().astype(int)]
    return ((T.task == u.task) & (T.zone != u.zone)).to_numpy() & ~inside


def predict_unit(model, raw: dict, ev: np.ndarray, names: list[str], chunk: int = 2_000_000) -> np.ndarray:
    X = assemble(raw, ev, names)
    s = np.concatenate([model.predict_proba(X[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)]) if len(X) \
        else np.zeros(0)
    out = np.full(ev.shape, np.nan, np.float32)
    out[ev] = s
    return out


def _champion_zone(sel: dict, task: str, champ: str, zone: int) -> dict:
    s = (sel.get(task) or {}).get(champ) or {}
    return next((r for r in s.get("per_zone", []) if int(r["zone"]) == int(zone)), None) or {}


def eval_unit(p6, u, stack_score: np.ndarray, raw: dict, sel: dict) -> list[dict]:
    """Official components of every (blend, rep, frac) on one held-out unit (eligible budget, ring 0)."""
    from ..gems3 import representation as R
    from ..m7.protocol import score_row

    sc, m7c = p6.cfg["stack"], p6.m9.sh
    reps = m7c["baseline"]["representations"]
    sigma = float(m7c["evaluation"]["ridge_sigma_px"])
    cz = _champion_zone(sel, u.task, p6.cfg["probes"]["champion"], u.zone)
    w = float(cz.get("w", 0.5))
    F = R.fuse(R.rank01(raw["cnn_b0"], u.ev), R.rank01(raw["cnn_b1"], u.ev), w)
    S = R.rank01(np.nan_to_num(stack_score, nan=0.0), u.ev)
    el, pk = p6.m9.elig(u, 0), p6.m9.pkey(u)
    rows = []
    for lam in sc["blend"]:
        s = ((1.0 - float(lam)) * F + float(lam) * S).astype(np.float32)
        crest = R.crest_map(s, u.ev, sigma)
        for rn in sc["reps"]:
            for f in sc["area_fracs"]:
                q = R.apply(reps[rn], s, u.ev, float(f), known=u.known, crest=crest, sigma_px=sigma, eligible=el)
                rows.append(score_row(u, q, el, pk, method="p6_stack", blend=float(lam), rep=rn, frac=float(f),
                                      champion_w=w))
    return rows


def heldout_q(p6, u, stack_score, raw, sel, params: dict) -> np.ndarray:
    from ..gems3 import representation as R

    m7c = p6.m9.sh
    cz = _champion_zone(sel, u.task, p6.cfg["probes"]["champion"], u.zone)
    F = R.fuse(R.rank01(raw["cnn_b0"], u.ev), R.rank01(raw["cnn_b1"], u.ev), float(cz.get("w", 0.5)))
    S = R.rank01(np.nan_to_num(stack_score, nan=0.0), u.ev)
    lam = float(params["blend"])
    s = ((1.0 - lam) * F + lam * S).astype(np.float32)
    return R.apply(m7c["baseline"]["representations"][params["rep"]], s, u.ev, float(params["frac"]), known=u.known,
                   sigma_px=float(m7c["evaluation"]["ridge_sigma_px"]), eligible=p6.m9.elig(u, 0))


def run(p6) -> dict:
    """Cross-fitted stacker on geo + disc_official, nested evaluation, unique systems, gate, candidate.json."""
    from ..gems3.metric_opt import nested_select
    from ..m7.baseline import heldout_map
    from ..m7.protocol import system_hits
    from .select import fallback_reference, gate

    sc = p6.cfg["stack"]
    sel = p6.selection()
    seed = int(p6.cfg["seed"])
    models = list(p6.m9.models)[:2]
    units = [u for t in sc["tasks"] for u in p6.m9.m7.units_of(t)]
    raws, provs = {}, []
    avail: set[str] | None = None
    for u in units:                                            # pass 1: which features exist in EVERY unit
        raw, prov = unit_raw(p6, u, models)
        raws[u.key] = raw
        provs.append({"unit": u.key, **prov})
        have = {k for k, v in raw.items() if v is not None and np.isfinite(np.asarray(v)[u.ev]).any()}
        avail = have if avail is None else avail & have
    names = select_features(sc, avail or set())
    p6.ws.save_table(pd.DataFrame(provs), "provenance", "stack")
    p6.ws.save_json({"features": names, "n": len(names), "dropped_absent": sorted(set(CNN + G02 + PF + KNOWN) - set(names))},
                    "stack/features/feature_set.json")
    T = pd.concat([unit_rows(p6, u, raws[u.key], names, seed + 101 * i) for i, u in enumerate(units)], ignore_index=True)
    p6.ws.save_table(T.groupby(["task", "zone", "y"]).size().reset_index(name="rows"), "row_counts", "stack")
    try:
        T.to_parquet(p6.ws.root / "stack" / "features" / "rows.parquet", index=False)
    except Exception as e:                                     # optional format
        p6.say(f"rows.parquet skipped: {type(e).__name__}: {e}")
    shape = p6.ctx.data.shape
    imps, grid, audit = [], [], []
    scores = {}
    for u in units:
        mp = p6.ws.root / "stack" / "maps" / f"{u.task}_{u.zone}.npy"
        tr = train_index(T, u, shape, float(sc["exclude_buffer_km"]))
        y = T.y.to_numpy()[tr]
        a = {"unit": u.key, "n_train": int(tr.sum()), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
        if a["n_pos"] < 50 or a["n_neg"] < 50:
            a["status"] = "too few training rows: unit skipped"
            audit.append(a)
            continue
        if mp.exists() and not p6.cfg.get("overwrite_existing"):
            scores[u.key] = np.load(mp)
            a["status"] = "reused"
        else:
            t0 = time.perf_counter()
            m = make_model(sc["lgbm"], seed)
            m.fit(T.loc[tr, names].to_numpy(np.float32), y, sample_weight=T.w.to_numpy()[tr])
            scores[u.key] = predict_unit(m, raws[u.key], u.ev, names)
            np.save(mp, scores[u.key])
            imps.append(pd.DataFrame({"unit": u.key, "feature": names,
                                      "gain": m.booster_.feature_importance("gain").astype(float)}))
            a.update(status="trained", seconds=round(time.perf_counter() - t0, 1))
        audit.append(a)
        grid += eval_unit(p6, u, scores[u.key], raws[u.key], sel)
    G = pd.DataFrame(grid)
    p6.ws.save_table(G, "grid", "stack")
    p6.ws.save_table(pd.DataFrame(audit), "crossfit_audit", "stack")
    if imps:
        I = pd.concat(imps, ignore_index=True)
        p6.ws.save_table(I.groupby("feature").gain.mean().sort_values(ascending=False).reset_index(),
                         "feature_importance", "stack")
    summ, per_all = {}, []
    for task in sc["tasks"]:
        g = G[G.task == task] if len(G) else G
        per, s = nested_select(g, ["blend", "rep", "frac"]) if len(g) else (pd.DataFrame(), {"dti_crossfit": np.nan})
        summ[task] = s
        per_all.append(per.assign(task=task))
    P = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    p6.ws.save_table(P, "per_zone", "stack")
    sysrows, champ = [], p6.cfg["probes"]["champion"]
    for r in P.itertuples() if len(P) else []:
        u = p6.m9.unit(f"{r.task}:{int(r.zone)}")
        if r.task == "geo" or u is None or u.key not in scores:
            continue
        q = heldout_q(p6, u, scores[u.key], raws[u.key], sel, {"blend": r.blend, "rep": r.rep, "frac": r.frac})
        p6.ws.save_npz({"q": q.astype(np.float32)}, f"validation/maps/p6_stack_{u.task}_{u.zone}.npz")
        hs = system_hits(p6.ctx, u, q, p6.m9.sh)
        qf = heldout_map(p6.view, u.task, champ, u.zone)
        hf = system_hits(p6.ctx, u, qf, p6.m9.sh) if qf is not None else hs.iloc[0:0]
        s_set = set(hs.system_id[hs.recovered]) if len(hs) else set()
        f_set = set(hf.system_id[hf.recovered]) if len(hf) else set()
        sysrows.append({"task": u.task, "zone": u.zone, "stack_systems": len(s_set), "fallback_systems": len(f_set),
                        "stack_only": len(s_set - f_set), "fallback_only": len(f_set - s_set),
                        "n_systems": int(len(hs))})
    SY = pd.DataFrame(sysrows)
    p6.ws.save_table(SY, "unique_systems", "stack")
    ref = fallback_reference(p6, sel)
    prim = p6.prim
    cand = {"name": "p6_stack", "model_source": "p6_stack",
            "disc_dti": summ.get(prim, {}).get("dti_crossfit"), "disc_worst": summ.get(prim, {}).get("fold_min"),
            "geo_dti": summ.get("geo", {}).get("dti_crossfit"), "geo_worst": summ.get("geo", {}).get("fold_min"),
            "unique_systems": int(SY.stack_only.sum()) if len(SY) else 0,
            "params_all": {t: s.get("params_all") for t, s in summ.items()}, "features": names}
    ok, why = gate(cand, ref, sc["gate"])
    cand.update(gate="PROMOTE" if ok else "REJECT", gate_reasons=why,
                incremental_disc=(cand["disc_dti"] or np.nan) - float(ref["disc_dti"]))
    p6.ws.save_json(json.loads(json.dumps(cand, default=float)), "stack/candidate.json")
    p6.say(f"W2 gate: {cand['gate']} {why or ''} (disc {cand['disc_dti']}, worst {cand['disc_worst']}, "
           f"geo {cand['geo_dti']}, +{cand['unique_systems']} systems)")
    return {"summary": summ, "per_zone": P, "systems": SY, "candidate": cand, "gate": {"decision": cand["gate"]}}


# =============================================================================== full region (after PROMOTE only)
def g02_full(p6) -> dict:
    """Full-grid G02 maps of the configured keys (resumable tiles under stack/features/g02_full)."""
    from ..m9.research.g02_experiment import MapStore

    g = _g02(p6)
    if g is None:
        return {}
    seed = int(p6.cfg["seed"])
    ref = np.asarray(p6.ctx.data.valid, bool)                  # full region: the whole footprint is "training"
    refs, rrefs = g.g01.fit_refs(ref, seed), g.fit_raw_refs(ref, seed)
    H, W = p6.ctx.data.shape
    ms = MapStore(p6.ws.root / "stack" / "features" / "g02_full", (H, W))
    tp = int(g.cfg["tiling"]["tile_px"])
    done = ms.tiles_done()
    for r in range(0, H, tp):
        for c in range(0, W, tp):
            if (r, c) in done:
                continue
            gs = (slice(r, min(r + tp, H)), slice(c, min(c + tp, W)))
            if g.src.valid[gs].any():
                for k, a in g.tile_maps(gs, refs, rrefs).items():
                    ms.put(k, gs, a)
                g.src.release()
            done.add((r, c))
            ms.mark_tile(done)
    keys = g02_keys(p6, ms.keys())
    return {k: np.asarray(ms[v], np.float32) for k, v in keys.items() if v in ms}


def full_region(p6) -> dict:
    """Refit on every labelled row of the primary task, score the full grid, write the validated GeoTIFF."""
    from ..gems3 import representation as R
    from ..gems3 import submission as S
    from ..m7 import export as E

    cand = json.loads((p6.ws.root / "stack" / "candidate.json").read_text(encoding="utf-8"))
    if cand.get("gate") != "PROMOTE":
        raise RuntimeError("W2 full region refused: the stacker did not pass its gate")
    sc, names = p6.cfg["stack"], list(cand["features"])
    T = pd.read_parquet(p6.ws.root / "stack" / "features" / "rows.parquet")
    tr = (T.task == p6.prim).to_numpy()
    m = make_model(sc["lgbm"], int(p6.cfg["seed"]))
    m.fit(T.loc[tr, names].to_numpy(np.float32), T.y.to_numpy()[tr], sample_weight=T.w.to_numpy()[tr])
    import joblib

    joblib.dump(m, p6.ws.root / "stack" / "models" / "stack_full.joblib")
    data = p6.ctx.data
    valid = np.asarray(data.valid, bool)
    maps, _ = p6.view.full_maps()
    raw = {"cnn_b0": maps[0], "cnn_b1": maps[1]}
    raw["cnn_disagree"] = np.abs(rank_in(maps[0], valid) - rank_in(maps[1], valid))
    raw.update(g02_full(p6))
    raw.update(pf_full(p6))
    raw.update(known_features(np.asarray(data.labels, bool), valid, 0.1, float(sc["known_cap_km"]),
                              float(sc["known_density_radius_km"]), float(sc["endpoint_max_km"])))
    miss = [n for n in names if n not in raw]
    if miss:
        raise FileNotFoundError(f"full-region features missing: {miss} (the stacker cannot be exported)")
    score = predict_unit(m, raw, valid, names)
    p6.ws.save_raster({"stack_score": score}, "stack/maps/full_score.tif", data.grid)
    pa = cand["params_all"][p6.prim]
    src = sc["full_region"]["area_frac_source"]
    task = "geo" if src == "geo" else p6.prim
    frac = float(src) if isinstance(src, (int, float)) else float(cand["params_all"][task]["frac"])
    w = float(p6.selection()[p6.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
    F = R.fuse(R.rank01(maps[0], valid), R.rank01(maps[1], valid), w)
    s = ((1 - float(pa["blend"])) * F + float(pa["blend"]) * R.rank01(np.nan_to_num(score, nan=0.0), valid))
    el = S.eligible_mask(data, 0, "eligible")
    rep = p6.m9.sh["baseline"]["representations"][pa["rep"]]
    sigma = float(p6.m9.sh["evaluation"]["ridge_sigma_px"])
    q = R.apply(rep, s.astype(np.float32), valid, frac, known=data.labels, sigma_px=sigma, eligible=el)
    led = [S.area_ledger(q, data, 0, f"stack {pa['rep']} blend {pa['blend']} frac {frac:g}")]
    man = {"model_id": "p6_stack", "method": "p6_stack", "parameters": {**pa, "frac": frac, "ring": 0},
           "features": names, "frac_source": str(src), "candidate": cand}
    info = E._write(p6.view, "p6_stack_fusion", q, led, man, 0)
    cand["submission_file"] = info["submission"]
    p6.ws.save_json(json.loads(json.dumps(cand, default=float)), "stack/candidate.json")
    return info
