"""P7 inputs: channel registry + full-region input contract, training units, patch batches, fold provenance.

Channels
    local   the M9 cube of the configured M9 variant (section-02 / 03 selected registry, cube order, footprint
            last); every local channel is label-free and exists on the full grid, so the full-region inference
            contract holds by construction.
    extra   optional full-grid layers appended after the cube: G02 / G01 maps from ``p6.stack.g02_full``
            (label-free, normalised on the whole footprint, the map the full-region export would use) and
            allow-listed label-free model maps (``B0_lineament``). Anything else raises ``LeakageError``: an OOF CNN
            map at fold j came from a model trained on the fold P7 holds out, and a discovery map exists only inside
            one split's evaluation zone, so neither satisfies the training = inference contract.
    The registry (name, index, modality, source, contract) is saved to ``features/channel_registry.csv``.

Units (``p7_units``): the M9 definitions with P7's own scope. ``geo_f<k>``: patches fully inside fold k's inner
training zone, checkpoint on its inner-validation zone, prediction = fold k. ``disc_s<d>``: patches anywhere valid,
withheld systems + ignore ring weight 0 (never negatives, never targets), fixed epochs, prediction = evaluation zone.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..p6.stack import LeakageError

FOOTPRINT = "m9:footprint"


# =============================================================================== registry / contract
def modality_of(name: str, patterns: dict) -> str:
    n = name.lower()
    if n == FOOTPRINT:
        return "footprint"
    for mod, pats in patterns.items():
        if any(str(p).lower() in n for p in pats):
            return mod
    return "other"


def channel_registry(local: list[str], extra: list[dict], patterns: dict) -> pd.DataFrame:
    """One row per network input channel, in tensor order (cube channels, then extra layers)."""
    rows = [{"index": i, "name": n, "modality": modality_of(n, patterns), "source": "m9_cube",
             "label_free": True, "full_region": True} for i, n in enumerate(local)]
    for j, e in enumerate(extra):
        rows.append({"index": len(local) + j, "name": e["name"], "modality": e.get("modality") or
                     modality_of(e["name"], patterns), "source": e["source"], "label_free": bool(e["label_free"]),
                     "full_region": bool(e["full_region"])})
    return pd.DataFrame(rows)


def assert_contract(reg: pd.DataFrame) -> None:
    """Every input channel must be label-free and reproducible on the full grid."""
    bad = reg[~(reg.label_free.astype(bool) & reg.full_region.astype(bool))]
    if len(bad):
        raise LeakageError(f"channels violating the full-region / label-free contract: {list(bad.name)}")


def modality_groups(reg: pd.DataFrame) -> dict[str, list[int]]:
    """{modality: [tensor indices]}; the footprint joins the first non-empty group (never its own stem)."""
    g: dict[str, list[int]] = {}
    for r in reg.itertuples():
        if r.modality != "footprint":
            g.setdefault(r.modality, []).append(int(r.index))
    fp = reg[reg.modality == "footprint"]
    if len(fp) and g:
        g[next(iter(g))].append(int(fp["index"].iloc[0]))
    return g


def extra_specs(cfg: dict) -> list[dict]:
    """Configured extra layers (before any data is read); refuses non-allow-listed model maps."""
    ic, out = cfg["inputs"], []
    if ic["geological"].get("enabled"):
        out += [{"name": f"g02:{k}", "key": v, "source": "g02_full", "label_free": True, "full_region": True,
                 "modality": "geological"} for k, v in ic["geological"]["keys"].items()]
    if ic["label_free_maps"].get("enabled"):
        for m in ic["label_free_maps"]["allow"]:
            if m != "B0_lineament":
                raise LeakageError(f"label_free_maps: {m!r} is not a verified label-free map")
            out.append({"name": f"map:{m}", "key": m, "source": "label_free_map", "label_free": True,
                        "full_region": True, "modality": "other"})
    return out


def build_extra(p7, specs: list[dict]) -> np.ndarray | None:
    """(E, H, W) float32 memory-mapped extra layers under ``features/extra_<key>.npy`` (built once)."""
    if not specs:
        return None
    from ..artifacts.atomic import atomic_write_json

    key = hashlib.sha1(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:12]
    root = p7.ws.root / "features"
    p, pm = root / f"extra_{key}.npy", root / f"extra_{key}.json"
    if p.exists() and pm.exists():
        return np.load(p, mmap_mode="r")
    H, W = p7.ctx.data.shape
    mm = np.lib.format.open_memmap(str(p) + ".partial", mode="w+", dtype=np.float32, shape=(len(specs), H, W))
    g02 = None
    for i, s in enumerate(specs):
        if s["source"] == "g02_full":
            if g02 is None:
                from ..p6.stack import g02_full

                g02 = g02_full(p7.p6, {x["name"].split(":", 1)[1]: x["key"] for x in specs if x["source"] == "g02_full"})
            a = g02.get(s["name"].split(":", 1)[1])
            if a is None:
                raise FileNotFoundError(f"G02 full-grid key {s['key']} unavailable (stack.g02_full)")
        else:
            a = p7.ctx.oof(s["key"])
        mm[i] = np.asarray(a, np.float32)                  # raw values, NaN kept (FoldScaler maps NaN -> 0)
    mm.flush()
    del mm
    import os

    os.replace(str(p) + ".partial", p)
    atomic_write_json(pm, {"key": key, "specs": specs, "shape": [len(specs), H, W]})
    return np.load(p, mmap_mode="r")


# =============================================================================== units
def p7_units(p7) -> list:
    from ..m9.config import zones_of
    from ..m9.dataset import TrainUnit

    ctx = p7.ctx
    valid = np.asarray(ctx.data.valid, bool)
    lab = np.asarray(ctx.data.labels, bool)
    sc = p7.cfg["scope"]
    out = []
    for k in zones_of(sc["train_folds"], list(ctx.fold_ids)):
        allowed = ctx.folds.inner_train(k) & valid
        out.append(TrainUnit(f"geo_f{k}", "cv", int(k), allowed, allowed.copy(), lab & allowed,
                             ctx.folds.inner_val(k) & valid, ctx.folds.val(k) & valid))
    for d in zones_of(sc["train_splits"], list(range(ctx.splits.n_splits))):
        ign = ctx.splits.ignore(d)
        out.append(TrainUnit(f"disc_s{d}", "disc", int(d), valid.copy(), valid & ~ign, lab & ~ign, None,
                             ctx.splits.eval_zone(d) & valid))
    return out


def eval_key(p7, tu) -> str:
    """Evaluation unit of a training unit: geo_f<k> -> geo:<k>, disc_s<d> -> <primary task>:<d>."""
    return f"geo:{tu.zone}" if tu.kind == "cv" else f"{p7.prim}:{tu.zone}"


def mask_hash(a: np.ndarray | None) -> str | None:
    if a is None:
        return None
    a = np.asarray(a, bool)
    return hashlib.sha1(np.packbits(a.ravel()).tobytes() + str(a.shape).encode()).hexdigest()[:12]


def fold_provenance(tu) -> dict:
    return {"unit": tu.name, "kind": tu.kind, "zone": int(tu.zone), "allowed": mask_hash(tu.allowed),
            "weight_mask": mask_hash(tu.weight_mask), "visible": mask_hash(tu.visible), "inner": mask_hash(tu.inner),
            "predict": mask_hash(tu.predict), "allowed_px": int(np.asarray(tu.allowed).sum()),
            "visible_px": int(np.asarray(tu.visible).sum()),
            "inner_px": int(np.asarray(tu.inner).sum()) if tu.inner is not None else 0,
            "predict_px": int(np.asarray(tu.predict).sum()) if tu.predict is not None else 0}


def overlap_audit(p7, units: list) -> pd.DataFrame:
    """Train / validation separation of every unit (pixels): allowed vs inner, allowed vs held-out prediction zone,
    patch-halo margin (pixels of allowed within P of the held-out zone: patches never cross, only their halo of
    label-free INPUTS at inference may), visible labels inside the held-out zone, withheld systems visible."""
    from scipy import ndimage

    P = int(p7.cfg["data"]["patch_px"])
    rows = []
    for tu in units:
        al = np.asarray(tu.allowed, bool)
        pr = np.asarray(tu.predict, bool) if tu.predict is not None else np.zeros_like(al)
        r = {"unit": tu.name, "kind": tu.kind, "zone": tu.zone,
             "visible_in_heldout_px": int((np.asarray(tu.visible, bool) & pr).sum()) if tu.kind == "cv" else None}
        if tu.kind == "cv":
            r["allowed_inner_overlap_px"] = int((al & np.asarray(tu.inner, bool)).sum())
            r["allowed_heldout_overlap_px"] = int((al & pr).sum())
            near = ndimage.distance_transform_edt(~pr) <= P if pr.any() else np.zeros_like(pr)
            r["allowed_within_patch_of_heldout_px"] = int((al & near).sum())
        else:
            G = np.asarray(p7.ctx.splits.withheld(tu.zone), bool)
            r["withheld_visible_px"] = int((G & np.asarray(tu.visible, bool)).sum())
            r["withheld_weighted_px"] = int((G & np.asarray(tu.weight_mask, bool)).sum())
        hard = [k for k in ("visible_in_heldout_px", "allowed_inner_overlap_px", "allowed_heldout_overlap_px",
                            "withheld_visible_px", "withheld_weighted_px") if (r.get(k) or 0) > 0]
        r["ok"], r["violations"] = not hard, ";".join(hard)
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================== batches
class P7Batch:
    """Random training patches of one unit: scaled cube (+ extra) channels, every target, augmentation."""

    def __init__(self, p7, tu, ecfg: dict, cube, extra, sc, esc, targets: dict, groups: dict, rng):
        from ..cnn import PatchSampler
        from .augment import PRESETS

        self.tu, self.cube, self.extra, self.sc, self.esc, self.t, self.rng = tu, cube, extra, sc, esc, targets, rng
        dc = ecfg["data"]
        self.P = int(dc["patch_px"])
        valid = np.asarray(p7.ctx.data.valid, bool)
        self.valid = valid
        self.sampler = PatchSampler(tu.allowed, valid, tu.visible, self.P, float(dc["min_valid_frac"]), rng)
        self.pos_frac = float(dc["pos_patch_frac"])
        self.preset = PRESETS[dc["augmentation"]]
        self.groups = groups
        self.fp_idx = [len(sc.names) - 1] if sc.names and sc.names[-1] == FOOTPRINT else []
        self.stats: dict = {}

    def inputs(self, sl) -> np.ndarray:
        x = self.sc.apply(self.cube[:, sl[0], sl[1]])
        if self.extra is not None:
            x = np.concatenate([x, self.esc.apply(self.extra[:, sl[0], sl[1]])], 0)
        return x

    def sample(self):
        from .augment import apply_geometric, photometric, sample_ops

        P = self.P
        rc = self.sampler.draw(self.rng.random() < self.pos_frac) or self.sampler.draw(False, 1000)
        if rc is None:
            raise RuntimeError(f"{self.tu.name}: no admissible patch (zone too fragmented for {P} px patches)")
        r0, c0 = rc
        sl = (slice(r0, r0 + P), slice(c0, c0 + P))
        x = self.inputs(sl)
        t = {k: np.asarray(v[sl], np.float32) for k, v in self.t.items() if k != "sysmap"}
        t["valid"] = self.valid[sl].astype(np.float32)
        for op in sample_ops(self.rng, self.preset):
            x, t = apply_geometric(x, t, op)
            self.stats[op] = self.stats.get(op, 0) + 1
        x, ph = photometric(x, t["valid"] > 0, self.groups, self.preset, self.rng, self.fp_idx)
        for o in ph:
            self.stats[o] = self.stats.get(o, 0) + 1
        return x, t

    def batch(self, n: int) -> dict:
        import torch

        xs, ts = [], []
        for _ in range(n):
            x, t = self.sample()
            xs.append(np.ascontiguousarray(x))
            ts.append(t)
        T = {k: np.stack([np.ascontiguousarray(t[k]) for t in ts])[:, None] for k in ts[0]}
        out = {"x": np.stack(xs), "w": T.pop("weight").astype(np.float32), **{k: v.astype(np.float32) for k, v in T.items()}}
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}


# =============================================================================== geographic shift (label-free)
def geographic_shift(cube, names: list[str], train_mask: np.ndarray, test_mask: np.ndarray, n: int,
                     seed: int) -> pd.DataFrame:
    """Per channel: standardised mean difference and two-sample KS statistic between a unit's training geography
    and its held-out zone (label-free inputs only; random pixel samples)."""
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(seed)
    flat = cube.reshape(cube.shape[0], -1)

    def idx(m):
        i = np.flatnonzero(np.asarray(m, bool).ravel())
        return np.sort(rng.choice(i, min(n, i.size), replace=False)) if i.size else i

    ia, ib = idx(train_mask), idx(test_mask)
    rows = []
    for j, nm in enumerate(names):
        if nm == FOOTPRINT or not ia.size or not ib.size:
            continue
        a, b = np.asarray(flat[j, ia], np.float32), np.asarray(flat[j, ib], np.float32)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 100 or b.size < 100:
            continue
        sd = float(np.sqrt(0.5 * (a.var() + b.var())) + 1e-9)
        rows.append({"channel": nm, "smd": float((b.mean() - a.mean()) / sd), "ks": float(ks_2samp(a, b).statistic)})
    return pd.DataFrame(rows)
