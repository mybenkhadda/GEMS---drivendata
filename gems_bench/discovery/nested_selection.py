"""M6.4: leakage-safe ensemble weight selection for existing FULL models (B0 + B1 first, larger sets after).

S_ens = sum_i w_i R(S_i), R a label-free per-zone normalisation (``rank``: average-rank quantile, the
``ensemble5`` convention; ``robust``: robust min-max between score quantiles), w on an explicit grid.

Geographic (nested): for held-out fold k the pair (weights, area fraction) maximises the pooled DTI of the folds
!= k; fold k is scored once with that choice. Optional refinement around the coarse choice is itself made on the
folds != k (``refine=True``: a finer local grid around fold k's own dev-selected weight, only when that weight is
interior). Stitching fold k's chosen combination gives an OOF map that is itself a held-out prediction.

Pseudo-discovery: the weights of split d never see split d's withheld systems:
``geo_masked``  chosen on the geographic OOF maps with split d's withheld systems + ignore ring removed from both
                truth and scored area (``ensemble5.weights_geo_masked`` on an explicit grid);
``loso``        chosen on the other discovery splits' predictions (leave-one-split-out).
The discovery area budget is then cross-fitted over splits by the shared scorecard (``discovery3``).

``index_audit`` lists, for every choice, the zones whose labels were read and the zone that was scored; the
choice is rejected (``LeakageError``) if they overlap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..p3.curves import DEFAULT_AREA_FRACS, BinaryCurve, crossfit_select, dti_from, zone_curves
from ..p3.ensemble3 import simplex
from ..p3.transforms import quantile_map

NORMALISATIONS = ("rank", "robust")


class LeakageError(RuntimeError):
    pass


# =============================================================================== normalisation / combination
def normalise(p: np.ndarray, valid: np.ndarray, zones: np.ndarray | None, how: str) -> np.ndarray:
    """Label-free per-zone normalisation (zones None = one zone over ``valid``)."""
    if how == "rank":
        return quantile_map(p, valid, zones, "average")
    if how == "robust":
        from ..evaluation.calibration import robust_quantile_map

        z = zones if zones is not None else np.where(valid, 0, -1)
        return robust_quantile_map(p, valid, z)
    raise ValueError(f"normalisation {how!r} not in {NORMALISATIONS}")


def normalised_members(ctx, maps: dict[str, np.ndarray], how: str) -> dict[str, np.ndarray]:
    return {m: normalise(p, ctx.data.valid, ctx.folds.fold_map, how) for m, p in maps.items()}


def combine(U: dict[str, np.ndarray], w: dict[str, float], valid: np.ndarray) -> np.ndarray:
    e = sum(float(w.get(m, 0.0)) * np.nan_to_num(U[m]) for m in U)
    return np.where(valid, e, np.nan).astype(np.float32)


def weight_grid(members: list[str], alphas=(0.0, 0.25, 0.5, 0.75, 1.0), simplex_step: float = 0.25) -> list[dict]:
    """Pairs: w = (alpha, 1 - alpha) on ``alphas``; larger sets: the simplex lattice of ``simplex_step``."""
    if len(members) == 2:
        a, b = members
        return [{a: round(float(x), 6), b: round(1.0 - float(x), 6)} for x in alphas]
    return [dict(zip(members, np.round(w, 6).tolist())) for w in simplex(len(members), simplex_step)]


def local_grid(members: list[str], centre: dict, step: float, halfwidth: float) -> list[dict]:
    """Finer pair grid around ``centre`` (pairs only; larger sets are not refined)."""
    if len(members) != 2:
        return []
    a, b = members
    c = centre[a]
    xs = np.round(np.arange(max(0.0, c - halfwidth), min(1.0, c + halfwidth) + 1e-9, step), 6)
    return [{a: float(x), b: round(1.0 - float(x), 6)} for x in xs]


def _wkey(w: dict) -> str:
    return json.dumps({k: round(float(v), 6) for k, v in w.items()}, sort_keys=True)


def weight_curves(ctx, U: dict[str, np.ndarray], weights: list[dict], fracs=DEFAULT_AREA_FRACS,
                  done: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-fold binary-mask curves of every weight vector (``done``: rows already computed, keyed by ``w``)."""
    have = set(done.w) if done is not None and len(done) else set()
    rows = [done] if done is not None and len(done) else []
    for w in weights:
        k = _wkey(w)
        if k in have:
            continue
        zc = zone_curves(combine(U, w, ctx.data.valid), ctx.data.labels, ctx.folds.fold_map, ctx.data.valid, fracs=fracs)
        zc["w"] = k
        rows.append(zc)
        have.add(k)
    return pd.concat(rows, ignore_index=True)


# =============================================================================== nested geographic selection
def _select_for_zone(table: pd.DataFrame, z, candidates: set[str] | None = None) -> dict:
    """Best (w, target_frac) on zones != z among ``candidates``; the row for z is never read."""
    t = table if candidates is None else table[table.w.isin(candidates)]
    dev = t[t.zone != z]
    s = dev.groupby(["w", "target_frac"]).apply(dti_from).rename("dti_dev").reset_index()
    best = s.sort_values(["dti_dev", "w"], ascending=[False, True]).iloc[0]
    held = t[(t.zone == z) & (t.w == best.w) & np.isclose(t.target_frac, best.target_frac)]
    return {"zone": z, "w": best.w, "target_frac": float(best.target_frac), "dti_dev": float(best.dti_dev),
            "tp_w": held.tp_w.sum(), "fp_w": held.fp_w.sum(), "fn_w": held.fn_w.sum(), "dti_heldout": dti_from(held),
            "selection_zones": ",".join(str(x) for x in sorted(dev.zone.unique())),
            "n_candidates": int(s.w.nunique())}


def nested_geo_selection(ctx, U: dict[str, np.ndarray], members: list[str], *, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                         simplex_step: float = 0.25, fracs=DEFAULT_AREA_FRACS, refine: bool = False,
                         refine_step: float = 0.05, refine_halfwidth: float = 0.2) -> dict:
    """{"per_fold", "summary", "curves", "audit"}: nested (weights, budget) choice per held-out fold."""
    coarse = weight_grid(members, alphas, simplex_step)
    coarse_keys = {_wkey(w) for w in coarse}
    table = weight_curves(ctx, U, coarse, fracs)
    per = []
    for z in ctx.fold_ids:
        # coarse grid only: weights added by another fold's refinement were centred with zone z's labels
        r = _select_for_zone(table, z, coarse_keys)
        r["stage"] = "coarse"
        if refine and len(members) == 2:
            c = json.loads(r["w"])
            if 0.0 < c[members[0]] < 1.0:                  # interior on the dev folds: refine there, on dev folds
                loc = local_grid(members, c, refine_step, refine_halfwidth)
                table = weight_curves(ctx, U, loc, fracs, done=table)
                r = _select_for_zone(table, z, coarse_keys | {_wkey(w) for w in loc})
                r["stage"] = "refined"
        per.append(r)
    per = pd.DataFrame(per)
    audit = index_audit(per, "geographic", ctx.fold_ids)
    summ = {"dti_crossfit": dti_from(per), "fold_min": float(per.dti_heldout.min()),
            "fold_max": float(per.dti_heldout.max()), "weights_per_fold": per.set_index("zone").w.to_dict(),
            "frac_per_fold": per.set_index("zone").target_frac.to_dict(),
            "weight_dispersion": _weight_dispersion(per.w, members)}
    _, insample = crossfit_select(table[table.w.isin(coarse_keys)], ["w", "target_frac"])
    summ["dti_insample_coarse"] = insample["dti_insample"]
    return {"per_fold": per, "summary": summ, "curves": table, "audit": audit}


def _weight_dispersion(ws, members) -> float:
    W = np.array([[json.loads(w).get(m, 0.0) for m in members] for w in ws])
    return float(np.abs(W - W.mean(0)).max()) if len(W) else np.nan


def stitched(ctx, U: dict[str, np.ndarray], per_fold: pd.DataFrame) -> np.ndarray:
    """OOF ensemble: fold k uses the weights chosen on folds != k (values outside fold zones stay NaN)."""
    out = np.full(ctx.data.shape, np.nan, np.float32)
    for r in per_fold.itertuples():
        z = (ctx.folds.fold_map == r.zone) & ctx.data.valid
        out[z] = combine(U, json.loads(r.w), ctx.data.valid)[z]
    return out


def index_audit(per: pd.DataFrame, task: str, zones) -> pd.DataFrame:
    """Selection vs evaluation zones of every choice; raises LeakageError on any overlap."""
    rows = []
    for r in per.to_dict("records"):
        sel = [int(x) for x in str(r["selection_zones"]).split(",") if x != ""]
        ok = int(r["zone"]) not in sel
        rows.append({"task": task, "heldout_zone": int(r["zone"]), "selection_zones": r["selection_zones"],
                     "all_zones": ",".join(map(str, zones)), "chosen": r.get("w", ""), "frac": r.get("target_frac"),
                     "heldout_labels_used_for_selection": not ok})
        if not ok:
            raise LeakageError(f"{task}: zone {r['zone']} was used to select its own parameters")
    return pd.DataFrame(rows)


# =============================================================================== discovery weights
def disc_normalised(ctx, members: list[str], d: int, how: str) -> dict[str, np.ndarray] | None:
    ez = ctx.splits.eval_zone(d) & ctx.data.valid
    out = {}
    for m in members:
        p = ctx.discovery_pred(m, d)
        if p is None:
            return None
        out[m] = normalise(np.where(ez, p, np.nan), ez, None, how)
    return out


def weights_geo_masked(ctx, U: dict[str, np.ndarray], d: int, weights: list[dict], fracs=DEFAULT_AREA_FRACS) -> tuple[dict, dict]:
    """Best weights on the geographic OOF maps with split d's withheld systems + ignore ring removed."""
    ign = ctx.splits.ignore(d)
    v = ctx.data.valid & ~ign
    lab = ctx.data.labels & ~ign
    best = (-1.0, None)
    for w in weights:
        zc = zone_curves(combine(U, w, v), lab, ctx.folds.fold_map, v, fracs=fracs)
        s = float(zc.groupby("target_frac").apply(dti_from).max())
        if s > best[0] or (s == best[0] and _wkey(w) < _wkey(best[1])):
            best = (s, w)
    return best[1], {"split": d, "method": "geo_masked", "dti_dev": best[0],
                     "selection_data": f"geographic OOF, split {d} withheld systems + ignore ring removed",
                     "uses_split_labels": False}


def weights_loso(ctx, members: list[str], d: int, weights: list[dict], how: str, policy: str = "excl",
                 fracs=DEFAULT_AREA_FRACS) -> tuple[dict, dict]:
    """Best weights on the other discovery splits (pooled best area under ``policy``)."""
    from ..p3 import discovery3

    others = [x for x in range(ctx.splits.n_splits) if x != d]
    Q = {x: disc_normalised(ctx, members, x, how) for x in others}
    masks = {x: discovery3.split_masks(ctx, x) for x in others}
    best = (-1.0, None)
    for w in weights:
        rows = []
        for x in others:
            if Q[x] is None:
                continue
            m = masks[x]
            truth, ex = discovery3.policy_truth_exclude(m, policy)
            c = BinaryCurve(combine(Q[x], w, m["ez"]), truth, m["ez"], ex)
            rows.append(c.at_area(fracs).assign(zone=x))
        if not rows:
            continue
        s = float(pd.concat(rows).groupby("target_frac").apply(dti_from).max())
        if s > best[0] or (s == best[0] and _wkey(w) < _wkey(best[1])):
            best = (s, w)
    return best[1], {"split": d, "method": "loso", "dti_dev": best[0],
                     "selection_data": f"discovery splits {others} under {policy}", "uses_split_labels": False}


def discovery_maps(ctx, members: list[str], weights_by_split: dict[int, dict], how: str) -> dict[int, np.ndarray]:
    out = {}
    for d, w in weights_by_split.items():
        Q = disc_normalised(ctx, members, d, how)
        if Q is not None:
            out[d] = combine(Q, w, ctx.splits.eval_zone(d) & ctx.data.valid)
    return out


# =============================================================================== candidate builder
@dataclass
class EnsembleCandidate:
    name: str
    members: list[str]
    method: str                 # mean | rank_mean | weighted
    normalisation: str          # none | rank | robust
    discovery_weights: str      # fixed | geo_masked | loso
    oof: np.ndarray | None = None
    disc: dict = field(default_factory=dict)
    geo_choice: pd.DataFrame = field(default_factory=pd.DataFrame)
    disc_choice: pd.DataFrame = field(default_factory=pd.DataFrame)
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)

    def record(self) -> dict:
        s = self.summary
        return {"candidate": self.name, "members": "+".join(self.members), "n_members": len(self.members),
                "method": self.method, "normalisation": self.normalisation,
                "discovery_weights": self.discovery_weights,
                "geo_weights_per_fold": json.dumps(s.get("weights_per_fold", {})),
                "geo_frac_per_fold": json.dumps(s.get("frac_per_fold", {})),
                "disc_weights_per_split": json.dumps(s.get("disc_weights", {})),
                "weight_dispersion": s.get("weight_dispersion", np.nan),
                "nested_dti_selection": s.get("dti_crossfit", np.nan),
                "area_policy": "per-fold area fraction cross-fitted (geographic) / per-split (discovery)"}


def build_candidate(ctx, maps: dict[str, np.ndarray], members: list[str], method: str, normalisation: str = "rank",
                    *, discovery_weights: str = "geo_masked", alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                    simplex_step: float = 0.25, fracs=DEFAULT_AREA_FRACS, refine: bool = False,
                    refine_step: float = 0.05, refine_halfwidth: float = 0.2, policy: str = "excl",
                    with_discovery: bool = True) -> EnsembleCandidate:
    """One ensemble candidate with its stitched OOF map, discovery maps and full selection record."""
    short = {"mean": "mean", "rank_mean": "rankmean", "weighted": f"w{normalisation}"}[method]
    tag = "+".join(m.replace("P5_", "") for m in members)
    dw = "fixed" if method in ("mean", "rank_mean") else discovery_weights
    name = f"M6_ENS_{tag}_{short}" + ("" if dw in ("fixed", "geo_masked") else f"_{dw}")
    cand = EnsembleCandidate(name, list(members), method, "none" if method == "mean" else
                             ("rank" if method == "rank_mean" else normalisation), dw)
    v = ctx.data.valid
    equal = {m: 1.0 / len(members) for m in members}
    if method == "mean":
        cand.oof = np.where(v, np.nanmean(np.stack([maps[m] for m in members]), 0), np.nan).astype(np.float32)
        cand.summary = {"weights_per_fold": {k: _wkey(equal) for k in ctx.fold_ids}}
    elif method == "rank_mean":
        U = normalised_members(ctx, {m: maps[m] for m in members}, "rank")
        cand.oof = combine(U, equal, v)
        cand.summary = {"weights_per_fold": {k: _wkey(equal) for k in ctx.fold_ids}}
    elif method == "weighted":
        U = normalised_members(ctx, {m: maps[m] for m in members}, normalisation)
        sel = nested_geo_selection(ctx, U, members, alphas=alphas, simplex_step=simplex_step, fracs=fracs,
                                   refine=refine, refine_step=refine_step, refine_halfwidth=refine_halfwidth)
        cand.oof = stitched(ctx, U, sel["per_fold"])
        cand.geo_choice, cand.audit, cand.summary = sel["per_fold"], sel["audit"], dict(sel["summary"])
    else:
        raise ValueError(method)
    if not with_discovery:
        return cand
    grid = weight_grid(members, alphas, simplex_step)
    by_split, rows = {}, []
    for d in range(ctx.splits.n_splits):
        if dw == "fixed":
            w, info = equal, {"split": d, "method": "fixed", "selection_data": "none (equal weights)",
                              "uses_split_labels": False}
        elif dw == "geo_masked":
            w, info = weights_geo_masked(ctx, U, d, grid, fracs)
        elif dw == "loso":
            w, info = weights_loso(ctx, members, d, grid, normalisation, policy, fracs)
        else:
            raise ValueError(dw)
        by_split[d] = w
        rows.append({**info, "weights": _wkey(w)})
    if method == "mean":
        cand.disc = {d: _disc_mean(ctx, members, d) for d in range(ctx.splits.n_splits)}
    else:
        cand.disc = discovery_maps(ctx, members, by_split, cand.normalisation)
    cand.disc_choice = pd.DataFrame(rows)
    cand.summary["disc_weights"] = {d: _wkey(w) for d, w in by_split.items()}
    if any(r.get("uses_split_labels") for r in rows):
        raise LeakageError(f"{name}: discovery weights used the scored split's labels")
    return cand


def _disc_mean(ctx, members, d):
    ez = ctx.splits.eval_zone(d) & ctx.data.valid
    ps = [ctx.discovery_pred(m, d) for m in members]
    return np.where(ez, np.nanmean(np.stack(ps), 0), np.nan).astype(np.float32)


def candidate_table(score: pd.DataFrame, cands: list[EnsembleCandidate], members: dict[str, str], refs: list[str],
                    cost: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per candidate (and per single member, ``members``: stem -> model) with scorecard metrics, deltas
    vs ``refs`` and the added inference / storage cost (fold models to run, checkpoint MB)."""
    s = score.set_index("model")
    cols = ("geo_dti", "geo_fold_min", "geo_fold_max", "disc_dti_excl", "disc_fold_min_excl", "disc_dti_retained_fp",
            "short_recall_w", "geo_frac", "geo_far_share_at_match", "disc_frac_excl")
    rows = []
    entries = [(m, {"candidate": m, "members": m, "n_members": 1, "method": "single", "normalisation": "none",
                    "discovery_weights": "n/a"}) for m in members.values()]
    entries += [(c.name, c.record()) for c in cands]
    for name, rec in entries:
        if name not in s.index:
            continue
        r = {**rec, **{c: s.loc[name].get(c, np.nan) for c in cols}}
        fc = [c for c in s.columns if c.startswith("dti_fold")]
        r["geo_fold_std"] = float(s.loc[name, fc].astype(float).std(ddof=0)) if fc else np.nan
        for ref in refs:
            if ref in s.index:
                for c in ("geo_dti", "geo_fold_min", "disc_dti_excl", "disc_dti_retained_fp", "short_recall_w"):
                    r[f"delta_{c}_vs_{ref}"] = s.loc[name].get(c, np.nan) - s.loc[ref].get(c, np.nan)
        mem = rec["members"].split("+")
        if cost:
            r["n_fold_models_inference"] = int(sum(cost.get(m, {}).get("n_checkpoints", 0) for m in mem))
            r["checkpoint_mb"] = float(sum(cost.get(m, {}).get("checkpoint_mb", 0.0) for m in mem))
            r["relative_inference_cost"] = float(len(mem))
        rows.append(r)
    return pd.DataFrame(rows)
