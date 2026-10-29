"""Fault-perpendicular profiles with orientation- and region-matched background.

Each *fault station* is a point on a mapped segment with its local strike; the
profile runs along the ground normal. Each *background station* is a random
location in the same region, far from any label, sampled with the same normal
direction, so that regional fabric and orientation effects cancel in the comparison.

Polarity: a step across a fault has an arbitrary sign relative to the chosen
normal, so aggregate profiles are mirrored so that the right-hand side is the
higher side ("polarity-aligned"). Background profiles get the same treatment.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from .io import Grid
from .morphology import FaultNetwork, axial_deg, region_ids
from .stats import auc_score, percentile_in


@dataclass
class ProfileSet:
    stations: pd.DataFrame       # sample_id, kind, seg_id, r, c, dr, dc, strike, region, matched_to
    offsets_km: np.ndarray
    signals: list[str]
    cube: np.ndarray             # (n_stations, n_signals, n_offsets)

    def kind(self, kind: str) -> np.ndarray:
        return (self.stations["kind"] == kind).to_numpy()

    def to_long_table(self) -> pd.DataFrame:
        """One row per (station, signal); one column per offset (named by offset in km)."""
        n, s, _ = self.cube.shape
        df = pd.DataFrame(self.cube.reshape(n * s, -1), columns=[f"{o:+.3f}" for o in self.offsets_km])
        df.insert(0, "signal", np.tile(self.signals, n))
        df.insert(0, "sample_id", np.repeat(self.stations.sample_id.to_numpy(), s))
        return df

    def save(self, stations_path, profiles_path) -> None:
        self.stations.to_parquet(stations_path, index=False)
        self.to_long_table().to_parquet(profiles_path, index=False)

    @classmethod
    def load(cls, stations_path, profiles_path) -> "ProfileSet":
        st = pd.read_parquet(stations_path)
        long = pd.read_parquet(profiles_path)
        signals = list(dict.fromkeys(long.signal))
        offset_cols = [c for c in long.columns if c not in ("sample_id", "signal")]
        cube = long[offset_cols].to_numpy(np.float32).reshape(len(st), len(signals), len(offset_cols))
        return cls(st, np.array([float(c) for c in offset_cols]), signals, cube)


def _nanmean(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=-1)


def _local_frame(e, n, s, pos, half):
    """PCA tangent of path points within +-half km arc of pos -> unit (te, tn)."""
    sel = np.abs(s - pos) <= half
    if sel.sum() < 3:
        sel = np.argsort(np.abs(s - pos))[:3]
    pe, pn = e[sel] - e[sel].mean(), n[sel] - n[sel].mean()
    cov = np.array([[pe @ pe, pe @ pn], [pe @ pn, pn @ pn]])
    w, v = np.linalg.eigh(cov)
    return v[0, 1], v[1, 1]


def sample_fault_stations(net: FaultNetwork, grid: Grid, valid: np.ndarray, spacing_km: float,
                          tangent_km: float, min_seg_km: float, edge_px: int,
                          region_grid, rng: np.random.Generator, max_stations: int) -> pd.DataFrame:
    rows = []
    for sid, path, length in zip(net.segments.seg_id, net.paths, net.segments.length_km):
        if length < min_seg_km or len(path) < 3:
            continue
        e, n = grid.rc_to_km(path[:, 0], path[:, 1])
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(e), np.diff(n)))])
        for pos in np.arange(spacing_km / 2, s[-1], spacing_km):
            i = int(np.argmin(np.abs(s - pos)))
            r, c = path[i]
            if not (edge_px <= r < grid.height - edge_px and edge_px <= c < grid.width - edge_px) or not valid[r, c]:
                continue
            te, tn = _local_frame(e, n, s, pos, tangent_km)
            ne, nn = tn, -te                                  # ground normal (east, north), unit length
            rows.append({"seg_id": int(sid), "r": int(r), "c": int(c),
                         "dr": -nn / grid.pixel_km_y, "dc": ne / grid.pixel_km_x,   # pixels per km
                         "strike": axial_deg(np.degrees(np.arctan2(te, tn)))})
    st = pd.DataFrame(rows, columns=["seg_id", "r", "c", "dr", "dc", "strike"])   # typed even when empty
    if len(st) > max_stations:
        st = st.sample(max_stations, random_state=int(rng.integers(1 << 31))).sort_values(["seg_id", "r"])
    st = st.reset_index(drop=True)
    st["kind"] = "fault"
    st["region"] = region_ids(st.r, st.c, grid, region_grid)
    return st


def sample_background_stations(fault_st: pd.DataFrame, net: FaultNetwork, grid: Grid, valid: np.ndarray,
                               min_dist_km: float, edge_px: int, region_grid,
                               rng: np.random.Generator) -> pd.DataFrame:
    edge_px = max(int(edge_px), 1)
    ok = valid & (net.distance_km > min_dist_km)
    ok[:edge_px], ok[-edge_px:], ok[:, :edge_px], ok[:, -edge_px:] = False, False, False, False
    rr, cc = np.nonzero(ok)
    if len(rr) == 0:
        raise ValueError("No background pixels: lower background_min_km")
    reg = region_ids(rr, cc, grid, region_grid)
    picks = np.empty(len(fault_st), dtype=np.int64)
    for rid, idx in fault_st.groupby("region").groups.items():
        pool = np.flatnonzero(reg == rid)
        if len(pool) < 50:
            pool = np.arange(len(rr))
        picks[np.asarray(idx)] = rng.choice(pool, size=len(idx))
    bg = fault_st[["dr", "dc", "strike"]].copy()
    bg["seg_id"] = -1
    bg["r"], bg["c"] = rr[picks], cc[picks]
    bg["kind"] = "background"
    bg["region"] = reg[picks]
    bg["matched_to"] = np.arange(len(fault_st))
    return bg


def build_stations(fault_st: pd.DataFrame, bg_st: pd.DataFrame) -> pd.DataFrame:
    fault_st = fault_st.copy()
    fault_st["matched_to"] = np.arange(len(fault_st))
    st = pd.concat([fault_st, bg_st], ignore_index=True)
    st.insert(0, "sample_id", np.arange(len(st)))
    return st


def extract_profiles(signals: dict[str, np.ndarray], stations: pd.DataFrame, offsets_km: np.ndarray) -> ProfileSet:
    rr = stations.r.to_numpy()[:, None] + offsets_km[None, :] * stations.dr.to_numpy()[:, None]
    cc = stations.c.to_numpy()[:, None] + offsets_km[None, :] * stations.dc.to_numpy()[:, None]
    coords = np.vstack([rr.ravel(), cc.ravel()])
    cube = np.empty((len(stations), len(signals), len(offsets_km)), np.float32)
    for i, arr in enumerate(signals.values()):
        cube[:, i, :] = ndimage.map_coordinates(arr, coords, order=1, mode="constant", cval=np.nan,
                                                prefilter=False).reshape(rr.shape)
    return ProfileSet(stations, offsets_km, list(signals), cube)


def profile_metrics(ps: ProfileSet, core_km: float) -> pd.DataFrame:
    """Per (station, signal): step across the fault and core anomaly relative to the flanks."""
    o = ps.offsets_km
    c = _nanmean(ps.cube[..., np.abs(o) <= core_km])
    lft = _nanmean(ps.cube[..., o < -core_km])
    rgt = _nanmean(ps.cube[..., o > core_km])
    step = rgt - lft
    peak = c - 0.5 * (lft + rgt)
    n, s = step.shape
    return pd.DataFrame({
        "sample_id": np.repeat(ps.stations.sample_id.to_numpy(), s),
        "kind": np.repeat(ps.stations.kind.to_numpy(), s),
        "seg_id": np.repeat(ps.stations.seg_id.to_numpy(), s),
        "signal": np.tile(ps.signals, n),
        "step": step.ravel(), "abs_step": np.abs(step).ravel(),
        "peak": peak.ravel(), "abs_peak": np.abs(peak).ravel(),
    })


def signal_effects(metrics: pd.DataFrame, modality_of) -> pd.DataFrame:
    """Fault-vs-background AUC for step, signed peak and |peak|, per signal."""
    rows = []
    for sig, g in metrics.groupby("signal", sort=False):
        f, b = g[g.kind == "fault"], g[g.kind == "background"]
        r = {"signal": sig, "modality": modality_of(sig), "n_fault": int(f.abs_step.notna().sum())}
        for col in ("abs_step", "peak", "abs_peak"):
            r[f"auc_{col}"] = auc_score(f[col], b[col])
        r["auc_trough"] = 1 - r["auc_peak"] if np.isfinite(r["auc_peak"]) else np.nan
        best = max(("abs_step", "abs_peak", "peak", "trough"), key=lambda k: np.nan_to_num(r[f"auc_{k}"], nan=0))
        r["best_metric"], r["best_auc"] = best, r[f"auc_{best}"]
        r["median_abs_step_fault"], r["median_abs_step_bg"] = f.abs_step.median(), b.abs_step.median()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("best_auc", ascending=False, ignore_index=True)


def aligned_cube(ps: ProfileSet, core_km: float) -> np.ndarray:
    """Mirror each profile so that the higher flank is on the positive-offset side."""
    o = ps.offsets_km
    step = _nanmean(ps.cube[..., o > core_km]) - _nanmean(ps.cube[..., o < -core_km])
    return np.where((step < 0)[..., None], ps.cube[..., ::-1], ps.cube)


def aggregate_profiles(ps: ProfileSet, core_km: float) -> pd.DataFrame:
    """Median and IQR of polarity-aligned profiles, per signal and station kind."""
    cube = aligned_cube(ps, core_km)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for kind in ("fault", "background"):
            sub = cube[ps.kind(kind)]
            q = np.nanpercentile(sub, [25, 50, 75], axis=0)       # (3, n_signals, n_offsets)
            for i, sig in enumerate(ps.signals):
                rows.append(pd.DataFrame({"signal": sig, "kind": kind, "offset_km": ps.offsets_km,
                                          "q25": q[0, i], "q50": q[1, i], "q75": q[2, i]}))
    return pd.concat(rows, ignore_index=True)


def response_width_km(agg: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """Full width at half maximum of the fault response (gradient of a step, or the peak itself)."""
    rows = []
    for sig, g in agg.groupby("signal", sort=False):
        f = g[g.kind == "fault"].sort_values("offset_km")
        b = g[g.kind == "background"].sort_values("offset_km")
        o = f.offset_km.to_numpy()
        metric = effects.set_index("signal").loc[sig, "best_metric"]
        if metric == "abs_step":
            resp = np.abs(np.gradient(f.q50.to_numpy() - b.q50.to_numpy(), o))
        else:
            resp = np.abs(f.q50.to_numpy() - b.q50.to_numpy())
        if not np.isfinite(resp).any() or np.nanmax(resp) <= 0:
            rows.append({"signal": sig, "fwhm_km": np.nan})
            continue
        resp = np.nan_to_num(resp)
        above = np.flatnonzero(resp >= 0.5 * resp.max())
        # contiguous run containing the maximum
        k = int(np.argmax(resp))
        lo, hi = k, k
        while lo - 1 in above:
            lo -= 1
        while hi + 1 in above:
            hi += 1
        rows.append({"signal": sig, "fwhm_km": float(o[hi] - o[lo] + (o[1] - o[0]))})
    return pd.DataFrame(rows)


def station_evidence(metrics: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Calibrated per-station evidence per modality group.

    For each signal, the |step| and |peak| of a station are converted to percentiles
    of the *background* distribution; a signal score is the max of the two; a group
    score is the max over the group's signals, re-calibrated against the background
    group scores. Background stations are therefore ~Uniform(0, 1) per group, and a
    threshold of 0.9 means a 10 % false-alarm rate per group.
    """
    m = metrics.copy()
    for col in ("abs_step", "abs_peak"):
        m[f"p_{col}"] = np.nan
        for sig, g in m.groupby("signal", sort=False):
            ref = g.loc[g.kind == "background", col]
            m.loc[g.index, f"p_{col}"] = percentile_in(g[col], ref)
    m["score"] = m[["p_abs_step", "p_abs_peak"]].max(axis=1)
    wide = m.pivot_table(index=["sample_id", "kind", "seg_id"], columns="signal", values="score").reset_index()
    out = wide[["sample_id", "kind", "seg_id"]].copy()
    bg = (out.kind == "background").to_numpy()
    for grp, sigs in groups.items():
        sigs = [s for s in sigs if s in wide.columns]
        if not sigs:
            continue
        raw = wide[sigs].max(axis=1).to_numpy()
        out[f"ev_{grp}"] = percentile_in(raw, raw[bg])
    return out


def segment_evidence(evidence: pd.DataFrame, strong: float) -> pd.DataFrame:
    ev_cols = [c for c in evidence.columns if c.startswith("ev_")]
    f = evidence[evidence.kind == "fault"]
    seg = f.groupby("seg_id")[ev_cols].median()
    seg["n_stations"] = f.groupby("seg_id").size()
    seg["n_strong_groups"] = (seg[ev_cols] >= strong).sum(axis=1)
    seg["max_ev"] = seg[ev_cols].max(axis=1)
    return seg.reset_index()
