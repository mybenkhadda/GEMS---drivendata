"""Section 05: USGS 1 m DEM inventory, coverage and a geologically representative pilot (no download here).

Reuses ``gems_bench.dem.inventory`` (URL parsing, study-area intersection, one URL per tile, storage estimate) and
the master pipeline's inventory table when it exists (``data/raw/dem_1m/inventory/dem_inventory.csv``). Coverage is
never assumed: only tiles listed in the inventory and intersecting the data footprint are candidates, and the
nominal footprints are checked against real GeoTIFF headers when ``inspect_headers`` is used.

Pilot design (5-10 tiles; adaptive to the inventory): quotas per scientific question, drawn at random inside each
qualifying pool (never the "best looking" tiles), then stratified random fill (region x relief tercile):

    fault_rich        mapped-fault density in the top quintile
    fault_poor        no (or bottom-quintile) mapped faults: false-positive behaviour of DEM evidence
    low_relief_basin  relief and slope in the bottom tercile
    mountain_front    relief and slope variability in the top tercile
    disagreement      B0 / B1 rank disagreement in the top quintile
    missed_by_both    tiles containing withheld systems that B0 and B1 both miss (section 02)

Orientation diversity: the fault-rich draw prefers a second dominant strike sector when one is available.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def load_inventory(m7) -> tuple[pd.DataFrame | None, str]:
    """(inventory, provenance) from the pipeline CSV, else parsed from the URL JSON; (None, reason) if absent."""
    from ..dem import inventory as I

    d = m7.cfg["dem"]
    csv = m7.local / d["inventory_csv"]
    js = m7.local / d["urls_json"]
    m7.stage_in([d["inventory_csv"], d["urls_json"], "data/raw/tnm_items.json"])
    if csv.exists():
        inv = pd.read_csv(csv)
        if "in_study_area" not in inv:
            inv = I.study_area_intersection(inv, SimpleNamespace(data=m7.ctx.data))
        return inv, f"pipeline inventory {csv.name}"
    for p in (js, m7.local / "data/raw/tnm_items.json"):
        if p.exists():
            inv = I.study_area_intersection(I.load_inventory(p), SimpleNamespace(data=m7.ctx.data))
            return inv, f"parsed from {p.name}"
    return None, ("no DEM inventory: run the master pipeline section 9 (dem_inventory) or place tnm_urls.json "
                  "under MyDrive/GEMS_2/data/raw/dem_1m/")


def _window(ctx, r):
    from ..dem.pilot import _window as w

    return w(ctx, r)


def tile_descriptors(m7, tiles: pd.DataFrame) -> pd.DataFrame:
    """Per tile (native-grid window): footprint, label density, relief / slope, dominant strike, disagreement,
    pixels of withheld systems missed by both B0 and B1, region."""
    from ..geo_eval import region_map
    from ..gems3.representation import rank01
    from ..postprocess.topology import orientation_field
    from .failure import strike_sector

    ctx = m7.ctx
    reg = region_map(ctx.data)
    v = ctx.data.valid
    lab = np.asarray(ctx.data.labels, bool)
    elev_name = "det_elev" if "det_elev" in ctx.data.band_names else None
    slope_name = "det_elev_slope" if "det_elev_slope" in ctx.data.band_names else None
    r0m, r1m = (np.asarray(ctx.oof(m)) for m in m7.models)
    dis = np.abs(rank01(r0m, v) - rank01(r1m, v))
    ang, _ = orientation_field(lab, 2.0)
    az = (90.0 - ang) % 180.0
    missed = np.zeros(ctx.data.shape, bool)
    st = m7.ws.load_table("system_table", "failure_analysis/tables")
    if len(st):
        nb = st[(st.task == m7.cfg["evaluation"]["primary_task"]) & (st["class"] == "neither")].system_id.unique()
        missed = np.isin(np.asarray(ctx.splits.system_map), nb)
    rows = []
    for r in tiles.itertuples():
        a0, a1, b0, b1 = _window(ctx, r)
        if a1 <= a0 or b1 <= b0:
            continue
        vv = v[a0:a1, b0:b1]
        if not vv.any():
            continue
        ll = lab[a0:a1, b0:b1]
        row = {"tile_id": r.tile_id, "url": r.url, "valid_frac": float(vv.mean()),
               "label_density": float(ll[vv].mean()), "missed_by_both_px": int(missed[a0:a1, b0:b1].sum()),
               "disagreement": float(np.nanmean(dis[a0:a1, b0:b1][vv])),
               "size_bytes": getattr(r, "size_bytes", np.nan), "file": getattr(r, "file", ""),
               "project_dir": getattr(r, "project_dir", ""), "window": json.dumps([int(a0), int(a1), int(b0), int(b1)])}
        rr = reg[a0:a1, b0:b1]
        row["region"] = int(np.bincount(rr[rr >= 0]).argmax()) if (rr >= 0).any() else -1
        if elev_name:
            e = ctx.data.read_window(elev_name, a0, a1, b0, b1)
            row["relief_std_m"] = float(np.nanstd(e[vv])) if np.isfinite(e[vv]).any() else np.nan
        if slope_name:
            s = ctx.data.read_window(slope_name, a0, a1, b0, b1)
            row["slope_mean"] = float(np.nanmean(s[vv])) if np.isfinite(s[vv]).any() else np.nan
            row["slope_std"] = float(np.nanstd(s[vv])) if np.isfinite(s[vv]).any() else np.nan
        if ll.any():
            aa = np.radians(2 * az[a0:a1, b0:b1][ll])
            row["dominant_strike_deg"] = float((np.degrees(np.arctan2(np.sin(aa).mean(), np.cos(aa).mean())) / 2) % 180)
            row["strike_sector"] = strike_sector(row["dominant_strike_deg"])
        else:
            row["dominant_strike_deg"], row["strike_sector"] = np.nan, "none"
        rows.append(row)
    return pd.DataFrame(rows)


def select_pilot(desc: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    d = cfg["dem"]
    n_total = int(d["n_pilot_tiles"] if d["coverage_mode"] == "pilot" else d["expanded_max_tiles"])
    rng = np.random.default_rng(seed)
    t = desc[desc.valid_frac >= 0.3].copy()
    if not len(t):
        return t.assign(reason=[])
    q = lambda c, p: t[c].quantile(p) if c in t and t[c].notna().any() else np.nan
    pools = {
        "fault_rich": t[t.label_density >= q("label_density", 0.8)] if t.label_density.max() > 0 else t.iloc[:0],
        "fault_poor": t[t.label_density <= max(q("label_density", 0.2), 0.0)],
        "low_relief_basin": t[(t.get("relief_std_m", pd.Series(np.nan, index=t.index)) <= q("relief_std_m", 1 / 3))
                              & (t.get("slope_mean", pd.Series(np.nan, index=t.index)) <= q("slope_mean", 1 / 3))],
        "mountain_front": t[(t.get("relief_std_m", pd.Series(np.nan, index=t.index)) >= q("relief_std_m", 2 / 3))
                            & (t.get("slope_std", pd.Series(np.nan, index=t.index)) >= q("slope_std", 2 / 3))],
        "disagreement": t[t.disagreement >= q("disagreement", 0.8)],
        "missed_by_both": t[t.missed_by_both_px > 0],
    }
    chosen: dict[str, str] = {}

    def take(pool: pd.DataFrame, n: int, reason: str, prefer_new_sector: bool = False):
        pool = pool[~pool.tile_id.isin(chosen)]
        if not len(pool) or n <= 0:
            return
        idx = rng.permutation(len(pool))
        picks = []
        if prefer_new_sector and "strike_sector" in pool:
            seen = set()
            for i in idx:
                s = pool.strike_sector.iloc[i]
                if s not in seen:
                    picks.append(i)
                    seen.add(s)
                if len(picks) >= n:
                    break
        for i in idx:
            if len(picks) >= n:
                break
            if i not in picks:
                picks.append(i)
        for i in picks[:n]:
            chosen[pool.tile_id.iloc[i]] = reason

    for k, n in (d.get("quotas") or {}).items():
        if k in pools and len(chosen) < n_total:
            take(pools[k], min(int(n), n_total - len(chosen)), k, prefer_new_sector=(k == "fault_rich"))
    rest = n_total - len(chosen)
    if rest > 0:
        base = t["relief_std_m"] if "relief_std_m" in t and t.relief_std_m.notna().sum() >= 3 else t.label_density
        t["relief_tercile"] = (pd.qcut(base.rank(method="first"), 3, labels=False) if len(t) >= 3
                               else np.zeros(len(t), int))
        strata = list(t.groupby(["region", "relief_tercile"]))
        order = rng.permutation(len(strata))
        i = 0
        while rest > 0 and i < 10 * max(len(strata), 1):
            _, g = strata[order[i % len(strata)]]
            before = len(chosen)
            take(g, 1, "stratified_random (region x relief tercile)")
            rest -= len(chosen) - before
            i += 1
    out = t[t.tile_id.isin(chosen)].copy()
    out["reason"] = out.tile_id.map(chosen)
    return out.reset_index(drop=True)


def storage(m7, inv: pd.DataFrame, sel: pd.DataFrame) -> dict:
    from ..dem import inventory as I

    d = m7.cfg["dem"]
    full = I.storage_estimate(inv) if len(inv) else {}
    pil = float(sel.size_bytes.fillna(I.TYPICAL_TILE_BYTES).astype(float).sum() / 2**30) if len(sel) else 0.0
    du = shutil.disk_usage(m7.local)
    return {"pilot_tiles": int(len(sel)), "pilot_gib_estimate": pil, "all_intersecting": full,
            "free_disk_gib": du.free / 2**30, "min_free_disk_gib": float(d["min_free_disk_gib"]),
            "max_gib": float(d["max_gib"]), "download_allowed": bool(d["download"]),
            "fits": bool(pil <= float(d["max_gib"]) and du.free / 2**30 - pil >= float(d["min_free_disk_gib"])),
            "note": "estimates use TNM sizes where present, else ~350 MB per tile; measured values come from section 06"}


def run(m7) -> dict:
    from ..dem import inventory as I

    inv, prov = load_inventory(m7)
    if inv is None:
        return {"status": "NO_INVENTORY", "reason": prov}
    tiles = I.choose_per_tile(inv[inv.in_study_area.fillna(False).astype(bool)], m7.cfg["dem"]["overlap_policy"])
    desc = tile_descriptors(m7, tiles)
    sel = select_pilot(desc, m7.cfg, int(m7.cfg["seed"]))
    if len(sel) < int(m7.cfg["dem"]["min_pilot_tiles"]):
        status = f"PILOT_TOO_SMALL ({len(sel)} < {m7.cfg['dem']['min_pilot_tiles']} qualifying tiles)"
    else:
        status = "OK"
    st = storage(m7, inv, sel)
    return {"status": status, "provenance": prov, "inventory": inv, "tiles": tiles, "descriptors": desc,
            "pilot": sel, "storage": st}


def download(m7, sel: pd.DataFrame) -> pd.DataFrame:
    """Explicit download of the pilot tiles (refused unless dem.download and the storage check pass)."""
    from ..dem.download import download_selected

    d = m7.cfg["dem"]
    if not d["download"]:
        raise RuntimeError("dem.download is False: review dem/pilot/pilot_tiles.csv and the storage estimate, then "
                           "set M7_CONFIG['dem']['download'] = True")
    st = storage(m7, sel.assign(in_study_area=True), sel)
    if not st["fits"]:
        raise RuntimeError(f"pilot does not fit: {st}")
    lim = int(d["max_pilot_tiles"] if d["coverage_mode"] == "pilot" else d["expanded_max_tiles"])
    return download_selected(sel, m7.local, max_tiles=lim, max_gib=float(d["max_gib"]),
                             sync=m7.pc.sync if m7.pc is not None else None)


def tile_path(m7, row) -> Path:
    return m7.local / "data" / "raw" / "dem_1m" / str(row["project_dir"]) / str(row["file"])
