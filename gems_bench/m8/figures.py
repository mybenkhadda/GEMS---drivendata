"""M8 figures: M7's 16:9 analytical panels and true-geometry maps (equal aspect, metre extents, labels drawn as
lines, predictions as filled rasters) + the M8 case-study panel set:

    1 raw magnetic field          2 magnetic structural response    3 gravity structural response
    4 surface / DEM evidence      5 B0 / B1 prediction              6 candidate lineaments
    7 Monte Carlo occupancy       8 known fault geometry + scored system

Every panel shares one window and one normalisation per layer type; the arrays are saved next to the figure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..m7.figures import bar_panels16, hillshade, line_panels16, map_panels  # noqa: F401 (re-exported)


def _win(m8, name: str, r0, r1, c0, c1):
    try:
        return m8.layers.window(name, r0, r1, c0, c1) if m8.layers.has(name) else None
    except KeyError:
        return None


def case_panels8(m8, task: str, zone: int, system_id: int, title: str) -> tuple[object, dict]:
    from ..m7.failure import case_window

    ctx = m8.ctx
    fc = m8.sh["failure"]
    r0, r1, c0, c1 = case_window(ctx, int(system_id), fc["case_margin_km"], fc["case_min_half_km"])
    win = (slice(r0, r1), slice(c0, c1))
    lab = np.asarray(ctx.data.labels, bool)[win]
    sm = np.asarray(ctx.splits.system_map)[win]
    this = sm == int(system_id)
    if task == "geo":
        known = lab & ~this
        maps = [np.asarray(ctx.oof(m))[win] for m in m8.models]
    else:
        G = ctx.splits.withheld(int(zone))[win]
        known = lab & ~G
        maps = [np.asarray(p)[win] if (p := ctx.discovery_pred(m, int(zone))) is not None else
                np.full(lab.shape, np.nan, np.float32) for m in m8.models]
    rd = lambda b: ctx.data.read_window(b, r0, r1, c0, c1) if b in ctx.data.band_names else np.full(lab.shape, np.nan)
    fams = m8.cfg["bands"]["families"]
    mag = rd(fams["magnetic"]["field"])
    mresp = _win(m8, "magnetic:tilt_edge@400", r0, r1, c0, c1)
    if mresp is None:
        mresp = _win(m8, "magnetic:thdr@400", r0, r1, c0, c1)
    gresp = _win(m8, "gravity:thdr@1600", r0, r1, c0, c1)
    hs = hillshade(rd(fams["terrain"]["field"]))
    dem = m8.dem_layer()
    demw = dem[win] if dem is not None else None
    occ = m8.mc_occupancy()
    occw = np.asarray(occ[win], np.float32) if occ is not None else None
    cand = m8.full_candidate_raster()
    candw = cand[win] > 0 if cand is not None else None
    nn = np.nanmean(np.stack([np.asarray(x, np.float32) for x in maps]), 0)
    layers = [
        {"title": f"1 raw magnetic ({fams['magnetic']['field']})", "raster": mag, "cmap": "RdBu_r", "truth": this},
        {"title": "2 magnetic structural response (tilt edge 400 m)", "raster": mresp, "cmap": "magma", "vmin": 0,
         "vmax": 1, "truth": this},
        {"title": "3 gravity structural response (THDR 1.6 km)", "raster": gresp, "cmap": "cividis", "truth": this},
        {"title": "4 surface: hillshade" + (" + 1 m DEM scarp percentile" if demw is not None else ""),
         "raster": demw if demw is not None and np.isfinite(demw).any() else hs, "cmap": "Greys_r", "vmin": 0, "vmax": 1,
         "truth": this},
        {"title": "5 B0 / B1 mean probability (honest map)", "raster": nn, "cmap": "viridis", "vmin": 0, "vmax": 1,
         "truth": this},
        {"title": "6 candidate lineaments (blue)", "raster": hs, "cmap": "Greys_r", "vmin": 0, "vmax": 1, "cand": candw,
         "truth": this, "legend": candw is not None},
        {"title": "7 Monte Carlo occupancy" + ("" if occw is not None else " (not computed here)"), "raster": occw,
         "cmap": "inferno", "vmin": 0, "vmax": 1, "truth": this},
        {"title": "8 known faults (black) + scored system (red)", "raster": None, "known": known, "truth": this,
         "legend": True},
    ]
    fig = map_panels(ctx.data.grid, (r0, r1, c0, c1), layers, title=title, ncols=4)
    data = {"window": np.array([r0, r1, c0, c1]), "system": this, "known": known, "raw_magnetic": mag,
            "b0b1_mean": nn, **({"magnetic_response": mresp} if mresp is not None else {}),
            **({"gravity_response": gresp} if gresp is not None else {}), **({"occupancy": occw} if occw is not None else {}),
            **({"candidates": candw} if candw is not None else {})}
    return fig, data


def export_cases(m8, cases: pd.DataFrame, sub: str, per_class: int) -> pd.DataFrame:
    import json

    if not len(cases):
        return pd.DataFrame()
    rng = np.random.default_rng(int(m8.cfg["seed"]))
    idx = []
    for cls, g in cases.groupby("case_class"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).head(per_class)
        for r in g.to_dict("records"):
            name = f"{cls}__{r['task']}_z{int(r['zone'])}_sys{int(r['system_id'])}"
            title = (f"{cls} | {r['task']} zone {int(r['zone'])} | system {int(r['system_id'])} | "
                     f"{r.get('trace_km', np.nan):.1f} km | class {r.get('class', '?')}")
            try:
                fig, data = case_panels8(m8, r["task"], int(r["zone"]), int(r["system_id"]), title)
            except Exception as e:                                  # a figure never stops a section
                m8.say(f"case {name} skipped: {type(e).__name__}: {e}")
                continue
            m8.ws.save_fig(fig, name, None, sub)
            m8.ws.save_npz({k: np.asarray(v) for k, v in data.items()}, f"{sub}/{name}.npz")
            meta = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in r.items()}
            m8.ws.save_json(json.loads(json.dumps(meta, default=str)), f"{sub}/{name}.json")
            idx.append({**meta, "figure": f"{name}.png", "arrays": f"{name}.npz"})
    t = pd.DataFrame(idx)
    m8.ws.save_table(t, "case_study_index", sub)
    return t


def stage_funnel(by: pd.DataFrame):
    d = by[(by.stratum == "all") & (by.modality == "any")] if len(by) else by
    if not len(d):
        return None, d
    from ..m7.figures import line_panels16

    order = {s: i for i, s in enumerate(["raw", "gradient", "normalized", "directional", "crest", "fragment",
                                         "persistent", "linked", "ranked", "m8_worm", "m8_linked", "m8_ranked"])}
    d = d.assign(stage_i=d.stage.map(order)).dropna(subset=["stage_i"])
    m = d.melt(id_vars=["task", "stage", "stage_i"], value_vars=["system_recall", "length_recall"], var_name="series",
               value_name="recall")
    return line_panels16(m, "stage_i", "recall", "series", "task", xlabel="extraction stage (0 = raw ... 8 = ranked; "
                         "9-11 = M8 worms / linked / ranked)", ylabel="share of withheld systems",
                         title="Where does geological information disappear? (any modality)"), m


def profile_gallery(P: pd.DataFrame):
    if not len(P):
        return None
    from ..m7.figures import line_panels16

    return line_panels16(P, "offset_m", "mean", "group", "field", xlabel="offset from the trace (m)",
                         ylabel="median |F(+o) - F(-o)| / IQR", title="Fault-normal contrast by group (unlabelled "
                         "boundaries are NOT negatives)")


def recall_density(pc: pd.DataFrame):
    if not len(pc):
        return None, pc
    from ..m7.figures import line_panels16

    d = pc.melt(id_vars=["task", "pool", "density_cap"], value_vars=["system_recall", "short_system_recall", "recall_w"],
                var_name="metric", value_name="recall")
    d["series"] = d.pool + " " + d.metric
    return line_panels16(d, "density_cap", "recall", "series", "task", logx=True,
                         xlabel="candidate density cap (km / 100 km^2, label-free order)", ylabel="recall",
                         title="Candidate recall vs density: M8 pool vs M7 pool (oracle-style coverage)"), d


def occupancy_map(m8, occ: np.ndarray, covered: np.ndarray, title: str):
    import matplotlib.pyplot as plt

    rows, cols = np.nonzero(covered)
    if not len(rows):
        return None
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    ext = m8.ctx.data.grid.window_extent(r0, r1, c0, c1)
    a = np.where(covered[r0:r1, c0:c1], occ[r0:r1, c0:c1], np.nan)
    h, w = r1 - r0, c1 - c0
    fig, ax = plt.subplots(figsize=(10, 10 * h / max(w, 1) + 0.5))
    im = ax.imshow(a, extent=ext, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", origin="upper")
    lab = np.asarray(m8.ctx.data.labels, bool)[r0:r1, c0:c1]
    from ..m7.figures import KNOWN, _lines

    _lines(ax, lab, ext, KNOWN, label="mapped faults")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="share of realisations with a line within 100 m")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig
