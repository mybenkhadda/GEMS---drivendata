"""Section-level analyses and figures for the GEMS geological EDA notebook.

Every ``sN_*`` function either computes (and persists) the tables for section N
or renders (and saves) one figure. Figures use the shared zoom windows in
``ctx.windows`` so that panels across sections are directly comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from .archetypes import (ARCHETYPE_LABELS, ARCHETYPE_SIGNALS, ARCHETYPES, archetype_counts,
                         assign_archetypes, evidence_by_geometry, pick_examples)
from .candidates import LineamentDetector, candidate_table, detector_validation
from .config import (DETECTION_GROUPS, EVIDENCE_GROUPS, MODALITY_COLORS, PRIMARY_FIELD, SCALES_KM,
                     band_modality)
from .context import EDAContext, evidence_signal_groups
from .crossmodal import (alignment_at_stations, edge_agreement, evidence_combinations, fabric_roses,
                         near_far_masks, orientation_fields, pixel_auc, rose_correlation)
from .enhance import TRANSFORM_LABELS, TRANSFORMS
from .io import write_geotiff, write_lines
from .morphology import (component_diagnostics, endpoint_gaps, length_weighted_rose, regional_summary,
                         region_name, strike_families)
from .plotting import (DIVERGING, Window, add_lines, add_window_boxes, axial_rose, cmap_for, frame,
                       hillshade_colored, paths_xy, rgb_composite, show_raster)
from .profiles import (aggregate_profiles, aligned_cube, build_stations, extract_profiles, profile_metrics,
                       response_width_km, sample_background_stations, sample_fault_stations,
                       segment_evidence, signal_effects, station_evidence)

HIGHLIGHT = "#00e5ff"
STATUS_COLORS = {"mapped": "#7f8c8d", "near_mapped": "#f5b041", "unmapped": "#e74c3c"}


# ------------------------------------------------------------------ helpers
def _panel(ctx: EDAContext, width: float = 3.0) -> tuple[float, float]:
    g = ctx.grid
    return width, width * (g.height * g.pixel_km_y) / (g.width * g.pixel_km_x)


def signal(ctx: EDAContext, name: str) -> np.ndarray:
    if name == "@fused":
        return ctx.extra_maps["fused"]
    if name in ctx.signals:
        return ctx.signals[name]
    return ctx.extra_maps[name]


def signal_label(name: str) -> str:
    return "fused lineament map" if name == "@fused" else name.replace(":", " · ")


def _show_signal(ctx, ax, name, window=None, title=None, faults=True, lw=0.5):
    show_raster(ax, signal(ctx, name), ctx.grid, window, cmap=cmap_for(name) if name != "@fused" else "magma",
                symmetric=name in DIVERGING, title=title or signal_label(name))
    if faults:
        ctx.faults_on(ax, lw=lw, alpha=0.8)


def _rgb_axis(ctx, ax, channels, window=None, title=None):
    sub = [c[window.slices] if window is not None else c for c in channels]
    ext = window.extent(ctx.grid) if window is not None else ctx.grid.extent
    ax.imshow(rgb_composite(sub), extent=ext, interpolation="nearest")
    frame(ax, ctx.grid, ext)
    if title:
        ax.set_title(title)


RGB_RECIPES = {
    "magnetic_structure": ("Magnetic structure\nR TMI-HG | G tilt(RTP) | B TMI-VG", ("tmi_hg", "magnetic:tilt", "tmi_vg")),
    "gravity_basement": ("Gravity / basement\nR iso-gravity | G gravity-HG | B depth to basement",
                         ("iso_grav_anom", "iso_grav_anom_hg", "depth_to_base_surf")),
    "terrain_strain": ("Terrain / strain\nR det. elevation | G slope | B shear rate",
                       ("det_elev", "det_elev_slope", "geod_shearrate")),
    "crossmodal_edges": ("Cross-modal edges (white = coincident)\nR magnetic | G gravity | B terrain gradient",
                         ("tmi_hg", "iso_grav_anom_hg", "det_elev_slope")),
}


def _recipe(ctx, key):
    return [signal(ctx, n) for n in RGB_RECIPES[key][1]]


def _hillshade(ctx, color_name="rtp", cmap="RdBu_r"):
    key = f"_hs_{color_name}"
    if key not in ctx.extra_maps:
        ctx.extra_maps[key] = hillshade_colored(ctx.raw("det_elev"), signal(ctx, color_name), ctx.grid, cmap)
    return ctx.extra_maps[key]


def _show_hillshade(ctx, ax, window=None, color_name="rtp", cmap="RdBu_r", title=None):
    hs = _hillshade(ctx, color_name, cmap)
    sub = hs[window.slices] if window is not None else hs
    ext = window.extent(ctx.grid) if window is not None else ctx.grid.extent
    ax.imshow(sub, extent=ext, interpolation="nearest")
    frame(ax, ctx.grid, ext)
    ax.set_title(title or f"Relief shaded, coloured by {color_name}")


# ======================================================== 1. visual overview
def s1_channel_maps(ctx: EDAContext):
    names = ctx.stack.names
    ncols = 5
    nrows = int(np.ceil(len(names) / ncols))
    pw, ph = _panel(ctx, 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, (ph + 0.35) * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        _show_signal(ctx, ax, name, title=f"{name}  [{band_modality(name)}]", lw=0.25)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    add_window_boxes(axes.flat[0], ctx.windows, ctx.grid, color="k")
    fig.suptitle("All geophysical channels (robust-scaled) with mapped faults; boxes = shared zoom windows", fontsize=11)
    return ctx.save_fig(fig, "s1_channels_with_faults", "All 19 channels, full region, mapped-fault overlay")


def s1_rgb_composites(ctx: EDAContext):
    pw, ph = _panel(ctx, 4.2)
    fig, axes = plt.subplots(2, 3, figsize=(pw * 3, (ph + 0.5) * 2))
    for ax, key in zip(axes.flat[:4], RGB_RECIPES):
        _rgb_axis(ctx, ax, _recipe(ctx, key), title=RGB_RECIPES[key][0])
        ctx.faults_on(ax, lw=0.3, alpha=0.7)
    _show_hillshade(ctx, axes.flat[4], color_name="rtp", cmap="RdBu_r")
    add_window_boxes(axes.flat[4], ctx.windows, ctx.grid, color="k")
    _show_hillshade(ctx, axes.flat[5], color_name="iso_grav_anom", cmap="PuOr_r")
    ctx.faults_on(axes.flat[5], lw=0.3, alpha=0.7)
    fig.suptitle("Geophysical RGB composites and colour-shaded relief", fontsize=11)
    return ctx.save_fig(fig, "s1_rgb_composites", "Magnetic, gravity/basement, terrain/strain and cross-modal edge RGBs")


def s1_zoom_overview(ctx: EDAContext):
    cols = ["hillshade", "magnetic_structure", "gravity_basement", "crossmodal_edges"]
    fig, axes = plt.subplots(len(ctx.windows), 4, figsize=(13, 3.3 * len(ctx.windows)), squeeze=False)
    for i, w in enumerate(ctx.windows):
        for j, key in enumerate(cols):
            ax = axes[i, j]
            if key == "hillshade":
                _show_hillshade(ctx, ax, w, title=f"{w.name}: {w.label}\nrelief coloured by RTP")
            else:
                _rgb_axis(ctx, ax, _recipe(ctx, key), w, title=RGB_RECIPES[key][0] if i == 0 else None)
            ctx.faults_on(ax, color="k", lw=0.8)
    fig.suptitle("Zoom windows (same windows reused in all sections)", fontsize=11)
    return ctx.save_fig(fig, "s1_zoom_windows", "Four named zoom windows: relief + RGB composites + faults")


# ======================================================== 2. fault morphology
def s2_morphology(ctx: EDAContext) -> dict:
    cfg, g, net = ctx.cfg, ctx.grid, ctx.net
    comps = component_diagnostics(net, g, cfg.short_fault_km)
    gaps = endpoint_gaps(net, g, cfg.gap_km, cfg.gap_misalign_deg)
    regional = regional_summary(net, g, ctx.valid, cfg.region_grid)
    families = strike_families(net.segments)
    n_end = int((net.nodes.kind == "endpoint").sum())
    bridged = set(gaps.seg_a).union(gaps.seg_b) if len(gaps) else set()
    summary = {
        "n_components": int(len(comps)),
        "n_segments": int(len(net.segments)),
        "n_junctions": int((net.nodes.kind == "junction").sum()),
        "n_endpoints": n_end,
        "total_trace_km": float(net.segments.length_km.sum()),
        "median_segment_km": float(net.segments.length_km.median()),
        "p90_segment_km": float(net.segments.length_km.quantile(0.9)),
        "label_width_px_median": float(comps.mean_width_px.median()) if len(comps) else np.nan,
        "component_kinds": comps.kind.value_counts().to_dict() if len(comps) else {},
        "component_km_by_kind": comps.groupby("kind").length_km.sum().to_dict() if len(comps) else {},
        "bridgeable_gaps": int(len(gaps)),
        "endpoint_share_bridgeable": float(2 * len(gaps) / max(n_end, 1)),
        "segments_touching_bridgeable_gap": int(len(bridged)),
        "strike_families_deg": families,
    }
    ctx.save_dataset(net.segments, "known_fault_segments", "Skeleton segments of mapped faults with geometry")
    ctx.save_dataset(net.nodes, "known_fault_nodes", "Skeleton nodes (endpoints / junctions) with degree")
    p = write_lines(ctx.paths.datasets / "known_fault_segments.gpkg", net.paths, net.segments, g, "segments")
    ctx.manifest.add("known_fault_segments_gpkg", p, "vector", "Mapped fault segments as LineStrings")
    ctx.save_table(comps, "fault_components", "Raster connected components: kind (single/network/fragment/blob)")
    ctx.save_table(gaps, "fault_endpoint_gaps", "Collinear endpoint pairs across components (likely broken traces)")
    ctx.save_table(regional, "fault_regional_summary", "Per-region density, orientation, length, curvature")
    ctx.save_json(summary, "fault_morphology_summary", "Headline fault-label morphology numbers")
    return {"components": comps, "gaps": gaps, "regional": regional, "families": families, "summary": summary}


def s2_label_suitability(ctx: EDAContext, morph: dict):
    comps, gaps = morph["components"], morph["gaps"]
    w = ctx.window("W2")
    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6])
    ax = fig.add_subplot(gs[0, 0])
    k = comps.groupby("kind").agg(n=("component", "size"), km=("length_km", "sum"))
    ax.barh(k.index, k.n, color="#5d6d7e")
    ax.set_title("Raster components by kind (count)")
    ax = fig.add_subplot(gs[0, 1])
    ax.barh(k.index, k.km, color="#1f618d")
    ax.set_title("... by mapped trace length (km)")
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(comps.n_segments.clip(upper=30), bins=30, color="#7d6608", log=True)
    ax.set_title("Segments per component (clipped at 30)")
    ax.set_xlabel("segments")
    ax = fig.add_subplot(gs[1, 1])
    if len(gaps):
        ax.hist(gaps.gap_km, bins=25, color="#ca6f1e")
    ax.set_title(f"Collinear endpoint gaps ({len(gaps)} pairs)")
    ax.set_xlabel("gap (km)")
    ax = fig.add_subplot(gs[:, 2])
    ext = w.extent(ctx.grid)
    ax.imshow(np.where(ctx.net.mask[w.slices], 1.0, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=2,
              interpolation="nearest")
    rng = np.random.default_rng(0)
    seg_in = [i for i, p in enumerate(ctx.net.paths) if w.contains(p[:, 0], p[:, 1]).any()]
    lines = paths_xy([ctx.net.paths[i] for i in seg_in], ctx.grid)
    add_lines(ax, lines, color=plt.cm.tab20(rng.integers(0, 20, len(lines))), lw=1.2)
    nd = ctx.net.nodes[w.contains(ctx.net.nodes.r, ctx.net.nodes.c)]
    for kind, col, s in (("junction", "#e74c3c", 18), ("endpoint", "#2e86c1", 8)):
        sub = nd[nd.kind == kind]
        x, y = ctx.grid.rc_to_xy(sub.r, sub.c)
        ax.scatter(x, y, s=s, c=col, label=kind, zorder=5, edgecolors="none")
    if len(gaps):
        gw = gaps[w.contains(gaps.r_a, gaps.c_a)]
        for r in gw.itertuples():
            (xa, xb), (ya, yb) = ctx.grid.rc_to_xy([r.r_a, r.r_b], [r.c_a, r.c_b])
            ax.plot([xa, xb], [ya, yb], ls="--", color="#ca6f1e", lw=1.2, zorder=4)
    frame(ax, ctx.grid, ext)
    ax.legend(loc="lower right")
    ax.set_title(f"{w.name}: label mask (grey), traced segments (colours),\njunctions, endpoints, bridgeable gaps (dashed)")
    fig.suptitle("Are raster components usable as fault objects?", fontsize=11)
    return ctx.save_fig(fig, "s2_label_suitability", "Component kinds, fragmentation and traced graph in W2")


def s2_orientation(ctx: EDAContext, morph: dict):
    seg, regional = ctx.net.segments, morph["regional"]
    nr, nc = ctx.cfg.region_grid
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(nr, nc + 2)
    ax = fig.add_subplot(gs[:, :2], projection="polar")
    centers, h = length_weighted_rose(seg)
    axial_rose(ax, centers, h, color=MODALITY_COLORS["faults"])
    for f in morph["families"]:
        ax.plot([np.radians(f)] * 2, [0, h.max()], color="#e74c3c", lw=1.5)
        ax.plot([np.radians(f + 180)] * 2, [0, h.max()], color="#e74c3c", lw=1.5)
    ax.set_title("Length-weighted strike rose (all mapped faults)\nred = dominant families: "
                 + ", ".join(f"{f:.0f}°" for f in morph["families"]))
    for rid in range(nr * nc):
        ax = fig.add_subplot(gs[rid // nc, 2 + rid % nc], projection="polar")
        s = seg[seg.region == rid]
        if len(s) >= 5:
            c2, h2 = length_weighted_rose(s)
            axial_rose(ax, c2, h2, color="#1f618d", lw=0.8)
        row = regional.loc[regional.region == rid].iloc[0]
        ax.set_title(f"{region_name(rid, ctx.cfg.region_grid)}  n={len(s)}  "
                     f"{row.density_km_per_100km2:.1f} km/100km²", fontsize=7)
        ax.set_xticklabels([])
    fig.suptitle("Fault orientations, overall and by region", fontsize=11)
    return ctx.save_fig(fig, "s2_orientation_by_region", "Strike roses overall and per region")


def s2_geometry(ctx: EDAContext):
    seg, nodes = ctx.net.segments, ctx.net.nodes
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    L = np.sort(seg.length_km.to_numpy())
    ax = axes[0, 0]
    ax.loglog(L, 1 - np.arange(len(L)) / len(L), ".", ms=2, color="k")
    ax.set_xlabel("segment length (km)")
    ax.set_ylabel("P(length ≥ x)")
    ax.set_title("Length distribution (CCDF)")
    ax = axes[0, 1]
    ax.hist(seg.sinuosity.clip(upper=2.0).dropna(), bins=40, color="#7d6608")
    ax.set_title("Sinuosity (path / chord, clipped at 2)")
    ax = axes[0, 2]
    ax.hist(seg.curv_mean_deg_km.dropna().clip(upper=seg.curv_mean_deg_km.quantile(0.99)), bins=40, color="#6c3483")
    ax.set_title("Mean curvature (deg / km)")
    ax = axes[1, 0]
    hb = ax.hexbin(seg.azimuth, np.log10(seg.length_km), C=seg.length_km, reduce_C_function=np.sum,
                   gridsize=(36, 15), cmap="viridis", mincnt=1)
    plt.colorbar(hb, ax=ax, label="trace km")
    ax.set_xlabel("strike (deg)")
    ax.set_ylabel("log10 length (km)")
    ax.set_title("Strike vs length (trace-km weighted)")
    ax = axes[1, 1]
    dc = nodes[nodes.degree > 0].degree.clip(upper=6).value_counts().sort_index()
    ax.bar(dc.index.astype(str), dc.values, color="#2e86c1")
    ax.set_title("Node degree (1 = tip, ≥3 = branch point)")
    ax = axes[1, 2]
    ax.scatter(seg.length_km, seg.sinuosity, s=3, alpha=0.4, c=seg.n_junction_ends, cmap="coolwarm")
    ax.set_xscale("log")
    ax.set_ylim(0.95, 2)
    ax.set_xlabel("length (km)")
    ax.set_ylabel("sinuosity")
    ax.set_title("Curved faults are long or branched? (colour = junction ends)")
    fig.suptitle("Mapped-fault geometry", fontsize=11)
    return ctx.save_fig(fig, "s2_geometry_distributions", "Length CCDF, sinuosity, curvature, branching")


def s2_density(ctx: EDAContext, morph: dict):
    pw, ph = _panel(ctx, 4.2)
    fig, axes = plt.subplots(1, 3, figsize=(pw * 2 + 4, ph + 0.6), gridspec_kw={"width_ratios": [pw, pw, 4]})
    show_raster(axes[0], ctx.extra_maps["trace_density"], ctx.grid, cmap="viridis", pct=(1, 99.5), colorbar=True,
                title=f"Mapped trace density (km / 100 km², r = {ctx.cfg.density_radius_km:g} km)")
    show_raster(axes[1], ctx.extra_maps["junction_density"], ctx.grid, cmap="magma", pct=(1, 99.5),
                title="Junction (branch-point) density")
    ctx.faults_on(axes[1], color="w", lw=0.25, alpha=0.5)
    add_window_boxes(axes[0], ctx.windows, ctx.grid)
    reg = morph["regional"]
    nr, nc = ctx.cfg.region_grid
    mat = reg.density_km_per_100km2.to_numpy().reshape(nr, nc)
    im = axes[2].imshow(mat, cmap="viridis")
    for rid, r in reg.iterrows():
        axes[2].text(rid % nc, rid // nc, f"{r.region_name}\n{r.density_km_per_100km2:.1f}\n"
                     f"strike {r.mean_strike:.0f}° (R={r.strike_R:.2f})\nL50 {r.median_length_km:.1f} km",
                     ha="center", va="center", fontsize=6.5, color="w")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Regional summary (density, mean strike, median length)")
    plt.colorbar(im, ax=axes[2], shrink=0.7)
    return ctx.save_fig(fig, "s2_density_and_regions", "Trace/junction density maps and regional contrasts")


# ======================================================== 3. profiles
def s3_profiles(ctx: EDAContext, derived: list[str]) -> dict:
    cfg, g = ctx.cfg, ctx.grid
    core = max(cfg.core_km, 1.5 * g.pixel_km)
    fault_st = sample_fault_stations(ctx.net, g, ctx.valid, cfg.station_spacing_km, cfg.tangent_km,
                                     min_seg_km=max(2 * cfg.tangent_km, 3 * g.pixel_km), edge_px=2,
                                     region_grid=cfg.region_grid, rng=ctx.rng, max_stations=cfg.max_stations)
    bg_st = sample_background_stations(fault_st, ctx.net, g, ctx.valid, cfg.background_min_km, 2,
                                       cfg.region_grid, ctx.rng)
    st = build_stations(fault_st, bg_st)
    offsets = np.linspace(-cfg.profile_half_km, cfg.profile_half_km, 61)
    names = ctx.stack.names + derived
    ps = extract_profiles({k: ctx.signals[k] for k in names}, st, offsets)
    metrics = profile_metrics(ps, core)
    effects = signal_effects(metrics, band_modality)
    agg = aggregate_profiles(ps, core)
    effects = effects.merge(response_width_km(agg, effects), on="signal", how="left")
    evidence = station_evidence(metrics, evidence_signal_groups(names))
    seg_ev = segment_evidence(evidence, cfg.strong_evidence)

    ps.save(ctx.paths.datasets / "profile_stations.parquet", ctx.paths.datasets / "profiles_long.parquet")
    ctx.manifest.add("profile_stations", ctx.paths.datasets / "profile_stations.parquet", "dataset",
                     "Fault and matched-background stations (r, c, normal, strike, region)")
    ctx.manifest.add("profiles_long", ctx.paths.datasets / "profiles_long.parquet", "dataset",
                     "Profiles: one row per (station, signal), one column per offset km; reload with ProfileSet.load")
    ctx.save_dataset(metrics, "profile_metrics", "Step / core-anomaly metrics per station and signal")
    ctx.save_dataset(evidence, "station_evidence", "Calibrated per-station modality evidence (background ~ U(0,1))")
    ctx.save_dataset(seg_ev, "segment_evidence", "Median station evidence per mapped segment")
    ctx.save_table(effects, "profile_signal_effects", "Fault-vs-background AUC per signal + response FWHM")
    ctx.save_table(agg, "profile_aggregates", "Polarity-aligned median/IQR profiles, fault vs background")
    return {"ps": ps, "metrics": metrics, "effects": effects, "agg": agg, "evidence": evidence,
            "seg_ev": seg_ev, "core_km": core}


def s3_aggregate_profiles(ctx: EDAContext, prof: dict, n: int = 12):
    eff, agg = prof["effects"], prof["agg"]
    top = eff.head(n).signal.tolist()
    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.7 * nrows), squeeze=False, sharex=True)
    for ax, sig in zip(axes.flat, top):
        col = MODALITY_COLORS.get(band_modality(sig), "k")
        for kind, c in (("background", "#95a5a6"), ("fault", col)):
            a = agg[(agg.signal == sig) & (agg.kind == kind)]
            ax.fill_between(a.offset_km, a.q25, a.q75, color=c, alpha=0.25, lw=0)
            ax.plot(a.offset_km, a.q50, color=c, lw=1.6 if kind == "fault" else 1.0, label=kind)
        r = eff.set_index("signal").loc[sig]
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{signal_label(sig)}\nAUC {r.best_auc:.2f} ({r.best_metric}), FWHM {r.fwhm_km:.1f} km", fontsize=8)
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    axes.flat[0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("offset across fault (km)")
    fig.suptitle("Polarity-aligned fault-perpendicular profiles (median ± IQR) vs matched background", fontsize=11)
    return ctx.save_fig(fig, "s3_aggregate_profiles", "Aggregate profiles of the most discriminative signals")


def s3_signal_effects(ctx: EDAContext, prof: dict):
    eff = prof["effects"].sort_values("best_auc")
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(eff) + 1.2))
    y = np.arange(len(eff))
    ax.barh(y, eff.best_auc - 0.5, left=0.5, color=[MODALITY_COLORS.get(m, "k") for m in eff.modality])
    ax.scatter(eff.auc_abs_step, y, marker="|", color="k", s=40, label="|step| AUC")
    ax.scatter(eff.auc_abs_peak, y, marker="x", color="k", s=12, label="|core anomaly| AUC")
    ax.set_yticks(y)
    ax.set_yticklabels([signal_label(s) for s in eff.signal], fontsize=7)
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set_xlabel("AUC, fault vs matched background stations")
    ax.legend(loc="lower right")
    ax.set_title("Which signals change across mapped faults? (bar = best metric, colour = modality)")
    return ctx.save_fig(fig, "s3_signal_effects", "Per-signal fault-vs-background AUC")


def s3_profile_examples(ctx: EDAContext, prof: dict, n_signals: int = 4):
    ps, metrics, eff = prof["ps"], prof["metrics"], prof["effects"]
    sigs = []
    for sig in eff.signal:
        if band_modality(sig) not in [band_modality(s) for s in sigs]:
            sigs.append(sig)
        if len(sigs) == n_signals:
            break
    cube = aligned_cube(ps, prof["core_km"])
    fig, axes = plt.subplots(len(sigs), 2, figsize=(10, 3.0 * len(sigs)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.4]})
    half = ctx.grid.km_to_px(2.2 * ctx.cfg.profile_half_km) / 2
    for i, sig in enumerate(sigs):
        metric = eff.set_index("signal").loc[sig, "best_metric"]
        col = {"abs_step": "abs_step", "abs_peak": "abs_peak", "peak": "peak", "trough": "peak"}[metric]
        m = metrics[(metrics.signal == sig) & (metrics.kind == "fault")].dropna(subset=[col])
        vals = -m[col] if metric == "trough" else m[col]
        target = vals.quantile(0.9)                                     # representative strong case, not the max
        sid = int(m.loc[(vals - target).abs().idxmin(), "sample_id"])
        st = ps.stations.set_index("sample_id").loc[sid]
        w = Window.around("ex", "", st.r, st.c, half, ctx.grid)
        ax = axes[i, 0]
        _show_signal(ctx, ax, sig, w, title=f"{signal_label(sig)} – station {sid}")
        rr = st.r + ps.offsets_km * st.dr
        cc = st.c + ps.offsets_km * st.dc
        x, y = ctx.grid.rc_to_xy(rr, cc)
        ax.plot(x, y, color=HIGHLIGHT, lw=1.5)
        ax.plot(x[-1], y[-1], "o", color=HIGHLIGHT, ms=3)
        ax = axes[i, 1]
        j = ps.signals.index(sig)
        bg = cube[ps.kind("background"), j]
        ax.fill_between(ps.offsets_km, np.nanpercentile(bg, 10, 0), np.nanpercentile(bg, 90, 0), color="#bdc3c7",
                        alpha=0.5, label="background 10–90 %")
        ax.plot(ps.offsets_km, np.nanmedian(bg, 0), color="#7f8c8d", lw=1)
        ax.plot(ps.offsets_km, cube[sid, j], color=MODALITY_COLORS.get(band_modality(sig), "k"), lw=1.8,
                label="this fault station (aligned)")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("offset (km)")
        ax.legend(fontsize=6)
    fig.suptitle("Representative fault profiles (90th-percentile contrast) vs background envelope", fontsize=11)
    return ctx.save_fig(fig, "s3_profile_examples", "Representative profiles with map location")


# ======================================================== 4. enhancement
def s4_transform_scale_auc(ctx: EDAContext, families: list[float], modalities=DETECTION_GROUPS,
                           scales=SCALES_KM, transforms=TRANSFORMS) -> dict:
    near, far = near_far_masks(ctx.net, ctx.grid, ctx.valid, ctx.cfg.near_tol_px, ctx.cfg.far_km)
    strike = families[0] if families else 0.0
    rows = []
    for m in modalities:
        a = ctx.primary(m)
        for t in transforms:
            for s in scales:
                resp = ctx.enh.transform(t, a, s, strike)
                rows.append({"modality": m, "field": PRIMARY_FIELD[m], "transform": t, "scale_km": s,
                             "sigma_px": ctx.enh.sigma_px(s), "pixel_auc": pixel_auc(resp, near, far, ctx.rng)})
    auc = pd.DataFrame(rows)
    raw = []
    for name in ctx.stack.names:
        v = ctx.signals[name]
        a_pos = pixel_auc(v, near, far, ctx.rng)
        raw.append({"band": name, "modality": band_modality(name), "auc_high": a_pos,
                    "auc_abs": pixel_auc(np.abs(v), near, far, ctx.rng), "auc_best": max(a_pos, 1 - a_pos)})
    raw = pd.DataFrame(raw).sort_values("auc_best", ascending=False)
    best = auc.loc[auc.groupby("modality").pixel_auc.idxmax()].sort_values("pixel_auc", ascending=False)
    ctx.save_table(auc, "transform_scale_pixel_auc", "Pixel AUC (near-fault vs far) per field x transform x scale")
    ctx.save_table(raw, "raw_band_pixel_auc", "Pixel AUC of each raw band (either polarity)")
    ctx.save_table(best, "best_transform_per_modality", "Best transform/scale per modality")
    return {"auc": auc, "raw": raw, "best": best, "strike": strike}


def s4_save_enhanced(ctx: EDAContext, enh_res: dict):
    layers = {}
    for r in enh_res["best"].itertuples():
        key = f"{r.modality}:{r.transform}@{r.scale_km:g}km"
        layers[key] = ctx.enh.transform(r.transform, ctx.primary(r.modality), r.scale_km, enh_res["strike"])
        ctx.extra_maps[f"best:{r.modality}"] = layers[key]
        ctx.extra_maps[f"best_name:{r.modality}"] = key
    for s in (2.0, 10.0):
        layers[f"magnetic:tilt@{s:g}km"] = ctx.enh.tilt(ctx.primary("magnetic"), s)
        layers[f"gravity:tilt@{s:g}km"] = ctx.enh.tilt(ctx.primary("gravity"), s)
    p = write_geotiff(ctx.paths.features / "enhanced_best.tif", layers, ctx.grid)
    ctx.manifest.add("enhanced_best", p, "raster", "Best transform per modality + tilt at 2/10 km (band descriptions)")
    p2 = write_geotiff(ctx.paths.features / "label_context.tif",
                       {"fault_distance_km": ctx.net.distance_km, "trace_density": ctx.extra_maps["trace_density"]},
                       ctx.grid)
    ctx.manifest.add("label_context", p2, "raster", "Label-derived layers (EDA only – NOT model inputs: leakage)")
    return p


def s4_auc_heatmaps(ctx: EDAContext, enh_res: dict):
    auc = enh_res["auc"]
    mods = list(dict.fromkeys(auc.modality))
    fig, axes = plt.subplots(1, len(mods), figsize=(3.0 * len(mods) + 1, 3.8), sharey=True)
    vmax = max(0.6, auc.pixel_auc.max())
    for ax, m in zip(np.atleast_1d(axes), mods):
        mat = auc[auc.modality == m].pivot(index="transform", columns="scale_km", values="pixel_auc").reindex(TRANSFORMS)
        im = ax.imshow(mat.to_numpy(), cmap="RdYlGn", vmin=1 - vmax, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(mat.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in mat.columns])
        ax.set_xlabel("scale (km)")
        ax.set_title(f"{m} ({PRIMARY_FIELD[m]})")
    np.atleast_1d(axes)[0].set_yticks(range(len(TRANSFORMS)))
    np.atleast_1d(axes)[0].set_yticklabels([TRANSFORM_LABELS[t] for t in TRANSFORMS], fontsize=7)
    plt.colorbar(im, ax=axes, shrink=0.8, label="pixel AUC (near vs far)")
    fig.suptitle("Fault sensitivity of each transformation across physical scales", fontsize=11)
    return ctx.save_fig(fig, "s4_transform_scale_auc", "Pixel AUC heatmaps: transform x scale per modality")


def _window_transform(ctx: EDAContext, t: str, a: np.ndarray, scale_km: float, strike: float, w: Window):
    """Transform computed on a padded crop around ``w`` (pasted into a full-size NaN array)."""
    pad = int(ctx.grid.km_to_px(2.5 * scale_km)) + 8
    r0, r1 = max(w.r0 - pad, 0), min(w.r1 + pad, ctx.grid.height)
    c0, c1 = max(w.c0 - pad, 0), min(w.c1 + pad, ctx.grid.width)
    out = np.full(ctx.grid.shape, np.nan, np.float32)
    out[r0:r1, c0:c1] = ctx.enh.transform(t, a[r0:r1, c0:c1], scale_km, strike)
    return out


def s4_transform_gallery(ctx: EDAContext, enh_res: dict, modality: str = "magnetic", window: str = "W1"):
    w = ctx.window(window)
    a = ctx.primary(modality)
    fig, axes = plt.subplots(len(TRANSFORMS), len(SCALES_KM), figsize=(2.7 * len(SCALES_KM), 2.6 * len(TRANSFORMS)))
    for i, t in enumerate(TRANSFORMS):
        for j, s in enumerate(SCALES_KM):
            ax = axes[i, j]
            resp = _window_transform(ctx, t, a, s, enh_res["strike"], w)
            show_raster(ax, resp, ctx.grid, w, cmap="magma", pct=(2, 99.5))
            ctx.faults_on(ax, color=HIGHLIGHT, lw=0.5, alpha=0.7)
            if i == 0:
                ax.set_title(f"{s:g} km", fontsize=9)
            if j == 0:
                ax.set_ylabel(TRANSFORM_LABELS[t], fontsize=7)
    fig.suptitle(f"{PRIMARY_FIELD[modality]}: transformations (rows) x physical scale (columns), window {w.name}",
                 fontsize=11)
    return ctx.save_fig(fig, f"s4_gallery_{modality}_{window}", "Transform x scale gallery in a zoom window")


def s4_fused_lineaments(ctx: EDAContext):
    cfg = ctx.cfg
    det = LineamentDetector(ctx.enh, cfg.candidate_scales_km, cfg.candidate_high_q, cfg.candidate_low_q,
                            cfg.candidate_min_km)
    det.fit({m: ctx.primary(m) for m in DETECTION_GROUPS}, ctx.valid)
    maps = det.fused(ctx.valid)
    ctx.extra_maps.update({"fused": maps["fused"], "fused_score": maps["score"], "fused_support": maps["support"]})
    layers = {**{f"fused_{k}": v for k, v in maps.items()}, **{f"response_{k}": v for k, v in det.responses.items()}}
    p = write_geotiff(ctx.paths.features / "lineament_maps.tif", layers, ctx.grid)
    ctx.manifest.add("lineament_maps", p, "raster", "Per-modality ridge responses + fused/support/coherence/score")
    near, far = near_far_masks(ctx.net, ctx.grid, ctx.valid, ctx.cfg.near_tol_px, ctx.cfg.far_km)
    aucs = {k: pixel_auc(v, near, far, ctx.rng) for k, v in layers.items()}
    ctx.save_table(pd.DataFrame({"layer": list(aucs), "pixel_auc": list(aucs.values())}), "lineament_layer_auc",
                   "Pixel AUC of fused lineament layers")
    return det, maps


# ======================================================== 5. cross-modal
def s5_crossmodal(ctx: EDAContext, prof: dict, scale_km: float = 5.0) -> dict:
    mods = ("magnetic", "gravity", "terrain", "conductivity", "basement", "strain")
    fields = {m: ctx.primary(m) for m in mods}
    orient = orientation_fields(fields, ctx.enh, scale_km)
    roses = fabric_roses(orient, ctx.valid, ctx.rng)
    c, h = length_weighted_rose(ctx.net.segments)
    roses = pd.concat([roses, pd.DataFrame({"field": "mapped_faults", "strike_bin": c, "density": h})], ignore_index=True)
    corr = rose_correlation(roses)
    align = alignment_at_stations(orient, prof["ps"].stations)
    edges = {m: ctx.enh.thdr(fields[m], scale_km) for m in ("magnetic", "gravity", "terrain", "conductivity", "basement")}
    binary, count = edge_agreement(edges, ctx.valid)
    near, far = near_far_masks(ctx.net, ctx.grid, ctx.valid, ctx.cfg.near_tol_px, ctx.cfg.far_km)
    lift = []
    for k in range(len(edges) + 1):
        pn, pf = (count[near] == k).mean(), (count[far] == k).mean()
        lift.append({"n_modalities_with_edge": k, "share_near_fault": pn, "share_far": pf,
                     "lift": pn / pf if pf else np.nan})
    lift = pd.DataFrame(lift)
    combos, jac, uniq = evidence_combinations(prof["evidence"], EVIDENCE_GROUPS, ctx.cfg.strong_evidence)
    align_summary = (align[align.coherence >= 0.3].groupby(["field", "kind"]).cos2
                     .agg(["mean", "count", "std"]).reset_index())
    align_summary["se"] = align_summary["std"] / np.sqrt(align_summary["count"])
    ctx.extra_maps["edge_agreement"] = count.astype(np.float32)
    ctx.save_table(roses, "fabric_roses", "Coherence-weighted fabric orientation histograms per modality")
    ctx.save_table(corr.reset_index(), "fabric_rose_correlation", "Correlation of orientation histograms")
    ctx.save_table(align_summary, "fabric_alignment_at_stations", "Mean cos(2Δstrike) modality fabric vs fault")
    ctx.save_table(lift, "edge_agreement_lift", "Near-fault enrichment by number of modalities with a strong edge")
    ctx.save_table(combos, "evidence_combinations", "UpSet-style modality support combinations at fault stations")
    ctx.save_table(jac.reset_index(), "evidence_jaccard", "Jaccard overlap of modality support")
    ctx.save_table(uniq, "evidence_unique_share", "Share of stations supported by each modality / uniquely")
    p = write_geotiff(ctx.paths.features / "crossmodal_orientation.tif",
                      {**{f"strike_{m}": o["strike"] for m, o in orient.items()},
                       **{f"coherence_{m}": o["coherence"] for m, o in orient.items()},
                       "edge_agreement_count": count.astype(np.float32)}, ctx.grid)
    ctx.manifest.add("crossmodal_orientation", p, "raster", "Structure-tensor strike/coherence per modality + edge agreement")
    return {"orient": orient, "roses": roses, "corr": corr, "align": align, "align_summary": align_summary,
            "count": count, "lift": lift, "combos": combos, "jaccard": jac, "unique": uniq}


def s5_fabric_figure(ctx: EDAContext, cm: dict):
    fields = [f for f in dict.fromkeys(cm["roses"].field) if f != "mapped_faults"]
    fr = cm["roses"][cm["roses"].field == "mapped_faults"]
    fig = plt.figure(figsize=(3.0 * len(fields), 6.2))
    gs = fig.add_gridspec(2, len(fields), height_ratios=[1, 1.15])
    for i, f in enumerate(fields):
        ax = fig.add_subplot(gs[0, i], projection="polar")
        r = cm["roses"][cm["roses"].field == f]
        axial_rose(ax, r.strike_bin.to_numpy(), r.density.to_numpy(), color=MODALITY_COLORS.get(f, "k"))
        axial_rose(ax, fr.strike_bin.to_numpy(), fr.density.to_numpy(), color="k", fill=False, lw=0.8)
        ax.set_title(f"{f}\n(r = {cm['corr'].loc[f, 'mapped_faults']:.2f} with faults)", fontsize=8)
    ax = fig.add_subplot(gs[1, : max(len(fields) // 2, 1)])
    corr = cm["corr"]
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Orientation-histogram correlation")
    ax = fig.add_subplot(gs[1, max(len(fields) // 2, 1):])
    s = cm["align_summary"]
    piv = s.pivot(index="field", columns="kind", values="mean")
    se = s.pivot(index="field", columns="kind", values="se")
    x = np.arange(len(piv))
    for k, (kind, col) in enumerate((("fault", "#c0392b"), ("background", "#7f8c8d"))):
        if kind in piv:
            ax.bar(x + (k - 0.5) * 0.38, piv[kind], 0.38, yerr=1.96 * se[kind], color=col, label=kind)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=30)
    ax.set_ylabel("mean cos(2Δ strike)")
    ax.legend()
    ax.set_title("Local fabric parallel to the mapped/matched strike? (+1 parallel, 0 random)")
    fig.suptitle("Shared structural orientations across modalities (black outline = mapped faults)", fontsize=11)
    return ctx.save_fig(fig, "s5_fabric_orientations", "Fabric roses, rose correlation, local alignment")


def s5_edge_agreement_figure(ctx: EDAContext, cm: dict):
    count = cm["count"].astype(float)
    count[count < 0] = np.nan
    n = int(np.nanmax(count))
    cmap = ListedColormap(plt.cm.YlOrRd(np.linspace(0.05, 1, n + 1)))
    norm = BoundaryNorm(np.arange(-0.5, n + 1.5), cmap.N)
    pw, ph = _panel(ctx, 4.0)
    fig, axes = plt.subplots(1, 4, figsize=(pw + 3.2 * 2 + 4, max(ph, 3.4) + 0.6),
                             gridspec_kw={"width_ratios": [pw, 3.2, 3.2, 3.6]})
    for ax, w in zip(axes[:3], [None, ctx.window("W1"), ctx.window("W4")]):
        show_raster(ax, count, ctx.grid, w, cmap=cmap, vmin=-0.5, vmax=n + 0.5)
        ax.images[-1].set_norm(norm)
        ctx.faults_on(ax, color="#1f618d", lw=0.35 if w is None else 0.8)
        ax.set_title("Modalities with a top-10 % edge" if w is None else f"{w.name}: {w.label}")
    add_window_boxes(axes[0], ctx.windows, ctx.grid, color="k")
    lift = cm["lift"]
    axes[3].bar(lift.n_modalities_with_edge, lift.lift, color=cmap(norm(lift.n_modalities_with_edge)))
    axes[3].axhline(1, color="k", lw=0.6)
    axes[3].set_xlabel("number of modalities with a strong edge")
    axes[3].set_ylabel("near-fault / far-from-fault share")
    axes[3].set_title("Enrichment of mapped faults")
    fig.suptitle("Do strong edges coincide across modalities, and do mapped faults sit on them?", fontsize=11)
    return ctx.save_fig(fig, "s5_edge_agreement", "Cross-modal edge-coincidence map and enrichment")


def s5_complementarity_figure(ctx: EDAContext, cm: dict):
    combos, jac, uniq = cm["combos"], cm["jaccard"], cm["unique"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    top = combos.head(15).iloc[::-1]
    axes[0].barh(top.combination, top.share, color="#5d6d7e")
    axes[0].set_xlabel("share of fault stations")
    axes[0].set_title(f"Supporting modality combinations (evidence ≥ {ctx.cfg.strong_evidence:g})")
    im = axes[1].imshow(jac.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(jac)))
    axes[1].set_xticklabels(jac.columns, rotation=45, ha="right")
    axes[1].set_yticks(range(len(jac)))
    axes[1].set_yticklabels(jac.index)
    for (i, j), v in np.ndenumerate(jac.to_numpy(dtype=float)):
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    axes[1].set_title("Jaccard overlap of support")
    plt.colorbar(im, ax=axes[1], shrink=0.7)
    u = uniq.set_index("group")
    axes[2].bar(u.index, u.support_share, color=[MODALITY_COLORS.get(g, "#bbb") for g in u.index], label="supports")
    axes[2].bar(u.index, u.unique_share.fillna(0), color="k", alpha=0.45, label="only modality")
    axes[2].tick_params(axis="x", rotation=40)
    axes[2].set_title("Support vs unique contribution")
    axes[2].legend()
    fig.suptitle("Complementary fault evidence (10 % false-alarm rate per modality on background)", fontsize=11)
    return ctx.save_fig(fig, "s5_complementarity", "UpSet-style combinations, Jaccard, unique share")


# ======================================================== 6. archetypes
def s6_archetypes(ctx: EDAContext, prof: dict) -> dict:
    cfg = ctx.cfg
    arch = assign_archetypes(ctx.net.segments, prof["seg_ev"], cfg.strong_evidence, cfg.weak_evidence,
                             cfg.short_fault_km)
    counts = archetype_counts(arch)
    geom = evidence_by_geometry(arch)
    sep = ctx.grid.km_to_px(2 * ctx.cfg.profile_half_km)
    picks = {a: pick_examples(arch, a, 3, 2 * cfg.short_fault_km, sep) for a in ARCHETYPES}
    parts = [p.assign(archetype=a) for a, p in picks.items() if len(p)]
    ex = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    ctx.save_dataset(arch, "segment_archetypes", "Segments with evidence + archetype scores/labels")
    ctx.save_table(counts, "archetype_counts", "Primary evidence archetype counts and trace-km share")
    ctx.save_table(geom, "evidence_by_geometry", "Median modality evidence by geometric class")
    if len(ex):
        ctx.save_table(ex[["archetype", "seg_id", "length_km", "r_mid", "c_mid", "x_mid", "y_mid"]
                          + [c for c in ex.columns if c.startswith("ev_")]], "archetype_examples",
                       "Segments chosen as archetype gallery examples")
    return {"arch": arch, "counts": counts, "geometry": geom, "picks": picks}


def _example_window(ctx, row, name) -> Window:
    path = ctx.net.paths[int(row.seg_id)]
    r, c = row.r_mid, row.c_mid
    if name == "intersection":
        a, b = ctx.net.graph.seg_nodes[int(row.seg_id)]
        nodes = ctx.net.nodes.set_index("node_id")
        for nid in (a, b):
            if nid >= 0 and nodes.loc[nid, "degree"] >= 3:
                r, c = nodes.loc[nid, "r"], nodes.loc[nid, "c"]
                break
    span = max(np.ptp(path[:, 0]), np.ptp(path[:, 1]))
    half = max(0.65 * span, ctx.grid.km_to_px(12.0))
    if name == "intersection":
        half = ctx.grid.km_to_px(15.0)
    return Window.around(f"seg{int(row.seg_id)}", name, r, c, half, ctx.grid)


def _example_row(ctx, axes_row, row, name, show_titles):
    w = _example_window(ctx, row, name)
    hl = paths_xy([ctx.net.paths[int(row.seg_id)]], ctx.grid)
    for ax, sig in zip(axes_row, ARCHETYPE_SIGNALS[name]):
        _show_signal(ctx, ax, sig, w, title=signal_label(sig) if show_titles else "", lw=0.7)
        add_lines(ax, hl, color=HIGHLIGHT, lw=1.6, zorder=4)
    ev = " ".join(f"{c[3:6]}={row[c]:.2f}" for c in row.index if c.startswith("ev_") and np.isfinite(row[c]))
    axes_row[0].set_ylabel(f"seg {int(row.seg_id)} · {row.length_km:.0f} km\n{ev}", fontsize=6.5)


def s6_archetype_figures(ctx: EDAContext, arc: dict) -> list:
    out = []
    for name in ARCHETYPES:
        p = arc["picks"][name]
        if not len(p):
            continue
        fig, axes = plt.subplots(len(p), 4, figsize=(12, 3.0 * len(p)), squeeze=False)
        for i, (_, row) in enumerate(p.iterrows()):
            _example_row(ctx, axes[i], row, name, show_titles=True)
        fig.suptitle(f"{ARCHETYPE_LABELS[name]} – highlighted segment (cyan), other mapped faults (black)", fontsize=11)
        out.append(ctx.save_fig(fig, f"s6_archetype_{name}", f"Gallery: {ARCHETYPE_LABELS[name]}"))
    return out


def s6_archetype_gallery(ctx: EDAContext, arc: dict):
    names = [a for a in ARCHETYPES if len(arc["picks"][a])]
    fig, axes = plt.subplots(len(names), 4, figsize=(12, 2.9 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        row = arc["picks"][name].iloc[0]
        _example_row(ctx, axes[i], row, name, show_titles=True)
        axes[i, 0].set_title(f"{ARCHETYPE_LABELS[name]}\n{signal_label(ARCHETYPE_SIGNALS[name][0])}", fontsize=8)
    fig.suptitle("Fault archetype gallery (≤ 4 signals per example)", fontsize=11)
    return ctx.save_fig(fig, "s6_archetype_gallery", "One example per archetype")


def s6_archetype_summary(ctx: EDAContext, arc: dict):
    counts, geom = arc["counts"], arc["geometry"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(counts.primary_archetype, counts.share_km, color="#5d6d7e")
    axes[0].set_xlabel("share of mapped trace length")
    axes[0].set_title("Primary evidence archetype")
    ev_cols = [c for c in geom.columns if c.startswith("ev_")]
    g = geom.set_index("geometry_class")[ev_cols]
    im = axes[1].imshow(g.to_numpy(float), cmap="viridis", vmin=0.4, vmax=0.95, aspect="auto")
    axes[1].set_yticks(range(len(g)))
    axes[1].set_yticklabels(g.index)
    axes[1].set_xticks(range(len(ev_cols)))
    axes[1].set_xticklabels([c[3:] for c in ev_cols], rotation=30)
    for (i, j), v in np.ndenumerate(g.to_numpy(float)):
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
    plt.colorbar(im, ax=axes[1], label="median calibrated evidence (0.5 = background)")
    axes[1].set_title("Is evidence weaker for curved / branched / short faults?")
    return ctx.save_fig(fig, "s6_archetype_summary", "Archetype shares and evidence by geometry class")


# ======================================================== 7. candidates
def s7_candidates(ctx: EDAContext, det: LineamentDetector, maps: dict) -> dict:
    cfg = ctx.cfg
    skel, graph = det.detect(maps, ctx.valid, ctx.grid)
    cands, cpaths = candidate_table(graph, ctx.grid, maps, det.responses, ctx.net,
                                    length_weighted_rose(ctx.net.segments), cfg.candidate_min_km,
                                    cfg.candidate_match_km, det.support_q)
    val = detector_validation(cands, cpaths, ctx.net, ctx.grid, ctx.valid, cfg.candidate_match_km)
    if len(cands):
        cands["rank_unmapped"] = np.nan
        um = cands.status == "unmapped"
        cands.loc[um, "rank_unmapped"] = cands.loc[um, "rank_score"].rank(ascending=False, method="first")
        ctx.save_dataset(cands, "lineament_candidates", "All detected lineaments with scores and status")
        p = write_lines(ctx.paths.datasets / "lineament_candidates.gpkg", cpaths, cands, ctx.grid, "candidates")
        ctx.manifest.add("lineament_candidates_gpkg", p, "vector", "Detected lineaments (exploratory, NOT confirmed faults)")
    ctx.save_json(val, "lineament_detector_validation", "Recall of mapped faults, precision proxy, lift vs random")
    return {"cands": cands, "paths": cpaths, "validation": val, "skeleton": skel}


def s7_candidate_map(ctx: EDAContext, cres: dict, n_label: int = 15):
    cands, paths = cres["cands"], cres["paths"]
    pw, ph = _panel(ctx, 7.5)
    fig, ax = plt.subplots(figsize=(pw + 1.5, ph + 0.6))
    show_raster(ax, ctx.extra_maps["fused_score"], ctx.grid, cmap="Greys", pct=(50, 99.7), colorbar=True)
    ctx.faults_on(ax, color="#1f618d", lw=0.5)
    for status, col in STATUS_COLORS.items():
        idx = cands.index[cands.status == status]
        add_lines(ax, paths_xy([paths[i] for i in idx], ctx.grid), color=col,
                  lw=1.4 if status == "unmapped" else 0.7, label=f"{status} ({len(idx)})", zorder=4)
    top = cands[cands.status == "unmapped"].nsmallest(n_label, "rank_unmapped")
    for r in top.itertuples():
        ax.text(r.x_mid, r.y_mid, f" #{int(r.rank_unmapped)}", color="#e74c3c", fontsize=7, fontweight="bold", zorder=6)
    add_window_boxes(ax, ctx.windows, ctx.grid, color="k")
    ax.plot([], [], color="#1f618d", label="mapped faults")
    ax.legend(loc="lower left", fontsize=7)
    v = cres["validation"]
    ax.set_title(f"Exploratory lineaments on fused score – recall of mapped faults {v['recall_known_within_km']:.2f}, "
                 f"precision proxy {v['candidate_precision_proxy']:.2f} (lift {v['lift']:.1f}x)\n"
                 "Unmapped candidates are hypotheses, not confirmed faults")
    return ctx.save_fig(fig, "s7_candidate_map", "Full-region candidate lineament map with status and ranks")


def s7_candidate_top(ctx: EDAContext, cres: dict, n: int = 6):
    cands, paths = cres["cands"], cres["paths"]
    top = cands[cands.status == "unmapped"].nsmallest(n, "rank_unmapped")
    sigs = ("rtp", "iso_grav_anom_hg", "det_elev", "@fused")
    fig, axes = plt.subplots(len(top), 4, figsize=(12, 3.0 * max(len(top), 1)), squeeze=False)
    for i, r in enumerate(top.itertuples()):
        path = paths[r.Index]
        half = max(0.65 * max(np.ptp(path[:, 0]), np.ptp(path[:, 1])), ctx.grid.km_to_px(12))
        w = Window.around(f"cand{r.cand_id}", "", r.r_mid, r.c_mid, half, ctx.grid)
        for ax, sig in zip(axes[i], sigs):
            _show_signal(ctx, ax, sig, w, title=signal_label(sig) if i == 0 else "", lw=0.7)
            add_lines(ax, paths_xy([path], ctx.grid), color="#e74c3c", lw=1.6, zorder=4)
        axes[i, 0].set_ylabel(f"#{int(r.rank_unmapped)} · {r.length_km:.0f} km · strike {r.azimuth:.0f}°\n"
                              f"struct {r.structural_score:.2f} · evid {r.evidence_score:.2f} · "
                              f"{int(r.n_supporting_modalities)} modalities", fontsize=6.5)
    fig.suptitle("Top-ranked unmapped lineaments (red) – exploratory, require geological verification", fontsize=11)
    return ctx.save_fig(fig, "s7_candidate_top", "Zooms of the highest-ranked unmapped candidates")


def s7_candidate_scores(ctx: EDAContext, cres: dict):
    c = cres["cands"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for status, col in STATUS_COLORS.items():
        s = c[c.status == status]
        axes[0].scatter(s.structural_score, s.evidence_score, s=8 + 4 * s.length_km.clip(upper=30), c=col,
                        alpha=0.6, label=status, edgecolors="none")
    axes[0].set_xlabel("structural coherence score")
    axes[0].set_ylabel("geophysical evidence score")
    axes[0].legend()
    axes[0].set_title("Candidate scores (size = length)")
    cs = [k for k in c.columns if k.startswith("supp_")]
    s = c.groupby("status")[cs].mean()
    s.columns = [k[5:] for k in cs]
    s.T.plot.bar(ax=axes[1], color=[STATUS_COLORS[k] for k in s.index])
    axes[1].set_ylabel("mean fraction of trace with strong response")
    axes[1].set_title("Which modality supports which lineaments?")
    return ctx.save_fig(fig, "s7_candidate_scores", "Candidate structural vs evidence scores; modality support")


def s7_abcd(ctx: EDAContext, cres: dict, window: Window | None, modalities=("magnetic", "gravity", "terrain")):
    """A raw | B raw + faults | C enhanced | D candidate structure map, one row per modality."""
    cands, paths = cres["cands"], cres["paths"]
    if window is None:
        pw, ph = _panel(ctx, 3.3)
        size = (pw * 4, (ph + 0.4) * len(modalities))
    else:
        size = (13, 3.3 * len(modalities))
    fig, axes = plt.subplots(len(modalities), 4, figsize=size, squeeze=False)
    clines = {s: paths_xy([paths[i] for i in cands.index[cands.status == s]], ctx.grid) for s in STATUS_COLORS}
    lw = 0.35 if window is None else 0.9
    for i, m in enumerate(modalities):
        name = PRIMARY_FIELD[m]
        _show_signal(ctx, axes[i, 0], name, window, title=f"A · {name} (raw)", faults=False)
        _show_signal(ctx, axes[i, 1], name, window, title=f"B · {name} + mapped faults", lw=lw)
        best = ctx.extra_maps.get(f"best:{m}")
        show_raster(axes[i, 2], best, ctx.grid, window, cmap="magma", pct=(2, 99.5),
                    title=f"C · {ctx.extra_maps.get(f'best_name:{m}', 'enhanced')}")
        show_raster(axes[i, 3], ctx.extra_maps["fused_score"], ctx.grid, window, cmap="Greys", pct=(50, 99.7),
                    title="D · fused lineament score + traces")
        ctx.faults_on(axes[i, 3], color="#1f618d", lw=lw)
        for s, col in STATUS_COLORS.items():
            add_lines(axes[i, 3], clines[s], color=col, lw=lw * 2 if s == "unmapped" else lw, zorder=4)
    where = "full region" if window is None else f"{window.name}: {window.label}"
    fig.suptitle(f"Raw → labelled → enhanced → candidate structures ({where}); D: blue mapped, red unmapped, "
                 "orange near-mapped, grey re-detected mapped", fontsize=10)
    tag = "full" if window is None else window.name
    return ctx.save_fig(fig, f"s7_abcd_{tag}", f"A/B/C/D comparison panel ({where})")
