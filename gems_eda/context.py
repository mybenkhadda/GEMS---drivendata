"""Shared EDA state: data on one grid, fault network, standardised signals, zoom windows."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from scipy import ndimage

from .config import MODALITIES, PRIMARY_FIELD, EDAConfig, EDAPaths, Manifest, find_project_root
from .enhance import Enhancer, standardize_signal
from .io import RasterStack, load_feature_stack, load_on_grid
from .morphology import FaultNetwork, build_fault_network, trace_density
from .plotting import Window, add_lines, paths_xy, pick_windows, savefig, windows_table
from .stats import rank_normalize


@dataclass
class EDAContext:
    cfg: EDAConfig
    paths: EDAPaths
    manifest: Manifest
    stack: RasterStack
    faults: np.ndarray             # raw label values on the working grid
    fault_info: dict
    submission: np.ndarray
    submission_info: dict
    net: FaultNetwork
    enh: Enhancer
    rng: np.random.Generator
    signals: dict[str, np.ndarray] = field(default_factory=dict)   # standardised raw + derived
    signal_notes: dict[str, str] = field(default_factory=dict)
    windows: list[Window] = field(default_factory=list)
    extra_maps: dict[str, np.ndarray] = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    @property
    def grid(self):
        return self.stack.grid

    @property
    def valid(self) -> np.ndarray:
        return self.stack.valid

    @property
    def fault_lines(self) -> list[np.ndarray]:
        if "_fault_lines" not in self.extra_maps:
            self.extra_maps["_fault_lines"] = paths_xy(self.net.paths, self.grid)
        return self.extra_maps["_fault_lines"]

    def raw(self, name: str) -> np.ndarray:
        return self.stack[name]

    def primary(self, modality: str) -> np.ndarray:
        return self.stack[PRIMARY_FIELD[modality]]

    def faults_on(self, ax, color="k", lw=0.45, alpha=0.85, label="mapped faults"):
        return add_lines(ax, self.fault_lines, color=color, lw=lw, alpha=alpha, label=label)

    def window(self, name: str) -> Window:
        return next(w for w in self.windows if w.name == name)

    # -- persistence ----------------------------------------------------
    def save_fig(self, fig, name: str, description: str = "", show: bool = True) -> Path:
        return savefig(fig, self.paths.figures / f"{name}.png", self.manifest, name, description, show)

    def save_table(self, df: pd.DataFrame, name: str, description: str = "", index: bool = False) -> Path:
        p = self.paths.tables / f"{name}.csv"
        df.to_csv(p, index=index)
        self.manifest.add(name, p, "table", description)
        return p

    def save_dataset(self, df: pd.DataFrame, name: str, description: str = "") -> Path:
        p = self.paths.datasets / f"{name}.parquet"
        df.to_parquet(p, index=False)
        self.manifest.add(name, p, "dataset", description)
        return p

    def save_json(self, obj: dict, name: str, description: str = "") -> Path:
        p = self.paths.tables / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
        self.manifest.add(name, p, "table", description)
        return p


def load_context(cfg: EDAConfig | None = None, root: Path | None = None) -> EDAContext:
    cfg = cfg or EDAConfig()
    paths = EDAPaths(find_project_root(root)).ensure()
    manifest = Manifest(paths)
    stack = load_feature_stack(paths.raw("features"), cfg.max_dim)
    grid = stack.grid
    faults, finfo = load_on_grid(paths.raw("faults"), grid, Resampling.max)
    sub, sinfo = load_on_grid(paths.raw("submission"), grid, Resampling.nearest)
    for arr, info in ((faults, finfo), (sub, sinfo)):
        v = arr[np.isfinite(arr)]
        uniq = np.unique(v[: 5_000_000])
        info["n_unique_values"] = int(len(uniq))
        info["values_preview"] = [float(x) for x in uniq[:10]]
        info["valid_frac"] = float(np.isfinite(arr).mean())
    mask = np.isfinite(faults) & (faults > 0)
    finfo["positive_frac"] = float(mask.mean())
    cfg.profile_half_km = max(cfg.profile_half_km, 20 * grid.pixel_km)
    cfg.station_spacing_km = max(cfg.station_spacing_km, 2 * grid.pixel_km)
    cfg.background_min_km = max(cfg.background_min_km, cfg.profile_half_km)
    net = build_fault_network(mask, grid, cfg.region_grid)
    ctx = EDAContext(cfg, paths, manifest, stack, faults, finfo, sub, sinfo, net, Enhancer(grid),
                     np.random.default_rng(cfg.seed))
    for name in stack.names:
        ctx.signals[name], ctx.signal_notes[name] = standardize_signal(stack[name])
    return ctx


def add_derived_signals(ctx: EDAContext, scale_km: float) -> list[str]:
    """Fault-sensitive derived signals used in profiles and archetypes ('<modality>:<transform>')."""
    e = ctx.enh
    rtp, grav, elev = ctx.primary("magnetic"), ctx.primary("gravity"), ctx.primary("terrain")
    derived = {
        "magnetic:thdr": e.thdr(rtp, scale_km),
        "magnetic:tilt": e.tilt(rtp, scale_km),
        "magnetic:thdr_ridge": e.ridge(e.thdr(rtp, scale_km), scale_km, "bright")[0],
        "gravity:thdr": e.thdr(grav, scale_km),
        "gravity:tilt": e.tilt(grav, scale_km),
        "terrain:curvature": e.curvature(elev, scale_km)["max_abs"],
        "terrain:mexican_hat": e.log_wavelet(elev, scale_km),
        "basement:thdr": e.thdr(ctx.primary("basement"), scale_km),
        "conductivity:thdr": e.thdr(ctx.primary("conductivity"), scale_km),
    }
    for k, v in derived.items():
        ctx.signals[k], note = standardize_signal(v)
        ctx.signal_notes[k] = f"{note}; scale {scale_km:g} km"
    ctx.extra_maps["magnetic:tilt_deg"] = derived["magnetic:tilt"]
    return list(derived)


def evidence_signal_groups(signals: list[str]) -> dict[str, list[str]]:
    from .config import EVIDENCE_GROUPS, band_modality

    return {g: [s for s in signals if band_modality(s) == g] for g in EVIDENCE_GROUPS}


def define_windows(ctx: EDAContext) -> pd.DataFrame:
    """Four named, non-overlapping zoom windows reused by every section."""
    g, net = ctx.grid, ctx.net
    density = trace_density(net, g, ctx.cfg.density_radius_km)
    jn = net.nodes[net.nodes.kind == "junction"]
    jmap = np.zeros(g.shape, np.float32)
    if len(jn):
        np.add.at(jmap, (jn.r.round().astype(int).clip(0, g.height - 1), jn.c.round().astype(int).clip(0, g.width - 1)), 1)
    jmap = ndimage.gaussian_filter(jmap, g.km_to_px(ctx.cfg.density_radius_km))
    curv = np.zeros(g.shape, np.float32)
    for path, k in zip(net.paths, net.segments.curv_mean_deg_km.fillna(0)):
        curv[path[:, 0], path[:, 1]] = k
    curv = ndimage.gaussian_filter(curv, g.km_to_px(ctx.cfg.density_radius_km))
    fabric = rank_normalize(ctx.enh.thdr(ctx.primary("magnetic"), 5.0), ctx.valid)
    sparse = np.nan_to_num(fabric) - np.nan_to_num(rank_normalize(density, ctx.valid))
    ctx.windows = pick_windows([
        ("W1", "dense mapped network", density),
        ("W2", "fault intersections", jmap),
        ("W3", "curved structures", curv),
        ("W4", "strong magnetic fabric, sparse labels", sparse),
    ], g, ctx.valid, ctx.cfg.zoom_km)
    ctx.extra_maps["trace_density"] = density
    ctx.extra_maps["junction_density"] = jmap
    df = windows_table(ctx.windows, g)
    ctx.save_table(df, "windows", "Named zoom windows (pixel and CRS bounds) reused by all figures")
    return df


__all__ = ["EDAContext", "load_context", "add_derived_signals", "define_windows", "evidence_signal_groups", "MODALITIES"]
