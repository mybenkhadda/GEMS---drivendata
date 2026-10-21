"""Phase-5 feature store: Phase-2 + Phase-4 features unchanged, plus two Phase-5 families.

``p5:<modality>:profile_<output>[_<tag>]@<s>km``  fault-normal structural profiles (features.profiles),
                                                 catalogue ``configs/features/structural_profiles.yaml``
``dem:<name>``                                   1 m DEM morphology aggregated to the native 100 m grid
                                                 (dem.morphology), read from the mosaic GeoTIFF in
                                                 ``data/processed/dem_1m_100m/dem_features_100m.tif``

Phase-5 features are cached in ``data/feature_store/phase5/`` with their own registry and build log; the
Phase-2 and Phase-4 registries are never rewritten. Same API as FeatureStore (get / has / load_stack /
build), so lgbm3, cnn3 and phase4_training read Phase-5 channels without modification.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from gems_eda.enhance import standardize_signal

from .. import rs_features
from ..phase4_context import Phase4FeatureStore
from . import _fname, profiles

DEM_MOSAIC = "dem_features_100m.tif"
DEM_UNSTANDARDIZED = ("coverage", "source_project", "n_tiles")


def load_profile_specs(band_names: list[str], path: Path | None = None) -> list:
    """RSFeatureSpecs of the profile catalogue (empty list when the YAML is absent)."""
    import yaml

    from ..paths import config_path

    p = Path(path) if path else config_path("features", "structural_profiles.yaml")
    if not p.exists():
        return []
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    dflt = cfg.get("defaults", {})
    specs = []
    for bname, b in cfg.get("blocks", {}).items():
        if not b.get("enabled", True):
            continue
        band = rs_features._band(b["src"])
        if band not in band_names:
            print(f"[profiles] block {bname}: band {band} missing -> skipped")
            continue
        params = {**dflt, **(b.get("params") or {}), "src": b["src"], "orient_src": b.get("orient_src", b["src"])}
        tag = b.get("tag", "")
        tg = f"_{tag}" if tag else ""
        modality = b.get("modality", b["src"])
        for s in b.get("scales_km", [1.0]):
            for out in b.get("outputs", list(profiles.PROFILE_OUTPUTS)):
                specs.append(rs_features.RSFeatureSpec(
                    name=f"p5:{modality}:profile_{out}{tg}@{float(s):g}km", groups=("P5_PROF",), modality=modality,
                    source=band, transform="profile", scale_km=float(s), kind="local", origin="native",
                    units="robust-z", support_km=profiles.support_km("profile", float(s), params),
                    note=b.get("note", ""), family=f"P5-profile_{out}", basis="fault-normal profile", output=out,
                    params=rs_features._freeze(params), standardize=out not in ("asymmetry", "offset", "lineation")))
        pers = b.get("persistence")
        if pers:
            pp = {**params, **pers}
            specs.append(rs_features.RSFeatureSpec(
                name=f"p5:{modality}:profile_persistence{tg}@{float(pp['scales_km'][0]):g}km", groups=("P5_PROF",),
                modality=modality, source=band, transform="profile_persistence", scale_km=float(pp["scales_km"][0]),
                kind="local", origin="native", units="fraction", support_km=profiles.support_km("profile_persistence",
                float(pp["scales_km"][0]), pp), note="fraction of scales with a top-decile cross-strike step",
                family="P5-profile_persistence", basis="fault-normal profile", output="persistence",
                params=rs_features._freeze(pp), standardize=False))
    return specs


def dem_specs(dem_dir: Path | None) -> list:
    """One spec per band of the aggregated DEM mosaic (descriptions = feature names)."""
    if dem_dir is None or not (Path(dem_dir) / DEM_MOSAIC).exists():
        return []
    import rasterio

    with rasterio.open(Path(dem_dir) / DEM_MOSAIC) as src:
        desc = [d or f"band{i + 1}" for i, d in enumerate(src.descriptions)]
    return [rs_features.RSFeatureSpec(
        name=f"dem:{d}", groups=("P5_DEM",), modality="terrain_1m", source=f"{DEM_MOSAIC}:{i + 1}",
        transform="dem_band", scale_km=0.1, kind="local", origin="dem_1m_aggregated",
        units="fraction" if d in DEM_UNSTANDARDIZED else "robust-z", support_km=0.1,
        note="1 m DEM morphology aggregated to 100 m (dem.morphology)", family=f"P5-dem_{d.split('_')[0]}",
        basis="1 m DEM", output=d, params=(("band", i + 1),), standardize=d not in DEM_UNSTANDARDIZED)
        for i, d in enumerate(desc)]


class Phase5FeatureStore(Phase4FeatureStore):
    def __init__(self, data, legacy_specs: list, rs_specs: list, rs_dir: Path, p5_specs: list, p5_dir: Path,
                 dem_dir: Path | None = None):
        super().__init__(data, legacy_specs, rs_specs, rs_dir)
        self.p5_dir = Path(p5_dir)
        self.p5_dir.mkdir(parents=True, exist_ok=True)
        self.dem_dir = Path(dem_dir) if dem_dir else None
        extra = list(p5_specs) + dem_specs(self.dem_dir)
        for s in extra:
            self.specs[s.name] = s
        self.p5_names = [s.name for s in extra]
        self._p5_set = set(self.p5_names)

    @classmethod
    def from_phase4(cls, store4: Phase4FeatureStore, p5_specs: list, p5_dir: Path, dem_dir: Path | None = None):
        legacy = [s for n, s in store4.specs.items() if n not in store4._rs_set]
        rs = [store4.specs[n] for n in store4.rs_names]
        return cls(store4.data, legacy, rs, store4.rs_dir, p5_specs, p5_dir, dem_dir)

    def path(self, name: str) -> Path:
        if name in self._p5_set:
            return self.p5_dir / _fname(name)
        return super().path(name)

    def compute(self, spec):
        if spec.name in self._p5_set:
            if spec.transform in profiles.TRANSFORMS:
                return profiles.TRANSFORMS[spec.transform](self, spec)
            if spec.transform == "dem_band":
                return self._dem_band(int(spec.p("band")))
            raise KeyError(spec.transform)
        return super().compute(spec)

    def _dem_band(self, i: int) -> np.ndarray:
        import rasterio

        with rasterio.open(self.dem_dir / DEM_MOSAIC) as src:
            if src.shape != self.data.shape or src.transform != self.data.profile["transform"]:
                raise ValueError("DEM mosaic is not on the native grid: rebuild it with dem.morphology.mosaic")
            return np.ma.filled(src.read(i, masked=True).astype(np.float32), np.nan)

    def build(self, names: list[str] | None = None, overwrite: bool = False, verbose: bool = True) -> pd.DataFrame:
        names = names or list(self.specs)
        mine = [n for n in names if n in self._p5_set]
        other = [n for n in names if n not in self._p5_set]
        if other:
            super().build(other, overwrite=overwrite, verbose=verbose)
        log = []
        try:
            for n in mine:
                if self.has(n) and not overwrite:
                    continue
                spec = self.specs[n]
                t0 = time.perf_counter()
                arr = self.compute(spec).astype(np.float32)
                arr[~self.data.valid] = np.nan
                z, note = standardize_signal(arr) if spec.standardize else (arr, "none (bounded / fraction feature)")
                from ..artifacts.atomic import atomic_save

                atomic_save(self.path(n), lambda t, z=z: np.save(t, z.astype(np.float32)))
                self._mm.pop(n, None)
                log.append({"name": n, "seconds": round(time.perf_counter() - t0, 2), "standardization": note})
                if verbose:
                    print(f"  feature {n:<56s} {log[-1]['seconds']:7.1f}s  {note}")
        finally:
            self._src.clear()
            rs_features.clear_memo(self)
            profiles.clear_memo(self)
        return self.p5_registry(pd.DataFrame(log))

    def p5_registry(self, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        rows = []
        for n in self.p5_names:
            d = rs_features.spec_table([self.specs[n]]).iloc[0].to_dict()
            d["cached"] = self.has(n)
            d["path"] = self.data.paths.rel(self.path(n)) if self.has(n) and _under(self.path(n), self.data.paths.root) else str(self.path(n))
            rows.append(d)
        df = pd.DataFrame(rows)
        if extra is not None and len(extra):
            prev = self.p5_dir / "p5_feature_build_log.csv"
            log = pd.concat([pd.read_csv(prev), extra]) if prev.exists() else extra
            log.drop_duplicates("name", keep="last").to_csv(prev, index=False)
        df.to_csv(self.p5_dir / "p5_feature_registry.csv", index=False)
        return df

    def family_names(self, prefix: str) -> list[str]:
        return [n for n in self.p5_names if n.startswith(prefix)]


def _under(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False
