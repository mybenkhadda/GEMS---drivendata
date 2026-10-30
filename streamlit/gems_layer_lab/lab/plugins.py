"""Trusted local feature plugins (Mode B).

TRUST MODEL: plugin files are ordinary Python modules that run with the full rights of the Streamlit / Colab process.
They are NOT sandboxed. Only files from explicitly configured local directories are loaded (default:
``gems_layer_lab/plugins/`` and ``$GEMS_LAB_PLUGIN_DIR``); the app never executes uploaded code. The restricted
formula editor (``featurelang``) can CALL registered plugins but cannot define them.

Registration API::

    from lab.plugins import register_feature
    from lab.ops import gradmag, local_std, safe_divide

    @register_feature(name="custom_disruption", inputs=["elevation", "slope"],
                      parameters={"sigma": 3.0, "eps": 0.01},
                      halo_px=lambda p: int(4 * p["sigma"]) + 4,        # context needed for exact tiling
                      description="terrain edge / slope texture", units="ratio")
    def custom_disruption(layers, sigma=3.0, eps=0.01):
        return safe_divide(gradmag(layers["elevation"], sigma), local_std(layers["slope"], sigma), eps)

Calling convention in a formula: one positional ARRAY argument per declared input, in declaration order (any
expression: a raw band, a saved feature, another formula), then keyword parameters::

    feature_B = custom_disruption(det_elev, det_elev_slope, sigma=4)

The function receives ``layers = {"elevation": <array>, "slope": <array>}`` and the parameters (declared defaults
filled in); it must return a 2-D array of the input shape (cast to float32; non-finite -> NaN).
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"


class PluginError(ValueError):
    pass


@dataclass
class PluginSpec:
    name: str
    fn: Callable
    inputs: tuple
    parameters: dict
    halo_px: Callable | int | None = None
    description: str = ""
    units: str = ""
    aoi_dependent: bool = False
    source_file: str = ""
    source_hash: str = ""
    warnings: list = field(default_factory=list)

    def halo(self, params: dict) -> int:
        p = {**self.parameters, **params}
        if self.halo_px is None:                  # conservative default from sigma-like parameters
            s = [float(v) for k, v in p.items() if "sigma" in k and isinstance(v, (int, float))]
            return int(math.ceil(4 * max(s))) + 4 if s else 0
        return int(self.halo_px(p) if callable(self.halo_px) else self.halo_px)

    def call(self, arrays: list, params: dict) -> np.ndarray:
        if len(arrays) != len(self.inputs):
            raise PluginError(f"{self.name} expects {len(self.inputs)} array inputs {list(self.inputs)}, "
                              f"got {len(arrays)}")
        unknown = set(params) - set(self.parameters)
        if unknown:
            raise PluginError(f"{self.name}: unknown parameters {sorted(unknown)} (declared {list(self.parameters)})")
        shapes = {np.shape(a) for a in arrays}
        if len(shapes) != 1:
            raise PluginError(f"{self.name}: input shapes differ {shapes}")
        out = self.fn({k: np.asarray(a, np.float32) for k, a in zip(self.inputs, arrays)},
                      **{**self.parameters, **params})
        out = np.asarray(out, np.float32)
        if out.shape != next(iter(shapes)):
            raise PluginError(f"{self.name} returned shape {out.shape}, expected {next(iter(shapes))}")
        bad = np.zeros(out.shape, bool)
        for a in arrays:
            bad |= ~np.isfinite(a)
        return np.where(np.isfinite(out) & ~bad, out, np.nan).astype(np.float32)

    def describe(self) -> dict:
        return {"name": self.name, "inputs": list(self.inputs), "parameters": self.parameters,
                "description": self.description, "units": self.units, "aoi_dependent": self.aoi_dependent,
                "source_file": self.source_file, "source_hash": self.source_hash,
                "call": f"{self.name}({', '.join(self.inputs)}"
                        + "".join(f", {k}={v!r}" for k, v in self.parameters.items()) + ")"}


_REGISTRY: dict[str, PluginSpec] = {}
_LOADING: dict = {"file": "", "hash": ""}


def register_feature(name: str, inputs: list, parameters: dict | None = None, halo_px=None, description: str = "",
                     units: str = "", aoi_dependent: bool = False):
    """Decorator registering a trusted feature function (see module docstring)."""
    if not name.isidentifier() or name.startswith("_"):
        raise PluginError(f"plugin name {name!r} must be a Python identifier")
    params = dict(parameters or {})
    for k, v in params.items():
        if not isinstance(v, (int, float, str, bool)):
            raise PluginError(f"{name}: parameter {k} default must be a number / string / bool")

    def deco(fn):
        sig = inspect.signature(fn)
        first = next(iter(sig.parameters), None)
        if first is None:
            raise PluginError(f"{name}: function must take the layers dict as first argument")
        missing = [k for k in params if k not in sig.parameters]
        if missing:
            raise PluginError(f"{name}: declared parameters {missing} are not function arguments")
        spec = PluginSpec(name, fn, tuple(inputs), params, halo_px, description or (inspect.getdoc(fn) or "")
                          .split("\n")[0], units, aoi_dependent, _LOADING["file"], _LOADING["hash"])
        if halo_px is None:
            spec.warnings.append("no halo_px declared: default 4 x max(sigma*) + 4 px used for tiling")
        _REGISTRY[name] = spec
        return fn
    return deco


def plugin_dirs(extra: list | None = None) -> list[Path]:
    dirs = [PLUGIN_DIR]
    env = os.getenv("GEMS_LAB_PLUGIN_DIR")
    if env:
        dirs += [Path(p) for p in env.split(os.pathsep) if p]
    return [Path(d) for d in dirs + list(extra or [])]


def load_plugins(dirs: list | None = None) -> tuple[dict[str, PluginSpec], list[str]]:
    """(registry, errors). Imports every ``*.py`` (not starting with '_') of the trusted directories; a failing file
    is reported and skipped. Re-loading replaces the registry (edited plugins are picked up)."""
    _REGISTRY.clear()
    errors = []
    for d in dirs or plugin_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_"):
                continue
            src = f.read_bytes()
            _LOADING.update(file=str(f), hash=hashlib.sha256(src).hexdigest()[:16])
            try:
                spec = importlib.util.spec_from_file_location(f"gll_plugin_{f.stem}_{_LOADING['hash']}", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception as e:                       # one broken plugin never disables the others
                errors.append(f"{f.name}: {type(e).__name__}: {e}")
            finally:
                _LOADING.update(file="", hash="")
    return dict(_REGISTRY), errors


def registry() -> dict[str, PluginSpec]:
    return dict(_REGISTRY)
