"""Restricted feature language (Mode A): multi-line Python-STYLE assignments evaluated by an AST interpreter.

    PARAMS = {"sigma": {"min": 1, "max": 20, "step": 1, "default": 5},
              "eps":   {"min": 0.001, "max": 0.2, "step": 0.001, "default": 0.01}}
    mag = gradmag(rtp, sigma=sigma)
    grav = gradmag(iso_grav_anom, sigma=sigma)
    ratio_mg = ratio(mag, grav, eps=eps)
    terrain = highpass(det_elev, sigma=8)
    feature_A = abs(terrain) * ratio_mg

Allowed: assignments ``name = expression`` (one target), numbers / booleans / short strings, + - * / ** (constant
exponent |p| <= 5), comparisons (one operator, result NaN where an operand is NaN), & | ~ on masks, calls of the
operations in ``lab.ops.OPS`` and of registered trusted plugins, names of input layers, saved library features,
earlier variables and PARAMS. Everything else (imports, attributes, subscripts, loops, comprehensions, lambdas,
f-strings, starred args, unknown functions, ...) is rejected at compile time. ``PARAMS`` is parsed as a strict
literal (dict of dicts of numbers), never executed.

Leakage guard: names with role ``target`` (mapped faults) or ``prediction`` (W2 / M9 maps) are refused as inputs.

Scalar arguments of operations (sigma, eps, ...) must be compile-time constants (numbers, PARAMS or negated
numbers), so the halo needed for exact windowed / tiled evaluation is known before reading any pixel.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import ops as O

MAX_STATEMENTS = 40
MAX_NODES = 220
MAX_STR = 24
NORMALIZERS = {"zscore": ("center", "scale"), "robust_z": ("center", "scale"), "minmax": ("lo", "hi"),
               "winsorize": ("lo", "hi"), "quantile_clip": ("lo", "hi"), "texture_entropy": ("lo", "hi")}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
        ast.Eq: np.equal, ast.NotEq: np.not_equal}


class FeatureError(ValueError):
    """Rejected or failed formula (message names the line / variable)."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    min: float
    max: float
    step: float
    default: float

    def clamp(self, v) -> float:
        return float(min(max(float(v), self.min), self.max))


@dataclass
class Statement:
    name: str
    expr: ast.AST
    source: str
    line: int
    raw: set = field(default_factory=set)          # input layers referenced directly
    locals: set = field(default_factory=set)       # earlier program variables referenced
    library: set = field(default_factory=set)      # saved library features referenced
    calls: set = field(default_factory=set)        # ops / plugins called
    params: set = field(default_factory=set)       # PARAMS used
    halo: int = 0
    aoi_dependent: bool = False


@dataclass
class Program:
    source: str
    statements: list
    params: dict                                   # name -> ParamSpec
    values: dict                                   # name -> current value
    warnings: list = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.statements]

    @property
    def raw_inputs(self) -> set:
        return set().union(*[s.raw for s in self.statements]) if self.statements else set()

    @property
    def library_refs(self) -> set:
        return set().union(*[s.library for s in self.statements]) if self.statements else set()

    @property
    def halo(self) -> int:
        return max([s.halo for s in self.statements] + [0])

    def statement(self, name: str) -> Statement:
        return next(s for s in self.statements if s.name == name)


@dataclass
class Env:
    """What a formula may reference. ``library`` maps a saved feature name to (halo_px, aoi_dependent)."""
    layers: set
    forbidden: dict = field(default_factory=dict)          # name -> role ('target' | 'prediction')
    library: dict = field(default_factory=dict)
    plugins: dict = field(default_factory=dict)             # name -> PluginSpec


# =============================================================================== literal PARAMS
def _num(node) -> float:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_num(node.operand)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float) and np.isfinite(node.value):
        return float(node.value)
    raise FeatureError(f"line {getattr(node, 'lineno', '?')}: PARAMS values must be numbers")


def parse_params(node: ast.AST) -> dict:
    if not isinstance(node, ast.Dict):
        raise FeatureError("PARAMS must be a dict literal {name: {min, max, step, default}}")
    out = {}
    for k, v in zip(node.keys, node.values):
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.isidentifier()):
            raise FeatureError("PARAMS keys must be identifier strings")
        if not isinstance(v, ast.Dict):
            raise FeatureError(f"PARAMS[{k.value!r}] must be a dict literal")
        d = {}
        for kk, vv in zip(v.keys, v.values):
            if not (isinstance(kk, ast.Constant) and kk.value in ("min", "max", "step", "default")):
                raise FeatureError(f"PARAMS[{k.value!r}]: only min / max / step / default keys")
            d[kk.value] = _num(vv)
        if set(d) != {"min", "max", "step", "default"}:
            raise FeatureError(f"PARAMS[{k.value!r}] needs min, max, step and default")
        if not (d["min"] <= d["default"] <= d["max"]) or d["step"] <= 0:
            raise FeatureError(f"PARAMS[{k.value!r}]: need min <= default <= max and step > 0")
        out[k.value] = ParamSpec(k.value, d["min"], d["max"], d["step"], d["default"])
    return out


# =============================================================================== compile
class _Compiler:
    def __init__(self, env: Env, params: dict, values: dict, defined: dict):
        self.env, self.params, self.values, self.defined = env, params, values, defined

    def scalar(self, node):
        """Compile-time scalar (number, bool, short string, PARAMS name, negation)."""
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or (type(v) in (int, float) and np.isfinite(v)):
                return v
            if isinstance(v, str) and len(v) <= MAX_STR and v.replace("_", "").isalnum():
                return v
            raise FeatureError(f"line {node.lineno}: unsupported constant {v!r}")
        if isinstance(node, ast.Name) and node.id in self.params:
            return self.values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = self.scalar(node.operand)
            if isinstance(v, str):
                raise FeatureError(f"line {node.lineno}: cannot negate a string")
            return -v if isinstance(node.op, ast.USub) else v
        raise FeatureError(f"line {getattr(node, 'lineno', '?')}: this argument must be a number or a PARAMS name "
                           "(needed at compile time for exact AOI halos)")

    def visit(self, node, st: Statement):
        """-> (halo_px, aoi_dependent) of the expression rooted at ``node``."""
        ln = getattr(node, "lineno", st.line)
        if isinstance(node, ast.Constant):
            self.scalar(node)
            return 0, False
        if isinstance(node, ast.Name):
            n = node.id
            if n in self.params:
                st.params.add(n)
                return 0, False
            if n in self.defined:
                st.locals.add(n)
                return self.defined[n]
            if n in self.env.forbidden:
                raise FeatureError(f"line {ln}: {n!r} is a {self.env.forbidden[n]} layer and cannot be a formula "
                                   "input (target / prediction leakage guard)")
            if n in self.env.layers:
                st.raw.add(n)
                return 0, False
            if n in self.env.library:
                st.library.add(n)
                return self.env.library[n]
            if n in O.OPS or n in self.env.plugins:
                raise FeatureError(f"line {ln}: {n!r} is a function; call it, e.g. {n}(...)")
            raise FeatureError(f"line {ln}: unknown name {n!r} (not a layer, saved feature, variable or PARAMS)")
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                p = self.scalar(node.right)
                if isinstance(p, str) or abs(float(p)) > 5:
                    raise FeatureError(f"line {ln}: exponent must be a number within [-5, 5]")
                return self.visit(node.left, st)
            if not isinstance(node.op, (*_BIN, ast.BitAnd, ast.BitOr)):
                raise FeatureError(f"line {ln}: operator {type(node.op).__name__} not allowed")
            a, b = self.visit(node.left, st), self.visit(node.right, st)
            return max(a[0], b[0]), a[1] or b[1]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            return self.visit(node.operand, st)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
                raise FeatureError(f"line {ln}: one comparison operator at a time (<, <=, >, >=, ==, !=)")
            a, b = self.visit(node.left, st), self.visit(node.comparators[0], st)
            return max(a[0], b[0]), a[1] or b[1]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FeatureError(f"line {ln}: only direct calls of library functions (no attributes / methods)")
            if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
                raise FeatureError(f"line {ln}: *args / **kwargs are not allowed")
            f = node.func.id
            st.calls.add(f)
            if f in O.OPS:
                return self.visit_op(f, node, st)
            if f in self.env.plugins:
                return self.visit_plugin(f, node, st)
            raise FeatureError(f"line {ln}: unknown function {f!r} (see the function reference)")
        raise FeatureError(f"line {ln}: {type(node).__name__} is not allowed in formulas (no imports, attributes, "
                           "subscripts, loops, comprehensions or lambdas)")

    def visit_op(self, f, node, st):
        spec = O.OPS[f]
        try:
            b = spec.signature().bind(*node.args, **{k.arg: k.value for k in node.keywords})
        except TypeError as e:
            raise FeatureError(f"line {node.lineno}: {f}{spec.signature()}: {e}") from e
        b.apply_defaults()
        halo, aoi, scal = 0, False, {}
        for pname, val in b.arguments.items():
            if pname in spec.arrays:
                if not isinstance(val, ast.AST):
                    raise FeatureError(f"line {node.lineno}: {f}: array argument {pname!r} missing")
                h, a = self.visit(val, st)
                halo, aoi = max(halo, h), aoi or a
            else:
                scal[pname] = self.scalar(val) if isinstance(val, ast.AST) else val
        try:
            for k, v in scal.items():                    # range checks before any pixel is read
                if v is None or isinstance(v, (str, bool)):
                    continue
                if "sigma" in k:
                    O._sigma(v, k, allow_zero=True)
                elif k == "size":
                    O._size(v, k)
            own = spec.halo(scal)
            aoi_own = spec.aoi_dependent(scal)
        except (TypeError, ValueError) as e:
            raise FeatureError(f"line {node.lineno}: {f}: invalid argument ({e})") from e
        return own + halo, aoi or aoi_own

    def visit_plugin(self, f, node, st):
        spec = self.env.plugins[f]
        if len(node.args) != len(spec.inputs):
            raise FeatureError(f"line {node.lineno}: {f} takes {len(spec.inputs)} array inputs {list(spec.inputs)} "
                               f"(positional), got {len(node.args)}")
        halo, aoi = 0, spec.aoi_dependent
        for a in node.args:
            h, d = self.visit(a, st)
            halo, aoi = max(halo, h), aoi or d
        kw = {}
        for k in node.keywords:
            if k.arg not in spec.parameters:
                raise FeatureError(f"line {node.lineno}: {f} has no parameter {k.arg!r} ({list(spec.parameters)})")
            kw[k.arg] = self.scalar(k.value)
        return spec.halo(kw) + halo, aoi


def compile_program(source: str, env: Env, values: dict | None = None) -> Program:
    """Parse + validate. ``values`` overrides PARAMS defaults (sliders); out-of-range values are clamped."""
    try:
        mod = ast.parse(source or "", mode="exec")
    except SyntaxError as e:
        raise FeatureError(f"syntax error line {e.lineno}: {e.msg}") from e
    body = list(mod.body)
    params: dict = {}
    if body and isinstance(body[0], ast.Assign) and len(body[0].targets) == 1 and \
            isinstance(body[0].targets[0], ast.Name) and body[0].targets[0].id == "PARAMS":
        params = parse_params(body[0].value)
        body = body[1:]
    vals = {k: p.clamp((values or {}).get(k, p.default)) for k, p in params.items()}
    if len(body) > MAX_STATEMENTS:
        raise FeatureError(f"at most {MAX_STATEMENTS} assignments")
    defined: dict = {}
    stmts, warns = [], []
    comp = _Compiler(env, params, vals, defined)
    reserved = set(O.OPS) | set(env.plugins) | set(env.layers) | set(env.forbidden) | set(params) | {"PARAMS"}
    for node in body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue                                     # a free-text note line
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            raise FeatureError(f"line {node.lineno}: write one assignment per statement: name = expression "
                               "(PARAMS, if used, must be the first statement)")
        name = node.targets[0].id
        if name.startswith("_") or name in reserved or name in defined:
            raise FeatureError(f"line {node.lineno}: {name!r} is reserved, a layer / function name, or defined twice")
        if sum(1 for _ in ast.walk(node.value)) > MAX_NODES:
            raise FeatureError(f"line {node.lineno}: expression of {name} too complex (> {MAX_NODES} nodes)")
        if name in env.library:
            warns.append(f"{name} shadows the saved library feature of the same name (this program's definition is "
                         "used; saving creates a new version)")
        st = Statement(name, node.value, ast.get_source_segment(source, node.value) or name, node.lineno)
        st.halo, st.aoi_dependent = comp.visit(node.value, st)
        defined[name] = (st.halo, st.aoi_dependent)
        stmts.append(st)
    return Program(source, stmts, params, vals, warns)


# =============================================================================== evaluate
def _as_mask(x, shape):
    a = np.asarray(x, np.float32)
    return np.broadcast_to(a, shape) if a.ndim == 0 else a


def run_statement(st: Statement, program: Program, lookup: Callable[[str], np.ndarray], env: Env,
                  shape: tuple) -> np.ndarray:
    """Evaluate one compiled statement; ``lookup(name)`` returns the array of a layer / variable / library feature."""

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in program.params:
                return program.values[node.id]
            return lookup(node.id)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return O.f_power(ev(node.left), _scalar(node.right))
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                A, B = _as_mask(a, shape), _as_mask(b, shape)
                r = ((A > 0.5) & (B > 0.5)) if isinstance(node.op, ast.BitAnd) else ((A > 0.5) | (B > 0.5))
                return np.where(np.isfinite(A) & np.isfinite(B), r.astype(np.float32), np.nan)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return np.asarray(_BIN[type(node.op)](a, b), np.float32)
        if isinstance(node, ast.UnaryOp):
            a = ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -np.asarray(a, np.float32) if np.ndim(a) else -a
            if isinstance(node.op, ast.Invert):
                A = _as_mask(a, shape)
                return np.where(np.isfinite(A), (A <= 0.5).astype(np.float32), np.nan)
            return a
        if isinstance(node, ast.Compare):
            a, b = ev(node.left), ev(node.comparators[0])
            A, B = np.asarray(a, np.float32), np.asarray(b, np.float32)
            with np.errstate(invalid="ignore"):
                r = _CMP[type(node.ops[0])](A, B).astype(np.float32)
            ok = np.isfinite(A) & np.isfinite(B)
            return np.where(ok, r, np.nan).astype(np.float32)
        if isinstance(node, ast.Call):
            f = node.func.id
            site = f"{st.name}:{node.lineno}:{node.col_offset}:{f}"
            if f in O.OPS:
                spec = O.OPS[f]
                b = spec.signature().bind(*node.args, **{k.arg: k.value for k in node.keywords})
                kw = {}
                for pname, val in b.arguments.items():
                    if pname in spec.arrays:
                        v = ev(val)
                        kw[pname] = np.broadcast_to(np.asarray(v, np.float32), shape) if np.ndim(v) == 0 else v
                    else:
                        kw[pname] = _scalar(val)
                prev, O._REC.site = O._REC.site, site
                try:
                    return spec.fn(**kw)
                except O.OpError as e:
                    raise FeatureError(f"{st.name} (line {node.lineno}): {f}: {e}") from e
                finally:
                    O._REC.site = prev
            spec = env.plugins[f]
            arrays = [ev(a) for a in node.args]
            arrays = [np.broadcast_to(np.asarray(a, np.float32), shape) if np.ndim(a) == 0 else a for a in arrays]
            return spec.call(arrays, {k.arg: _scalar(k.value) for k in node.keywords})
        raise FeatureError(f"{st.name}: unsupported node {type(node).__name__}")

    def _scalar(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return program.values[node.id]
        if isinstance(node, ast.UnaryOp):
            v = _scalar(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        return node                                       # signature default (python value)

    with np.errstate(all="ignore"):
        out = ev(st.expr)
    out = np.asarray(out, np.float32)
    if out.ndim == 0:
        out = np.full(shape, float(out), np.float32)
    if out.shape != tuple(shape):
        raise FeatureError(f"{st.name}: result shape {out.shape} != {tuple(shape)}")
    return out


# =============================================================================== diagnostics / hashing / freezing
def diagnose(arr: np.ndarray, valid_inputs: np.ndarray | None = None, division: list | None = None) -> dict:
    """NaN / Inf / constant / outlier / division-instability warnings of one result."""
    a = np.asarray(arr, np.float32)
    n = a.size
    inf = int(np.isinf(a).sum())
    fin = np.isfinite(a)
    v = a[fin]
    d = {"n_pixels": int(n), "valid_share": float(fin.mean()) if n else 0.0, "n_inf": inf}
    if v.size:
        med = float(np.median(v))
        mad = float(1.4826 * np.median(np.abs(v - med)))
        d.update(min=float(v.min()), p02=float(np.percentile(v, 2)), median=med, p98=float(np.percentile(v, 98)),
                 max=float(v.max()), std=float(v.std()),
                 outlier_share=float(np.mean(np.abs(v - med) > 20 * mad)) if mad > 0 else 0.0)
    w = []
    if inf:
        w.append(f"{inf} infinite values (converted to NaN): division by ~0 or overflow - prefer ratio()/safe_divide()")
    if valid_inputs is not None and np.any(valid_inputs & ~fin):
        w.append(f"{int((valid_inputs & ~fin).sum())} pixels became NaN although their inputs were valid "
                 "(log/sqrt domain, 0/0, overflow)")
    if v.size and float(v.std()) < 1e-12:
        w.append("constant array (no variation in this AOI)")
    if v.size and d.get("outlier_share", 0) > 0.005:
        w.append(f"{100 * d['outlier_share']:.2f} % extreme outliers (> 20 robust SD): consider winsorize / log1p")
    if not v.size:
        w.append("no valid pixel")
    for div in division or []:
        if div["floored_share"] > 0.01:
            w.append(f"{div['op']} at {div['site']}: {100 * div['floored_share']:.1f} % of denominators below eps "
                     "(division instability)")
    d["warnings"] = w
    return d


def expr_hash(st: Statement, program: Program, dep_hashes: dict, extra: str = "") -> str:
    """Content hash of a statement: its AST, the PARAMS values it uses, the hashes of everything it references and
    the ops version. Any upstream change changes it, so dependent caches invalidate and nothing else does."""
    payload = {"ast": ast.dump(st.expr, annotate_fields=False), "params": {p: program.values[p] for p in
                                                                          sorted(st.params)},
               "deps": {k: dep_hashes.get(k) for k in sorted(st.raw | st.locals | st.library | st.calls)},
               "ops": O.OPS_VERSION, "extra": extra}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def freeze_normalization(source: str, recorded: dict) -> tuple[str, list[str]]:
    """Insert the statistics recorded by AOI-dependent normalisations (``ops.Recorder.normalization``, keyed by call
    site) as explicit keyword constants, e.g. ``zscore(x)`` -> ``zscore(x, center=1.23, scale=4.5)``. Text edits keep
    comments. Use statistics from a DEVELOPMENT area only."""
    mod = ast.parse(source)
    edits, done = [], []
    raw = source.encode("utf-8")                         # ast column offsets are UTF-8 byte offsets
    lines = raw.splitlines(keepends=True)
    offs = np.cumsum([0] + [len(x) for x in lines])
    for stmt in mod.body:
        if not (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)):
            continue
        for node in ast.walk(stmt.value):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in NORMALIZERS:
                site = f"{stmt.targets[0].id}:{node.lineno}:{node.col_offset}:{node.func.id}"
                rec = recorded.get(site)
                keys = NORMALIZERS[node.func.id]
                if not rec or any(k in {kw.arg for kw in node.keywords} for k in keys):
                    continue
                ins = "".join(f", {k}={rec[k]:.8g}" for k in keys if k in rec)
                pos = int(offs[node.end_lineno - 1] + node.end_col_offset - 1)   # before the closing parenthesis
                edits.append((pos, ins))
                done.append(f"{site} <- {ins.strip(', ')}")
    out = raw
    for pos, ins in sorted(edits, reverse=True):
        out = out[:pos] + ins.encode("utf-8") + out[pos:]
    return out.decode("utf-8"), done
