"""Safe, formula-based NumPy raster transformations (no eval / exec)."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

MAX_FORMULAS = 30
MAX_EXPR_NODES = 130
MAX_SIGMA_PX = 60.0
MAX_ABS_EXPONENT = 5


class FormulaError(ValueError):
    pass


def _array(x):
    return np.asarray(x, dtype=np.float32)


def _gaussian(a, sigma=3.0):
    sigma = float(sigma)
    if not np.isfinite(sigma) or not 0.25 <= sigma <= MAX_SIGMA_PX:
        raise FormulaError(f"sigma must be between 0.25 and {MAX_SIGMA_PX} pixels")
    x = _array(a)
    mask = np.isfinite(x)
    if not mask.any():
        return np.full_like(x, np.nan, dtype=np.float32)
    weights = gaussian_filter(mask.astype(np.float32), sigma, mode="reflect", truncate=4)
    values = gaussian_filter(np.where(mask, x, 0), sigma, mode="reflect", truncate=4)
    return np.divide(values, weights, out=np.full_like(x, np.nan), where=weights > 1e-5)


def _local_std(a, sigma=3.0):
    mean = _gaussian(a, sigma)
    second = _gaussian(_array(a) ** 2, sigma)
    return np.sqrt(np.maximum(second - mean ** 2, 0.0))


def _gradient(a, sigma=1.0):
    # 100 m GEMS pixels: differences are in 'value per pixel', not per metre.
    x = _gaussian(a, sigma)
    y, z = np.gradient(x)
    return np.hypot(z, y).astype(np.float32)


def _highpass(a, sigma=5.0):
    return _array(a) - _gaussian(a, sigma)


def _ratio(a, b, epsilon=1e-5):
    epsilon = float(epsilon)
    if not (0 < epsilon <= 1e10) or not np.isfinite(epsilon):
        raise FormulaError("ratio epsilon must be positive and finite")
    den = _array(b)
    safe = np.where(np.abs(den) < epsilon, np.where(den < 0, -epsilon, epsilon), den)
    return np.divide(_array(a), safe)


def _normdiff(a, b, epsilon=1e-5):
    epsilon = float(epsilon)
    if not (0 < epsilon <= 1e10) or not np.isfinite(epsilon):
        raise FormulaError("normdiff epsilon must be positive and finite")
    a, b = _array(a), _array(b)
    return (a - b) / (np.abs(a) + np.abs(b) + epsilon)


def _zscore(a):
    a = _array(a)
    good = np.isfinite(a)
    if not good.any():
        return np.full_like(a, np.nan)
    mu = float(np.mean(a[good]))
    std = float(np.std(a[good]))
    return (a - mu) / std if std > 1e-12 else np.full_like(a, np.nan)


def _robust_z(a):
    a = _array(a)
    good = np.isfinite(a)
    if not good.any():
        return np.full_like(a, np.nan)
    med = float(np.median(a[good]))
    mad = float(1.4826 * np.median(np.abs(a[good] - med)))
    return (a - med) / mad if mad > 1e-12 else np.full_like(a, np.nan)


FUNCTIONS = {
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
    "log1p": np.log1p, "exp": np.exp,
    "clip": np.clip, "minimum": np.minimum, "maximum": np.maximum,
    "where": np.where, "ratio": _ratio, "normdiff": _normdiff,
    "gaussian": _gaussian, "local_mean": _gaussian,
    "local_std": _local_std, "gradmag": _gradient,
    "highpass": _highpass, "zscore": _zscore, "robust_z": _robust_z,
}

ALLOWED_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
ALLOWED_COMPARISON = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Invert)


@dataclass(frozen=True)
class Formula:
    name: str
    expression: ast.AST
    source: str
    dependencies: tuple[str, ...]


def _validate_node(node, available):
    if isinstance(node, ast.Constant):
        if type(node.value) not in (float, int, bool):
            raise FormulaError("Only finite numeric and Boolean constants are allowed")
        if isinstance(node.value, (float, int)) and not np.isfinite(node.value):
            raise FormulaError("Nonfinite constants are not allowed")
        return set()
    if isinstance(node, ast.Name):
        if node.id not in available:
            raise FormulaError(f"Unknown layer: {node.id!r}")
        return {node.id}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ALLOWED_BINARY):
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or type(node.right.value) not in (float, int) or abs(node.right.value) > MAX_ABS_EXPONENT:
                raise FormulaError("Power must be a constant exponent within [-5, 5]")
        return _validate_node(node.left, available) | _validate_node(node.right, available)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ALLOWED_UNARY):
        return _validate_node(node.operand, available)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ALLOWED_COMPARISON):
        return _validate_node(node.left, available) | _validate_node(node.comparators[0], available)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
        # Local kernels need a predictable, literal sigma for correct AOI halo selection.
        if node.func.id in {"gaussian", "local_mean", "local_std", "gradmag", "highpass"}:
            sig_nodes = ([node.args[1]] if len(node.args) > 1 else []) + [
                kw.value for kw in node.keywords if kw.arg == "sigma"
            ]
            for sig in sig_nodes:
                if not isinstance(sig, ast.Constant) or type(sig.value) not in (int, float):
                    raise FormulaError("Local filter sigma must be a numeric constant")
                if not (0.25 <= float(sig.value) <= MAX_SIGMA_PX):
                    raise FormulaError(f"Local filter sigma must be in [0.25, {MAX_SIGMA_PX}] pixels")
        seen = set()
        for arg in node.args:
            seen |= _validate_node(arg, available)
        for kw in node.keywords:
            if not kw.arg or kw.arg.startswith("_"):
                raise FormulaError("Unsupported keyword")
            seen |= _validate_node(kw.value, available)
        return seen
    raise FormulaError(f"Disallowed syntax: {type(node).__name__} (attribute access, imports, subscripts and arbitrary calls are forbidden)")


def compile_formulas(source: str, available: list[str] | set[str]) -> list[Formula]:
    try:
        module = ast.parse(source or "", mode="exec")
    except SyntaxError as exc:
        raise FormulaError(f"Formula syntax at line {exc.lineno}: {exc.msg}") from exc
    if len(module.body) > MAX_FORMULAS:
        raise FormulaError(f"Maximum {MAX_FORMULAS} formulas")
    names = set(available)
    formulas = []
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise FormulaError("Write one assignment per line: new_layer = expression")
        name = stmt.targets[0].id
        if not name.isidentifier() or name.startswith("_") or name in names or name in FUNCTIONS:
            raise FormulaError(f"Invalid, reserved, or repeated layer name: {name}")
        if sum(1 for _ in ast.walk(stmt.value)) > MAX_EXPR_NODES:
            raise FormulaError(f"Expression too complex: {name}")
        dependencies = _validate_node(stmt.value, names)
        src = ast.get_source_segment(source, stmt) or name
        formulas.append(Formula(name, stmt.value, src, tuple(sorted(dependencies))))
        names.add(name)
    return formulas


def _run(node, context):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return context[node.id]
    if isinstance(node, ast.BinOp):
        a, b = _run(node.left, context), _run(node.right, context)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Pow): return a ** b
    if isinstance(node, ast.UnaryOp):
        a = _run(node.operand, context)
        if isinstance(node.op, ast.UAdd): return a
        if isinstance(node.op, ast.USub): return -a
        if isinstance(node.op, ast.Invert): return ~a
    if isinstance(node, ast.Compare):
        a, b = _run(node.left, context), _run(node.comparators[0], context)
        return {
            ast.Lt: np.less, ast.LtE: np.less_equal,
            ast.Gt: np.greater, ast.GtE: np.greater_equal,
            ast.Eq: np.equal, ast.NotEq: np.not_equal,
        }[type(node.ops[0])](a, b)
    if isinstance(node, ast.Call):
        f = FUNCTIONS[node.func.id]
        return f(*[_run(n, context) for n in node.args], **{k.arg: _run(k.value, context) for k in node.keywords})
    raise FormulaError(f"Unsupported expression: {type(node).__name__}")


def evaluate_formulas(formulas: list[Formula], raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate validated assignments in order; failure is isolated by formula name."""
    if not raw:
        raise FormulaError("No raster layers loaded")
    shape = next(iter(raw.values())).shape
    context = {k: _array(v) for k, v in raw.items()}
    for f in formulas:
        try:
            with np.errstate(all="ignore"):
                result = np.asarray(_run(f.expression, context), dtype=np.float32)
            if result.ndim == 0:
                result = np.full(shape, result.item(), dtype=np.float32)
            if result.shape != shape:
                raise FormulaError(f"shape {result.shape} instead of {shape}")
            if result.nbytes > 128 * 1024 * 1024:
                raise FormulaError("Formula output exceeds 128 MiB; select a smaller AOI")
            result = np.where(np.isfinite(result), result, np.nan).astype(np.float32, copy=False)
            context[f.name] = result
        except Exception as e:
            raise FormulaError(f"{f.name}: {e}") from e
    return context


def suggested_halo(formulas: list[Formula]) -> int:
    """Conservative 4*sigma halo for explicit local operators; no halo for ROI statistics."""
    sigmas = [2.0]
    ops = {"gaussian", "local_mean", "local_std", "gradmag", "highpass"}
    for f in formulas:
        for node in ast.walk(f.expression):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ops:
                sigma = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    sigma = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "sigma" and isinstance(kw.value, ast.Constant):
                        sigma = kw.value.value
                sigmas.append(float(sigma if sigma is not None else (5 if node.func.id == "highpass" else 3)))
    return min(260, int(np.ceil(4 * max(sigmas))) + 2)
