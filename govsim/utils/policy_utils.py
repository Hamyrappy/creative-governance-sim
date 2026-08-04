# govsim/utils/policy_utils.py

"""
Policy sandbox on RestrictedPython.

Совместимо по API:
- validate_and_compile_policy_expression(expr, context_vars) -> code
- evaluate_safe_policy_code(code, context) -> value | None (на ошибке)
- PolicyValidationError (поднимается на этапе компиляции/валидации)

Настраивается через SandboxConfig: какие builtins/math/np доступны, короткие имена и т.п.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Set

import ast as _ast
import math as _math
import numpy as _np
import keyword
import re
import types as _types

from RestrictedPython import compile_restricted_eval
from RestrictedPython import safe_builtins, utility_builtins
from RestrictedPython.Eval import (
    default_guarded_getattr as _getattr_,
    default_guarded_getitem as _getitem_,
    default_guarded_getiter as _getiter_,
)

class PolicyValidationError(ValueError):
    pass


# Конфиг песочницы


@dataclass(frozen=True)
class SandboxConfig:
    """Определяет, что разрешено внутри выражений политики."""
    # Узкий набор встроенных поверх safe_builtins/utility_builtins
    allowed_builtins: Iterable[str] = ("abs", "min", "max", "round", "float", "int", "len")

    # Разрешённые функции из math:
    math_funcs: Mapping[str, Callable[..., Any]] = field(default_factory=lambda: {
        # базовые
        "sin": _math.sin, "cos": _math.cos, "tan": _math.tan,
        "asin": _math.asin, "acos": _math.acos, "atan": _math.atan, "atan2": _math.atan2,
        "sinh": _math.sinh, "cosh": _math.cosh, "tanh": _math.tanh,
        "exp": _math.exp, "log": _math.log, "log10": _math.log10, "log1p": _math.log1p,
        "sqrt": _math.sqrt, "pow": _math.pow, "hypot": _math.hypot,
        "floor": _math.floor, "ceil": _math.ceil, "fabs": _math.fabs, "fmod": _math.fmod,
        "erf": _math.erf, "erfc": _math.erfc, "gamma": _math.gamma, "lgamma": _math.lgamma,
        "copysign": _math.copysign, "isfinite": _math.isfinite, "isnan": _math.isnan, "isinf": _math.isinf,
    })

    # Разрешённые функции NumPy (проверенные безопасные, полезные для контроллеров).
    # NB: arange/linspace/ones/zeros/full deliberately EXCLUDED — они материализуют массив
    # произвольного размера из скаляра (напр. np.arange(10**9)) → неограниченная аллокация памяти
    # каждый шаг. Оставлены только редьюсеры/поэлементные функции, чей размер ограничен входом.
    numpy_funcs: Mapping[str, Callable[..., Any]] = field(default_factory=lambda: {
        "array": _np.array, "asarray": _np.asarray,
        "clip": _np.clip, "mean": _np.mean, "std": _np.std, "var": _np.var,
        "sum": _np.sum, "min": _np.min, "max": _np.max,
        "argmin": _np.argmin, "argmax": _np.argmax,
        "dot": _np.dot, "matmul": _np.matmul,
        "minimum": _np.minimum, "maximum": _np.maximum,
        "where": _np.where,
        # простые преобразования
        "reshape": _np.reshape, "ravel": _np.ravel, "stack": _np.stack, "concatenate": _np.concatenate,
        # удобные «активации»
        "tanh": _np.tanh, "sign": _np.sign, "abs": _np.abs,
    })

    # Дополнительные безопасные глобалы (свои функции/константы)
    extra_globals: Mapping[str, Any] = field(default_factory=dict)

    # Давать короткие имена (sin, clip, …) рядом с math/np
    expose_short_names: bool = True
    # Экспонировать неймспейсы math/np
    expose_namespaces: bool = True


DEFAULT_CONFIG = SandboxConfig()


# Внутренние утилиты

# Real stdlib MODULES that RestrictedPython's ``utility_builtins`` injects into ``__builtins__``
# (``random``, ``string``, ``unicodedata``, …). They are escape hatches — most dangerously the
# un-seeded global ``random`` module, which would inject entropy outside the system's seeded
# numpy Generator and destroy the byte-exact reproducibility guarantee. Strip them.
_BANNED_UTILITY_NAMES = frozenset({"random", "string", "unicodedata", "whrandom"})


def _build_safe_builtins(cfg: SandboxConfig) -> Dict[str, Any]:
    b: Dict[str, Any] = {}
    b.update(safe_builtins)
    b.update(utility_builtins)
    for name in _BANNED_UTILITY_NAMES:  # defense-in-depth even though the whitelist normally blocks them
        b.pop(name, None)
    py_builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__  # type: ignore
    for name in cfg.allowed_builtins:
        if name in py_builtins:
            b[name] = py_builtins[name]
    return b

def _namespace_from_dict(d: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**d)

def _coerce_numpy_scalars(d: MutableMapping[str, Any]) -> None:
    for k, v in list(d.items()):
        if isinstance(v, _np.generic):
            d[k] = v.item()

def _allowed_identifier_set(cfg: SandboxConfig, extra_allowed: Iterable[str] = ()) -> Set[str]:
    allowed: Set[str] = set(extra_allowed)
    if cfg.expose_short_names:
        allowed.update(cfg.math_funcs.keys())
        allowed.update(cfg.numpy_funcs.keys())
    if cfg.expose_namespaces:
        allowed.update({"math", "np"})
    allowed.update(cfg.extra_globals.keys())
    allowed.update(cfg.allowed_builtins)
    return allowed

def _extract_identifiers(expr: str) -> Set[str]:
    toks = set(re.findall(r"[A-Za-z_]\w*", expr))
    return {t for t in toks if not keyword.iskeyword(t)}


# Exponentiation is the one operator that turns a tiny expression into an unbounded CPU/memory bomb
# (``9**9**9`` is a multi-gigabyte integer). We reject CHAINED powers and any power with a large
# literal exponent; ordinary control terms like ``current_x ** 3`` stay allowed.
_POW_EXPONENT_CAP = 8


def _literal_number(node: "_ast.AST") -> float | None:
    """The numeric value of a constant (optionally unary-signed), else None."""
    if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.USub, _ast.UAdd)):
        inner = _literal_number(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, _ast.USub) else inner
    return None


def _check_expression_safety(expression: str) -> None:
    """Reject exponentiation bombs (chained ``a**b**c`` / a huge literal exponent) before compiling.

    Best-effort: if the expression does not parse as a Python expression here, we leave the error to
    the RestrictedPython compile step (which produces the proper feedback)."""
    try:
        tree = _ast.parse(expression, mode="eval")
    except SyntaxError:
        return
    for node in _ast.walk(tree):
        if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Pow):
            if isinstance(node.right, _ast.BinOp) and isinstance(node.right.op, _ast.Pow):
                raise PolicyValidationError("chained exponentiation (a ** b ** c) is not allowed.")
            exponent = _literal_number(node.right)
            if exponent is not None and abs(exponent) > _POW_EXPONENT_CAP:
                raise PolicyValidationError(
                    f"exponent {exponent:g} exceeds the safe cap {_POW_EXPONENT_CAP} "
                    "(guards against an exponentiation CPU/memory bomb)."
                )


# Публичный API

def validate_and_compile_policy_expression(
    expression: str,
    context_variable_names: Iterable[str] | None = None,
    *,
    config: SandboxConfig = DEFAULT_CONFIG,
):
    """
    Валидация идентификаторов и компиляция RestrictedPython-код-объекта.
    Бросает PolicyValidationError при проблемах.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise PolicyValidationError("Выражение политики не может быть пустым.")

    # Non-ASCII rejection closes the Unicode-confusable whitelist bypass: Python NFKC-normalizes
    # identifiers at compile time, so a fullwidth token like ``ｒａｎｄｏｍ`` compiles to the real
    # ``random`` — yet the ASCII-only identifier extractor never sees it to whitelist-check it. A
    # legitimate control law is pure ASCII, so requiring ASCII is both safe and sufficient here.
    if not expression.isascii():
        raise PolicyValidationError(
            "Выражение политики должно содержать только ASCII-символы "
            "(защита от обхода whitelist через Unicode-нормализацию идентификаторов)."
        )

    if context_variable_names is None:
        context_variable_names = ()

    ids_in_expr = _extract_identifiers(expression)
    allowed_ids = _allowed_identifier_set(config, extra_allowed=context_variable_names)
    unknown = sorted(x for x in ids_in_expr if x not in allowed_ids)
    if unknown:
        raise PolicyValidationError(
            "В выражении встречены неожиданные имена (не входят в контекст/whitelist): "
            + ", ".join(unknown[:10])
        )

    _check_expression_safety(expression)  # raises on an exponentiation bomb

    try:
        result = compile_restricted_eval(expression, filename="<policy_expression>")
    except Exception as e:
        raise PolicyValidationError(f"Ошибка компиляции выражения: {e}") from e
    # ``compile_restricted_eval`` does NOT raise on a syntax/restriction error — it returns a
    # CompileResult with ``.code is None`` and the reason in ``.errors``. Honor the documented
    # "raises PolicyValidationError on problems" contract so both sandbox entry points agree.
    errors = getattr(result, "errors", None)
    if errors:
        raise PolicyValidationError("Ошибка компиляции выражения: " + "; ".join(str(e) for e in errors))
    if getattr(result, "code", result) is None:
        raise PolicyValidationError("Выражение не скомпилировалось (code=None).")
    return result



_CodeType = type(compile("0", "<x>", "eval"))

def _ensure_code_object(maybe_code, *, expression_fallback: str | None = None):
    """
    Приводит что угодно к реальному code-объекту или возвращает None.
    Поддерживает:
      - уже скомпилированный code (CodeType)
      - строку с выражением (скомпилируем RestrictedPython'ом)
      - объекты с атрибутом .code (иногда так возвращают RP-врапперы)
      - dict с ключом 'code'
    """
    # 1) уже code
    if isinstance(maybe_code, _CodeType):
        return maybe_code

    # 2) иногда RP-хелперы возвращают обёртку с .code
    if hasattr(maybe_code, "code") and isinstance(getattr(maybe_code, "code"), _CodeType):
        return getattr(maybe_code, "code")

    # 3) словарик вида {"code": <CodeType>}
    if isinstance(maybe_code, dict) and isinstance(maybe_code.get("code"), _CodeType):
        return maybe_code["code"]

    # 4) пришла строка: скомпилируем сейчас (compile_restricted_eval возвращает CompileResult,
    #    поэтому берём из него .code — вернуть сам CompileResult в eval() нельзя)
    if isinstance(maybe_code, (str, bytes, bytearray)):
        try:
            src = maybe_code if isinstance(maybe_code, str) else maybe_code.decode("utf-8")
            compiled = compile_restricted_eval(src, filename="<policy_expression>")
            return getattr(compiled, "code", None)
        except Exception:
            return None

    # 5) как крайний случай: если дали fallback-строку выражения — попробуем по ней
    if expression_fallback and isinstance(expression_fallback, str):
        try:
            compiled = compile_restricted_eval(expression_fallback, filename="<policy_expression>")
            return getattr(compiled, "code", None)
        except Exception:
            return None

    return None



def evaluate_safe_policy_code(
    compiled_code,
    simulation_context: Mapping[str, Any],
    *,
    config: SandboxConfig = DEFAULT_CONFIG,
):
    """
    Выполняет выражение в ограниченной среде.
    По контракту проекта — на ошибке возвращаем None (а не кидаем исключение).
    """
    # Нормализуем то, что нам передали как "скомпилированное"
    code_obj = _ensure_code_object(compiled_code)
    if code_obj is None:
        print("Ошибка ВЫПОЛНЕНИЯ выражения политики: передан неподдерживаемый формат 'compiled_code'.")
        return None

    safe_locals: Dict[str, Any] = dict(simulation_context)
    _coerce_numpy_scalars(safe_locals)

    math_ns = _namespace_from_dict(config.math_funcs)
    numpy_ns = _namespace_from_dict(config.numpy_funcs)

    builtins_map = _build_safe_builtins(config)

    safe_globals: Dict[str, Any] = {
        "__builtins__": builtins_map,
        "_getattr_": _getattr_,
        "_getitem_": _getitem_,
        "_getiter_": _getiter_,
    }

    if config.expose_namespaces:
        safe_globals["math"] = math_ns
        safe_globals["np"] = numpy_ns

    if config.expose_short_names:
        short = {**config.math_funcs, **config.numpy_funcs}
        for name, func in short.items():
            # не перезаписывать builtins
            if name not in safe_locals and name not in config.allowed_builtins:
                safe_globals[name] = func

    for k, v in config.extra_globals.items():
        if k not in safe_locals:
            safe_globals[k] = v

    try:
        return eval(code_obj, safe_globals, safe_locals)
    except Exception as e:
        print(f"Ошибка ВЫПОЛНЕНИЯ выражения политики: {e}")
        return None


# --- не обязательные, но удобные алиасы ---
def compile_policy_expression(expr: str, ctx_vars: Iterable[str] | None = None, *, config: SandboxConfig = DEFAULT_CONFIG):
    return validate_and_compile_policy_expression(expr, ctx_vars, config=config)

def evaluate_policy_code(code, ctx: Mapping[str, Any], *, config: SandboxConfig = DEFAULT_CONFIG):
    return evaluate_safe_policy_code(code, ctx, config=config)
