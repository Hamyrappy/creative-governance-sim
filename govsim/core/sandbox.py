"""
sandbox — the policy-expression compiler/evaluator, re-homed onto the core seam.

This is a thin bridge over ``govsim.utils.policy_utils`` (RestrictedPython; KEPT per the
grand plan) that speaks the core's ``ValidationResult`` vocabulary: a bad expression returns a
structured *reject-with-feedback*, never a silent ``None`` (doc-08 dead-end fix). It is
domain-agnostic — only "a single Python expression over a whitelist of names" — so it lives in
``govsim.core`` and is reused by every ``ActionInterface`` that accepts an ``expr`` payload
(today the scalar lever interface; later the economy interface's institution bodies).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from govsim.core.action import ValidationResult
from govsim.utils.policy_utils import (
    PolicyValidationError,
    SandboxConfig,
    DEFAULT_CONFIG,
    evaluate_safe_policy_code,
    validate_and_compile_policy_expression,
)

__all__ = [
    "compile_expr",
    "eval_safe",
    "SandboxConfig",
    "DEFAULT_CONFIG",
    "PolicyValidationError",
]


def compile_expr(
    expr: str,
    allowed: Iterable[str] | None = None,
    *,
    config: SandboxConfig = DEFAULT_CONFIG,
) -> ValidationResult:
    """Validate identifiers + compile ``expr`` under RestrictedPython.

    Returns ``ValidationResult(ok=True, compiled=<code>)`` on success, or
    ``ValidationResult(ok=False, feedback=<why>)`` on a compile/whitelist failure — the
    feedback string is what a ``TraceFeedback`` harness component shows the regent next turn.
    """
    try:
        result = validate_and_compile_policy_expression(expr, allowed, config=config)
    except PolicyValidationError as e:
        return ValidationResult(ok=False, feedback=str(e))
    # ``compile_restricted_eval`` returns a CompileResult(code, errors, warnings, used_names);
    # it does NOT raise on a syntax error — code is None and the reason is in ``errors``.
    errors = getattr(result, "errors", None)
    code = getattr(result, "code", result)
    if errors:
        return ValidationResult(ok=False, feedback="; ".join(str(e) for e in errors))
    if code is None:
        return ValidationResult(ok=False, feedback="Expression failed to compile (no code produced).")
    return ValidationResult(ok=True, compiled=code)


def eval_safe(
    compiled: Any,
    variables: Mapping[str, Any],
    *,
    config: SandboxConfig = DEFAULT_CONFIG,
) -> Any:
    """Evaluate compiled policy code in the restricted environment.

    Returns the value, or ``None`` on a runtime error (the project contract). Callers that
    need the failure reason should pre-validate with :func:`compile_expr`.
    """
    return evaluate_safe_policy_code(compiled, variables, config=config)
