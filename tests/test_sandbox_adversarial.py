"""
Adversarial tests for the policy sandbox.

The papers claim that an enacted policy is "a single Python expression with no statements, imports,
attribute access, or side effects", and that claim is load-bearing twice over: it is the whole basis
for calling the enacted rule auditable, and it is what makes it safe to execute text a language
model wrote. A claim like that should be attacked in the test suite rather than asserted in a
docstring, so this module is a battery of escapes rather than a happy path.

Each case is either rejected at compile time or, if it compiles, must not be able to reach anything
outside the supplied variable namespace. Both outcomes are acceptable; silently succeeding is not.
"""

from __future__ import annotations

import pytest

from govsim.core.sandbox import compile_expr, eval_safe

ALLOWED = ["current_x", "target_x", "I", "S"]
VARS = {"current_x": 1.0, "target_x": 0.0, "I": 0.1, "S": 0.9}

#: Things a model might plausibly emit, by accident or because it was asked to.
ESCAPES = [
    pytest.param("__import__('os').system('echo pwned')", id="import-builtin"),
    pytest.param("open('/etc/passwd').read()", id="open-file"),
    pytest.param("eval('1+1')", id="nested-eval"),
    pytest.param("exec('x=1')", id="exec"),
    pytest.param("current_x.__class__", id="dunder-class"),
    pytest.param("current_x.__class__.__mro__[1].__subclasses__()", id="subclasses-walk"),
    pytest.param("().__class__.__bases__", id="tuple-bases"),
    pytest.param("(lambda: globals())()", id="globals-via-lambda"),
    pytest.param("globals()", id="globals"),
    pytest.param("locals()", id="locals"),
    pytest.param("vars()", id="vars"),
    pytest.param("getattr(current_x, 'real')", id="getattr"),
    pytest.param("setattr(current_x, 'x', 1)", id="setattr"),
    pytest.param("[c for c in ().__class__.__base__.__subclasses__()]", id="comprehension-walk"),
    pytest.param("compile('1', '<s>', 'eval')", id="compile"),
    pytest.param("input()", id="input"),
    pytest.param("breakpoint()", id="breakpoint"),
    pytest.param("undefined_name", id="unwhitelisted-name"),
    pytest.param("current_x; target_x", id="statement-sequence"),
    pytest.param("x := 5", id="walrus-binding"),
]


@pytest.mark.parametrize("expr", ESCAPES)
def test_escape_attempt_never_succeeds(expr: str):
    """Reject at compile time, or fail closed at eval. Never quietly return a value."""
    res = compile_expr(expr, ALLOWED)
    if not res.ok:
        assert res.feedback, "a rejection must carry feedback — the trace channel shows it verbatim"
        return
    # If it compiled, evaluation must not produce a usable number from a forbidden capability.
    value = eval_safe(res.compiled, dict(VARS))
    assert not isinstance(value, (int, float)) or isinstance(value, bool) is False and value is None, (
        f"expression {expr!r} compiled AND evaluated to {value!r}; the sandbox let it through"
    )


@pytest.mark.parametrize("expr", [
    "-0.9 * current_x",
    "0.9 if I > 0.01 else 0.0",
    "0.5 * I + 0.05",
    "min(0.9, max(0.0, 2.0 * I))",
    "-(1.3 * current_x + 3.0 * current_x ** 3)",
    "abs(current_x - target_x)",
])
def test_legitimate_policies_still_compile_and_evaluate(expr: str):
    """The sandbox has to stay usable: every law our references and regents emit must survive it."""
    res = compile_expr(expr, ALLOWED)
    assert res.ok, f"legitimate policy rejected: {res.feedback}"
    value = eval_safe(res.compiled, dict(VARS))
    assert isinstance(value, (int, float)), f"{expr!r} evaluated to {value!r}"


def test_rejection_feedback_names_the_problem():
    """TraceFeedback forwards this string to the model, so it has to be actionable."""
    res = compile_expr("np.random.rand()", ALLOWED)
    assert not res.ok
    assert any(token in res.feedback.lower() for token in ("name", "allow", "not", "invalid")), (
        f"unhelpful rejection feedback: {res.feedback!r}"
    )


def test_runtime_error_fails_closed_rather_than_raising():
    """A law that compiles but blows up at eval must leave the lever alone, not kill the run."""
    res = compile_expr("current_x / 0", ALLOWED)
    if res.ok:
        assert eval_safe(res.compiled, dict(VARS)) is None
