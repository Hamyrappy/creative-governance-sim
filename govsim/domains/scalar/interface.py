"""
ScalarLeverInterface — the trivial sibling of the (later) economy interface: clip a sandboxed
expression into a bounded lever. NO ledger, NO conservation. This is literally the old linear
world's apply path, generalized and pulled *out* of the System (the inversion of doc-09 §1.2:
acting is the ActionInterface's job, not the System's).

Used by ``CubicSystem`` (cubic/linear control), ``SIRSystem`` (epidemic), and ``CompanySystem``
(a firm) — two of them non-economic, which is the proof the core abstraction is not secretly
economy-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from govsim.core.action import (
    ActionInterface,
    ActionRequest,
    ActionSpace,
    ApplyReport,
    ValidationResult,
    VerbSpec,
)
from govsim.core.sandbox import compile_expr
from govsim.core.system import System


@dataclass(frozen=True)
class Lever:
    """One bounded control surface: a verb the regent emits, mapped to a system attribute."""

    name: str  # the verb the regent uses, e.g. "set_control_input"
    value_range: tuple[float, float]
    attr: str  # the system attribute the clipped value is written to, e.g. "current_u"
    description: str = ""


class ScalarLeverInterface(ActionInterface):
    def __init__(self, levers: list[Lever]) -> None:
        self.levers: dict[str, Lever] = {lev.name: lev for lev in levers}

    def action_space(self, system: System, regent_id: str) -> ActionSpace:
        context_vars = list(system.observe(regent_id).vars.keys())
        verbs = [
            VerbSpec(
                name=lev.name,
                value_range=lev.value_range,
                value_type="float",
                description=lev.description or f"Set lever '{lev.name}' (written to '{lev.attr}').",
            )
            for lev in self.levers.values()
        ]
        return ActionSpace(verbs=verbs, context_vars=context_vars)

    def validate(self, req: ActionRequest, system: System, regent_id: str) -> ValidationResult:
        lev = self.levers.get(req.verb)
        if lev is None:
            return ValidationResult(ok=False, feedback=f"Unknown verb '{req.verb}'. Allowed: {sorted(self.levers)}.")
        expr = req.payload.get("expr")
        if not isinstance(expr, str) or not expr.strip():
            return ValidationResult(ok=False, feedback=f"Verb '{req.verb}' needs payload {{'expr': '<python expression>'}}.")
        context_vars = list(system.observe(regent_id).vars.keys())
        return compile_expr(expr, context_vars)

    def apply(self, reqs: list[ActionRequest], system: System) -> ApplyReport:
        """Validate the batch; install each accepted lever expression onto the system.

        The expression is NOT evaluated here — ``System.step`` re-evaluates the installed lever
        every tick (the eval-cadence contract). ``install_lever`` stores it as pure data, so a
        cloned system re-evaluates against itself (sound rollout).
        """
        applied: list[ActionRequest] = []
        rejected: list[tuple[ActionRequest, str]] = []
        for req in reqs:
            result = self.validate(req, system, req.regent_id)
            if not result.ok:
                rejected.append((req, result.feedback))
                continue
            lev = self.levers[req.verb]
            system.install_lever(lev.attr, result.compiled, lev.value_range, req.regent_id)
            applied.append(req)
        return ApplyReport(applied=applied, rejected=rejected)
