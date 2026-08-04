"""
Regime calibration — the domain-general way to bound how much adaptation is worth.

Every adaptation claim ("the regent recovers from an unseen structural break better than a rule
that cannot") is only meaningful relative to two reference policies, and both are *computable*
rather than argued:

    frozen  = argmin over a policy family Θ of the loss on the PRE-shock world,
              then deployed unchanged through the shock. The best a designer could have done
              with everything knowable before the break — and nothing after it.
    oracle  = argmin over the SAME Θ of the loss on the POST-shock world.
              The best that family can do once the break is known. Clairvoyant by construction.

Their gap is the **headroom**: the entire budget any adaptive mechanism is competing for.

    headroom = L(frozen) / L(oracle)          (both scored on the post-shock window)
    R(arm)   = (L(arm) − L(oracle)) / (L(frozen) − L(oracle))

``R = 0`` is clairvoyant, ``R = 1`` is "never adapted", and ``R`` is comparable across systems,
severities, and models in a way raw MSE is not. A regime with headroom ≈ 1 has no budget to
compete for, so a null result there is a fact about the *environment*, not about the regent —
which is why this module runs before an experiment, not after it.

Both references are held to the same policy family as each other. That is deliberate: it keeps the
comparison about *when the parameters were chosen*, never about who had the richer language.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from govsim.core.experiment import Experiment, Hypothesis
from govsim.core.objective import Objective
from govsim.core.regent import MultiScriptedRegent, ScriptedRegent
from govsim.core.runner import Runner
from govsim.core.schedule import Schedule


@dataclass(frozen=True)
class PolicyFamily:
    """A parameterized family of control laws: a template plus the grid its knobs range over.

    ``template`` is a ``str.format`` pattern over the grid's parameter names that renders to ONE
    sandboxed expression, e.g. ``"{k} * (target_x - current_x)"`` with ``grid={"k": [...]}``, or
    ``"{a} if I > {thr} else 0.0"`` for a threshold institution. The family is the *language* both
    references share, so it should be expressive enough that "the oracle is weak" is never the
    explanation for a small headroom.

    ``extra_laws`` lets one family govern SEVERAL instruments jointly: a second mapping of
    ``verb -> template`` rendered from the same grid. A regime whose correct response to a broken
    instrument is "use the other one" cannot be scored against a reference confined to the broken
    one, because that reference cannot express the response the experiment is about.
    """

    verb: str
    template: str
    grid: dict[str, list[float]]
    extra_laws: dict[str, str] = field(default_factory=dict)

    def render(self, params: dict[str, float]) -> str:
        return self.template.format(**{k: repr(v) for k, v in params.items()})

    def render_all(self, params: dict[str, float]) -> dict[str, str]:
        """``{verb: expression}`` for every instrument this family governs."""
        p = {k: repr(v) for k, v in params.items()}
        return {self.verb: self.template.format(**p),
                **{v: t.format(**p) for v, t in self.extra_laws.items()}}

    def combinations(self) -> Iterable[dict[str, float]]:
        names = list(self.grid)
        for values in itertools.product(*(self.grid[n] for n in names)):
            yield dict(zip(names, values))

    def size(self) -> int:
        n = 1
        for v in self.grid.values():
            n *= len(v)
        return n


@dataclass
class CalibrationResult:
    best_params: dict[str, float]
    best_expr: str
    best_loss: float
    #: ``{verb: expression}`` for every instrument the family governs (== ``{verb: best_expr}``
    #: for a single-instrument family).
    best_laws: dict[str, str] = field(default_factory=dict)
    all_losses: list[tuple[dict[str, float], float]] = field(default_factory=list)


def _score_expr(
    expr: str | dict[str, str],
    *,
    verb: str,
    system_factory: Callable[[int], Any],
    action_interface: Any,
    objective: Objective,
    schedule: Schedule,
    seeds: list[int],
    horizon: int,
    metric: str,
) -> float:
    """Mean of ``metric`` over seeds for a fixed control law. ``inf`` if any seed blew up.

    ``expr`` may be a single expression for ``verb`` or a ``{verb: expression}`` mapping when the
    reference policy governs several instruments at once.
    """
    regent = (MultiScriptedRegent(expr) if isinstance(expr, dict)
              else ScriptedRegent(verb=verb, expr=expr))
    exp = Experiment(
        name=f"calib:{verb}",
        system_factory=system_factory,
        action_interface=action_interface,
        regents={"regent:0": regent},
        objectives={"regent:0": objective},
        schedule=schedule,
        seeds=seeds,
        horizon=horizon,
        hypothesis=Hypothesis(
            id="calibration",
            claim="reference-policy calibration probe (not a scientific claim)",
            baseline="the same policy family under a different regime",
            primary_metric=metric,
        ),
    )
    recs = Runner().run(exp)
    vals = [r.components["regent:0"][metric] for r in recs]
    if any(not math.isfinite(v) for v in vals):
        return float("inf")
    return statistics.mean(vals)


def calibrate(
    family: PolicyFamily,
    *,
    system_factory: Callable[[int], Any],
    action_interface: Any,
    objective: Objective,
    schedule: Schedule,
    seeds: list[int],
    horizon: int,
    metric: str,
) -> CalibrationResult:
    """Exhaustively score every law in ``family`` on this world; return the best (lowest ``metric``).

    ``metric`` is a *loss* component (lower is better) — the same one the headline comparison uses,
    so the reference is optimal for exactly the quantity being reported and cannot be accused of
    having been tuned for something else.
    """
    scored: list[tuple[dict[str, float], float]] = []
    for params in family.combinations():
        expr = family.render_all(params) if family.extra_laws else family.render(params)
        loss = _score_expr(
            expr, verb=family.verb, system_factory=system_factory,
            action_interface=action_interface, objective=objective, schedule=schedule,
            seeds=seeds, horizon=horizon, metric=metric,
        )
        scored.append((params, loss))
    scored.sort(key=lambda kv: kv[1])
    best_params, best_loss = scored[0]
    return CalibrationResult(
        best_params=best_params,
        best_expr=family.render(best_params),
        best_laws=family.render_all(best_params),
        best_loss=best_loss,
        all_losses=scored,
    )


def calibrate_families(
    families: dict[str, PolicyFamily],
    **kw: Any,
) -> tuple[str, CalibrationResult]:
    """Calibrate several policy families and return the globally best ``(family_name, result)``.

    This exists to answer the objection that sinks an otherwise clean result: *of course* a
    code-emitting regent beats a reference that was only allowed to pick parameters inside one
    fixed functional form. If the regent's advantage is really "it wrote a shape the reference
    could not express", then widening the reference's vocabulary until it contains that shape
    should erase the advantage — and if it does not, the advantage is about something else.

    Reporting normalized regret against both a narrow institutional family and a widened one keeps
    those two readings apart instead of letting the narrow number stand in for both.
    """
    results = {name: calibrate(fam, **kw) for name, fam in families.items()}
    best_name = min(results, key=lambda n: results[n].best_loss)
    return best_name, results[best_name]


def normalized_regret(arm_loss: float, frozen_loss: float, oracle_loss: float) -> float:
    """``R = (arm − oracle) / (frozen − oracle)``: 0 = clairvoyant, 1 = never adapted.

    ``R`` is deliberately NOT clipped. R < 0 means the arm beat the calibrated oracle — which is
    informative rather than impossible, because the oracle is only optimal *within its policy
    family*, and a code-emitting regent is not confined to that family (it may use conditionals or
    state history the family cannot express). R > 1 means the arm did actively worse than never
    adapting, the failure mode a governance paper most needs to be able to report.
    """
    denom = frozen_loss - oracle_loss
    if denom == 0 or not math.isfinite(denom):
        return float("nan")
    return (arm_loss - oracle_loss) / denom


def headroom(frozen_loss: float, oracle_loss: float) -> float:
    """``L(frozen)/L(oracle)`` — the factor by which adaptation could reduce post-shock loss."""
    if oracle_loss <= 0 or not math.isfinite(oracle_loss):
        return float("inf")
    return frozen_loss / oracle_loss
