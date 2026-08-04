"""
rollout — the domain-neutral counterfactual-evaluation primitive (doc-09 §2.1 / §5.2).

A rollout clones a ``RollableSystem``, installs a candidate action batch through the (domain-routed)
``ActionInterface``, advances ``horizon`` ticks, and scores the resulting trajectory with an
``Objective``. It is the shared fitness oracle the rollout-dependent harness components
(``RolloutProbe``, later ``Evolution``) and the ``OPRORegent`` baseline use. It mutates only the
clone — never the live system — so a regent that scores candidates this way still holds NO write
handle to real state (the inversion of doc-09 §1.2 is preserved).

Soundness (doc-08 HARD rule): the clone carries the system's own ``np.random.Generator``, so a
single rollout (``seed=None``) is a FAITHFUL continuation of the same stochastic stream. Passing an
explicit ``seed`` instead resamples an *independent future from the SAME current state* — a
deliberate Monte-Carlo over futures for variance-aware selection (``mean − λ·std``), not the
re-seeded-restart bug doc-08 warned about (state is preserved; only the future noise stream is drawn
fresh). This lives in ``core`` because nothing here names a domain — it is pure clone/step/score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from govsim.core.action import ActionInterface, ActionRequest
from govsim.core.objective import Objective
from govsim.core.system import RollableSystem


def rollout(
    system: RollableSystem,
    action_interface: ActionInterface,
    objective: Objective,
    requests: Sequence[ActionRequest],
    horizon: int,
    *,
    regent_id: str = "regent:0",
    seed: int | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Clone ``system``, install ``requests``, step ``horizon`` ticks, return ``(score, trajectory)``.

    ``seed=None`` continues the cloned Generator faithfully; an explicit ``seed`` resamples an
    independent future from the current state (for variance-aware selection over seeds).
    """
    clone = system.clone()
    if seed is not None:
        clone.rng = np.random.default_rng(seed)
    if requests:
        action_interface.apply(list(requests), clone)
    # Record ONLY post-step rows — matching the real Runner, which appends metrics after each
    # ``system.step()``. Seeding the trajectory with a pre-step row (where the freshly-installed
    # lever has not yet been evaluated, so current_u is still 0) would score the objective on a
    # different slice than the realized run this oracle is meant to predict.
    traj: list[dict[str, float]] = []
    for _ in range(horizon):
        info = clone.step()
        traj.append(clone.metrics())
        if getattr(info, "terminated", False):
            break
    return objective.evaluate(traj, regent_id), traj


@dataclass(frozen=True)
class RolloutContext:
    """A handle the ``Runner`` injects into a regent's ``scratch`` on decision steps, *only when the
    system is a ``RollableSystem``* (the rollout-soundness precondition gate). Ephemeral: set right
    before ``propose`` and popped right after, so no live system reference is persisted/serialized.
    Rollout-dependent components/regents read ``scratch["_rollout"]``; its absence IS the gate.
    """

    system: RollableSystem
    action_interface: ActionInterface
    objective: Objective
    regent_id: str = "regent:0"

    def rollout(self, requests: Sequence[ActionRequest], horizon: int, seed: int | None = None) -> float:
        return rollout(self.system, self.action_interface, self.objective, requests, horizon,
                       regent_id=self.regent_id, seed=seed)[0]

    def score(self, requests: Sequence[ActionRequest], horizon: int, seeds: Sequence[int]) -> list[float]:
        """Per-future scores of a candidate batch over ``seeds`` (paired across candidates upstream)."""
        return [self.rollout(requests, horizon, seed=s) for s in seeds]
