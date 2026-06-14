"""
System — a controllable complex process. NOT specific to economies.

The load-bearing inversion (vs the old ``BaseEconomicSystem``, which fused *being a
system* with *being acted upon*): a ``System`` only advances dynamics and exposes a
read-only, jurisdiction-scoped view. *Acting* on the system is the job of an
``ActionInterface`` (``govsim.core.action``) — the only domain-coupled seam. A regent
therefore never holds a write handle to system state; it sees an ``Observation`` and an
``ActionSpace`` and returns ``ActionRequest``s.

Reproducibility (doc-08, HARD rule): a concrete system owns a ``numpy.random.Generator``
seeded in ``reset(seed)``; it must never draw from the module-global ``random`` /
``numpy.random`` state. ``clone()`` (on ``RollableSystem``) carries that Generator so a
counterfactual rollout is a *faithful* continuation, not a re-seeded restart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    """A read-only, jurisdiction-scoped snapshot the regent sees on a decision step.

    ``vars`` are the only identifiers a sandboxed policy expression may reference for this
    viewer (the single source of truth for the sandbox whitelist + the prompt's context).
    """

    vars: dict[str, float]
    scope: str = "regent:0"  # the viewer/regent id this view is scoped to (jurisdiction / info asymmetry)
    t: int = 0


@dataclass(frozen=True)
class StepInfo:
    """What one ``System.step()`` reports back to the runner."""

    terminated: bool = False  # the process reached a terminal/absorbing state
    truncated: bool = False  # cut off by an external limit (e.g. horizon), not termination
    info: dict[str, Any] = field(default_factory=dict)


class System(ABC):
    """Any controllable process: ``reset → (observe / step)*``.

    Concrete subclasses set ``self.rng`` (a ``numpy.random.Generator``) in ``reset``.
    The annotation is a string (``from __future__ import annotations``) so importing this
    module does not require numpy.
    """

    rng: "Any"  # numpy.random.Generator on concrete systems; not imported here on purpose

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Reset to the initial state and (re)seed ``self.rng`` deterministically from ``seed``."""

    @abstractmethod
    def step(self) -> StepInfo:
        """Advance exactly one tick. Any installed action is re-evaluated by the
        ``ActionInterface`` against this system *before* dynamics advance."""

    @abstractmethod
    def observe(self, viewer_id: str = "regent:0") -> Observation:
        """Return the read-only view for ``viewer_id`` (jurisdiction-scoped; identical for all
        viewers when there is a single full-jurisdiction regent)."""

    @property
    @abstractmethod
    def time(self) -> int:
        """The current integer tick."""

    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """Flat ``name -> float`` metrics for objectives and logging (a superset of any
        single viewer's ``Observation.vars``)."""


class RollableSystem(System):
    """A ``System`` that can be cloned for counterfactual rollout.

    ``clone()`` is a *capability*, deliberately not on the base ``System``: a live external
    world (a real datacenter, a remote API) is a controllable system but cannot be cloned,
    and rollout-based harnesses simply do not apply to it. Rollout components require this
    subtype; a precondition test gates them on it.
    """

    @abstractmethod
    def clone(self) -> "RollableSystem":
        """Return a deep, independent copy whose ``rng`` is carried so a rollout continues
        the same stochastic stream (no re-seed restart)."""
