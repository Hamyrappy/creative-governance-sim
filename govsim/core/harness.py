"""
Harness — the regent's scaffolding, and a first-class *research object* (Goal B: not only
study regents, but learn how to make them more effective).

A ``Harness`` is an ordered stack of independently-toggleable ``HarnessComponent``s decorating
a fixed ``on_observe → propose → on_outcome`` lifecycle around a thin ``Regent``. Components
communicate ONLY through the additive ``scratch`` dict and the ``Outcome`` — never by importing
each other — which is what makes leave-one-out attribution valid (removing X cannot silently
disable Y). The ``enabled`` flag is the *entire* ablation apparatus (H3): an ablation is a
config sweep over which components are on.

Phase 0 ships the machinery with ZERO components; the component taxonomy (TraceFeedback,
EpisodicMemory, RolloutProbe, Critic, Reflection, Evolution, InterRegentComms) lands under
``govsim/harness/`` as results justify each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from govsim.core.action import ActionRequest, ActionSpace, ApplyReport
from govsim.core.system import Observation

# The base "propose" signature a component wraps: (view, space, scratch) -> requests.
ProposeFn = Callable[[Observation, ActionSpace, dict], list[ActionRequest]]


@dataclass
class Outcome:
    """What happened after a decision was applied — fed to ``on_outcome`` for reflection/memory."""

    requests: list[ActionRequest]
    report: ApplyReport | None = None
    error: str | None = None  # e.g. a validation/conservation rejection or runtime error
    metrics: dict[str, float] = field(default_factory=dict)


class HarnessComponent:
    """A pluggable, ablatable piece of scaffolding. All hooks default to no-ops; a component
    overrides only what it needs. ``propose_hook`` wraps the next callable in the chain."""

    name: str = "component"
    enabled: bool = True

    def on_observe(self, view: Observation, space: ActionSpace, scratch: dict) -> None:
        """Inject memory / trace / critic notes / inbox into ``scratch`` before the regent decides."""

    def propose_hook(
        self,
        regent: "Any",
        view: Observation,
        space: ActionSpace,
        scratch: dict,
        base: ProposeFn,
    ) -> list[ActionRequest]:
        """Wrap proposal generation (rollout-probe / critic / evolution). Default: pass through."""
        return base(view, space, scratch)

    def on_outcome(
        self, view: Observation, requests: list[ActionRequest], outcome: Outcome, scratch: dict
    ) -> None:
        """React to the realized outcome (reflect, write memory, post to the inter-regent bus)."""


class Harness:
    def __init__(self, components: list[HarnessComponent] | None = None) -> None:
        self.components: list[HarnessComponent] = list(components or [])

    def _active(self) -> list[HarnessComponent]:
        return [c for c in self.components if c.enabled]  # <- the entire ablation switch

    def act(self, regent: "Any", view: Observation, space: ActionSpace, scratch: dict) -> list[ActionRequest]:
        active = self._active()
        for c in active:
            c.on_observe(view, space, scratch)

        def base(v: Observation, s: ActionSpace, sc: dict) -> list[ActionRequest]:
            return regent.decide(v, s, sc)

        chain: ProposeFn = base
        for c in reversed(active):  # outermost component runs first
            chain = self._wrap(c, regent, chain)
        return chain(view, space, scratch)

    @staticmethod
    def _wrap(component: HarnessComponent, regent: "Any", nxt: ProposeFn) -> ProposeFn:
        def wrapped(v: Observation, s: ActionSpace, sc: dict) -> list[ActionRequest]:
            return component.propose_hook(regent, v, s, sc, nxt)

        return wrapped

    def on_outcome(self, view: Observation, requests: list[ActionRequest], outcome: Outcome, scratch: dict) -> None:
        for c in self._active():
            c.on_outcome(view, requests, outcome, scratch)
