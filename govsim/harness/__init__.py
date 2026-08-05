"""
govsim.harness — the pluggable, ablatable scaffolding components (Goal B: make controllers better).

A ``Harness`` (in ``govsim.core``) is an ordered stack of these ``HarnessComponent``s decorating a
thin ``Regent``. Components talk ONLY through the additive ``scratch`` dict + the ``Outcome`` — never
by importing each other — so leave-one-out attribution is valid (the ``enabled`` flag is the whole
ablation apparatus, H3).

Rollout-FREE components ship first (doc-09 §5.2): ``TraceFeedback`` (the failure channel),
``OutcomeFeedback`` (the performance channel) and ``EpisodicMemory``. Rollout-dependent ones
(``RolloutProbe``, ``Evolution``) are gated behind the ``RollableSystem``/Generator precondition and
land once a result justifies them.

The three rollout-free components are deliberately *different kinds of information*, not three
flavours of the same one — errors, realized performance, and retrieved precedent. That is what makes
a factorial ablation over them worth running: they can substitute for or interfere with each other,
and a one-at-a-time sweep would not show it.
"""

from govsim.harness.components import (
    ContextualOutcomeFeedback,
    Critic,
    ContrastiveMemory,
    DistantMemory,
    ForeignMemory,
    UnscoredMemory,
    EpisodicMemory,
    OutcomeFeedback,
    RolloutProbe,
    TraceFeedback,
)

__all__ = ["TraceFeedback", "OutcomeFeedback", "ContextualOutcomeFeedback",
           "EpisodicMemory", "ContrastiveMemory", "DistantMemory", "UnscoredMemory", "ForeignMemory", "RolloutProbe", "Critic"]
