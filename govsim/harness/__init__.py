"""
govsim.harness — the pluggable, ablatable scaffolding components (Goal B: make controllers better).

A ``Harness`` (in ``govsim.core``) is an ordered stack of these ``HarnessComponent``s decorating a
thin ``Regent``. Components talk ONLY through the additive ``scratch`` dict + the ``Outcome`` — never
by importing each other — so leave-one-out attribution is valid (the ``enabled`` flag is the whole
ablation apparatus, H3).

Rollout-FREE components ship first (doc-09 §5.2): ``TraceFeedback`` (the cheapest upgrade) and
``EpisodicMemory``. Rollout-dependent ones (``RolloutProbe``, ``Evolution``) are gated behind the
``RollableSystem``/Generator precondition and land once a result justifies them.
"""

from govsim.harness.components import TraceFeedback, EpisodicMemory, RolloutProbe

__all__ = ["TraceFeedback", "EpisodicMemory", "RolloutProbe"]
