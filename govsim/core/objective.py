"""
Objective — pluggable, per-system, per-regent. "Which objective" is an experimental
variable (H4), never a hardcoded config detail: the chosen scalar *becomes* the fitness an
evolutionary/selection harness optimizes, so it is a first-class scientific decision.

Only the ABC lives in core; concrete objectives are *domain-bound plugins* (welfare/Gini
logic cannot be typed without importing domain observables) and live in
``govsim/domains/<domain>/objectives.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

# A trajectory is the sequence of per-step ``metrics()`` dicts produced over a run/rollout.
Trajectory = Sequence[dict[str, float]]


class Objective(ABC):
    @abstractmethod
    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        """The scalar score (higher is better) for ``regent_id`` over ``trajectory``."""

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        """Named sub-metrics always logged alongside the scalar (e.g. Gini, MSE, MSU, cost).

        Distributional metrics belong here even when not in ``evaluate`` so every run records
        them — making "proxy up / true-welfare down" Goodhart episodes detectable (H5).
        """
        return {}

    def describe(self) -> str:
        """A plain-language statement of what is being optimized, for the regent's prompt.

        This is not decoration. A regent that is not told its objective is being scored against
        references that were exhaustively optimized for one — the comparison then measures whether
        the agent guessed the mandate, not whether it governs well. We shipped exactly that for a
        while: the epidemic prompt named the levers and the observables and never mentioned
        infections, cost, or the trade-off between them, while every calibrated reference was
        optimal for ``burden + λ·cost``.

        Concrete objectives should state the quantity, the trade-off, and the weight, because all
        three change what a competent controller would do. The default is deliberately useless so
        that an objective which has not written one is visible in the transcript rather than
        silently absent.
        """
        return f"(objective: {type(self).__name__}; no description provided)"
