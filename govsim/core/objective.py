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
