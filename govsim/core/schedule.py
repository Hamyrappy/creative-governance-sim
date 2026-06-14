"""
Schedule — when regents decide. Extracted from the tangled inline condition in the old
``simulation.py`` (``(current_step + 1) % freq == 0 and current_step+1 >= 1 and
current_step+1 < total_steps``), which conflated frequency with edge-trimming and silently
suppressed the decision on the final boundary. Here it is one clean, testable object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable


class Schedule(ABC):
    @abstractmethod
    def should_decide(self, step: int) -> bool:
        """Whether a regent decision happens at this (0-based) tick."""


class EveryN(Schedule):
    """Decide at steps 0, n, 2n, ... (``n == 1`` => every step)."""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("EveryN requires n >= 1")
        self.n = n

    def should_decide(self, step: int) -> bool:
        return step % self.n == 0


class AtSteps(Schedule):
    """Decide only at an explicit set of steps (e.g. the thesis's 200/600/800 interventions)."""

    def __init__(self, steps: Iterable[int]) -> None:
        self.steps = frozenset(steps)

    def should_decide(self, step: int) -> bool:
        return step in self.steps
