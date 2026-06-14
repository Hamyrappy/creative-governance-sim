"""
Regent — a controller. Domain-blind: it sees an ``Observation`` + an ``ActionSpace`` and
returns ``ActionRequest``s. No regent imports a domain context class (this is what killed
the old ``IntelligentLLMAgent``'s hard binding to ``LinearSystemAgentContext``).

This module ships the trivial baselines that need no LLM. The ``LLMRegent`` and the
control baselines (``PIDRegent`` / ``LQRRegent`` / ``OPRORegent``) arrive in Phase 1 under
``govsim/regents/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.system import Observation

# A per-regent mutable scratchpad that harness components read/write (memory, trace, inbox, ...).
Scratch = dict[str, Any]


class Regent(ABC):
    def __init__(self, id: str = "regent:0") -> None:
        self.id = id

    @abstractmethod
    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        """Return zero or more action requests for this decision step."""


class StaticRegent(Regent):
    """Passive baseline: never proposes anything (the no-government scenario)."""

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        return []


class ScriptedRegent(Regent):
    """Deterministic baseline: emit a fixed expression for a fixed verb each decision step.

    Used for the N=1 golden-master test (reproduce a pre-rewrite run with no live LLM) and as
    a quick smoke regent. Mirrors the old ``TestPoliciesAgent`` but on the new seam. (Named
    ``ScriptedRegent``, not ``TestRegent``, to avoid pytest's ``Test*`` collection heuristic.)
    """

    def __init__(self, verb: str, expr: str, id: str = "regent:0") -> None:
        super().__init__(id)
        self.verb = verb
        self.expr = expr

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        return [ActionRequest(regent_id=self.id, verb=self.verb, payload={"expr": self.expr})]
