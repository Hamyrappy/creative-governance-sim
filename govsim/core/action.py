"""
ActionInterface — THE keystone seam, and the *only* domain-coupled one.

It is the per-domain vocabulary of *what a controller may do* plus *the domain invariants*.
economy-vs-epidemic-vs-company differ HERE and nowhere above. Two siblings ship:
  - ``ScalarLeverInterface`` (``govsim.domains.scalar``): clip a sandboxed expression into a
    bounded lever; NO ledger. Used by the cubic, SIR, company systems.
  - ``EconomyActionInterface`` (``govsim.domains.economy``, later): the conserved Effect/Ledger/
    Mediator/Institutions "Chancery" of ``agents/07-governance-interface.md`` lives *entirely
    inside* this one implementation.

The regent above only ever holds an ``Observation`` (read) and an ``ActionSpace`` (what it may
do); it returns ``ActionRequest``s. System state is mutated *only* inside ``apply`` — which is why
conservation ("no resources from thin air") is a structural consequence of the layering, not a
special property of the governance kernel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from govsim.core.system import System


@dataclass(frozen=True)
class ActionRequest:
    """One thing a regent wants to do, attributed to a regent.

    ``payload`` is a tagged union with exactly one arm — the *one* place the core action type
    is domain-aware (see grand-plan §3.2):
      - ``{"expr": "<python expression over the view's context_vars>"}``  (scalar levers)
      - ``{"value": <number>}``                                          (a direct set)
      - ``{"effect": {...}}``                                            (opaque to the core;
            only an ``ActionInterface`` like ``EconomyActionInterface`` interprets it)
    """

    regent_id: str
    verb: str  # a lever name ("set_lockdown") or a domain verb ("Transfer")
    payload: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerbSpec:
    """One thing the regent *may* do, as advertised by an ``ActionSpace``."""

    name: str
    value_range: tuple[float, float] | None = None
    value_type: str = "float"
    description: str = ""


@dataclass(frozen=True)
class ActionSpace:
    """The machine- and prompt-renderable description of what a (specific) regent may do now."""

    verbs: list[VerbSpec]
    context_vars: list[str]  # identifiers a payload ``expr`` may reference (== Observation.vars keys)

    def verb_names(self) -> list[str]:
        return [v.name for v in self.verbs]

    def as_tools(self) -> list[dict[str, Any]]:
        """Render the verbs as OpenAI-*compatible* function tools (one tool per verb).

        Default schema: each verb takes an ``expr`` (a single Python expression over
        ``context_vars``) and an optional ``reasoning``. A richer domain interface (e.g. the
        economy plugin) may override ``as_tools`` with structured ``effect`` parameters.
        """
        ctx = ", ".join(self.context_vars)
        tools: list[dict[str, Any]] = []
        for v in self.verbs:
            rng = f" Result is clipped to {v.value_range}." if v.value_range else ""
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": v.name,
                        "description": (v.description or f"Set lever '{v.name}'.")
                        + f" Provide ONE Python expression over: [{ctx}]." + rng,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expr": {
                                    "type": "string",
                                    "description": "A single Python expression over the allowed context variables.",
                                },
                                "reasoning": {"type": "string", "description": "Brief justification."},
                            },
                            "required": ["expr"],
                        },
                    },
                }
            )
        return tools


@dataclass(frozen=True)
class ValidationResult:
    """Structured reject-WITH-FEEDBACK (doc-08 dead-end fix: never a silent ``None``)."""

    ok: bool
    feedback: str = ""  # on failure: why, so the regent can correct on the next turn
    compiled: Any = None  # on success: an opaque compiled artifact the interface caches


@dataclass(frozen=True)
class ApplyReport:
    """What ``apply`` did with a batch of requests — fed back to each regent."""

    applied: list[ActionRequest] = field(default_factory=list)
    rejected: list[tuple[ActionRequest, str]] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


class ActionInterface(ABC):
    """Per-domain action vocabulary + invariants. The only domain-coupled seam."""

    @abstractmethod
    def action_space(self, system: System, regent_id: str) -> ActionSpace:
        """The verbs + context this regent may use now (jurisdiction-scoped)."""

    @abstractmethod
    def validate(self, req: ActionRequest, system: System, regent_id: str) -> ValidationResult:
        """Check a request against the domain rules; return ok+compiled or reject+feedback."""

    @abstractmethod
    def apply(self, reqs: list[ActionRequest], system: System) -> ApplyReport:
        """Apply a (possibly multi-regent) batch atomically, through the domain invariants.

        This is the single mutation checkpoint: lever-clipping (scalar) or conservation/solvency
        (economy) is enforced over the *merged* batch, so guarantees hold regardless of regent count.
        """
