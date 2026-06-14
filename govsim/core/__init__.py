"""
govsim.core — the domain-agnostic seams of the experiment machine.

This package is the top of the architecture described in ``agents/09-grand-plan.md``:
an experiment machine for LLM "regents" (controllers) that try to control complex
systems (economies are *domain #1*, not the framework). Nothing in this package may
encode a domain noun (no "economy", "tax", "ledger", ...) — that lives in ``govsim.domains.*``.
A CI test enforces this (see ``tests/``).

The six seams:
  - System / RollableSystem   (a controllable process; acting is NOT its concern)
  - ActionInterface           (the ONLY domain-coupled seam: what a regent may do + invariants)
  - Regent                    (a controller; domain-blind)
  - Harness / HarnessComponent(the regent's pluggable, ablatable scaffolding — Goal B)
  - Objective                 (pluggable, per-system, per-regent)
  - Schedule                  (when regents decide)

plus the LLM seam (``govsim.core.llm``): an OpenAI-*compatible* client (any endpoint /
provider / local model — configured by ``base_url`` + ``model``, never hardcoded) wrapped
by a cache/replay tape so an LLM-in-the-loop experiment is reproducible and cheap to re-run.
"""

from govsim.core.system import System, RollableSystem, Observation, StepInfo
from govsim.core.action import (
    ActionInterface,
    ActionRequest,
    ActionSpace,
    VerbSpec,
    ValidationResult,
    ApplyReport,
)
from govsim.core.regent import Regent, StaticRegent, ScriptedRegent
from govsim.core.objective import Objective
from govsim.core.harness import Harness, HarnessComponent, Outcome
from govsim.core.schedule import Schedule, EveryN, AtSteps
from govsim.core.experiment import (
    Experiment,
    RunRecord,
    Hypothesis,
    CreativityMetric,
    JurisdictionSpec,
)
from govsim.core.runner import Runner
from govsim.core.result_store import ResultStore
from govsim.core.rollout import rollout, RolloutContext

__all__ = [
    "System",
    "RollableSystem",
    "Observation",
    "StepInfo",
    "ActionInterface",
    "ActionRequest",
    "ActionSpace",
    "VerbSpec",
    "ValidationResult",
    "ApplyReport",
    "Regent",
    "StaticRegent",
    "ScriptedRegent",
    "Objective",
    "Harness",
    "HarnessComponent",
    "Outcome",
    "Schedule",
    "EveryN",
    "AtSteps",
    "Experiment",
    "RunRecord",
    "Hypothesis",
    "CreativityMetric",
    "JurisdictionSpec",
    "Runner",
    "ResultStore",
    "rollout",
    "RolloutContext",
]
