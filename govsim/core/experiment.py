"""
The experiment spine — the WHAT-first gated experiment description + the data model that shapes
everything downstream (doc-08 §3.2 / doc-09 §2.4).

An ``Experiment`` binds a system factory, an ``ActionInterface``, one-or-more ``Regent``s (N=1 is
the trivial case), their ``Objective``s, a ``Harness``, a ``Schedule``, seeds, a horizon — and,
as **required gate fields**, a ``Hypothesis`` (claim + NAMED baseline + primary metric). The
``Runner`` refuses to run if the gate is unset: deciding WHAT (which hypothesis, which baseline)
is an executable precondition that cannot rot, not a discipline that erodes (doc-09 §1.3).

``regent_id`` / ``polity_id`` are threaded from day one even at N=1 (filled ``regent:0`` /
``polity:0``); widening the loop to N>1 later then touches nothing structural (doc-09 §4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from govsim.core.action import ActionInterface
from govsim.core.harness import Harness
from govsim.core.objective import Objective
from govsim.core.regent import Regent
from govsim.core.schedule import Schedule
from govsim.core.system import System


@dataclass(frozen=True)
class Hypothesis:
    """A falsifiable claim with a NAMED rival — the WHAT-first gate (doc-08 §3.1).

    Without ``claim`` + ``baseline`` the experiment has no measurable dependent variable, so the
    ``Runner`` will not run. ``primary_metric`` names the pre-registered series/component the
    comparison is decided on.
    """

    id: str
    claim: str
    baseline: str
    primary_metric: str
    falsification: str = ""  # what observation would falsify the claim


@dataclass(frozen=True)
class CreativityMetric:
    """The (domain-scoped) creativity construct, or ``None`` to honestly drop the word (doc-09 §6.3).

    ``kind`` ∈ {generalization_gap, functional_novelty, policy_innovation_score, qd_diversity}.
    """

    name: str
    kind: str
    description: str = ""


@dataclass(frozen=True)
class JurisdictionSpec:
    """What a regent may govern: a partition of the already-advertised action space (doc-09 §4.1).

    ``verbs == "*"`` = all advertised verbs; ``observable_scope is None`` = the full view. Threaded
    now (N=1 uses the permissive default); the N>1 partitioning logic lands in Phase 5.
    """

    verbs: frozenset[str] | str = "*"
    observable_scope: frozenset[str] | None = None


@dataclass
class Experiment:
    """A runnable experiment. ``hypothesis`` is a required gate; ``creativity_metric`` may be None."""

    name: str
    system_factory: Callable[[int], System]  # seed -> a freshly reset System
    action_interface: ActionInterface
    regents: dict[str, Regent]
    objectives: dict[str, Objective]
    schedule: Schedule
    seeds: list[int]
    horizon: int
    # ---- WHAT-FIRST GATES (Runner refuses to run if hypothesis/baseline absent) ----
    hypothesis: Hypothesis
    creativity_metric: CreativityMetric | None = None
    # ---- multi-regent hooks (N=1 trivial) ----
    harness: Harness = field(default_factory=Harness)
    jurisdictions: dict[str, JurisdictionSpec] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """One (experiment, seed) result — the unit the ResultStore aggregates over.

    Heavy series (``metrics_series``) and raw LLM I/O (``llm_io``) are persisted as side
    artifacts by the store; the row carries the keys, scores, components and provenance needed for
    cross-run queries and reproducibility (doc-09 §2.4).
    """

    experiment: str
    system_id: str
    action_interface_id: str
    schedule_id: str
    hypothesis_id: str
    seed: int
    horizon: int
    git_commit: str
    regent_specs: dict[str, dict[str, Any]]
    objective_ids: dict[str, str]
    score: dict[str, float]
    components: dict[str, dict[str, float]]
    metrics_series: list[dict[str, float]] = field(default_factory=list)
    llm_io: list[dict[str, Any]] = field(default_factory=list)
    token_cost: float = 0.0
    terminated_at_step: int | None = None
    creativity_metric: str | None = None
    created_at: float = field(default_factory=time.time)
