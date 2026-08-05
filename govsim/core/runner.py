"""
Runner — the paired-seed orchestration loop, extracted from the tangled ``simulation.py``.

It enforces the WHAT-first gate (no hypothesis+baseline ⇒ no run), then for each seed runs the
decision loop of doc-09 §4.2:

    for step in range(horizon):
        if schedule.should_decide(step):
            for each regent: observe (jurisdiction-scoped) → action_space → harness.act → reqs
            report = action_interface.apply(all_reqs, system)   # SINGLE atomic mutation
            feed (applied/rejected + reason) back to each regent's harness (on_outcome)
        system.step()          # re-evaluates installed actions, then advances dynamics
        record system.metrics()

The single ``apply([all reqs], system)`` is the one invariant checkpoint, so guarantees hold
regardless of regent count. With a ``CachingReplayClient`` in replay mode, the only varying source
across a paired comparison is the world seed (the basis of the H3 ablation / stats protocol).
"""

from __future__ import annotations

import functools
import subprocess
from typing import Any

from govsim.core.experiment import Experiment, JurisdictionSpec, RunRecord
from govsim.core.harness import Outcome
from govsim.core.regent import Regent
from govsim.core.rollout import RolloutContext
from govsim.core.system import RollableSystem


@functools.lru_cache(maxsize=1)
def _git_commit() -> str:
    """The provenance stamp for a RunRecord, resolved once per process.

    Cached because a calibration sweep is thousands of short runs, and spawning a git subprocess per
    run made the sweep spend most of its wall-clock in process creation rather than simulation. The
    commit cannot change under a running process in any way we would want to record differently, so
    there is nothing to lose by resolving it once.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - environment-dependent
        return "unknown"


#: Regent attributes worth persisting as provenance. ``laws``/``pre_expr``/``post_expr`` are here
#: because the anchor-consistency gate compares a stored reference run against the calibration
#: artifact, and it can only do that if the enacted policy was actually recorded. It was not:
#: switching the references to ``MultiScriptedRegent`` (which holds ``laws``, not ``expr``) left the
#: gate reading a field that no longer existed, so it skipped every check and passed vacuously —
#: a gate that had caught a real stale-anchor bug days earlier.
_SPEC_ATTRS = ("model", "prompt_file", "temperature", "verb", "expr", "laws",
               "pre_expr", "post_expr", "switch_step", "kp", "kd", "gain")


def _regent_spec(regent: Regent) -> dict[str, Any]:
    """Best-effort provenance for a regent (type + any pinned model/prompt/policy fields)."""
    spec: dict[str, Any] = {"type": type(regent).__name__}
    for attr in _SPEC_ATTRS:
        if hasattr(regent, attr):
            spec[attr] = getattr(regent, attr)
    return spec


class Runner:
    """Runs an ``Experiment`` and (optionally) persists each ``RunRecord`` to a ``ResultStore``."""

    def __init__(self, result_store: Any = None) -> None:
        self.result_store = result_store

    def run(self, exp: Experiment) -> list[RunRecord]:
        # ---- WHAT-first gate (doc-09 §1.3): an executable precondition, not a discipline ----
        if exp.hypothesis is None or not exp.hypothesis.claim or not exp.hypothesis.baseline:
            raise ValueError(
                "WHAT-first gate (doc-08 §3.1): an Experiment needs a Hypothesis with a "
                "non-empty claim AND a named baseline before it may run."
            )
        records = [self._run_seed(exp, seed) for seed in exp.seeds]
        if self.result_store is not None:
            for rec in records:
                self.result_store.add(rec)
        return records

    def _run_seed(self, exp: Experiment, seed: int) -> RunRecord:
        # Every seed runs against the SAME Experiment object, so any component that accumulates
        # would otherwise carry one run's history into the next — episodes retrieved from other
        # world realisations, outcome scores earned in runs this world never saw. A paired seed
        # design assumes independent draws; this is what makes that true.
        exp.harness.reset()
        system = exp.system_factory(seed)
        regents = exp.regents
        scratch: dict[str, dict] = {rid: {} for rid in regents}

        trajectory: list[dict[str, float]] = []
        llm_io: list[dict[str, Any]] = []
        terminated_at: int | None = None
        last_decision_len = {rid: 0 for rid in regents}

        for step in range(exp.horizon):
            if exp.schedule.should_decide(step):
                # Realized-feedback channel: the Objective over the window since this regent's LAST
                # decision (i.e. the realized score of the law it then deployed). Lets a learning
                # regent (e.g. OPRO in 'realized' mode) score its own past actions without a model.
                for rid in regents:
                    window = trajectory[last_decision_len[rid]:]
                    if window:
                        scratch[rid]["_last_realized_score"] = exp.objectives[rid].evaluate(window, rid)
                    last_decision_len[rid] = len(trajectory)
                self._decision_step(exp, system, scratch)
            info = system.step()
            trajectory.append(system.metrics())
            if info.terminated:
                terminated_at = step
                break

        # collect any LLM I/O the regents recorded into their scratch
        token_cost = 0.0
        for rid in regents:
            for call in scratch[rid].get("_llm_calls", []):
                llm_io.append({"regent_id": rid, **call})
                token_cost += float(call.get("cost_usd") or 0.0)

        score = {rid: exp.objectives[rid].evaluate(trajectory, rid) for rid in regents}
        components = {rid: exp.objectives[rid].components(trajectory, rid) for rid in regents}

        return RunRecord(
            experiment=exp.name,
            system_id=type(system).__name__,
            action_interface_id=type(exp.action_interface).__name__,
            schedule_id=type(exp.schedule).__name__,
            hypothesis_id=exp.hypothesis.id,
            seed=seed,
            horizon=exp.horizon,
            git_commit=_git_commit(),
            regent_specs={rid: _regent_spec(r) for rid, r in regents.items()},
            objective_ids={rid: type(o).__name__ for rid, o in exp.objectives.items()},
            score=score,
            components=components,
            metrics_series=trajectory,
            llm_io=llm_io,
            token_cost=token_cost,
            terminated_at_step=terminated_at,
            creativity_metric=exp.creativity_metric.name if exp.creativity_metric else None,
            harness_components=[
                {"name": getattr(c, "name", type(c).__name__), "enabled": bool(getattr(c, "enabled", True))}
                for c in exp.harness.components
            ],
        )

    def _decision_step(self, exp: Experiment, system: Any, scratch: dict[str, dict]) -> None:
        all_reqs = []
        per_regent: dict[str, tuple] = {}
        rollable = isinstance(system, RollableSystem)
        for rid, regent in exp.regents.items():
            # The regent's MANDATE. Publishing it is not a courtesy: the calibrated references it
            # is compared against were exhaustively optimized for this objective, so a regent that
            # has to guess it is being scored on a different task from its rivals.
            scratch[rid]["_objective"] = exp.objectives[rid].describe()
            view = system.observe(rid)
            space = exp.action_interface.action_space(system, rid)
            space = self._scope(space, exp.jurisdictions.get(rid))
            # Inject the rollout context ONLY when the system is rollable (the soundness gate):
            # rollout-dependent components/regents read scratch["_rollout"]; its absence is the gate.
            if rollable:
                scratch[rid]["_rollout"] = RolloutContext(system, exp.action_interface, exp.objectives[rid], rid)
            reqs = exp.harness.act(regent, view, space, scratch[rid])
            scratch[rid].pop("_rollout", None)  # ephemeral: never persist a live system handle
            per_regent[rid] = (view, reqs)
            all_reqs.extend(reqs)

        report = exp.action_interface.apply(all_reqs, system)
        rejected_by_req = {id(req): reason for req, reason in report.rejected}

        for rid, (view, reqs) in per_regent.items():
            errors = [rejected_by_req[id(r)] for r in reqs if id(r) in rejected_by_req]
            if not reqs:
                # A decision that produced ZERO requests (LLM emitted no parseable tool-call / empty
                # or truncated reply) is otherwise a silent no-op the TraceFeedback loop never sees —
                # yet it is the single most common LLM failure mode. Surface it as an error so the
                # corrective channel fires next turn. (A deliberately-passive regent has no harness
                # reading this, so it stays harmless there.)
                errors.append("no action produced: empty or unparseable regent reply; nothing was applied")
            outcome = Outcome(
                requests=reqs,
                report=report,
                error="; ".join(errors) if errors else None,
                metrics=system.metrics(),
            )
            exp.harness.on_outcome(view, reqs, outcome, scratch[rid])

    @staticmethod
    def _scope(space, jurisdiction: JurisdictionSpec | None):
        """Restrict the advertised verbs to this regent's jurisdiction (a partition of the space).

        N=1 / no jurisdiction ⇒ the full space. The partition logic is exercised by the Phase-5
        N=2 smoke test; here it is a no-op for the permissive default."""
        if jurisdiction is None or jurisdiction.verbs == "*":
            return space
        allowed = jurisdiction.verbs
        verbs = [v for v in space.verbs if v.name in allowed]
        return type(space)(verbs=verbs, context_vars=space.context_vars)
