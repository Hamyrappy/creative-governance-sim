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

import subprocess
from typing import Any

from govsim.core.experiment import Experiment, JurisdictionSpec, RunRecord
from govsim.core.harness import Outcome
from govsim.core.regent import Regent


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - environment-dependent
        return "unknown"


def _regent_spec(regent: Regent) -> dict[str, Any]:
    """Best-effort provenance for a regent (type + any pinned model/prompt/expr fields)."""
    spec: dict[str, Any] = {"type": type(regent).__name__}
    for attr in ("model", "prompt_file", "temperature", "verb", "expr", "kp", "kd", "gain"):
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
        system = exp.system_factory(seed)
        regents = exp.regents
        scratch: dict[str, dict] = {rid: {} for rid in regents}

        trajectory: list[dict[str, float]] = []
        llm_io: list[dict[str, Any]] = []
        terminated_at: int | None = None

        for step in range(exp.horizon):
            if exp.schedule.should_decide(step):
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
        )

    def _decision_step(self, exp: Experiment, system: Any, scratch: dict[str, dict]) -> None:
        all_reqs = []
        per_regent: dict[str, tuple] = {}
        for rid, regent in exp.regents.items():
            view = system.observe(rid)
            space = exp.action_interface.action_space(system, rid)
            space = self._scope(space, exp.jurisdictions.get(rid))
            reqs = exp.harness.act(regent, view, space, scratch[rid])
            per_regent[rid] = (view, reqs)
            all_reqs.extend(reqs)

        report = exp.action_interface.apply(all_reqs, system)
        rejected_by_req = {id(req): reason for req, reason in report.rejected}

        for rid, (view, reqs) in per_regent.items():
            errors = [rejected_by_req[id(r)] for r in reqs if id(r) in rejected_by_req]
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
