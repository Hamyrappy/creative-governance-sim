"""
Harness components.

``TraceFeedback`` and ``EpisodicMemory`` are rollout-FREE: they write into ``scratch`` keys the LLM
prompt assembler reads (``trace``, ``memory``), upgrading an ``LLMRegent`` without the regent knowing
they exist — and toggling ``enabled`` (the H3 ablation switch) cleanly removes their effect. Neither
needs ``clone()``/rollout, so they are sound on any system.

``RolloutProbe`` is rollout-DEPENDENT (doc-09 §5.2): it scores candidate laws by cloning the system
and rolling it forward, so it is gated behind the ``RollableSystem`` precondition — it is a pure
pass-through unless the ``Runner`` injected a ``RolloutContext`` into ``scratch`` (which it does only
for rollable systems). This is why the rollout-free components ship first and this one is gated.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.harness import HarnessComponent, Outcome
from govsim.core.system import Observation


def _extract_json_obj(text: str) -> dict:
    """Best-effort parse of a JSON object from a possibly fenced/surrounded reply (self-contained,
    so the harness layer does not import the regents layer)."""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start: end + 1] if start != -1 and end > start else None
    if candidate is None:
        return {}
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


class TraceFeedback(HarnessComponent):
    """Feed the last action's failure (sandbox/validate rejection, conservation reject, runtime
    error) back into the next prompt as a corrective note — the cheapest upgrade (doc-08 §6).

    Flow: ``on_outcome`` stashes ``Outcome.error``; the next ``on_observe`` moves it to
    ``scratch["trace"]`` (and clears it when there was no error), where the prompt assembler shows it.
    """

    name = "trace_feedback"

    def on_observe(self, view: Observation, space: ActionSpace, scratch: dict) -> None:
        pending = scratch.pop("_pending_trace", None)
        if pending:
            scratch["trace"] = pending
        else:
            scratch.pop("trace", None)

    def on_outcome(self, view: Observation, requests: list[ActionRequest], outcome: Outcome, scratch: dict) -> None:
        if outcome.error:
            scratch["_pending_trace"] = outcome.error


class EpisodicMemory(HarnessComponent):
    """Retrieve the k most similar past (state, action, outcome) episodes by current metrics and
    inject them as ``scratch["memory"]`` — the cheap, RAG-style learning baseline (doc-08 §6).

    Similarity is Euclidean over the numeric keys shared between the current view and a stored
    state. Append-only; no rollout, no clone.
    """

    name = "episodic_memory"

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.episodes: list[dict] = []

    def on_observe(self, view: Observation, space: ActionSpace, scratch: dict) -> None:
        if not self.episodes:
            return
        ranked = sorted(self.episodes, key=lambda ep: self._distance(view.vars, ep["state"]))
        lines = []
        for ep in ranked[: self.k]:
            state = ", ".join(f"{k}={v:.4g}" for k, v in ep["state"].items())
            acts = "; ".join(f"{a['verb']}:{a['expr']}" for a in ep["actions"])
            lines.append(f"- when [{state}] you did [{acts}] → score≈{ep['score']:.4g}")
        scratch["memory"] = "\n".join(lines)

    def on_outcome(self, view: Observation, requests: list[ActionRequest], outcome: Outcome, scratch: dict) -> None:
        actions = [{"verb": r.verb, "expr": str(r.payload.get("expr", r.payload))} for r in requests]
        score = sum(v for v in outcome.metrics.values() if isinstance(v, (int, float)))
        self.episodes.append({"state": dict(view.vars), "actions": actions, "score": score})

    @staticmethod
    def _distance(a: dict[str, float], b: dict[str, float]) -> float:
        shared = set(a) & set(b)
        if not shared:
            return math.inf
        return math.sqrt(sum((a[k] - b[k]) ** 2 for k in shared))


class RolloutProbe(HarnessComponent):
    """Variance-aware rollout selection (H2: experimentation > reasoning).

    Ask the regent for up to ``n_candidates`` candidate laws (deduped), score each by cloning the
    system and rolling it forward ``horizon`` ticks over a set of future ``seeds``, and keep the
    candidate maximizing ``mean − λ·std`` (worst-case-aware; never select on the mean alone —
    stats-protocol). Diversity is requested via ``scratch["_probe_seed_offset"]``: regents that honor
    it (``LLMRegent`` perturbs its sampling seed) yield distinct candidates; deterministic baselines
    ignore it, so the probe degrades to robustly *scoring* the single candidate (still useful, never
    harmful).

    ROLLOUT-DEPENDENT: a pure pass-through unless ``scratch["_rollout"]`` (a ``RolloutContext``) was
    injected by the Runner — i.e. unless the System is a ``RollableSystem``. That absence is the
    precondition gate (doc-09 §5.2): on a non-rollable world this component does nothing.
    """

    name = "rollout_probe"

    def __init__(self, n_candidates: int = 4, horizon: int = 20, seeds: tuple[int, ...] = (0, 1, 2, 3),
                 lam: float = 0.5) -> None:
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.seeds = tuple(seeds)
        self.lam = lam

    def propose_hook(self, regent, view, space, scratch, base):
        ctx = scratch.get("_rollout")
        if ctx is None:  # precondition not met (non-RollableSystem) ⇒ pure pass-through
            return base(view, space, scratch)

        candidates: list[list[ActionRequest]] = []
        seen: set[tuple] = set()
        for i in range(self.n_candidates):
            scratch["_probe_seed_offset"] = i  # regents that support it diversify their proposal
            reqs = base(view, space, scratch)
            if not reqs:
                continue
            key = tuple((r.verb, str(r.payload)) for r in reqs)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(reqs)
        scratch.pop("_probe_seed_offset", None)

        if not candidates:
            return []

        best, best_obj, scored = candidates[0], float("-inf"), []
        for reqs in candidates:
            futures = ctx.score(reqs, self.horizon, self.seeds)
            obj = statistics.fmean(futures) - self.lam * (statistics.pstdev(futures) if len(futures) > 1 else 0.0)
            scored.append(obj)
            if obj > best_obj:
                best_obj, best = obj, reqs
        scratch["rollout_probe"] = {"n_candidates": len(candidates), "best_obj": best_obj, "scores": scored}
        return best


class Critic(HarnessComponent):
    """A second LLM audits the regent's proposed control law against the goal/constraints and either
    approves it or returns a critique; on veto, the regent is asked to REVISE once with the critique
    injected as ``scratch["critic"]`` (which the prompt assemblers surface). Rollout-FREE — a single
    extra LLM call, no clone — so it is sound on any system. (doc-09 §5.2; an H3 ablation arm.)

    The critic is deliberately conservative: it approves unless it can name a concrete problem, so it
    cannot silently stall a run. Its audit calls are recorded into ``scratch["_llm_calls"]`` like the
    regent's, so the Runner persists them in the RunRecord.
    """

    name = "critic"

    def __init__(self, llm: Any, model: str, *, temperature: float = 0.0, max_revisions: int = 1,
                 max_tokens: int | None = None, extra: dict[str, Any] | None = None) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_revisions = max_revisions
        self.max_tokens = max_tokens
        self.extra = extra

    def propose_hook(self, regent, view, space, scratch, base):
        reqs = base(view, space, scratch)
        revisions = 0
        while reqs and revisions < self.max_revisions:
            approve, critique = self._audit(view, space, reqs, scratch)
            if approve:
                break
            scratch["critic"] = critique  # the assembler surfaces this so the regent revises
            reqs = base(view, space, scratch)
            revisions += 1
        scratch.pop("critic", None)
        scratch.setdefault("critic_log", []).append({"revisions": revisions})
        return reqs

    def _audit(self, view: Observation, space: ActionSpace, reqs: list[ActionRequest],
               scratch: dict) -> tuple[bool, str]:
        laws = "; ".join(f"{r.verb}: {r.payload.get('expr', r.payload)}" for r in reqs)
        obs = ", ".join(f"{k}={v:.6g}" for k, v in view.vars.items())
        messages = [
            {"role": "system", "content":
                "You are a control-policy critic. Decide whether the proposed control law is sensible "
                "and safe for driving the state to target without excessive control effort or instability. "
                "Respond with STRICT JSON only: {\"approve\": true|false, \"critique\": \"<one sentence>\"}. "
                "Approve unless you can name a CONCRETE problem (e.g. wrong sign, unstable gain, ignores state)."},
            {"role": "user", "content":
                f"State: {obs}\nAllowed variables: {space.context_vars}\nProposed law(s): {laws}\nJudge it."},
        ]
        opt: dict[str, Any] = {}
        if self.max_tokens is not None:
            opt["max_tokens"] = self.max_tokens
        if self.extra:
            opt["extra"] = self.extra
        resp = self.llm.complete(messages, model=self.model, temperature=self.temperature, **opt)
        scratch.setdefault("_llm_calls", []).append({
            "step": view.t, "regent": "critic", "messages": messages,
            "response_text": getattr(resp, "text", ""), "tool_calls": getattr(resp, "tool_calls", []),
            "model": getattr(resp, "model", self.model), "usage": getattr(resp, "usage", {}),
            "cost_usd": getattr(resp, "cost_usd", None), "cached": getattr(resp, "cached", False),
        })
        obj = _extract_json_obj(getattr(resp, "text", "") or "")
        return bool(obj.get("approve", True)), str(obj.get("critique", ""))
