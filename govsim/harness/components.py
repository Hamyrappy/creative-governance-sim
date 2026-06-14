"""
Rollout-free harness components.

Both write into ``scratch`` keys the LLM prompt assembler reads (``trace``, ``memory``), so they
upgrade an ``LLMRegent`` without the regent knowing they exist — and toggling ``enabled`` (the H3
ablation switch) cleanly removes their effect. Neither needs ``clone()``/rollout, so they are sound
on any system (the rollout-soundness gate does not apply).
"""

from __future__ import annotations

import math

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.harness import HarnessComponent, Outcome
from govsim.core.system import Observation


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
