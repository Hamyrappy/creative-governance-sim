"""
OPRORegent — Optimization-by-PROmpting, the named H1 rival the LLM-with-harness must beat
(doc-09 §6.1 / §6.4).

Mechanism: keep an archive of ``(control_law, score)`` across decision steps; each decision, show the
LLM the best-so-far laws sorted worst→best (the OPRO meta-prompt), ask for a NEW law expected to
score higher, score that candidate by **rollout** on a clone, append it, and emit the *incumbent best*
law in the archive. It is **trace-LESS**: its only feedback is the ``(solution, score)`` optimization
trajectory — there is no error/conservation/runtime trace channel (that is exactly the ``TraceFeedback``
arm it is compared against). So "code-as-policy + trace beats trace-less OPRO" is a clean contrast.

Scoring candidates by rollout (rather than waiting for realized multi-step feedback) makes OPRO
self-contained and deterministic under the replay tape — a faithful, if generous, version of the
baseline. It therefore needs the rollout context the Runner injects for ``RollableSystem``s; on a
non-rollable system it degrades to emitting the LLM's latest raw proposal (no archive).
"""

from __future__ import annotations

import statistics
from typing import Any

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.regent import Regent, Scratch
from govsim.core.system import Observation
from govsim.regents.llm_regent import parse_action_requests


def _format_archive(archive: list[tuple[str, float]], top: int) -> str:
    return "\n".join(f"  score={s:.4f}  ->  {e}" for e, s in archive[-top:])


class OPRORegent(Regent):
    def __init__(self, verb: str, llm: Any, model: str, *, id: str = "regent:0",
                 temperature: float = 0.8, seed: int = 0, probe_horizon: int = 30,
                 probe_seeds: tuple[int, ...] = (0, 1, 2), archive_cap: int = 16, top_k: int = 8,
                 max_tokens: int | None = None, extra: dict[str, Any] | None = None) -> None:
        super().__init__(id)
        self.verb = verb
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.probe_horizon = probe_horizon
        self.probe_seeds = tuple(probe_seeds)
        self.archive_cap = archive_cap
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.extra = extra

    def _meta_prompt(self, view: Observation, space: ActionSpace, archive: list[tuple[str, float]]) -> list[dict]:
        system = (
            "You are optimizing a single control law (ONE Python expression) for a dynamical system. "
            f"The law sets the lever '{self.verb}'; it is re-evaluated every step and clipped to range. "
            f"Allowed variables (use ONLY these): {space.context_vars}. No statements/imports/side effects.\n"
            "Below are laws already tried with their scores (HIGHER is better). Propose a NEW law, "
            "different from those, that you expect to score higher."
        )
        if archive:
            system += "\n\nTried so far (worst to best):\n" + _format_archive(archive, self.top_k)
        else:
            system += "\n\nNothing tried yet — propose a sensible first control law."
        obs = ", ".join(f"{k}={v:.6g}" for k, v in view.vars.items())
        user = f"Current state (step {view.t}): {obs}\nReturn the new control law now (call the tool)."
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        archive: list[tuple[str, float]] = scratch.setdefault("_opro_archive", [])
        messages = self._meta_prompt(view, space, archive)
        opt: dict[str, Any] = {}
        if self.max_tokens is not None:
            opt["max_tokens"] = self.max_tokens
        if self.extra:
            opt["extra"] = self.extra
        resp = self.llm.complete(messages, tools=space.as_tools(), model=self.model,
                                 temperature=self.temperature, seed=self.seed + len(archive), **opt)
        self._record(scratch, view, messages, resp)
        reqs = parse_action_requests(resp, space, self.id)

        ctx = scratch.get("_rollout")
        if reqs and ctx is not None:
            expr = reqs[0].payload.get("expr")
            valid = ctx.action_interface.validate(reqs[0], ctx.system, self.id)
            if isinstance(expr, str) and valid.ok:
                futures = ctx.score(reqs[:1], self.probe_horizon, self.probe_seeds)
                archive.append((expr, statistics.fmean(futures)))
                archive.sort(key=lambda t: t[1])  # ascending ⇒ best is last
                if len(archive) > self.archive_cap:
                    del archive[: len(archive) - self.archive_cap]

        if archive:  # commit the incumbent best law (OPRO does not chase the latest noisy proposal)
            return [ActionRequest(self.id, self.verb, {"expr": archive[-1][0]})]
        return reqs

    def _record(self, scratch: Scratch, view: Observation, messages: list[dict], resp: Any) -> None:
        scratch.setdefault("_llm_calls", []).append({
            "step": view.t, "regent": "opro", "messages": messages,
            "response_text": getattr(resp, "text", ""), "tool_calls": getattr(resp, "tool_calls", []),
            "model": getattr(resp, "model", self.model), "usage": getattr(resp, "usage", {}),
            "cost_usd": getattr(resp, "cost_usd", None), "cached": getattr(resp, "cached", False),
        })
