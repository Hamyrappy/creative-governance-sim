"""
OPRORegent — Optimization-by-PROmpting, the named H1 rival the LLM-with-harness must beat
(doc-09 §6.1 / §6.4).

Mechanism: keep an archive of ``(control_law, score)`` across decision steps; each decision, show the
LLM the best-so-far laws sorted worst→best (the OPRO meta-prompt), ask for a NEW law expected to
score higher, score that candidate by **rollout** on a clone, append it, and emit the *incumbent best*
law in the archive. It is **trace-LESS**: its only feedback is the ``(solution, score)`` optimization
trajectory — there is no error/conservation/runtime trace channel (that is exactly the ``TraceFeedback``
arm it is compared against). So "code-as-policy + trace beats trace-less OPRO" is a clean contrast.

Two scoring modes:
  - ``realized`` (the FAIR H1 baseline): the archive credits each law with the realized ``Objective``
    over the interval it was actually deployed (the Runner supplies ``scratch["_last_realized_score"]``).
    OPRO follows a deterministic explore/exploit schedule — it deploys a fresh proposal to *measure* it
    during warm-up and periodically, but otherwise re-deploys its incumbent-best law — exactly like a
    trace-less reasoner that both learns from observed outcomes AND exploits what it has learned. (A
    pure-explore variant that never re-deploys its best law would be a silently crippled rival.) Use
    this against the harnessed LLM.
  - ``rollout`` (generous, self-contained): score candidates on a true-plant clone before committing the
    incumbent best. This makes OPRO an oracle optimizer (more information than the harnessed LLM gets),
    so it is a *strong* baseline, kept for ablations and replay-determinism. Needs a ``RollableSystem``.
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
                 temperature: float = 0.8, seed: int = 0, scoring: str = "rollout",
                 probe_horizon: int = 30, probe_seeds: tuple[int, ...] = (0, 1, 2),
                 archive_cap: int = 16, top_k: int = 8, explore_warmup: int = 3,
                 explore_period: int = 4,
                 max_tokens: int | None = None, extra: dict[str, Any] | None = None) -> None:
        super().__init__(id)
        if scoring not in ("rollout", "realized"):
            raise ValueError("scoring must be 'rollout' or 'realized'")
        self.verb = verb
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.seed = seed
        # 'rollout': score candidates on a true-plant clone (generous — an oracle optimizer).
        # 'realized': score the law you ACTUALLY deployed by its realized objective over the next
        #   interval (fair — the trace-less reasoning baseline H1 should be measured against).
        self.scoring = scoring
        self.probe_horizon = probe_horizon
        self.probe_seeds = tuple(probe_seeds)
        self.archive_cap = archive_cap
        self.top_k = top_k
        # Realized-mode explore/exploit schedule (deterministic ⇒ replay-stable): explore (deploy &
        # measure a fresh proposal) during the first ``explore_warmup`` decisions and every
        # ``explore_period``-th decision thereafter; otherwise EXPLOIT the incumbent-best archived law.
        # Without this, realized OPRO would deploy a fresh temp-0.8 proposal at EVERY decision and never
        # its best law, so its realized loss would be permanent exploration cost — a silently crippled
        # H1 rival (the harnessed LLM would "beat" it on OPRO's un-exploited exploration, not on trace).
        self.explore_warmup = explore_warmup
        self.explore_period = max(1, explore_period)
        self.max_tokens = max_tokens
        self.extra = extra

    def _trim(self, archive: list[tuple[str, float]]) -> None:
        archive.sort(key=lambda t: t[1])  # ascending ⇒ best last
        if len(archive) > self.archive_cap:
            del archive[: len(archive) - self.archive_cap]

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

        # 'realized' mode: first credit the previously-deployed law with its realized score.
        if self.scoring == "realized":
            pending = scratch.get("_opro_pending")
            realized = scratch.get("_last_realized_score")
            if pending is not None and realized is not None:
                archive.append((pending, float(realized)))
                self._trim(archive)

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
        new_expr = reqs[0].payload.get("expr") if reqs else None

        if self.scoring == "rollout":
            ctx = scratch.get("_rollout")
            if reqs and ctx is not None and isinstance(new_expr, str):
                valid = ctx.action_interface.validate(reqs[0], ctx.system, self.id)
                if valid.ok:
                    archive.append((new_expr, statistics.fmean(ctx.score(reqs[:1], self.probe_horizon, self.probe_seeds))))
                    self._trim(archive)
            if archive:  # commit the incumbent best law (the true plant was the oracle)
                return [ActionRequest(self.id, self.verb, {"expr": archive[-1][0]})]
            return reqs

        # 'realized' mode: epsilon-greedy over the archive.
        idx = scratch.get("_opro_realized_idx", 0)
        scratch["_opro_realized_idx"] = idx + 1

        # Validate the proposal BEFORE deploying: a malformed law would be rejected by ``apply`` and the
        # system would keep the PREVIOUS lever, so crediting the malformed law with that realized score
        # (the old bug) is wrong. Only an actually-installable law may be deployed & credited.
        new_valid = isinstance(new_expr, str)
        if new_valid:
            ctx = scratch.get("_rollout")
            if ctx is not None:
                try:
                    new_valid = ctx.action_interface.validate(reqs[0], ctx.system, self.id).ok
                except Exception:
                    new_valid = False

        explore = (not archive) or idx < self.explore_warmup or (idx % self.explore_period == 0)
        if explore and new_valid:  # deploy & measure a fresh proposal
            scratch["_opro_pending"] = new_expr
            return [ActionRequest(self.id, self.verb, {"expr": new_expr})]
        if archive:  # EXPLOIT the incumbent best (re-credit it so its realized estimate refines)
            best = archive[-1][0]
            scratch["_opro_pending"] = best
            return [ActionRequest(self.id, self.verb, {"expr": best})]
        if new_valid:  # nothing archived yet ⇒ deploy the valid proposal to seed the archive
            scratch["_opro_pending"] = new_expr
            return [ActionRequest(self.id, self.verb, {"expr": new_expr})]
        scratch["_opro_pending"] = None  # no valid proposal and empty archive ⇒ nothing to credit
        return reqs

    def _record(self, scratch: Scratch, view: Observation, messages: list[dict], resp: Any) -> None:
        scratch.setdefault("_llm_calls", []).append({
            "step": view.t, "regent": "opro", "messages": messages,
            "response_text": getattr(resp, "text", ""), "tool_calls": getattr(resp, "tool_calls", []),
            "model": getattr(resp, "model", self.model), "usage": getattr(resp, "usage", {}),
            "cost_usd": getattr(resp, "cost_usd", None), "cached": getattr(resp, "cached", False),
        })
