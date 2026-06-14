"""
LLMRegent — the LLM controller, on the new domain-blind seam.

It assembles a prompt from four sources (doc-06 §2.3: task instruction · action space · current
observation · harness scratch[trace/memory]), calls an OpenAI-*compatible* client with the action
space rendered as tools, and parses the reply (native tool-call preferred, JSON fallback, legacy
``value_expression`` tolerated) into validated ``ActionRequest``s. Every raw call is recorded into
``scratch["_llm_calls"]`` so the Runner persists it in the RunRecord (reproducibility + a reasoning
corpus). With a ``CachingReplayClient`` in replay mode the regent is deterministic and key-free.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.regent import Regent, Scratch
from govsim.core.system import Observation

PromptAssembler = Callable[[Observation, ActionSpace, dict], list[dict[str, Any]]]


def default_prompt_assembler(view: Observation, space: ActionSpace, scratch: dict) -> list[dict[str, Any]]:
    """The 4-source prompt. Domain-blind: it never names a domain — only verbs + context vars."""
    verb_lines = []
    for v in space.verbs:
        rng = f" in range {v.value_range}" if v.value_range else ""
        verb_lines.append(f"  - {v.name}{rng}: {v.description}")
    system = (
        "You are a regent (controller) governing a dynamical system. At each decision you choose a "
        "control law for one or more levers. A law is ONE Python expression over the allowed context "
        "variables (no statements, imports, or side effects); the system re-evaluates it every step "
        "and clips the result into the lever's range.\n\n"
        "Levers you may set:\n" + "\n".join(verb_lines) + "\n\n"
        f"Allowed context variables (use ONLY these): {space.context_vars}\n\n"
        "Respond by CALLING one of the provided tools with {\"expr\": \"<expression>\"} (preferred), "
        "or, if you cannot call a tool, return strict JSON: {\"verb\": \"<lever>\", \"expr\": \"<expression>\"}."
    )
    obs = ", ".join(f"{k}={v:.6g}" for k, v in view.vars.items())
    user_parts = [f"Current observation (step {view.t}): {obs}"]
    if scratch.get("trace"):
        user_parts.append(f"Feedback on your last action: {scratch['trace']}")
    if scratch.get("memory"):
        user_parts.append(f"Relevant past episodes:\n{scratch['memory']}")
    if scratch.get("critic"):
        user_parts.append(f"A critic flagged your previous control law: {scratch['critic']} Revise it.")
    user_parts.append("Choose the control law now.")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(user_parts)}]


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a possibly fenced/surrounded reply."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _payload_from_args(args: Any) -> dict | None:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    if "expr" in args:
        return {"expr": str(args["expr"])}
    if "value" in args:
        return {"value": args["value"]}
    if "value_expression" in args:  # legacy
        return {"expr": str(args["value_expression"])}
    return None


def parse_action_requests(resp: Any, space: ActionSpace, regent_id: str) -> list[ActionRequest]:
    """Turn an ``LLMResponse`` into ActionRequests (tool-calls preferred; JSON / legacy fallback)."""
    verbs = set(space.verb_names())
    reqs: list[ActionRequest] = []

    for tc in getattr(resp, "tool_calls", None) or []:
        verb = tc.get("name")
        payload = _payload_from_args(tc.get("arguments"))
        if verb in verbs and payload:
            reqs.append(ActionRequest(regent_id=regent_id, verb=verb, payload=payload))
    if reqs:
        return reqs

    obj = _extract_json(getattr(resp, "text", "") or "")
    if not obj:
        return []

    if "actions" in obj and isinstance(obj["actions"], list):
        for a in obj["actions"]:
            payload = _payload_from_args(a)
            verb = a.get("verb") if isinstance(a, dict) else None
            if verb in verbs and payload:
                reqs.append(ActionRequest(regent_id=regent_id, verb=verb, payload=payload))
        return reqs

    payload = _payload_from_args(obj)
    verb = obj.get("verb") or obj.get("policy_type_id")
    # If the model omitted the verb but there is exactly one lever, target it.
    if verb not in verbs and len(verbs) == 1:
        verb = next(iter(verbs))
    if verb in verbs and payload:
        reqs.append(ActionRequest(regent_id=regent_id, verb=verb, payload=payload))
    return reqs


class LLMRegent(Regent):
    def __init__(self, llm: Any, model: str, *, id: str = "regent:0", temperature: float = 0.0,
                 seed: int = 0, prompt_assembler: PromptAssembler | None = None,
                 prompt_file: str | None = None, max_tokens: int | None = None,
                 extra: dict[str, Any] | None = None) -> None:
        super().__init__(id)
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.prompt_assembler = prompt_assembler or default_prompt_assembler
        self.prompt_file = prompt_file
        self.max_tokens = max_tokens
        self.extra = extra  # e.g. {"reasoning_effort": "low"} for gpt-oss

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        messages = self.prompt_assembler(view, space, scratch)
        tools = space.as_tools()
        # RolloutProbe asks for several distinct candidates per decision by bumping this offset;
        # perturbing the seed gives a different sample (and a distinct cache key) per candidate.
        seed = self.seed + int(scratch.get("_probe_seed_offset", 0))
        # Pass max_tokens/extra only when set, so lean fake clients (tests) need not accept them.
        opt: dict[str, Any] = {}
        if self.max_tokens is not None:
            opt["max_tokens"] = self.max_tokens
        if self.extra:
            opt["extra"] = self.extra
        resp = self.llm.complete(
            messages, tools=tools, model=self.model, temperature=self.temperature, seed=seed, **opt
        )
        scratch.setdefault("_llm_calls", []).append(
            {
                "step": view.t,
                "messages": messages,
                "response_text": getattr(resp, "text", ""),
                "tool_calls": getattr(resp, "tool_calls", []),
                "model": getattr(resp, "model", self.model),
                "usage": getattr(resp, "usage", {}),
                "cost_usd": getattr(resp, "cost_usd", None),
                "cached": getattr(resp, "cached", False),
            }
        )
        return parse_action_requests(resp, space, self.id)
