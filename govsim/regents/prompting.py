"""
Prompt assembly + the boot-time ``check_prompt`` validator (doc-06 §2.3, repositioned by doc-09 §9).

The prompt is a FOUR-source contract — observables (the view) ∪ action space (verbs/ranges/context
vars) ∪ agent-supplied text (history/trace/KPIs) — assembled by the agent, NOT auto-derived. The one
honest cost of that is a placeholder can reference a name no source supplies; ``check_prompt`` closes
it by failing LOUD at *construction* (boot), instead of silently blanking the field at the LLM.

``TemplatePromptAssembler`` realizes a ``{placeholder}`` template against the four sources with that
boot validator. ``make_obfuscated_assembler`` builds the **partial-information** assembler for the H1
nonlinear arm: it tells the regent the update rule is ``x_(k+1) = f(x_k, u_k, noise)`` with ``f``
UNKNOWN and possibly nonlinear — so the regent must *infer the structure from observed history* and
synthesize a (possibly nonlinear) law, the regime where code-as-policy can beat a fixed-form PID.
"""

from __future__ import annotations

import re
import string
from typing import Any

from govsim.core.action import ActionSpace
from govsim.core.system import Observation

# A ``{name}`` placeholder, but NOT a ``{{escaped}}`` brace (negative lookbehind on a leading '{').
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-zA-Z_]\w*)\}")

# The non-observable legs of the four-source contract the AGENT supplies (not the world's observables).
AGENT_SUPPLIED = {"available_context_vars", "history_text", "trace", "memory", "current_step", "critic"}


def placeholders(template: str) -> set[str]:
    """The set of ``{name}`` fields a template references (ignoring ``{{escaped}}`` braces)."""
    return set(_PLACEHOLDER_RE.findall(template))


def check_prompt(template: str, suppliable: set[str]) -> None:
    """Fail LOUD if ``template`` references a name no source can supply (doc-06 §2.3).

    Run at assembler construction (boot), so a typo'd ``{param_Q}`` raises immediately instead of
    silently rendering blank at the LLM and corrupting a run.
    """
    missing = placeholders(template) - set(suppliable)
    if missing:
        raise ValueError(
            f"prompt template references unsuppliable names: {sorted(missing)} "
            f"(suppliable: {sorted(suppliable)})"
        )


def suppliable_names(observation: Observation, space: ActionSpace) -> set[str]:
    """The names a prompt MAY reference for this world: its observables + the agent-supplied legs +
    each verb's ``<verb>_range`` (and the conventional ``u_range``)."""
    names = set(observation.vars) | set(AGENT_SUPPLIED)
    names |= {f"{v.name}_range" for v in space.verbs}
    if space.verbs:
        names.add("u_range")
    return names


def _fmt(v: Any) -> Any:
    return f"{v:.6g}" if isinstance(v, float) else v


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - check_prompt prevents reaching this
        return "{" + key + "}"


class TemplatePromptAssembler:
    """A four-source ``{placeholder}`` template assembler with a boot-time ``check_prompt``.

    Validated at CONSTRUCTION against the world's ``suppliable`` names, so a bad placeholder fails at
    boot, not silently at the LLM. Callable as a ``PromptAssembler`` (``(view, space, scratch) -> messages``).
    """

    def __init__(self, template: str, suppliable: set[str], *,
                 nudge: str = "Choose the control law now (call the provided tool).") -> None:
        check_prompt(template, suppliable)
        self.template = template
        self.suppliable = set(suppliable)
        self.nudge = nudge

    def __call__(self, view: Observation, space: ActionSpace, scratch: dict) -> list[dict[str, Any]]:
        data: dict[str, Any] = {k: _fmt(v) for k, v in view.vars.items()}
        data["available_context_vars"] = ", ".join(space.context_vars)
        data["current_step"] = view.t
        data["history_text"] = scratch.get("memory") or "(no history yet)"
        data["trace"] = f"Feedback on your last action: {scratch['trace']}" if scratch.get("trace") else ""
        data["critic"] = f"A critic flagged your previous law: {scratch['critic']} Revise it." if scratch.get("critic") else ""
        for v in space.verbs:
            if v.value_range:
                data[f"{v.name}_range"] = v.value_range
        if space.verbs and space.verbs[0].value_range:
            data["u_range"] = space.verbs[0].value_range
        text = string.Formatter().vformat(self.template, (), _SafeDict(data))
        return [{"role": "system", "content": text}, {"role": "user", "content": self.nudge}]


OBFUSCATED_TEMPLATE = """You govern a stochastic dynamical system whose update rule is

    x_(k+1) = f(x_k, u_k, noise)

where the function f is UNKNOWN to you and MAY BE NONLINEAR. You are NOT told f. You must INFER its
structure from the observed history of states and your past controls, then choose a control law that
drives the state to target. Warning: a purely linear control law can be insufficient if the plant is
nonlinear (for example, if it grows faster than linearly in the state, a fixed proportional gain that
works near the target may fail to contain large excursions).

Objective: drive current_x to target_x = {target_x} while keeping control effort moderate.

You may use ONLY these context variables in your expression: {available_context_vars}

Recent history (most-similar past situations: state -> control -> score), if any:
{history_text}

Current state (step {current_step}): current_x={current_x}, previous_x={previous_x}, current_u={current_u}.
{trace}
{critic}

Return ONE Python expression for the control u_k (no statements, imports, or side effects), using only
the allowed variables and safe math/np helpers. The system clips the result into its range for you, so
do not reference any range name in the expression. Consider conditionals or terms in the state's higher
powers if the history suggests the plant is nonlinear."""


def make_obfuscated_assembler(suppliable: set[str]) -> TemplatePromptAssembler:
    """The partial-information assembler for the H1 nonlinear arm (f unknown ⇒ infer from history)."""
    return TemplatePromptAssembler(OBFUSCATED_TEMPLATE, suppliable)
