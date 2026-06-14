"""
Tests for the prompt-assembly layer: the doc-06 §2.3 boot validator (``check_prompt``), the
``TemplatePromptAssembler``, and the partial-information ``make_obfuscated_assembler`` (the H1 arm).
"""

from __future__ import annotations

import pytest

from govsim.core.action import ActionSpace, VerbSpec
from govsim.core.system import Observation
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface
from govsim.regents import LLMRegent, make_obfuscated_assembler, suppliable_names
from govsim.regents.prompting import (
    OBFUSCATED_TEMPLATE, TemplatePromptAssembler, check_prompt, placeholders,
)
from govsim.core.llm import LLMResponse


def _space():
    return ActionSpace(verbs=[VerbSpec(name="set_control_input", value_range=(-2.0, 2.0))],
                       context_vars=["current_x", "previous_x", "current_u", "target_x", "step"])


def _obs(**vars_):
    base = {"current_x": 1.0, "previous_x": 1.2, "current_u": 0.0, "target_x": 0.0, "step": 0.0}
    base.update(vars_)
    return Observation(vars=base, t=int(base["step"]))


# --- check_prompt / placeholders ------------------------------------------------------------

def test_placeholders_ignores_escaped_braces():
    assert placeholders("set {x} to {y}, literal {{not_a_field}}") == {"x", "y"}


def test_check_prompt_raises_on_unsuppliable():
    with pytest.raises(ValueError) as e:
        check_prompt("use {current_x} and {param_Q}", {"current_x", "trace"})
    assert "param_Q" in str(e.value)


def test_check_prompt_passes_when_all_suppliable():
    check_prompt("use {current_x} and {trace}", {"current_x", "trace", "extra"})  # no raise


# --- suppliable_names + TemplatePromptAssembler ---------------------------------------------

def test_suppliable_names_covers_obs_and_agent_legs():
    sys = CubicSystem({"target_x": 0.0})
    sys.reset(0)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])
    names = suppliable_names(sys.observe(), iface.action_space(sys, "regent:0"))
    assert {"current_x", "previous_x", "current_u", "target_x", "step"} <= names
    assert {"available_context_vars", "history_text", "trace", "current_step"} <= names
    assert "set_control_input_range" in names


def test_template_assembler_formats_and_validates():
    tpl = "x={current_x}, ctx={available_context_vars}, step={current_step}.{trace}"
    asm = TemplatePromptAssembler(tpl, {"current_x", "available_context_vars", "current_step", "trace"})
    msgs = asm(_obs(current_x=2.5), _space(), {"trace": "rejected: bad name"})
    sys_msg = msgs[0]["content"]
    assert "x=2.5" in sys_msg and "current_x" in sys_msg
    assert "rejected: bad name" in sys_msg


def test_template_assembler_rejects_bad_template_at_construction():
    with pytest.raises(ValueError):
        TemplatePromptAssembler("uses {nonexistent_field}", {"current_x"})


# --- obfuscated (partial-information) assembler ---------------------------------------------

def test_obfuscated_assembler_hides_structure_and_asks_to_infer():
    sys = CubicSystem({"target_x": 0.0, "cubic_coeff": 0.05})
    sys.reset(0)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])
    asm = make_obfuscated_assembler(suppliable_names(sys.observe(), iface.action_space(sys, "regent:0")))
    msgs = asm(sys.observe(), iface.action_space(sys, "regent:0"), {})
    text = msgs[0]["content"].lower()
    assert "unknown" in text and "f(x_k" in text             # f is presented as unknown
    assert "infer" in text                                    # must infer structure from history
    assert "cubic_coeff" not in text and "0.05" not in text   # the true nonlinearity is NOT revealed
    assert "x_(k+1)" in msgs[0]["content"]


def test_llm_regent_with_obfuscated_assembler_runs():
    class _ToolClient:
        def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None,
                     response_format=None, max_tokens=None, extra=None):
            return LLMResponse(tool_calls=[{"id": "1", "name": "set_control_input",
                                            "arguments": '{"expr": "-1.2*current_x - 0.1*current_x**3"}'}],
                               model=model or "fake")

    sys = CubicSystem({"target_x": 0.0, "cubic_coeff": 0.05})
    sys.reset(0)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])
    asm = make_obfuscated_assembler(suppliable_names(sys.observe(), iface.action_space(sys, "regent:0")))
    r = LLMRegent(llm=_ToolClient(), model="fake", prompt_assembler=asm)
    reqs = r.decide(sys.observe(), iface.action_space(sys, "regent:0"), {})
    assert reqs[0].payload["expr"] == "-1.2*current_x - 0.1*current_x**3"
