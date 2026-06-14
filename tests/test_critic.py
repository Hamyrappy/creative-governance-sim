"""
Tests for the Critic harness component (rollout-free 2nd-LLM audit → veto → revise once) and that
the prompt assemblers surface the critique on ``scratch["critic"]``.
"""

from __future__ import annotations

from govsim.core.action import ActionRequest, ActionSpace, VerbSpec
from govsim.core.llm import LLMResponse
from govsim.core.system import Observation
from govsim.harness import Critic
from govsim.regents.llm_regent import default_prompt_assembler


def _space():
    return ActionSpace(verbs=[VerbSpec(name="set_control_input", value_range=(-2.0, 2.0))],
                       context_vars=["current_x"])


def _obs():
    return Observation(vars={"current_x": 1.0}, t=0)


class _Veto:
    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None,
                 response_format=None, max_tokens=None, extra=None):
        return LLMResponse(text='{"approve": false, "critique": "the gain has the wrong sign"}',
                           model=model or "fake")


class _Approve:
    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None,
                 response_format=None, max_tokens=None, extra=None):
        return LLMResponse(text='{"approve": true, "critique": ""}', model=model or "fake")


def _base_factory(calls):
    def base(v, s, sc):
        calls.append(sc.get("critic"))
        expr = "-0.9*current_x" if sc.get("critic") else "0.9*current_x"  # fix the sign once critiqued
        return [ActionRequest("regent:0", "set_control_input", {"expr": expr})]
    return base


def test_critic_veto_triggers_one_revision():
    calls: list = []
    critic = Critic(_Veto(), "fake")
    scratch: dict = {}
    out = critic.propose_hook(None, _obs(), _space(), scratch, _base_factory(calls))
    assert out[0].payload["expr"] == "-0.9*current_x"          # the revised (sign-fixed) law
    assert calls == [None, "the gain has the wrong sign"]      # base re-called WITH the critique
    assert scratch["_llm_calls"][0]["regent"] == "critic"      # the audit call was recorded
    assert "critic" not in scratch                             # transient key popped after revising
    assert scratch["critic_log"][0]["revisions"] == 1


def test_critic_approve_passes_through_unchanged():
    calls: list = []
    critic = Critic(_Approve(), "fake")
    scratch: dict = {}
    out = critic.propose_hook(None, _obs(), _space(), scratch, _base_factory(calls))
    assert out[0].payload["expr"] == "0.9*current_x"           # original, no revision
    assert calls == [None]                                     # base called exactly once
    assert scratch["critic_log"][0]["revisions"] == 0


def test_critic_respects_max_revisions_zero():
    calls: list = []
    critic = Critic(_Veto(), "fake", max_revisions=0)
    out = critic.propose_hook(None, _obs(), _space(), {}, _base_factory(calls))
    assert out[0].payload["expr"] == "0.9*current_x"           # never audited/revised
    assert calls == [None]


def test_default_assembler_surfaces_critic_channel():
    msgs = default_prompt_assembler(_obs(), _space(), {"critic": "wrong sign on the gain"})
    assert "wrong sign on the gain" in msgs[1]["content"]
