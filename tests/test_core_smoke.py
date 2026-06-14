"""
Phase-0 smoke tests for the domain-agnostic core seams (govsim.core).

These run without numpy, without an API key, and without the `openai` package: they exercise
the seams' logic and the cache/replay tape with a fake in-memory LLM client. CI runs them
key-free (the WHAT-first / reproducibility discipline of agents/08 & 09).
"""

from __future__ import annotations

import pytest

from govsim.core import (
    ActionRequest,
    ActionSpace,
    AtSteps,
    EveryN,
    Harness,
    HarnessComponent,
    StaticRegent,
    ScriptedRegent,
    VerbSpec,
)
from govsim.core.llm import CachingReplayClient, LLMResponse


# --- Schedule -------------------------------------------------------------------------------

def test_everyn_decides_on_multiples():
    s = EveryN(50)
    assert s.should_decide(0)
    assert s.should_decide(50)
    assert s.should_decide(100)
    assert not s.should_decide(49)
    assert not s.should_decide(51)


def test_everyn_rejects_nonpositive():
    with pytest.raises(ValueError):
        EveryN(0)


def test_atsteps():
    s = AtSteps([200, 600, 800])
    assert s.should_decide(600)
    assert not s.should_decide(599)


# --- Regents + ActionSpace ------------------------------------------------------------------

def _space():
    return ActionSpace(
        verbs=[VerbSpec(name="set_u", value_range=(-2.0, 2.0), description="control input")],
        context_vars=["current_x", "previous_x", "t"],
    )


def test_static_regent_proposes_nothing():
    assert StaticRegent().decide(None, _space(), {}) == []


def test_test_regent_emits_fixed_request():
    r = ScriptedRegent(verb="set_u", expr="-0.9 * current_x", id="regent:0")
    reqs = r.decide(None, _space(), {})
    assert len(reqs) == 1
    req = reqs[0]
    assert isinstance(req, ActionRequest)
    assert req.regent_id == "regent:0"
    assert req.verb == "set_u"
    assert req.payload == {"expr": "-0.9 * current_x"}


def test_action_space_as_tools_is_openai_shaped():
    tools = _space().as_tools()
    assert len(tools) == 1
    fn = tools[0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "set_u"
    assert "expr" in fn["function"]["parameters"]["properties"]


# --- Harness (Phase 0: machinery, components exercised here as fakes) ------------------------

def test_harness_passthrough_with_no_components():
    r = ScriptedRegent(verb="set_u", expr="0.0")
    h = Harness()
    reqs = h.act(r, None, _space(), {})
    assert len(reqs) == 1 and reqs[0].verb == "set_u"


def test_harness_component_hooks_and_ablation_switch():
    calls: list[str] = []

    class Tracer(HarnessComponent):
        name = "tracer"

        def on_observe(self, view, space, scratch):
            calls.append("observe")
            scratch["note"] = "seen"

        def propose_hook(self, regent, view, space, scratch, base):
            calls.append("propose")
            return base(view, space, scratch)

    r = ScriptedRegent(verb="set_u", expr="0.0")
    comp = Tracer()
    h = Harness([comp])
    scratch: dict = {}
    h.act(r, None, _space(), scratch)
    assert calls == ["observe", "propose"]
    assert scratch["note"] == "seen"

    # the `enabled` flag is the entire ablation apparatus
    calls.clear()
    comp.enabled = False
    h.act(r, None, _space(), {})
    assert calls == []


# --- LLM cache / replay tape ----------------------------------------------------------------

class _FakeClient:
    """Records calls and returns a canned response; satisfies the LLMClient protocol."""

    def __init__(self):
        self.n_calls = 0

    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None, response_format=None):
        self.n_calls += 1
        return LLMResponse(text=f"resp#{self.n_calls}", model=model or "fake", usage={"total_tokens": 1})


def test_cache_serves_second_identical_call(tmp_path):
    fake = _FakeClient()
    client = CachingReplayClient(fake, tmp_path, mode="cache")
    msgs = [{"role": "user", "content": "hi"}]

    a = client.complete(msgs, model="m", temperature=0.0, seed=1)
    b = client.complete(msgs, model="m", temperature=0.0, seed=1)

    assert fake.n_calls == 1  # second call served from disk
    assert a.text == b.text == "resp#1"
    assert b.cached is True and a.cached is False


def test_cache_key_varies_with_seed(tmp_path):
    fake = _FakeClient()
    client = CachingReplayClient(fake, tmp_path, mode="cache")
    msgs = [{"role": "user", "content": "hi"}]
    client.complete(msgs, model="m", seed=1)
    client.complete(msgs, model="m", seed=2)
    assert fake.n_calls == 2  # different seed => different key => real call


def test_replay_miss_raises_and_never_calls(tmp_path):
    fake = _FakeClient()
    client = CachingReplayClient(fake, tmp_path, mode="replay")
    with pytest.raises(KeyError):
        client.complete([{"role": "user", "content": "x"}], model="m")
    assert fake.n_calls == 0
