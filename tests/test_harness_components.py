"""
Tests for the rollout-free harness components (TraceFeedback, EpisodicMemory) and their
interaction with the Harness ablation switch + an LLMRegent through the Runner.
"""

from __future__ import annotations

from govsim.core import EveryN, Experiment, Harness, Hypothesis, Runner
from govsim.core.action import ActionRequest, ActionSpace, VerbSpec
from govsim.core.harness import Outcome
from govsim.core.llm import LLMResponse
from govsim.core.system import Observation
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.harness import EpisodicMemory, TraceFeedback
from govsim.regents import LLMRegent


def _space():
    return ActionSpace(verbs=[VerbSpec(name="set_control_input", value_range=(-2.0, 2.0))], context_vars=["current_x"])


# --- TraceFeedback --------------------------------------------------------------------------

def test_trace_feedback_round_trips_error_into_next_prompt():
    tf = TraceFeedback()
    scratch: dict = {}
    obs = Observation(vars={"current_x": 1.0})
    # an outcome with an error → next on_observe surfaces it as scratch["trace"]
    tf.on_outcome(obs, [], Outcome(requests=[], error="rejected: unknown name 'foo'"), scratch)
    tf.on_observe(obs, _space(), scratch)
    assert scratch["trace"] == "rejected: unknown name 'foo'"
    # a clean outcome → trace is cleared on the following observe
    tf.on_outcome(obs, [], Outcome(requests=[], error=None), scratch)
    tf.on_observe(obs, _space(), scratch)
    assert "trace" not in scratch


# --- EpisodicMemory -------------------------------------------------------------------------

def test_episodic_memory_retrieves_nearest_episode():
    em = EpisodicMemory(k=1)
    req = ActionRequest("regent:0", "set_control_input", {"expr": "-0.9*current_x"})
    em.on_outcome(Observation(vars={"current_x": 5.0}), [req], Outcome(requests=[req], metrics={"current_x": 5.0}), {})
    em.on_outcome(Observation(vars={"current_x": 0.1}), [req], Outcome(requests=[req], metrics={"current_x": 0.1}), {})
    scratch: dict = {}
    em.on_observe(Observation(vars={"current_x": 0.0}), _space(), scratch)  # nearest = the 0.1 episode
    assert "memory" in scratch and "current_x=0.1" in scratch["memory"]


def test_episodic_memory_noop_when_empty():
    scratch: dict = {}
    EpisodicMemory().on_observe(Observation(vars={"current_x": 0.0}), _space(), scratch)
    assert "memory" not in scratch


# --- Harness ablation switch with real components -------------------------------------------

def test_harness_ablation_disables_component_effect():
    em = EpisodicMemory(k=1)
    req = ActionRequest("regent:0", "set_control_input", {"expr": "0"})
    em.episodes.append({"state": {"current_x": 1.0}, "actions": [{"verb": "set_control_input", "expr": "0"}], "score": 1.0})
    h = Harness([em])

    class _R:
        id = "regent:0"

        def decide(self, view, space, scratch):
            return [req]

    scratch: dict = {}
    h.act(_R(), Observation(vars={"current_x": 1.0}), _space(), scratch)
    assert "memory" in scratch  # component active

    em.enabled = False
    scratch2: dict = {}
    h.act(_R(), Observation(vars={"current_x": 1.0}), _space(), scratch2)
    assert "memory" not in scratch2  # the enabled flag is the whole ablation apparatus


# --- end-to-end: LLMRegent + harness through the Runner (key-free) --------------------------

class _ToolClient:
    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None,
                 response_format=None, max_tokens=None, extra=None):
        # echo back whether memory/trace reached the prompt, so we can assert the harness fired
        saw_mem = any("past episodes" in m["content"].lower() for m in messages)
        return LLMResponse(tool_calls=[{"id": "1", "name": "set_control_input", "arguments": '{"expr": "-0.9*current_x"}'}],
                           model=model or "fake", usage={"saw_memory": saw_mem})


def test_llm_regent_with_harness_runs_end_to_end():
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"sigma_epsilon": 0.05})
        s.reset(seed)
        return s

    exp = Experiment(
        name="cubic_llm_harness",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": LLMRegent(llm=_ToolClient(), model="fake")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(10),
        seeds=[0],
        horizon=60,
        hypothesis=Hypothesis(id="H-harness", claim="harness runs", baseline="no-harness", primary_metric="mse"),
        harness=Harness([TraceFeedback(), EpisodicMemory(k=2)]),
    )
    rec = Runner().run(exp)[0]
    assert len(rec.llm_io) >= 1
    # by the 2nd+ decision EpisodicMemory has stored an episode → memory reaches the prompt
    assert any(call["usage"].get("saw_memory") for call in rec.llm_io)
