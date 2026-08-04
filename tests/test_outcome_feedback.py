"""
Tests for the two performance-feedback channels.

These matter more than most component tests because the experiment's headline null hinged on one of
them: ``OutcomeFeedback`` produced no effect, and the first question was whether it was wired at all.
It was — but a *silent* wiring failure is indistinguishable from a genuine null in the results table,
so the wiring is pinned here rather than re-verified by hand next time.

``ContextualOutcomeFeedback`` exists because of what the transcripts showed once the wiring was
cleared: scores drawn from different phases of an evolving world are confounded with the phase. Its
distinguishing behaviour is the repeat-detection note, so that is what is tested.
"""

from __future__ import annotations

from govsim.core.action import ActionRequest, ActionSpace, VerbSpec
from govsim.core.harness import Outcome
from govsim.core.system import Observation
from govsim.harness import ContextualOutcomeFeedback, OutcomeFeedback, TraceFeedback

SPACE = ActionSpace(verbs=[VerbSpec(name="set_lockdown", value_range=(0.0, 0.9))],
                    context_vars=["I", "S"])


def _obs(t: int, I: float, S: float = 0.5) -> Observation:
    return Observation(vars={"I": I, "S": S, "t": float(t)}, scope="regent:0", t=t)


def _req(expr: str) -> ActionRequest:
    return ActionRequest(regent_id="regent:0", verb="set_lockdown", payload={"expr": expr})


def _cycle(comp, scratch, t, I, expr, realized, S=0.5):
    """One decision cycle: observe, act, then be told what the deployed law realized."""
    comp.on_observe(_obs(t, I, S), SPACE, scratch)
    reqs = [_req(expr)]
    comp.on_outcome(_obs(t, I, S), reqs, Outcome(requests=reqs, report=None, error=None,
                                                 metrics={"I": I}), scratch)
    scratch["_last_realized_score"] = realized


def test_outcome_feedback_reaches_the_scratch_the_prompt_reads():
    comp, scratch = OutcomeFeedback(k=4), {}
    _cycle(comp, scratch, 0, 0.10, "0.9", -11.0)
    _cycle(comp, scratch, 10, 0.12, "0.5", -13.0)
    comp.on_observe(_obs(20, 0.13), SPACE, scratch)

    text = scratch.get("outcome", "")
    assert "0.9" in text and "-11.0" in text, "the deployed law and its realized score must appear"
    assert "WORSE than" in text, "a decline must be called out, not left for the model to infer"


def test_outcome_feedback_is_silent_before_anything_has_been_realized():
    comp, scratch = OutcomeFeedback(k=4), {}
    comp.on_observe(_obs(0, 0.1), SPACE, scratch)
    assert "outcome" not in scratch, "an empty history must not render an empty section"


def test_trace_feedback_stays_silent_on_well_formed_policy():
    """The pre-registered null control. If this ever fires without an error, the design is broken."""
    comp, scratch = TraceFeedback(), {}
    for t in (0, 10, 20):
        comp.on_observe(_obs(t, 0.1), SPACE, scratch)
        reqs = [_req("0.5")]
        comp.on_outcome(_obs(t, 0.1), reqs,
                        Outcome(requests=reqs, report=None, error=None, metrics={}), scratch)
        assert "trace" not in scratch


def test_trace_feedback_fires_on_a_rejection():
    comp, scratch = TraceFeedback(), {}
    reqs = [_req("import os")]
    comp.on_outcome(_obs(0, 0.1), reqs,
                    Outcome(requests=reqs, report=None, error="rejected: import", metrics={}), scratch)
    comp.on_observe(_obs(10, 0.1), SPACE, scratch)
    assert "rejected" in scratch["trace"]


def test_contextual_outcome_attaches_the_state_each_score_was_earned_in():
    comp, scratch = ContextualOutcomeFeedback(k=4), {}
    _cycle(comp, scratch, 0, 0.10, "0.9", -11.0)
    _cycle(comp, scratch, 10, 0.30, "0.9", -14.0)
    comp.on_observe(_obs(20, 0.31), SPACE, scratch)

    text = scratch["outcome"]
    assert "from state [" in text, "each score must carry the state it was earned in"
    assert "I=0.1" in text and "I=0.3" in text


def test_contextual_outcome_flags_the_same_law_scoring_differently_in_a_similar_state():
    """The load-bearing signal: same policy, comparable conditions, different result."""
    comp, scratch = ContextualOutcomeFeedback(k=4, similar_within=0.25), {}
    _cycle(comp, scratch, 0, 0.200, "0.9", -11.0)
    _cycle(comp, scratch, 10, 0.205, "0.9", -16.0)   # same law, ~same state, much worse
    comp.on_observe(_obs(20, 0.21), SPACE, scratch)

    text = scratch["outcome"]
    assert "SAME law" in text and "WORSE" in text
    assert "outside your policy has changed" in text


def test_contextual_outcome_does_not_flag_a_different_state():
    """A worse score in a much worse state is explained by the state; flagging it would be noise."""
    comp, scratch = ContextualOutcomeFeedback(k=4, similar_within=0.10), {}
    _cycle(comp, scratch, 0, 0.05, "0.9", -11.0)
    _cycle(comp, scratch, 10, 0.40, "0.9", -16.0)    # same law, VERY different state
    comp.on_observe(_obs(20, 0.41), SPACE, scratch)
    assert "SAME law" not in scratch["outcome"]


def test_contextual_outcome_does_not_flag_a_different_law():
    comp, scratch = ContextualOutcomeFeedback(k=4), {}
    _cycle(comp, scratch, 0, 0.20, "0.9", -11.0)
    _cycle(comp, scratch, 10, 0.20, "0.3", -16.0)    # different law ⇒ the change explains itself
    comp.on_observe(_obs(20, 0.21), SPACE, scratch)
    assert "SAME law" not in scratch["outcome"]
