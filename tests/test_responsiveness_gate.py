"""The fifth validity gate: did the harness change BEHAVIOUR, or only the prompt?

These cases pin the distinction the gate exists to draw. The four earlier gates all pass on a model
that reads nothing, and the resulting flat ablation table is easy to misreport as "these components
do not help". The true statement is narrower and more interesting: below some capability, a model
does not consult its scaffold, so the scaffold cannot be measured on it.

Measured, and the reason this file exists: a 0.8B model emitted ``set_lockdown=0.5`` on 400/400
decisions with episodic memory live in 399 of those prompts. A 27B model on the identical harness,
worlds, and seeds spread its decisions over many policies (top share 50%, TV 0.32-0.38).
"""

from __future__ import annotations

from govsim.analysis.stats import policy_responsiveness


def test_constant_policy_is_not_responsive_even_when_the_channel_is_live():
    """The headline failure: 400 identical decisions, in both arms."""
    base = ["set_lockdown=0.5"] * 400
    treated = ["set_lockdown=0.5"] * 400
    rep = policy_responsiveness(base, treated)
    assert rep["responsive"] is False
    assert rep["degenerate"] is True
    assert rep["tv_distance"] == 0.0
    assert "ONE policy" in rep["reason"]


def test_a_trivial_perturbation_of_a_constant_is_still_not_responsive():
    """Measured qwen behaviour: 385/400 on one policy, 15 on a nonsense value.

    A naive "distributions differ" check passes here — the distributions genuinely do differ. It is
    the concentration that disqualifies the arm, so the gate must test both.
    """
    base = ["set_lockdown=0.5"] * 400
    treated = ["set_lockdown=0.5"] * 385 + ["set_lockdown=-0.5"] * 15
    rep = policy_responsiveness(base, treated)
    assert rep["tv_distance"] > 0.0          # they DO differ ...
    assert rep["top_share"] >= 0.96          # ... but 96% of decisions are one policy
    assert rep["responsive"] is False


def test_a_genuinely_responsive_arm_passes():
    """Measured gemma behaviour: a spread of policies that shifts when the channel is added."""
    base = ["set_lockdown=0.5"] * 50 + ["set_lockdown=0.9*I"] * 50
    treated = ["set_lockdown=0.5"] * 20 + ["set_lockdown=0.9*I"] * 40 + ["set_vaccination=0.5"] * 40
    rep = policy_responsiveness(base, treated)
    assert rep["responsive"] is True
    assert rep["degenerate"] is False
    assert rep["reason"] == ""


def test_a_live_channel_that_moves_nothing_is_flagged_even_without_degeneracy():
    """The subtler case: a varied policy mix that the channel fails to shift.

    Not degenerate — the agent is genuinely governing — but the treated distribution is the control
    distribution, so the contrast still measures nothing about the component.
    """
    mix = ["set_lockdown=0.5"] * 34 + ["set_lockdown=0.9*I"] * 33 + ["set_vaccination=0.2"] * 33
    rep = policy_responsiveness(mix, list(mix))
    assert rep["degenerate"] is False
    assert rep["responsive"] is False
    assert "not the behaviour" in rep["reason"]


def test_an_arm_that_enacted_nothing_is_never_silently_passed():
    rep = policy_responsiveness(["set_lockdown=0.5"], [])
    assert rep["responsive"] is False
    assert "enacted nothing" in rep["reason"]


def test_tv_distance_is_symmetric_and_bounded():
    a = ["x"] * 30 + ["y"] * 70
    b = ["x"] * 80 + ["z"] * 20
    fwd = policy_responsiveness(a, b)["tv_distance"]
    rev = policy_responsiveness(b, a)["tv_distance"]
    assert abs(fwd - rev) < 1e-12
    assert 0.0 <= fwd <= 1.0
    # Disjoint supports are maximally distant.
    assert policy_responsiveness(["p"] * 10, ["q"] * 10)["tv_distance"] == 1.0
