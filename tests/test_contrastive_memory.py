"""ContrastiveMemory must change the FRAMING of precedent and nothing else.

The claim it exists to support is causal and narrow: that plain episodic memory suppresses policy
revision because of how precedent is presented, not because of what is retrieved. That claim is only
testable if the two components retrieve the same episodes and report the same numbers. If the
contrastive variant also retrieved differently, or added a number the parent withholds, any
behavioural difference would be attributable to the extra information and the experiment would
answer a question nobody asked.

So these tests are mostly equalities.
"""

from __future__ import annotations

import re

import pytest

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.harness import Outcome
from govsim.core.system import Observation
from govsim.harness import ContrastiveMemory, EpisodicMemory

EPISODES = [
    (0.050, "0.9 if I > 0.02 else 0.0", 2.1),
    (0.052, "0.5", 1.4),
    (0.048, "0.9*(I/0.02)", 3.3),
    (0.090, "0.1", 0.2),
]


def _fill(component, episodes=EPISODES):
    scratch: dict = {}
    for i, (inf, expr, score) in enumerate(episodes):
        view = Observation(vars={"I": inf, "S": 0.6, "t": float(i)}, scope="regent:0", t=i)
        reqs = [ActionRequest("regent:0", "set_lockdown", {"expr": expr})]
        component.on_outcome(view, reqs, Outcome(requests=reqs, metrics={"x": score}), scratch)
    return scratch


def _render(component, current_i=0.050):
    scratch = _fill(component)
    component.on_observe(
        Observation(vars={"I": current_i, "S": 0.6, "t": 9.0}, scope="regent:0", t=9),
        ActionSpace(verbs=[], context_vars=[]),
        scratch,
    )
    return scratch.get("memory", "")


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"-?\d+\.?\d*", text))


# --- same information -----------------------------------------------------------------------------

def test_both_variants_retrieve_the_same_episodes():
    """Retrieval is inherited unchanged; only the rendering is overridden."""
    plain, contrastive = EpisodicMemory(k=3), ContrastiveMemory(k=3)
    _fill(plain), _fill(contrastive)
    view = {"I": 0.050, "S": 0.6, "t": 9.0}
    rank_p = sorted(plain.episodes, key=lambda e: plain._distance(view, e["state"]))[:3]
    rank_c = sorted(contrastive.episodes, key=lambda e: contrastive._distance(view, e["state"]))[:3]
    assert [e["actions"] for e in rank_p] == [e["actions"] for e in rank_c]


def test_the_same_laws_and_scores_appear_in_both_renderings():
    plain, contrastive = _render(EpisodicMemory(k=3)), _render(ContrastiveMemory(k=3))
    for law in ("0.9 if I > 0.02 else 0.0", "0.5", "0.9*(I/0.02)"):
        assert law in plain and law in contrastive, law
    # Every score the parent shows must appear in the child. The child additionally states the
    # min and max of that same set, which are values already on screen.
    assert _numbers(plain) <= _numbers(contrastive)


def test_the_contrastive_variant_adds_no_episode_the_parent_withheld():
    """The 4th, distant episode must not leak in — that would be extra information, not framing."""
    assert "0.1" not in _render(ContrastiveMemory(k=3)).split("(comparable situations")[0]


# --- different framing ------------------------------------------------------------------------

def test_episodes_are_ordered_best_outcome_first():
    """The parent orders by similarity, which says which precedent is most APPLICABLE and nothing
    about which is most successful."""
    lines = [ln for ln in _render(ContrastiveMemory(k=3)).splitlines() if ln.startswith("-")]
    scores = [float(re.search(r"scored (-?\d+\.?\d*)", ln).group(1)) for ln in lines]
    assert scores == sorted(scores, reverse=True), scores


def test_the_second_person_is_gone():
    """"you did X" is an invitation to imitate, and the measured behaviour is that it is accepted."""
    text = _render(ContrastiveMemory(k=3)).lower()
    assert "you did" not in text
    assert "you " not in text.replace("outcomes", "")
    assert "a law of the form" in text


def test_the_spread_between_outcomes_is_stated_explicitly():
    """The fact that makes a choice a choice: comparable situations produced different results."""
    text = _render(ContrastiveMemory(k=3))
    assert "1.4" in text and "3.3" in text
    assert "options to weigh rather than a precedent to follow" in text


def test_no_spread_line_when_every_retrieved_outcome_is_identical():
    """Announcing a range of zero would be noise, and would assert a choice that does not exist."""
    c = ContrastiveMemory(k=3)
    scratch = _fill(c, [(0.05, "0.5", 2.0), (0.051, "0.6", 2.0), (0.049, "0.7", 2.0)])
    c.on_observe(Observation(vars={"I": 0.05, "S": 0.6, "t": 9.0}, scope="regent:0", t=9),
                 ActionSpace(verbs=[], context_vars=[]), scratch)
    assert "comparable situations have produced outcomes" not in scratch["memory"]


# --- inherited contracts must survive the override --------------------------------------------

def test_scorer_only_keys_are_still_hidden():
    c = ContrastiveMemory(k=2)
    scratch: dict = {}
    view = Observation(vars={"I": 0.05, "_regime_on": 1.0, "cum_cost": 99.0}, scope="regent:0", t=0)
    reqs = [ActionRequest("regent:0", "set_lockdown", {"expr": "0.5"})]
    c.on_outcome(view, reqs, Outcome(requests=reqs, metrics={"x": 1.0}), scratch)
    c.on_observe(Observation(vars={"I": 0.05}, scope="regent:0", t=1),
                 ActionSpace(verbs=[], context_vars=[]), scratch)
    assert "_regime_on" not in scratch["memory"]
    assert "cum_cost" not in scratch["memory"]


def test_an_empty_memory_injects_nothing():
    c = ContrastiveMemory(k=3)
    scratch: dict = {}
    c.on_observe(Observation(vars={"I": 0.05}, scope="regent:0", t=0),
                 ActionSpace(verbs=[], context_vars=[]), scratch)
    assert "memory" not in scratch


def test_a_single_episode_renders_without_a_spread_claim():
    c = ContrastiveMemory(k=3)
    scratch = _fill(c, [(0.05, "0.5", 2.0)])
    c.on_observe(Observation(vars={"I": 0.05, "S": 0.6, "t": 9.0}, scope="regent:0", t=9),
                 ActionSpace(verbs=[], context_vars=[]), scratch)
    assert "a law of the form" in scratch["memory"]
    assert "comparable situations have produced" not in scratch["memory"]
