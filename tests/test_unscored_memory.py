"""UnscoredMemory must differ from its parent by the SCORE and by nothing else.

The claim it tests is that episodic memory's lock-in is carried by the outcome scores attached to
precedent rather than by recall of the precedent itself. That is only testable if the two components
retrieve the same episodes, in the same order, phrased the same way, so the single difference is the
deleted `-> outcome=X` clause.

Written after ContrastiveMemory failed by changing three things at once (ordering, person, and an
added spread line) and making lock-in worse. This one changes exactly one thing.
"""

from __future__ import annotations

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.harness import Outcome
from govsim.core.system import Observation
from govsim.harness import EpisodicMemory, UnscoredMemory

EPISODES = [(0.050, "0.9 if I > 0.02 else 0.0", 2.1), (0.052, "0.5", 1.4), (0.048, "0.2", 3.3)]


def _render(cls, k=3):
    c = cls(k=k)
    scratch: dict = {}
    for i, (inf, expr, score) in enumerate(EPISODES):
        view = Observation(vars={"I": inf, "S": 0.6, "t": float(i)}, scope="regent:0", t=i)
        reqs = [ActionRequest("regent:0", "set_lockdown", {"expr": expr})]
        c.on_outcome(view, reqs, Outcome(requests=reqs, metrics={"x": score}), scratch)
    c.on_observe(Observation(vars={"I": 0.050, "S": 0.6, "t": 9.0}, scope="regent:0", t=9),
                 ActionSpace(verbs=[], context_vars=[]), scratch)
    return scratch.get("memory", "")


def test_the_only_difference_is_the_deleted_score_clause():
    """Line for line, the parent's text minus everything from the arrow onward."""
    parent = _render(EpisodicMemory).splitlines()
    child = _render(UnscoredMemory).splitlines()
    assert len(parent) == len(child)
    for p, c in zip(parent, child):
        assert p.startswith(c), (p, c)
        assert "→ outcome" in p and "→ outcome" not in c


def test_no_score_survives_anywhere_in_the_rendering():
    text = _render(UnscoredMemory)
    for score in ("2.1", "1.4", "3.3", "outcome"):
        assert score not in text, score


def test_retrieval_order_is_unchanged():
    """Similarity ordering is inherited. Reordering would confound the score contrast — which is the
    mistake ContrastiveMemory made by changing ordering and phrasing at the same time."""
    parent, child = _render(EpisodicMemory), _render(UnscoredMemory)
    laws = lambda t: [ln.split("you did [")[1].split("]")[0] for ln in t.splitlines()]
    assert laws(parent) == laws(child)


def test_the_second_person_is_deliberately_KEPT():
    """The parent's phrasing is held fixed so the contrast isolates the score."""
    assert "you did" in _render(UnscoredMemory)


def test_states_and_laws_still_appear():
    text = _render(UnscoredMemory)
    assert "I=0.05" in text and "set_lockdown:0.5" in text


def test_an_empty_memory_injects_nothing():
    c = UnscoredMemory(k=3)
    scratch: dict = {}
    c.on_observe(Observation(vars={"I": 0.05}, scope="regent:0", t=0),
                 ActionSpace(verbs=[], context_vars=[]), scratch)
    assert "memory" not in scratch


def test_scorer_only_keys_stay_hidden():
    c = UnscoredMemory(k=2)
    scratch: dict = {}
    view = Observation(vars={"I": 0.05, "_regime_on": 1.0}, scope="regent:0", t=0)
    reqs = [ActionRequest("regent:0", "set_lockdown", {"expr": "0.5"})]
    c.on_outcome(view, reqs, Outcome(requests=reqs, metrics={"x": 1.0}), scratch)
    c.on_observe(Observation(vars={"I": 0.05}, scope="regent:0", t=1),
                 ActionSpace(verbs=[], context_vars=[]), scratch)
    assert "_regime_on" not in scratch["memory"]
