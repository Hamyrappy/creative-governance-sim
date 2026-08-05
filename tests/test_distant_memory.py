"""`DistantMemory` must differ from its control in exactly one respect: the retrieval order.

The arm exists to isolate proximity from bank contents. That only works if nothing else moved, and
"nothing else moved" is the kind of claim that is true when written and false three edits later. Two
of this repository's more expensive mistakes were of exactly this shape — a memory arm that silently
carried a different episode count, and a foreign-memory arm whose donor bank turned out to be a
different size from the one the probe measured.

So these tests compare the two components' rendered prompts directly rather than inspecting either
in isolation: same bank, same episodes retrieved by count, same wording, opposite end of the ranking.
"""

from __future__ import annotations

import pytest

from govsim.core.action import ActionSpace
from govsim.core.system import Observation
from govsim.harness import DistantMemory, EpisodicMemory

SPACE = ActionSpace(verbs=[], context_vars=[])


def _bank() -> list[dict]:
    """Episodes at increasing distance from the query state below, each with a distinct law."""
    return [
        {"state": {"I": 0.020, "S": 0.80}, "actions": [{"verb": "set_lockdown", "expr": "0.90"}],
         "score": -2.0},
        {"state": {"I": 0.021, "S": 0.79}, "actions": [{"verb": "set_lockdown", "expr": "0.85"}],
         "score": -2.1},
        {"state": {"I": 0.120, "S": 0.60}, "actions": [{"verb": "set_lockdown", "expr": "0.40"}],
         "score": -5.0},
        {"state": {"I": 0.300, "S": 0.40}, "actions": [{"verb": "set_lockdown", "expr": "0.10"}],
         "score": -9.0},
    ]


def _view() -> Observation:
    return Observation(vars={"I": 0.020, "S": 0.80}, scope="regent:0", t=10)


def _render(cls, k: int) -> str:
    m = cls(k=k)
    m.episodes = [dict(e) for e in _bank()]
    scratch: dict = {}
    m.on_observe(_view(), SPACE, scratch)
    return scratch.get("memory", "")


def test_it_retrieves_the_opposite_end_of_the_same_bank():
    near, far = _render(EpisodicMemory, 1), _render(DistantMemory, 1)
    assert "0.9" in near and "I=0.02" in near, near
    assert "0.1" in far and "I=0.3" in far, far
    assert near != far


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_it_returns_the_same_number_of_episodes_as_its_control(k):
    """Episode count is the classic silent confound: an arm that shows fewer precedents is testing
    dose, not order."""
    near, far = _render(EpisodicMemory, k), _render(DistantMemory, k)
    assert len(far.splitlines()) == len(near.splitlines()) == min(k, len(_bank()))


def test_the_wording_is_byte_identical_apart_from_the_episodes_chosen():
    """Same second person, same arrow, same score clause. If the phrasing diverges, the arm is a
    presentation manipulation as well as a retrieval one, and the three failed presentation repairs
    say that is not a neutral change."""
    near, far = _render(EpisodicMemory, 4), _render(DistantMemory, 4)
    # With k = |bank| both return every episode, so the RENDERED SET must be identical and only the
    # order differs. That is the strongest available check on the wording.
    assert sorted(near.splitlines()) == sorted(far.splitlines())
    assert near.splitlines() != far.splitlines(), "the ordering did not actually invert"


def test_it_still_accumulates_the_agents_own_episodes():
    """The whole point is that the BANK is unchanged. If it ever stops appending, the arm silently
    becomes a fixed-bank arm like ForeignMemory and confounds the thing it exists to isolate."""
    assert DistantMemory.on_outcome is EpisodicMemory.on_outcome
    assert DistantMemory.reset is EpisodicMemory.reset


def test_a_single_episode_bank_cannot_distinguish_the_two():
    """Stated in the component docstring and load-bearing for how the result is read: at the first
    transition the nearest and furthest episode are the same one, so this arm cannot differ there.
    Any reported difference at transition 1 would be noise or a bug."""
    m_near, m_far = EpisodicMemory(k=3), DistantMemory(k=3)
    one = [_bank()[0]]
    for m in (m_near, m_far):
        m.episodes = [dict(e) for e in one]
    s1: dict = {}
    s2: dict = {}
    m_near.on_observe(_view(), SPACE, s1)
    m_far.on_observe(_view(), SPACE, s2)
    assert s1["memory"] == s2["memory"]


def test_an_empty_bank_emits_nothing_rather_than_an_empty_section():
    """An empty string in `scratch["memory"]` would render a headed but blank precedent section,
    which is a different prompt from having no section at all."""
    scratch: dict = {}
    DistantMemory(k=3).on_observe(_view(), SPACE, scratch)
    assert "memory" not in scratch
