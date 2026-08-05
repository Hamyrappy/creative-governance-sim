"""The pinned monetary anchors must still describe the world they were calibrated on.

``economy_experiments._World.calibrated`` hardcodes three reference laws so a published ``R`` is
reproducible from the source tree and cannot silently move when someone re-runs a calibration. The
cost of pinning is drift: if the world's prices, shock or horizon change, the pinned laws quietly
stop being the frozen/best-fixed/switching policies and every ``R`` computed against them becomes a
ratio to nothing in particular.

That failure is invisible in the numbers — the arms still run, the table still fills in — which is
why it needs a test rather than vigilance. This one re-derives the anchors' ORDERING and their
measured ratios from the live world and fails when the world has moved out from under them.

These are the same checks ``scripts/analyze_matrix.py`` runs as its anchor-consistency gate, at a
tolerance loose enough to survive seed noise and tight enough to catch a re-priced world.
"""

from __future__ import annotations

import pytest

from govsim.core.runner import Runner
from govsim.experiments import get

SEEDS = list(range(8))

#: MEASURED at 20 seeds by ``scripts/run_matrix.py --arms monetary-refs``. The 8-seed calibration
#: that chose the laws reported 2980.56 / 2597.61 / 1910.81; re-deriving at 20 seeds moved every
#: anchor by under 3% and left adaptation headroom at 1.358x against the calibrated 1.359x.
EXPECTED = {
    "monetary_frozen": 2927.81,
    "monetary_best_fixed": 2541.01,
    "monetary_switching": 1871.44,
}


def _loss(name: str, seeds: list[int]) -> float:
    exp = get(name)
    exp.seeds = list(seeds)
    recs = Runner().run(exp)
    return sum(r.components["regent:0"]["loss"] for r in recs) / len(recs)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_each_anchor_still_scores_what_it_scored_when_it_was_pinned(name):
    """A 25% band: wide enough for an 8-seed subsample of a 20-seed mean, narrow enough that a
    re-priced world or a moved shock step cannot slip through."""
    got = _loss(name, SEEDS)
    expected = EXPECTED[name]
    assert abs(got - expected) / expected < 0.25, (name, got, expected)


def test_the_anchors_are_ordered_the_way_their_definitions_require():
    """switching <= best_fixed <= frozen, by construction and not by luck.

    Switching SUBSUMES not switching — it may always play the best fixed law on both legs — so a
    switching reference that loses to ``best_fixed`` is a broken search, not a finding. That exact
    contradiction is what exposed the joint-vs-composed bug in ``calibrate_switching``, and later
    the ``top_k`` shortlist truncation, so it is worth asserting every time.
    """
    frozen = _loss("monetary_frozen", SEEDS)
    best_fixed = _loss("monetary_best_fixed", SEEDS)
    switching = _loss("monetary_switching", SEEDS)
    assert switching <= best_fixed, (switching, best_fixed)
    assert best_fixed <= frozen, (best_fixed, frozen)


def test_adaptation_headroom_is_still_large_enough_to_host_the_experiment():
    """The reason this world is the flagship: adaptation headroom, not staleness.

    Staleness alone would be satisfied by a world where a better CONSTANT recovers everything, and
    an agent that adapts brilliantly would score the same there as one that cannot adapt at all.
    """
    best_fixed = _loss("monetary_best_fixed", SEEDS)
    switching = _loss("monetary_switching", SEEDS)
    adaptation = best_fixed / switching
    assert adaptation > 1.25, adaptation


def test_the_optimum_is_interior_so_the_shock_can_move_it():
    """A corner optimum is regime-invariant: no shock can move it, so the world reports a null
    whatever governs it. Bracketing the standing rule between both corners is the cheap check."""
    standing = _loss("monetary_standing_rule", SEEDS)
    assert standing < _loss("monetary_do_nothing", SEEDS)
    assert standing < _loss("monetary_max_lever", SEEDS)


def test_the_factorial_registers_all_eight_cells_and_each_carries_its_components():
    from govsim.experiments import available

    arms = [n for n in available() if n.startswith("monetary_llm_")]
    assert len(arms) == 8, arms
    bare = get("monetary_llm_bare")
    full = get("monetary_llm_trace_outcome_memory")
    assert len(bare.harness.components) == 0
    assert len(full.harness.components) == 3
    # The single-factor cells must carry exactly one component, or the "factorial" is mislabelled
    # arms and every main effect is confounded with whatever was actually installed.
    for suffix, n in (("trace", 1), ("outcome", 1), ("memory", 1), ("trace_outcome", 2)):
        assert len(get(f"monetary_llm_{suffix}").harness.components) == n, suffix
