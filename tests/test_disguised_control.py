"""The semantic-obfuscation control must differ from the monetary world in NAMES and nothing else.

If any dynamic, any number, or any part of the loss differs between the two, the comparison they
exist to support — "is the regent's monetary failure a domain prior, or is the control problem just
hard?" — is confounded by whatever else moved, and the experiment answers neither question.

These tests are the guarantee. They are strict on purpose: exact equality on trajectories, not
approximate agreement.
"""

from __future__ import annotations

import pytest

from govsim.core.action import ActionRequest
from govsim.domains.economy.disguised import (
    ALIASES,
    DisguisedEconomy,
    DisguisedLoss,
    undisguise,
)
from govsim.domains.economy.monetary import DualMandateLoss, MonetaryEconomy
from govsim.domains.scalar import Lever, ScalarLeverInterface

SEEDS = range(5)
STEPS = 120  # past the break at 100


def _roll(cls, seed: int, expr: str | None = None, verb: str = "set_policy_rate"):
    system = cls({"seed": seed})
    if expr is not None:
        iface = ScalarLeverInterface([Lever(verb, (0.0, 12.0), "policy_rate")])
        report = iface.apply([ActionRequest("regent:0", verb, {"expr": expr})], system)
        assert report.applied, report.rejected
    traj = []
    for _ in range(STEPS):
        system.step()
        traj.append(system.metrics())
    return traj


# --- the dynamics must be identical --------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_uncontrolled_trajectories_are_exactly_identical(seed):
    a = _roll(DisguisedEconomy, seed)
    b = _roll(MonetaryEconomy, seed)
    assert [sorted(r.items()) for r in a] == [sorted(r.items()) for r in b]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_control_law_produces_the_same_trajectory(seed):
    """A constant is used because it is expressible in BOTH namespaces without translation."""
    a = _roll(DisguisedEconomy, seed, "5.0")
    b = _roll(MonetaryEconomy, seed, "5.0")
    assert [sorted(r.items()) for r in a] == [sorted(r.items()) for r in b]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_loss_is_numerically_identical(seed):
    traj = _roll(MonetaryEconomy, seed, "5.0")
    assert DisguisedLoss(lam=0.25, post_shock_step=100).components(traj) == \
           DualMandateLoss(lam=0.25, post_shock_step=100).components(traj)


def test_the_break_still_lands_and_still_breaks_transmission():
    """The control is worthless if the disguised world quietly lost its shock."""
    system = DisguisedEconomy({"shock_step": 100, "shock_params": {"transmission": 0.10}})
    before = system.transmission
    for _ in range(120):
        system.step()
    assert system.transmission < before
    assert system.transmission == pytest.approx(0.10)


# --- only the names differ -----------------------------------------------------------------------

def test_no_economic_name_survives_into_what_the_regent_observes():
    """The whole point. A leak here silently restores the prior and the control measures nothing."""
    system = DisguisedEconomy({})
    for _ in range(30):
        system.step()
    published = " ".join(system.observe().vars).lower()
    for giveaway in ("inflation", "output", "rate", "r_star", "policy", "gap", "target_rate"):
        assert giveaway not in published, (giveaway, published)


def test_every_original_observable_has_an_alias_and_none_is_dropped():
    """A silently dropped variable would make the disguised regent strictly less informed, which
    would confound the comparison in the direction that flatters our hypothesis."""
    plain = MonetaryEconomy({}).observe().vars
    disguised = DisguisedEconomy({}).observe().vars
    assert len(plain) == len(disguised)
    assert {ALIASES.get(k, k) for k in plain} == set(disguised)
    for k, v in plain.items():
        assert disguised[ALIASES.get(k, k)] == v


def test_metrics_are_NOT_renamed():
    """metrics() feeds the objective and the result store, never a prompt. Renaming it would break
    every analysis script that reads policy_rate, silently and only for this world."""
    assert sorted(DisguisedEconomy({}).metrics()) == sorted(MonetaryEconomy({}).metrics())


def test_the_mandate_states_the_same_numbers_with_no_economics():
    plain = DualMandateLoss(lam=0.25).describe()
    hidden = DisguisedLoss(lam=0.25).describe()
    assert plain != hidden
    for giveaway in ("inflation", "output gap", "policy rate", "central bank", "economy",
                     "borrowing", "monetary"):
        assert giveaway not in hidden.lower(), giveaway
    # The numbers, and the warning that the instrument may stop working, must both survive: without
    # the warning this stops testing "does the prior override the instruction" and starts testing
    # "was there an instruction", which is a question already answered elsewhere in this project.
    for kept in ("0.25", "0.5", "2.0", "whether or not the lever affects the system"):
        assert kept in hidden, kept


def test_undisguise_round_trips():
    for original, alias in ALIASES.items():
        assert undisguise(alias) == original
    assert undisguise("t") == "t"  # untouched names pass through
