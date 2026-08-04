"""
Tests for the stock-flow-consistent fiscal world.

The load-bearing one is ``test_money_is_conserved_*``: this world's claim to be an economy rather
than a difference equation with economic variable names is the accounting identity, and an identity
that is not asserted is a comment. The rest pin the properties the platform needs before it will
believe a headroom number from this domain — bounded dynamics, per-seed variation, an unobservable
instrument efficacy, an interior optimum, and a control problem that a do-nothing policy loses.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from govsim.core.action import ActionRequest
from govsim.domains.economy.fiscal_sfc import FiscalSFCEconomy, FiscalStabilizationLoss
from govsim.domains.scalar.interface import Lever, ScalarLeverInterface

SHOCKED = {"shock_step": 100, "shock_params": {"tax_compliance": 0.35}}

#: A hand-written rule a competent finance ministry could plausibly legislate: lean spending
#: against the output gap, lean the rate against household balances (which is the only trace the
#: collapse leaves in the observables).
FEEDBACK_TAX = "min(0.4, max(0.02, 0.10 + 0.004 * (M_h - 250.0)))"
FEEDBACK_TRANSFER = "max(0.0, min(40.0, 10.0 + 0.35 * (y_potential - output)))"


def _iface() -> ScalarLeverInterface:
    return ScalarLeverInterface([
        Lever("set_tax_rate", (0.0, 0.6), "tax_rate"),
        Lever("set_transfer", (0.0, 40.0), "transfer"),
    ])


def _run(cfg: dict, seed: int, tax_expr: str, transfer_expr: str, horizon: int = 200):
    """Install the two lever expressions and roll the world forward; return (trajectory, system)."""
    system = FiscalSFCEconomy(cfg)
    system.reset(seed)
    report = _iface().apply(
        [
            ActionRequest("regent:0", "set_tax_rate", {"expr": tax_expr}),
            ActionRequest("regent:0", "set_transfer", {"expr": transfer_expr}),
        ],
        system,
    )
    assert not report.rejected, report.rejected
    trajectory = []
    for _ in range(horizon):
        system.step()
        trajectory.append(system.metrics())
    return trajectory, system


def _loss(trajectory, **kwargs) -> float:
    return FiscalStabilizationLoss(**kwargs).components(trajectory)["loss"]


#: Policy corners plus the interesting interior ones — every boundedness/conservation claim is
#: checked against all of them, because a clip that only holds for reasonable policies is not a bound.
_POLICIES = [
    ("0.0", "0.0"),                      # never act
    ("0.6", "40.0"),                     # max both levers
    ("0.0", "40.0"),                     # spend the state into its debt ceiling with no revenue
    ("0.6", "0.0"),                      # tax households down to zero balance
    ("0.10", "10.0"),                    # the interior optimum
    (FEEDBACK_TAX, FEEDBACK_TRANSFER),   # a feedback rule
]


# --- conservation: the identity this world is built on ---------------------------------------

@pytest.mark.parametrize("tax_expr,transfer_expr", _POLICIES)
def test_money_is_conserved_under_every_policy(tax_expr, transfer_expr):
    trajectory, system = _run(SHOCKED, 4, tax_expr, transfer_expr, horizon=1000)
    drift = max(abs(row["money_total"] - system.money_total) for row in trajectory)
    # Each period moves ONE number between two balances, so the only error possible is float
    # round-off accumulating over 1000 additions — nothing structural hides under a loose tolerance.
    assert drift < 1e-9, f"money leaked: {drift}"


def test_money_is_conserved_across_the_shock():
    trajectory, system = _run(SHOCKED, 1, "0.35", "30.0", horizon=200)
    before = trajectory[99]["money_total"]
    after = trajectory[-1]["money_total"]
    assert abs(before - system.money_total) < 1e-9
    assert abs(after - system.money_total) < 1e-9  # a collapsing tax take moves money, never destroys it


def test_stocks_respect_their_institutional_floors():
    # M_h >= 0 and M_g >= -debt_limit are not numerical guards: they are "households cannot remit
    # what they do not hold" and "the state cannot pay past its ceiling". Conservation turns each
    # floor into the other stock's ceiling, which is what makes the world bounded at all.
    for tax_expr, transfer_expr in _POLICIES:
        trajectory, system = _run(SHOCKED, 2, tax_expr, transfer_expr, horizon=1000)
        assert min(row["M_h"] for row in trajectory) >= -1e-9
        assert min(row["M_g"] for row in trajectory) >= -system.debt_limit - 1e-9
        assert max(row["M_h"] for row in trajectory) <= system.money_total + system.debt_limit + 1e-9
        assert max(row["M_g"] for row in trajectory) <= system.money_total + 1e-9


# --- bounded dynamics -------------------------------------------------------------------------

def test_no_policy_diverges_over_a_long_run():
    for tax_expr, transfer_expr in _POLICIES:
        trajectory, system = _run(SHOCKED, 5, tax_expr, transfer_expr, horizon=3000)
        outputs = [row["output"] for row in trajectory]
        assert all(np.isfinite(v) for v in outputs)
        assert min(outputs) >= 0.0  # demand-determined output is clipped positive
        # The analytic ceiling: a bounded wealth stock and a capped spending flow over a multiplier
        # denominator bounded below by (1 - alpha_income), with slack for the demand noise.
        ceiling = (
            system.alpha_wealth * (system.money_total + system.debt_limit)
            + system.gov_base_spend + system.transfer_cap + 10.0 * system.noise_sigma
        ) / (1.0 - system.alpha_income)
        assert max(outputs) <= ceiling


def test_zero_compliance_does_not_blow_the_world_up():
    # Total collection failure is the harshest point on the shock axis: revenue is nil, the state
    # runs a permanent deficit, and only the debt ceiling stops it. Check it saturates rather than
    # runs away — a comparison between two blow-ups is not a result.
    cfg = {"shock_step": 50, "shock_params": {"tax_compliance": 0.0}}
    trajectory, system = _run(cfg, 6, "0.30", "30.0", horizon=3000)
    assert all(np.isfinite(row["output"]) for row in trajectory)
    assert trajectory[-1]["M_g"] >= -system.debt_limit - 1e-9
    assert abs(trajectory[-1]["money_total"] - system.money_total) < 1e-9


# --- per-seed heterogeneity -------------------------------------------------------------------

def test_seeds_produce_genuinely_different_trajectories():
    a, _ = _run({}, 1, "0.25", "25.0")
    b, _ = _run({}, 2, "0.25", "25.0")
    assert [row["output"] for row in a] != [row["output"] for row in b]
    # Not merely different noise draws: the seed moves the structural parameters, so the same
    # policy leaves the two economies at different mean output. Without that a paired shared-seed
    # design pairs identical numbers and reports a zero-width interval as a finding.
    mean_a = np.mean([row["output"] for row in a])
    mean_b = np.mean([row["output"] for row in b])
    assert abs(mean_a - mean_b) > 1e-6


def test_same_seed_reproduces_the_trajectory():
    a, _ = _run({}, 11, "0.25", "25.0")
    b, _ = _run({}, 11, "0.25", "25.0")
    assert [row["output"] for row in a] == [row["output"] for row in b]


def test_reset_draws_new_parameters_and_restores_shocked_ones():
    system = FiscalSFCEconomy(SHOCKED)
    system.reset(3)
    alpha_first = system.alpha_income
    for _ in range(150):
        system.step()
    assert system.tax_compliance == 0.35
    system.reset(3)
    assert system.tax_compliance == 1.0  # a fresh run must start from the pre-shock instrument
    assert system.alpha_income == alpha_first  # …and reproduce the same seed's economy
    assert system.M_h + system.M_g == pytest.approx(system.money_total)


#: Every scalar parameter a ``shock_params`` entry can overwrite, with a value distinct from its
#: default. Enumerated rather than spot-checked because the failure is silent: an unrestored
#: parameter leaks into the NEXT run of the same object, and that run still looks self-consistent.
_SHOCKABLE = {
    "tax_compliance": 0.35, "tax_distortion": 0.9, "gov_base_spend": 25.0,
    "noise_sigma": 5.0, "debt_limit": 50.0, "alpha_income": 0.42, "alpha_wealth": 0.33,
    "y_potential": 85.0, "transfer_cap": 5.0, "tax_rate_cap": 0.2,
}


@pytest.mark.parametrize("name,value", sorted(_SHOCKABLE.items()))
def test_reset_restores_every_shockable_parameter(name, value):
    # ``y_potential`` is the one that made this a real bug rather than a tidiness rule: a fall in
    # potential output is the textbook macro shock, it is published in observe() AND read by the
    # objective, so a version that survived reset() would move the target of every later "fresh"
    # run of the same object — silently rescoring the whole arm against a goalpost nobody set.
    system = FiscalSFCEconomy({"shock_step": 2, "shock_params": {name: value}})
    system.reset(0)
    pristine = getattr(system, name)
    assert pristine != value, f"{name}: pick a shock value that differs from the default"
    for _ in range(5):
        system.step()
    assert getattr(system, name) == value, f"{name}: the shock never landed"
    system.reset(0)
    assert getattr(system, name) == pristine, f"{name} leaked across reset()"


def test_a_potential_output_shock_does_not_leak_into_the_next_run():
    # The end-to-end version of the above: what a re-used system object PUBLISHES after a reset.
    system = FiscalSFCEconomy({"shock_step": 2, "shock_params": {"y_potential": 85.0}})
    system.reset(0)
    for _ in range(10):
        system.step()
    system.reset(0)
    system.step()
    assert system.observe().vars["y_potential"] == 100.0
    assert system.metrics()["y_potential"] == 100.0


# --- the shock actually changes behaviour ------------------------------------------------------

def test_shock_changes_behaviour_only_after_the_shock_step():
    clean, _ = _run({}, 3, "0.10", "10.0")
    broken, _ = _run(SHOCKED, 3, "0.10", "10.0")
    pre = [(x["output"], y["output"]) for x, y in zip(clean[:100], broken[:100])]
    assert all(abs(x - y) < 1e-12 for x, y in pre)  # identical stochastic stream before the break
    post_clean = np.mean([row["output"] for row in clean[100:]])
    post_broken = np.mean([row["output"] for row in broken[100:]])
    # The same enacted policy now collects a third of the revenue, so the fiscal stance is far
    # looser than the institution believes it is and output overshoots potential.
    assert post_broken > post_clean * 1.2


def test_efficacy_shock_leaves_the_instrument_price_untouched():
    """The asymmetry the whole design rests on: same enacted rate, same bill, a fraction of the yield."""
    clean, _ = _run({}, 7, "0.30", "20.0")
    broken, _ = _run({"shock_step": 0, "shock_params": {"tax_compliance": 0.2}}, 7, "0.30", "20.0")
    assert clean[-1]["cum_tax_cost"] == pytest.approx(broken[-1]["cum_tax_cost"])
    assert broken[-1]["M_g"] < clean[-1]["M_g"]  # the yield, however, is gone


# --- the efficacy parameter is unobservable ----------------------------------------------------

def test_tax_compliance_is_absent_from_observe():
    system = FiscalSFCEconomy(SHOCKED)
    system.reset(0)
    for _ in range(120):
        system.step()
    assert system.tax_compliance == 0.35  # it is doing work…
    variables = system.observe().vars
    assert "tax_compliance" not in variables
    # …and nothing published stands in for it: no observable equals it, at any step.
    assert all(value != system.tax_compliance for value in variables.values())


def test_a_policy_expression_cannot_reference_the_efficacy():
    system = FiscalSFCEconomy(SHOCKED)
    system.reset(0)
    report = _iface().apply(
        [ActionRequest("regent:0", "set_tax_rate", {"expr": "0.3 / tax_compliance"})], system
    )
    assert report.applied == []
    assert "tax_compliance" in report.rejected[0][1]


def test_metrics_record_what_the_institution_did():
    _, system = _run(SHOCKED, 0, "0.22", "18.0", horizon=10)
    metrics = system.metrics()
    assert metrics["tax_rate"] == pytest.approx(0.22)
    assert metrics["transfer"] == pytest.approx(18.0)
    assert "tax_compliance" not in metrics


# --- levers -------------------------------------------------------------------------------------

def test_levers_are_reevaluated_and_clipped_every_step():
    system = FiscalSFCEconomy({})
    system.reset(0)
    _iface().apply(
        [
            ActionRequest("regent:0", "set_tax_rate", {"expr": "10.0 * output"}),
            ActionRequest("regent:0", "set_transfer", {"expr": "-5.0"}),
        ],
        system,
    )
    for _ in range(5):
        system.step()
        assert 0.0 <= system.tax_rate <= 0.6
        assert 0.0 <= system.transfer <= 40.0
    assert system.tax_rate == 0.6 and system.transfer == 0.0


# --- the objective --------------------------------------------------------------------------------

def test_empty_post_shock_window_scores_worst_case():
    trajectory = [{"output": 100.0, "y_potential": 100.0, "deficit": 0.0,
                   "cum_tax_cost": 0.1 * i, "t": float(i)} for i in range(5)]
    components = FiscalStabilizationLoss(post_shock_step=100).components(trajectory)
    assert components["post_loss"] == float("inf")
    assert components["loss"] < float("inf")  # the full-horizon score is unaffected


def test_tax_cost_is_differenced_across_the_window_not_read_as_a_running_total():
    # Reading cum_tax_cost undifferenced would bill the post-shock policy for every rate enacted
    # before the break, turning the score into a clock. That bug shipped here once.
    trajectory = [{"output": 100.0, "y_potential": 100.0, "deficit": 0.0,
                   "cum_tax_cost": 0.1 * i, "t": float(i)} for i in range(10)]
    components = FiscalStabilizationLoss(lam=1.0, post_shock_step=5).components(trajectory)
    assert components["post_tax_cost"] == pytest.approx(0.4)
    assert components["cum_tax_cost"] == pytest.approx(0.9)


def test_objective_prefers_output_at_potential_and_a_flat_balance():
    on_target = [{"output": 100.0, "y_potential": 100.0, "deficit": 0.0, "cum_tax_cost": 0.0}] * 10
    off_target = [{"output": 60.0, "y_potential": 100.0, "deficit": 0.0, "cum_tax_cost": 0.0}] * 10
    drifting = [{"output": 100.0, "y_potential": 100.0, "deficit": 20.0, "cum_tax_cost": 0.0}] * 10
    objective = FiscalStabilizationLoss()
    assert objective.evaluate(on_target) > objective.evaluate(off_target)
    assert objective.evaluate(on_target) > objective.evaluate(drifting)


def test_objective_loud_fails_on_a_mis_wired_system():
    with pytest.raises(KeyError):
        FiscalStabilizationLoss().evaluate([{"infected": 0.1}])


def test_describe_states_the_mandate_including_the_cost_asymmetry():
    text = FiscalStabilizationLoss().describe()
    assert "MANDATE" in text
    assert "ENACT" in text and "not on what it collects" in text


# --- the world has a control problem ---------------------------------------------------------------

@pytest.mark.parametrize("cfg", [{}, SHOCKED])
def test_a_sensible_policy_beats_doing_nothing(cfg):
    seeds = range(6)
    passive = np.mean([_loss(_run(cfg, seed, "0.0", "0.0")[0]) for seed in seeds])
    governed = np.mean([_loss(_run(cfg, seed, FEEDBACK_TAX, FEEDBACK_TRANSFER)[0]) for seed in seeds])
    assert governed < passive / 2.0, f"passive={passive}, governed={governed}"


def test_the_optimum_is_interior_in_both_levers():
    # If the best policy were a corner — always max, or never act — no shock could move it and the
    # measured adaptation headroom would be zero by construction, whatever the agent did.
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    transfers = [0.0, 10.0, 20.0, 30.0, 40.0]
    scores = {
        (rate, transfer): np.mean([_loss(_run({}, seed, f"{rate}", f"{transfer}")[0]) for seed in range(3)])
        for rate, transfer in itertools.product(rates, transfers)
    }
    best_rate, best_transfer = min(scores, key=scores.get)
    assert 0.0 < best_rate < 0.6
    assert 0.0 < best_transfer < 40.0
