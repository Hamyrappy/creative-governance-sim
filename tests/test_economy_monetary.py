"""
Tests for ``MonetaryEconomy`` — the stabilization economy whose policy rate can be disconnected
from demand mid-run.

The tests are organized around the preconditions this platform learned to check BEFORE believing
any result from a world: the dynamics cannot diverge (a comparison between two blow-ups is not a
result), seeds are genuinely different economies (or the paired statistics are vacuous), the
instrument's efficacy is unobservable (or there is nothing to infer), the optimum is interior and
the break MOVES it (or no policy could have adapted), and the objective's window handling does not
turn the score into a clock.
"""

from __future__ import annotations

import numpy as np
import pytest

from govsim.core.action import ActionRequest
from govsim.core.sandbox import compile_expr
from govsim.domains.economy.monetary import DualMandateLoss, MonetaryEconomy
from govsim.domains.scalar.interface import Lever, ScalarLeverInterface

SHOCK_STEP = 100
HORIZON = 200
RATE_RANGE = (0.0, 12.0)

#: The intended break: transmission collapses, the rate's cost is untouched.
SHOCKED = {"shock_step": SHOCK_STEP, "shock_params": {"transmission": 0.10}}
UNSHOCKED: dict = {"shock_step": None}


def _iface() -> ScalarLeverInterface:
    return ScalarLeverInterface([Lever("set_policy_rate", RATE_RANGE, "policy_rate")])


def _run(expr: str, seed: int, cfg: dict, horizon: int = HORIZON) -> list[dict[str, float]]:
    system = MonetaryEconomy(cfg)
    system.reset(seed)
    report = _iface().apply([ActionRequest("regent:0", "set_policy_rate", {"expr": expr})], system)
    assert not report.rejected, report.rejected
    trajectory = [system.metrics()]
    for _ in range(horizon):
        system.step()
        trajectory.append(system.metrics())
    return trajectory


def _mean_loss(expr: str, cfg: dict, seeds: range, key: str = "loss",
               post_shock_step: int | None = None, lam: float = 0.15) -> float:
    objective = DualMandateLoss(lam=lam, post_shock_step=post_shock_step)
    return float(np.mean([objective.components(_run(expr, s, cfg))[key] for s in seeds]))


# --- bounded dynamics -----------------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    "12.0",                                             # permanently punitive (maximal lever)
    "0.0",                                              # permanently stimulative
    "r_star + inflation_target",                        # do-nothing: hold the neutral stance
    "0.0 if inflation > inflation_target else 12.0",    # perverse: cut into a boom, hike into a slump
    "policy_rate + 5.0 * output_gap",                   # a runaway integrator
    "12.0 - 4.0 * inflation",                           # violates the Taylor principle
])
def test_no_policy_can_make_the_economy_diverge(expr):
    """The claimed invariant: both states stay inside the clip band, forever, under ANY policy."""
    system = MonetaryEconomy(SHOCKED)
    system.reset(7)
    _iface().apply([ActionRequest("regent:0", "set_policy_rate", {"expr": expr})], system)
    for _ in range(5_000):
        info = system.step()
        assert not info.terminated
        assert np.isfinite(system.output_gap) and np.isfinite(system.inflation)
        assert abs(system.output_gap) <= system.gap_band
        assert abs(system.inflation) <= system.inflation_band
        lo, hi = system.rate_range
        assert lo <= system.policy_rate <= hi


def _spectral_radius(rho_y: float, phi: float, kappa: float, transmission: float,
                     rho_pi: float) -> float:
    return float(max(abs(np.linalg.eigvals(
        np.array([[rho_y, kappa * transmission], [phi, rho_pi]])))))


def test_open_loop_dynamics_are_a_contraction_for_every_seed():
    """The band is the guarantee of last resort; the per-seed draws are clipped so the plant is a
    contraction at a constant rate in the first place — no seed is handed an explosive economy."""
    system = MonetaryEconomy(UNSHOCKED)
    for seed in range(200):
        system.reset(seed)
        assert _spectral_radius(system.rho_y, system.phi, system.kappa, system.transmission,
                                system.rho_pi) < 1.0


def test_contraction_holds_at_the_CORNERS_of_the_draw_bands_not_merely_on_sampled_seeds():
    """Sampling 200 seeds shows no seed *happened* to be explosive; the claim is that none *can*
    be. The draws are clipped into ``rho_y_bounds`` × ``phi_bounds``, the spectral radius is
    increasing in both, and transmission ≤ 1 — so checking the four corners at the worst
    transmission settles every seed at once. Without this, widening a band by one edit would pass
    the sampled test for a long time and then hand some unlucky seed an explosive economy."""
    system = MonetaryEconomy(UNSHOCKED)
    for rho_y in system.rho_y_bounds:
        for phi in system.phi_bounds:
            assert _spectral_radius(rho_y, phi, system.kappa, 1.0, system.rho_pi) < 1.0


# --- per-seed heterogeneity -----------------------------------------------------------------

def test_seeds_draw_genuinely_different_economies():
    drawn = []
    system = MonetaryEconomy(UNSHOCKED)
    for seed in range(12):
        system.reset(seed)
        drawn.append((system.rho_y, system.phi, system.noise_scale))
    assert len({d[0] for d in drawn}) == 12
    assert len({d[1] for d in drawn}) == 12
    assert len({d[2] for d in drawn}) == 12


def test_seeds_produce_different_trajectories_and_a_seed_reproduces_its_own():
    a = _run("5.0", 0, UNSHOCKED, horizon=50)
    b = _run("5.0", 1, UNSHOCKED, horizon=50)
    a_again = _run("5.0", 0, UNSHOCKED, horizon=50)
    assert [r["output_gap"] for r in a] == [r["output_gap"] for r in a_again]
    assert [r["output_gap"] for r in a] != [r["output_gap"] for r in b]
    assert [r["inflation"] for r in a] != [r["inflation"] for r in b]


def test_no_module_global_rng():
    """A second seeding of the same object must reproduce, i.e. nothing leaks in from a global."""
    system = MonetaryEconomy(UNSHOCKED)
    system.reset(3)
    first = [(system.step(), system.output_gap)[1] for _ in range(20)]
    system.reset(3)
    assert [(system.step(), system.output_gap)[1] for _ in range(20)] == first


# --- the shock ------------------------------------------------------------------------------

def test_shock_leaves_the_pre_break_run_identical_and_changes_what_follows():
    expr = "6.5"
    plain = _run(expr, 5, UNSHOCKED)
    broken = _run(expr, 5, SHOCKED)
    for row_a, row_b in zip(plain[:SHOCK_STEP + 1], broken[:SHOCK_STEP + 1]):
        assert row_a["output_gap"] == row_b["output_gap"]
        assert row_a["inflation"] == row_b["inflation"]
    assert plain[-1]["output_gap"] != broken[-1]["output_gap"]
    # A restrictive stance that no longer reaches demand leaves the economy hotter, not cooler.
    assert broken[-1]["output_gap"] > plain[-1]["output_gap"]


def test_shock_overwrites_only_efficacy_and_reset_restores_it():
    system = MonetaryEconomy(SHOCKED)
    system.reset(0)
    assert system.transmission == 1.0
    for _ in range(SHOCK_STEP + 1):
        system.step()
    assert system.transmission == 0.10
    assert system.rate_cost == system.rate_cost0  # the instrument's PRICE is untouched
    system.reset(0)
    assert system.transmission == 1.0


@pytest.mark.parametrize("attr,shocked_value", [
    ("transmission", 0.10), ("kappa", 0.90), ("rate_cost", 5.0), ("demand_drift", 4.0),
    ("rho_pi", 0.90), ("r_star", 9.0), ("inflation_target", 9.0),
    # The two that used to leak. They are the boundedness guarantee itself, so a leaked shrunken
    # band silently re-parameterizes every later run of the same object — including the pre-shock
    # arm the shocked arm is compared against.
    ("gap_band", 3.0), ("inflation_band", 3.0),
])
def test_no_shocked_parameter_survives_a_reset(attr, shocked_value):
    """``reset`` must hand back a pristine plant for EVERY parameter ``shock_params`` can reach."""
    system = MonetaryEconomy({"shock_step": 0, "shock_params": {attr: shocked_value}})
    system.reset(0)
    pristine = getattr(system, attr)
    assert pristine != shocked_value, "pick a shocked value that differs from the default"
    system.step()
    assert getattr(system, attr) == shocked_value  # the shock really did land
    system.reset(0)
    assert getattr(system, attr) == pristine


def test_cost_accrues_on_the_rate_set_not_on_what_it_achieves():
    """The asymmetry that makes an efficacy collapse expensive to ignore."""
    working = _run("7.0", 4, {"shock_step": None, "transmission": 1.0})
    broken = _run("7.0", 4, {"shock_step": None, "transmission": 0.02})
    assert working[-1]["cum_cost"] == pytest.approx(broken[-1]["cum_cost"])
    assert working[-1]["output_gap"] != broken[-1]["output_gap"]


# --- unobservability ------------------------------------------------------------------------

def test_efficacy_is_not_observable():
    system = MonetaryEconomy(UNSHOCKED)
    system.reset(0)
    variables = system.observe().vars
    for hidden in ("transmission", "kappa", "demand_drift", "rho_y", "phi"):
        assert hidden not in variables
    # …and it is not merely absent from the dict: it is outside the sandbox whitelist, so a policy
    # cannot read it either. Inferring the break from the trace is the task.
    assert not compile_expr("transmission * 4.0", list(variables)).ok
    assert _iface().apply(
        [ActionRequest("regent:0", "set_policy_rate", {"expr": "transmission * 4.0"})], system
    ).rejected


def test_observation_and_metrics_carry_the_enacted_rate():
    system = MonetaryEconomy(UNSHOCKED)
    system.reset(0)
    _iface().apply([ActionRequest("regent:0", "set_policy_rate", {"expr": "9.25"})], system)
    system.step()
    assert system.observe().vars["policy_rate"] == 9.25
    assert system.metrics()["policy_rate"] == 9.25  # what the institution DID, in the record
    assert system.time == 1


def test_lever_is_re_evaluated_every_step_and_clipped_into_range():
    system = MonetaryEconomy(UNSHOCKED)
    system.reset(0)
    _iface().apply(
        [ActionRequest("regent:0", "set_policy_rate", {"expr": "100.0 * output_gap"})], system)
    system.output_gap = 1.0
    system.step()
    assert system.policy_rate == RATE_RANGE[1]
    system.output_gap = -1.0
    system.step()
    assert system.policy_rate == RATE_RANGE[0]


# --- the objective --------------------------------------------------------------------------

def test_empty_post_shock_window_scores_infinite():
    trajectory = [{"t": float(t), "inflation": 2.0, "output_gap": 0.0, "cum_cost": 16.0 * t,
                   "policy_rate": 4.0, "inflation_target": 2.0} for t in range(5)]
    components = DualMandateLoss(post_shock_step=100).components(trajectory)
    assert components["post_loss"] == float("inf")
    assert components["post_mandate_burden"] == float("inf")
    assert np.isfinite(components["loss"])  # the full-horizon score is unaffected


def test_a_run_that_produced_no_rows_at_all_also_scores_infinite():
    """The degenerate case of the same bug, and the one that scored BEST instead of worst.

    ``post_loss`` is a sum of non-negative terms, so 0.0 is its global minimum. An arm that
    collapsed before emitting a single row used to be handed exactly that — the empty-window guard
    additionally required a non-empty trajectory, so the emptiest window of all took the ordinary
    path and won. A reference calibrated against it would be measuring collapse, not governance."""
    empty_window = DualMandateLoss(post_shock_step=100).components([])
    assert empty_window["post_loss"] == float("inf")
    assert empty_window["post_mandate_burden"] == float("inf")
    real_run = [{"t": float(t), "inflation": 3.0, "output_gap": 1.0, "cum_cost": 16.0 * t,
                 "policy_rate": 4.0, "inflation_target": 2.0} for t in range(200)]
    assert empty_window["post_loss"] > DualMandateLoss(post_shock_step=100).components(real_run)["post_loss"]


def test_post_window_cost_is_differenced_not_read_as_a_running_total():
    """Reading ``cum_cost`` undifferenced turns the score into a clock — a bug that shipped here."""
    def rows(offset: float):
        return [{"t": float(t), "inflation": 2.0, "output_gap": 0.0, "policy_rate": 4.0,
                 "inflation_target": 2.0, "cum_cost": offset + 16.0 * t} for t in range(20)]

    objective = DualMandateLoss(post_shock_step=10)
    early = objective.components(rows(0.0))
    late = objective.components(rows(10_000.0))  # same policy, just later in a longer run
    assert early["post_loss"] == pytest.approx(late["post_loss"])
    assert objective.evaluate(rows(0.0)) == pytest.approx(objective.evaluate(rows(10_000.0)))


def test_objective_fails_loudly_when_wired_to_the_wrong_system():
    with pytest.raises(KeyError):
        DualMandateLoss().components([{"t": 0.0, "current_x": 1.0}])


def test_mandate_description_states_the_trade_off():
    text = DualMandateLoss(lam=0.15).describe()
    assert "MANDATE" in text and "0.15" in text
    assert "target" in text and "rate" in text


# --- the control problem --------------------------------------------------------------------

def test_a_sensible_rule_beats_doing_nothing():
    """Excess demand is persistent, so holding the neutral rate is not a neutral choice."""
    seeds = range(8)
    taylor = "6.4 + 1.5 * (inflation - inflation_target) + 0.5 * output_gap"
    do_nothing = "r_star + inflation_target"
    assert _mean_loss(taylor, UNSHOCKED, seeds) < 0.5 * _mean_loss(do_nothing, UNSHOCKED, seeds)
    assert _mean_loss(taylor, SHOCKED, seeds) < _mean_loss(do_nothing, SHOCKED, seeds)


def test_the_optimum_is_interior_and_the_break_moves_it():
    """Both halves matter. An interior optimum is what a shock can move; a corner one cannot,
    and a world whose best policy is 'always max' or 'never act' has no headroom to measure."""
    seeds = range(6)
    grid = [0.0, 2.0, 3.5, 5.0, 6.5, 8.0, 12.0]
    pre = {rate: _mean_loss(f"{rate}", UNSHOCKED, seeds) for rate in grid}
    post = {rate: _mean_loss(f"{rate}", SHOCKED, seeds, key="post_loss",
                             post_shock_step=SHOCK_STEP) for rate in grid}
    best_pre = min(pre, key=lambda r: pre[r])
    best_post = min(post, key=lambda r: post[r])
    assert grid[0] < best_pre < grid[-1]
    assert grid[0] < best_post < grid[-1]
    # The instrument got weaker while its price stayed: the authority should buy less of it.
    assert best_post < best_pre
    # …and freezing the pre-break answer is measurably worse than the post-break one.
    assert post[best_pre] > post[best_post]
