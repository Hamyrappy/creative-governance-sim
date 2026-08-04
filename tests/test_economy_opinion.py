"""
Tests for ``OpinionPolity`` + ``PolarizationLoss``.

Beyond the usual interface checks, these pin the four properties the platform needs before it will
believe a headroom number out of this world: the dynamics are bounded and mass-conserving, seeds
differ, the efficacy parameter is unobservable, and the pre-shock optimum is interior. The last one
is the expensive one — a world whose optimum sits at a corner reports a null for every agent, so it
is asserted here rather than left to a calibration script nobody re-runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from govsim.core.action import ActionRequest
from govsim.core.sandbox import compile_expr
from govsim.domains.economy.opinion import OpinionPolity, PolarizationLoss
from govsim.domains.scalar.interface import Lever, ScalarLeverInterface

SHOCK = 120
HORIZON = 240
#: A severe instrument collapse: 5% of ordered moderation reaches anyone. See the module docstring
#: on why a *mild* collapse is correctly a null here.
COLLAPSE = {"moderation_efficacy": 0.05}


def _iface() -> ScalarLeverInterface:
    return ScalarLeverInterface([Lever("set_moderation", (0.0, 1.0), "moderation")])


def _drive(system: OpinionPolity, expr: str) -> None:
    result = compile_expr(expr, list(system.observe().vars.keys()))
    assert result.ok, result.feedback
    system.install_lever("moderation", result.compiled, (0.0, 1.0), "regent:0")


def _run(expr: str, seed: int, horizon: int = HORIZON, **params) -> list[dict[str, float]]:
    system = OpinionPolity(dict(params, seed=seed))
    system.reset(seed)
    _drive(system, expr)
    rows = [system.metrics()]
    for _ in range(horizon):
        system.step()
        rows.append(system.metrics())
    return rows


def _loss(expr: str, seed: int, lam: float = 0.2, **params) -> float:
    return PolarizationLoss(lam=lam).components(_run(expr, seed, **params))["loss"]


# --- interface contract ---------------------------------------------------------------------

def test_lever_attrs_and_metrics_report_the_enacted_policy():
    system = OpinionPolity({"shock_step": None})
    assert system.lever_attrs == {"moderation": (0.0, 1.0)}
    _drive(system, "0.7")
    system.step()
    metrics = system.metrics()
    # The governance question is what the institution DID, so the lever has to be in the record.
    assert metrics["moderation"] == pytest.approx(0.7)
    assert metrics["cum_cost"] == pytest.approx(0.7)
    assert system.time == 1


def test_lever_expression_is_reevaluated_every_step_and_clipped():
    system = OpinionPolity({"shock_step": None})
    report = _iface().apply([ActionRequest("regent:0", "set_moderation", {"expr": "100.0 * polarization"})], system)
    assert len(report.applied) == 1
    system.step()
    assert system.moderation == 1.0  # clipped into range rather than rejected
    _drive(system, "polarization")
    seen = []
    for _ in range(5):
        system.step()
        seen.append(system.moderation)
    assert len(set(seen)) > 1  # tracking a moving observable, not frozen at apply time


def test_clone_continues_the_same_stochastic_stream():
    system = OpinionPolity({"shock_step": None})
    _drive(system, "0.4")
    for _ in range(20):
        system.step()
    twin = system.clone()
    for _ in range(20):
        system.step()
        twin.step()
    assert np.allclose(system.mass, twin.mass)


# --- bounded dynamics + the conservation invariant ------------------------------------------

def test_mass_is_conserved_and_non_negative_over_a_long_hostile_run():
    """The boundedness claim, at parameters far past anything a scenario would use."""
    hostile = {
        "shock_step": None, "polar_strength": 5.0, "attraction": 4.0, "churn": 0.25,
        "dt": 2.0, "agitation_sigma": 0.8, "civic_gain": 20.0, "camp_pole": 1.0,
    }
    system = OpinionPolity(dict(hostile, seed=3))
    system.reset(3)
    _drive(system, "1.0 if polarization > 0.2 else 0.0")
    for _ in range(3000):
        system.step()
        assert system.mass.min() >= 0.0
        assert system.mass.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.isfinite(system.mass))


def test_every_reported_statistic_stays_in_range():
    for expr in ("0.0", "1.0", "0.5"):
        for seed in range(4):
            rows = _run(expr, seed, horizon=400, shock_step=SHOCK, shock_params=COLLAPSE)
            for row in rows:
                assert 0.0 <= row["polarization"] <= 1.0
                assert -1.0 <= row["mean_opinion"] <= 1.0
                assert 0.0 <= row["extreme_mass"] <= 1.0
                assert 0.0 <= row["dispersion"] <= 1.0


def test_support_never_leaves_the_grid():
    """Nothing accumulates as a boundary artefact of the transfer clamp."""
    system = OpinionPolity({"shock_step": None, "polar_strength": 3.0, "camp_pole": 1.0, "seed": 7})
    system.reset(7)
    _drive(system, "0.0")
    for _ in range(500):
        system.step()
    assert system.mass.shape == system.grid.shape
    assert abs(system._mean_opinion()) <= 1.0


# --- per-seed heterogeneity ------------------------------------------------------------------

def test_seeds_produce_genuinely_different_trajectories():
    finals = [_run("0.3", seed, horizon=120, shock_step=None)[-1]["polarization"] for seed in range(8)]
    assert len(set(np.round(finals, 6))) == len(finals)
    # Not merely different in the last digit: the seeds have to disagree about the problem.
    assert float(np.std(finals)) > 1e-3


def test_reset_with_the_same_seed_is_reproducible():
    a = _run("0.4", 11, horizon=60, shock_step=None)
    b = _run("0.4", 11, horizon=60, shock_step=None)
    assert [row["polarization"] for row in a] == [row["polarization"] for row in b]


def test_reset_restores_parameters_the_shock_overwrote():
    system = OpinionPolity({"shock_step": 2, "shock_params": {"moderation_efficacy": 0.05, "civic_gain": 0.0}})
    _drive(system, "1.0")
    for _ in range(5):
        system.step()
    assert system.moderation_efficacy == pytest.approx(0.05)
    system.reset(0)
    assert system.moderation_efficacy == pytest.approx(1.0)
    assert system.civic_gain == pytest.approx(3.0)
    assert system.moderation == 0.0 and system.cum_cost == 0.0 and system.time == 0


def test_reset_restores_EVERY_parameter_a_shock_can_reach():
    """``shock_params`` setattrs arbitrary names, so ``reset`` must undo arbitrary names.

    Pinned by reflection rather than by a list, because a hand-kept list of restorable parameters is
    exactly what rotted here: seven of them (`attraction`, `camp_pole`, `camp_width`, `churn`, `dt`,
    `moderation_cost`, `agitation_sigma`) had no shadow and leaked past ``reset`` into the next run
    of the same object. A leaked ``moderation_cost`` re-prices the instrument for a later arm, so
    the leak would show up as an arm difference and be read as a result.
    """
    pristine = OpinionPolity({"shock_step": None})
    # Every scalar parameter the world exposes, perturbed to a value it cannot reach on its own.
    # `n_bins`/`h` describe the grid rather than the physics, so they are asserted-restored but not
    # perturbed — a fractional bin count is not a shock, it is a broken object.
    structural = {"n_bins", "h"}
    hostile = {name: value + 0.37 for name, value in pristine._pristine.items()
               if name not in structural}
    system = OpinionPolity({"shock_step": 1, "shock_params": hostile})
    _drive(system, "1.0")
    for _ in range(3):
        system.step()
    assert system.attraction == pytest.approx(hostile["attraction"])  # the shock really landed
    system.reset(0)
    for name, value in pristine._pristine.items():
        assert getattr(system, name) == pytest.approx(value), f"{name} leaked past reset()"
    # And the derived draw is rebuilt from the restored base, not left at the shocked value.
    assert system.polar_strength != pytest.approx(hostile["polar_strength_init"])


# --- the shock: it bites, and it is invisible -------------------------------------------------

def test_the_shock_changes_behaviour_under_an_unchanged_policy():
    calm = [_run("0.6", s, shock_step=None)[-1]["polarization"] for s in range(8)]
    broken = [_run("0.6", s, shock_step=SHOCK, shock_params=COLLAPSE)[-1]["polarization"] for s in range(8)]
    assert float(np.mean(broken)) > float(np.mean(calm)) + 0.1
    # Same policy, same bill: the collapse costs exactly as much as the working instrument did.
    assert _run("0.6", 0, shock_step=None)[-1]["cum_cost"] == pytest.approx(
        _run("0.6", 0, shock_step=SHOCK, shock_params=COLLAPSE)[-1]["cum_cost"])


def test_the_shock_lands_exactly_at_shock_step():
    system = OpinionPolity({"shock_step": 5, "shock_params": COLLAPSE})
    _drive(system, "0.5")
    for _ in range(5):  # steps 0..4 leave the instrument intact
        assert system.moderation_efficacy == pytest.approx(1.0)
        system.step()
    assert system.time == 5
    system.step()  # the step taken AT t == shock_step is the one that breaks it
    assert system.moderation_efficacy == pytest.approx(0.05)


def test_efficacy_is_absent_from_the_regent_view():
    system = OpinionPolity({"shock_step": SHOCK, "shock_params": COLLAPSE})
    for _ in range(3):
        system.step()
    variables = system.observe().vars
    assert "moderation_efficacy" not in variables
    # Nor smuggled in under another name, nor recoverable from the action space the regent is shown.
    assert not any("efficacy" in name for name in variables)
    assert not any("efficacy" in name for name in system.metrics())
    assert not any("efficacy" in name for name in _iface().action_space(system, "regent:0").context_vars)


def test_efficacy_is_not_recoverable_from_the_view_alone():
    """Two worlds differing only in efficacy look identical to the regent on the shock step."""
    working = OpinionPolity({"shock_step": 3, "shock_params": {"moderation_efficacy": 1.0}, "seed": 2})
    broken = OpinionPolity({"shock_step": 3, "shock_params": COLLAPSE, "seed": 2})
    for system in (working, broken):
        system.reset(2)
        _drive(system, "0.8")
    for _ in range(4):  # steps 0..3; the shock is applied inside step 3, before dynamics
        working.step()
        broken.step()
    assert working.moderation_efficacy != broken.moderation_efficacy
    assert working.observe().vars["moderation"] == broken.observe().vars["moderation"]
    assert working.observe().vars["cum_cost"] == pytest.approx(broken.observe().vars["cum_cost"])


# --- the objective ----------------------------------------------------------------------------

def test_empty_post_shock_window_scores_worst_case_not_zero():
    objective = PolarizationLoss(lam=0.2, post_shock_step=SHOCK)
    stunted = _run("0.4", 0, horizon=10, shock_step=SHOCK, shock_params=COLLAPSE)
    components = objective.components(stunted)
    assert components["post_loss"] == float("inf")
    assert components["post_polarization"] == float("inf")
    assert components["post_cost"] == float("inf")
    assert np.isfinite(components["loss"])  # the full-horizon metric is unaffected


def test_post_shock_cost_is_differenced_across_the_window():
    """Reading ``cum_cost`` undifferenced would bill the post-shock window for pre-shock spending."""
    rows = _run("0.5", 0, shock_step=SHOCK, shock_params=COLLAPSE)
    components = PolarizationLoss(lam=0.2, post_shock_step=SHOCK).components(rows)
    assert components["post_cost"] == pytest.approx(0.5 * (HORIZON - SHOCK))
    assert components["cum_cost"] == pytest.approx(0.5 * HORIZON)


def test_evaluate_on_a_window_scores_the_window_not_the_clock():
    """Two identical windows at different times must score identically (the running-total bug)."""
    rows = _run("0.5", 0, shock_step=None)
    objective = PolarizationLoss(lam=0.2)
    early = [dict(row, polarization=0.3) for row in rows[10:30]]
    late = [dict(row, polarization=0.3) for row in rows[200:220]]
    assert objective.evaluate(early) == pytest.approx(objective.evaluate(late))


def test_evaluate_is_the_negated_loss_and_describe_states_the_mandate():
    rows = _run("0.4", 0, shock_step=None)
    objective = PolarizationLoss(lam=0.2)
    assert objective.evaluate(rows) == pytest.approx(-objective.components(rows)["loss"])
    mandate = objective.describe()
    assert "MANDATE" in mandate and "0.2" in mandate
    assert "not on the mixing it achieves" in mandate  # the cost asymmetry has to be stated


def test_empty_trajectory_is_scored_without_raising():
    assert PolarizationLoss().evaluate([]) == 0.0
    assert PolarizationLoss().components([])["total_polarization"] == 0.0


# --- the world has a control problem, and its optimum is interior -----------------------------

def test_a_hand_written_policy_beats_doing_nothing_on_every_seed():
    seeds = range(10)
    standing = [_loss("0.4", s, shock_step=None) for s in seeds]
    feedback = [_loss("min(1.0, 0.15 + 1.5 * polarization)", s, shock_step=None) for s in seeds]
    passive = [_loss("0.0", s, shock_step=None) for s in seeds]
    assert all(a < b for a, b in zip(standing, passive))
    assert all(a < b for a, b in zip(feedback, passive))
    assert float(np.mean(passive)) > 2.0 * float(np.mean(standing))


def test_the_pre_shock_optimum_is_interior():
    """Both corners must be worse than an interior level, or no shock can move the optimum."""
    seeds = range(10)
    scores = {level: float(np.mean([_loss(f"{level:.2f}", s, shock_step=None) for s in seeds]))
              for level in (0.0, 0.4, 1.0)}
    assert scores[0.4] < scores[0.0]
    assert scores[0.4] < scores[1.0]


def test_a_severe_collapse_moves_the_post_shock_optimum_toward_abandoning_the_instrument():
    """The adaptation the world is built to reward: stop paying for an instrument that stopped working.

    The pre-shock policy is pinned at the calibrated 0.4 in every arm, so what is compared is purely
    the post-shock decision. Keeping 0.4 must lose to abandoning it once efficacy has collapsed, and
    must NOT lose when the instrument still works — otherwise the score is tracking something other
    than the break.
    """
    seeds = range(10)
    objective = PolarizationLoss(lam=0.2, post_shock_step=SHOCK)

    def post_loss(after: str, efficacy: float) -> float:
        params = {"shock_step": SHOCK, "shock_params": {"moderation_efficacy": efficacy}}
        expr = f"0.4 if t < {SHOCK} else {after}"
        return float(np.mean([objective.components(_run(expr, s, **params))["post_loss"] for s in seeds]))

    assert post_loss("0.0", 0.05) < post_loss("0.4", 0.05)   # collapsed: the drive is a pure bill
    assert post_loss("0.4", 1.0) < post_loss("0.0", 1.0)     # intact: abandoning it would be wrong


# --- guards ------------------------------------------------------------------------------------

def test_even_bin_count_is_rejected():
    with pytest.raises(ValueError):
        OpinionPolity({"n_bins": 20})
