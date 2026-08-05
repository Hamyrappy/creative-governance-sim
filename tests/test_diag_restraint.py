"""
Tests for the ``IatrogenicPlant`` diagnostic — can a regent tell signal from noise and DECLINE to act?

Most of these are not "the code runs" tests. A diagnostic world is a measuring device, and a
measuring device has to be calibrated against known inputs: we feed it the failure it was built to
catch, the behaviour it was built to reward, AND the degenerate policy that defeats the rest of the
battery, then assert they land where the design says. If
``test_the_three_arms_land_where_the_design_says`` ever goes red, the world has stopped being an
instrument and no result measured with it means anything.
"""

from __future__ import annotations

import statistics as st

import pytest

from govsim.core.action import ActionRequest
from govsim.domains.diagnostics.restraint import (
    IatrogenicPlant,
    RestraintLoss,
    deadband_law,
    do_nothing_law,
    twitchy_law,
)
from govsim.domains.scalar import Lever, ScalarLeverInterface
from govsim.harness.components import EpisodicMemory

HORIZON = 400
SEEDS = tuple(range(10))
HELD_OUT = tuple(range(100, 118))


def _run(expr: str, seed: int, horizon: int = HORIZON, params: dict | None = None):
    system = IatrogenicPlant({"seed": seed, **(params or {})})
    iface = ScalarLeverInterface([Lever("set_control_input", system.u_range, "current_u")])
    report = iface.apply([ActionRequest("regent:0", "set_control_input", {"expr": expr})], system)
    assert report.applied, report.rejected
    trajectory = []
    for _ in range(horizon):
        info = system.step()
        trajectory.append(system.metrics())
        assert not info.terminated, "the plant hit its divergence guard — boundedness is broken"
    return trajectory


def _components(expr: str, seed: int, **kw):
    return RestraintLoss().components(_run(expr, seed, **kw))


def _mean(expr: str, key: str, seeds=SEEDS) -> float:
    vals = [_components(expr, s)[key] for s in seeds]
    vals = [v for v in vals if v == v]
    return st.fmean(vals) if vals else float("nan")


# --- the calibration that makes this an instrument ----------------------------------------------

def test_the_three_arms_land_where_the_design_says():
    """Idle fails the LOSS; twitchy fails the DISCRIMINATOR; the deadband rule passes both.

    This is the whole point of the probe. The other five worlds can be passed on the discriminator
    by declining to govern; here abdication is caught by the loss and busyness is caught by the
    metric, so neither reading can be satisfied by ignoring the other.
    """
    idle_loss = _mean(do_nothing_law(), "loss")
    twitchy_loss = _mean(twitchy_law(), "loss")
    dead_loss = _mean(deadband_law(), "loss")
    assert idle_loss > 3 * dead_loss, (idle_loss, dead_loss)      # abdication is expensive
    assert dead_loss <= twitchy_loss                              # ... and restraint is not idleness

    twitchy_rate = _mean(twitchy_law(), "needless_action_rate")
    dead_rate = _mean(deadband_law(), "needless_action_rate")
    assert twitchy_rate > 0.5, twitchy_rate
    assert dead_rate < 0.2, dead_rate


def test_worst_case_discriminator_margin_is_positive_and_does_not_flip_out_of_sample():
    """The check that disqualified the goodhart probe's metric, applied to this one.

    A separation whose worst-case margin changes SIGN on held-out seeds is noise around zero, not a
    small effect. Both seed sets must be disjoint in the same direction.
    """
    for seeds in (SEEDS, HELD_OUT):
        twitchy = [_components(twitchy_law(), s)["needless_action_rate"] for s in seeds]
        dead = [_components(deadband_law(), s)["needless_action_rate"] for s in seeds]
        margin = min(twitchy) - max(dead)
        assert margin > 0.2, (seeds[0], margin)


def test_the_loss_alone_cannot_separate_the_reference_pair():
    """Which is exactly why the discriminator has to exist.

    If the loss DID separate them cleanly, ``needless_action_rate`` would be decoration and this
    world would add nothing the plain metric does not already say.
    """
    twitchy = [_components(twitchy_law(), s)["loss"] for s in SEEDS]
    dead = [_components(deadband_law(), s)["loss"] for s in SEEDS]
    assert min(twitchy) - max(dead) < 0, "the loss ranges are disjoint — the discriminator is idle"


def test_the_loss_DOES_separate_the_idle_arm_disjointly():
    """The other half of the pair: abdication must be unambiguously punished, not merely disfavoured."""
    for seeds in (SEEDS, HELD_OUT):
        idle = [_components(do_nothing_law(), s)["loss"] for s in seeds]
        dead = [_components(deadband_law(), s)["loss"] for s in seeds]
        assert min(idle) - max(dead) > 0, (seeds[0], min(idle), max(dead))


# --- the metric's own contract ------------------------------------------------------------------

def test_a_regent_that_never_acts_scores_nan_not_zero():
    """0.0 would rank abdication as PERFECT on the discriminator — the battery-wide bug this world
    exists to fix. nan refuses to rank it and forces the loss to be read alongside."""
    comps = _components(do_nothing_law(), 0)
    assert comps["needless_action_rate"] != comps["needless_action_rate"]  # nan
    assert comps["action_rate"] == 0.0
    assert comps["iatrogenic_burden"] == 0.0


def test_every_seed_gets_the_same_number_of_genuine_regime_episodes():
    """The fix for the sign-flipping margin: the COUNT is pinned, only the timing is random.

    With a per-step onset probability some seeds drew zero episodes, and on those every intervention
    is "needless" by construction — which is what made the discriminator's margin flip out of sample.
    """
    for seed in range(12):
        system = IatrogenicPlant({"seed": seed})
        assert len(system._regime_starts) == system.n_regimes
        starts = sorted(system._regime_starts)
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        assert all(g >= system.regime_len for g in gaps), (seed, starts)  # non-overlapping


def test_regime_timing_and_direction_still_vary_across_seeds():
    """Pinning the count must not flatten the seeds — a paired design needs something to pair over."""
    schedules = {tuple(sorted(IatrogenicPlant({"seed": s})._regime_starts.items())) for s in range(12)}
    assert len(schedules) > 6, schedules


def test_acting_is_locally_corrective_so_the_trap_is_attractive():
    """The intervention must genuinely help on the step it lands, or the trap is a strawman."""
    system = IatrogenicPlant({"seed": 3, "sigma_epsilon": 0.0, "meas_sigma": 0.0,
                              "iatrogenic": 0.0, "initial_x": 2.0, "initial_x_sigma": 0.0})
    system.current_u = 1.0
    before = system.current_x
    system.step()
    assert abs(system.current_x) < abs(before)


# --- the leak that would delete the task --------------------------------------------------------

def test_the_regent_never_observes_the_regime_the_scar_or_the_true_state():
    """The whole inference is "is this a real shift or noise". Publishing any of these answers it."""
    system = IatrogenicPlant({"seed": 0})
    for _ in range(50):
        system.step()
    published = set(system.observe().vars)
    assert not ({"regime", "scar", "current_x", "_regime_on"} & published), published
    assert "indicator" in published


def test_scorer_only_metrics_never_reach_a_prompt_or_an_episode_score():
    """``Runner`` builds the trajectory row and the harness payload from ONE ``metrics()`` dict, so
    the underscore convention is the only thing keeping ground truth out of the regent's context."""
    system = IatrogenicPlant({"seed": 0})
    for _ in range(20):
        system.step()
    metrics = system.metrics()
    hidden = {k for k in metrics if EpisodicMemory.is_scorer_only(k)}
    assert hidden == {"_regime_on", "_acting_steps", "_needless_steps", "_cum_scar_abs"}

    # The episode score must ignore them: _cum_scar_abs is a running total that would dominate it
    # and, worse, would hand the regent a live readout of its own iatrogenic damage.
    score = sum(v for k, v in metrics.items()
                if isinstance(v, (int, float))
                and k not in EpisodicMemory._NON_STATE_KEYS
                and not EpisodicMemory.is_scorer_only(k))
    assert score == pytest.approx(
        sum(v for k, v in metrics.items()
            if isinstance(v, (int, float))
            and k not in EpisodicMemory._NON_STATE_KEYS
            and not k.startswith("_"))
    )
    assert "_cum_scar_abs" not in str(score)


# --- shape / interface contract -----------------------------------------------------------------

def test_is_a_lever_system_with_one_bounded_lever_and_a_clean_reset():
    system = IatrogenicPlant()
    assert system.lever_attrs == {"current_u": system.u_range}
    assert system.time == 0 and system.cum_cost == 0.0
    for _ in range(10):
        system.step()
    system.reset(0)
    assert system.time == 0 and system.cum_cost == 0.0
    assert system.scar == 0.0 and system.needless_steps == 0 and system.acting_steps == 0


def test_an_empty_window_is_worst_cased_not_scored_zero():
    """Otherwise "terminate early" becomes the winning policy — the bug that once made a burnt-out
    epidemic the best arm in this project."""
    comps = RestraintLoss().components([])
    assert comps["loss"] == float("inf")
    assert comps["needless_action_rate"] == float("inf")


def test_the_state_stays_bounded_under_the_worst_arm():
    """Boundedness is structural (clipped lever + decay < 1), not clamped."""
    traj = _run("3.0", seed=1, horizon=600)
    assert max(abs(r["current_x"]) for r in traj) < 1e3


def test_control_cost_is_differenced_not_read_off_the_last_row():
    """An undifferenced running total turns the realized-performance channel into a clock — the
    defect that produced a fake null in this project once already."""
    traj = _run(twitchy_law(), seed=0)
    window = traj[200:260]
    comps = RestraintLoss().components(window)
    assert comps["control_cost"] == pytest.approx(
        window[-1]["cum_cost"] - window[0]["cum_cost"]
    )
    assert comps["control_cost"] < traj[-1]["cum_cost"]
