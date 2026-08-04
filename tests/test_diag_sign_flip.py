"""
Tests for the ``SignFlipPlant`` diagnostic — does the instrument actually measure what it claims?

Most of these are not "the code runs" tests. A diagnostic world is a measuring device, and a
measuring device has to be *calibrated against known inputs*: we feed it the failure it was built to
catch and the behaviour it was built to reward, and assert they land far apart on the discriminating
metric. If ``test_naive_law_fails_the_discriminator_and_the_corrected_law_passes`` ever goes red, the
world has stopped being an instrument and no result measured with it means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from govsim.core.action import ActionRequest
from govsim.core.system import RollableSystem
from govsim.domains.diagnostics.sign_flip import (
    NOMINAL_FEEDBACK_GAIN,
    SignFlipLoss,
    SignFlipPlant,
    detuned_law,
    frozen_law,
    loss_for,
    reversed_law,
    true_post_break_gain_sign,
)
from govsim.domains.scalar import Lever, ScalarLeverInterface

BREAK = 150
HORIZON = 400
SEEDS = tuple(range(8))


def _run(expr: str | None, seed: int, horizon: int = HORIZON, params: dict | None = None):
    """Install ``expr`` as the control law (``None`` = do nothing) and collect the trajectory."""
    system = SignFlipPlant({**(params or {}), "seed": seed})
    if expr is not None:
        iface = ScalarLeverInterface([Lever("set_control_input", system.u_range, "current_u")])
        report = iface.apply([ActionRequest("regent:0", "set_control_input", {"expr": expr})], system)
        assert report.applied, report.rejected
    trajectory = []
    for _ in range(horizon):
        info = system.step()
        trajectory.append(system.metrics())
        assert not info.terminated, "the plant hit its divergence guard — boundedness is broken"
    return trajectory


def _components(expr: str | None, seed: int, **kw):
    return SignFlipLoss(post_shock_step=BREAK).components(_run(expr, seed, **kw))


# --- shape / interface contract ---------------------------------------------------------------

def test_is_a_rollable_lever_system_with_one_bounded_lever():
    system = SignFlipPlant()
    assert isinstance(system, RollableSystem)
    assert system.lever_attrs == {"current_u": system.u_range}
    assert system.time == 0


def test_lever_is_re_evaluated_every_step_and_clipped():
    """The eval-cadence contract: the installed law is re-read each tick, not once at apply time."""
    system = SignFlipPlant({"seed": 0, "u_range": (-3.0, 3.0)})
    iface = ScalarLeverInterface([Lever("set_control_input", system.u_range, "current_u")])
    iface.apply([ActionRequest("regent:0", "set_control_input", {"expr": "-100.0 * current_x"})], system)
    system.current_x = 5.0
    system.step()
    assert system.current_u == -3.0  # clipped, and computed from the state at THIS tick
    system.current_x = -5.0
    system.step()
    assert system.current_u == 3.0   # re-evaluated: a one-shot apply could not have flipped it


# --- boundedness (requirement 5: no arm may diverge) -------------------------------------------

@pytest.mark.parametrize(
    "name,expr",
    [
        ("do_nothing", None),
        ("max_lever", "3.0"),
        ("min_lever", "-3.0"),
        ("naive_frozen", frozen_law()),          # the runaway arm: feedback through a reversed lever
        ("corrected", reversed_law(BREAK)),
    ],
)
def test_bounded_over_500_steps(name, expr):
    """Saturation, not divergence.

    The frozen-rule arm is a genuine positive-feedback runaway after the break: it must be expensive
    and must NOT blow up. Nothing clamps the state — the bound comes from the clipped lever plus the
    open-loop pole ``decay < 1``, whose saturated fixed point is ``|g|·U/(1 - decay)`` ≈ 20-45 here.
    """
    for seed in SEEDS:
        xs = [row["current_x"] for row in _run(expr, seed, horizon=500)]
        assert all(np.isfinite(x) for x in xs), f"{name}: non-finite state on seed {seed}"
        assert max(abs(x) for x in xs) < 100.0, f"{name}: state ran to {max(map(abs, xs)):.1f} on seed {seed}"


# --- per-seed heterogeneity (paired statistics need something to pair over) ---------------------

def test_seeds_give_different_plants_and_different_trajectories():
    plants = [SignFlipPlant({"seed": s}) for s in (0, 1, 2, 3)]
    for field in ("decay", "control_gain", "current_x", "disturbance"):
        values = [getattr(p, field) for p in plants]
        assert len(set(values)) == len(values), f"{field} is identical across seeds — no heterogeneity"
    finals = [_run(frozen_law(), s, horizon=60)[-1]["current_x"] for s in (0, 1, 2, 3)]
    assert len(set(finals)) == len(finals)
    # …but a repeated seed reproduces exactly (the paired design's other half).
    assert _run(frozen_law(), 1, horizon=60) == _run(frozen_law(), 1, horizon=60)


def test_clone_continues_the_same_stochastic_stream():
    system = SignFlipPlant({"seed": 3})
    for _ in range(20):
        system.step()
    twin = system.clone()
    assert [system.step() and system.current_x for _ in range(30)] == [twin.step() and twin.current_x for _ in range(30)]


# --- the hidden parameter --------------------------------------------------------------------

def test_control_gain_is_invisible_to_the_regent():
    """The task IS inferring that the lever reversed. Publishing the gain would delete the task."""
    system = SignFlipPlant({"seed": 0})
    for _ in range(BREAK + 5):
        system.step()
    observed = system.observe("regent:0").vars
    assert "control_gain" not in observed
    assert "disturbance" not in observed          # what makes feedback necessary stays hidden too
    assert "flip_factor" not in observed
    # No published value equals the gain either — magnitude or sign, direct or negated.
    for key, value in observed.items():
        assert not np.isclose(abs(value), abs(system.control_gain)), f"'{key}' leaks the gain magnitude"
    # metrics() is a channel to the regent as well: the Runner feeds it into harness.on_outcome.
    reported = system.metrics()
    for banned in ("control_gain", "disturbance", "flip_factor", "wrong_sign", "cum_wrong_sign"):
        assert banned not in reported
    # …and the observable the policy language needs is present.
    assert {"current_x", "previous_x", "current_u", "target_x", "step", "decay"} <= set(observed)


def test_the_break_reverses_the_gain_after_the_lever_is_re_evaluated():
    """Order contract: re-eval, THEN shock, THEN dynamics.

    The break step therefore already pushes the old rule's control through the NEW gain — the regent
    could not have acted on information it did not have.
    """
    quiet = {"decay": 0.8, "decay_sigma": 0.0, "control_gain": 1.0, "gain_sigma": 0.0,
             "sigma_epsilon": 0.0, "dist_sigma": 0.0, "initial_x": 0.0, "initial_x_sigma": 0.0,
             "shock_step": 3, "seed": 0}
    system = SignFlipPlant(quiet)
    iface = ScalarLeverInterface([Lever("set_control_input", system.u_range, "current_u")])
    iface.apply([ActionRequest("regent:0", "set_control_input", {"expr": "1.0"})], system)
    for expected in (1.0, 1.8, 2.44):            # x_{k+1} = 0.8·x_k + (+1)·1
        system.step()
        assert system.current_x == pytest.approx(expected)
    assert system.control_gain == pytest.approx(1.0)
    system.step()                                 # the break: gain flips, u is still the old +1
    assert system.control_gain == pytest.approx(-1.0)
    assert system.current_x == pytest.approx(0.8 * 2.44 - 1.0)


def test_reset_restores_a_pristine_pre_break_plant():
    system = SignFlipPlant({"seed": 0, "shock_step": 5})
    for _ in range(10):
        system.step()
    assert system.control_gain < 0                # flipped
    system.reset(1)
    assert system.control_gain > 0, "reset left the plant in its post-break regime"
    assert system.u_range == system.u_range0 and system.target_x == system.target_x0
    assert system.time == 0 and system.cum_cost == 0.0 and system._levers == {}


# --- THE TEST THAT MATTERS ---------------------------------------------------------------------

def test_naive_law_fails_the_discriminator_and_the_corrected_law_passes():
    """The calibration of the instrument: a known-wrong input and a known-right one, far apart.

    Both laws have the SAME gain magnitude and differ only in sign after the break, so this cannot
    be passed by tuning — only by reversing.
    """
    for seed in SEEDS:
        naive = _components(frozen_law(), seed)
        correct = _components(reversed_law(BREAK), seed)
        assert naive["wrong_sign_fraction"] >= 0.95, f"seed {seed}: the trap did not catch the frozen rule"
        assert correct["wrong_sign_fraction"] <= 0.05, f"seed {seed}: the correct law was flagged as wrong"
        # The discriminator is conditional on acting; both arms must actually be acting.
        assert naive["active_fraction"] >= 0.9 and correct["active_fraction"] >= 0.9
        # …and the failure is expensive, not merely detectable (requirement: costly).
        assert naive["post_loss"] > 20.0 * correct["post_loss"], f"seed {seed}: the trap is not costly"


def test_the_discriminator_separates_by_far_more_than_the_seed_to_seed_noise():
    """THE VALIDITY CONDITION for the whole instrument, stated as a margin rather than a threshold.

    A diagnostic is only worth reading if the gap between the arm it condemns and the arm it clears
    is wide compared to the scatter within each arm. Measured over 10 seeds: naive 1.0000 (sd 0.0),
    correct 0.0031 (sd 0.0033), worst-case gap 0.9932 against a within-arm sd of 0.0033 — a
    separation of ~300x the noise, with no overlap between the arms' ranges at all.
    """
    seeds = range(10)
    naive = [_components(frozen_law(), s)["wrong_sign_fraction"] for s in seeds]
    correct = [_components(reversed_law(BREAK), s)["wrong_sign_fraction"] for s in seeds]
    gap = min(naive) - max(correct)          # worst case over seeds, not a difference of means
    noise = max(float(np.std(naive, ddof=1)), float(np.std(correct, ddof=1)))
    assert gap > 0.0, "the arms OVERLAP — the world does not discriminate and no result means anything"
    assert gap > 20.0 * noise, f"separation {gap:.4f} is not wide against seed noise {noise:.4f}"
    assert gap > 0.9, f"worst-case separation collapsed to {gap:.4f}"


def test_the_discriminator_is_graded_and_not_a_two_valued_read_off():
    """Both reference laws are affine in the error, so ``g·u·e`` has a constant sign for each and
    the separation above is guaranteed by algebra. That would be worthless if the metric could ONLY
    return 0 or 1 — it would be restating the policy's sign rather than measuring behaviour.

    So feed it policies outside the reference family. A rule that reverses on alternating steps
    reads ~0.5; bang-bang and over-gained corrections (nothing affine about the first) read ~0.
    """
    pre = f"(-{NOMINAL_FEEDBACK_GAIN} * (current_x - target_x)) if step < {BREAK} else "
    right = f"({NOMINAL_FEEDBACK_GAIN} * (current_x - target_x))"
    wrong = f"(-{NOMINAL_FEEDBACK_GAIN} * (current_x - target_x))"
    # deadbands off: a deadband skips steps just after a corrected one (the state landed near
    # target), which biases a genuinely 50/50 policy DOWN to ~0.44. Ungated it reads 0.502.
    scorer = SignFlipLoss(post_shock_step=BREAK, state_deadband=0.0, control_deadband=0.0)
    seeds = range(6)

    half = [scorer.components(_run(pre + f"({right} if (step % 2 == 0) else {wrong})", s)) for s in seeds]
    assert 0.42 <= float(np.mean([c["wrong_sign_fraction"] for c in half])) <= 0.58

    bang = [scorer.components(_run(pre + "(3.0 if current_x > target_x else -3.0)", s)) for s in seeds]
    assert float(np.mean([c["wrong_sign_fraction"] for c in bang])) <= 0.05
    over = [scorer.components(_run(pre + "(3.0 * (current_x - target_x))", s)) for s in seeds]
    assert float(np.mean([c["wrong_sign_fraction"] for c in over])) <= 0.05
    # …and a constant, feedback-free push is condemned: it is right by luck at best.
    flat = [scorer.components(_run(pre + "3.0", s)) for s in seeds]
    assert float(np.mean([c["wrong_sign_fraction"] for c in flat])) >= 0.90


def test_the_separation_does_not_depend_on_tuned_deadbands():
    """If the gap only appeared at one setting of the two deadbands, they would be researcher
    degrees of freedom and the result would be a fit. It holds with them off entirely."""
    for sdb, cdb in [(0.0, 0.0), (0.05, 0.01), (0.25, 0.05), (1.0, 0.5)]:
        scorer = SignFlipLoss(post_shock_step=BREAK, state_deadband=sdb, control_deadband=cdb)
        naive = [scorer.components(_run(frozen_law(), s))["wrong_sign_fraction"] for s in range(4)]
        correct = [scorer.components(_run(reversed_law(BREAK), s))["wrong_sign_fraction"] for s in range(4)]
        assert min(naive) - max(correct) > 0.9, f"deadbands ({sdb}, {cdb}) collapsed the separation"


def test_the_trap_is_attractive_because_the_naive_law_is_optimal_before_the_break():
    """A failure nobody would commit measures carelessness, not a failure mode.

    Pre-break the frozen law is not a strawman: it is roughly 10x better than passivity, and it is
    the answer to the problem as posed for the first 150 steps.
    """
    scorer = SignFlipLoss()
    for seed in SEEDS:
        naive_pre = scorer.components([r for r in _run(frozen_law(), seed) if r["step"] < BREAK])["mse"]
        idle_pre = scorer.components([r for r in _run(None, seed) if r["step"] < BREAK])["mse"]
        assert naive_pre < idle_pre / 5.0, f"seed {seed}: the naive law is not worth learning pre-break"


def test_passivity_is_not_a_free_retreat():
    """Doing nothing must be punished too, or "stop acting" would be a legitimate answer and the
    world would measure timidity rather than direction."""
    for seed in SEEDS:
        idle = _components(None, seed)
        correct = _components(reversed_law(BREAK), seed)
        assert idle["post_loss"] > 5.0 * correct["post_loss"], f"seed {seed}: passivity is too cheap"
        assert idle["active_fraction"] == 0.0  # the companion metric exposes it as passive, not correct


def test_abdication_scores_nan_on_the_discriminator_not_zero():
    """An empty denominator must not read as a perfect score.

    Measured before this was fixed: the do-nothing arm scored ``wrong_sign_fraction = 0.000000``
    against the corrected law's ``0.003105`` — i.e. abdication OUTRANKED the known-correct policy on
    the headline metric, and would have kept outranking it through any mean-over-seeds summary. 0/0
    is undefined, not perfect, so it is ``nan``: it propagates through an aggregation instead of
    flattering the arm that never governed.
    """
    for seed in SEEDS:
        idle = _components(None, seed)
        assert np.isnan(idle["wrong_sign_fraction"]), f"seed {seed}: abdication scored as correct"
        assert idle["active_steps"] == 0.0 and idle["called_for_steps"] > 0.0
        # …while the arm that actually governs gets a real number on a real denominator.
        correct = _components(reversed_law(BREAK), seed)
        assert np.isfinite(correct["wrong_sign_fraction"])
        assert correct["active_steps"] > 100.0


def test_components_key_set_is_the_same_on_both_branches():
    """A ragged components() dict makes a ragged results table — and the arm that goes ragged is
    the degenerate one, which is exactly the row a reader skims past."""
    full = SignFlipLoss(post_shock_step=BREAK).components(_run(frozen_law(), 0))
    empty = SignFlipLoss(post_shock_step=BREAK).components(_run(frozen_law(), 0, horizon=10))
    assert set(full) == set(empty)
    assert {"wrong_steps", "active_steps", "called_for_steps"} <= set(full)
    # the counts are the pooled numerator/denominator behind the ratio
    assert full["wrong_steps"] / full["active_steps"] == pytest.approx(full["wrong_sign_fraction"])


def test_shrinking_the_gain_improves_the_loss_but_stays_wrong_signed():
    """THE POINT OF THE INSTRUMENT: loss alone cannot diagnose the failure.

    Reacting to the break by shrinking the gain — keeping the direction — is what a hill-climber
    finds, and the loss rewards it richly: at gain 0.02 the post-break loss drops by 10-75x on every
    seed. The discriminator does not move at all. That gap between "the number went down" and "the
    mistake was corrected" is the whole reason this world exists.
    """
    for seed in SEEDS:
        naive = _components(frozen_law(), seed)
        timid = _components(detuned_law(BREAK, post_gain=0.02), seed)
        correct = _components(reversed_law(BREAK), seed)
        # Loss says: real progress.
        assert timid["post_loss"] < 0.25 * naive["post_loss"], f"seed {seed}"
        # The discriminator says: nothing was learned about the direction — on a denominator big
        # enough to mean something (the timid rule still acts on a third of the called-for steps).
        assert timid["wrong_sign_fraction"] >= 0.95, f"seed {seed}"
        assert timid["active_fraction"] >= 0.30, f"seed {seed}"
        # And loss confirms the re-tuner never reaches the reversal it was avoiding.
        assert timid["post_loss"] > 5.0 * correct["post_loss"], f"seed {seed}"


def test_a_half_measure_can_be_worse_than_the_frozen_rule():
    """Re-tuning is not even monotone, which is what makes the trap sticky.

    Post-break the closed-loop pole is ``decay + |g|·k`` for a wrong-signed gain ``k``: shrinking
    ``k`` walks that pole DOWN through 1.0, and near the unit root the persistent disturbance random-
    walks the state across the whole saturation band. So on the slow-decay seeds a regent that
    cautiously cuts its gain to 0.10 sees things get WORSE than leaving the rule alone — evidence
    that reads as "my direction was right, my nerve was wrong". Only crossing all the way through
    zero, to the reversed sign, is reliably better. Pinning this because it is a property of the
    world worth keeping, not an accident: it is what stops the diagnostic from being solvable by
    timidity.
    """
    ratios = [
        _components(detuned_law(BREAK, post_gain=0.10), seed)["post_loss"]
        / _components(frozen_law(), seed)["post_loss"]
        for seed in SEEDS
    ]
    assert any(r > 1.0 for r in ratios), "half-measures always help — the trap has gone soft"
    assert any(r < 0.5 for r in ratios), "half-measures never help — the trap is no longer tempting"


# --- objective bookkeeping ---------------------------------------------------------------------

def test_empty_post_break_window_scores_worst_case_not_zero():
    """A run that ends before the break has no post-break evidence. Scoring that 0.0 would make
    early termination the winning policy — the trap ``EpidemicLoss`` documents."""
    trajectory = _run(frozen_law(), 0, horizon=10)
    components = SignFlipLoss(post_shock_step=BREAK).components(trajectory)
    assert components["post_loss"] == float("inf")
    assert components["wrong_sign_fraction"] == float("inf")
    assert components["post_steps"] == 0.0
    assert np.isfinite(components["loss"])  # the full-horizon score is untouched


def test_window_cost_is_differenced_not_read_off_the_last_row():
    """``cum_cost`` is a running total from t=0. Reading it undifferenced would bill the post-break
    window for control bought before the lever ever reversed, and would turn the Runner's
    per-interval realized-score signal into a clock that only ever reports "worse than last time"."""
    rows = [{"step": float(i), "current_x": 0.0, "previous_x": 0.0, "target_x": 0.0,
             "current_u": 0.0, "cum_cost": 100.0 + i} for i in range(10)]
    components = SignFlipLoss(lam=1.0, post_shock_step=5).components(rows)
    assert components["post_cost"] == pytest.approx(4.0 / 5.0)   # (109 - 105) / 5 rows
    assert components["mean_cost"] == pytest.approx(9.0 / 10.0)  # NOT 109.0
    # evaluate() differences too — the Runner calls it on short windows.
    assert SignFlipLoss(lam=1.0).evaluate(rows[5:]) == pytest.approx(-4.0 / 5.0)


def test_post_window_scoring_matches_scoring_that_window_alone():
    """A consistency check the undifferenced bug would have failed loudly."""
    trajectory = _run(reversed_law(BREAK), 0)
    windowed = SignFlipLoss(post_shock_step=BREAK).components(trajectory)
    standalone = SignFlipLoss().components([r for r in trajectory if r["step"] >= BREAK])
    assert windowed["post_loss"] == pytest.approx(standalone["loss"])


def test_objective_loud_fails_on_a_mis_wired_system():
    with pytest.raises(KeyError):
        SignFlipLoss().components([{"t": 0.0, "infected": 0.1}])


# --- pairing the objective to the plant (the silent mis-scoring mode) ---------------------------

def test_a_mis_paired_loss_reads_the_instrument_exactly_backwards():
    """Why ``loss_for`` exists. ``post_break_gain_sign`` is a property of the PLANT, but nothing
    couples them, and the failure is silent: every arm still returns a plausible number in [0, 1].

    The natural way to hit this is the no-reversal control arm (``flip_factor=+1``) — the ablation
    any honest experiment with this world needs — scored by a default ``SignFlipLoss``.
    """
    control_arm = {"flip_factor": 1.0}   # the lever never reverses; the frozen law stays CORRECT
    wrong_scorer = SignFlipLoss(post_shock_step=BREAK)          # default sign = -1: WRONG for this plant
    right_scorer = loss_for(SignFlipPlant({**control_arm, "seed": 0}))
    for seed in SEEDS:
        trajectory = _run(frozen_law(), seed, params=control_arm)
        assert wrong_scorer.components(trajectory)["wrong_sign_fraction"] >= 0.95   # backwards…
        assert right_scorer.components(trajectory)["wrong_sign_fraction"] <= 0.05   # …and corrected


def test_loss_for_matches_the_plant_and_derives_the_sign_from_its_configuration():
    flipping = SignFlipPlant({"seed": 0})
    assert true_post_break_gain_sign(flipping) == -1.0
    scorer = loss_for(flipping)
    assert scorer.post_shock_step == flipping.shock_step
    assert scorer.post_break_gain_sign == -1.0
    # …read off the configuration, so it is right whether or not the plant has already been stepped
    # past its break (the live ``control_gain`` flips underneath you; ``control_gain0`` does not).
    for _ in range(flipping.shock_step + 5):
        flipping.step()
    assert flipping.control_gain < 0
    assert true_post_break_gain_sign(flipping) == -1.0
    assert true_post_break_gain_sign(SignFlipPlant({"flip_factor": 1.0})) == 1.0
    # a shock that overwrites the gain outright wins over the flip factor
    assert true_post_break_gain_sign(SignFlipPlant({"shock_params": {"control_gain": -2.5}})) == -1.0
    assert true_post_break_gain_sign(SignFlipPlant({"flip_factor": -1.0, "shock_params": {"control_gain": 0.4}})) == 1.0
    # explicit kwargs still win, so the factory is a default and not a straitjacket
    assert loss_for(flipping, lam=0.9, post_break_gain_sign=1.0).lam == 0.9


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_a_degenerate_gain_sign_is_refused_at_construction(bad):
    """With ``g = 0`` the condition ``g·u·e > 0`` is unsatisfiable, so EVERY policy scores
    ``wrong_sign_fraction = 0.0`` and the instrument silently certifies the failure it exists to
    catch. That must be a construction error, not a table of zeros."""
    with pytest.raises(ValueError, match="post_break_gain_sign"):
        SignFlipLoss(post_break_gain_sign=bad)
    with pytest.raises(ValueError):
        loss_for(SignFlipPlant({"seed": 0}), post_break_gain_sign=0.0)
    # …and a plant whose lever does nothing after the break has no direction to judge at all
    with pytest.raises(ValueError, match="zero/non-finite"):
        true_post_break_gain_sign(SignFlipPlant({"flip_factor": 0.0}))


def test_mandate_names_the_trade_off_without_giving_away_the_gain():
    text = SignFlipLoss(post_shock_step=BREAK).describe().lower()
    assert "target" in text and "control" in text
    assert "not guaranteed to stay the same" in text  # honest warning that the regime may move
    for giveaway in ("sign", "revers", "flip", "negat", "opposite"):
        assert giveaway not in text, f"the mandate hands the regent the answer ('{giveaway}')"


def test_reference_laws_compile_against_the_published_observables():
    """Both references must be policies the regent could actually have emitted."""
    system = SignFlipPlant({"seed": 0})
    iface = ScalarLeverInterface([Lever("set_control_input", system.u_range, "current_u")])
    for expr in (frozen_law(), reversed_law(BREAK), detuned_law(BREAK)):
        result = iface.validate(ActionRequest("regent:0", "set_control_input", {"expr": expr}), system, "regent:0")
        assert result.ok, f"{expr!r} is not expressible in the policy language: {result.feedback}"
    assert str(NOMINAL_FEEDBACK_GAIN) in frozen_law()
