"""
Tests for the validity gates and the corrected statistics.

These are the paper's most transferable claim, so they need to be more than assertions in a
docstring. Each test encodes a failure this study actually shipped and then caught:

* a channel that fires but reports a *clock* rather than performance;
* a percentile bootstrap that rejects at ~8% while claiming 5%;
* a null reported without saying what it could have detected.

Key-free and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from govsim.analysis.stats import (
    bootstrap_ci,
    bootstrap_ci_t,
    channel_liveness,
    minimum_detectable_effect,
    wilcoxon_signed_rank_p,
)


# --- channel liveness ------------------------------------------------------------------------

def test_liveness_passes_a_channel_that_tracks_the_truth():
    truth = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    reported = [t + 0.1 * ((-1) ** i) for i, t in enumerate(truth)]  # noisy but monotone-ish
    res = channel_liveness(reported, truth)
    assert res["live"], res["reason"]
    assert res["rho"] > 0.8


def test_liveness_fails_a_clock():
    """The exact defect we shipped: the channel reported cumulative spend, so it tracked TIME.

    Performance wandered; the reported number marched. A loss table cannot tell this from a
    component that simply did not help.
    """
    rng = np.random.default_rng(0)
    truth = list(rng.normal(0.0, 1.0, 30))          # actual window quality, no trend
    reported = [-float(i) for i in range(30)]        # a clock
    res = channel_liveness(reported, truth)
    assert not res["live"]
    assert "rank correlation" in res["reason"]


def test_liveness_fails_a_channel_that_never_fired():
    res = channel_liveness([], [1.0, 2.0, 3.0])
    assert not res["live"]
    assert "never fired" in res["reason"]


def test_liveness_fails_a_constant_report():
    res = channel_liveness([1.0] * 12, list(range(12)))
    assert not res["live"]
    assert "constant" in res["reason"]


def test_liveness_reports_sign_agreement_separately_from_correlation():
    """Direction is what the channel is practically for, so it is reported on its own."""
    truth = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    res = channel_liveness(truth, truth)
    assert res["sign_agreement"] == pytest.approx(1.0)


# --- interval coverage -----------------------------------------------------------------------

def _coverage(fn, trials: int = 400, n: int = 20) -> float:
    rng = np.random.default_rng(11)
    hit = 0
    for i in range(trials):
        x = rng.normal(0.0, 1.0, n)
        _, lo, hi = fn(x, n_boot=400, seed=int(i))
        hit += lo <= 0.0 <= hi
    return hit / trials


def test_studentized_interval_covers_better_than_the_percentile_one():
    """The decision rule every claim rests on is 'the CI excludes 0'; its level has to be real.

    Kept as a comparison rather than an absolute threshold because both estimates are themselves
    noisy at this trial count — what must hold is the ORDERING, which is the reason we switched.
    """
    pct = _coverage(bootstrap_ci)
    stu = _coverage(bootstrap_ci_t)
    assert stu > pct, f"bootstrap-t ({stu:.3f}) should cover better than percentile ({pct:.3f})"
    assert stu > 0.90, f"studentized coverage {stu:.3f} is too far below nominal 0.95 to use"


def test_intervals_degenerate_gracefully_on_tiny_samples():
    for fn in (bootstrap_ci, bootstrap_ci_t):
        point, lo, hi = fn([1.0])
        assert point == lo == hi == 1.0
        assert fn([]) == (0.0, 0.0, 0.0)


# --- distribution-free second opinion --------------------------------------------------------

def test_wilcoxon_detects_a_clear_shift_and_ignores_noise():
    rng = np.random.default_rng(3)
    assert wilcoxon_signed_rank_p(list(rng.normal(2.0, 1.0, 20))) < 0.01
    assert wilcoxon_signed_rank_p(list(rng.normal(0.0, 1.0, 20))) > 0.05


def test_wilcoxon_refuses_to_opine_on_too_few_points():
    assert wilcoxon_signed_rank_p([1.0, 2.0, 3.0]) == 1.0


def test_wilcoxon_handles_ties_and_zeros_without_blowing_up():
    p = wilcoxon_signed_rank_p([0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, 2.0])
    assert 0.0 <= p <= 1.0


# --- minimum detectable effect ---------------------------------------------------------------

def test_mde_shrinks_with_n_and_grows_with_correction():
    rng = np.random.default_rng(5)
    small = minimum_detectable_effect(list(rng.normal(0.0, 1.0, 20)))
    large = minimum_detectable_effect(list(rng.normal(0.0, 1.0, 80)))
    assert large["mde"] < small["mde"], "more seeds must resolve smaller effects"

    uncorrected = minimum_detectable_effect(list(rng.normal(0.0, 1.0, 20)), n_comparisons=1)
    corrected = minimum_detectable_effect(list(rng.normal(0.0, 1.0, 20)), n_comparisons=7)
    assert corrected["alpha_effective"] < uncorrected["alpha_effective"]


def test_mde_states_the_cost_of_halving_it():
    """The obvious follow-up to a null is 'add seeds'; the answer is quadratic and should be said."""
    res = minimum_detectable_effect(list(np.random.default_rng(7).normal(0.0, 1.0, 20)))
    assert res["n_for_half_mde"] == 80


def test_mde_is_undefined_rather_than_optimistic_on_one_point():
    res = minimum_detectable_effect([1.0])
    assert res["mde"] == float("inf")
