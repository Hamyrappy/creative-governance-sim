"""
Tests for the statistics layer (paired bootstrap CI, variance-aware selection, collapse detector,
metric extraction). Pure functions are tested directly; ``compare`` is also exercised on REAL
RunRecords from two scripted experiments that share a system + seeds but differ in control gain.
"""

from __future__ import annotations

from govsim.analysis import (
    bootstrap_ci, collapse_summary, compare, infer_lower_is_better, metric_by_seed,
    paired_diff, robust_score, variance_aware_select,
)
from govsim.core import EveryN, Experiment, Hypothesis, Runner
from govsim.core.regent import ScriptedRegent
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss


# --- pure functions -------------------------------------------------------------------------

def test_robust_score_penalizes_variance():
    assert robust_score([1.0, 1.0, 1.0], lam=0.5) == 1.0
    assert robust_score([0.0, 2.0], lam=0.5) < robust_score([1.0, 1.0], lam=0.5)  # same mean, more std


def test_variance_aware_select_prefers_low_variance_at_equal_mean():
    # candidate 1 mean=1 std=0 ; candidate 0 mean=1 std=1 ⇒ pick 1
    assert variance_aware_select([[0.0, 2.0], [1.0, 1.0]], lam=0.5) == 1


def test_bootstrap_ci_zero_width_on_constant_diffs():
    point, lo, hi = bootstrap_ci([0.3, 0.3, 0.3, 0.3], n_boot=2000, seed=0)
    assert point == 0.3 and lo == 0.3 and hi == 0.3


def test_bootstrap_ci_brackets_mean():
    point, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=5000, seed=0)
    assert abs(point - 3.0) < 1e-9 and lo < 3.0 < hi


def test_compare_detects_significant_paired_improvement():
    # A uniformly lower (better) loss than B on every seed ⇒ CI excludes 0, A beats B
    a = {0: 0.10, 1: 0.12, 2: 0.09, 3: 0.11, 4: 0.10}
    b = {0: 0.30, 1: 0.31, 2: 0.29, 3: 0.32, 4: 0.30}
    res = compare(a, b, lower_is_better=True, seed=0)
    assert res["excludes_zero"] and res["a_better_than_b"]
    assert res["ci_high"] < 0  # A - B significantly negative


def test_compare_no_difference_includes_zero():
    a = {0: 0.10, 1: 0.30, 2: 0.20}
    b = {0: 0.30, 1: 0.10, 2: 0.20}  # same values, swapped ⇒ mean diff ~0
    res = compare(a, b, lower_is_better=True, seed=0)
    assert not res["a_better_than_b"]


def test_infer_lower_is_better():
    assert infer_lower_is_better("mse") and infer_lower_is_better("cost")
    assert not infer_lower_is_better("profit") and not infer_lower_is_better("score")


# --- integration on real RunRecords ---------------------------------------------------------

def _scripted_exp(name: str, expr: str) -> Experiment:
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.05})
        s.reset(seed)
        s.current_x = 1.0
        return s

    return Experiment(
        name=name,
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-3.0, 3.0), "current_u")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr=expr)},
        objectives={"regent:0": StabilizationLoss(lam=0.05)},
        schedule=EveryN(10),
        seeds=[0, 1, 2, 3, 4],
        horizon=120,
        hypothesis=Hypothesis(id="H", claim="c", baseline="b", primary_metric="mse"),
    )


def test_compare_on_real_runs_strong_gain_beats_weak():
    strong = Runner().run(_scripted_exp("strong", "-1.9 * current_x"))  # pole-cancelling
    weak = Runner().run(_scripted_exp("weak", "-0.3 * current_x"))      # barely controls
    a = metric_by_seed(strong, "mse")
    b = metric_by_seed(weak, "mse")
    res = compare(a, b, lower_is_better=True, seed=0)
    assert res["n"] == 5
    assert res["a_better_than_b"]  # the strong-gain regent has significantly lower MSE


def test_collapse_summary_flags_terminated_runs():
    # an explosive plant with no control ⇒ terminates (|x|>1e9)
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 2.0, "param_B": 0.0, "sigma_epsilon": 0.0})
        s.reset(seed)
        s.current_x = 5.0
        return s

    exp = Experiment(
        name="boom",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-1.0, 1.0), "current_u")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="0.0")},
        objectives={"regent:0": StabilizationLoss(lam=0.0)},
        schedule=EveryN(10),
        seeds=[0, 1],
        horizon=200,
        hypothesis=Hypothesis(id="H", claim="c", baseline="b", primary_metric="mse"),
    )
    recs = Runner().run(exp)
    cs = collapse_summary(recs)
    assert cs["n_terminated"] == 2 and set(cs["terminated_seeds"]) == {0, 1}
