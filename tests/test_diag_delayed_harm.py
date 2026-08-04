"""
Tests for the ``DelayedHarm`` diagnostic (credit assignment across a long delay).

These are not smoke tests. A diagnostic world is a measuring instrument, and an instrument that
merely runs measures nothing — so the load-bearing tests here are the ones that check the world
DISCRIMINATES: that the naive policy it was built to catch scores badly on ``deferred_damage``
while the known-correct policy scores well, on every seed, with a margin that survives the
per-seed heterogeneity. If those fail, the world is broken, not the test.

The rest pin the interface contract (bounded arms, per-seed trajectories, the hidden parameter
absent from ``observe``) and two properties that were WRONG in the first cut of this world and are
therefore worth a regression test: that the DELAY shock actually bites, and that the
discriminating metric separates policies far more sharply than loss does.

Three of these tests exist to answer challenges a sceptic should raise, and are worth naming so
nobody deletes them as redundant:

  - separation is asserted against the SEED NOISE, not just as a mean gap — the per-seed ranges of
    the naive and correct arms must be disjoint, for both the coarse pair (naive vs correct) and
    the subtle one (stale vs retuned). A mean difference with overlapping ranges is not an
    instrument, and nothing else here would have caught that.
  - ``deferred_damage`` must not be a lever-use gauge wearing a costume. It scores ``u = 0`` at
    exactly zero, which is what such a gauge would do, so two counterexamples are pinned: a policy
    that pulls the lever half as much as the reference and scores far worse, and one that pulls it
    twice as much and scores better. Plus the clean version — a CONSTANT policy, bit-identical
    action path, scoring 1.68x higher in the shocked world than the frozen one.
  - the reference policy is checked against a brute-force sweep, because "known-correct" is an
    assumption until someone sweeps it.
"""

from __future__ import annotations

import numpy as np
import pytest

from govsim.core.sandbox import compile_expr
from govsim.domains.diagnostics.delayed_harm import (
    LENGTHENING_LAG,
    DeferredHarmLoss,
    DelayedHarm,
    effective_harm,
    myopic_expr,
    reference_expr,
    reference_level,
)
from govsim.scenarios import ShockKind

SHOCK_STEP = 120
HORIZON = 260
SEEDS = range(12)
LAM = 1.0


def _rollout(expr: str | None, seed: int, params: dict | None = None, horizon: int = HORIZON):
    """Install ``expr`` as the lever policy and run ``horizon`` steps, returning (system, trajectory).

    Goes through ``compile_expr`` + ``install_lever`` rather than poking the attribute, so the
    policies under test are genuinely expressible in the regent's language (diagnostic criterion 3)
    and are re-evaluated every step by the same code path a real run uses.
    """
    system = DelayedHarm(dict(params or {}))
    system.reset(seed)
    if expr is not None:
        result = compile_expr(expr, list(system.observe().vars.keys()))
        assert result.ok, f"reference policy {expr!r} is not expressible: {result.feedback}"
        system.install_lever("expedite", result.compiled, system.u_range, "regent:0")
    trajectory = [system.metrics()]
    for _ in range(horizon):
        system.step()
        trajectory.append(system.metrics())
    return system, trajectory


def _components(expr_for, seed: int, params: dict | None = None) -> dict[str, float]:
    """Score one seed. ``expr_for`` is a callable so per-seed policies see that seed's own world."""
    probe = DelayedHarm(dict(params or {}, seed=seed))
    _, trajectory = _rollout(expr_for(probe), seed, params)
    return DeferredHarmLoss(lam=LAM, post_shock_step=SHOCK_STEP).components(trajectory)


def _across_seeds(expr_for, key: str, params: dict | None = None) -> np.ndarray:
    return np.array([_components(expr_for, s, params)[key] for s in SEEDS])


# -- the arms -------------------------------------------------------------------------------

DO_NOTHING = lambda s: "0.0"                                        # noqa: E731
MAX_LEVER = lambda s: "1.0"                                         # noqa: E731
NAIVE = myopic_expr                                                 # the trap: u = backlog / relief
CORRECT = lambda s: reference_expr(s, LAM, delay=30)                # noqa: E731  (retuned to the new lag)
STALE = lambda s: reference_expr(s, LAM, delay=8)                   # noqa: E731  (tuned to the old lag)


# --- interface contract ---------------------------------------------------------------------

def test_lever_attrs_and_metrics_report_the_enacted_lever():
    system = DelayedHarm({})
    assert system.lever_attrs == {"expedite": (0.0, 1.0)}
    _, trajectory = _rollout("0.4", seed=0, horizon=5)
    assert trajectory[-1]["expedite"] == pytest.approx(0.4)


def test_hidden_parameters_are_absent_from_observe():
    """The lag, the harm coefficient and the pipeline are the inference task — publishing any of
    them would delete the problem. ``metrics`` may carry the pipeline (the objective needs it);
    ``observe`` may not, and the sandbox whitelist is built from ``observe``."""
    # A lever must be INSTALLED for this to be a real concealment. With no lever the enacted
    # ``expedite`` stays 0.0, every deposit is 0.0, and ``pipeline_load`` is exactly 0.0 — so
    # asserting the pipeline is hidden while it is empty asserts nothing at all.
    system, _ = _rollout("0.6", seed=0, horizon=20)
    assert system.expedite > 0.0
    assert system.metrics()["pipeline_load"] > 0.0, "the pipeline must actually hold something"
    visible = set(system.observe().vars)
    assert visible == {"backlog", "previous_backlog", "expedite", "t"}
    for hidden in ("delay", "harm_coeff", "staleness_growth", "pipeline_load",
                   "cum_arrived", "cum_deposited", "relief", "rho", "base_inflow"):
        assert hidden not in visible, f"{hidden} leaked into observe()"
    # …and a policy expression genuinely cannot reach them.
    assert not compile_expr("delay", list(visible)).ok
    assert not compile_expr("pipeline_load", list(visible)).ok
    assert "delay" not in system.metrics()


def test_reset_restores_shock_mutated_parameters():
    """A shock overwrites ``delay``; a later reset must give a pristine pre-shock world back,
    otherwise the second "fresh" run of the same object is silently a different experiment."""
    system = DelayedHarm({})
    for _ in range(SHOCK_STEP + 5):
        system.step()
    assert system.delay == 30
    system.reset(1)
    assert system.delay == 8 and system.time == 0
    assert sum(system._pipeline) == 0.0 and system.cum_deposited == 0.0


def test_scenario_is_a_delay_family_shock():
    assert LENGTHENING_LAG.kind is ShockKind.DELAY
    armed = LENGTHENING_LAG.applied_to({"harm_coeff": 4.0})
    assert armed["shock_step"] == 120 and armed["shock_params"] == {"delay": 30}
    assert armed["harm_coeff"] == 4.0  # applied_to must not drop the base config


# --- criterion 5: bounded ---------------------------------------------------------------------

@pytest.mark.parametrize("label,expr", [("do-nothing", "0.0"), ("max-lever", "1.0"),
                                        ("naive", "backlog / 4.0")])
def test_bounded_over_500_steps(label, expr):
    """No arm diverges over 500 steps on any seed — AND the clip is not what stops it.

    Asserting only ``< backlog_max`` would be vacuous: the clip guarantees it. Asserting a peak
    comfortably below the cap is what shows the calibration itself is stable, so the trap is a
    property of the dynamics rather than of a saturating bound.
    """
    for seed in SEEDS:
        system, trajectory = _rollout(expr, seed, horizon=500)
        peak = max(row["backlog"] for row in trajectory)
        assert np.isfinite(peak)
        assert peak < 0.75 * system.backlog_max, f"{label} seed {seed} peaked at {peak}"
        assert np.isfinite(system.cum_deposited) and np.isfinite(sum(system._pipeline))


def test_seeds_give_different_trajectories():
    """Per-seed heterogeneity is drawn in ``reset`` from the system's own Generator. Without it a
    paired shared-seed design has nothing to pair over and its CI is a zero-width interval."""
    finals = [_rollout("0.3", seed)[1][-1]["backlog"] for seed in SEEDS]
    assert len({round(x, 9) for x in finals}) == len(finals)
    inflows = [DelayedHarm({"seed": s}).base_inflow for s in SEEDS]
    harms = [DelayedHarm({"seed": s}).harm_coeff for s in SEEDS]
    assert len({round(x, 9) for x in inflows}) == len(inflows)
    assert len({round(x, 9) for x in harms}) == len(harms)
    # …and the same seed reproduces exactly.
    assert _rollout("0.3", 4)[1][-1]["backlog"] == _rollout("0.3", 4)[1][-1]["backlog"]


# --- criteria 1 & 2: a naive-but-reasonable policy fails, and the failure is attractive -------

def test_the_lever_always_looks_good_one_step_ahead():
    """Criterion 2. The trap must be locally REWARDED, or the world measures carelessness.

    From any state, using the lever strictly improves the very next observed backlog, at every
    level of use. One-step reasoning therefore has no way to prefer restraint.
    """
    for seed in (0, 5, 9):
        base = DelayedHarm({})
        base.reset(seed)
        for _ in range(40):
            base.step()
        outcomes = []
        for u in (0.0, 0.25, 0.5, 1.0):
            branch = base.clone()  # clone carries the Generator AND the pipeline: same stochastic
            branch.expedite = u    # stream, same standing liability, only the action differs
            branch.step()
            outcomes.append(branch.backlog)
        assert outcomes == sorted(outcomes, reverse=True), outcomes
        assert outcomes[-1] < outcomes[0] - 1.0, "the immediate payoff must be large, not marginal"


def test_naive_policy_is_worse_than_doing_nothing():
    """Criterion 1. The myopic controller must actually FAIL, not merely be suboptimal — it is
    beaten by the null policy of never touching the lever, which is the strongest available
    statement that local reasoning is actively harmful here."""
    naive = _across_seeds(NAIVE, "loss")
    nothing = _across_seeds(DO_NOTHING, "loss")
    assert np.all(naive > nothing), (naive, nothing)
    assert naive.mean() > 1.7 * nothing.mean()


# --- criterion 4: THE DISCRIMINATING METRIC (the test that matters) ---------------------------

def test_naive_scores_badly_and_correct_scores_well_on_deferred_damage():
    """THE diagnostic test. If this fails the world is not an instrument and must be fixed.

    ``deferred_damage`` is the liability the policy is standing in at the horizon plus what it
    already paid after the break. The myopic controller must be buried by it; the known-correct
    standing level must not be.
    """
    naive = _across_seeds(NAIVE, "deferred_damage")
    correct = _across_seeds(CORRECT, "deferred_damage")
    assert np.all(naive > correct), list(zip(naive, correct))
    ratios = naive / correct
    assert ratios.min() > 5.0, f"margin too thin on some seed: {ratios}"
    assert naive.mean() > 10.0 * correct.mean()
    # THE MARGIN MUST BEAT THE SEED NOISE, not merely be positive on average. Two statements of
    # that, because a mean gap with overlapping per-seed ranges is not a usable instrument:
    # the ranges are disjoint (measured 538.5 vs 70.9), and the standardized gap is enormous.
    assert naive.min() > correct.max(), (
        f"ranges overlap: worst naive {naive.min():.1f} vs best correct {correct.max():.1f}")
    pooled = float(np.sqrt((naive.var(ddof=1) + correct.var(ddof=1)) / 2.0))
    assert (naive.mean() - correct.mean()) / pooled > 3.0, f"Cohen d too small (pooled sd {pooled:.1f})"
    # The correct policy is not merely better on the diagnostic — it is also the better GOVERNOR,
    # so the metric is not rewarding a policy that bought its low liability with a bad mandate.
    assert _across_seeds(CORRECT, "loss").mean() < _across_seeds(NAIVE, "loss").mean()
    assert _across_seeds(CORRECT, "loss").mean() < _across_seeds(DO_NOTHING, "loss").mean()


def test_deferred_damage_separates_where_loss_barely_does():
    """Why ``loss`` alone cannot diagnose (criterion 4), stated as a measurement.

    The stale rule (correct for the OLD lag) and the retuned rule (correct for the new one) differ
    by ~10% in post-shock loss — noise, on a single run — but by ~3x in deferred damage, on every
    seed. Loss is collected only if the clock runs long enough to collect it; the liability is
    there either way. That gap IS the instrument.
    """
    stale_loss = _across_seeds(STALE, "post_loss")
    correct_loss = _across_seeds(CORRECT, "post_loss")
    stale_def = _across_seeds(STALE, "deferred_damage")
    correct_def = _across_seeds(CORRECT, "deferred_damage")

    loss_gap = stale_loss.mean() / correct_loss.mean()
    damage_gap = stale_def.mean() / correct_def.mean()
    assert loss_gap < 1.25, f"loss was supposed to be nearly blind here, got {loss_gap:.2f}x"
    assert damage_gap > 2.0, f"the metric must see what loss cannot, got {damage_gap:.2f}x"
    # The separations are not merely both positive: the metric's is an order of magnitude larger.
    assert (damage_gap - 1.0) > 8.0 * (loss_gap - 1.0)
    assert np.all(stale_def > correct_def), "and it must see it on every seed, not on average"
    # …and separates them cleanly, not just on average: even the mildest stale seed is worse than
    # the worst correct seed (measured 124.9 vs 70.9). This is the subtler of the two contrasts the
    # world has to resolve, so it is the one whose resolution is worth pinning.
    assert stale_def.min() > correct_def.max(), (
        f"the subtle pair overlaps: {stale_def.min():.1f} vs {correct_def.max():.1f}")


def test_do_nothing_scores_zero_deferred_damage_but_loses_on_loss():
    """The metric is a LIABILITY gauge, not a fitness. Never touching the lever creates no deferred
    damage at all and is still a bad policy. Pinning this stops anyone reading a low
    ``deferred_damage`` as a pass — the diagnosis is the PAIR (competitive loss, high liability)."""
    assert _across_seeds(DO_NOTHING, "deferred_damage").max() == 0.0
    assert _across_seeds(DO_NOTHING, "loss").mean() > 1.4 * _across_seeds(CORRECT, "loss").mean()


def test_the_three_arm_ordering_is_the_one_the_probe_claims():
    """The whole instrument in one assertion block: naive / correct / do-nothing, same seeds.

    Kept separate from the metric tests because it is the table a reader checks first, and because
    it pins the ORDERING (correct < do-nothing < naive on loss) rather than any single margin. If a
    recalibration ever makes doing nothing beat the reference, or makes the trap merely suboptimal
    instead of actively harmful, this is the test that says so.
    """
    loss = {arm: _across_seeds(f, "loss") for arm, f in
            (("naive", NAIVE), ("correct", CORRECT), ("nothing", DO_NOTHING))}
    dd = {arm: _across_seeds(f, "deferred_damage") for arm, f in
          (("naive", NAIVE), ("correct", CORRECT), ("nothing", DO_NOTHING))}
    # loss: the correct policy governs best, the trap is worse than not governing at all.
    assert loss["correct"].mean() < loss["nothing"].mean() < loss["naive"].mean()
    assert np.all(loss["naive"] > loss["nothing"])   # on every seed, not on average
    assert np.all(loss["naive"] > loss["correct"])
    # deferred damage: the trap is buried, the reference is not, the null is exactly zero.
    assert dd["nothing"].max() == 0.0
    assert dd["naive"].min() > dd["correct"].max()
    # and the metric separates the two ACTIVE arms far more sharply than loss does.
    assert (dd["naive"].mean() / dd["correct"].mean()) > 4.0 * (
        loss["naive"].mean() / loss["correct"].mean())


def test_deferred_damage_is_not_a_lever_use_gauge_in_disguise():
    """The first thing to suspect of a metric that scores ``u = 0`` at exactly zero.

    If ``deferred_damage`` were monotone in how hard the lever was pulled it would carry no
    information the regent's own action log lacks, and it would not be evidence of anything. Two
    counterexamples, both on the same seeds as everything else: a policy that uses the lever HALF as
    much as the reference and scores far worse, and one that uses it TWICE as much and scores
    better. What the metric actually prices is when the pull happened — under which hidden lag, and
    how close to the horizon it landed.
    """
    late_dump = lambda s: "1.0 if t >= 230 else 0.0"     # noqa: E731  only the last 30 steps
    early_only = lambda s: "1.0 if t < 120 else 0.0"     # noqa: E731  saturate, then stop dead

    use = {k: _across_seeds(f, "mean_expedite") for k, f in
           (("late", late_dump), ("early", early_only), ("ref", CORRECT))}
    dd = {k: _across_seeds(f, "deferred_damage") for k, f in
          (("late", late_dump), ("early", early_only), ("ref", CORRECT))}

    # Uses the lever about half as much as the reference…
    assert use["late"].mean() < 0.7 * use["ref"].mean()
    # …and is buried by the metric anyway, because it dumped under the long lag and near the end.
    assert dd["late"].mean() > 3.0 * dd["ref"].mean()

    # Uses it more than twice as much as the reference…
    assert use["early"].mean() > 2.0 * use["ref"].mean()
    # …and scores BETTER, because by the horizon it is genuinely standing in almost nothing.
    assert dd["early"].mean() < dd["ref"].mean()


def test_identical_actions_score_differently_when_the_hidden_lag_differs():
    """The cleanest statement that the metric is not computable from the action log.

    ``test_the_shock_raises_deferred_damage_for_a_rule_tuned_on_the_old_lag`` uses feedback rules,
    whose enacted lever path differs between the frozen and shocked worlds — so its comparison is
    confounded by what the policy did. A CONSTANT policy removes that: the action path is bit-for-bit
    identical in both worlds, and the only thing that changed is the lag the regent cannot see.
    """
    const = lambda s: "0.3"  # noqa: E731
    frozen_u = _across_seeds(const, "mean_expedite", {"shock_step": None})
    shocked_u = _across_seeds(const, "mean_expedite")
    assert np.allclose(frozen_u, shocked_u), "the action path must be identical for this to be clean"

    frozen = _across_seeds(const, "deferred_damage", {"shock_step": None})
    shocked = _across_seeds(const, "deferred_damage")
    assert np.all(shocked > 1.5 * frozen), (frozen, shocked)
    # The ratio is a pure function of the lag (the per-seed ``harm_coeff`` and the constant ``u``
    # both cancel), so it is identical on every seed — which is itself the point: the metric is
    # reading the world, not the policy.
    ratios = shocked / frozen
    assert np.allclose(ratios, ratios[0]), ratios
    # It tracks the compounding factor ``growth^30 / growth^8 = 1.722`` but sits slightly BELOW it
    # (1.683, measured), and the gap is not noise: the post-shock window opens with 8 deposits still
    # in flight at the OLD size, so a little of the window's damage is priced at the old lag.
    lag_ratio = (1.025 ** 30) / (1.025 ** 8)
    assert ratios[0] == pytest.approx(lag_ratio, rel=0.05)
    assert ratios[0] < lag_ratio


def test_the_reference_level_really_is_the_best_constant_standing_level():
    """Criterion 3 has teeth only if the "known-correct" policy is actually near-optimal.

    ``reference_level`` is a closed form for the STEADY-STATE optimum; this checks it against a
    brute-force sweep over constant levels on the same seeds. They do not have to agree exactly —
    a finite window under-prices deposits written in its last ``delay`` steps, so its empirical
    argmin sits a little higher (u ≈ 0.26 vs u* ≈ 0.217) — but the reference must be within a few
    percent of the best constant, or it is not a credible oracle to score regents against.
    """
    grid = [0.16, 0.20, 0.24, 0.28, 0.32, 0.40]
    swept = {u: _across_seeds(lambda s, u=u: f"{u}", "post_loss").mean() for u in grid}
    best = min(swept.values())
    ref = _across_seeds(CORRECT, "post_loss").mean()
    assert ref < 1.03 * best, f"reference {ref:.1f} is more than 3% off the best constant {best:.1f}"
    # …and it is nowhere near the stale level or the extremes, so "near-optimal" is not vacuous.
    assert ref < 0.9 * swept[0.40]
    assert ref < _across_seeds(STALE, "post_loss").mean()


# --- the DELAY shock actually bites -----------------------------------------------------------

def test_lengthening_the_lag_raises_the_cost_of_the_lever_and_lowers_the_correct_level():
    """The property the first cut of this world did NOT have.

    Without compounding, a longer lag only reschedules damage: every deposit still lands, so the
    correct standing level is lag-invariant and a DELAY shock has no adaptation headroom at all —
    it is strictly GOOD for a fixed policy inside a fixed horizon. ``staleness_growth`` is what
    makes the lag governable, so it gets a regression test.
    """
    system = DelayedHarm({})
    system.reset(0)
    assert effective_harm(system, 30) > 1.6 * effective_harm(system, 8)
    assert reference_level(system, LAM, delay=30) < 0.7 * reference_level(system, LAM, delay=8)


def test_the_shock_raises_deferred_damage_for_a_rule_tuned_on_the_old_lag():
    """A rule tuned on the old lag over-uses the lever once the lag lengthens — measured, on the
    same seeds, against the identical policy in a world where the lag never moves."""
    for policy in (NAIVE, STALE):
        frozen = _across_seeds(policy, "deferred_damage", {"shock_step": None})
        shocked = _across_seeds(policy, "deferred_damage")
        assert shocked.mean() > 1.3 * frozen.mean(), (frozen.mean(), shocked.mean())
        assert np.all(shocked > frozen)


def test_the_break_makes_the_evidence_actively_misleading():
    """The cruel part, and the reason a competent-looking agent can still fail.

    New work goes to the back of a much longer queue while work already in flight lands on the old
    schedule, so there is an arrival holiday right after the break. Under heavy use the observed
    backlog COLLAPSES during it — while the pipeline is at its fullest and a unit of lever has just
    got twice as expensive. Every observable says the lever started working better.
    """
    _, trajectory = _rollout(myopic_expr(DelayedHarm({"seed": 0})), 0)
    def mean_backlog(lo, hi):
        return float(np.mean([r["backlog"] for r in trajectory if lo <= r["t"] < hi]))
    before = mean_backlog(SHOCK_STEP - 25, SHOCK_STEP)
    holiday = mean_backlog(SHOCK_STEP, SHOCK_STEP + 25)
    after = mean_backlog(SHOCK_STEP + 60, HORIZON + 1)
    assert holiday < 0.6 * before, "the break must LOOK like an improvement"
    assert after > 1.5 * before, "…and then be much worse than before it"
    pipeline = [r["pipeline_load"] for r in trajectory if SHOCK_STEP <= r["t"] < SHOCK_STEP + 25]
    assert min(pipeline) > 0.0  # the liability never went away while the backlog was falling


# --- the objective's window and differencing contract -----------------------------------------

def test_empty_post_shock_window_scores_worst_case_not_zero():
    """A run that ends before the break has no post-shock evidence. Scoring the empty window 0.0
    would make "end the run early" the optimal post-shock policy AND would report zero liability
    for a policy that never lived long enough to pay any."""
    _, short = _rollout("1.0", seed=0, horizon=10)
    comps = DeferredHarmLoss(lam=LAM, post_shock_step=SHOCK_STEP).components(short)
    assert comps["post_loss"] == float("inf")
    assert comps["deferred_damage"] == float("inf")
    assert np.isfinite(comps["loss"])  # the full-horizon score is unaffected


def test_cumulative_damage_is_differenced_across_the_window():
    """``cum_deposited``/``cum_arrived`` are running totals from t=0. Read undifferenced they turn
    the score into a clock — the bug that once made this platform's realized-performance signal
    report "worse than last time" 359 times and "better" never."""
    _, trajectory = _rollout("0.5", seed=0)
    obj = DeferredHarmLoss(lam=LAM, post_shock_step=SHOCK_STEP)
    comps = obj.components(trajectory)
    post = [r for r in trajectory if r["t"] >= SHOCK_STEP]
    assert comps["post_damage"] == pytest.approx(
        post[-1]["cum_deposited"] - post[0]["cum_deposited"])
    assert comps["post_damage"] < comps["cum_deposited"]  # i.e. NOT the undifferenced total
    # ``pipeline_load`` is a LEVEL, so it is read off the last row and NOT differenced; getting
    # that pair backwards is the same class of error and would be invisible in the number.
    assert comps["deferred_damage"] == pytest.approx(
        post[-1]["pipeline_load"] + (post[-1]["cum_arrived"] - post[0]["cum_arrived"]))

    # The damage term must TELESCOPE across adjacent windows sharing a boundary row, which is the
    # property that makes the Runner's per-interval feedback signal mean anything.
    mid = len(trajectory) // 2
    first, second = trajectory[:mid + 1], trajectory[mid:]
    span = lambda rows: rows[-1]["cum_deposited"] - rows[0]["cum_deposited"]  # noqa: E731
    assert span(first) + span(second) == pytest.approx(span(trajectory), rel=1e-9)


def test_the_score_of_a_window_does_not_depend_on_when_it_happened():
    """The undifferenced-total bug, pinned directly.

    Two equal-length windows, same constant policy, both in the pre-shock steady state: they must
    score the same, because the policy did the same thing in both. Reading ``cum_deposited`` off
    the last row instead of differencing would bill the later window for every deposit made since
    t=0 and turn the score into a clock that only ever gets worse.
    """
    _, trajectory = _rollout("0.5", seed=0)
    obj = DeferredHarmLoss(lam=LAM)
    early = [r for r in trajectory if 60 <= r["t"] < 90]
    late = [r for r in trajectory if 90 <= r["t"] < 120]
    assert obj.evaluate(early) == pytest.approx(obj.evaluate(late), rel=0.05)

    # …and the buggy reading really would have separated them, so this test has teeth: score the
    # same two windows the wrong way (last row's running total, undifferenced) and watch the later
    # window come out materially worse for no reason but the clock.
    def undifferenced(rows):
        return -(sum(r["backlog"] for r in rows) + LAM * rows[-1]["cum_deposited"])
    assert undifferenced(late) < undifferenced(early)
    drift = abs(undifferenced(late) - undifferenced(early)) / abs(undifferenced(early))
    assert drift > 0.05, f"the wrong reading must break the 5% tolerance above, drifted {drift:.3f}"


def test_end_of_horizon_dumping_beats_the_null_on_loss_and_is_caught_by_the_metric():
    """A known, deliberately unpatched artefact — pinned so it is a finding and not a surprise.

    Saturating the lever over only the final 30 steps scores BETTER than never touching it: the
    relief lands inside the horizon, the backlog consequence does not. Charging damage on creation
    blunts this but cannot remove it, and the only clean fix — a terminal charge on the pipeline
    residual — would break the additivity across adjacent windows that the Runner's per-interval
    feedback depends on. So the artefact stays and ``deferred_damage`` is what catches it.

    This matters for interpretation, not just bookkeeping: a regent that finds this has not failed
    the probe, it has solved the credit-assignment problem and then arbitraged the clock. The pair
    (loss better than the null, deferred damage far above it) is what names that case.
    """
    late_dump = lambda s: "1.0 if t >= 230 else 0.0"  # noqa: E731
    assert _across_seeds(late_dump, "loss").mean() < _across_seeds(DO_NOTHING, "loss").mean()
    assert _across_seeds(late_dump, "deferred_damage").min() > 0.0
    assert _across_seeds(late_dump, "deferred_damage").mean() > 3.0 * _across_seeds(
        CORRECT, "deferred_damage").mean()


def test_mandate_states_the_trade_off_without_leaking_the_lag():
    text = DeferredHarmLoss(lam=LAM).describe()
    assert "not told how long that delay is" in text
    assert "8" not in text and "30" not in text  # the lag's actual length stays hidden
    for word in ("backlog", "delay", "rework"):
        assert word in text
