"""
Tests for the ``StrategicPopulation`` diagnostic.

These do not merely check that the code runs. A diagnostic world is only an instrument if the
failure it was built to catch actually happens to the policy it was built to catch, and does NOT
happen to the policy that anticipates it. The load-bearing tests are:

  - ``test_naive_sharp_rule_is_gamed_but_smooth_response_is_not`` — the discrimination itself;
  - ``test_moving_threshold_also_defeats_bunching`` — the second correct answer;
  - ``test_naive_rule_is_attractive_before_the_population_learns`` — the trap must be BAITED, or we
    are measuring carelessness rather than foresight;
  - ``test_the_decay_is_endogenous_no_exogenous_shock_needed`` — the property that distinguishes
    this world from every other one on the platform.

If any of those fail, the world is broken, not the test.
"""

from __future__ import annotations

import numpy as np

from govsim.core.action import ActionRequest
from govsim.domains.diagnostics.strategic_population import (
    StrategicComplianceLoss,
    StrategicPopulation,
)
from govsim.domains.scalar import Lever, ScalarLeverInterface

SHOCK_STEP = 120
HORIZON = 240

# The three policies the world exists to tell apart, all in the platform's one-expression language.
NAIVE_SHARP = {"set_threshold": "0.6", "set_response_width": "0.0"}
SMOOTH_BAND = {"set_threshold": "0.6", "set_response_width": "1.2"}   # width > penalty/evasion_unit_cost
MOVING_LINE = {"set_threshold": "0.30 if (t % 6) < 3 else 0.95", "set_response_width": "0.0"}
DO_NOTHING: dict[str, str] = {}
MAX_LEVER = {"set_threshold": "0.0", "set_response_width": "1.5"}


def _iface(system: StrategicPopulation) -> ScalarLeverInterface:
    return ScalarLeverInterface(
        [Lever(f"set_{attr}", rng, attr) for attr, rng in system.lever_attrs.items()]
    )


def _run(policy: dict[str, str], seed: int, horizon: int = HORIZON,
         params: dict | None = None) -> tuple[StrategicPopulation, list[dict[str, float]]]:
    system = StrategicPopulation(params if params is not None else {"shock_step": SHOCK_STEP})
    system.reset(seed)
    if policy:
        report = _iface(system).apply(
            [ActionRequest("regent:0", verb, {"expr": expr}) for verb, expr in policy.items()], system
        )
        assert not report.rejected, report.rejected  # a rejected reference silently becomes do-nothing
    trajectory = []
    for _ in range(horizon):
        system.step()
        trajectory.append(system.metrics())
    return system, trajectory


def _post_gap(policy: dict[str, str], seeds: range) -> list[float]:
    obj = StrategicComplianceLoss(post_shock_step=SHOCK_STEP)
    return [obj.components(_run(policy, s)[1])["post_evasion_gap"] for s in seeds]


# --- THE test: the trap catches the naive rule and spares the two correct ones ----------------

def test_naive_sharp_rule_is_gamed_but_smooth_response_is_not():
    """A fixed sharp threshold is bunched against; a proportional band is not.

    The mechanism, restated correctly (an earlier version of this comment claimed a break-even width
    of ``penalty / evasion_unit_cost = 0.75`` above which gaming is a strict loss — there is no such
    width, see ``test_perceived_cliff_sets_the_temptation_depth``): a unit buys concealment while it
    sits within the temptation depth ``D(w) = (penalty/evasion_unit_cost)·min(1, 2·bunch_margin/w)``
    of the believed line. A step rule saturates ``D`` at 0.75; ``w = 1.2`` cuts it to 0.0625. The
    incentive is never abolished, only priced down — but priced down by enough that the two ranges
    do not come close to overlapping.
    """
    seeds = range(8)
    naive = _post_gap(NAIVE_SHARP, seeds)
    smooth = _post_gap(SMOOTH_BAND, seeds)

    assert min(naive) > 0.10, f"the trap did not spring on the naive rule: {naive}"
    assert max(smooth) < 0.01, f"the smooth reference is being gamed too: {smooth}"
    # Separated per-seed, not merely on average — every naive run is worse than every smooth run.
    assert min(naive) > 10 * max(smooth)
    # …and the gap is what makes the two policies distinguishable, not a rounding difference.
    assert np.mean(naive) - np.mean(smooth) > 0.10


def test_the_separation_is_wider_than_the_seed_to_seed_noise():
    """The validity condition for the whole instrument, asserted as such rather than implied.

    A diagnostic that separates by less than its own sampling noise measures the seed, not the
    policy. Measured: naive 0.193 (sd 0.020) vs smooth 0.0008 (sd 0.0005), so the gap between the
    arm means is ~10 naive-sds wide and the worst naive seed still clears the best smooth seed by
    +0.162. The thresholds below are deliberately slack versions of those numbers — this test should
    fail on a real regression, not on the third decimal.
    """
    seeds = range(10)
    naive = _post_gap(NAIVE_SHARP, seeds)
    smooth = _post_gap(SMOOTH_BAND, seeds)
    separation = float(np.mean(naive) - np.mean(smooth))
    noise = max(float(np.std(naive)), float(np.std(smooth)))
    assert separation > 6.0 * noise, f"separation {separation:.4f} vs noise {noise:.4f}"
    # No overlap at all between the two arms' per-seed ranges.
    assert min(naive) - max(smooth) > 0.05, (min(naive), max(smooth))


def test_every_sharp_line_is_trapped_not_just_the_one_we_picked():
    """The trap must be a property of the DISCONTINUITY, not of ``threshold == 0.6``.

    If only the hand-picked naive setting were gamed, this world would be a single tuned
    counterexample dressed up as an instrument — a regent that happened to choose 0.9 would look
    prescient for no reason. Sweeping the sharp rule across the plausible range of thresholds, every
    one of them is bunched against, and all of them by margins far above the correct policies'
    ~0.001.
    """
    seeds = range(4)
    gaps = {thr: float(np.mean(_post_gap(
        {"set_threshold": f"{thr}", "set_response_width": "0.0"}, seeds)))
        for thr in (0.4, 0.6, 0.8, 1.0, 1.2)}
    assert min(gaps.values()) > 0.03, gaps
    assert max(_post_gap(SMOOTH_BAND, seeds)) < 0.01
    # Monotone in the threshold: a line drawn higher sits above more of the population, so fewer
    # units are close enough to it to find concealment worth buying. A non-monotone sweep would mean
    # something other than the advertised mechanism is driving the result.
    values = [gaps[t] for t in (0.4, 0.6, 0.8, 1.0, 1.2)]
    assert values == sorted(values, reverse=True), gaps


def test_perceived_cliff_sets_the_temptation_depth_so_the_gap_decays_as_one_over_w_squared():
    """Pins the mechanism as DERIVED, not as observed-and-rationalized.

    The population estimates the cliff by comparing peers ``±bunch_margin`` either side of the
    believed line, so a ramp of width ``w`` shows it only ``min(1, 2·bunch_margin/w)`` of the
    penalty. Temptation depth is therefore ``D(w) ∝ 1/w``; both the share of tempted units and the
    concealment each buys scale with ``D``, so the realized gap should fall like ``1/w²``.

    This is the test that would have caught the false 'break-even width' claim: a break-even at
    ``penalty/evasion_unit_cost = 0.75`` predicts gap == 0 for w > 0.75, and the gap is not zero
    there. It is also the guard on ``bunch_margin``, which silently sets the scale of the whole
    strategic layer.
    """
    seeds = range(4)
    gaps = {w: float(np.mean(_post_gap(
        {"set_threshold": "0.6", "set_response_width": f"{w}"}, seeds)))
        for w in (0.4, 0.8, 1.5)}
    # The incentive is never abolished — a wide band still leaves a tempted slice.
    assert all(g > 0.0 for g in gaps.values()), gaps
    # …and gap·w² is the invariant, flat to well within a factor of two across a 3.75× range of w.
    scaled = [gaps[w] * w * w for w in (0.4, 0.8, 1.5)]
    assert max(scaled) / min(scaled) < 1.6, dict(zip((0.4, 0.8, 1.5), scaled))


def test_brute_force_enforcement_does_not_win():
    """The world must reward ANTICIPATION, not enforcement volume.

    The cheapest way for a diagnostic like this to be secretly trivial is for 'intervene on everyone,
    hard' to score as well as the policy that understands the population — then the probe is a test
    of aggression and any regent that turns the dial to eleven passes. ``lam_intervention`` is what
    stops that: brute force drives harm to ~0 and then pays for it, landing at roughly twice the
    smooth band's post-break loss. Note that brute force is ALSO gamed (a line at 0 is still a line),
    which is the second reason it loses.
    """
    obj = StrategicComplianceLoss(post_shock_step=SHOCK_STEP)
    seeds = range(6)

    def post_loss(policy):
        return float(np.mean([obj.components(_run(policy, s)[1])["post_loss"] for s in seeds]))

    smooth = post_loss(SMOOTH_BAND)
    for name, brute in (("wide-at-zero", {"set_threshold": "0.0", "set_response_width": "1.5"}),
                        ("sharp-at-zero", {"set_threshold": "0.0", "set_response_width": "0.0"})):
        assert post_loss(brute) > 1.5 * smooth, (name, post_loss(brute), smooth)
    # The optimum is INTERIOR in the width lever, so 'max the band' is not a free win either.
    interior = post_loss({"set_threshold": "0.6", "set_response_width": "0.6"})
    widest = post_loss({"set_threshold": "0.6", "set_response_width": "1.5"})
    assert interior < widest, (interior, widest)


def test_the_audit_makes_the_trap_diagnosable_and_reports_do_not_fall():
    """The world must be fair rather than blind — and fair in the direction it actually is.

    The signature is NOT 'reports fall'. Under the gamed rule the reported indicator is pinned at
    the enforcement line and even drifts slightly up, while the truth underneath it climbs. A regent
    told to look for a falling indicator would be looking for the wrong thing, so this test asserts
    the direction as well as the magnitude.
    """
    seeds = range(6)
    early_rep, late_rep, early_true, late_true, late_audit = [], [], [], [], []
    for seed in seeds:
        system, trajectory = _run(NAIVE_SHARP, seed)
        early_rep.append(np.mean([r["reported_mean"] for r in trajectory[80:120]]))
        late_rep.append(np.mean([r["reported_mean"] for r in trajectory[200:240]]))
        early_true.append(np.mean([r["true_mean"] for r in trajectory[80:120]]))
        late_true.append(np.mean([r["true_mean"] for r in trajectory[200:240]]))
        late_audit.append(system.observe().vars["audit_harm"])
    # The indicator does not fall — that is the whole point of bunching.
    assert np.mean(late_rep) >= np.mean(early_rep) * 0.98, (np.mean(early_rep), np.mean(late_rep))
    # …while the truth it proxies runs away from it.
    assert np.mean(late_true) > np.mean(early_true) * 1.3, (np.mean(early_true), np.mean(late_true))
    # And the published audit carries that divergence well clear of its own between-seed noise.
    assert np.mean(late_audit) > 6.0 * np.std(late_audit), (np.mean(late_audit), np.std(late_audit))


def test_moving_threshold_also_defeats_bunching():
    """A threshold that keeps moving leaves a cliff that is never where the population thinks it is.

    Worth asserting separately: it proves the world is measuring *anticipation of adaptation* rather
    than a preference for one lever setting. Two structurally different answers pass, and they have
    nothing in common except that neither presents a stable notch.
    """
    seeds = range(8)
    moving = _post_gap(MOVING_LINE, seeds)
    naive = _post_gap(NAIVE_SHARP, seeds)
    assert max(moving) < 0.06, f"the moving threshold is being bunched against: {moving}"
    assert min(naive) > 3 * max(moving)


def test_naive_rule_is_attractive_before_the_population_learns():
    """Requirement (2): the wrong behaviour must be locally REWARDED.

    Over the pre-break window the sharp line out-scores the correct answer — it reaches the same
    units for a fraction of the enforcement effort, because it does not intervene on everyone a
    little (17.05 vs 18.61 mean loss, 8.4% cheaper, every seed). A world where the naive policy were
    dominated from step 0 would measure carelessness rather than foresight.

    Precision about the claim, since it is easy to overstate: the sharp line beats the *correct
    reference*, which is all a trap needs. It is not the pre-break optimum — a narrow band at
    ``w = 0.4`` scores 14.64 over the same window and beats both.
    """
    obj = StrategicComplianceLoss()
    wins = 0
    for seed in range(8):
        naive = obj.components(_run(NAIVE_SHARP, seed, SHOCK_STEP)[1])["loss"]
        smooth = obj.components(_run(SMOOTH_BAND, seed, SHOCK_STEP)[1])["loss"]
        wins += naive < smooth
    assert wins == 8, f"the naive rule is not tempting pre-break (won {wins}/8)"


def test_correct_policies_beat_the_naive_one_after_the_break():
    """The discriminating metric is not decorative: the failure it isolates is also expensive."""
    obj = StrategicComplianceLoss(post_shock_step=SHOCK_STEP)
    for seed in range(6):
        naive = obj.components(_run(NAIVE_SHARP, seed)[1])["post_loss"]
        smooth = obj.components(_run(SMOOTH_BAND, seed)[1])["post_loss"]
        moving = obj.components(_run(MOVING_LINE, seed)[1])["post_loss"]
        assert naive > 2.0 * smooth, (seed, naive, smooth)
        assert naive > 2.0 * moving, (seed, naive, moving)


def test_the_decay_is_endogenous_no_exogenous_shock_needed():
    """The property this world exists for: with ``shock_step=None`` nothing outside the policy ever
    changes, and the sharp rule still rots while the smooth one does not.

    Every other world on the platform needs a parameter overwrite to create adaptation headroom.
    Here the regent's own rule is the perturbation; the armed shock only shortens the fuse.
    """
    stationary = {"shock_step": None}
    horizon, tail = 800, 200
    naive, smooth = [], []
    for seed in range(4):
        _, tn = _run(NAIVE_SHARP, seed, horizon, stationary)
        _, ts = _run(SMOOTH_BAND, seed, horizon, stationary)
        naive.append(float(np.mean([r["evasion_gap"] for r in tn[-tail:]])))
        smooth.append(float(np.mean([r["evasion_gap"] for r in ts[-tail:]])))
    assert min(naive) > 0.08, f"no endogenous decay without a shock: {naive}"
    assert max(smooth) < 0.01, f"the smooth reference decays too: {smooth}"


def test_evasion_gap_separates_two_policies_of_similar_loss():
    """Requirement (4): the metric must diagnose where total loss cannot.

    Doing nothing and enforcing a gamed sharp line both end up with most of the population's true
    activity unsuppressed, so their post-break losses are the same order of magnitude. Only the gap
    says WHY: one authority never intervened, the other was lied to.
    """
    obj = StrategicComplianceLoss(post_shock_step=SHOCK_STEP)
    idle = [obj.components(_run(DO_NOTHING, s)[1]) for s in range(6)]
    gamed = [obj.components(_run(NAIVE_SHARP, s)[1]) for s in range(6)]
    idle_harm = np.mean([c["post_harm"] for c in idle])
    gamed_harm = np.mean([c["post_harm"] for c in gamed])
    assert 0.4 < gamed_harm / idle_harm < 2.5, (gamed_harm, idle_harm)  # comparable outcomes…
    assert max(c["post_evasion_gap"] for c in idle) < 1e-9                # …opposite diagnoses
    assert min(c["post_evasion_gap"] for c in gamed) > 0.10


# --- boundedness ------------------------------------------------------------------------------

def test_bounded_over_500_steps_under_do_nothing_and_max_lever():
    """No arm may diverge. Bounds are structural (clips in ``step``), so assert the real ones."""
    for policy in (DO_NOTHING, MAX_LEVER, NAIVE_SHARP, SMOOTH_BAND, MOVING_LINE):
        system, trajectory = _run(policy, 3, 500)
        b_max = system.behaviour_max
        for row in trajectory:
            for key, value in row.items():
                assert np.isfinite(value), (policy, key, value)
            assert 0.0 <= row["harm"] <= b_max - system.safe_level
            assert 0.0 <= row["evasion_gap"] <= b_max
            assert 0.0 <= row["bunched_share"] <= 1.0
            assert 0.0 <= row["intervention_rate"] <= 1.0
            assert 0.0 <= row["true_mean"] <= b_max
            assert 0.0 <= row["reported_mean"] <= b_max
        # the running totals grow at most linearly in the per-step bound
        assert trajectory[-1]["cum_intervention"] <= 500.0
        assert trajectory[-1]["cum_evasion_cost"] <= 500.0 * system.evasion_unit_cost * b_max
        # internal (hidden) state stays inside its own ranges too
        assert 0.0 <= float(system.behaviour.min()) and float(system.behaviour.max()) <= b_max
        assert 0.0 <= system.line_estimate <= b_max
        assert 0.0 <= system.cliff_estimate <= 1.0


def test_bounded_under_pathological_lever_expressions():
    """Boundedness must survive policies no sane regent would write, because an LLM regent will.

    The suite above only exercises well-formed constant and periodic policies. A lever that raises at
    runtime, that returns absurd magnitudes, or that feeds an observable straight back into itself
    must still leave every published quantity finite and inside its structural clip — otherwise a
    single malformed emission turns into a NaN in the results store.
    """
    pathological = [
        {"set_threshold": "1/0", "set_response_width": "0.0"},                       # runtime raise
        {"set_threshold": "t*1000000000.0", "set_response_width": "1000000000.0"},   # +absurd
        {"set_threshold": "-1000000000.0", "set_response_width": "-1000000000.0"},   # -absurd
        {"set_threshold": "audit_harm*1000.0", "set_response_width": "reported_max*-50.0"},
    ]
    for policy in pathological:
        system, trajectory = _run(policy, 3, 500)
        for row in trajectory:
            for key, value in row.items():
                assert np.isfinite(value), (policy, key, value)
        b_max = system.behaviour_max
        assert 0.0 <= float(system.behaviour.min()) and float(system.behaviour.max()) <= b_max
        assert 0.0 <= system.line_estimate <= b_max
        assert 0.0 <= system.cliff_estimate <= 1.0
        # the levers themselves stayed inside the declared ranges despite the garbage
        assert 0.0 <= system.threshold <= b_max
        assert 0.0 <= system.response_width <= 1.5


# --- reproducibility, heterogeneity, and the per-system RNG ------------------------------------

def test_seeds_give_different_trajectories_and_a_seed_reproduces():
    a = _run(NAIVE_SHARP, 1, 80)[1]
    b = _run(NAIVE_SHARP, 2, 80)[1]
    assert [r["harm"] for r in a] != [r["harm"] for r in b]
    again = _run(NAIVE_SHARP, 1, 80)[1]
    assert [r["harm"] for r in a] == [r["harm"] for r in again]


def test_per_seed_heterogeneity_is_real_not_cosmetic():
    """Identical seeds would make the platform's paired statistics vacuous — a zero-width CI."""
    natural_means, initial_beliefs = [], []
    for seed in range(6):
        system = StrategicPopulation()
        system.reset(seed)
        natural_means.append(float(system.natural.mean()))
        initial_beliefs.append(system.line_estimate)
    assert np.std(natural_means) > 0.01, natural_means
    assert np.std(initial_beliefs) > 0.05, initial_beliefs


def test_trajectory_ignores_the_global_numpy_seed():
    """Behavioural companion to the repo-wide AST ban on module-global RNG use."""
    np.random.seed(11)
    a = [r["harm"] for r in _run(NAIVE_SHARP, 5, 60)[1]]
    np.random.seed(999)
    b = [r["harm"] for r in _run(NAIVE_SHARP, 5, 60)[1]]
    assert a == b


# --- the interface contract -------------------------------------------------------------------

def test_population_beliefs_and_true_behaviour_are_not_observable():
    """The hidden state whose inference IS the task must never leak into the policy's namespace."""
    system, _ = _run(NAIVE_SHARP, 0, 30)
    published = set(system.observe().vars)
    for hidden in ("line_estimate", "cliff_estimate", "adaptation_rate", "true_mean",
                   "evasion_gap", "bunched_share", "harm", "behaviour", "natural"):
        assert hidden not in published, f"'{hidden}' leaked into observe()"
    # …and the audit that IS published is a noisy subsample, not the truth itself
    assert system.observe().vars["audit_harm"] != system.metrics()["harm"]


def test_levers_are_reevaluated_and_clipped_every_step():
    system = StrategicPopulation({"shock_step": None})
    system.reset(0)
    report = _iface(system).apply(
        [ActionRequest("regent:0", "set_threshold", {"expr": "100.0 * t"}),
         ActionRequest("regent:0", "set_response_width", {"expr": "-5.0"})],
        system,
    )
    assert len(report.applied) == 2
    system.step()                                    # t=0 → 100*0 = 0, in range
    assert system.threshold == 0.0
    assert system.response_width == 0.0              # clipped up to the low bound
    system.step()                                    # t=1 → 100, clipped to the high bound
    assert system.threshold == system.behaviour_max
    system.step()                                    # re-evaluated every step, still clipped
    assert system.threshold == system.behaviour_max


def test_shock_raises_the_adaptation_rate_and_reset_restores_it():
    system = StrategicPopulation({"shock_step": 5, "shock_params": {"adaptation_rate": 0.10}})
    system.reset(0)
    for _ in range(5):  # steps 0..4 are pre-shock
        assert system.adaptation_rate == system.adaptation_rate0
        system.step()
    system.step()  # the shock fires at the top of step() when _t == 5
    assert system.adaptation_rate == 0.10
    system.reset(0)
    assert system.adaptation_rate == system.adaptation_rate0


def test_metrics_record_the_enacted_levers():
    _, trajectory = _run(SMOOTH_BAND, 0, 5)
    assert trajectory[-1]["threshold"] == 0.6
    assert trajectory[-1]["response_width"] == 1.2


def test_clone_continues_the_same_stochastic_stream():
    system, _ = _run(NAIVE_SHARP, 7, 20)
    twin = system.clone()
    original = [(system.step(), system.metrics()["harm"])[1] for _ in range(15)]
    cloned = [(twin.step(), twin.metrics()["harm"])[1] for _ in range(15)]
    assert original == cloned


# --- the objective's window handling ------------------------------------------------------------

def test_empty_post_shock_window_scores_worst_case_not_zero():
    """A run that ends before the break must not earn a flattering zero gap."""
    obj = StrategicComplianceLoss(post_shock_step=SHOCK_STEP)
    _, short = _run(NAIVE_SHARP, 0, 30)
    comps = obj.components(short)
    for key in ("post_loss", "post_harm", "post_evasion_gap", "post_bunched_share"):
        assert comps[key] == float("inf"), (key, comps[key])
    assert np.isfinite(comps["loss"])  # the full-horizon score is unaffected


def test_cumulative_quantities_are_differenced_across_the_window():
    """Reading a running total undifferenced turns the score into a clock. Two windows with the
    same per-step spending must score the same regardless of when they happen."""
    obj = StrategicComplianceLoss(lam_intervention=1.0, lam_evasion=1.0)
    early = [{"t": float(i), "harm": 0.0, "cum_intervention": float(i), "cum_evasion_cost": 0.0,
              "evasion_gap": 0.0, "bunched_share": 0.0} for i in range(10)]
    late = [{"t": float(i), "harm": 0.0, "cum_intervention": float(i), "cum_evasion_cost": 0.0,
             "evasion_gap": 0.0, "bunched_share": 0.0} for i in range(500, 510)]
    assert obj.evaluate(early) == obj.evaluate(late) == -9.0
    assert obj.components(late)["enforcement"] == 9.0


def test_describe_states_the_mandate_and_the_trade_off():
    text = StrategicComplianceLoss().describe()
    assert "MANDATE" in text
    for word in ("harm", "enforcement", "conceal", "report", "width"):
        assert word in text.lower()
