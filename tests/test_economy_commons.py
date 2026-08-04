"""
Tests for the ``CommonsEconomy`` world (a renewable common-pool resource under a quota that needs
enforcement and a reserve that does not).

The list is not generic hygiene. Each case pins one property this world has to have before any
result measured on it is worth reading:

  - the state cannot leave ``[0, capacity]`` (a comparison between two divergences is not a result);
  - seeds are genuinely different worlds (else the paired statistics pair a number with itself);
  - ``enforcement_efficacy`` is invisible to the regent (else the inference task is deleted);
  - the efficacy collapse actually changes the trajectory, and the adapted instrument recovers what
    the stale one loses (else there is no adaptation headroom to measure);
  - the optimum is interior in both levers (else no shock can move it);
  - an empty post-shock window scores worst-case, and cumulative totals are DIFFERENCED across a
    window rather than read off its last row (that one shipped as a real bug in this codebase).
"""

from __future__ import annotations

import numpy as np

from govsim.core.action import ActionRequest
from govsim.domains.economy.commons import CommonsEconomy, CommonsWelfare
from govsim.domains.scalar import Lever, ScalarLeverInterface

SHOCK_STEP = 100
HORIZON = 200
NEVER = {"shock_step": 10_000}
COLLAPSE = {"shock_step": SHOCK_STEP, "shock_params": {"enforcement_efficacy": 0.15}}


def _iface() -> ScalarLeverInterface:
    return ScalarLeverInterface([
        Lever("set_quota", (0.0, 0.5), "quota", "Allowed catch per step."),
        Lever("set_reserve", (0.0, 0.8), "reserve", "Share of the stock closed to fishing."),
    ])


def _run(cfg: dict, seed: int, quota: str | None = None, reserve: str | None = None,
         horizon: int = HORIZON) -> list[dict[str, float]]:
    """Run one governed trajectory through the real action interface (not by poking attributes)."""
    system = CommonsEconomy(dict(cfg, seed=seed))
    system.reset(seed)
    reqs = []
    if quota is not None:
        reqs.append(ActionRequest("regent:0", "set_quota", {"expr": quota}))
    if reserve is not None:
        reqs.append(ActionRequest("regent:0", "set_reserve", {"expr": reserve}))
    if reqs:
        report = _iface().apply(reqs, system)
        assert not report.rejected, report.rejected
    rows = []
    for _ in range(horizon):
        system.step()
        rows.append(system.metrics())
    return rows


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


# --- bounded dynamics + the claimed invariants -----------------------------------------------

def test_stock_stays_in_zero_to_capacity_over_a_long_run():
    """The docstring's headline claim: no configuration and no policy can make the state leave the
    interval. Includes lever expressions that ask for absurd values, which the range clip absorbs."""
    for seed in range(4):
        for quota, reserve in (
            (None, None),                    # open access — the collapse arm
            ("0.0", None),                   # total ban — the other corner
            ("1000000.0 * stock", "-1000000.0"),  # adversarial: clipped into range before it is read
            ("0.18 * stock", "0.3"),
        ):
            rows = _run(COLLAPSE, seed, quota, reserve, horizon=2000)
            capacity = rows[0]["capacity"]
            for row in rows:
                assert np.isfinite(row["stock"])
                assert -1e-12 <= row["stock"] <= capacity + 1e-12
                assert 0.0 <= row["stock_frac"] <= 1.0 + 1e-12


def test_harvest_never_exceeds_the_accessible_stock():
    """The conservation-flavoured invariant: catch is taken out of biomass that exists, and only out
    of the part the reserve leaves open. Checked with a constant reserve so the accessible share is
    known for the step that is about to run."""
    system = CommonsEconomy(dict(COLLAPSE, seed=3))
    system.reset(3)
    report = _iface().apply([ActionRequest("regent:0", "set_reserve", {"expr": "0.5"})], system)
    assert not report.rejected
    system.step()  # the first step is what installs reserve = 0.5; from here it is constant
    for _ in range(400):
        before = system.stock
        system.step()
        assert 0.0 <= system.catch <= 0.5 * before + 1e-12


def test_a_collapsed_stock_is_absorbing_and_the_run_continues():
    """'Partly irreversible' is a property, not a slogan: open access drives the stock to zero, it
    stays there, and the world keeps stepping (terminating on collapse would empty the post-shock
    window and flatter whichever arm broke the fishery fastest)."""
    system = CommonsEconomy({"seed": 1, **NEVER})
    system.reset(1)
    for _ in range(400):
        info = system.step()
        assert not info.terminated
    assert system.stock == 0.0
    for _ in range(50):
        system.step()
        assert system.stock == 0.0
        assert system.catch == 0.0


# --- per-seed heterogeneity -------------------------------------------------------------------

def test_seeds_produce_genuinely_different_worlds():
    params = [(s.growth_rate, s.capacity, s.stock) for s in
              (CommonsEconomy({"seed": seed}) for seed in range(6))]
    assert len({round(g, 9) for g, _, _ in params}) == 6
    assert len({round(k, 9) for _, k, _ in params}) == 6
    assert len({round(b, 9) for _, _, b in params}) == 6
    tails = [_run(NEVER, seed, "0.18 * stock", horizon=60)[-1]["stock"] for seed in range(6)]
    assert len({round(x, 9) for x in tails}) == 6


def test_same_seed_replays_exactly():
    a = _run(COLLAPSE, 7, "0.18 * stock", "0.2", horizon=120)
    b = _run(COLLAPSE, 7, "0.18 * stock", "0.2", horizon=120)
    assert [row["stock"] for row in a] == [row["stock"] for row in b]


# --- the shock: unobservable, and it bites -----------------------------------------------------

def test_efficacy_is_absent_from_the_observation():
    system = CommonsEconomy({"seed": 0})
    keys = set(system.observe().vars)
    assert keys == {"stock", "catch", "capacity", "quota", "reserve", "t"}
    assert not any("effica" in k or "compl" in k or "illegal" in k for k in keys)
    # Two worlds that differ ONLY in efficacy look identical to the regent at decision time.
    broken = CommonsEconomy({"seed": 0, "enforcement_efficacy": 0.1})
    assert broken.observe().vars == system.observe().vars


def test_the_efficacy_collapse_changes_behaviour():
    """Same seed, same rule: after the break the fleet fishes past the quota, the stock falls, and
    the authority is still paying for the quota it is no longer getting."""
    for seed in range(3):
        quiet = _run(NEVER, seed, "0.18 * stock")
        broken = _run(COLLAPSE, seed, "0.18 * stock")
        assert [r["stock"] for r in quiet[:SHOCK_STEP]] == [r["stock"] for r in broken[:SHOCK_STEP]]
        assert broken[-1]["stock"] < quiet[-1]["stock"] - 1e-6
        assert broken[-1]["illegal_catch"] >= 0.0 and quiet[-1]["illegal_catch"] == 0.0
        # The bill for the (now useless) quota keeps arriving at the same rate.
        quiet_bill = quiet[-1]["cum_cost"] - quiet[SHOCK_STEP]["cum_cost"]
        broken_bill = broken[-1]["cum_cost"] - broken[SHOCK_STEP]["cum_cost"]
        assert broken_bill > 0.0 and broken_bill >= quiet_bill * 0.9


def test_reset_restores_every_shocked_parameter():
    system = CommonsEconomy(dict(COLLAPSE, seed=2))
    for _ in range(SHOCK_STEP + 1):
        system.step()
    assert system.enforcement_efficacy == 0.15
    system.reset(2)
    assert system.enforcement_efficacy == 1.0
    assert system.quota == system.quota_max and system.reserve == 0.0
    assert system.cum_welfare == 0.0 and system.cum_cost == 0.0 and system.time == 0


# --- the control problem: an interior optimum, and headroom after the break --------------------

def test_a_sensible_quota_beats_doing_nothing():
    """If the null wins there is no control problem to study. Open access collapses the fishery;
    a quota sized near the stock's own regrowth does not."""
    objective = CommonsWelfare()
    nothing = _mean(objective.evaluate(_run(NEVER, seed)) for seed in range(8))
    governed = _mean(objective.evaluate(_run(NEVER, seed, "0.18 * stock")) for seed in range(8))
    assert governed > nothing + 1.0
    assert _mean(_run(NEVER, seed)[-1]["stock_frac"] for seed in range(8)) < 0.02
    assert _mean(_run(NEVER, seed, "0.18 * stock")[-1]["stock_frac"] for seed in range(8)) > 0.2


def test_both_corners_lose_to_an_interior_policy():
    """No shock can move an optimum that sits at a corner, so this world is only usable if
    restraint is priced. Banning the fishery outright and leaving it open both lose."""
    objective = CommonsWelfare()
    scores = {
        expr: _mean(objective.evaluate(_run(NEVER, seed, expr)) for seed in range(6))
        for expr in ("0.0", "0.08 * stock", "0.18 * stock", "0.5", None)
    }
    assert scores["0.18 * stock"] > scores["0.0"]        # not a total ban
    assert scores["0.18 * stock"] > scores["0.5"]        # not an unrestricted quota
    assert scores["0.18 * stock"] > scores["0.08 * stock"]  # not "as tight as possible" either
    assert scores["0.18 * stock"] > scores[None]
    # …and the same interiority in the second lever, which is what adaptation has to reach for.
    reserves = {
        rho: _mean(objective.evaluate(_run(NEVER, seed, None, rho)) for seed in range(6))
        for rho in ("0.0", "0.65", "0.8")
    }
    assert reserves["0.65"] > reserves["0.0"] and reserves["0.65"] > reserves["0.8"]


def test_adapting_the_instrument_recovers_what_the_stale_rule_loses():
    """The whole point of the world. After compliance collapses, no setting of the quota helps —
    the correct move is to stop paying for it and close water instead. Measured on the post-shock
    window only, since a pre-shock-optimal rule is optimal before the break by definition."""
    post = CommonsWelfare(post_shock_step=SHOCK_STEP)
    stale = _mean(post.components(_run(COLLAPSE, seed, "0.18 * stock"))["post_loss"] for seed in range(6))
    tighter = _mean(post.components(_run(COLLAPSE, seed, "0.08 * stock"))["post_loss"] for seed in range(6))
    adapted = _mean(post.components(_run(COLLAPSE, seed, None, "0.65"))["post_loss"] for seed in range(6))
    assert tighter > adapted, "tightening the dead instrument must not rescue the run"
    assert adapted < stale - 1.0, "the surviving instrument must buy back a real share of the loss"
    # …while pre-shock the reserve is the DOMINATED choice, so a rule calibrated before the break
    # would not already be holding the answer.
    objective = CommonsWelfare()
    quota_arm = _mean(objective.evaluate(_run(NEVER, seed, "0.18 * stock")) for seed in range(6))
    reserve_arm = _mean(objective.evaluate(_run(NEVER, seed, None, "0.65")) for seed in range(6))
    assert quota_arm > reserve_arm


# --- the objective ------------------------------------------------------------------------------

def test_empty_post_shock_window_scores_worst_case():
    """A run that ends before the break has no post-shock evidence. Scoring the empty window as 0.0
    would make 'end the run early' the optimal post-shock policy."""
    rows = _run(NEVER, 0, "0.18 * stock", horizon=40)
    comps = CommonsWelfare(post_shock_step=1000).components(rows)
    assert comps["post_loss"] == float("inf")
    assert comps["post_welfare"] == float("-inf")
    assert np.isfinite(comps["loss"])  # the full-horizon score is unaffected


def test_cumulative_totals_are_differenced_across_the_window():
    """The bug this codebase actually shipped: reading a running total off the last row scores a
    window by WHEN it happened, turning the realized-performance signal into a clock."""
    rows = _run(COLLAPSE, 0, "0.18 * stock")
    objective = CommonsWelfare(post_shock_step=SHOCK_STEP)
    comps = objective.components(rows)
    window = [r for r in rows if r["t"] >= SHOCK_STEP]
    assert np.isclose(comps["post_cost"], window[-1]["cum_cost"] - window[0]["cum_cost"])
    assert comps["post_cost"] < rows[-1]["cum_cost"]
    assert np.isclose(comps["post_welfare"], window[-1]["cum_welfare"] - window[0]["cum_welfare"])
    # evaluate() must difference too — the runner calls it on the interval since the last decision.
    assert np.isclose(objective.evaluate(window),
                      comps["post_welfare"] - comps["post_cost"] - 0.02 * comps["post_depletion"])


def test_objective_is_empty_trajectory_safe_and_describes_its_mandate():
    objective = CommonsWelfare()
    assert objective.evaluate([]) == 0.0
    assert objective.components([])["loss"] == 0.0
    text = objective.describe()
    assert "MANDATE" in text and "quota" in text and "cost" in text
    assert "effica" not in text  # the mandate must not leak the hidden parameter


# --- interface + metrics contract ---------------------------------------------------------------

def test_levers_are_re_evaluated_every_step_and_clipped():
    system = CommonsEconomy({"seed": 5, **NEVER})
    system.reset(5)
    report = _iface().apply([ActionRequest("regent:0", "set_quota", {"expr": "10.0 * stock"})], system)
    assert not report.rejected
    seen = []
    for _ in range(30):
        system.step()
        seen.append(system.quota)
    assert max(seen) <= 0.5 + 1e-12  # clipped into the lever range
    assert len({round(q, 9) for q in seen}) > 1  # re-evaluated per step, not once at apply time


def test_metrics_record_what_the_authority_did():
    rows = _run(NEVER, 0, "0.18 * stock", "0.25", horizon=5)
    for key in ("quota", "reserve", "stock", "catch", "cum_welfare", "cum_cost", "t"):
        assert key in rows[-1]
    assert np.isclose(rows[-1]["reserve"], 0.25)


def test_clone_continues_the_same_stochastic_stream():
    system = CommonsEconomy(dict(COLLAPSE, seed=11))
    system.reset(11)
    _iface().apply([ActionRequest("regent:0", "set_quota", {"expr": "0.18 * stock"})], system)
    for _ in range(30):
        system.step()
    twin = system.clone()
    original = [system.step() or system.stock for _ in range(20)]
    rollout = [twin.step() or twin.stock for _ in range(20)]
    assert original == rollout
