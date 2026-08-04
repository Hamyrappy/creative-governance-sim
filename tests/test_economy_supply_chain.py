"""
Tests for ``SupplyChainEconomy`` — the single-echelon inventory world with an order delay.

These pin the properties that make the world usable as an experiment rather than a demo: it cannot
diverge, its two conservation laws hold exactly (so the clipping is auditable and not a hole units
leak through), seeds are genuinely heterogeneous, both shocks bite, neither is observable, the
objective refuses to flatter an empty post-shock window, and the optimum is INTERIOR — a sensible
rule beats both corners, so there is something for a shock to move.
"""

from __future__ import annotations

import math

import pytest

from govsim.core.action import ActionRequest
from govsim.domains.economy.supply_chain import SupplyChainCost, SupplyChainEconomy
from govsim.domains.scalar.interface import Lever, ScalarLeverInterface

SHOCK_AT = 60
ORDER_CAP = 40.0


def _iface() -> ScalarLeverInterface:
    return ScalarLeverInterface([Lever("set_order", (0.0, ORDER_CAP), "order")])


def _run(expr: str, seed: int = 0, steps: int = 200, params: dict | None = None):
    """Install ``expr`` as the ordering rule and roll the world forward; returns (system, trajectory)."""
    system = SupplyChainEconomy(params or {})
    system.reset(seed)
    report = _iface().apply([ActionRequest("regent:0", "set_order", {"expr": expr})], system)
    assert report.applied, report.rejected
    trajectory = []
    for _ in range(steps):
        system.step()
        trajectory.append(system.metrics())
    return system, trajectory


def _base_stock(mult: float, gain: float = 0.4, boost: float = 1.0) -> str:
    """The natural rule for this world: replace what sold, then close the gap to a target buffer."""
    return f"{boost} * (recent_demand + {gain} * ({mult} * recent_demand + backlog - inventory))"


SENSIBLE = _base_stock(1.0)
DO_NOTHING = "0.0"
ALWAYS_MAX = str(ORDER_CAP)


# --- bounded dynamics -----------------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    DO_NOTHING,
    ALWAYS_MAX,
    SENSIBLE,
    _base_stock(20.0, gain=5.0, boost=10.0),  # a wildly over-geared rule: the clip must hold, not the rule
    "1000000000.0 * (backlog - inventory)",
    "-1000000000.0",
])
def test_state_stays_bounded_over_a_long_run(expr):
    system, trajectory = _run(expr, steps=600, params={"shock_step": SHOCK_AT,
                                                       "shock_params": {"lead_time": 6}})
    for row in trajectory:
        assert 0.0 <= row["inventory"] <= system.inventory_cap
        assert 0.0 <= row["backlog"] <= system.backlog_cap
        assert 0.0 <= row["order"] <= ORDER_CAP
        assert 0.0 <= row["demand"] <= system.demand_cap
        assert math.isfinite(row["on_order"])
    assert len(system.pipeline) == system.lead_time
    assert sum(system.pipeline) <= ORDER_CAP * system.lead_time_cap + 1e-9


def test_lever_expression_that_errors_at_runtime_leaves_state_bounded():
    """A rule that divides by zero keeps its previous value (the LeverSystem contract) — it must
    not leave the world in a state the caps no longer describe."""
    system, trajectory = _run("recent_demand / (inventory - inventory)", steps=100)
    assert all(0.0 <= r["order"] <= ORDER_CAP for r in trajectory)
    assert all(0.0 <= r["inventory"] <= system.inventory_cap for r in trajectory)


# --- the two conservation laws ----------------------------------------------------------------

@pytest.mark.parametrize("shock", [None, {"lead_time": 6}, {"lead_time": 1},
                                   {"fulfilment_efficacy": 0.35}])
def test_goods_are_conserved(shock):
    """Everything delivered is on the shelf, in transit, sold, or explicitly spoiled by the cap.

    A lead-time change re-lengths the pipeline mid-run; this is the test that says the re-lengthing
    moves goods rather than inventing or destroying them.
    """
    params = {"shock_step": SHOCK_AT, "shock_params": shock} if shock else {}
    system, _ = _run(SENSIBLE, steps=300, params=params)
    lhs = system.init_inventory + system.cum_delivered
    rhs = system.inventory + sum(system.pipeline) + system.cum_shipped + system.cum_spoiled
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("expr", [DO_NOTHING, ALWAYS_MAX, SENSIBLE])
def test_demand_is_conserved(expr):
    """Every unit demanded was shipped, is still owed, or was explicitly dropped by the backlog cap."""
    system, _ = _run(expr, steps=300)
    assert system.cum_demand == pytest.approx(
        system.cum_shipped + system.backlog + system.cum_lost, rel=1e-9, abs=1e-6
    )


def test_efficacy_collapse_delivers_less_than_it_bills():
    """The asymmetry the whole design rests on: cost accrues on the order, goods on the fill."""
    system, _ = _run(SENSIBLE, steps=200,
                     params={"shock_step": SHOCK_AT, "shock_params": {"fulfilment_efficacy": 0.4}})
    assert system.cum_delivered < system.cum_order_units


# --- per-seed heterogeneity -------------------------------------------------------------------

def test_seeds_produce_different_trajectories():
    runs = [_run(SENSIBLE, seed=s, steps=80)[1] for s in range(5)]
    finals = [r[-1]["inventory"] for r in runs]
    assert len(set(finals)) == len(finals)
    # Not merely a different noise stream around a shared mean: reset() draws the firm itself.
    systems = [SupplyChainEconomy() for _ in range(5)]
    for s, sysm in enumerate(systems):
        sysm.reset(s)
    assert len({sysm.demand_mu for sysm in systems}) == 5
    assert len({sysm.init_inventory for sysm in systems}) == 5


def test_same_seed_reproduces_exactly():
    a = _run(SENSIBLE, seed=3, steps=120)[1]
    b = _run(SENSIBLE, seed=3, steps=120)[1]
    assert a == b


def test_reset_restores_shocked_parameters():
    """Without this, a second 'fresh' run of the same object inherits the previous run's regime."""
    system = SupplyChainEconomy({"shock_step": 2, "shock_params": {"lead_time": 7,
                                                                   "fulfilment_efficacy": 0.3}})
    system.reset(0)
    for _ in range(5):
        system.step()
    assert (system.lead_time, system.fulfilment_efficacy) == (7, 0.3)
    system.reset(0)
    assert system.lead_time == system.lead_time0
    assert system.fulfilment_efficacy == system.fulfilment_efficacy0
    assert len(system.pipeline) == system.lead_time0
    assert system._levers == {}


def test_clone_continues_the_same_stochastic_stream():
    system, _ = _run(SENSIBLE, seed=1, steps=40)
    clone = system.clone()
    original = [system.step() or system.metrics() for _ in range(20)]
    copied = [clone.step() or clone.metrics() for _ in range(20)]
    assert original == copied


# --- the shocks actually change behaviour ------------------------------------------------------

def test_delay_shock_changes_the_trajectory():
    params = {"shock_step": SHOCK_AT, "shock_params": {"lead_time": 6}}
    _, plain = _run(SENSIBLE, seed=0, steps=200)
    _, shocked = _run(SENSIBLE, seed=0, steps=200, params=params)
    assert [r["inventory"] for r in plain[:SHOCK_AT]] == [r["inventory"] for r in shocked[:SHOCK_AT]]
    assert [r["inventory"] for r in plain[SHOCK_AT:]] != [r["inventory"] for r in shocked[SHOCK_AT:]]
    obj = SupplyChainCost(post_shock_step=SHOCK_AT)
    assert obj.components(shocked)["post_loss"] > obj.components(plain)["post_loss"]


def test_delay_shock_opens_an_arrival_gap():
    """The delay is felt as goods that stop turning up, which is the only trace it leaves.

    Goods already in transit still land on their old schedule, so the gap opens once that queue
    drains (here: lead_time 2 → 6 at step 60 ⇒ two more arrivals, then four empty periods, then the
    orders placed under the new regime).
    """
    system = SupplyChainEconomy({"shock_step": SHOCK_AT, "shock_params": {"lead_time": 6}})
    system.reset(0)
    _iface().apply([ActionRequest("regent:0", "set_order", {"expr": SENSIBLE})], system)
    arrivals = []
    for _ in range(SHOCK_AT + 8):
        system.step()
        arrivals.append(system.arrivals)
    assert all(a > 0 for a in arrivals[SHOCK_AT - 5:SHOCK_AT + 2])
    assert arrivals[SHOCK_AT + 2:SHOCK_AT + 6] == [0.0, 0.0, 0.0, 0.0]
    assert arrivals[SHOCK_AT + 6] > 0
    assert system.lead_time == 6


def test_efficacy_shock_changes_the_trajectory_and_costs_more():
    params = {"shock_step": SHOCK_AT, "shock_params": {"fulfilment_efficacy": 0.35}}
    _, plain = _run(SENSIBLE, seed=0, steps=200)
    _, shocked = _run(SENSIBLE, seed=0, steps=200, params=params)
    assert [r["inventory"] for r in plain[SHOCK_AT:]] != [r["inventory"] for r in shocked[SHOCK_AT:]]
    obj = SupplyChainCost(post_shock_step=SHOCK_AT)
    assert obj.components(shocked)["post_loss"] > obj.components(plain)["post_loss"]


@pytest.mark.parametrize("shock", [{"lead_time": 6}, {"fulfilment_efficacy": 0.35}])
def test_a_re_tuned_rule_beats_the_frozen_one_after_the_shock(shock):
    """There is adaptation headroom: the frozen rule is not still optimal in the new regime.

    Without this the world is unusable for the experiment regardless of the agent — it would report
    a null by construction, and the null would be a property of the world, not a finding.
    """
    params = {"shock_step": SHOCK_AT, "shock_params": shock}
    obj = SupplyChainCost(post_shock_step=SHOCK_AT)
    frozen = adapted = 0.0
    for seed in range(6):
        frozen += obj.components(_run(SENSIBLE, seed, 200, params)[1])["post_loss"]
        adapted += obj.components(_run(_base_stock(2.0, gain=0.15, boost=1.6), seed, 200, params)[1])["post_loss"]
    assert adapted < frozen


# --- the shocked parameters are not observable -------------------------------------------------

def test_observe_hides_efficacy_and_lead_time():
    system = SupplyChainEconomy({"shock_step": 1, "shock_params": {"fulfilment_efficacy": 0.2,
                                                                   "lead_time": 9}})
    system.reset(0)
    for _ in range(5):
        system.step()
    published = system.observe("regent:0").vars
    assert set(published) == {"inventory", "backlog", "last_demand", "recent_demand", "order", "t"}
    for value in published.values():
        assert value != pytest.approx(system.fulfilment_efficacy)
    assert not any("effica" in k or "lead" in k or "pipeline" in k for k in published)
    # metrics feed the trajectory and can reach a harness prompt, so the parameters stay out there too.
    assert not any("effica" in k or "lead" in k for k in system.metrics())


def test_observe_publishes_the_enacted_order_and_metrics_record_it():
    """The governance question is what the institution DID; a record of stock levels cannot answer it."""
    system, trajectory = _run("7.5", steps=10)
    assert system.observe().vars["order"] == pytest.approx(7.5)
    assert all(r["order"] == pytest.approx(7.5) for r in trajectory)


def test_action_space_offers_only_what_is_observable():
    system = SupplyChainEconomy()
    space = _iface().action_space(system, "regent:0")
    assert space.verb_names() == ["set_order"]
    assert "fulfilment_efficacy" not in space.context_vars
    assert "lead_time" not in space.context_vars
    assert system.lever_attrs == {"order": (0.0, ORDER_CAP)}


# --- the objective ------------------------------------------------------------------------------

def test_empty_post_shock_window_scores_worst_case_not_zero():
    """An empty window must not be free, or 'end the run early' becomes the optimal policy."""
    obj = SupplyChainCost(post_shock_step=500)
    _, trajectory = _run(SENSIBLE, steps=50)
    comps = obj.components(trajectory)
    assert comps["post_loss"] == float("inf")
    assert comps["post_short_units"] == float("inf")
    assert math.isfinite(comps["loss"])


def test_costs_are_differenced_across_the_window_not_read_as_a_running_total():
    """The bug that once turned a realized-performance signal into a clock: the cumulative keys run
    from t=0, so an undifferenced read scores a window by when it happened."""
    obj = SupplyChainCost(post_shock_step=100)
    _, trajectory = _run(SENSIBLE, steps=200)
    comps = obj.components(trajectory)
    assert comps["post_hold_units"] < comps["hold_units"]
    assert comps["post_order_units"] == pytest.approx(
        trajectory[-1]["cum_order_units"] - trajectory[99]["cum_order_units"]
    )
    # Two windows with identical per-step behaviour must score identically wherever they sit.
    early = obj.evaluate(trajectory[10:60])
    late = obj.evaluate(trajectory[110:160])
    assert early == pytest.approx(late, rel=0.5)


def test_evaluate_is_the_negated_loss_and_higher_is_better():
    obj = SupplyChainCost()
    _, good = _run(SENSIBLE, seed=0, steps=150)
    _, bad = _run(DO_NOTHING, seed=0, steps=150)
    assert obj.evaluate(good) == pytest.approx(-obj.components(good)["loss"])
    assert obj.evaluate(good) > obj.evaluate(bad)
    assert obj.evaluate([]) == 0.0


def test_describe_states_the_mandate_including_the_hidden_lag():
    text = SupplyChainCost().describe().lower()
    assert "mandate" in text
    for term in ("inventory", "backlog", "order"):
        assert term in text
    assert "lag" in text or "not arrive" in text


# --- there is a control problem, and its optimum is interior --------------------------------------

def test_a_sensible_policy_beats_doing_nothing_and_beats_ordering_the_maximum():
    obj = SupplyChainCost()
    sensible = sum(obj.components(_run(SENSIBLE, s, 200)[1])["loss"] for s in range(6))
    nothing = sum(obj.components(_run(DO_NOTHING, s, 200)[1])["loss"] for s in range(6))
    maximal = sum(obj.components(_run(ALWAYS_MAX, s, 200)[1])["loss"] for s in range(6))
    assert sensible < nothing
    assert sensible < maximal


def test_the_optimal_buffer_is_interior():
    """Neither corner wins the sweep. If one did, no shock could move the optimum and the world
    would report a null by construction (grand-plan requirement 5)."""
    obj = SupplyChainCost()
    multipliers = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    losses = [
        sum(obj.components(_run(_base_stock(m), s, 200)[1])["loss"] for s in range(6))
        for m in multipliers
    ]
    best = losses.index(min(losses))
    assert 0 < best < len(multipliers) - 1, dict(zip(multipliers, losses))
