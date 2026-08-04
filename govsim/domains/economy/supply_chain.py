"""
SupplyChainEconomy — a single-echelon inventory system with an ORDER DELAY (the beer-game
structure), plus the objective that prices it.

**Why this world exists.** The calibration results say only INSTRUMENT_EFFICACY shocks leave
adaptation headroom: a feedback rule keyed on the governed state absorbs a state-dynamics shock by
itself, because the observable moves and the rule fires differently. Inventory control adds a second
family that has never been measured here — DELAY. An ordering rule is tuned, implicitly, to a lead
time. Double the lead time and the rule keeps ordering as if goods arrived on the old schedule; the
correction it makes today lands two periods later than it believes, so it over-corrects, and the
over-correction arrives just as the earlier one does. That is the bullwhip, and it is not a
threshold error a feedback rule can absorb — the gain itself is wrong. Whether that leaves headroom
is an open question this world is built to answer, next to an efficacy shock in the same dynamics as
a within-world control.

Both shocks are expressed through ``shock_params`` at ``shock_step`` (the SIRSystem mechanism,
unchanged):
  - ``lead_time`` doubles                  → ShockKind.DELAY
  - ``fulfilment_efficacy`` collapses      → ShockKind.INSTRUMENT_EFFICACY (an order is only
    partly filled, while the authority is billed for the whole of it)

**Neither is observable.** ``observe`` publishes inventory, backlog, realized demand and the order
last placed — never the lead time, never the fill rate, never the pipeline. Both breaks are visible
only as a pattern in arrivals, which is the same evidence a real planner has. Publishing either
parameter would delete the inference task that is the whole point.

**Bounded dynamics (no arm can diverge).** Every state variable is hard-clipped every step:
``inventory ∈ [0, inventory_cap]``, ``backlog ∈ [0, backlog_cap]``, ``demand ∈ [0, demand_cap]``,
the order lever ∈ ``[0, order_cap]``, and the pipeline holds at most ``lead_time`` slots with
``lead_time ∈ [1, lead_time_cap]``, so goods in transit are bounded by ``order_cap · lead_time_cap``.
There is no unbounded accumulator in the state; the cumulative counters are diagnostics that the
objective differences across its window. A comparison between two arms is therefore always a
comparison between two finite trajectories.

**Two conservation laws hold exactly** (both are pinned by tests), which is what makes the clipping
auditable rather than a place for units to quietly vanish:

    init_inventory + cum_delivered == inventory + on_order + cum_shipped + cum_spoiled
    cum_demand                     == cum_shipped + backlog  + cum_lost

``cum_spoiled`` and ``cum_lost`` are exactly the mass the two caps removed; if the caps never bind
they stay 0 and the identities are the plain textbook ones.

**Cost accrues on the ORDER PLACED, not on the goods delivered.** An order that the supplier fills
at 35% still costs what it cost to place. That asymmetry is what makes an efficacy collapse
expensive to ignore, and it is why the correct post-shock response is a genuine re-optimization
(buy some of the shortfall back, accept a worse service level for the rest) rather than a different
threshold on the same rule.

The system carries no prices at all — it accumulates *physical* quantities (unit-periods of stock
held, unit-periods of backlog, units ordered) and ``SupplyChainCost`` owns the prices. Prices decide
whether the optimum is interior, and an interior optimum is a precondition for measuring anything
here, so they belong to the mandate the regent is told and the reference is optimized against.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem

#: Shock parameters that are integers. A scenario is data (``dict[str, float]``), so a delay shock
#: arrives as ``4.0``; writing that straight onto ``lead_time`` would make the pipeline length a
#: float and break the list arithmetic silently.
_INT_SHOCK_PARAMS = frozenset({"lead_time"})


class SupplyChainEconomy(LeverSystem):
    """Single-echelon inventory control with a lead time: order today, receive in ``lead_time``.

    The authority sets one lever, ``order`` — the quantity to place this period, naturally written
    as an expression over what it can see (``inventory``, ``backlog``, ``recent_demand``). Orders
    join a pipeline of ``lead_time`` slots and arrive at the front of it.

    Demand is stochastic and mean-reverting (AR(1) around a per-seed mean), so it is neither a
    constant a rule can be hard-coded to nor a random walk that would make the problem unsolvable.

    See the module docstring for the bounds, the two conservation laws, and why efficacy and lead
    time are kept out of ``observe``.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        # -- demand process (AR(1) in levels, mean-reverting) --
        self.demand_mu_init: float = float(p.get("demand_mu", 10.0))
        self.demand_rho: float = float(p.get("demand_rho", 0.6))
        self.demand_sigma: float = float(p.get("demand_sigma", 2.0))
        self.demand_cap_mult: float = float(p.get("demand_cap_mult", 4.0))
        self.recent_demand_alpha: float = float(p.get("recent_demand_alpha", 0.3))
        # -- the instrument and its delay (BOTH shockable, NEITHER observable) --
        self.lead_time0: int = int(p.get("lead_time", 2))
        self.lead_time_cap: int = int(p.get("lead_time_cap", 16))
        self.fulfilment_efficacy0: float = float(p.get("fulfilment_efficacy", 1.0))
        # -- hard bounds; see the class docstring's boundedness claim --
        self.order_cap: float = float(p.get("order_cap", 40.0))
        self.inventory_cap: float = float(p.get("inventory_cap", 300.0))
        self.backlog_cap: float = float(p.get("backlog_cap", 200.0))
        # -- per-seed heterogeneity. Without a genuine draw here every seed runs the same
        # trajectory up to the demand noise, and a paired shared-seed design has almost nothing to
        # pair over: the two arms would differ by a common shift, not by how each handled a
        # different firm. Drawn once in reset() from the system's own Generator. --
        self.demand_mu_sigma: float = float(p.get("demand_mu_sigma", 0.15))
        self.init_inventory_mult: float = float(p.get("init_inventory_mult", 2.0))
        self.init_inventory_sigma: float = float(p.get("init_inventory_sigma", 0.25))
        # -- the unseen structural break (CubicSystem/SIRSystem mechanism, verbatim) --
        ss = p.get("shock_step")
        self.shock_step: int | None = None if ss is None else int(ss)
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {}))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"order": (0.0, self.order_cap)}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        # Restore EVERY parameter a shock may overwrite from its snapshot, or a second "fresh" run
        # of the same object silently inherits the previous run's regime.
        self.lead_time: int = self.lead_time0
        self.fulfilment_efficacy: float = self.fulfilment_efficacy0
        self.demand_mu: float = self.demand_mu_init
        if self.demand_mu_sigma > 0:
            self.demand_mu *= float(np.exp(self.rng.normal(0.0, self.demand_mu_sigma)))
        self.demand_cap: float = self.demand_cap_mult * self.demand_mu
        self.demand: float = self.demand_mu
        self.recent_demand: float = self.demand_mu
        inv0 = self.init_inventory_mult * self.demand_mu
        if self.init_inventory_sigma > 0:
            inv0 *= float(np.exp(self.rng.normal(0.0, self.init_inventory_sigma)))
        self.init_inventory: float = float(min(inv0, self.inventory_cap))
        self.inventory: float = self.init_inventory
        self.backlog: float = 0.0
        self.pipeline: list[float] = [0.0] * self.lead_time
        self.order: float = 0.0
        self.arrivals: float = 0.0
        self.shipped: float = 0.0
        # Physical accumulators (no prices — SupplyChainCost owns those). The first three are what
        # the objective differences across its window; the last three close the conservation laws.
        self.cum_hold_units: float = 0.0
        self.cum_short_units: float = 0.0
        self.cum_order_units: float = 0.0
        self.cum_demand: float = 0.0
        self.cum_shipped: float = 0.0
        self.cum_delivered: float = 0.0
        self.cum_spoiled: float = 0.0
        self.cum_lost: float = 0.0
        self._t: int = 0
        self._levers.clear()

    def _apply_shock(self) -> None:
        for name, value in self.shock_params.items():
            if not hasattr(self, name):
                # A misspelled parameter name would otherwise ``setattr`` a brand-new attribute that
                # nothing reads: the scenario reports itself as shocked, the dynamics never change,
                # and the arm returns a null that is indistinguishable from a finding. Refuse.
                raise KeyError(
                    f"SupplyChainEconomy: shock_params names '{name}', which is not a parameter of "
                    f"this system, so the shock would be silently discarded. Known shockable "
                    f"parameters include 'lead_time' and 'fulfilment_efficacy'."
                )
            setattr(self, name, int(round(value)) if name in _INT_SHOCK_PARAMS else float(value))
        # A scenario is free-form data; clamp the delay so a typo cannot allocate an arbitrarily
        # long pipeline (the boundedness claim has to survive a hostile config, not just a careful one).
        self.lead_time = int(min(max(1, self.lead_time), self.lead_time_cap))

    def _receive(self) -> float:
        """Pop this period's arrivals and re-length the pipeline to the CURRENT lead time.

        Re-lengthing is where a delay shock becomes physical, and it must not create or destroy
        goods. A lengthened lead time pads with empty slots, so nothing arrives for a while and
        orders already in transit still land on their old schedule — the arrival gap is the only
        trace the shock leaves. A shortened one folds the orphaned tail into the last slot rather
        than dropping it, which is what keeps the goods identity exact.
        """
        arrivals = self.pipeline.pop(0) if self.pipeline else 0.0
        want = self.lead_time - 1
        if len(self.pipeline) < want:
            self.pipeline.extend([0.0] * (want - len(self.pipeline)))
        elif len(self.pipeline) > want:
            orphaned = sum(self.pipeline[want:])
            del self.pipeline[want:]
            if want > 0:
                self.pipeline[want - 1] += orphaned
            else:
                arrivals += orphaned
        return arrivals

    def step(self) -> StepInfo:
        self._reeval_levers()  # the eval-cadence contract: the lever expression is re-read EVERY step
        if self.shock_step is not None and self._t == self.shock_step:
            self._apply_shock()
        placed = float(min(max(0.0, self.order), self.order_cap))

        self.arrivals = self._receive()
        self.inventory += self.arrivals
        spoiled = max(0.0, self.inventory - self.inventory_cap)
        self.inventory -= spoiled

        eps = float(self.rng.normal(0.0, self.demand_sigma)) if self.demand_sigma > 0 else 0.0
        raw = self.demand_mu + self.demand_rho * (self.demand - self.demand_mu) + eps
        self.demand = float(min(max(0.0, raw), self.demand_cap))
        self.recent_demand += self.recent_demand_alpha * (self.demand - self.recent_demand)

        # Backlog is a queue, not a lost sale: today's shipments serve the outstanding orders too,
        # so a stockout keeps costing until it is worked off. Otherwise "let it go short" is free
        # after one period and the stockout price stops disciplining anything.
        required = self.demand + self.backlog
        self.shipped = min(self.inventory, required)
        self.inventory -= self.shipped
        outstanding = required - self.shipped
        lost = max(0.0, outstanding - self.backlog_cap)
        self.backlog = outstanding - lost

        # The lever is the ORDER; efficacy is how much of it the supplier actually ships.
        delivered = placed * self.fulfilment_efficacy
        self.pipeline.append(delivered)

        self.cum_hold_units += self.inventory
        self.cum_short_units += self.backlog
        # Billed on the policy set, not on what arrived: an order filled at 35% costs what it cost
        # to place. That is what makes an efficacy collapse expensive to ignore.
        self.cum_order_units += placed
        self.cum_demand += self.demand
        self.cum_shipped += self.shipped
        self.cum_delivered += delivered
        self.cum_spoiled += spoiled
        self.cum_lost += lost
        self._t += 1
        # Nothing terminates: demand is endemic and every state variable is clipped, so there is no
        # absorbing state a passive policy could reach to make its post-shock window empty.
        return StepInfo(terminated=False, truncated=False,
                        info={"arrivals": self.arrivals, "shipped": self.shipped})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # What a planner has: stock on hand, orders owed, what sold, what it last ordered.
            # NOT lead_time, NOT fulfilment_efficacy, NOT the pipeline — inferring that the
            # instrument broke (or slowed) from the trace is the task being measured.
            vars={
                "inventory": self.inventory,
                "backlog": self.backlog,
                "last_demand": self.demand,
                "recent_demand": self.recent_demand,
                "order": self.order,
                "t": float(self._t),
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # ``order`` is in the record because the governance question is what the institution DID,
        # and a trajectory of stock levels alone cannot answer it. The two shocked parameters stay
        # out: they are unobservable by design, and a metric is one leak away from a prompt.
        return {
            "inventory": self.inventory,
            "backlog": self.backlog,
            "demand": self.demand,
            "order": self.order,
            "arrivals": self.arrivals,
            "shipped": self.shipped,
            "on_order": sum(self.pipeline),
            "cum_hold_units": self.cum_hold_units,
            "cum_short_units": self.cum_short_units,
            "cum_order_units": self.cum_order_units,
            "cum_lost": self.cum_lost,
            "t": float(self._t),
        }


class SupplyChainCost(Objective):
    """Minimize holding + stockout + ordering cost (negated → higher is better).

    **The three prices are the substantive knobs.** They decide where the optimum sits, and a corner
    optimum cannot be moved by any shock — a world calibrated at "always order the cap" or "never
    order" has zero adaptation headroom by construction, whatever the agent. So they have to make
    restraint rational in both directions: ``holding_cost`` punishes the always-max corner (stock
    piles up against the cap and is billed every period it sits there), ``stockout_cost`` punishes
    the do-nothing corner (backlog is a queue and keeps costing until it is worked off), and
    ``order_cost`` is what turns an efficacy collapse into a real trade-off rather than a free
    "just order more" — buying the shortfall back costs ``order_cost / fulfilment_efficacy`` per
    unit actually delivered, so at some collapse severity the correct answer becomes a worse
    service level, which no re-threshold of the old rule expresses.

    ``post_shock_step`` mirrors ``EpidemicLoss``: ``components`` emits ``post_*`` over the rows
    at-or-after the break, so the pre-shock half — where a pre-shock-optimal rule is near-optimal by
    definition — is not averaged into the headline comparison.
    """

    def __init__(self, holding_cost: float = 1.0, stockout_cost: float = 6.0,
                 order_cost: float = 0.5, hold_key: str = "cum_hold_units",
                 short_key: str = "cum_short_units", order_key: str = "cum_order_units",
                 post_shock_step: int | None = None, step_key: str = "t") -> None:
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.hold_key = hold_key
        self.short_key = short_key
        self.order_key = order_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Run the warehouse at least total cost over the horizon. Every period you "
            "pay:\n"
            f"  - {self.holding_cost} per unit of stock sitting in inventory at the end of the "
            "period;\n"
            f"  - {self.stockout_cost} per unit of demand you still owe (backlog is a queue: an "
            "unfilled unit keeps costing every period until you ship it);\n"
            f"  - {self.order_cost} per unit YOU ORDER — billed on the order you place, not on what "
            "the supplier actually delivers.\n"
            "Orders do not arrive the period you place them; there is a delivery lag, and you are "
            "not told what it is. Lower total is better. Both failure modes are real: a warehouse "
            "stacked to the roof and a warehouse that is always out of stock are both bad."
        )

    def _cost_parts(self, rows: Trajectory) -> tuple[float, float, float]:
        """(holding, stockout, ordering) unit-totals over ``rows``, DIFFERENCED across the window.

        The keys are running totals from t=0, so reading the last row undifferenced would score a
        post-shock window by *when it happened* rather than by what the policy did in it — the
        metric becomes a clock that only ever goes up. That exact bug has bitten this project once
        already (``EpidemicLoss.evaluate``); it is not a hypothetical.
        """
        if not rows:
            return 0.0, 0.0, 0.0
        first, last = rows[0], rows[-1]
        return (
            last.get(self.hold_key, 0.0) - first.get(self.hold_key, 0.0),
            last.get(self.short_key, 0.0) - first.get(self.short_key, 0.0),
            last.get(self.order_key, 0.0) - first.get(self.order_key, 0.0),
        )

    def _loss(self, rows: Trajectory) -> float:
        hold, short, ordered = self._cost_parts(rows)
        return self.holding_cost * hold + self.stockout_cost * short + self.order_cost * ordered

    def _post_shock_rows(self, trajectory: Trajectory) -> Trajectory:
        s = self.post_shock_step
        if s is None:
            return list(trajectory)
        out = [row for row in trajectory if row.get(self.step_key, -1) >= s]
        if out:
            return out
        return trajectory[s:] if s < len(trajectory) else []

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        """Negated loss over ``trajectory``, which may be a WINDOW rather than a whole run.

        The Runner calls this on the interval since the regent's last decision to build the
        realized-performance signal a harness shows the model, so the differencing in
        ``_cost_parts`` is load-bearing here and not only in ``components``: undifferenced, every
        interval would report "worse than last time" forever and a regent reading that signal would
        rationally abandon a policy that was working.
        """
        return -self._loss(trajectory)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        hold, short, ordered = self._cost_parts(trajectory)
        base = {
            "hold_units": hold,
            "short_units": short,
            "order_units": ordered,
            "loss": self.holding_cost * hold + self.stockout_cost * short + self.order_cost * ordered,
            "mean_backlog": (sum(r.get("backlog", 0.0) for r in trajectory) / len(trajectory)) if trajectory else 0.0,
            "peak_backlog": max((r.get("backlog", 0.0) for r in trajectory), default=0.0),
            "mean_inventory": (sum(r.get("inventory", 0.0) for r in trajectory) / len(trajectory)) if trajectory else 0.0,
        }
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and len(post_rows) < 2:
            # The run ended at or before the break, so there is no post-shock evidence. Scoring that
            # 0.0 would make "get the run terminated early" the winning post-shock policy and a
            # reference calibrated on it would be measuring termination, not governance.
            #
            # The threshold is TWO rows, not one, and that is the whole point: every component here
            # is a DIFFERENCE across the window, so a lone row is differenced against itself and
            # scores a flawless 0.0 — strictly better than any window containing real post-shock
            # time. A horizon of exactly ``post_shock_step`` produces precisely that, which leaves
            # the exploit the empty-window guard was written to close open one row further along.
            inf = float("inf")
            base.update({"post_hold_units": inf, "post_short_units": inf,
                         "post_order_units": inf, "post_loss": inf, "post_peak_backlog": inf})
            return base
        p_hold, p_short, p_ordered = self._cost_parts(post_rows)
        base.update({
            "post_hold_units": p_hold,
            "post_short_units": p_short,
            "post_order_units": p_ordered,
            "post_loss": self.holding_cost * p_hold + self.stockout_cost * p_short + self.order_cost * p_ordered,
            "post_peak_backlog": max((r.get("backlog", 0.0) for r in post_rows), default=0.0),
        })
        return base
