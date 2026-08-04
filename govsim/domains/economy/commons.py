"""
CommonsEconomy — a renewable common-pool resource under an authority that can restrict access
two ways, only one of which depends on anybody obeying it.

Why this world exists. The platform's calibration results say that a shock to the *governed state*
is absorbed by any feedback rule (the observable moves, the rule fires harder, a policy nobody
touched stays near-optimal), and that only a shock to the *efficacy of an instrument* leaves
adaptation headroom. A fishery is the textbook case: the quota is a legal object, and a legal
object is worth exactly the compliance behind it. When compliance collapses the quota keeps costing
what it always cost — the fleet is still told it may not fish, the paperwork is still filed, the
political price is still paid — and it stops doing anything at all. No threshold on the quota can
recover that, because at zero compliance the quota→harvest channel has gain zero. The correct
answer is a different instrument: close water instead of writing rules. That substitution is the
adaptation headroom here, and it is the Lucas critique in a fishery.

The two instruments, and why the pair is the point:
  - ``quota`` — the legally allowed catch. Cheap, precise, and worth nothing without enforcement.
  - ``reserve`` — the fraction of the stock placed in a physically inaccessible refuge (deep or
    remote grounds). Blunt and expensive, but it binds through geography rather than obedience,
    so ``enforcement_efficacy`` cannot touch it.
Pre-shock the quota dominates on price; post-shock it is a pure deadweight cost and the reserve is
the only instrument left. A rule frozen before the break keeps buying the dead one.

BOUNDED BY CONSTRUCTION. The stock is hard-clipped into ``[0, capacity]`` every step, harvest is
capped at the accessible stock (so it can never mine biomass that is not there), and logistic
growth vanishes at both ends of that interval. No configuration and no policy — including an
adversarial lever expression, which is clipped into its own range before it is ever read — can make
the state leave the interval or diverge. Recruitment noise is multiplicative on growth, so it
cannot push the stock outside the clip either. A collapsed fishery is therefore a valid state the
run continues to inhabit rather than a termination: terminating on collapse would empty the
post-shock window and hand a flattering score to whichever arm broke the world fastest.

PARTLY IRREVERSIBLE. With ``allee > 0`` per-capita growth is depressed at low stock
(``B/(B + allee·K)``), so a depleted stock recovers on a much slower clock than it collapsed on,
and below ``extinction_floor`` the remnant is written off to exactly zero, which is absorbing.
The floor is there because a logistic stock decays geometrically without ever reaching zero in
floating point, and a fishery holding 1e-119 of its carrying capacity is not a fishery that can
come back — it is a rounding error the model would otherwise let regrow. Over-harvesting is not a
mistake the next step undoes, which is what makes governing on a stale rule expensive rather than
merely suboptimal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem


class CommonsEconomy(LeverSystem):
    """A logistic fishery governed by a quota (enforcement-dependent) and a reserve (not).

    Dynamics, per step, on stock ``B`` with carrying capacity ``K``:

    ``accessible = B·(1 - reserve)``
    ``desired    = catchability·accessible``                      (open-access effort, fixed)
    ``harvest    = min(desired, quota) + (1 - c)·max(0, desired - quota)``, capped at ``accessible``
    ``growth     = r·B·(1 - B/K)·B/(B + allee·K)``                (× lognormal recruitment noise)
    ``B'         = clip(B + growth - harvest, 0, K)``

    where ``c = enforcement_efficacy`` is the share of the fleet that respects the quota. At
    ``c = 1`` the quota is a hard cap; at ``c = 0`` the fleet fishes as if no quota existed while
    the authority is billed for one.

    The tragedy is in the default numbers, not in the story: ``catchability`` (0.50) exceeds the
    intrinsic growth rate ``r`` (0.40), so unregulated effort drives the stock to zero. Doing
    nothing is not a neutral baseline here, it is the losing arm.

    The optimum is interior in both levers. Restricting the quota costs
    ``quota_cost·(1 - quota/quota_max)`` per step — the tighter the rule the more it costs to
    impose — so a total ban buys zero yield at maximum political price, while an unrestricted quota
    buys a collapsed stock. Neither corner survives; the authority has to find the harvest rate the
    stock can carry and pay for exactly that much restriction.

    Cost accrues on the POLICY SET, never on its effect: both terms above read only lever values.
    A quota nobody obeys still closes the season on paper, and that asymmetry is the whole reason an
    efficacy collapse is expensive to ignore rather than merely disappointing.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        # -- stock dynamics -------------------------------------------------------------------
        self.growth_rate_init: float = float(p.get("growth_rate", 0.40))
        self.capacity_init: float = float(p.get("capacity", 1.0))
        self.initial_stock_frac: float = float(p.get("initial_stock_frac", 0.80))
        self.allee_init: float = float(p.get("allee", 0.05))
        self.recruit_sigma_init: float = float(p.get("recruit_sigma", 0.05))
        self.extinction_floor_init: float = float(p.get("extinction_floor", 1e-3))  # share of K
        # Per-seed heterogeneity. Without it every seed replays one trajectory, the paired
        # shared-seed design pairs a number with itself, and the bootstrap CI is a zero-width
        # interval wearing the costume of a result. Drawn once per reset from this system's own
        # Generator, never from the module-global RNG (a CI AST test enforces that).
        self.growth_rate_sigma: float = float(p.get("growth_rate_sigma", 0.15))
        self.capacity_sigma: float = float(p.get("capacity_sigma", 0.12))
        self.initial_stock_sigma: float = float(p.get("initial_stock_sigma", 0.15))
        # -- the fleet ------------------------------------------------------------------------
        self.catchability_init: float = float(p.get("catchability", 0.50))
        # What share of the fleet actually respects the quota. 1.0 = a hard cap; a shock drives it
        # toward 0 = a quota that exists only on paper.
        self.enforcement_efficacy_init: float = float(p.get("enforcement_efficacy", 1.0))
        # -- instruments ----------------------------------------------------------------------
        self.quota_max_init: float = float(p.get("quota_max", 0.50))
        self.reserve_max_init: float = float(p.get("reserve_max", 0.80))
        self.quota_cost_init: float = float(p.get("quota_cost", 0.04))
        self.reserve_cost_init: float = float(p.get("reserve_cost", 0.08))
        self.price: float = float(p.get("price", 1.0))
        self.discount: float = float(p.get("discount", 0.99))
        # -- the unseen structural break -------------------------------------------------------
        # Same mechanism as SIRSystem: named parameters are overwritten at ``shock_step``. Leaving
        # ``shock_params`` empty gives the stationary world (the golden-master / pre-shock arm).
        self.shock_step: int = int(p.get("shock_step", 100))
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {}))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"quota": (0.0, self.quota_max), "reserve": (0.0, self.reserve_max)}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.growth_rate: float = self.growth_rate_init
        if self.growth_rate_sigma > 0:
            self.growth_rate *= float(np.exp(self.rng.normal(0.0, self.growth_rate_sigma)))
        self.capacity: float = self.capacity_init
        if self.capacity_sigma > 0:
            self.capacity *= float(np.exp(self.rng.normal(0.0, self.capacity_sigma)))
        frac = self.initial_stock_frac
        if self.initial_stock_sigma > 0:
            frac = float(np.clip(frac * np.exp(self.rng.normal(0.0, self.initial_stock_sigma)), 0.05, 1.0))
        self.stock: float = frac * self.capacity
        # Restore EVERY parameter a shock may overwrite, so a reused object starts pristine rather
        # than inheriting the previous run's regime.
        self.allee: float = self.allee_init
        self.recruit_sigma: float = self.recruit_sigma_init
        self.extinction_floor: float = self.extinction_floor_init
        self.catchability: float = self.catchability_init
        self.enforcement_efficacy: float = self.enforcement_efficacy_init
        self.quota_max: float = self.quota_max_init
        self.reserve_max: float = self.reserve_max_init
        self.quota_cost: float = self.quota_cost_init
        self.reserve_cost: float = self.reserve_cost_init
        # Absent any policy the commons is unregulated: the quota is set at its permissive ceiling
        # and no water is closed. "Do nothing" therefore means open access, which is the correct
        # null for a tragedy-of-the-commons world and not a free pass.
        self.quota: float = self.quota_max
        self.reserve: float = 0.0
        self.catch: float = 0.0
        self.illegal_catch: float = 0.0
        self.cum_welfare: float = 0.0
        self.cum_cost: float = 0.0
        self._t: int = 0
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()
        if self._t == self.shock_step:
            for name, value in self.shock_params.items():
                setattr(self, name, float(value))
        stock = self.stock
        accessible = stock * (1.0 - self.reserve)
        desired = self.catchability * accessible
        compliance = min(1.0, max(0.0, self.enforcement_efficacy))
        over_quota = max(0.0, desired - self.quota) * (1.0 - compliance)
        harvest = min(min(desired, self.quota) + over_quota, accessible)
        growth = self.growth_rate * stock * (1.0 - stock / self.capacity)
        if self.allee > 0.0:
            growth *= stock / (stock + self.allee * self.capacity)
        if self.recruit_sigma > 0.0:
            # Drawn unconditionally, before it is known whether growth is zero: a stochastic stream
            # whose consumption depends on the state would desynchronize two arms sharing a seed,
            # and the paired design would then be comparing different worlds.
            shock = float(self.rng.normal(0.0, self.recruit_sigma))
            growth *= float(np.exp(shock - 0.5 * self.recruit_sigma ** 2))  # median-corrected: E[factor] = 1
        self.stock = float(np.clip(stock + growth - harvest, 0.0, self.capacity))
        if self.stock < self.extinction_floor * self.capacity:
            self.stock = 0.0  # a remnant this thin is extinction, not a stock with a slow recovery
        self.catch = float(harvest)
        self.illegal_catch = float(over_quota)
        weight = self.discount ** self._t
        self.cum_welfare += weight * self.price * harvest
        # Billed on the levers alone — no term below reads the stock, the catch, or compliance.
        self.cum_cost += weight * (
            self.quota_cost * (1.0 - self.quota / self.quota_max) + self.reserve_cost * self.reserve
        )
        self._t += 1
        # A collapsed stock is a state, not an ending: terminating here would truncate the
        # post-shock window and reward the arm that destroyed the fishery earliest.
        return StepInfo(terminated=not np.isfinite(self.stock), truncated=False, info={})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # ``enforcement_efficacy`` is deliberately absent. The authority runs stock surveys and
            # sees landings; it does not get told how much of its own law is being obeyed. Noticing
            # that catch has broken away from the quota, and that the stock is falling faster than
            # the rules permit, IS the task — publishing the parameter would delete the problem.
            vars={
                "stock": self.stock,
                "catch": self.catch,
                "capacity": self.capacity,
                "quota": self.quota,
                "reserve": self.reserve,
                "t": float(self._t),
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # The enacted levers are part of the record: the governance question is what the authority
        # DID, and a trajectory of stock alone cannot answer it. ``illegal_catch`` is logged for
        # post-hoc analysis only — it is not in ``observe``, so no policy can key on it.
        return {
            "stock": self.stock,
            "stock_frac": self.stock / self.capacity if self.capacity > 0 else 0.0,
            "capacity": self.capacity,
            "catch": self.catch,
            "illegal_catch": self.illegal_catch,
            "quota": self.quota,
            "reserve": self.reserve,
            "cum_welfare": self.cum_welfare,
            "cum_cost": self.cum_cost,
            "t": float(self._t),
        }


class CommonsWelfare(Objective):
    """Discounted catch value minus the cost of the restrictions ordered, minus a depletion charge.

    Negated in the usual direction: ``evaluate`` returns higher-is-better, ``components["loss"]``
    is the same quantity as a loss.

    **λ prices restriction against fish, and it decides whether this world can host an experiment
    at all.** Set it too low and the optimum is a corner ("ban everything, it's free"); too high and
    the optimum is the other corner ("never restrict"). A corner optimum is invariant to the
    regime — no efficacy collapse can move it — so the measured adaptation headroom would be 1.0x
    by construction. The defaults put the optimum strictly inside both lever ranges.

    **The depletion charge is a terminal condition, not a taste.** A finite horizon prices the stock
    at the end of the run at zero, so the unbeatable endgame is to liquidate the fishery on the last
    steps. Charging for each step the stock spends below ``target_frac`` of carrying capacity
    restores the standing value of the resource the authority is supposed to hand on. It is
    deliberately NOT discounted: a guard rail that decays stops guarding exactly where the incentive
    to strip-mine is strongest.

    ``post_shock_step`` mirrors ``EpidemicLoss``: the ``post_*`` components cover only the rows
    at-or-after the break, so the pre-shock stretch — where a pre-shock-optimal rule is optimal by
    definition — is not averaged into the headline comparison and cannot dilute it away.
    """

    def __init__(self, lam: float = 1.0, depletion_weight: float = 0.02, target_frac: float = 0.40,
                 welfare_key: str = "cum_welfare", cost_key: str = "cum_cost",
                 stock_key: str = "stock_frac", post_shock_step: int | None = None,
                 step_key: str = "t") -> None:
        self.lam = lam
        self.depletion_weight = depletion_weight
        self.target_frac = target_frac
        self.welfare_key = welfare_key
        self.cost_key = cost_key
        self.stock_key = stock_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Maximize the discounted value of the catch, MINUS "
            f"{self.lam} times the cost of the restrictions you order, MINUS "
            f"{self.depletion_weight} for every step the stock sits below {self.target_frac:.0%} of "
            "carrying capacity.\n"
            "  - the fleet fishes harder than the stock can regrow, so with no restriction the "
            "fishery collapses and the catch goes to zero for good;\n"
            "  - restriction is billed on the POLICY YOU SET, not on its effect: you pay for how "
            "far below the ceiling you set the quota, and for how much water you close, whether or "
            "not anyone complies with either;\n"
            "  - so tightening the quota is worth it only while it protects more fish than it "
            "costs, and closing water is worth it only when the quota cannot do the job.\n"
            "Higher total is better. Both corners lose: a fishery banned outright yields nothing at "
            "full political cost, and a fishery left open yields nothing once the stock is gone."
        )

    def _welfare_cost_depletion(self, rows: Trajectory) -> tuple[float, float, float]:
        """(catch value, restriction cost, depletion charge) over ``rows``.

        Welfare and cost are running totals from t=0, so both are DIFFERENCED across the window.
        Reading either off the last row would score a window by when it happened rather than by
        what the policy did in it — that mistake shipped here once and turned the runner's
        realized-performance signal into a clock that reported "worse than last time" forever.
        """
        if not rows:
            return 0.0, 0.0, 0.0
        welfare = rows[-1].get(self.welfare_key, 0.0) - rows[0].get(self.welfare_key, 0.0)
        cost = rows[-1].get(self.cost_key, 0.0) - rows[0].get(self.cost_key, 0.0)
        depletion = sum(
            max(0.0, 1.0 - row.get(self.stock_key, 0.0) / self.target_frac) for row in rows
        ) if self.target_frac > 0 else 0.0
        return welfare, cost, depletion

    def _post_shock_rows(self, trajectory: Trajectory) -> Trajectory:
        s = self.post_shock_step
        if s is None:
            return list(trajectory)
        out = [row for row in trajectory if row.get(self.step_key, -1) >= s]
        if out:
            return out
        return trajectory[s:] if s < len(trajectory) else []

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        welfare, cost, depletion = self._welfare_cost_depletion(trajectory)
        return welfare - self.lam * cost - self.depletion_weight * depletion

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        welfare, cost, depletion = self._welfare_cost_depletion(trajectory)
        stocks = [row.get(self.stock_key, 0.0) for row in trajectory]
        catches = [row.get("catch", 0.0) for row in trajectory]
        base = {
            "welfare": welfare,
            "cost": cost,
            "depletion": depletion,
            "loss": self.lam * cost + self.depletion_weight * depletion - welfare,
            "total_catch": sum(catches),
            "final_stock_frac": stocks[-1] if stocks else 0.0,
            "min_stock_frac": min(stocks) if stocks else 0.0,
        }
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and not post_rows:
            # The run ended before the break, so there is no post-shock evidence. Scoring an empty
            # window as 0.0 would make "end the run early" the optimal post-shock policy and the
            # reference would be calibrated on early termination rather than on governance.
            # NOTE the absence of an `and trajectory` guard. A run that produced NO rows at all is
            # the most extreme version of exactly this failure — zero post-shock evidence — and it
            # used to fall through to the ordinary branch and score post_loss = 0.0, which beat
            # every genuine policy. "Terminate before the first step" must not be the winning move.
            # ``evaluate`` and the full-horizon ``loss`` still return 0.0 on an empty trajectory:
            # only the post-shock window, which is the pre-registered H1 comparison metric, is
            # worst-cased here.
            inf = float("inf")
            base.update({"post_welfare": -inf, "post_cost": inf, "post_depletion": inf,
                         "post_loss": inf, "post_min_stock_frac": 0.0})
            return base
        post_welfare, post_cost, post_depletion = self._welfare_cost_depletion(post_rows)
        post_stocks = [row.get(self.stock_key, 0.0) for row in post_rows]
        base.update({
            "post_welfare": post_welfare,
            "post_cost": post_cost,
            "post_depletion": post_depletion,
            "post_loss": self.lam * post_cost + self.depletion_weight * post_depletion - post_welfare,
            "post_min_stock_frac": min(post_stocks) if post_stocks else 0.0,
        })
        return base
