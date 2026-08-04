"""
``MonetaryEconomy`` — a stabilization economy governed by a single policy rate, plus its
dual-mandate objective.

This is the Lucas critique's home ground: the critique was written about exactly this authority,
saying that the estimated relations a policy rule is tuned on are not invariant to the policy
regime, so a rule extrapolated across a structural break is being applied to a world it no longer
describes. Here the break is made explicit and unobservable — the *transmission* of the policy rate
into demand collapses, while the rate keeps costing what it always cost.

**The break is an instrument-efficacy shock, not a state shock, and the distinction is the whole
design.** A feedback rule ("raise the rate when inflation runs") absorbs a demand or cost-push
shock by triggering harder; the governed state moves, the rule reads it, and a frozen rule stays
near-optimal. A transmission collapse is not absorbed, and worse, a feedback rule responds to it in
exactly the wrong direction: with the rate no longer reaching demand, the boom runs, the rule reads
the boom and *raises the rate further*, paying more for an instrument that has stopped working.
The correct response — stop leaning, because the lever is disconnected — is not expressible as a
different threshold on the same observables. That gap is the adaptation headroom.

Dynamics (a 3-equation New Keynesian toy, in percentage points):

    y_{t+1}  = ρ_y·y_t + drift − κ·transmission·(i_t − r* − π_t) + ε_y
    π_{t+1}  = π* + ρ_π·(π_t − π*) + φ·y_t + ε_π

The inflation equation is the usual ``ρ_π·π + φ·y`` written in deviations from the target π*, i.e.
with expectations anchored at target: with a closed output gap inflation rests *at* the mandate
rather than at zero, so the two halves of the mandate are jointly attainable and the world does not
secretly force a permanent hot economy to hold target.

``drift`` is a persistent exogenous demand impulse and is the reason the instrument has a job worth
paying for. Without it the optimal policy is "sit at the neutral rate forever", which is invariant
to transmission — a world where no efficacy shock can move the optimum, and therefore a world with
zero headroom by construction. It sits *outside* the transmission multiplier on purpose: the boom
does not go away when the ability to fight it does.

**Bounded dynamics — two independent guarantees.** (1) Open-loop contraction: at a constant rate the
state matrix is ``[[ρ_y, κ·T], [φ, ρ_π]]``, and the per-seed draws are clipped to bands that keep
``(1−ρ_y)(1−ρ_π) > κ·T·φ`` for every seed and every ``transmission ≤ 1``, so its spectral radius is
below 1 and a bounded policy input yields a bounded state. (2) Hard clipping: the lever is clipped
into ``rate_range`` every step and both states are clipped into a wide finite band after every
update, so even a perverse feedback rule (or an unlucky tail of the Gaussian noise, the only
unbounded input in the model) cannot make this system diverge. No arm can blow up; the comparison
is always between two finite numbers.

Efficacy is deliberately absent from ``observe()``: the regent sees the rate it set and what the
economy did, never the parameter connecting them. Inferring "the instrument broke" from the trace
is the task.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem


class MonetaryEconomy(LeverSystem):
    """Output gap + inflation, governed by one policy rate, breakable at the transmission channel.

    The lever is ``policy_rate`` (a nominal rate in percentage points, clipped into ``rate_range``,
    whose floor is a zero lower bound by default). The regent installs an expression over the
    observables and it is re-evaluated every step, as everywhere else in this platform.

    ``shock_params`` overwrites any named plant parameter at ``shock_step`` (the mechanism
    ``CubicSystem``/``SIRSystem`` use). The intended break is ``{"transmission": 0.1}``: a liquidity
    trap / broken monetary transmission, with ``rate_cost`` untouched.

    Severity and price both have to be calibrated, and neither is a taste question. Sweeping the
    best *constant* rate over 24 seeds at the defaults (horizon 200, break at 100) gives an interior
    optimum in every cell and this adaptation headroom (frozen pre-break optimum ÷ post-break
    optimum, on the post-break window)::

        λ \\ transmission     0.30   0.20   0.10   0.05
        0.10                 1.00   1.02   1.09   1.16
        0.15                 1.03   1.09   1.22   1.32
        0.25                 1.13   1.26   1.47   1.61

    A mild collapse is worth nothing: at ``transmission=0.30`` the optimal rate barely moves and a
    frozen rule is already right. Headroom needs BOTH a severe break and an instrument priced
    highly enough that abandoning it is the correct response.

    Cost accrues on the rate the authority *sets*, never on the effect it achieves: a restrictive
    stance is politically and financially expensive whether or not it reaches anybody's borrowing
    decision. That asymmetry is what makes an efficacy collapse expensive to ignore.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        # -- the authority's own numbers: it knows its target and its estimate of r*, so both are
        # -- observable. They are still snapshotted: a shock could in principle move r*.
        self.inflation_target0: float = float(p.get("inflation_target", 2.0))
        self.r_star0: float = float(p.get("r_star", 2.0))
        # -- structural parameters. Every one of them is snapshotted because ``shock_params`` may
        # -- overwrite any of them, and a shock that leaked into the next "fresh" run of the same
        # -- object would silently contaminate the pre-shock arm.
        self.rho_y_init: float = float(p.get("rho_y", 0.65))
        self.rho_pi0: float = float(p.get("rho_pi", 0.55))
        self.phi_init: float = float(p.get("phi", 0.20))
        self.kappa0: float = float(p.get("kappa", 0.35))
        self.demand_drift0: float = float(p.get("demand_drift", 1.0))
        self.transmission0: float = float(p.get("transmission", 1.0))
        self.sigma_gap0: float = float(p.get("sigma_gap", 0.30))
        self.sigma_inflation0: float = float(p.get("sigma_inflation", 0.20))
        self.rate_cost0: float = float(p.get("rate_cost", 1.0))
        lo, hi = p.get("rate_range", (0.0, 12.0))
        self.rate_range0: tuple[float, float] = (float(lo), float(hi))
        # -- per-seed heterogeneity. Without it every seed replays one trajectory, the paired
        # -- shared-seed design pairs identical numbers, and a bootstrap CI of width zero gets
        # -- reported as a finding. Drawn in reset() from the system's own Generator.
        self.rho_y_sigma: float = float(p.get("rho_y_sigma", 0.06))
        self.phi_sigma: float = float(p.get("phi_sigma", 0.15))
        self.noise_scale_sigma: float = float(p.get("noise_scale_sigma", 0.30))
        # The draws are clipped into bands narrow enough to keep the open-loop system a contraction
        # for EVERY seed: (1-rho_y)(1-rho_pi) > kappa*transmission*phi at the corners. Heterogeneity
        # that occasionally hands one seed an explosive economy would be measuring the draw.
        self.rho_y_bounds: tuple[float, float] = tuple(p.get("rho_y_bounds", (0.40, 0.74)))  # type: ignore[assignment]
        self.phi_bounds: tuple[float, float] = tuple(p.get("phi_bounds", (0.10, 0.30)))  # type: ignore[assignment]
        # -- saturation band (the boundedness guarantee of last resort) --
        # Snapshotted like every other plant parameter, and for a sharper reason than the rest:
        # these two ARE the boundedness guarantee. Left un-snapshotted they were both the live
        # value and the pristine one, so ``shock_params={"gap_band": 3.0}`` permanently shrank the
        # band on the object and every later ``reset()`` handed the next run a quietly different
        # world — including the pre-shock arm it is compared against.
        self.gap_band0: float = float(p.get("gap_band", 25.0))
        self.inflation_band0: float = float(p.get("inflation_band", 25.0))
        # -- the unseen structural break --
        ss = p.get("shock_step")
        self.shock_step: int | None = None if ss is None else int(ss)
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {}))
        self.initial_gap: float = float(p.get("initial_gap", 0.0))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"policy_rate": self.rate_range}

    @property
    def neutral_rate(self) -> float:
        """The nominal rate that leaves the real rate at ``r*`` when inflation is at target."""
        return self.r_star + self.inflation_target

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.inflation_target: float = self.inflation_target0
        self.r_star: float = self.r_star0
        self.rho_pi: float = self.rho_pi0
        self.kappa: float = self.kappa0
        self.demand_drift: float = self.demand_drift0
        self.transmission: float = self.transmission0
        self.rate_cost: float = self.rate_cost0
        self.rate_range: tuple[float, float] = self.rate_range0
        self.gap_band: float = self.gap_band0
        self.inflation_band: float = self.inflation_band0
        self.rho_y: float = float(np.clip(
            self.rho_y_init * np.exp(self.rng.normal(0.0, self.rho_y_sigma)), *self.rho_y_bounds))
        self.phi: float = float(np.clip(
            self.phi_init * np.exp(self.rng.normal(0.0, self.phi_sigma)), *self.phi_bounds))
        self.noise_scale: float = float(np.exp(self.rng.normal(0.0, self.noise_scale_sigma)))
        self.sigma_gap: float = self.sigma_gap0 * self.noise_scale
        self.sigma_inflation: float = self.sigma_inflation0 * self.noise_scale
        self.output_gap: float = self.initial_gap
        self.inflation: float = self.inflation_target
        self.prev_output_gap: float = self.output_gap
        self.prev_inflation: float = self.inflation
        # A passive authority starts at the neutral rate, so "do nothing" is a defensible policy
        # (hold the textbook stance) rather than an accidentally maximal stimulus at rate zero.
        self.policy_rate: float = self.neutral_rate
        self.cum_cost: float = 0.0
        self._t: int = 0
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()
        if self.shock_step is not None and self._t == self.shock_step:
            for name, value in self.shock_params.items():
                setattr(self, name, float(value))
        stance = self.policy_rate - self.r_star - self.inflation  # the real-rate gap, in points
        eps_gap = float(self.rng.normal(0.0, self.sigma_gap))
        eps_inflation = float(self.rng.normal(0.0, self.sigma_inflation))
        next_gap = (
            self.rho_y * self.output_gap
            + self.demand_drift
            - self.kappa * self.transmission * stance
            + eps_gap
        )
        next_inflation = (
            self.inflation_target
            + self.rho_pi * (self.inflation - self.inflation_target)
            + self.phi * self.output_gap
            + eps_inflation
        )
        self.prev_output_gap = self.output_gap
        self.prev_inflation = self.inflation
        self.output_gap = float(np.clip(next_gap, -self.gap_band, self.gap_band))
        self.inflation = float(np.clip(next_inflation, -self.inflation_band, self.inflation_band))
        # Billed on the stance the authority set, at full price, whatever reached the economy.
        self.cum_cost += self.rate_cost * self.policy_rate ** 2
        self._t += 1
        finite = bool(np.isfinite(self.output_gap) and np.isfinite(self.inflation))
        # Unreachable while the clip band holds; kept so a mis-parameterized plant fails loudly
        # instead of feeding NaNs into a comparison.
        return StepInfo(terminated=not finite, truncated=False, info={"stance": stance})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # ``transmission`` (and κ, and the demand drift) are NOT here. The regent sees the data
            # a central bank reads off its own dashboard — the gap, inflation, the rate it set, its
            # target, its estimate of r* — and must infer from the trace that the lever stopped
            # working. Publishing the efficacy would delete the problem this world exists to pose.
            vars={
                "output_gap": self.output_gap,
                "inflation": self.inflation,
                "prev_output_gap": self.prev_output_gap,
                "prev_inflation": self.prev_inflation,
                "policy_rate": self.policy_rate,
                "inflation_target": self.inflation_target,
                "r_star": self.r_star,
                "t": float(self._t),
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # The enacted rate is part of the record, not only the economy it produced: the governance
        # question is what the institution DID. Efficacy stays out — unobservable by design.
        return {
            "output_gap": self.output_gap,
            "inflation": self.inflation,
            "inflation_gap": self.inflation - self.inflation_target,
            "policy_rate": self.policy_rate,
            "inflation_target": self.inflation_target,
            "cum_cost": self.cum_cost,
            "t": float(self._t),
        }


class DualMandateLoss(Objective):
    """``Σ[(π−π*)² + w·y²] + λ·Σ(rate cost)``, negated so higher is better.

    **λ is the substantive knob, not a nuisance parameter** (the same lesson ``EpidemicLoss``
    carries). It prices a point of policy rate against a point of inflation miss, so it decides
    whether the optimal authority is interior or a corner. Priced too cheaply, the answer is "hold
    the rate as high as the range allows, the boom is what hurts" — a corner that no transmission
    collapse can move, and therefore a world with no adaptation headroom to measure. Priced high
    enough that restraint is rational, the optimal rate is interior on both sides of the break and
    the break *moves* it: a weaker instrument buys less per point of rate, so the authority that
    knows the channel is broken stops paying for it and accepts a hotter economy, while a frozen
    rule keeps buying at full price.

    ``post_shock_step`` mirrors ``EpidemicLoss``/``StabilizationLoss``: the ``post_*`` components
    cover only rows at-or-after the break, so the pre-shock half — where a pre-shock-optimal rule is
    by definition near-optimal — is not averaged into the headline comparison.
    """

    def __init__(self, lam: float = 0.15, gap_weight: float = 0.5, target: float = 2.0,
                 inflation_key: str = "inflation", gap_key: str = "output_gap",
                 cost_key: str = "cum_cost", target_key: str = "inflation_target",
                 rate_key: str = "policy_rate", post_shock_step: int | None = None,
                 step_key: str = "t") -> None:
        self.lam = lam
        self.gap_weight = gap_weight
        self.target = target
        self.inflation_key = inflation_key
        self.gap_key = gap_key
        self.cost_key = cost_key
        self.target_key = target_key
        self.rate_key = rate_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Minimize, summed over the whole horizon, the squared deviation of "
            f"inflation from its {self.target}% target PLUS {self.gap_weight} times the squared "
            f"output gap, PLUS {self.lam} times the total cost of the policy rate you set.\n"
            "  - the rate's cost accrues on the LEVEL YOU SET, every step, and is charged whether "
            "or not the rate reaches the economy: a restrictive stance is politically and "
            "financially expensive even when it changes nobody's behaviour;\n"
            "  - so a point of policy rate is worth setting only while it buys back more than "
            f"{self.lam} times its own cost in inflation misses and lost output;\n"
            "  - the economy is running with persistent excess demand, so leaving the rate at its "
            "neutral level is not a neutral choice.\n"
            "Lower total is better. Both halves matter: an authority that holds inflation exactly "
            "at target with a punitive rate forever is not better than one that lets inflation run "
            "a little and spends nothing."
        )

    def _mandate_cost(self, rows: Trajectory) -> tuple[float, float]:
        """``(mandate burden, rate cost)`` over ``rows``.

        The cost is DIFFERENCED across the window rather than read off the last row: ``cum_cost`` is
        a running total from t=0, so on a post-shock window (or on the per-decision interval the
        Runner scores) the undifferenced value bills the current policy for every rate set before
        the window opened, and the score becomes a clock — it falls with elapsed time no matter what
        the authority does. That exact bug shipped once in this project's epidemic objective.
        """
        if not rows:
            return 0.0, 0.0
        for row in rows:
            if self.inflation_key not in row:  # loud-fail a mis-wired objective/system pairing
                raise KeyError(                # rather than score a missing state as a perfect 0.0
                    f"DualMandateLoss: inflation_key '{self.inflation_key}' absent from a trajectory "
                    f"row (keys: {sorted(row)}); the objective is wired to the wrong system."
                )
        burden = sum(
            (row[self.inflation_key] - row.get(self.target_key, self.target)) ** 2
            + self.gap_weight * row.get(self.gap_key, 0.0) ** 2
            for row in rows
        )
        cost = rows[-1].get(self.cost_key, 0.0) - rows[0].get(self.cost_key, 0.0)
        return burden, cost

    def _post_shock_rows(self, trajectory: Trajectory) -> Trajectory:
        s = self.post_shock_step
        if s is None:
            return list(trajectory)
        out = [row for row in trajectory if row.get(self.step_key, -1) >= s]
        if out:
            return out
        return trajectory[s:] if s < len(trajectory) else []

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        burden, cost = self._mandate_cost(trajectory)
        return -(burden + self.lam * cost)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        burden, cost = self._mandate_cost(trajectory)
        rates = [row.get(self.rate_key, 0.0) for row in trajectory]
        gaps = [row.get(self.gap_key, 0.0) for row in trajectory]
        misses = [row.get(self.inflation_key, 0.0) - row.get(self.target_key, self.target)
                  for row in trajectory]
        base = {
            "mandate_burden": burden,
            "rate_cost": cost,
            "loss": burden + self.lam * cost,
            "mean_rate": (sum(rates) / len(rates)) if rates else 0.0,
            "mean_abs_gap": (sum(abs(g) for g in gaps) / len(gaps)) if gaps else 0.0,
            "mean_abs_inflation_miss": (sum(abs(m) for m in misses) / len(misses)) if misses else 0.0,
            "final_rate": rates[-1] if rates else 0.0,
        }
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and not post_rows:
            # The run ended BEFORE the break, so there is no post-shock data. An empty window must
            # not score 0.0: that would make "end the run early" the optimal post-shock policy and a
            # reference calibrated against it would be measuring termination, not governance.
            # Note the guard does NOT also require a non-empty ``trajectory``. It used to, which
            # left the worst case scoring best: a run that produced NO rows at all (an arm that
            # collapsed at t=0) fell through to the ordinary path, and since burden and cost are
            # both non-negative its ``post_loss`` came out 0.0 — the global minimum of the metric,
            # beating every real run. A window with no rows is a window with no rows.
            inf = float("inf")
            base.update({"post_mandate_burden": inf, "post_rate_cost": inf,
                         "post_mean_rate": inf, "post_loss": inf})
            return base
        post_burden, post_cost = self._mandate_cost(post_rows)
        post_rates = [row.get(self.rate_key, 0.0) for row in post_rows]
        base.update({
            "post_mandate_burden": post_burden,
            "post_rate_cost": post_cost,
            "post_mean_rate": (sum(post_rates) / len(post_rates)) if post_rates else 0.0,
            "post_loss": post_burden + self.lam * post_cost,
        })
        return base
