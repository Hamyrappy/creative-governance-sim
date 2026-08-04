"""
A stock-flow-consistent fiscal world: households, a government, and one accounting identity.

The model is Godley & Lavoie's SIM with two governed levers. Output is demand-determined,
consumption runs off disposable income and accumulated wealth, and the only thing that moves
money is the government: it spends, it taxes, and the difference lands on household balances.

**Why this world exists.** The platform's calibration results say that a shock to the governed
state buys nothing — a feedback rule keyed on the state absorbs it by firing more often, and the
measured adaptation headroom comes back at ~1.0x. The family that leaves headroom is a shock to
the *instrument*, and fiscal policy has a textbook one: tax compliance. When compliance collapses,
the enacted rate keeps distorting the economy exactly as much as it always did — the wedge between
what a household earns and what it keeps is a function of the statutory rate, not of the
collection rate — while the revenue that arrives is a fraction of it. So the price of the
instrument is unchanged and its yield is gone, which is precisely the asymmetry a rule tuned on the
old regime cannot express as a different threshold on the same lever. ``tax_compliance`` is absent
from ``observe`` on purpose: a governor sees output, its own balance sheet, and its own enacted
policy, and has to work out from the drift that the tax system stopped delivering.

**Boundedness (no arm may diverge).** Three mechanisms, and none of them is a tuning constant:

1. Money is conserved by construction. ``M_h + M_g`` is invariant to machine precision, because
   every period computes a single net transfer and applies it with opposite signs. There is no
   path in ``step`` that creates or destroys a unit of money.
2. Both stocks are hard-bounded by that conservation plus two clips that are institutional facts
   rather than numerical guards: the government may not pay out more than its debt ceiling allows
   (``spending <= M_g + debt_limit``), and households may not remit more tax than they hold
   (``taxes <= M_h + spending``). Together these pin ``M_g >= -debt_limit`` and ``M_h >= 0``, and
   conservation then pins the other end of each: ``M_h <= money_total + debt_limit`` and
   ``M_g <= money_total``.
3. Output is an affine function of a bounded wealth stock and a bounded spending flow, divided by
   a multiplier denominator that is bounded below by ``1 - alpha_income > 0``, then clipped at
   zero. A bounded numerator over a denominator bounded away from zero cannot run away.

So the worst a catastrophic policy can do is saturate: spend the state into its ceiling and let
output sit at the level that bounded wealth supports. That is a bad outcome with a finite score,
which is what a comparison needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem


class FiscalSFCEconomy(LeverSystem):
    """Households + a government on one conserved money stock, governed by a tax rate and a transfer.

    Each period, in order:

    * the government pays out ``gov_base_spend + transfer`` (capped by its debt ceiling);
    * output is whatever that spending plus household demand adds up to —
      ``Y = (alpha_wealth·M_h + spending + noise) / (1 - alpha_income_eff·(1 - tau_eff))``,
      the fixed point of ``Y = C + G`` with ``C = alpha_income_eff·(Y - taxes) + alpha_wealth·M_h``;
    * households remit ``tau_eff·Y``, where ``tau_eff = tax_compliance · tax_rate``.

    The two rates are deliberately different objects. ``tax_rate`` is what the institution
    *enacted*: it sets the wedge that shrinks the effective propensity to consume
    (``alpha_income_eff = alpha_income·(1 - tax_distortion·tax_rate)``) and it is what the objective
    bills, every step, whether or not a cent arrives. ``tax_compliance`` is what fraction of the
    enacted rate is actually collected, it multiplies revenue only, and it is not observable.
    Driving it toward zero therefore leaves the cost of taxation untouched and deletes its yield.

    Per-seed heterogeneity is real, not decorative: the two propensities and the opening household
    balance are drawn per seed from ``self.rng``, so seeds differ in how much government the economy
    needs to reach potential, and a paired shared-seed design has something to pair over.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        # -- behaviour --------------------------------------------------------------------------
        self.alpha_income_init: float = float(p.get("alpha_income", 0.60))   # consume out of income
        self.alpha_wealth_init: float = float(p.get("alpha_wealth", 0.15))   # consume out of wealth
        # The enacted rate's deadweight. This is the channel that makes the shock asymmetric: it
        # keys on ``tax_rate`` and never on ``tax_compliance``, so an uncollected tax still bites.
        self.tax_distortion0: float = float(p.get("tax_distortion", 0.50))
        self.gov_base_spend0: float = float(p.get("gov_base_spend", 0.0))
        self.noise_sigma0: float = float(p.get("noise_sigma", 1.0))          # autonomous demand noise
        # -- instrument efficacy (the shockable, UNOBSERVABLE parameter) -------------------------
        self.tax_compliance0: float = float(p.get("tax_compliance", 1.0))
        # -- stocks and institutional limits ------------------------------------------------------
        self.m_h_init: float = float(p.get("m_h_init", 120.0))
        self.m_g_init: float = float(p.get("m_g_init", 0.0))
        self.debt_limit0: float = float(p.get("debt_limit", 400.0))
        # ``*0`` snapshots of EVERY remaining parameter a shock may overwrite. The three below are
        # not hypothetical targets: a fall in potential output is the standard macro shock, and
        # ``y_potential`` is both published in ``observe`` and read by the objective, so a shock that
        # survived ``reset`` would silently move the target of every later "fresh" run of the same
        # object — and the contaminated runs would still look self-consistent.
        self.transfer_cap0: float = float(p.get("transfer_cap", 40.0))
        self.tax_rate_cap0: float = float(p.get("tax_rate_cap", 0.6))
        self.y_potential0: float = float(p.get("y_potential", 100.0))
        # -- per-seed heterogeneity (lognormal, so the draws stay positive) -----------------------
        self.alpha_income_sigma: float = float(p.get("alpha_income_sigma", 0.10))
        self.alpha_wealth_sigma: float = float(p.get("alpha_wealth_sigma", 0.18))
        self.m_h_sigma: float = float(p.get("m_h_sigma", 0.25))
        # -- the unseen structural break ----------------------------------------------------------
        ss = p.get("shock_step")
        self.shock_step: int | None = None if ss is None else int(ss)
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {}))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"tax_rate": (0.0, self.tax_rate_cap), "transfer": (0.0, self.transfer_cap)}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        # ``alpha_income`` is clipped strictly below 1: it is the multiplier's denominator, and at
        # 1.0 with a zero effective tax the fixed point Y = C + G has no finite solution.
        self.alpha_income = float(
            np.clip(self.alpha_income_init * np.exp(self.rng.normal(0.0, self.alpha_income_sigma)), 0.05, 0.95)
        )
        self.alpha_wealth = float(
            np.clip(self.alpha_wealth_init * np.exp(self.rng.normal(0.0, self.alpha_wealth_sigma)), 0.01, 0.90)
        )
        self.M_h = float(max(0.0, self.m_h_init * np.exp(self.rng.normal(0.0, self.m_h_sigma))))
        self.M_g = self.m_g_init
        # The conserved quantity, fixed at reset because the opening household balance is a per-seed
        # draw. Every later step leaves this untouched; ``test_money_is_conserved`` is the proof.
        self.money_total = self.M_h + self.M_g
        # restore EVERY parameter a shock may overwrite, so a re-run of the same object is pristine
        self.tax_compliance = self.tax_compliance0
        self.tax_distortion = self.tax_distortion0
        self.gov_base_spend = self.gov_base_spend0
        self.noise_sigma = self.noise_sigma0
        self.debt_limit = self.debt_limit0
        self.transfer_cap = self.transfer_cap0
        self.tax_rate_cap = self.tax_rate_cap0
        self.y_potential = self.y_potential0
        self.tax_rate = 0.0
        self.transfer = 0.0
        self.output = 0.0
        self.prev_output = 0.0
        self.deficit = 0.0
        self.cum_tax_cost = 0.0
        self._t = 0
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()
        if self.shock_step is not None and self._t == self.shock_step:
            for name, value in self.shock_params.items():
                setattr(self, name, float(value))
        # The enacted rate distorts regardless of what it collects; compliance scales only revenue.
        alpha_income_eff = self.alpha_income * max(0.0, 1.0 - self.tax_distortion * self.tax_rate)
        tau_eff = max(0.0, self.tax_compliance) * self.tax_rate
        # A government cannot pay out money it does not have and may not borrow past its ceiling.
        spending = max(0.0, min(self.gov_base_spend + self.transfer, self.M_g + self.debt_limit))
        noise = float(self.rng.normal(0.0, self.noise_sigma)) if self.noise_sigma > 0 else 0.0
        denom = 1.0 - alpha_income_eff * (1.0 - tau_eff)  # >= 1 - alpha_income > 0 by the reset clip
        output = max(0.0, (self.alpha_wealth * self.M_h + spending + noise) / denom)
        # Households remit out of the balance they hold after receiving this period's spending, so
        # the payment can never overdraw them — the floor that makes M_h >= 0 structural.
        taxes = min(tau_eff * output, self.M_h + spending)
        net = spending - taxes  # ONE number, applied with opposite signs: money cannot leak
        self.M_h += net
        self.M_g -= net
        self.prev_output = self.output
        self.output = output
        self.deficit = net
        # Billed on the POLICY SET, not on the revenue it raised. A tax nobody pays is still a tax
        # everybody arranges their affairs around, and that asymmetry is what makes an efficacy
        # collapse expensive to ignore rather than merely disappointing.
        self.cum_tax_cost += self.tax_rate
        self._t += 1
        terminated = not (np.isfinite(self.M_h) and np.isfinite(self.M_g) and np.isfinite(self.output))
        return StepInfo(terminated=terminated, truncated=False, info={"deficit": net})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # ``tax_compliance`` is NOT here, and neither is the tax take that would reveal it by
            # division. The governor sees what it enacted, what the economy produced, and where its
            # own balance sits; that the tax system stopped delivering is an inference from the
            # drift between them, which is the whole task. Publishing the parameter deletes it.
            vars={
                "output": self.output,
                "prev_output": self.prev_output,
                "y_potential": self.y_potential,
                "M_h": self.M_h,
                "M_g": self.M_g,
                "tax_rate": self.tax_rate,
                "transfer": self.transfer,
                "t": float(self._t),
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # The enacted levers are part of the record, not just the state they produced: the
        # governance question is what the institution DID. ``money_total`` rides along so the
        # conservation invariant is auditable from a stored trajectory, not only from a live object.
        return {
            "output": self.output,
            "y_potential": self.y_potential,
            "M_h": self.M_h,
            "M_g": self.M_g,
            "money_total": self.M_h + self.M_g,
            "deficit": self.deficit,
            "tax_rate": self.tax_rate,
            "transfer": self.transfer,
            "cum_tax_cost": self.cum_tax_cost,
            "t": float(self._t),
        }


class FiscalStabilizationLoss(Objective):
    """Hold output at potential and the public balance flat, at the least statutory taxation.

    ``loss = Σ ((Y - Y*)/Y*)² + mu·Σ (deficit/deficit_scale)² + lam·Σ tax_rate`` (negated, so higher
    is better). The three terms are not interchangeable and each one is load-bearing:

    * the output gap alone is minimized by spending the maximum, which is a corner;
    * the deficit term alone is minimized by taxing and spending nothing, which is the other corner
      and would make "never act" optimal — a policy no shock can move, and therefore a world with
      zero adaptation headroom by construction;
    * the tax term prices the instrument, and it prices the *enacted* rate. Without it the cheapest
      way to hold both targets is a punitive rate, which is again a corner.

    With all three, the optimum is interior: enough spending to lift demand to potential, a rate
    high enough to fund it without drifting, and no more rate than that. ``lam`` is the substantive
    knob — set it too low and the optimum walks to the tax ceiling, where no compliance shock can
    move it, and the world stops being able to host the experiment.

    ``post_shock_step`` mirrors ``EpidemicLoss``: it emits ``post_*`` over the rows at-or-after the
    break, so the pre-shock half — where a pre-shock-optimal rule is near-optimal by definition —
    is not averaged into the headline comparison.
    """

    def __init__(self, lam: float = 0.2, mu: float = 1.0, y_potential: float = 100.0,
                 deficit_scale: float = 20.0, output_key: str = "output",
                 potential_key: str = "y_potential", deficit_key: str = "deficit",
                 cost_key: str = "cum_tax_cost", post_shock_step: int | None = None,
                 step_key: str = "t") -> None:
        self.lam = lam
        self.mu = mu
        self.y_potential = y_potential
        self.deficit_scale = deficit_scale
        self.output_key = output_key
        self.potential_key = potential_key
        self.deficit_key = deficit_key
        self.cost_key = cost_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Keep output at potential and the public finances flat, and do it with as "
            "little taxation as that takes. Your score sums, over the horizon:\n"
            "  - the squared relative gap between output and potential, every step;\n"
            f"  - {self.mu} times the squared per-step government deficit (spending minus revenue), "
            f"scaled by {self.deficit_scale} — a balance that drifts in either direction is a "
            "failure, and borrowing has a hard ceiling you cannot spend past;\n"
            f"  - {self.lam} times the tax rate you ENACT, every step, charged on the rate itself "
            "and not on what it collects: the statutory rate distorts the economy whether or not "
            "the revenue ever arrives, so a rate that raises nothing still costs you everything.\n"
            "Lower total is better. All three matter: spending output up to potential while the "
            "deficit runs, and balancing the books on a collapsed economy, are both failures."
        )

    def _terms(self, rows: Trajectory) -> tuple[float, float, float]:
        """``(output gap, deficit penalty, tax cost)`` over ``rows``.

        The tax cost is DIFFERENCED across the window rather than read off the last row.
        ``cum_tax_cost`` is a running total from t=0, so on a post-shock window the undifferenced
        value bills the post-shock policy for every rate enacted before the break existed — which
        turns the score into a clock that only ever gets worse. That exact bug shipped here once.
        """
        if not rows:
            return 0.0, 0.0, 0.0
        for row in rows:
            if self.output_key not in row:  # loud-fail a mis-wired objective/system pairing rather
                raise KeyError(             # than score a missing output as a constant gap of 1.0
                    f"FiscalStabilizationLoss: output_key '{self.output_key}' absent from a "
                    f"trajectory row (keys: {sorted(row)}); the objective is wired to the wrong system."
                )
        gap = 0.0
        for row in rows:
            potential = row.get(self.potential_key, self.y_potential) or self.y_potential
            gap += ((row[self.output_key] - potential) / potential) ** 2
        drift = sum((row.get(self.deficit_key, 0.0) / self.deficit_scale) ** 2 for row in rows)
        cost = rows[-1].get(self.cost_key, 0.0) - rows[0].get(self.cost_key, 0.0)
        return gap, drift, cost

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
        realized-performance signal the harness shows the model, so the differencing in ``_terms``
        is not cosmetic: reading ``cum_tax_cost`` undifferenced would score a ten-step window by
        when it happened rather than by what was done in it.
        """
        gap, drift, cost = self._terms(trajectory)
        return -(gap + self.mu * drift + self.lam * cost)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        gap, drift, cost = self._terms(trajectory)
        outputs = [row.get(self.output_key, 0.0) for row in trajectory]
        post_rows = self._post_shock_rows(trajectory)
        base = {
            "output_gap": gap,
            "deficit_penalty": drift,
            "tax_cost": cost,
            "loss": gap + self.mu * drift + self.lam * cost,
            "mean_output": (sum(outputs) / len(outputs)) if outputs else 0.0,
            "final_position": trajectory[-1].get("M_g", 0.0) if trajectory else 0.0,
            "cum_tax_cost": trajectory[-1].get(self.cost_key, 0.0) if trajectory else 0.0,
        }
        if self.post_shock_step is not None and trajectory and not post_rows:
            # The run ended BEFORE the break, so there is no post-shock evidence. Scoring that as
            # 0.0 would crown any policy that gets the run terminated early; worst-case it instead,
            # exactly as StabilizationLoss and EpidemicLoss do.
            inf = float("inf")
            return {**base, "post_output_gap": inf, "post_deficit_penalty": inf,
                    "post_tax_cost": inf, "post_loss": inf}
        post_gap, post_drift, post_cost = self._terms(post_rows)
        return {**base,
                "post_output_gap": post_gap,
                "post_deficit_penalty": post_drift,
                "post_tax_cost": post_cost,
                "post_loss": post_gap + self.mu * post_drift + self.lam * post_cost}
