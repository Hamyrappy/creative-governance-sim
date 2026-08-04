"""
Objectives for the scalar domain. Concrete objectives are domain-bound plugins (welfare/loss
logic needs the domain's observables), so they live here, not in ``govsim.core`` — only the
``Objective`` ABC is in core (grand plan §10).

Convention: ``evaluate`` returns a score where **higher is better** (so selection/evolution can
``argmax`` uniformly across objectives); a loss is therefore returned negated. ``components``
always logs the raw sub-metrics (MSE, MSU, peak, cost, …) so a "proxy up / true-down" Goodhart
episode (H5) is detectable after the fact.
"""

from __future__ import annotations

from govsim.core.objective import Objective, Trajectory


def _window(trajectory: Trajectory, window: int | None) -> Trajectory:
    if window is None or window <= 0 or window >= len(trajectory):
        return trajectory
    return trajectory[-window:]


class StabilizationLoss(Objective):
    """Dual-mandate control loss: ``MSE(x→target) + λ·MSU(u)`` (generalizes the thesis loss & the
    LQR cost). Returned negated so higher is better.

    ``post_shock_step`` is the H1 headline knob (doc-09 §6.4): when set, ``components`` also emits
    ``post_mse``/``post_msu``/``post_loss`` computed over the rows AT-OR-AFTER that step — the
    *post-shock* window the H1 claim is actually about. The pre-shock half (where a frozen
    pre-shock-optimal controller is near-optimal) is thus not averaged into the pre-registered
    comparison metric, which would otherwise dilute — and can invert — the post-shock verdict.
    The full-horizon ``mse``/``score`` are unchanged, so golden-master/score semantics are stable.
    """

    def __init__(self, target_key: str = "target_x", state_key: str = "current_x",
                 control_key: str = "current_u", lam: float = 0.1, target: float = 0.0,
                 window: int | None = None, post_shock_step: int | None = None,
                 step_key: str = "step") -> None:
        self.target_key = target_key
        self.state_key = state_key
        self.control_key = control_key
        self.lam = lam
        self.target = target
        self.window = window
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def _mse_msu(self, rows: Trajectory) -> tuple[float, float]:
        if not rows:
            return 0.0, 0.0
        n = len(rows)
        for row in rows:
            if self.state_key not in row:  # loud-fail a mis-wired objective/system pairing (do not
                raise KeyError(              # silently score a missing state as 0.0 → tiny fake MSE)
                    f"StabilizationLoss: state_key '{self.state_key}' absent from a trajectory row "
                    f"(keys: {sorted(row)}); the objective is wired to the wrong system/observable."
                )
        mse = sum((row[self.state_key] - row.get(self.target_key, self.target)) ** 2 for row in rows) / n
        msu = sum(row.get(self.control_key, 0.0) ** 2 for row in rows) / n
        return mse, msu

    def _post_shock_rows(self, trajectory: Trajectory) -> Trajectory:
        """Rows at-or-after ``post_shock_step`` (by the row's ``step`` value if present, else index)."""
        s = self.post_shock_step
        if s is None:
            return list(trajectory)
        out = [row for row in trajectory if row.get(self.step_key, -1) >= s]
        if out:
            return out
        return trajectory[s:] if s < len(trajectory) else []

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        mse, msu = self._mse_msu(_window(trajectory, self.window))
        return -(mse + self.lam * msu)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        rows = _window(trajectory, self.window)
        mse, msu = self._mse_msu(rows)
        final_x = rows[-1].get(self.state_key, 0.0) if rows else 0.0
        mean_abs_x = (sum(abs(r.get(self.state_key, 0.0)) for r in rows) / len(rows)) if rows else 0.0
        # Post-shock window (== full window when no post_shock_step is set) — the H1 headline metric.
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and trajectory and not post_rows:
            # The run ended BEFORE the shock (e.g. diverged pre-shock): there is no post-shock data,
            # so the post-shock metric is worst-case (inf), NOT a flattering 0.0. The collapse detector
            # tracks this separately; here we just refuse to reward a run that never reached the regime.
            post_mse = post_msu = float("inf")
        else:
            post_mse, post_msu = self._mse_msu(post_rows)
        return {"mse": mse, "msu": msu, "loss": mse + self.lam * msu, "final_x": final_x,
                "mean_abs_x": mean_abs_x, "post_mse": post_mse, "post_msu": post_msu,
                "post_loss": post_mse + self.lam * post_msu}


class EpidemicLoss(Objective):
    """Minimize cumulative infection-burden + λ·intervention-cost (negated → higher is better).

    **λ is the substantive knob, not a nuisance parameter.** It prices a unit of lockdown against a
    unit of infection, so it is what decides whether the optimal institution is interior
    ("lock down at severity θ") or a corner ("always lock down at the maximum"). A corner optimum is
    invariant to the epidemiological regime — no variant shock can move it — so a domain calibrated
    at a corner has *zero* adaptation headroom by construction. λ must therefore be set large enough
    to make restraint rational before this world can host an adaptation experiment at all.

    ``post_shock_step`` mirrors ``StabilizationLoss``: it emits ``post_*`` components over the rows
    at-or-after the variant's arrival, so the pre-shock window (where a pre-shock-optimal rule is by
    definition near-optimal) is not averaged into the headline comparison.
    """

    def __init__(self, lam: float = 1.0, infected_key: str = "infected", cost_key: str = "cum_cost",
                 post_shock_step: int | None = None, step_key: str = "t") -> None:
        self.lam = lam
        self.infected_key = infected_key
        self.cost_key = cost_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def _burden_cost(self, rows: Trajectory) -> tuple[float, float]:
        """(infection burden, intervention cost) over ``rows``.

        The cost is differenced across the window rather than read off the last row: ``cum_cost`` is
        a running total from t=0, so on a post-shock window the undifferenced value would silently
        bill the post-shock policy for lockdowns bought before the variant existed.
        """
        if not rows:
            return 0.0, 0.0
        burden = sum(row.get(self.infected_key, 0.0) for row in rows)
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
        """Negated loss over ``trajectory``, which may be a WINDOW rather than a whole run.

        The cost is differenced across the window, exactly as ``components`` does. It has to be:
        ``cum_cost`` is a running total from t=0, so reading its last value scores a 10-step window
        by *when it happened* rather than by what the policy did in it. Two windows with identical
        infection burden and identical per-step spending then differ by the whole cost accrued
        before either of them started.

        This is not a hypothetical. The Runner calls this method on the interval since the regent's
        previous decision to produce the realized-performance signal a harness shows the model
        (``core/runner.py``), and with the undifferenced version that signal was a monotonically
        falling clock: across 20 seeds it reported "worse than last time" 359 times and "better"
        never. A regent shown that signal rationally abandons whatever it is doing, and an ablation
        built on it measures the passage of time.

        For a full trajectory the two definitions differ only by the first step's cost, so the
        reported ``components["loss"]`` — which was always differenced — is unaffected.
        """
        burden, cost = self._burden_cost(trajectory)
        return -(burden + self.lam * cost)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        infected = [row.get(self.infected_key, 0.0) for row in trajectory]
        burden, cost = self._burden_cost(trajectory)
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and trajectory and not post_rows:
            # The run ENDED before the shock — e.g. the epidemic burnt out, so the variant never
            # arrived. An empty window must not score 0.0: that would make "let it rip until the
            # run terminates" the optimal post-shock policy, and a reference calibrated that way is
            # measuring early termination, not governance. Worst-case it, exactly as
            # StabilizationLoss does; the collapse detector reports the termination separately.
            inf = float("inf")
            return {
                "total_infected": sum(infected),
                "peak_infected": max(infected) if infected else 0.0,
                "cum_cost": trajectory[-1].get(self.cost_key, 0.0),
                "loss": burden + self.lam * cost,
                "post_infected": inf, "post_peak_infected": inf,
                "post_cost": inf, "post_loss": inf,
            }
        post_burden, post_cost = self._burden_cost(post_rows)
        post_infected = [row.get(self.infected_key, 0.0) for row in post_rows]
        return {
            "total_infected": sum(infected),
            "peak_infected": max(infected) if infected else 0.0,
            "cum_cost": trajectory[-1].get(self.cost_key, 0.0) if trajectory else 0.0,
            # The full trade-off — the quantity a reference policy must actually be optimal for.
            "loss": burden + self.lam * cost,
            "post_infected": post_burden,
            "post_peak_infected": max(post_infected) if post_infected else 0.0,
            "post_cost": post_cost,
            "post_loss": post_burden + self.lam * post_cost,
        }


class CompanyProfit(Objective):
    """Maximize mean per-step profit."""

    def __init__(self, profit_key: str = "profit") -> None:
        self.profit_key = profit_key

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        if not trajectory:
            return 0.0
        return sum(row.get(self.profit_key, 0.0) for row in trajectory) / len(trajectory)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        profits = [row.get(self.profit_key, 0.0) for row in trajectory]
        return {
            "total_profit": sum(profits),
            "mean_profit": (sum(profits) / len(profits)) if profits else 0.0,
            "final_cash": trajectory[-1].get("cash", 0.0) if trajectory else 0.0,
        }
