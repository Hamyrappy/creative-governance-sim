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
    LQR cost). Returned negated so higher is better."""

    def __init__(self, target_key: str = "target_x", state_key: str = "current_x",
                 control_key: str = "current_u", lam: float = 0.1, target: float = 0.0,
                 window: int | None = None) -> None:
        self.target_key = target_key
        self.state_key = state_key
        self.control_key = control_key
        self.lam = lam
        self.target = target
        self.window = window

    def _mse_msu(self, trajectory: Trajectory) -> tuple[float, float]:
        rows = _window(trajectory, self.window)
        if not rows:
            return 0.0, 0.0
        n = len(rows)
        mse = sum((row.get(self.state_key, 0.0) - row.get(self.target_key, self.target)) ** 2 for row in rows) / n
        msu = sum(row.get(self.control_key, 0.0) ** 2 for row in rows) / n
        return mse, msu

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        mse, msu = self._mse_msu(trajectory)
        return -(mse + self.lam * msu)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        mse, msu = self._mse_msu(trajectory)
        rows = _window(trajectory, self.window)
        final_x = rows[-1].get(self.state_key, 0.0) if rows else 0.0
        mean_abs_x = (sum(abs(r.get(self.state_key, 0.0)) for r in rows) / len(rows)) if rows else 0.0
        return {"mse": mse, "msu": msu, "loss": mse + self.lam * msu, "final_x": final_x, "mean_abs_x": mean_abs_x}


class EpidemicLoss(Objective):
    """Minimize cumulative infection-burden + λ·intervention-cost (negated → higher is better)."""

    def __init__(self, lam: float = 1.0, infected_key: str = "infected", cost_key: str = "cum_cost") -> None:
        self.lam = lam
        self.infected_key = infected_key
        self.cost_key = cost_key

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        total_infected = sum(row.get(self.infected_key, 0.0) for row in trajectory)
        final_cost = trajectory[-1].get(self.cost_key, 0.0) if trajectory else 0.0
        return -(total_infected + self.lam * final_cost)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        infected = [row.get(self.infected_key, 0.0) for row in trajectory]
        return {
            "total_infected": sum(infected),
            "peak_infected": max(infected) if infected else 0.0,
            "cum_cost": trajectory[-1].get(self.cost_key, 0.0) if trajectory else 0.0,
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
