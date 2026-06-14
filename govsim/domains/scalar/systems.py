"""
Scalar dynamical systems behind the ``ScalarLeverInterface``.

Reproducibility (doc-08 HARD rule): every system owns its own ``numpy.random.Generator``,
seeded in ``reset(seed)``; nothing draws from the module-global ``random`` / ``numpy.random``
state (a CI AST test enforces this). ``clone()`` deepcopies the system *including* the
Generator, so a counterfactual rollout is a faithful continuation of the same stochastic
stream — the precondition for a sound rollout-based fitness oracle.

The apply/step contract (the bug doc-08 §3.2 flagged: the old ``SingleMarketModel`` eval'd the
policy once at apply time while every other world re-eval'd per step): here the
``ScalarLeverInterface`` *installs* a compiled lever expression onto the system as **pure data**
(a code object + range + viewer id), and ``System.step()`` re-evaluates it *every step* via
``_reeval_levers``. Storing the action as data (not a closure over the interface or the system)
is what keeps it clone-safe — a deepcopied clone re-evaluates against *itself*, not the original.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from govsim.core.sandbox import eval_safe
from govsim.core.system import Observation, RollableSystem, StepInfo


@dataclass(frozen=True)
class _InstalledLever:
    """A compiled lever expression installed on a system (pure, deepcopy-safe data)."""

    compiled: Any  # an immutable RestrictedPython code object (deepcopy returns itself)
    low: float
    high: float
    regent_id: str


class LeverSystem(RollableSystem):
    """Base for scalar systems controlled by clipped expression-levers.

    Subclasses implement the dynamics in ``step`` (calling ``self._reeval_levers()`` first),
    expose state via ``observe``/``metrics``, and list their lever attributes. The lever
    plumbing (install + per-step re-eval + clip) lives here so ``CubicSystem``/``SIRSystem``/
    ``CompanySystem`` stay short and the eval-cadence contract is implemented in exactly one place.
    """

    def __init__(self) -> None:
        self._levers: dict[str, _InstalledLever] = {}

    # -- lever plumbing (called by ScalarLeverInterface.apply / by step) --------------------

    def install_lever(self, attr: str, compiled: Any, value_range: tuple[float, float], regent_id: str) -> None:
        """Install/replace the compiled expression that drives ``attr`` each step."""
        low, high = value_range
        self._levers[attr] = _InstalledLever(compiled=compiled, low=float(low), high=float(high), regent_id=regent_id)

    def set_lever(self, attr: str, value: float) -> None:
        setattr(self, attr, float(value))

    def _reeval_levers(self) -> None:
        """Re-evaluate every installed lever against the current view and clip into range.

        On a runtime eval error (``eval_safe`` returns ``None``) or a non-numeric result, the
        lever keeps its previous value — matching the legacy ``LinearStochasticSystem`` contract.
        """
        for attr, lev in self._levers.items():
            variables = self.observe(lev.regent_id).vars
            value = eval_safe(lev.compiled, variables)
            if value is None or not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            self.set_lever(attr, max(lev.low, min(lev.high, float(value))))

    # -- RollableSystem capability ----------------------------------------------------------

    def clone(self) -> "LeverSystem":
        return copy.deepcopy(self)


class CubicSystem(LeverSystem):
    """A scalar stochastic plant: ``x_{k+1} = A·x_k + B·u_k + C + g·x_k**p + ε_k``.

    The ``g·x_k**p`` term (``cubic_coeff`` × ``state_exponent``, default ``p = 3``) is the
    project's most novel arm: with ``cubic_coeff = 0`` it is the linear plant of the thesis
    (LQR is the exact ground-truth ceiling, the sanity arm); with ``cubic_coeff != 0`` it is an
    UNKNOWN nonlinear plant the regent must infer from partial information + history (the H1
    adaptation arm). ``u`` is the single lever ``current_u``, clipped to ``u_range``.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        self.initial_x: float = float(p.get("initial_x", 0.0))
        self.param_A: float = float(p.get("param_A", 0.95))
        self.param_B: float = float(p.get("param_B", 0.5))
        self.param_C: float = float(p.get("param_C", 0.0))
        self.sigma_epsilon: float = float(p.get("sigma_epsilon", 0.1))
        tx = p.get("target_x", 0.0)
        self.target_x: float | None = None if tx is None else float(tx)
        lo, hi = p.get("u_range", (-2.0, 2.0))
        self.u_range: tuple[float, float] = (float(lo), float(hi))
        self.cubic_coeff: float = float(p.get("cubic_coeff", 0.0))
        self.state_exponent: int = int(p.get("state_exponent", 3))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"current_u": self.u_range}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.current_x: float = self.initial_x
        self.previous_x: float = self.initial_x
        self.current_u: float = 0.0
        self._t: int = 0
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()  # re-eval the installed lever expression EACH step (the contract)
        shock = float(self.rng.normal(0.0, self.sigma_epsilon))
        next_x = (
            self.param_A * self.current_x
            + self.param_B * self.current_u
            + self.param_C
            + self.cubic_coeff * (self.current_x ** self.state_exponent)
            + shock
        )
        self.previous_x = self.current_x
        self.current_x = next_x
        self._t += 1
        terminated = not np.isfinite(self.current_x) or abs(self.current_x) > 1e9
        return StepInfo(terminated=terminated, truncated=False, info={"shock": shock})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        variables: dict[str, float] = {
            "step": float(self._t),
            "current_x": self.current_x,
            "previous_x": self.previous_x,
            "current_u": self.current_u,
        }
        if self.target_x is not None:
            variables["target_x"] = self.target_x
        return Observation(vars=variables, scope=viewer_id, t=self._t)

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        m = {
            "step": float(self._t),
            "current_x": self.current_x,
            "current_u": self.current_u,
            "previous_x": self.previous_x,
        }
        if self.target_x is not None:
            m["target_x"] = self.target_x
        return m


class SIRSystem(LeverSystem):
    """An SIR epidemic with lockdown + vaccination levers — a NON-economic proof of generality.

    The same core (Regent / Harness / Runner / ResultStore / cache) runs this unchanged: the
    regent emits ``{verb: "set_lockdown", payload: {"expr": "0.7 if I > 0.1 else 0.2"}}`` — a
    sandboxed expression, no ``Mint``/``Transfer``/``Ledger`` anywhere. The β-jump at ``t==120``
    is exactly H1's "unseen structural shock" (a variant), the same scientific spine as the cubic.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        self.beta0_init: float = float(p.get("beta0", 0.35))
        self.gamma: float = float(p.get("gamma", 0.10))
        self.noise_sigma: float = float(p.get("noise_sigma", 0.0))
        self.shock_step: int = int(p.get("shock_step", 120))
        self.shock_factor: float = float(p.get("shock_factor", 1.8))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"lockdown": (0.0, 0.9), "vacc": (0.0, 0.5)}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.S, self.I, self.R = 0.99, 0.01, 0.0
        self.beta0 = self.beta0_init
        self.lockdown = 0.0
        self.vacc = 0.0
        self._t = 0
        self.cum_cost = 0.0
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()
        beta = self.beta0 * (1.0 - self.lockdown)
        noise = float(self.rng.normal(0.0, self.noise_sigma)) if self.noise_sigma > 0 else 0.0
        new_i = max(0.0, beta * self.S * self.I + noise)
        vaccinated = self.vacc * self.S * 0.02
        self.S += -new_i - vaccinated
        self.I += new_i - self.gamma * self.I
        self.R += self.gamma * self.I + vaccinated
        self.S = max(0.0, self.S)
        if self._t == self.shock_step:
            self.beta0 *= self.shock_factor  # H1 unseen structural shock: a more transmissible variant
        self.cum_cost += self.lockdown * 1.0 + self.vacc * 0.5
        self._t += 1
        return StepInfo(terminated=self.I < 1e-4, truncated=False, info={})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            vars={"S": self.S, "I": self.I, "R": self.R, "lockdown": self.lockdown, "vacc": self.vacc, "t": float(self._t)},
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        return {"S": self.S, "I": self.I, "R": self.R, "infected": self.I, "cum_cost": self.cum_cost, "t": float(self._t)}


class CompanySystem(LeverSystem):
    """A single firm with price + production levers — a *dynamics* model, not the economy plugin.

    Demand responds to price (with noise); the regent maximizes ``CompanyProfit``. Ledger-free,
    so it rides the trivial ``ScalarLeverInterface`` exactly like the SIR system — a second
    non-economic domain on the same core.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        self.demand_a: float = float(p.get("demand_a", 50.0))
        self.demand_b: float = float(p.get("demand_b", 2.0))
        self.unit_cost: float = float(p.get("unit_cost", 5.0))
        self.fixed_cost: float = float(p.get("fixed_cost", 10.0))
        self.demand_sigma: float = float(p.get("demand_sigma", 1.0))
        self.init_price: float = float(p.get("init_price", 15.0))
        self.init_production: float = float(p.get("init_production", 20.0))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"price": (0.0, 100.0), "production": (0.0, 200.0)}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.price = self.init_price
        self.production = self.init_production
        self.last_profit = 0.0
        self.last_demand = 0.0
        self.cash = 0.0
        self._t = 0
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()
        noise = float(self.rng.normal(0.0, self.demand_sigma))
        demand = max(0.0, self.demand_a - self.demand_b * self.price + noise)
        sold = min(self.production, demand)
        profit = sold * self.price - self.unit_cost * self.production - self.fixed_cost
        self.last_demand = demand
        self.last_profit = profit
        self.cash += profit
        self._t += 1
        return StepInfo(terminated=False, truncated=False, info={})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            vars={
                "price": self.price,
                "production": self.production,
                "last_demand": self.last_demand,
                "last_profit": self.last_profit,
                "cash": self.cash,
                "t": float(self._t),
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        return {"profit": self.last_profit, "cash": self.cash, "price": self.price, "demand": self.last_demand, "t": float(self._t)}
