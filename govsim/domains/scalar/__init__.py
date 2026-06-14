"""
The scalar domain — bounded-lever control of low-dimensional dynamical systems.

A regent emits a single sandboxed Python expression per lever (e.g.
``{"verb": "set_control_input", "payload": {"expr": "-0.9 * current_x"}}``); the
``ScalarLeverInterface`` re-evaluates it each step and clips the result into the lever's range.
There is NO ledger, NO conservation — that machinery is confined to the (later) economy plugin.

Three systems ship here:
  - ``CubicSystem``  — the migrated linear/cubic stochastic scalar plant; the H1 partial-
    information nonlinear-control arm (set ``cubic_coeff != 0``) and the LQR sanity arm
    (``cubic_coeff = 0``).
  - ``SIRSystem``    — an epidemic with lockdown/vaccination levers; a NON-economic proof that
    the same core runs unchanged.
  - ``CompanySystem``— a firm with price/production levers (a dynamics model, not the economy).
"""

from govsim.domains.scalar.interface import Lever, ScalarLeverInterface
from govsim.domains.scalar.systems import CubicSystem, SIRSystem, CompanySystem, LeverSystem
from govsim.domains.scalar.objectives import StabilizationLoss, EpidemicLoss, CompanyProfit

__all__ = [
    "Lever",
    "ScalarLeverInterface",
    "LeverSystem",
    "CubicSystem",
    "SIRSystem",
    "CompanySystem",
    "StabilizationLoss",
    "EpidemicLoss",
    "CompanyProfit",
]
