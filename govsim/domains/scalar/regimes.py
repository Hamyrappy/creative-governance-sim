"""
The pinned experimental regimes — one place where "what world are we studying" is written down.

Every arm of a comparison must bind to the *identical* plant, or the comparison measures the plant.
So the world configs live here as module constants rather than being re-typed per experiment, and
the calibrated reference policies are read from a generated artifact
(``govsim/docs_gates/calibration.json``, produced by ``scripts/recalibrate.py``) rather than
hand-copied — a hand-copied gain is a silent fork waiting to happen.

Two regimes, chosen by measurement (``scripts/headroom_audit.py``), not by taste:

``EPIDEMIC``  — the flagship. An endemic SIRS disease; at t=100 lockdown efficacy collapses to a
                quarter while lockdown still costs what it always cost. Headroom ≈ 1.7x, with an
                interior optimum on both sides of the break, and the plant cannot diverge because
                S+I+R is conserved.

``SCALAR``    — the declared **negative control**. The scalar cubic plant under the original
                state-dynamics shock, where the measured headroom is ≈ 1.1x. It is kept precisely
                *because* it is near-null: it is where the headroom diagnostic predicts no effect is
                findable, so finding none there is a check on the instrument rather than a
                disappointment. Reporting only the domain that worked would be the whole problem.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from govsim.domains.scalar.systems import CubicSystem, SIRSystem

CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "docs_gates" / "calibration.json"

# ---------------------------------------------------------------------------------------------
# EPIDEMIC — endemic SIRS + an instrument-efficacy collapse
# ---------------------------------------------------------------------------------------------
EPIDEMIC_SHOCK_STEP = 100
EPIDEMIC_HORIZON = 200
EPIDEMIC_LAMBDA = 0.08  # prices lockdown against infection; interior optimum on both sides
EPIDEMIC_DECIDE_EVERY = 10  # ⇒ 20 decisions per run: the per-arm LLM call budget

#: The stationary pre-shock world. A reference policy calibrated here is what a competent authority
#: could have legislated with everything knowable before the break.
EPIDEMIC_PRE: dict[str, Any] = {
    "beta0": 0.35, "gamma": 0.10,
    "waning": 0.02, "import_rate": 0.0005,     # endemic: the disease is still here at t=100
    "lockdown_efficacy": 1.0, "vacc_efficacy": 1.0, "vacc_rate": 0.02,
    "lockdown_cost": 1.0, "vacc_cost": 0.5,
    "shock_step": 10_000,                       # i.e. never, within the horizon
    # Per-seed population heterogeneity + per-step incidence noise. A seed has to mean something:
    # with a noiseless SIR every seed gives the same trajectory, so the paired shared-seed design
    # would pair identical numbers and report a zero-width confidence interval as a finding.
    "beta0_sigma": 0.18, "gamma_sigma": 0.10, "initial_i": 0.01, "initial_i_sigma": 0.40,
    "noise_sigma": 0.0015,
}

#: The world every arm actually runs on. ``shock_factor=1.0`` keeps transmissibility fixed, so the
#: break is purely an instrument failure — a threshold rule cannot absorb it by triggering more often.
EPIDEMIC_SHOCKED: dict[str, Any] = dict(
    EPIDEMIC_PRE,
    shock_step=EPIDEMIC_SHOCK_STEP,
    shock_factor=1.0,
    shock_params={"lockdown_efficacy": 0.25},
)

# ---------------------------------------------------------------------------------------------
# SCALAR — the near-null negative control
# ---------------------------------------------------------------------------------------------
SCALAR_SHOCK_STEP = 100
SCALAR_HORIZON = 200
SCALAR_LAMBDA = 0.1
SCALAR_DECIDE_EVERY = 10
SCALAR_U_RANGE = (-2.0, 2.0)

SCALAR_PRE: dict[str, Any] = {
    "param_A": 0.95, "param_B": 0.5, "param_C": 0.0, "sigma_epsilon": 0.08,
    "target_x": 0.0, "u_range": SCALAR_U_RANGE, "cubic_coeff": 0.05, "state_exponent": 3,
}
SCALAR_SHOCKED: dict[str, Any] = dict(
    SCALAR_PRE,
    shock_step=SCALAR_SHOCK_STEP,
    shock_params={"param_A": 1.03, "param_B": 0.12, "cubic_coeff": 0.10},
    shock_state_kick=0.8,
)


def sir_factory(cfg: dict[str, Any]):
    def factory(seed: int) -> SIRSystem:
        s = SIRSystem(cfg)
        s.reset(seed)
        return s
    return factory


def cubic_factory(cfg: dict[str, Any]):
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem(cfg)
        s.reset(seed)
        return s
    return factory


@lru_cache(maxsize=1)
def calibration() -> dict[str, Any]:
    """The generated frozen/oracle reference policies, or ``{}`` if not yet calibrated.

    Kept as a *generated artifact* on purpose: the reference policies are empirical results with a
    seed count and a search grid behind them, so they belong in a file with that provenance next to
    them, not in a literal that nobody can trace back to a run.
    """
    if not CALIBRATION_PATH.exists():
        return {}
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


def reference_expr(regime: str, which: str) -> str:
    """Look up a calibrated reference law, e.g. ``reference_expr("epidemic", "frozen")``."""
    data = calibration().get(regime)
    if not data or which not in data:
        raise RuntimeError(
            f"no calibrated '{which}' policy for regime '{regime}'. "
            f"Run `uv run python scripts/recalibrate.py` to generate {CALIBRATION_PATH.name}."
        )
    return data[which]["expr"]
