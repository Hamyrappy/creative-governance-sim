"""
Scalar-domain experiments (rung 1 + the rung-1.5 generality proof). All are key-free: they use
the deterministic ``ScriptedRegent`` baseline so they run on CI without an LLM. The LLM-regent
variants land with ``govsim/regents/`` (Phase 1); the wiring here is what they slot into.

Each carries a WHAT-first ``Hypothesis`` so the ``Runner`` gate passes — modeling the discipline
that an experiment names its claim + baseline up front (docs in ``govsim/docs_gates/``).
"""

from __future__ import annotations

from govsim.core.experiment import CreativityMetric, Experiment, Hypothesis
from govsim.core.schedule import EveryN
from govsim.core.regent import ScriptedRegent
from govsim.domains.scalar import (
    CompanyProfit,
    CompanySystem,
    CubicSystem,
    EpidemicLoss,
    Lever,
    ScalarLeverInterface,
    SIRSystem,
    StabilizationLoss,
)
from govsim.experiments import register


@register("cubic_stabilization")
def cubic_stabilization() -> Experiment:
    """LINEAR scalar plant (cubic_coeff=0): the LQR-sanity arm. ScriptedRegent proportional law.

    This is the deterministic golden-master experiment (fixed seed → fixed trajectory, no LLM)."""
    def factory(seed: int) -> CubicSystem:
        sys = CubicSystem({"param_A": 0.95, "param_B": 0.5, "param_C": 0.0,
                           "sigma_epsilon": 0.1, "target_x": 0.0, "u_range": (-2.0, 2.0)})
        sys.reset(seed)
        return sys

    return Experiment(
        name="cubic_stabilization",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-0.9 * current_x")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(50),
        seeds=[0, 1, 2, 3, 4],
        horizon=200,
        hypothesis=Hypothesis(
            id="H0-linear-sanity",
            claim="a proportional code-as-policy regent matches LQR/OPRO on the linear plant (sanity)",
            baseline="LQR ground-truth + OPRO (Phase 1); no-control as the trivial floor",
            primary_metric="mse",
            falsification="MSE not within CI of the LQR optimum over seeds",
        ),
        creativity_metric=None,  # honestly: the linear arm has no creativity headroom (doc-09 §6.4)
    )


@register("cubic_nonlinear")
def cubic_nonlinear() -> Experiment:
    """The H1 partial-information NONLINEAR arm: an unknown cubic plant (state_exponent=3).

    With cubic_coeff != 0 the plant is nonlinear; the obfuscated prompt (Phase 1, for the LLM
    regent) reveals only "a third-degree main term", so the regent must infer structure. Here the
    baseline ScriptedRegent simply demonstrates the wiring runs and stays bounded."""
    def factory(seed: int) -> CubicSystem:
        sys = CubicSystem({"param_A": 0.95, "param_B": 0.5, "param_C": 0.0,
                           "sigma_epsilon": 0.08, "target_x": 0.0, "u_range": (-2.0, 2.0),
                           "cubic_coeff": 0.05, "state_exponent": 3})
        sys.reset(seed)
        return sys

    return Experiment(
        name="cubic_nonlinear",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-1.2 * current_x")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(50),
        seeds=[0, 1, 2, 3, 4],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-adaptation",
            claim="a code-as-policy regent infers the nonlinear plant from partial info and lowers "
                  "post-shock regret vs a frozen pre-shock-optimal controller and trace-less OPRO",
            baseline="frozen LQR/numeric-DP for the linearization + trace-less OPRO",
            primary_metric="mse",
            falsification="no lower regret than the frozen-optimal baseline over seeds",
        ),
        creativity_metric=CreativityMetric(
            name="functional-novelty", kind="functional_novelty",
            description="residual of the best-fit PID; use of conditionals/state-history a PID cannot",
        ),
    )


@register("sir_lockdown")
def sir_lockdown() -> Experiment:
    """NON-economic generality proof: SIR epidemic on the SAME core (zero ledger), threshold lockdown."""
    def factory(seed: int) -> SIRSystem:
        sys = SIRSystem({"beta0": 0.35, "gamma": 0.10, "noise_sigma": 0.0, "shock_step": 120})
        sys.reset(seed)
        return sys

    return Experiment(
        name="sir_lockdown",
        system_factory=factory,
        action_interface=ScalarLeverInterface([
            Lever("set_lockdown", (0.0, 0.9), "lockdown"),
            Lever("set_vaccination", (0.0, 0.5), "vacc"),
        ]),
        regents={"regent:0": ScriptedRegent(verb="set_lockdown", expr="0.8 if I > 0.05 else 0.0")},
        objectives={"regent:0": EpidemicLoss(lam=1.0)},
        schedule=EveryN(5),
        seeds=[0, 1, 2],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-sir-transfer",
            claim="the SAME rung-1 regent/harness/objective stabilizes a non-economic system unchanged",
            baseline="no-intervention SIR (StaticRegent)",
            primary_metric="total_infected",
        ),
        creativity_metric=None,
    )


@register("company_pricing")
def company_pricing() -> Experiment:
    """NON-economic generality proof #2: a firm with price/production levers, maximizing profit."""
    def factory(seed: int) -> CompanySystem:
        sys = CompanySystem({"demand_a": 50.0, "demand_b": 2.0, "unit_cost": 5.0,
                             "fixed_cost": 10.0, "demand_sigma": 1.0})
        sys.reset(seed)
        return sys

    return Experiment(
        name="company_pricing",
        system_factory=factory,
        action_interface=ScalarLeverInterface([
            Lever("set_price", (0.0, 100.0), "price"),
            Lever("set_production", (0.0, 200.0), "production"),
        ]),
        regents={"regent:0": ScriptedRegent(verb="set_production", expr="max(0.0, last_demand)")},
        objectives={"regent:0": CompanyProfit()},
        schedule=EveryN(1),
        seeds=[0, 1, 2],
        horizon=100,
        hypothesis=Hypothesis(
            id="H-company",
            claim="a code-as-policy regent raises mean profit vs a fixed-production baseline",
            baseline="fixed production = initial",
            primary_metric="mean_profit",
        ),
        creativity_metric=None,
    )
