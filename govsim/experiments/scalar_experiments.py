"""
Scalar-domain experiments (rung 1 + the rung-1.5 generality proof). All are key-free: they use
the deterministic ``ScriptedRegent`` baseline so they run on CI without an LLM. The LLM-regent
variants land with ``govsim/regents/`` (Phase 1); the wiring here is what they slot into.

Each carries a WHAT-first ``Hypothesis`` so the ``Runner`` gate passes — modeling the discipline
that an experiment names its claim + baseline up front (docs in ``govsim/docs_gates/``).
"""

from __future__ import annotations

import os

from govsim.core.experiment import CreativityMetric, Experiment, Hypothesis
from govsim.core.harness import Harness
from govsim.core.llm import CachingReplayClient, OpenAICompatClient
from govsim.core.schedule import EveryN
from govsim.core.regent import ScriptedRegent
from govsim.harness import EpisodicMemory, TraceFeedback
from govsim.regents import LLMRegent
from govsim.domains.scalar import (
    CompanyProfit,
    CompanySystem,
    CoupledSystem,
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


def _replay_client() -> CachingReplayClient:
    """An OpenAI-compatible client wrapped in the cache/replay tape, configured purely by env
    (no vendor/model hardcoded): OPENAI_BASE_URL, OPENAI_MODEL, GOVSIM_LLM_MODE (live|cache|replay),
    GOVSIM_LLM_CACHE. Construction is lazy/key-free, so registering + listing this experiment needs
    no API key; only ``govsim run`` of it calls the model (or replays a recorded tape)."""
    inner = OpenAICompatClient(base_url=os.environ.get("OPENAI_BASE_URL"))
    return CachingReplayClient(inner, os.environ.get("GOVSIM_LLM_CACHE", "logs/llm_cache"),
                               mode=os.environ.get("GOVSIM_LLM_MODE", "cache"))


@register("cubic_nonlinear_llm")
def cubic_nonlinear_llm() -> Experiment:
    """The H1 nonlinear arm governed by an actual ``LLMRegent`` (OpenAI-compatible, cache/replay).

    Identical wiring to ``cubic_nonlinear`` but the regent is the LLM instead of the scripted
    baseline — the end-to-end demonstration that the machine runs an LLM-in-the-loop experiment.
    Run with a key (cache mode) once to record the tape, then replay for free/reproducible reruns:
        OPENAI_API_KEY=... OPENAI_MODEL=<model> govsim run cubic_nonlinear_llm --store logs/runs
    """
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def factory(seed: int) -> CubicSystem:
        sys = CubicSystem({"param_A": 0.95, "param_B": 0.5, "param_C": 0.0,
                           "sigma_epsilon": 0.08, "target_x": 0.0, "u_range": (-2.0, 2.0),
                           "cubic_coeff": 0.05, "state_exponent": 3})
        sys.reset(seed)
        return sys

    return Experiment(
        name="cubic_nonlinear_llm",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": LLMRegent(llm=_replay_client(), model=model, temperature=0.0)},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        harness=Harness([TraceFeedback(), EpisodicMemory(k=3)]),  # the cheapest upgrades (doc-09 §5.2)
        schedule=EveryN(25),
        seeds=[0, 1, 2],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-adaptation",
            claim="an LLM code-as-policy regent infers the nonlinear plant from partial info and lowers "
                  "post-shock regret vs a frozen pre-shock-optimal controller and trace-less OPRO",
            baseline="frozen LQR/numeric-DP for the linearization + trace-less OPRO",
            primary_metric="mse",
            falsification="no lower regret than the frozen-optimal baseline over seeds",
        ),
        creativity_metric=CreativityMetric(
            name="functional-novelty", kind="functional_novelty",
            description="residual of the best-fit PID; conditionals/state-history a PID cannot use",
        ),
    )


def _coupled_factory(extra: dict | None = None):
    base = {"param_A": 0.95, "param_B": 0.4, "param_C": 0.0, "sigma_epsilon": 0.10,
            "target_x": 0.0, "u_range": (-2.0, 2.0), "u_smoothing_rho": 0.70}
    base.update(extra or {})

    def factory(seed: int) -> CoupledSystem:
        sys = CoupledSystem(base)
        sys.reset(seed)
        return sys

    return factory


@register("coupled_stabilization")
def coupled_stabilization() -> Experiment:
    """Coupled multi-state plant, (near-)stationary sanity arm: shocks effectively off, mild drift.

    Control inertia (rho_u) + cross-coupling already make this a non-trivial control problem that a
    single fixed gain only partly solves; it is the coupled analogue of ``cubic_stabilization``."""
    return Experiment(
        name="coupled_stabilization",
        system_factory=_coupled_factory({"shock_period": 0, "param_B_drift_sigma": 0.005}),
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "u_commanded")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-1.2 * current_x")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(25),
        seeds=[0, 1, 2, 3, 4],
        horizon=300,
        hypothesis=Hypothesis(
            id="H0-coupled-sanity",
            claim="a proportional code-as-policy regent keeps the coupled plant bounded near target (sanity)",
            baseline="no-control (StaticRegent) + tuned PID",
            primary_metric="mse",
            falsification="MSE no better than no-control over seeds",
        ),
        creativity_metric=None,
    )


@register("coupled_regime_shift")
def coupled_regime_shift() -> Experiment:
    """The coupled H1 arm: periodic external regime shocks (every 60 steps) kick the hidden aux
    states, breaking a worked-out policy — the multi-state analogue of the cubic adaptation arm."""
    return Experiment(
        name="coupled_regime_shift",
        system_factory=_coupled_factory({"shock_period": 60, "shock_magnitude_aux1": 1.0,
                                          "shock_magnitude_aux2": -0.8, "param_B_drift_sigma": 0.01}),
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "u_commanded")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-1.2 * current_x")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(20),
        seeds=[0, 1, 2, 3, 4],
        horizon=300,
        hypothesis=Hypothesis(
            id="H1-coupled-adaptation",
            claim="a code-as-policy regent recovers from periodic unseen regime shocks with lower "
                  "post-shock regret than a frozen pre-shock-optimal controller and trace-less OPRO",
            baseline="frozen LQR for the main-state linearization + trace-less OPRO",
            primary_metric="mse",
            falsification="no lower post-shock regret than the frozen-optimal baseline over seeds",
        ),
        creativity_metric=CreativityMetric(
            name="functional-novelty", kind="functional_novelty",
            description="regime-detecting / state-history-using law a fixed-gain PID structurally cannot express",
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
