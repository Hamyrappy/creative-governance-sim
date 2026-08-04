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
from govsim.harness import Critic, EpisodicMemory, TraceFeedback
from govsim.regents import LLMRegent, LQRRegent, OPRORegent, make_obfuscated_assembler, suppliable_names
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
    factory = _cubic_h1_factory()

    return Experiment(
        name="cubic_nonlinear",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-1.2 * current_x")},
        objectives={"regent:0": _h1_loss()},
        schedule=EveryN(50),
        seeds=[0, 1, 2, 3, 4],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-adaptation",
            claim="a code-as-policy regent infers the nonlinear plant from partial info and lowers "
                  "post-shock regret vs a frozen pre-shock-optimal controller and trace-less OPRO",
            baseline="frozen LQR/numeric-DP for the linearization + trace-less OPRO",
            primary_metric="post_mse",  # POST-shock MSE (steps >= shock); not diluted by the pre-shock half
            falsification="no lower post-shock regret than the frozen-optimal baseline over seeds",
        ),
        creativity_metric=CreativityMetric(
            name="functional-novelty", kind="functional_novelty",
            description="residual of the best-fit PID; use of conditionals/state-history a PID cannot",
        ),
    )


def _replay_client() -> CachingReplayClient:
    """An OpenAI-compatible client wrapped in the cache/replay tape, configured purely by env
    (no vendor/model hardcoded): OPENAI_BASE_URL, OPENAI_API_KEY_ENV, OPENAI_MODEL,
    GOVSIM_LLM_MODE (live|cache|replay), GOVSIM_LLM_CACHE. Construction is lazy/key-free, so
    registering + listing this experiment needs no API key; only ``govsim run`` of it calls the
    model (or replays a recorded tape)."""
    inner = OpenAICompatClient(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key_env=os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY"),
        # GOVSIM_LLM_DROP_PARAMS: comma-separated wire params this endpoint rejects (e.g. "seed"
        # for the Gemini OpenAI-compat layer, which 400s on it). Cache keys are unaffected.
        drop_params=frozenset(
            p.strip() for p in os.environ.get("GOVSIM_LLM_DROP_PARAMS", "").split(",") if p.strip()
        ),
        min_interval=float(os.environ.get("GOVSIM_LLM_MIN_INTERVAL", "0") or 0),
        timeout=float(os.environ.get("GOVSIM_LLM_TIMEOUT", "600") or 600),
    )
    return CachingReplayClient(inner, os.environ.get("GOVSIM_LLM_CACHE", "logs/llm_cache"),
                               mode=os.environ.get("GOVSIM_LLM_MODE", "cache"))


def _llm_opts() -> tuple[int | None, dict | None]:
    """Per-provider knobs from env: GOVSIM_LLM_MAX_TOKENS and GOVSIM_LLM_REASONING_EFFORT
    (e.g. ``low`` for gpt-oss, so it does not over-think and return empty content)."""
    mt = os.environ.get("GOVSIM_LLM_MAX_TOKENS")
    eff = os.environ.get("GOVSIM_LLM_REASONING_EFFORT")
    return (int(mt) if mt else None), ({"reasoning_effort": eff} if eff else None)


# Pre-shock linearization the FROZEN baselines (LQR) are built from — they cannot see the shock.
_CUBIC_H1_A, _CUBIC_H1_B = 0.95, 0.5
# The single structural-shock step for the cubic H1 arm. The headline metric is POST-shock only, so
# this same constant seeds both the plant's ``shock_step`` and the objective's ``post_shock_step``
# (keeping "when the shock fires" and "which window we score" provably in sync).
_CUBIC_H1_SHOCK_STEP = 100


def _h1_loss() -> "StabilizationLoss":
    """The H1 objective: full-horizon score, but its pre-registered comparison metric (``post_mse``)
    is windowed to AT-OR-AFTER the shock — so a paired ``compare`` measures the *post-shock regret*
    the H1 claim is about, not a pre/post-shock average (the pre-shock half would dilute/invert it)."""
    return StabilizationLoss(lam=0.1, post_shock_step=_CUBIC_H1_SHOCK_STEP)


def _cubic_h1_factory():
    """The shared H1 nonlinear arm: a cubic plant suffering an UNSEEN structural shock at t=100
    (``param_A`` jumps unstable + the nonlinearity strengthens). Every H1 regent/baseline binds to
    THIS same system so a paired comparison isolates the regent, and a frozen pre-shock-optimal
    controller provably cannot anticipate the regime change (doc-09 §6.4)."""
    # Severity NOTE (☐ AUTHOR knob): these values set whether H1 has headroom. Post-shock the plant is
    # mildly unstable AND the nonlinearity strengthens AND the state is kicked away from target, so a
    # frozen gentle controller (pre-shock LQR) recovers slowly / may diverge while an adaptive law that
    # infers the cubic recovers. Tune to keep the strong arms bounded but the frozen baseline stressed.
    cfg = {"param_A": _CUBIC_H1_A, "param_B": _CUBIC_H1_B, "param_C": 0.0, "sigma_epsilon": 0.08,
           "target_x": 0.0, "u_range": (-2.0, 2.0), "cubic_coeff": 0.05, "state_exponent": 3,
           "shock_step": _CUBIC_H1_SHOCK_STEP, "shock_params": {"param_A": 1.03, "cubic_coeff": 0.10},
           "shock_state_kick": 0.8}

    def factory(seed: int) -> CubicSystem:
        sys = CubicSystem(cfg)
        sys.reset(seed)
        return sys

    return factory


@register("cubic_nonlinear_llm")
def cubic_nonlinear_llm() -> Experiment:
    """The H1 nonlinear arm governed by an actual ``LLMRegent`` (OpenAI-compatible, cache/replay).

    Identical wiring to ``cubic_nonlinear`` but the regent is the LLM instead of the scripted
    baseline — the end-to-end demonstration that the machine runs an LLM-in-the-loop experiment.
    Run with a key (cache mode) once to record the tape, then replay for free/reproducible reruns:
        OPENAI_API_KEY=... OPENAI_MODEL=<model> govsim run cubic_nonlinear_llm --store logs/runs
    """
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens, extra = _llm_opts()

    factory = _cubic_h1_factory()

    return Experiment(
        name="cubic_nonlinear_llm",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": LLMRegent(llm=_replay_client(), model=model, temperature=0.0,
                                       max_tokens=max_tokens, extra=extra)},
        objectives={"regent:0": _h1_loss()},
        harness=Harness([TraceFeedback(), EpisodicMemory(k=3)]),  # the cheapest upgrades (doc-09 §5.2)
        schedule=EveryN(25),
        seeds=[0, 1, 2],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-adaptation",
            claim="an LLM code-as-policy regent infers the nonlinear plant from partial info and lowers "
                  "post-shock regret vs a frozen pre-shock-optimal controller and trace-less OPRO",
            baseline="frozen LQR/numeric-DP for the linearization + trace-less OPRO",
            primary_metric="post_mse",  # POST-shock MSE (steps >= shock); not diluted by the pre-shock half
            falsification="no lower post-shock regret than the frozen-optimal baseline over seeds",
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
        # Score from AT-OR-AFTER the first periodic shock (step 60), so the pre-shock warm-up where a
        # frozen controller is near-optimal is not averaged into the post-shock adaptation metric.
        objectives={"regent:0": StabilizationLoss(lam=0.1, post_shock_step=60)},
        schedule=EveryN(20),
        seeds=[0, 1, 2, 3, 4],
        horizon=300,
        hypothesis=Hypothesis(
            id="H1-coupled-adaptation",
            claim="a code-as-policy regent recovers from periodic unseen regime shocks with lower "
                  "post-shock regret than a frozen pre-shock-optimal controller and trace-less OPRO",
            baseline="frozen LQR for the main-state linearization + trace-less OPRO",
            primary_metric="post_mse",  # POST-first-shock MSE (steps >= 60); not diluted by the warm-up
            falsification="no lower post-shock regret than the frozen-optimal baseline over seeds",
        ),
        creativity_metric=CreativityMetric(
            name="functional-novelty", kind="functional_novelty",
            description="regime-detecting / state-history-using law a fixed-gain PID structurally cannot express",
        ),
    )


@register("cubic_nonlinear_opro")
def cubic_nonlinear_opro() -> Experiment:
    """The trace-LESS OPRO baseline on the H1 nonlinear arm — the named rival the harnessed LLM must
    beat (doc-09 §6.4). Same system/objective/seeds as ``cubic_nonlinear_llm``; the difference is the
    regent (OPRO archive, no trace channel) — so a paired comparison isolates the trace side-channel."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens, extra = _llm_opts()

    factory = _cubic_h1_factory()

    return Experiment(
        name="cubic_nonlinear_opro",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": OPRORegent("set_control_input", _replay_client(), model, temperature=0.8,
                                        scoring="realized",  # FAIR: learns only from realized outcomes
                                        max_tokens=max_tokens, extra=extra)},
        objectives={"regent:0": _h1_loss()},
        harness=Harness([]),  # trace-LESS by construction: no components
        schedule=EveryN(25),
        seeds=[0, 1, 2],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-adaptation",
            claim="trace-less OPRO is the baseline the harnessed LLM regent must beat on the nonlinear arm",
            baseline="this IS the named baseline (trace-less OPRO)",
            primary_metric="post_mse",  # POST-shock MSE (steps >= shock); not diluted by the pre-shock half
            falsification="the harnessed LLM does NOT lower post-shock regret vs this baseline over seeds",
        ),
        creativity_metric=None,
    )


@register("cubic_nonlinear_lqr")
def cubic_nonlinear_lqr() -> Experiment:
    """The FROZEN pre-shock-optimal baseline (doc-09 §6.4): an analytic LQR gain computed from the
    PRE-shock linearization (A=0.95, B=0.5) under the objective's Q/R (R=λ=0.1), held fixed through
    the regime shock. It cannot anticipate the shock or the nonlinearity — the controller H1 must beat
    on post-shock regret. Key-free (no LLM)."""
    return Experiment(
        name="cubic_nonlinear_lqr",
        system_factory=_cubic_h1_factory(),
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": LQRRegent("set_control_input", A=_CUBIC_H1_A, B=_CUBIC_H1_B, Q=1.0, R=0.1)},
        objectives={"regent:0": _h1_loss()},
        schedule=EveryN(25),
        seeds=[0, 1, 2],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-adaptation",
            claim="the frozen pre-shock-optimal LQR is the baseline the adaptive regent must beat post-shock",
            baseline="this IS the named frozen-optimal baseline",
            primary_metric="post_mse",  # POST-shock MSE (steps >= shock); not diluted by the pre-shock half
            falsification="the adaptive regent does NOT lower post-shock regret vs this frozen LQR",
        ),
        creativity_metric=None,
    )


@register("cubic_nonlinear_llm_obfuscated")
def cubic_nonlinear_llm_obfuscated() -> Experiment:
    """The true H1 PARTIAL-INFORMATION arm: the LLM regent is told only ``x_(k+1)=f(x_k,u_k,noise)``
    with f UNKNOWN/possibly-nonlinear, and must INFER the cubic structure from observed history (the
    obfuscated prompt). The boot-time ``check_prompt`` validates the template against the world's
    suppliable names at construction. EpisodicMemory feeds the history the regent reasons over."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens, extra = _llm_opts()

    factory = _cubic_h1_factory()

    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])
    sample = factory(0)
    assembler = make_obfuscated_assembler(suppliable_names(sample.observe(), iface.action_space(sample, "regent:0")))

    return Experiment(
        name="cubic_nonlinear_llm_obfuscated",
        system_factory=factory,
        action_interface=iface,
        regents={"regent:0": LLMRegent(llm=_replay_client(), model=model, temperature=0.0,
                                       prompt_assembler=assembler, prompt_file="obfuscated",
                                       max_tokens=max_tokens, extra=extra)},
        objectives={"regent:0": _h1_loss()},
        harness=Harness([TraceFeedback(), EpisodicMemory(k=4)]),
        schedule=EveryN(25),
        seeds=[0, 1, 2],
        horizon=200,
        hypothesis=Hypothesis(
            id="H1-adaptation",
            claim="under partial information (f unknown) the LLM infers the nonlinear plant from history "
                  "and lowers post-shock regret vs frozen-optimal and trace-less OPRO",
            baseline="frozen LQR/numeric-DP for the linearization + trace-less OPRO (cubic_nonlinear_opro)",
            primary_metric="post_mse",  # POST-shock MSE (steps >= shock); not diluted by the pre-shock half
            falsification="no lower post-shock regret than the frozen-optimal baseline over seeds",
        ),
        creativity_metric=CreativityMetric(
            name="functional-novelty", kind="functional_novelty",
            description="infers and exploits the cubic term from history; conditionals/state-powers a PID cannot",
        ),
    )


@register("cubic_nonlinear_llm_critic")
def cubic_nonlinear_llm_critic() -> Experiment:
    """The H1 nonlinear arm with a Critic in the harness (2nd-LLM audit→revise) on top of trace +
    memory — an H3 ablation arm (does the critic yield a separable gain?). The regent and critic
    share one cache/replay client."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens, extra = _llm_opts()
    client = _replay_client()

    factory = _cubic_h1_factory()

    return Experiment(
        name="cubic_nonlinear_llm_critic",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": LLMRegent(llm=client, model=model, temperature=0.0,
                                       max_tokens=max_tokens, extra=extra)},
        objectives={"regent:0": _h1_loss()},
        harness=Harness([TraceFeedback(), EpisodicMemory(k=3),
                         Critic(client, model, max_tokens=max_tokens, extra=extra)]),
        schedule=EveryN(25),
        seeds=[0, 1, 2],
        horizon=200,
        hypothesis=Hypothesis(
            id="H3-critic",
            claim="adding a critic (audit→revise) yields a separable gain over trace+memory alone",
            baseline="the same harness minus the Critic (leave-one-out, doc-09 §5.4)",
            primary_metric="post_mse",
            falsification="the critic's paired bootstrap CI does not exclude 0 in single-add AND LOO",
        ),
        creativity_metric=None,
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
