"""
The flagship experiment suite: governing an endemic epidemic through an instrument failure.

**The setting.** A disease is endemic (waning immunity + imported cases, so it does not burn out).
The authority holds two levers, lockdown and vaccination, each with a standing cost. At t=100
lockdown efficacy collapses to a quarter of what it was — compliance fatigue — while a lockdown
still closes the same shops for the same price. Nothing announces this. The authority sees
prevalence, its own past choices, and whatever its harness remembers; efficacy is never observable.

**Why this world and not the scalar plant.** Measured, not assumed
(``scripts/headroom_audit.py``): a threshold-on-prevalence rule is *feedback*, so it absorbs a
transmissibility shock on its own — prevalence rises, the rule triggers more often, and a frozen
policy stays near-optimal. Headroom ≈ 1.0x, and no controller could have shown an effect there. An
**instrument** shock is not absorbed, because the observable no longer says what the lever is worth.
Headroom ≈ 1.7x, with an interior optimum on both sides. The scalar arms below are kept as a
declared negative control at ≈1.14x: the regime where the diagnostic predicts no findable effect.

**The two anchors** are calibrated by exhaustive search over one shared threshold-policy family
(``scripts/recalibrate.py`` → ``docs_gates/calibration.json``), so they differ only in *when* they
were allowed to look:

    frozen  ``0.9 if I > 0.01 else 0.0``   optimal before the break, deployed unchanged (R = 1)
    oracle  ``0.9 if I > 0.25 else 0.0``   optimal after it, clairvoyant                  (R = 0)

The oracle's answer is the substantive one: keep the intensity, raise the trigger twenty-fold. Stop
paying for an instrument that no longer works, except in a genuine emergency.

**The ablation** is a 2×2×2 factorial over three *different kinds* of information, not one-at-a-time:
``TraceFeedback`` (failures), ``OutcomeFeedback`` (realized performance), ``EpisodicMemory``
(retrieved precedent). Factorial because the interaction is the interesting part and because
one-at-a-time attribution over interacting scaffold components is known to mislead. The
pre-registered prediction is asymmetric: trace alone should do nothing here (it fires only on
malformed actions, and an instrument failure is not a malformed action), while outcome feedback is
the only channel from which an unobservable efficacy collapse is inferable at all.

Every LLM arm decides on the same schedule and therefore spends the same call budget — 20 decisions
per run — so no arm can win by simply being allowed to think more often. The Critic arm is the one
exception and is labelled as such: it buys extra calls, so it is reported against a call-matched
comparison rather than against the others.
"""

from __future__ import annotations

import os

from govsim.core.experiment import CreativityMetric, Experiment, Hypothesis
from govsim.core.harness import Harness
from govsim.core.llm import CachingReplayClient, OpenAICompatClient
from govsim.core.regent import ScriptedRegent
from govsim.core.schedule import EveryN
from govsim.domains.scalar import (
    EpidemicLoss,
    Lever,
    ScalarLeverInterface,
    StabilizationLoss,
)
from govsim.domains.scalar import regimes as R
from govsim.experiments import register
from govsim.harness import Critic, EpisodicMemory, OutcomeFeedback, TraceFeedback
from govsim.regents import LLMRegent, OPRORegent

# Seeds: the pre-registered headline count. 20 paired seeds is the stats-protocol floor for a
# headline claim; the development floor of 5 is for smoke runs only.
SEEDS = list(range(20))

EPIDEMIC_LEVERS = [
    Lever("set_lockdown", (0.0, 0.9), "lockdown"),
    Lever("set_vaccination", (0.0, 0.5), "vacc"),
]


def _iface() -> ScalarLeverInterface:
    return ScalarLeverInterface(list(EPIDEMIC_LEVERS))


def _objective() -> EpidemicLoss:
    return EpidemicLoss(lam=R.EPIDEMIC_LAMBDA, post_shock_step=R.EPIDEMIC_SHOCK_STEP)


def _client() -> CachingReplayClient:
    inner = OpenAICompatClient(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key_env=os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY"),
        drop_params=frozenset(
            p.strip() for p in os.environ.get("GOVSIM_LLM_DROP_PARAMS", "").split(",") if p.strip()
        ),
        # GOVSIM_LLM_MIN_INTERVAL: seconds to hold between calls. Gemini's free tier allows 15
        # requests/minute/model, so 4.2 keeps a long sweep under the limit instead of relying on
        # retries to absorb it.
        min_interval=float(os.environ.get("GOVSIM_LLM_MIN_INTERVAL", "0") or 0),
    )
    return CachingReplayClient(inner, os.environ.get("GOVSIM_LLM_CACHE", "logs/llm_cache"),
                               mode=os.environ.get("GOVSIM_LLM_MODE", "cache"))


def _llm_opts() -> tuple[int | None, dict | None]:
    mt = os.environ.get("GOVSIM_LLM_MAX_TOKENS")
    eff = os.environ.get("GOVSIM_LLM_REASONING_EFFORT")
    return (int(mt) if mt else None), ({"reasoning_effort": eff} if eff else None)


def _model() -> str:
    return os.environ.get("OPENAI_MODEL", "gemini-3.5-flash-lite")


_EPIDEMIC_CLAIM = (
    "under an unobservable collapse of instrument efficacy, a code-as-policy regent with a "
    "performance-feedback harness lowers post-shock governance loss relative to the pre-shock-optimal "
    "rule it started from"
)
_EPIDEMIC_BASELINE = (
    "calibrated frozen threshold institution (epidemic_frozen, R=1) and budget-matched trace-less "
    "OPRO (epidemic_opro); calibrated clairvoyant oracle (epidemic_oracle) anchors R=0"
)


def _epidemic_experiment(name: str, regent, harness: Harness, *, claim: str = _EPIDEMIC_CLAIM,
                         hyp_id: str = "H1-instrument-collapse", falsification: str = "",
                         creativity: CreativityMetric | None = None,
                         metadata: dict | None = None) -> Experiment:
    """Every epidemic arm, wired identically apart from the regent and the harness stack."""
    return Experiment(
        name=name,
        system_factory=R.sir_factory(R.EPIDEMIC_SHOCKED),
        action_interface=_iface(),
        regents={"regent:0": regent},
        objectives={"regent:0": _objective()},
        harness=harness,
        schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY),
        seeds=list(SEEDS),
        horizon=R.EPIDEMIC_HORIZON,
        hypothesis=Hypothesis(
            id=hyp_id,
            claim=claim,
            baseline=_EPIDEMIC_BASELINE,
            primary_metric="post_loss",
            falsification=falsification or (
                "the paired bootstrap CI of (arm - frozen) on post_loss includes 0 over 20 seeds"
            ),
        ),
        creativity_metric=creativity,
        metadata={
            "regime": "epidemic",
            "decisions_per_run": R.EPIDEMIC_HORIZON // R.EPIDEMIC_DECIDE_EVERY,
            **(metadata or {}),
        },
    )


# ---------------------------------------------------------------------------------------------
# The two calibrated anchors (key-free, deterministic)
# ---------------------------------------------------------------------------------------------


@register("epidemic_frozen")
def epidemic_frozen() -> Experiment:
    """R = 1. The threshold institution that was optimal before the break, held through it."""
    return _epidemic_experiment(
        "epidemic_frozen",
        ScriptedRegent(verb="set_lockdown", expr=R.reference_expr("epidemic", "frozen")),
        Harness([]),
        claim="this IS the named non-adaptive baseline: the pre-shock-optimal threshold institution",
        falsification="n/a — reference arm",
        metadata={"role": "reference:frozen", "normalized_regret": 1.0},
    )


@register("epidemic_oracle")
def epidemic_oracle() -> Experiment:
    """R = 0. Clairvoyant: the same policy family, re-optimized with the break already known."""
    return _epidemic_experiment(
        "epidemic_oracle",
        ScriptedRegent(verb="set_lockdown", expr=R.reference_expr("epidemic", "oracle")),
        Harness([]),
        claim="this IS the clairvoyant upper bound: the post-shock-optimal policy in the same family",
        falsification="n/a — reference arm",
        metadata={"role": "reference:oracle", "normalized_regret": 0.0},
    )


# ---------------------------------------------------------------------------------------------
# The 2x2x2 harness factorial (LLM regent, identical call budget)
# ---------------------------------------------------------------------------------------------

#: (suffix, builder) for each rollout-free component the factorial toggles.
_FACTORS = {
    "trace": lambda: TraceFeedback(),
    "outcome": lambda: OutcomeFeedback(k=4),
    "memory": lambda: EpisodicMemory(k=4),
}


def _register_factorial() -> None:
    """Register all 8 cells of the {trace} x {outcome} x {memory} design.

    Registered programmatically because writing eight near-identical factories by hand is how one
    of them silently ends up with a different seed list or horizon — the confound this design exists
    to avoid.
    """
    names = list(_FACTORS)
    for bits in range(8):
        on = [n for i, n in enumerate(names) if bits >> i & 1]
        suffix = "_".join(on) if on else "bare"
        exp_name = f"epidemic_llm_{suffix}"

        def make(on=tuple(on), exp_name=exp_name) -> Experiment:
            max_tokens, extra = _llm_opts()
            return _epidemic_experiment(
                exp_name,
                LLMRegent(llm=_client(), model=_model(), temperature=0.0,
                          max_tokens=max_tokens, extra=extra),
                Harness([_FACTORS[n]() for n in on]),
                metadata={"role": "treatment", "factors": list(on),
                          "trace": "trace" in on, "outcome": "outcome" in on,
                          "memory": "memory" in on, "budget_matched": True},
            )

        register(exp_name)(make)


_register_factorial()


@register("epidemic_llm_critic")
def epidemic_llm_critic() -> Experiment:
    """The full harness plus a Critic. NOT budget-matched — the critic buys extra LLM calls, so this
    arm is compared against a call-matched control rather than against the factorial cells."""
    max_tokens, extra = _llm_opts()
    client = _client()
    return _epidemic_experiment(
        "epidemic_llm_critic",
        LLMRegent(llm=client, model=_model(), temperature=0.0, max_tokens=max_tokens, extra=extra),
        Harness([TraceFeedback(), OutcomeFeedback(k=4), EpisodicMemory(k=4),
                 Critic(client, _model(), max_tokens=max_tokens, extra=extra)]),
        hyp_id="H3-critic",
        claim="a second-LLM audit yields a gain over the full rollout-free harness that survives "
              "matching the extra call budget",
        falsification="the critic's paired bootstrap CI does not exclude 0 once calls are matched",
        metadata={"role": "treatment", "factors": ["trace", "outcome", "memory", "critic"],
                  "budget_matched": False},
    )


@register("epidemic_opro")
def epidemic_opro() -> Experiment:
    """The named rival: trace-less OPRO on the identical world, at the identical call budget.

    OPRO learns only from an archive of (law, realized score) pairs — the optimization trajectory,
    with no error channel and no retrieved precedent. It is the right comparison for "is the harness
    doing anything a plain optimizer over past scores would not?", which is the objection the harness
    literature is currently built around answering badly.
    """
    max_tokens, extra = _llm_opts()
    return _epidemic_experiment(
        "epidemic_opro",
        OPRORegent("set_lockdown", _client(), _model(), temperature=0.8, scoring="realized",
                   max_tokens=max_tokens, extra=extra),
        Harness([]),
        claim="this IS the named budget-matched adaptive rival (trace-less OPRO)",
        falsification="the harnessed regent does NOT lower post_loss relative to this arm",
        metadata={"role": "reference:opro", "budget_matched": True},
    )


# ---------------------------------------------------------------------------------------------
# The scalar NEGATIVE CONTROL (declared near-null: measured headroom ~1.14x)
# ---------------------------------------------------------------------------------------------

_SCALAR_NULL_NOTE = (
    "declared negative control: the measured headroom in this regime is ~1.14x, so the diagnostic "
    "predicts almost no findable effect here. It is reported to show the headroom measure has "
    "discriminating power, not because an effect is expected."
)


def _scalar_experiment(name: str, regent, harness: Harness, claim: str, role: str) -> Experiment:
    return Experiment(
        name=name,
        system_factory=R.cubic_factory(R.SCALAR_SHOCKED),
        action_interface=ScalarLeverInterface(
            [Lever("set_control_input", R.SCALAR_U_RANGE, "current_u")]),
        regents={"regent:0": regent},
        objectives={"regent:0": StabilizationLoss(lam=R.SCALAR_LAMBDA,
                                                  post_shock_step=R.SCALAR_SHOCK_STEP)},
        harness=harness,
        schedule=EveryN(R.SCALAR_DECIDE_EVERY),
        seeds=list(SEEDS),
        horizon=R.SCALAR_HORIZON,
        hypothesis=Hypothesis(
            id="H1-negative-control",
            claim=claim,
            baseline="calibrated frozen (scalar_frozen) and clairvoyant oracle (scalar_oracle)",
            primary_metric="post_loss",
            falsification="an effect IS found here despite ~1.14x headroom, which would indicate "
                          "the headroom measure understates what is achievable outside the "
                          "calibration family",
        ),
        metadata={"regime": "scalar", "role": role, "note": _SCALAR_NULL_NOTE},
    )


@register("scalar_frozen")
def scalar_frozen() -> Experiment:
    return _scalar_experiment(
        "scalar_frozen",
        ScriptedRegent(verb="set_control_input", expr=R.reference_expr("scalar", "frozen")),
        Harness([]), "pre-shock-optimal law held through the break (R=1 anchor)", "reference:frozen")


@register("scalar_oracle")
def scalar_oracle() -> Experiment:
    return _scalar_experiment(
        "scalar_oracle",
        ScriptedRegent(verb="set_control_input", expr=R.reference_expr("scalar", "oracle")),
        Harness([]), "post-shock-optimal law in the same family (R=0 anchor)", "reference:oracle")


@register("scalar_llm_full")
def scalar_llm_full() -> Experiment:
    max_tokens, extra = _llm_opts()
    return _scalar_experiment(
        "scalar_llm_full",
        LLMRegent(llm=_client(), model=_model(), temperature=0.0, max_tokens=max_tokens, extra=extra),
        Harness([TraceFeedback(), OutcomeFeedback(k=4), EpisodicMemory(k=4)]),
        "the full-harness regent in a regime with almost no headroom to win", "treatment")
