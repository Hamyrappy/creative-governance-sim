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
policy stays near-optimal. No controller could have shown an effect there. An **instrument** shock
is not absorbed, because the observable no longer says what the lever is worth. The scalar arms
below are kept as a declared negative control.

**Four calibrated references**, all by exhaustive search over the same policy vocabulary — which
spans BOTH instruments, because the arms can use both and a reference confined to one is not a
weaker opponent but an unfair one. The laws themselves live in the generated artifact
(``scripts/recalibrate.py`` → ``docs_gates/calibration.json``) and are read from it rather than
written here, so this docstring cannot drift out of step with what actually ran:

    frozen      optimal before the break, held unchanged through it                        (R = 1)
    best_fixed  the best FIXED law over the whole broken horizon, chosen in hindsight —
                the non-adaptive ceiling, and what "the regent adapted" has to beat
    oracle      the best fixed law for the post-break window (reported, not targeted)
    switching   the (pre-leg, post-leg) pair searched JOINTLY — the clairvoyant ADAPTOR    (R = 0)

The switching pair is searched jointly rather than composed from two separately-optimal legs,
because the pre-break leg decides the state the post-break leg inherits. Composed, it was beaten by
a fixed law in the harsher regime — impossible for a real upper bound.

**Why the metric is the full horizon and not the post-break window.** Scoring only the post-break
window rewards *passivity*: a policy that never intervenes is wrong before the break and, because
the break disables the instrument, close to right after it — so it scores well without having
adapted to anything. We found this when the no-harness control arm landed within 11% of the
post-window oracle while emitting mostly constants. Over the full horizon the free lunch is gone,
because no fixed law is optimal on both sides of a break that moves the optimum. **Beating
``best_fixed`` is therefore what "the regent adapted" has to mean**, and it is the pre-registered
falsification target (deviation logged in ``docs_gates/preregistration.md`` §8).

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
from govsim.core.regent import MultiScriptedRegent, ScriptedRegent
from govsim.core.schedule import EveryN
from govsim.domains.scalar import (
    EpidemicLoss,
    Lever,
    ScalarLeverInterface,
    StabilizationLoss,
)
from govsim.domains.scalar import regimes as R
from govsim.experiments import register
from govsim.harness import (
    ContextualOutcomeFeedback, ContrastiveMemory, Critic, EpisodicMemory, OutcomeFeedback,
    TraceFeedback,
)
from govsim.regents import LLMRegent, OPRORegent, SwitchingRegent

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
        timeout=float(os.environ.get("GOVSIM_LLM_TIMEOUT", "600") or 600),
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
    "the best FIXED law in hindsight (epidemic_best_fixed) — the non-adaptive ceiling — plus the "
    "calibrated frozen institution (epidemic_frozen, R=1) and budget-matched trace-less OPRO "
    "(epidemic_opro); the clairvoyant switching adaptor (epidemic_switching) anchors R=0"
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
            primary_metric="loss",  # FULL horizon — see the note on passivity below
            falsification=falsification or (
                "the paired bootstrap CI of (arm - best_fixed) on full-horizon loss includes 0 "
                "over 20 seeds"
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
        MultiScriptedRegent(R.calibration()["epidemic"]["frozen"]["laws"]),
        Harness([]),
        claim="this IS the named non-adaptive baseline: the pre-shock-optimal threshold institution",
        falsification="n/a — reference arm",
        metadata={"role": "reference:frozen", "normalized_regret": 1.0},
    )


@register("epidemic_best_fixed")
def epidemic_best_fixed() -> Experiment:
    """The NON-ADAPTIVE CEILING: the single best fixed law over the whole broken horizon, chosen
    with hindsight. Beating this is what "the regent adapted" has to mean — a fixed law cannot be
    optimal on both sides of a break that moves the optimum, so any arm that beats it must have
    changed its behaviour, and no arm can reach it by being passive."""
    return _epidemic_experiment(
        "epidemic_best_fixed",
        MultiScriptedRegent(R.calibration()["epidemic"]["best_fixed"]["laws"]),
        Harness([]),
        claim="this IS the non-adaptive ceiling: the best fixed law in hindsight",
        falsification="n/a — reference arm",
        metadata={"role": "reference:best_fixed"},
    )


@register("epidemic_switching")
def epidemic_switching() -> Experiment:
    """R = 0. The clairvoyant ADAPTOR: pre-break optimum until the break, post-break optimum after.
    It is handed both laws and the exact switch time, none of which any other arm can see."""
    d = R.calibration()["epidemic"]["switching"]
    return _epidemic_experiment(
        "epidemic_switching",
        SwitchingRegent("set_lockdown", d["pre_laws"], d["post_laws"], R.EPIDEMIC_SHOCK_STEP),
        Harness([]),
        claim="this IS the clairvoyant upper bound: the optimal policy switch at the exact break",
        falsification="n/a — reference arm",
        metadata={"role": "reference:switching", "normalized_regret": 0.0},
    )


@register("epidemic_oracle")
def epidemic_oracle() -> Experiment:
    """R = 0. Clairvoyant: the same policy family, re-optimized with the break already known."""
    return _epidemic_experiment(
        "epidemic_oracle",
        MultiScriptedRegent(R.calibration()["epidemic"]["oracle"]["laws"]),
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


@register("epidemic_llm_ctx_outcome")
def epidemic_llm_ctx_outcome() -> Experiment:
    """Outcome feedback that reports each score NEXT TO the state it was earned in.

    The plain outcome channel produced no effect, and the recorded prompts say why: its scores come
    from different phases of an evolving epidemic, so the signal is confounded with the very
    non-stationarity it is meant to reveal. This arm is the minimal fix — same channel, same call
    budget, states attached, and an explicit flag when the same law scores differently in comparable
    conditions. Its control is ``epidemic_llm_outcome``, so the contrast isolates *contextualization*
    rather than *feedback*."""
    max_tokens, extra = _llm_opts()
    return _epidemic_experiment(
        "epidemic_llm_ctx_outcome",
        LLMRegent(llm=_client(), model=_model(), temperature=0.0, max_tokens=max_tokens, extra=extra),
        Harness([ContextualOutcomeFeedback(k=4)]),
        hyp_id="H3b-contextual-outcome",
        claim="attaching the state to each realized score recovers the regime signal that a bare "
              "score history confounds with the regime change itself",
        falsification="the paired CI of (ctx_outcome - outcome) on full-horizon loss includes 0",
        metadata={"role": "treatment", "factors": ["ctx_outcome"], "budget_matched": True,
                  "control_arm": "epidemic_llm_outcome"},
    )


@register("epidemic_llm_contrastive")
def epidemic_llm_contrastive() -> Experiment:
    """Episodic memory reframed as a CHOICE rather than a precedent, at identical information.

    The measured problem this answers: plain ``EpisodicMemory`` is the only channel in the factorial
    that significantly hurts (+0.630 loss, p_Holm=0.002), and the mechanism is lock-in rather than
    misinformation. Policy churn falls from 0.850 to 0.082 and distinct policies from 14.4 to 2.1 —
    shown what it did before, the regent does it again. The obvious alternative explanation, that
    retrieval serves stale pre-break precedent, was tested and is false: only 39.6% of episodes
    retrieved post-break predate it, against a chance baseline near 69%.

    ``ContrastiveMemory`` retrieves the SAME episodes and reports the SAME scores, ranked best-first,
    in the third person, with the spread between outcomes named. Nothing is added and nothing is
    withheld, so any difference is attributable to framing rather than to content.

    Its control is ``epidemic_llm_memory``, so the contrast isolates *presentation* rather than
    *recall*. PREDICTION, recorded before the arm was run: churn rises materially above 0.082."""
    max_tokens, extra = _llm_opts()
    return _epidemic_experiment(
        "epidemic_llm_contrastive",
        LLMRegent(llm=_client(), model=_model(), temperature=0.0, max_tokens=max_tokens, extra=extra),
        Harness([ContrastiveMemory(k=4)]),
        hyp_id="H3c-contrastive-memory",
        claim="presenting the same retrieved precedents as ranked options rather than as the "
              "agent's own past commitments restores policy revision and recovers the loss that "
              "episodic memory costs",
        falsification="the paired CI of (contrastive - memory) on full-horizon loss includes 0",
        metadata={"role": "treatment", "factors": ["contrastive"], "budget_matched": True,
                  "control_arm": "epidemic_llm_memory"},
    )


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
# The SEVERE severity point — a check on the diagnostic, not a second shot at a result
# ---------------------------------------------------------------------------------------------
# Taken a priori from the headroom surface computed before any treatment arm ran: total instrument
# failure at a higher price on intervention. If measured headroom bounds what adaptation is worth,
# then the measured advantage of an adaptive regent should be LARGER here, and roughly in the ratio
# the headroom predicts. A bound that does not track the thing it bounds is not a bound.

_SEVERE_CLAIM = (
    "the advantage an adaptive regent shows over the non-adaptive ceiling scales with the measured "
    "adaptation headroom of the regime"
)


def _severe_experiment(name: str, regent, harness: Harness, *, claim: str = _SEVERE_CLAIM,
                       metadata: dict | None = None) -> Experiment:
    return Experiment(
        name=name,
        system_factory=R.sir_factory(R.EPIDEMIC_SEVERE_SHOCKED),
        action_interface=_iface(),
        regents={"regent:0": regent},
        objectives={"regent:0": EpidemicLoss(lam=R.EPIDEMIC_SEVERE_LAMBDA,
                                             post_shock_step=R.EPIDEMIC_SHOCK_STEP)},
        harness=harness,
        schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY),
        seeds=list(SEEDS),
        horizon=R.EPIDEMIC_HORIZON,
        hypothesis=Hypothesis(
            id="H5-headroom-tracks-effect",
            claim=claim,
            baseline="the same arms on the milder `epidemic_*` regime; best_fixed anchors R=1",
            primary_metric="loss",
            falsification="the regent's advantage over best_fixed does NOT increase with headroom "
                          "across the two severity points",
        ),
        metadata={"regime": "epidemic_severe",
                  "decisions_per_run": R.EPIDEMIC_HORIZON // R.EPIDEMIC_DECIDE_EVERY,
                  **(metadata or {})},
    )


@register("severe_frozen")
def severe_frozen() -> Experiment:
    return _severe_experiment(
        "severe_frozen",
        MultiScriptedRegent(R.calibration()["epidemic_severe"]["frozen"]["laws"]),
        Harness([]), claim="the pre-break-optimal rule, held through a total instrument failure",
        metadata={"role": "reference:frozen"})


@register("severe_best_fixed")
def severe_best_fixed() -> Experiment:
    return _severe_experiment(
        "severe_best_fixed",
        MultiScriptedRegent(R.calibration()["epidemic_severe"]["best_fixed"]["laws"]),
        Harness([]), claim="the non-adaptive ceiling for the severe regime",
        metadata={"role": "reference:best_fixed"})


@register("severe_switching")
def severe_switching() -> Experiment:
    d = R.calibration()["epidemic_severe"]["switching"]
    return _severe_experiment(
        "severe_switching",
        SwitchingRegent("set_lockdown", d["pre_laws"], d["post_laws"],
                        R.EPIDEMIC_SHOCK_STEP),
        Harness([]), claim="the clairvoyant adaptor for the severe regime",
        metadata={"role": "reference:switching", "normalized_regret": 0.0})


def _register_severe_treatments() -> None:
    """The same three rungs as the cross-model panel: no harness / outcome only / all three."""
    rungs = {"bare": [], "outcome": ["outcome"], "trace_outcome_memory": ["trace", "outcome", "memory"]}
    for suffix, on in rungs.items():
        exp_name = f"severe_llm_{suffix}"

        def make(on=tuple(on), exp_name=exp_name) -> Experiment:
            max_tokens, extra = _llm_opts()
            return _severe_experiment(
                exp_name,
                LLMRegent(llm=_client(), model=_model(), temperature=0.0,
                          max_tokens=max_tokens, extra=extra),
                Harness([_FACTORS[n]() for n in on]),
                metadata={"role": "treatment", "factors": list(on), "budget_matched": True})

        register(exp_name)(make)


_register_severe_treatments()


# ---------------------------------------------------------------------------------------------
# The scalar NEGATIVE CONTROL (declared near-null: measured headroom ~1.14x)
# ---------------------------------------------------------------------------------------------

_SCALAR_NULL_NOTE = (
    "declared negative control: the best FIXED law in hindsight comes within ~4% of the "
    "clairvoyant adaptor in this regime, so the diagnostic "
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
            primary_metric="loss",
            falsification="an effect IS found here despite ~1.04x adaptation headroom, which would indicate "
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
