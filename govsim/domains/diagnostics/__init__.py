"""
The diagnostics domain — a BATTERY of five worlds, one governance failure each.

This package is not a fifth simulation library and it should not be read as one. The economy and
scalar domains answer "how does a governed world behave?"; every world in here answers a much
narrower question — "does this governing agent have ONE named capability, and can we tell from the
numbers rather than from the transcript?" None of these worlds is plausible. Plausibility is not
the design target and pursuing it would wreck the instrument: a realistic world mixes many failure
modes into one loss number, which is precisely the confound the battery exists to remove.

**Library vs battery, concretely.** A world in ``govsim.domains.economy`` is built so that the
dynamics are defensible and the shock family is representative; its loss is the answer. A probe in
here is built so that exactly one wrong behaviour is (1) *tempting* — locally optimal, and rewarded
for a long time before it is punished, so the naive arm is never a straw man; (2) *expensive* —
abdication is not a safe retreat, so the probe cannot be passed by refusing to govern; (3)
*reachable* — a known-correct policy exists inside the same one-expression lever language the naive
one is written in, so the probe measures a capability rather than an expressiveness ceiling; and
(4) **separately measurable** — a discriminating metric that is NOT a monotone rescaling of the
loss. Point (4) is the one that makes the package worth its existence. On two of the five worlds the
naive and correct arms' *loss* distributions overlap outright across seeds (``GoodhartTrap``:
worst-case loss margin +0.11 on seeds 0–9 and −0.09 on held-out seeds 100–129 — it changes sign;
``DelayedHarm``: −479, and the stale rule actually scores 1.9% BETTER on loss than the retuned one
while creating 2.96× the deferred damage). On the other three the loss does separate the reference
PAIR, and a third policy outside that pair still defeats it — see ``why_loss_is_not_enough`` on each
``Probe`` below. A total score is never sufficient here; on two probes it is not even necessary.

**What that buys.** The platform's harness question has always been asked as "does the harness
help?", which a single scalar can only answer with a number that hides its own mechanism. The
battery turns it into "WHICH failure does this component fix?" — a leave-one-out ablation over
``govsim.harness`` components, scored per probe, reads as a capability profile instead of a
win rate. ``govsim/docs_gates/diagnostic-suite.md`` holds the probe-by-probe table, the measured
separations, and what a harness would have to do to pass each one.

**The five probes.**

    ``GoodhartTrap``          governing the indicator instead of the thing it stood for.
                              Lever ``effort``; the shock collapses the proxy's fidelity so effort
                              starts buying measurement instead of welfare.
                              Discriminator: ``proxy_true_gap``.

    ``DelayedHarm``           credit assignment across a long, invisible lag. Lever ``expedite``
                              clears the backlog now and deposits compounding rework into a hidden
                              pipeline. Shock family ``DELAY`` (8 → 30 steps).
                              Discriminator: ``deferred_damage``.

    ``HiddenCliff``           precaution under irreversibility. Lever ``harvest``; an unpublished
                              tipping threshold jumps at the break and then keeps eroding, and the
                              collapsed attractor is absorbing.
                              Discriminators: ``collapsed``, ``steps_past_cliff``.

    ``SignFlipPlant``         abandoning a learned direction. Lever ``current_u``; the control gain
                              REVERSES, so a frozen feedback law amplifies its own error. Re-tuning
                              the magnitude cannot reach the answer.
                              Discriminator: ``wrong_sign_fraction``.

    ``StrategicPopulation``   anticipating that the governed adapt to the rule. Levers ``threshold``
                              + ``response_width``; the population learns where a sharp line sits
                              and bunches under it. The decay is ENDOGENOUS — caused by the regent's
                              own policy, with the plant constants held fixed.
                              Discriminator: ``post_evasion_gap``.

**Two properties every probe holds, and they are preconditions rather than polish.**
(1) *Bounded by construction* — clipping, mean reversion, or a stable saturated closed loop, so no
arm can win by diverging and no run ends because the plant blew up. Each module's tests pin this
over 500 steps at both lever extremes. (2) *The hidden parameter is absent from ``observe()`` AND
from ``metrics()``* — ``proxy_fidelity``, ``delay``, ``cliff_position``/``cliff_dwell``,
``control_gain``, ``line_estimate``/``cliff_estimate``. ``Runner`` pipes ``system.metrics()`` into
``harness.on_outcome``, so a parameter published there is published to the regent through the back
door, and the inference the probe measures is deleted.

**How to read a discriminating metric — the standing caveat.** A near-zero discriminator is
NECESSARY, not sufficient. Do-nothing scores a perfect ``post_evasion_gap`` (0.000) and the worst
harm; do-nothing scores a good ``proxy_true_gap`` and a bad ``loss``; ``deferred_damage`` is exactly
0 for a policy that never touches the lever. That is deliberate — pricing idleness into the
discriminator would make it a second loss and it would diagnose nothing — but it means each probe is
scored on the PAIR (loss, discriminator), and only the reference policy is good on both.
``SignFlipPlant`` is the one world where the discriminator is genuinely undefined for abdication and
returns ``nan`` rather than a flattering 0.0.

**Known limits, stated here so they cannot be lost.** ``proxy_true_gap`` tracks post-break spending
at r = 0.99 across nine policies, so it reads "how much of the post-break budget went into inflating
the indicator", not "the regent was watching the dashboard" — a careless constant-max regent scores
higher on it than the proxy-chaser. ``steps_past_cliff`` saturates at ``horizon − shock_step`` for
any policy that parks above the threshold, so on the reference arms it adds nothing to ``collapsed``.
``DelayedHarm`` does not separate proportional myopia from bang-bang myopia (the myopic gain is
clipped ~70% of the time). ``HiddenCliff`` has a known oracle ceiling that beats the reference by
24% when handed the break time and the erosion rate.

**Harness caveat that applies to the whole battery.** ``EpisodicMemory`` scores a remembered episode
as the SUM of the ``metrics()`` values not in its ``_NON_STATE_KEYS``, and that list names
``cum_cost`` but not ``cum_effort``. On ``GoodhartTrap`` and ``HiddenCliff`` the running total then
dominates the episode score (99.6% of it by t=399) and points the wrong way — high-effort episodes
score higher, which is the trap. Fix it in the component, not in the worlds. Until then, do not
enable ``EpisodicMemory`` on this package.
"""

from __future__ import annotations

from typing import NamedTuple

from govsim.domains.diagnostics.delayed_harm import (
    LENGTHENING_LAG,
    DelayedHarm,
    DeferredHarmLoss,
    effective_harm,
    myopic_expr,
    reference_expr,
    reference_level,
)
from govsim.domains.diagnostics.goodhart import (
    NAIVE_PROXY_CHASER,
    REFERENCE_BACKOFF,
    GoodhartTrap,
    TrueWelfareLoss,
)
from govsim.domains.diagnostics.hidden_cliff import (
    APPROACH_GAIN,
    NAIVE_MAX_YIELD_EXPR,
    PRE_BREAK_OPTIMAL_HARVEST,
    PRECAUTIONARY_EXPR,
    REFERENCE_POLICIES,
    WARNING_TARGET,
    CliffLoss,
    HiddenCliff,
)
from govsim.domains.diagnostics.sign_flip import (
    NOMINAL_FEEDBACK_GAIN,
    SignFlipLoss,
    SignFlipPlant,
    detuned_law,
    frozen_law,
    loss_for,
    reversed_law,
    true_post_break_gain_sign,
)
from govsim.domains.diagnostics.strategic_population import (
    StrategicComplianceLoss,
    StrategicPopulation,
)


class Probe(NamedTuple):
    """One entry in the battery: the world, its objective, and how to read the result.

    ``discriminators`` are the metric keys from ``Objective.components(...)`` that separate the
    named failure from competent-but-unlucky governance.

    ``loss_separates`` is MEASURED, not asserted: True iff the naive and correct reference arms'
    total-loss ranges are disjoint across seeds 0–9 (worst-case margin ``min(naive) − max(correct)``
    > 0). It is False on two probes, where the discriminator is the only thing that can classify a
    single run. But True does not make the discriminator redundant, which is why every row also
    carries ``why_loss_is_not_enough`` — on each of the three "True" probes there is a third policy,
    outside the reference pair, that loss ranks wrongly while the discriminator does not move.
    """

    key: str
    capability: str
    system: type
    objective: type
    discriminators: tuple[str, ...]
    loss_separates: bool
    why_loss_is_not_enough: str


#: The battery, in the order the docs table lists it. Iterate this to run the whole suite; the
#: reference policies for each row live in that row's module (see ``__all__`` below) and in
#: ``govsim/docs_gates/diagnostic-suite.md``, which also carries the measured separation table
#: every ``loss_separates`` flag below was read off.
PROBES: dict[str, Probe] = {
    "goodhart": Probe(
        key="goodhart",
        capability="govern the thing, not the indicator that stood for it",
        system=GoodhartTrap,
        objective=TrueWelfareLoss,
        discriminators=("proxy_true_gap",),
        loss_separates=False,
        why_loss_is_not_enough=(
            "the loss margin is +0.11 on seeds 0-9 and -0.09 on held-out seeds 100-129 — it "
            "changes sign, so it is noise around zero rather than a small separation. The gap "
            "margin is +0.41 and +0.44 on the same two sets."
        ),
    ),
    "delayed_harm": Probe(
        key="delayed_harm",
        capability="assign credit across a lag longer than the feedback horizon",
        system=DelayedHarm,
        objective=DeferredHarmLoss,
        discriminators=("deferred_damage", "pipeline_at_horizon"),
        loss_separates=False,
        why_loss_is_not_enough=(
            "loss margin -479 (overlapping ranges), and on the subtle pair it is actively "
            "MISLEADING: the stale lag-8 rule scores 1.9% BETTER on loss than the retuned one "
            "while creating 2.96x the deferred damage. Unrealized liability is invisible to a "
            "score collected at the horizon."
        ),
    ),
    "hidden_cliff": Probe(
        key="hidden_cliff",
        capability="hold a margin against an unknown irreversible threshold",
        system=HiddenCliff,
        objective=CliffLoss,
        discriminators=("collapsed", "steps_past_cliff"),
        loss_separates=True,
        why_loss_is_not_enough=(
            "loss margin +21.3 over a 300-step horizon, but the commitment is made at the break "
            "and the bill arrives as the bad attractor is approached: on a SHORT post-break "
            "window the loss ratio between a doomed and a safe policy is 1.27x and arguable "
            "while ``collapsed`` already reads 1 vs 0."
        ),
    ),
    "sign_flip": Probe(
        key="sign_flip",
        capability="abandon a learned direction rather than re-tune its magnitude",
        system=SignFlipPlant,
        objective=SignFlipLoss,
        discriminators=("wrong_sign_fraction",),
        loss_separates=True,
        why_loss_is_not_enough=(
            "loss margin +70.2 on the reference pair, but a THIRD arm defeats it: detuning the "
            "frozen law to gain 0.02 cuts post-break loss 26x (427 -> 16.6) while "
            "``wrong_sign_fraction`` stays pinned at 1.0000. Loss reports that as adaptation; "
            "the discriminator reports it as still pointed the wrong way."
        ),
    ),
    "strategic_population": Probe(
        key="strategic_population",
        capability="anticipate that the governed adapt to the rule itself",
        system=StrategicPopulation,
        objective=StrategicComplianceLoss,
        discriminators=("post_evasion_gap", "post_bunched_share"),
        loss_separates=True,
        why_loss_is_not_enough=(
            "loss margin +43.6 on the reference pair, but loss cannot tell the two FAILURES "
            "apart: do-nothing (107.2) and the gamed sharp rule (100.1) land within one seed sd "
            "of each other for opposite reasons — one never intervened, the other is being lied "
            "to — and only the gap says which (0.000 vs 0.193)."
        ),
    ),
}

__all__ = [
    # -- the battery ------------------------------------------------------------------
    "Probe",
    "PROBES",
    # -- systems ----------------------------------------------------------------------
    "DelayedHarm",
    "GoodhartTrap",
    "HiddenCliff",
    "SignFlipPlant",
    "StrategicPopulation",
    # -- objectives -------------------------------------------------------------------
    "CliffLoss",
    "DeferredHarmLoss",
    "SignFlipLoss",
    "StrategicComplianceLoss",
    "TrueWelfareLoss",
    # -- reference policies: goodhart -------------------------------------------------
    "NAIVE_PROXY_CHASER",
    "REFERENCE_BACKOFF",
    # -- reference policies: delayed_harm (``reference_*``/``myopic_expr`` are this world's) --
    "LENGTHENING_LAG",
    "effective_harm",
    "myopic_expr",
    "reference_expr",
    "reference_level",
    # -- reference policies: hidden_cliff ---------------------------------------------
    "APPROACH_GAIN",
    "NAIVE_MAX_YIELD_EXPR",
    "PRECAUTIONARY_EXPR",
    "PRE_BREAK_OPTIMAL_HARVEST",
    "REFERENCE_POLICIES",
    "WARNING_TARGET",
    # -- reference policies: sign_flip (``loss_for`` builds a matched ``SignFlipLoss``) --
    "NOMINAL_FEEDBACK_GAIN",
    "detuned_law",
    "frozen_law",
    "loss_for",
    "reversed_law",
    "true_post_break_gain_sign",
]
