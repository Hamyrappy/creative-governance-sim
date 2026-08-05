"""
The economy library's runnable arms: five worlds x (three key-free references + one LLM regent).

**What this module is for.** ``govsim/domains/economy`` ships five governed worlds; this file turns
each of them into arms the Runner can execute and the analysis layer can pair. The point of a
*library* rather than a flagship is that the platform's central claim — only a shock to an
instrument's efficacy leaves adaptation headroom, a shock to the governed state does not — was
measured on one epidemic. Five unrelated mechanics under the same ``ShockKind`` is what turns that
into a statement about the shock family instead of a statement about SIR dynamics.

**Every world gets the same three references, and they are references rather than decoration.**

    ``*_standing_rule``  a plausible institution: the rule a competent authority would actually
                         legislate in that world (a Taylor rule, a constant-escapement harvest
                         rule, an order-up-to policy). Feedback, and tuned to the pre-shock world.
    ``*_do_nothing``     the passive floor. Note that "do nothing" is world-specific and is NOT
                         always the same as "set the lever to zero": in the monetary world it is
                         the neutral rate (a passive authority holds the textbook stance, and
                         rate zero would be a maximal stimulus, not an abstention); in the commons
                         it is the permissive quota ceiling, i.e. open access, which is the losing
                         arm rather than a free pass.
    ``*_max_lever``      maximal intervention: spend the instrument as hard as its range allows.
                         In the commons the quota lever runs *backwards* (a higher quota is a
                         weaker restriction), so maximal intervention pins the quota at 0 and the
                         reserve at its ceiling.

The two extremes are there because a *corner* optimum is regime-invariant: no shock can move it, so
a world whose best policy sits at one of these has zero adaptation headroom by construction and
will report a null whatever governs it. Bracketing the standing rule between both corners is the
cheapest available check that the optimum is interior, and it is a check that has to be re-run
whenever a price moves. ``scripts/headroom_audit.py`` is the full version of the same question.

**The LLM arm** (``*_llm``) carries the full rollout-free harness — trace feedback, outcome
feedback, episodic memory — and decides on the same ``EveryN`` schedule as every other arm, so it
spends a fixed and world-independent call budget. It uses the identical
``_client()``/``_llm_opts()``/``_model()`` env-var plumbing as ``governance_experiments``, so a
cached tape replays key-free and a live run needs only ``OPENAI_BASE_URL``/``OPENAI_MODEL``.

**Shocks are expressed as ``govsim.scenarios.Scenario`` values**, never as inline dicts. That is
what makes "the same perturbation applied to all of them" auditable: each world names which of its
parameters plays the instrument-efficacy role, the scenario names the family, and a reader can diff
the five and see that four of them are the same experiment in different clothes — and that the
supply chain's is deliberately not, because ``DELAY`` was the family this platform had never
measured. It is measured now, and it came back near-null; see the table below.

**These arms are registered, not blessed.** Whether any of these worlds can host an adaptation
experiment is an empirical question answered by ``scripts/headroom_audit.py``, and the answer for
some of them is no. Registering an arm in a headroom-free world is fine and useful — it is how a
declared negative control gets reported — but the hypothesis text below says which is which, and
the audit's number is what settles it.

**MEASURED** (``uv run python scripts/headroom_audit.py --seeds 5``, on the post-shock window):

    monetary      1.98x   the strongest world in the whole library, epidemic included
    opinion       1.46x   real, and inside the narrow band the world's own sweep predicted
    commons       ratio undefined, absolute gap 2.57 on a frozen loss of 2.42 — the oracle
                          scores NEGATIVE (net welfare), so the ratio is meaningless here and the
                          gap is what to read. The oracle substitutes quota -> reserve exactly as
                          designed.
    supply_chain  1.06x   near-null. This is a RESULT, not a disappointment: DELAY was the one
                          unmeasured family in ``govsim/scenarios.py`` and it now has a number.
    fiscal        1.00x   DEGENERATE — cannot host an adaptation experiment. See below.

The fiscal world's null has a specific and checkable cause, and it is not "the shock was too mild".
The optimal fiscal policy in this configuration is ``tax_rate = 0``: the government reaches
potential output by running early deficits into household balances and then living off
``alpha_wealth * M_h``, so taxation only destroys wealth and distorts consumption. And
``tau_eff = tax_compliance * tax_rate``, so at ``tax_rate = 0`` a compliance collapse multiplies
zero — the shock is not merely small, it is *arithmetically inert*. The audit says so plainly:
frozen and oracle are the identical policy. Fixing this is a change to the world's prices (the
world would have to make taxation worth enacting), which belongs to whoever owns
``fiscal_sfc.py``; the arms are registered here as a declared null so the fact is reported rather
than quietly dropped.
"""

from __future__ import annotations

import itertools

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from govsim.core.experiment import Experiment, Hypothesis
from govsim.core.harness import Harness
from govsim.core.llm import CachingReplayClient, OpenAICompatClient
from govsim.core.objective import Objective
from govsim.core.regent import MultiScriptedRegent, ScriptedRegent
from govsim.regents.baselines import SwitchingRegent
from govsim.core.schedule import EveryN
from govsim.domains.economy import (
    CommonsEconomy,
    CommonsWelfare,
    DualMandateLoss,
    FiscalSFCEconomy,
    FiscalStabilizationLoss,
    MonetaryEconomy,
    OpinionPolity,
    PolarizationLoss,
    SupplyChainCost,
    SupplyChainEconomy,
)
from govsim.domains.scalar import Lever, ScalarLeverInterface
from govsim.experiments import register
from govsim.harness import EpisodicMemory, OutcomeFeedback, TraceFeedback
from govsim.regents import LLMRegent
from govsim.scenarios import Scenario, ShockKind

#: The stats-protocol floor for a headline paired comparison (``govsim/docs_gates/stats-protocol.md``).
#: Held identical across all five worlds so a cross-world table pairs seed-for-seed.
SEEDS = list(range(20))


# =============================================================================================
# LLM plumbing — byte-for-byte the ``governance_experiments`` pattern, so one env var configures
# every arm in the project and a cache written by one suite is readable by the other.
# =============================================================================================


def _client() -> CachingReplayClient:
    inner = OpenAICompatClient(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key_env=os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY"),
        drop_params=frozenset(
            p.strip() for p in os.environ.get("GOVSIM_LLM_DROP_PARAMS", "").split(",") if p.strip()
        ),
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


# =============================================================================================
# The scenario catalogue for this library
# =============================================================================================
# Four of the five are the SAME family — INSTRUMENT_EFFICACY — deliberately, because that is the
# only family this platform has measured as leaving headroom, and a cross-world claim about it
# needs the worlds to differ in their mechanics and not in their break. The supply chain's is the
# exception on purpose: DELAY has never been measured here, and a library with only the family we
# already believe in cannot discover that we were wrong about the others.

MONETARY_SHOCK = Scenario(
    name="monetary_transmission_collapse",
    kind=ShockKind.INSTRUMENT_EFFICACY,
    step=100,
    params={"transmission": 0.10},
    story="Monetary transmission breaks: the policy rate stops reaching borrowing decisions while "
          "costing the authority exactly what it always cost. The rate the central bank sets is "
          "still on every front page; it simply no longer moves demand.",
)

FISCAL_SHOCK = Scenario(
    name="fiscal_compliance_collapse",
    kind=ShockKind.INSTRUMENT_EFFICACY,
    step=100,
    params={"tax_compliance": 0.25},
    story="Tax collection fails. The statutory rate keeps distorting every household's choices — "
          "the wedge is a function of the enacted rate, not of the collection rate — while three "
          "quarters of the revenue stops arriving.",
)

COMMONS_SHOCK = Scenario(
    name="commons_enforcement_collapse",
    kind=ShockKind.INSTRUMENT_EFFICACY,
    step=100,
    params={"enforcement_efficacy": 0.15},
    story="Enforcement collapses: the fleet stops respecting the quota. The season is still closed "
          "on paper and the political price is still paid, and the catch is now whatever the "
          "fleet wants. The reserve still binds, because geography does not need obedience.",
)

SUPPLY_CHAIN_SHOCK = Scenario(
    name="supply_chain_delay_lengthens",
    kind=ShockKind.DELAY,
    step=100,
    params={"lead_time": 6},
    story="Shipping times triple with no announcement. Every order still costs what it cost, and "
          "the correction placed today now lands four periods later than the standing rule "
          "believes — so corrections pile onto corrections. This is the bullwhip.",
)

OPINION_SHOCK = Scenario(
    name="opinion_moderation_collapse",
    kind=ShockKind.INSTRUMENT_EFFICACY,
    step=120,
    params={"moderation_efficacy": 0.05},
    story="The audience leaves for platforms the authority does not reach. The moderation drive is "
          "funded, staffed and billed exactly as before, and almost none of it lands.",
)


# =============================================================================================
# World specs
# =============================================================================================


@dataclass(frozen=True)
class _World:
    """One world's complete wiring, plus the three reference laws and the hypothesis text.

    Held as data and registered in a loop rather than written out as twenty near-identical
    factories. Twenty hand-written factories is how one of them silently ends up with a different
    seed list, horizon or shock step — and a cross-world headroom table built on arms that differ
    in their horizon is not a cross-world table at all, it is five unrelated numbers in a grid.
    """

    key: str
    system_cls: type
    base: dict[str, Any]
    scenario: Scenario
    levers: list[Lever]
    objective: Callable[[], Objective]
    horizon: int
    decide_every: int
    #: role -> {verb: sandboxed expression}. Roles are fixed: standing_rule / do_nothing / max_lever.
    laws: dict[str, dict[str, str]]
    #: What each reference is, in one line, for the hypothesis text and the ``list`` output.
    role_notes: dict[str, str]
    claim: str
    hypothesis_id: str
    headroom_note: str
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The three CALIBRATED references, pinned from ``scripts/decompose_library.py``. Present only
    #: for worlds that have actually been decomposed; a world without them registers no ``R`` arms,
    #: which is the honest outcome — ``R`` is meaningless without a measured frozen/switching pair.
    #:
    #: Pinned rather than read from ``logs/decomposition.json`` at import time so that a run is
    #: reproducible from the source tree alone, and so a re-calibration cannot silently move the
    #: denominator of every published ``R``. ``tests/test_economy_anchors.py`` re-derives them and
    #: fails if the world has drifted from what they were calibrated on.
    calibrated: dict[str, dict[str, str]] = field(default_factory=dict)
    #: The step at which ``switching`` is allowed to change its mind (== the scenario's break).
    switch_step: int | None = None

    @property
    def shocked_config(self) -> dict[str, Any]:
        return self.scenario.applied_to(self.base)

    def factory(self) -> Callable[[int], Any]:
        cfg = self.shocked_config
        cls = self.system_cls

        def make(seed: int):
            system = cls(dict(cfg))
            system.reset(seed)  # __init__ resets at seed 0; the Runner's seed is the real one
            return system

        return make

    @property
    def baseline_text(self) -> str:
        """The NAMED rivals, spelled out. The Runner gate wants a name, not an adjective."""
        return "; ".join(
            f"{self.key}_{role} ({note})" for role, note in self.role_notes.items()
        )


ECONOMY_HORIZON = 200
ECONOMY_DECIDE_EVERY = 10  # => 20 decisions per run, the project-wide per-arm LLM call budget


WORLDS: list[_World] = [
    # -----------------------------------------------------------------------------------------
    _World(
        key="monetary",
        system_cls=MonetaryEconomy,
        base={},  # the module's calibrated defaults; see MonetaryEconomy's headroom table
        scenario=MONETARY_SHOCK,
        levers=[Lever("set_policy_rate", (0.0, 12.0), "policy_rate",
                      "The nominal policy rate in percentage points. Its floor is a zero lower "
                      "bound. Billed on the level you set, every step.")],
        # lam=0.25 is not a default: MonetaryEconomy's docstring table reports adaptation headroom
        # of 1.47x at (lam=0.25, transmission 0.10) and only 1.09x at lam=0.10. A cheap instrument
        # makes abandoning it worthless, and a world where the shock cannot move the optimum
        # reports a null no matter which regent governs it.
        objective=lambda: DualMandateLoss(lam=0.25, post_shock_step=MONETARY_SHOCK.step),
        horizon=ECONOMY_HORIZON,
        decide_every=ECONOMY_DECIDE_EVERY,
        laws={
            # A Taylor-type rule: lean against inflation misses and the output gap from the neutral
            # rate. Gains kept modest because at lam=0.25 a hard-leaning rule pays more for the
            # stance than the stabilization is worth — which is itself the interior optimum showing.
            "standing_rule": {"set_policy_rate":
                              "r_star + inflation + 0.5 * (inflation - inflation_target) "
                              "+ 0.25 * output_gap"},
            # NOT rate zero. Zero is maximal stimulus; the passive stance is the neutral rate.
            "do_nothing": {"set_policy_rate": "r_star + inflation_target"},
            "max_lever": {"set_policy_rate": "12.0"},
        },
        role_notes={
            "standing_rule": "a Taylor rule leaning on inflation and the gap",
            "do_nothing": "hold the neutral rate forever, never lean",
            "max_lever": "the ceiling rate, held forever",
        },
        # MEASURED by scripts/decompose_library.py --seeds 8 (top_k=24, converged against an
        # exhaustive search). frozen 2980.56 / best_fixed 2597.61 / switching 1910.81, giving
        # adaptation 1.359x x robustness 1.147x = staleness 1.560x — the largest ADAPTATION headroom
        # in the library, epidemic included.
        #
        # Read the three laws together, because the substance of the result is in their shapes.
        # The frozen rule leans hard (gain 1.0) from a high intercept. The best FIXED response to a
        # transmission collapse is to abandon feedback entirely (gain 0.0) and sit at a constant.
        # The clairvoyant ADAPTOR does neither: it keeps leaning at half strength and drops the
        # intercept to the zero lower bound. Adapting here means changing the LEVEL while keeping
        # the rule, which is a response no re-tuning of the frozen rule's threshold can express.
        calibrated={
            "frozen": {"set_policy_rate":
                       "5.0 + 1.0 * (inflation - inflation_target + 0.5 * output_gap)"},
            "best_fixed": {"set_policy_rate":
                           "6.0 + 0.0 * (inflation - inflation_target + 0.5 * output_gap)"},
            "switching_pre": {"set_policy_rate":
                              "6.0 + 0.5 * (inflation - inflation_target + 0.5 * output_gap)"},
            "switching_post": {"set_policy_rate":
                               "0.0 + 0.5 * (inflation - inflation_target + 0.5 * output_gap)"},
        },
        switch_step=MONETARY_SHOCK.step,
        claim="under an unobservable collapse of monetary transmission, an adaptive regent lowers "
              "full-horizon dual-mandate loss relative to the Taylor institution that was tuned "
              "before the break — by ceasing to pay for a disconnected instrument, which is a "
              "response no re-threshold of the same rule can express",
        hypothesis_id="H-econ-monetary-transmission",
        headroom_note="MEASURED 1.98x (5 seeds) — the largest in the library, epidemic included. "
                      "The Lucas critique's own case. A feedback rule answers a transmission "
                      "collapse by leaning HARDER — the boom runs, the rule reads the boom, the "
                      "rate rises, and every point of it now buys a tenth of what it did at the "
                      "same price. The oracle drops the intercept from 5.0 to 0.0 and halves the "
                      "gain; the frozen rule keeps paying full price for a disconnected lever.",
        metadata={"shock_kind": ShockKind.INSTRUMENT_EFFICACY.value, "lam": 0.25,
                  "measured_headroom": 1.98, "can_host_adaptation": True},
    ),
    # -----------------------------------------------------------------------------------------
    _World(
        key="fiscal",
        system_cls=FiscalSFCEconomy,
        base={},
        scenario=FISCAL_SHOCK,
        levers=[
            Lever("set_tax_rate", (0.0, 0.6), "tax_rate",
                  "The statutory tax rate. It distorts the economy — and is billed to you — "
                  "whether or not the revenue ever arrives."),
            Lever("set_transfer", (0.0, 40.0), "transfer",
                  "Government transfer paid out each period, on top of base spending. Capped by "
                  "the debt ceiling you cannot spend past."),
        ],
        objective=lambda: FiscalStabilizationLoss(lam=0.2, post_shock_step=FISCAL_SHOCK.step),
        horizon=ECONOMY_HORIZON,
        decide_every=ECONOMY_DECIDE_EVERY,
        laws={
            # Automatic stabilizer: transfer scales with the output shortfall, funded by a modest
            # flat rate near the calibrated interior optimum (tax 0.10, transfer ~10).
            "standing_rule": {"set_tax_rate": "0.10",
                              "set_transfer": "min(40.0, 10.0 + 0.25 * (y_potential - output))"},
            "do_nothing": {"set_tax_rate": "0.0", "set_transfer": "0.0"},
            "max_lever": {"set_tax_rate": "0.6", "set_transfer": "40.0"},
        },
        role_notes={
            "standing_rule": "flat 10% rate + a transfer that scales with the output shortfall",
            "do_nothing": "no tax, no transfer — the economy runs on private wealth alone",
            "max_lever": "the tax ceiling and the transfer ceiling, both held forever",
        },
        claim="under an unobservable collapse of tax compliance, an adaptive regent lowers "
              "full-horizon fiscal stabilization loss relative to the automatic stabilizer tuned "
              "before the break — because the enacted rate keeps distorting at full strength while "
              "its yield is gone, so the correct response is to stop enacting it",
        hypothesis_id="H-econ-fiscal-compliance",
        headroom_note="MEASURED 1.00x (5 seeds), frozen and oracle the IDENTICAL policy — this "
                      "world CANNOT host an adaptation experiment as priced. The optimum is "
                      "tax_rate = 0 (the economy reaches potential off accumulated household "
                      "wealth, so taxing only distorts), and tau_eff = tax_compliance * tax_rate, "
                      "so a compliance collapse multiplies zero. The shock is arithmetically "
                      "inert, not merely mild. Registered as a DECLARED NULL.",
        metadata={"shock_kind": ShockKind.INSTRUMENT_EFFICACY.value, "lam": 0.2,
                  "measured_headroom": 1.00, "can_host_adaptation": False,
                  "role_note": "declared null: the shocked parameter multiplies a lever the "
                               "optimal policy sets to zero"},
    ),
    # -----------------------------------------------------------------------------------------
    _World(
        key="commons",
        system_cls=CommonsEconomy,
        base={},
        scenario=COMMONS_SHOCK,
        levers=[
            Lever("set_quota", (0.0, 0.5), "quota",
                  "The legally allowed catch per period. Cheap and precise — and worth exactly "
                  "the compliance behind it. You are billed for how far below the ceiling you set "
                  "it, obeyed or not."),
            Lever("set_reserve", (0.0, 0.8), "reserve",
                  "The fraction of the stock placed in a physically inaccessible refuge. Blunt and "
                  "expensive, and it binds through geography rather than through obedience."),
        ],
        objective=lambda: CommonsWelfare(lam=1.0, post_shock_step=COMMONS_SHOCK.step),
        horizon=ECONOMY_HORIZON,
        decide_every=ECONOMY_DECIDE_EVERY,
        laws={
            # Constant-escapement: take half of whatever sits above 40% of carrying capacity, and
            # keep a fifth of the water closed as insurance. This is what fisheries management
            # actually looks like, and it is feedback on the one observable that matters.
            "standing_rule": {"set_quota": "max(0.0, 0.5 * (stock - 0.4 * capacity))",
                              "set_reserve": "0.2"},
            # Open access. In a world where fleet capacity exceeds the growth rate this is the
            # LOSING arm, not a neutral null — which is the whole content of "tragedy".
            "do_nothing": {"set_quota": "0.5", "set_reserve": "0.0"},
            # Maximal INTERVENTION, which for the quota means its minimum: the quota lever runs
            # backwards (a higher quota is a weaker restriction).
            "max_lever": {"set_quota": "0.0", "set_reserve": "0.8"},
        },
        role_notes={
            "standing_rule": "constant-escapement harvest rule + a 20% reserve",
            "do_nothing": "open access: the permissive quota ceiling and no closed water",
            "max_lever": "total ban plus the maximum reserve — maximal restriction",
        },
        claim="under an unobservable collapse of quota enforcement, an adaptive regent lowers "
              "full-horizon commons loss relative to the escapement rule tuned before the break — "
              "by SUBSTITUTING to the instrument that does not need obedience, which is a "
              "different lever rather than a different threshold on the broken one",
        hypothesis_id="H-econ-commons-enforcement",
        headroom_note="MEASURED as an absolute gap of 2.57 against a frozen loss of 2.42; the "
                      "RATIO is undefined because CommonsWelfare is net welfare and the oracle "
                      "scores below zero (-0.15). The substitution works exactly as designed: the "
                      "oracle moves the reserve 0.0 -> 0.8 and stops pretending the quota binds. "
                      "Collapse is partly irreversible, so a stale rule is expensive rather than "
                      "merely suboptimal.",
        metadata={"shock_kind": ShockKind.INSTRUMENT_EFFICACY.value, "lam": 1.0,
                  "measured_headroom_gap": 2.57, "measured_headroom": None,
                  "can_host_adaptation": True,
                  "role_note": "report the GAP, never the ratio: this objective can go negative"},
    ),
    # -----------------------------------------------------------------------------------------
    _World(
        key="supply_chain",
        system_cls=SupplyChainEconomy,
        base={"lead_time": 2},
        scenario=SUPPLY_CHAIN_SHOCK,
        levers=[Lever("set_order", (0.0, 40.0), "order",
                      "Units to order this period. They arrive after a delivery lag you are not "
                      "told, and you are billed for the order you place, not for what turns up.")],
        objective=lambda: SupplyChainCost(post_shock_step=SUPPLY_CHAIN_SHOCK.step),
        horizon=ECONOMY_HORIZON,
        decide_every=ECONOMY_DECIDE_EVERY,
        laws={
            # Order-up-to: bring the inventory position to three periods of recent demand. The
            # textbook policy, and one whose target level is implicitly a function of the lead time
            # — which is exactly why a delay shock is the interesting break to hand it.
            "standing_rule": {"set_order": "max(0.0, 3.0 * recent_demand + backlog - inventory)"},
            "do_nothing": {"set_order": "0.0"},
            "max_lever": {"set_order": "40.0"},
        },
        role_notes={
            "standing_rule": "order-up-to three periods of recent demand",
            "do_nothing": "never order; the warehouse drains and the backlog queue runs",
            "max_lever": "order the cap every period, forever",
        },
        claim="under an unannounced lengthening of the delivery lag, an adaptive regent lowers "
              "full-horizon inventory cost relative to the order-up-to rule tuned to the old lag — "
              "the standing rule's GAIN is wrong rather than its threshold, so corrections land "
              "late and stack on one another",
        hypothesis_id="H-econ-supply-chain-delay",
        headroom_note="MEASURED 1.06x (5 seeds) — NEAR-NULL, and that is the finding. This is the "
                      "library's one DELAY arm, and DELAY was listed in govsim/scenarios.py as NOT "
                      "YET MEASURED. It now has a number, and the number says a lengthened lead "
                      "time behaves like a state shock: an order-up-to rule with a free target "
                      "absorbs it by ordering more, and the oracle differs from the frozen rule "
                      "only in that target. Treat as a declared negative control until a delay "
                      "regime with a genuinely mis-tuned gain is found. Checked for seed "
                      "sensitivity before believing it: 1.06x / 1.07x / 1.06x at 5 / 8 / 12 seeds. "
                      "At 2 seeds it reports 2.59x — too few seeds to pick a stable frozen rule, "
                      "and a reminder that a headroom number from a smoke run is not a result.",
        metadata={"shock_kind": ShockKind.DELAY.value, "measured_headroom": 1.06,
                  "can_host_adaptation": False,
                  "role_note": "declared near-null: the first measurement of the DELAY family"},
    ),
    # -----------------------------------------------------------------------------------------
    _World(
        key="opinion",
        system_cls=OpinionPolity,
        base={},
        scenario=OPINION_SHOCK,
        levers=[Lever("set_moderation", (0.0, 1.0), "moderation",
                      "Effort spent widening who hears whom and funding a neutral common ground. "
                      "Billed on the effort you order, not on the mixing it achieves.")],
        objective=lambda: PolarizationLoss(lam=0.2, post_shock_step=OPINION_SHOCK.step),
        # 240, not 200: the break lands at 120 here (the polity needs longer to reach its
        # equilibrium than the other worlds), so a 200-step horizon would leave the post-shock
        # window less than half the length of everyone else's.
        horizon=240,
        decide_every=ECONOMY_DECIDE_EVERY,
        laws={
            # The calibrated best CONSTANT lever from OpinionPolity's own sweep (16 seeds, H=240):
            # 0.4, with never-moderating 4.0x worse and always-maximum 1.8x worse. A threshold rule
            # scores worse here, because polarization rises monotonically until the instrument
            # bites and a threshold on a rising observable just pins itself to its high branch.
            "standing_rule": {"set_moderation": "0.4"},
            "do_nothing": {"set_moderation": "0.0"},
            "max_lever": {"set_moderation": "1.0"},
        },
        role_notes={
            "standing_rule": "the calibrated best constant moderation level (0.4)",
            "do_nothing": "never moderate; the polity hollows out and splits into two blocs",
            "max_lever": "maximum moderation every step, forever",
        },
        claim="under an unobservable collapse of moderation efficacy, an adaptive regent lowers "
              "full-horizon polarization loss relative to the calibrated constant institution — by "
              "spending NOTHING, which is the one response a threshold on a rising observable "
              "cannot produce",
        hypothesis_id="H-econ-opinion-moderation",
        headroom_note="MEASURED 1.46x (5 seeds) — real, and the second-strongest in the library. "
                      "Non-economic mechanics under an economic mandate, so a result here is not a "
                      "result about money. One caveat worth carrying: the oracle sits AT the top "
                      "of the lever range (constant 1.0), i.e. at a corner of the family, so the "
                      "1.46x is a lower bound on what a wider family would find and the corner "
                      "should be re-checked if the price moves. Relatedly, opinion_max_lever "
                      "slightly BEATS opinion_standing_rule over the shocked full horizon "
                      "(61.6 vs 65.4, 2 seeds) even though 0.4 is verifiably the pre-shock optimum "
                      "(re-measured: 29.83 at 0.4, 120.45 at 0.0, 52.89 at 1.0 over 16 seeds, "
                      "reproducing OpinionPolity's own sweep). Polarization is a STOCK, so the "
                      "post-break leg inherits the state the pre-break leg left it in, and "
                      "over-moderating early pays after the instrument dies. That is robustness "
                      "headroom, not adaptation headroom — a different standing rule, no "
                      "adaptation required — and the two must not be reported as one number.",
        metadata={"shock_kind": ShockKind.INSTRUMENT_EFFICACY.value, "lam": 0.2,
                  "measured_headroom": 1.46, "can_host_adaptation": True,
                  "role_note": "oracle sits at the family's upper corner — 1.46x is a lower bound"},
    ),
]

WORLDS_BY_KEY = {w.key: w for w in WORLDS}


# =============================================================================================
# Registration
# =============================================================================================


def _experiment(world: _World, name: str, regent, harness: Harness, *, claim: str,
                falsification: str, metadata: dict[str, Any]) -> Experiment:
    """Every arm of a world, wired identically apart from the regent and the harness.

    Identical wiring is not tidiness. The system factory, interface, objective, schedule, seeds and
    horizon are built from the SAME ``_World``, so any difference the analysis layer measures
    between two arms of a world is a difference between the regent and the harness, and cannot be a
    difference between two subtly different plants.
    """
    return Experiment(
        name=name,
        system_factory=world.factory(),
        action_interface=ScalarLeverInterface(list(world.levers)),
        regents={"regent:0": regent},
        objectives={"regent:0": world.objective()},
        harness=harness,
        schedule=EveryN(world.decide_every),
        seeds=list(SEEDS),
        horizon=world.horizon,
        hypothesis=Hypothesis(
            id=world.hypothesis_id,
            claim=claim,
            baseline=world.baseline_text,
            # The FULL horizon, for the reason ``governance_experiments`` documents at length:
            # scoring only the post-shock window rewards passivity, because a break that disables
            # an instrument makes "never intervene" close to right afterwards — so a policy that
            # never adapted to anything scores well. No fixed law is optimal on both sides of a
            # break that moves the optimum, so the full horizon removes that free lunch.
            # ``post_loss`` is emitted by every objective here and is reported alongside.
            primary_metric="loss",
            falsification=falsification,
        ),
        metadata={
            "regime": world.key,
            "shock": world.scenario.name,
            "shock_kind": world.scenario.kind.value,
            "shock_step": world.scenario.step,
            "decisions_per_run": world.horizon // world.decide_every,
            "headroom_note": world.headroom_note,
            **world.metadata,
            **metadata,
        },
    )


def _reference_regent(laws: dict[str, str]):
    """``ScriptedRegent`` for a one-lever world, ``MultiScriptedRegent`` for the rest.

    Not cosmetic. A reference confined to one instrument in a world whose correct answer to a
    broken instrument is "use the other one" cannot express the response the experiment is about,
    so it is not a weaker opponent — it is an unfair one, and every arm beats it for the wrong
    reason.
    """
    if len(laws) == 1:
        (verb, expr), = laws.items()
        return ScriptedRegent(verb=verb, expr=expr)
    return MultiScriptedRegent(laws)


def _register_world(world: _World) -> None:
    for role, laws in world.laws.items():
        name = f"{world.key}_{role}"

        def make_reference(world=world, role=role, laws=laws, name=name) -> Experiment:
            return _experiment(
                world, name, _reference_regent(laws), Harness([]),
                claim=f"this IS the named '{role}' reference for the {world.key} world: "
                      f"{world.role_notes[role]}",
                falsification="n/a — key-free reference arm, not a claim",
                metadata={"role": f"reference:{role}", "laws": dict(laws)},
            )

        register(name)(make_reference)

    # -- the CALIBRATED references: the denominators of R ----------------------------------------
    # Registered only where a decomposition exists. Without a measured (frozen, switching) pair
    # there is no adaptation budget to normalize against, and an R computed anyway would be a
    # number with no referent.
    if world.calibrated and world.switch_step is not None:
        for role in ("frozen", "best_fixed"):
            name = f"{world.key}_{role}"

            def make_calibrated(world=world, role=role, name=name) -> Experiment:
                laws = world.calibrated[role]
                return _experiment(
                    world, name, _reference_regent(laws), Harness([]),
                    claim=f"the CALIBRATED '{role}' reference for {world.key}, from "
                          f"scripts/decompose_library.py",
                    falsification="n/a — key-free calibrated reference, not a claim",
                    metadata={"role": f"calibrated:{role}", "laws": dict(laws)},
                )

            register(name)(make_calibrated)

        def make_switching(world=world) -> Experiment:
            pre, post = world.calibrated["switching_pre"], world.calibrated["switching_post"]
            (verb, pre_expr), = pre.items()
            return _experiment(
                world, f"{world.key}_switching",
                SwitchingRegent(verb, pre, post, world.switch_step), Harness([]),
                claim=f"the CLAIRVOYANT ADAPTOR for {world.key}: the best (pre, post) law pair, "
                      f"searched jointly rather than composed from two separate optima",
                falsification="n/a — key-free calibrated reference, not a claim",
                metadata={"role": "calibrated:switching", "laws": dict(post),
                          "pre_expr": pre_expr, "switch_step": world.switch_step},
            )

        register(f"{world.key}_switching")(make_switching)

    # -- the 2^3 harness factorial ---------------------------------------------------------------
    # One arm per (trace, outcome, memory) cell, so a component's contribution is estimated from
    # eight cells with interactions rather than from one all-on arm against one all-off arm. The
    # all-on cell keeps the plain ``{key}_llm`` name so existing runs and scripts still resolve.
    for cell in itertools.product((False, True), repeat=3):
        trace, outcome, memory = cell
        on = [n for n, b in zip(("trace", "outcome", "memory"), cell) if b]
        suffix = "_".join(on) if on else "bare"
        arm_name = f"{world.key}_llm_{suffix}"

        def make_llm_arm(world=world, arm_name=arm_name, cell=cell, on=tuple(on)) -> Experiment:
            trace, outcome, memory = cell
            max_tokens, extra = _llm_opts()
            components = []
            if trace:
                components.append(TraceFeedback())
            if outcome:
                components.append(OutcomeFeedback(k=4))
            if memory:
                components.append(EpisodicMemory(k=4))
            return _experiment(
                world, arm_name,
                LLMRegent(llm=_client(), model=_model(), temperature=0.0,
                          max_tokens=max_tokens, extra=extra),
                Harness(components),
                claim=world.claim,
                falsification="the paired bootstrap CI of (arm - standing_rule) on full-horizon "
                              "loss includes 0 over the seed set",
                metadata={"role": "treatment", "factors": list(on),
                          "cell": {"trace": trace, "outcome": outcome, "memory": memory},
                          "budget_matched": True,
                          "control_arm": f"{world.key}_standing_rule"},
            )

        register(arm_name)(make_llm_arm)

    llm_name = f"{world.key}_llm"

    def make_llm(world=world, llm_name=llm_name) -> Experiment:
        max_tokens, extra = _llm_opts()
        return _experiment(
            world, llm_name,
            LLMRegent(llm=_client(), model=_model(), temperature=0.0,
                      max_tokens=max_tokens, extra=extra),
            Harness([TraceFeedback(), OutcomeFeedback(k=4), EpisodicMemory(k=4)]),
            claim=world.claim,
            falsification="the paired bootstrap CI of (arm - standing_rule) on full-horizon loss "
                          "includes 0 over the seed set",
            metadata={"role": "treatment", "factors": ["trace", "outcome", "memory"],
                      "budget_matched": True,
                      "control_arm": f"{world.key}_standing_rule"},
        )

    register(llm_name)(make_llm)


for _world in WORLDS:
    _register_world(_world)
