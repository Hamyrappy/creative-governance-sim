"""
The economy domain — five governed worlds that share the scalar lever contract.

These are *not* a second action interface. Every system here subclasses
``govsim.domains.scalar.systems.LeverSystem`` and is driven through the same
``ScalarLeverInterface``: the regent emits one sandboxed Python expression per lever, the system
re-evaluates it every step and clips the result into range. What the package adds is not new
plumbing but new **worlds** — five economies with genuinely different dynamics, different
instruments, and different ways for an instrument to break.

**Why five and not one.** The platform's central result (``scripts/headroom_audit.py``,
``govsim/scenarios.py``) is that a shock to the *governed state* is absorbed by any feedback rule
and leaves ≈1.0x adaptation headroom, while a shock to the *efficacy of an instrument* does not.
That was measured on one epidemic. A claim about a shock *family* needs more than one world, or it
is a claim about SIR dynamics wearing a general costume. These five were built so the same
``ShockKind`` can be applied across unrelated mechanics and the headroom compared:

    ``MonetaryEconomy``    3-equation New Keynesian toy; lever = the policy rate.
                           Breaks at ``transmission`` — the Lucas critique's own case.
    ``FiscalSFCEconomy``   Godley & Lavoie SIM, money conserved to machine precision;
                           levers = tax rate + transfer. Breaks at ``tax_compliance``.
    ``CommonsEconomy``     logistic fishery with an Allee effect and an absorbing collapse;
                           levers = quota (needs obedience) + reserve (needs only geography).
                           Breaks at ``enforcement_efficacy``, which kills one instrument and
                           leaves the other — the substitution case.
    ``SupplyChainEconomy`` single-echelon inventory with an order pipeline; lever = the order.
                           Breaks at ``lead_time`` (a DELAY shock, a family never measured here)
                           or at ``fulfilment_efficacy``.
    ``OpinionPolity``      bounded-confidence opinion mass on a fixed grid; lever = moderation.
                           Breaks at ``moderation_efficacy``. Non-economic mechanics under an
                           economic mandate, which is the cross-check on "economy" meaning
                           anything structural here.

**Two properties every world in this package holds, and they are preconditions rather than
polish.** (1) *Bounded by construction* — conservation, hard clipping, or both, so no arm can win
a comparison by diverging and no run ends because the plant blew up. (2) *The efficacy parameter
is absent from* ``observe()`` — the regent sees what it enacted and what the world did, never the
coefficient joining them. Publishing it would delete the inference the experiment measures.

Each system ships with its own ``Objective``, and the objective owns the **prices**. That is
deliberate: a price decides whether the optimal policy is interior or at a corner, and a corner
optimum is regime-invariant — no shock can move it, so the world reports a null whatever governs
it. The λ in each loss below is therefore a substantive parameter of the experiment, not a
nuisance constant, and every one of them is documented with the sweep that set it.

Wired into runnable arms in ``govsim/experiments/economy_experiments.py``; measured for headroom
by ``scripts/headroom_audit.py``.
"""

from govsim.domains.economy.commons import CommonsEconomy, CommonsWelfare
from govsim.domains.economy.fiscal_sfc import FiscalSFCEconomy, FiscalStabilizationLoss
from govsim.domains.economy.monetary import DualMandateLoss, MonetaryEconomy
from govsim.domains.economy.opinion import OpinionPolity, PolarizationLoss
from govsim.domains.economy.supply_chain import SupplyChainCost, SupplyChainEconomy

__all__ = [
    # systems
    "CommonsEconomy",
    "FiscalSFCEconomy",
    "MonetaryEconomy",
    "OpinionPolity",
    "SupplyChainEconomy",
    # objectives
    "CommonsWelfare",
    "DualMandateLoss",
    "FiscalStabilizationLoss",
    "PolarizationLoss",
    "SupplyChainCost",
]
