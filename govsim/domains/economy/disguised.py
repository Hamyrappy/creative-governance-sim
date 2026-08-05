"""
A semantic-obfuscation control: the monetary world with its economics removed.

WHY THIS EXISTS. On the monetary world our regent is worse than every scripted reference, including
doing nothing. Its failure has a specific shape — it holds the policy rate at $9.51$ after a break
that severed the rate's transmission, achieving the best inflation-and-output stabilization of any
arm and paying three times the clairvoyant's price for it — and it writes recognizable Taylor rules
with conventional feedback gains while the calibrated optimum sets that gain to zero. The mandate
states, in every prompt, that the rate is charged "whether or not the rate reaches the economy".

That reads like a **domain prior overriding an explicit instruction**. But "reads like" is not a
measurement, and the alternative explanation is mundane: the monetary control problem may simply be
harder than the epidemic one, in which case the failure says nothing about priors at all.

The two are separable, and this module is the separation. It presents the IDENTICAL dynamics, the
identical break, the identical loss and the identical numbers, with every economic name removed:

    inflation        -> signal_a          policy_rate       -> lever
    output_gap       -> signal_b          r_star            -> baseline
    inflation_target -> target_a          "central bank"    -> (nothing)

Nothing about the control problem changes. Only the recognizability of the domain does.

    If the disguised regent does BETTER, the failure was the prior.
    If it does the same, the failure was task difficulty and the prior story is wrong.

This is a **pre-registered directional prediction**, recorded before the arm was run: we expect the
disguised arm to improve, because the mechanism we propose predicts it. If it does not, that is
reported as a refutation of our own explanation rather than quietly reinterpreted — the obvious
temptation here is to run the control, find no effect, and describe the result as "robustness of the
finding to surface framing", which would be exactly backwards.

The disguise is deliberately shallow: a capable model may well recognize a Taylor-rule problem from
its structure alone. That works AGAINST our hypothesis rather than for it — a shallow disguise can
only underestimate how much of the failure the prior explains — so a positive result here is a lower
bound on the effect.
"""

from __future__ import annotations

from typing import Any

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.economy.monetary import DualMandateLoss, MonetaryEconomy

#: observable name -> its neutral alias. Applied only at the ``observe()`` boundary, so the world's
#: own mechanics, metrics and objective are untouched and the two arms remain numerically identical.
ALIASES: dict[str, str] = {
    "inflation": "signal_a",
    "prev_inflation": "prev_signal_a",
    "inflation_target": "target_a",
    "output_gap": "signal_b",
    "prev_output_gap": "prev_signal_b",
    "policy_rate": "lever",
    "r_star": "baseline",
}


class DisguisedEconomy(MonetaryEconomy):
    """``MonetaryEconomy`` with de-economized observable names.

    Subclassed rather than reimplemented on purpose: any divergence in the dynamics would confound
    the very comparison this class exists to make. The only override is ``observe()``.

    ``metrics()`` is deliberately NOT renamed. It feeds the objective and the result store, never a
    prompt, and renaming it would silently break every analysis script that reads ``policy_rate``.
    """

    #: The lever verb the regent is offered. Renamed with the observables, or the disguise leaks
    #: through the action space — being asked to ``set_policy_rate`` names the domain outright.
    LEVER_VERB = "set_lever"

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        base = super().observe(viewer_id)
        return Observation(
            vars={ALIASES.get(k, k): v for k, v in base.vars.items()},
            scope=base.scope,
            t=base.t,
        )


class DisguisedLoss(DualMandateLoss):
    """``DualMandateLoss`` with an identically-valued, de-economized mandate.

    Every number is the same. Every economic noun is gone. The warning about paying for an
    instrument that no longer works is preserved VERBATIM in structure, because removing it would
    change the experiment from "does the prior override the instruction" to "does the instruction
    exist" — a question this project has already answered elsewhere.
    """

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Minimize, summed over the whole horizon, the squared deviation of "
            f"signal_a from its target_a of {self.target} PLUS {self.gap_weight} times the squared "
            f"value of signal_b, PLUS {self.lam} times the total cost of the lever you set.\n"
            "  - the lever's cost accrues on the LEVEL YOU SET, every step, and is charged whether "
            "or not the lever affects the system: a high setting is expensive even when it changes "
            "nothing;\n"
            "  - so a unit of lever is worth setting only while it buys back more than "
            f"{self.lam} times its own cost in signal_a deviation and signal_b magnitude;\n"
            "  - the system carries a persistent positive drift, so leaving the lever at its "
            "baseline is not a neutral choice.\n"
            "Lower total is better. Both halves matter: a controller that holds signal_a exactly at "
            "target with a punitive lever forever is not better than one that lets signal_a run a "
            "little and spends nothing."
        )


def undisguise(name: str) -> str:
    """Map a neutral alias back to its economic name, for reading transcripts."""
    inverse = {v: k for k, v in ALIASES.items()}
    return inverse.get(name, name)
