"""
The scenario catalogue — shock families, declared once and applied uniformly across every system.

A library of worlds is only a research instrument if the *same* perturbation can be applied to all
of them and compared. Otherwise each world comes with its own bespoke shock and cross-system claims
are apples to oranges.

The taxonomy is not arbitrary. It comes out of the measurement in ``govsim.analysis.calibration``:
across every regime we have calibrated, **only shocks that degrade the instrument leave adaptation
headroom**. Shocks to the governed state are absorbed by any feedback rule — the observable rises,
the rule fires more often, and a policy nobody touched stays near-optimal. So the families below are
ordered by whether we expect them to be answerable, and a system that cannot express an
``INSTRUMENT_EFFICACY`` shock probably cannot host an adaptation experiment at all.

Each family is a *named parameter overwrite* at a step. Systems opt in by naming which of their
parameters play which role; nothing here knows what a lockdown or a tax rate is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ShockKind(str, Enum):
    """What part of the control problem the break moves.

    The ordering is by expected adaptation headroom, highest first, per the calibration results.
    """

    #: The lever keeps costing what it cost and stops working. The mapping from observation to
    #: correct action changes while the observation does not, so feedback cannot absorb it. This is
    #: the Lucas critique's own case and the only family we have found with real headroom.
    INSTRUMENT_EFFICACY = "instrument_efficacy"

    #: The lever still works but its price changes. Moves the interior optimum; whether that leaves
    #: headroom depends on where the optimum sits (at a corner, no shock can move it).
    INSTRUMENT_COST = "instrument_cost"

    #: The regent's view degrades — an observable becomes noisier, lagged, or biased. Distinct from
    #: the above because the *action* is still right; the evidence for choosing it is worse.
    OBSERVABILITY = "observability"

    #: A delay between acting and taking effect appears or lengthens. Breaks any rule tuned on the
    #: old lag, and is not visible in a snapshot of the state.
    DELAY = "delay"

    #: The plant's own dynamics move: growth rate, transmissibility, volatility. Feedback rules
    #: largely absorb these — we measure ~1.0x headroom — so a benchmark built here reports nulls
    #: regardless of the agent. Included precisely so that can be demonstrated rather than assumed.
    STATE_DYNAMICS = "state_dynamics"

    #: What "good" means changes: the objective's weights move mid-run. The agent is not told.
    PREFERENCE = "preference"


@dataclass(frozen=True)
class Scenario:
    """One named, reproducible perturbation of a world.

    ``params`` are written onto the system at ``step`` by the system's own shock mechanism, so a
    scenario is data rather than code and can be logged, diffed, and cited in a paper.
    """

    name: str
    kind: ShockKind
    step: int
    params: dict[str, Any] = field(default_factory=dict)
    #: Free-text description of what a governing authority would *experience*. Worth writing: it is
    #: what makes a scenario legible to a reader who does not know the parameter names.
    story: str = ""

    def applied_to(self, base: dict[str, Any]) -> dict[str, Any]:
        """A world config with this scenario armed. Never mutates ``base``."""
        return dict(base, shock_step=self.step, shock_params=dict(self.params))


def expected_headroom(kind: ShockKind) -> str:
    """What the calibration machinery has found for this family, as a prior — not a substitute.

    Always run ``scripts/headroom_audit.py`` on the actual regime. This is here to stop someone
    spending a week building a benchmark on a family we have already measured as near-null.
    """
    return {
        ShockKind.INSTRUMENT_EFFICACY: "measured 1.0-1.19x; the only family with real headroom, and "
                                       "only inside a narrow band of severity and price",
        ShockKind.INSTRUMENT_COST: "measured ~1.00x at the prices tried; moves the optimum only if "
                                   "it was interior to begin with",
        ShockKind.OBSERVABILITY: "not yet measured",
        ShockKind.DELAY: "not yet measured",
        ShockKind.STATE_DYNAMICS: "measured 1.00-1.26x across many families; near-null by "
                                  "construction, because feedback absorbs it",
        ShockKind.PREFERENCE: "not yet measured",
    }[kind]
