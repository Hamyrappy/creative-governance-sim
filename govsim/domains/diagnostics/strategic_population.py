"""
StrategicPopulation — a diagnostic world in which the governed LEARN the rule and move under it.

**The shock here is partly ENDOGENOUS, and that is the whole point.** Every other world on this
platform models the Lucas critique as an exogenous parameter overwrite: at ``shock_step`` some
number the regent cannot see changes, and the question is whether the regent notices and re-tunes.
Here the break is *caused by the regent's own policy*. Nothing about the environment has to change
for a sharp threshold rule to decay: the population watches where enforcement lands, forms an
estimate of the line, and slides its reported indicator to just below it. The longer a fixed sharp
rule is used, the better the estimate, the more mass bunches, and the less the rule does — with the
plant parameters held perfectly constant. The exogenous shock this world *does* arm
(``adaptation_rate`` rises at the break) only makes the endogenous process faster; it is an
accelerant, not the cause. A regent that treats the decay as an external event to be re-tuned
against will re-tune into the same trap.

WHAT THE INSTRUMENT MEASURES. Does the regent anticipate that the governed adapt to its rule? The
tempting answer is the one every regulator writes first: "intervene when the indicator exceeds θ".
It is legible, cheap (it touches only the units that deserve touching), and for a while it beats the
correct answer — that is requirement (2), the failure must be locally rewarded. What it also is, is
a cliff: a unit sitting a hair above θ can drop below it for a hair's worth of concealment cost and
escape intervention entirely. Once the population knows where the cliff is, the indicator stalls at
the line while the behaviour it was supposed to proxy walks away from it.

THE MECHANISM, in one line: **bunching pays exactly when the rule has a discontinuity.** A unit
compares the intervention it avoids, ``penalty × cliff_estimate``, against what concealment costs,
``evasion_unit_cost × need``. Concealment has a constant marginal price, so the choice is a
threshold on ``need`` and the whole mechanism collapses to one number — the **temptation depth**

    ``D(w) = (penalty / evasion_unit_cost) × min(1, 2·bunch_margin / w)``

which is how far above the believed line a unit can sit and still find concealment worth buying.
The ``min(1, 2m/w)`` factor is the cliff the population actually perceives: it estimates the drop by
comparing peers a margin ``m`` either side of the believed line, so a ramp of width ``w`` shows it
only the fraction ``2m/w`` of the full penalty. At the defaults (``m = 0.05``) a step rule saturates
at ``D = 0.75`` — three quarters of the behaviour range is in play — while ``D(w) = 0.075/w`` for any
ramp wider than ``2m``.

**There is no width at which gaming becomes impossible, and the docstring used to claim there was.**
``D`` decays like ``1/w`` and never reaches zero inside the lever's range. Both the share of units
inside the tempted slice and the concealment each of them buys scale with ``D``, so the realized
evasion gap falls like ``D² ∝ 1/w²``. Measured on this world, ``gap × w²`` is constant to within 10%
across ``w ∈ [0.4, 1.5]`` (≈1.2e-3), with no kink anywhere near the ``penalty/evasion_unit_cost``
value that an earlier version of this note advertised as a break-even. Quadratically unprofitable is
enough, and it is the honest claim: a wide band does not abolish the incentive to shade a report, it
prices it out for all but a vanishing slice. Enforcement is itself priced, which is why the
loss-optimal width is interior (≈0.6 at the defaults) rather than as wide as the lever will go.

THREE POLICIES, all writable in the platform's one-expression lever language:
  - NAIVE (fails):    ``set_threshold "0.6"`` + ``set_response_width "0.0"`` — the sharp line.
    Not one hand-picked setting: *every* sharp line in the plausible range is caught. Sweeping the
    threshold over 0.3–1.5 with width 0 gives post-break gaps of 0.29 down to 0.027, all of them
    far above the ≈0.001 the correct policies produce. The trap is a property of the discontinuity,
    which is what makes this an instrument rather than a single tuned counterexample.
  - CORRECT (smooth): ``set_threshold "0.6"`` + ``set_response_width "1.2"`` — a proportional band
    wide enough that the slice of units for which shading still pays is negligible (``D ≈ 0.06``).
    No usable cliff, almost nothing left to bunch against.
  - CORRECT (moving): a threshold that oscillates with ``t`` and a sharp response. The cliff still
    exists but is never where the population believes it is, so the *experienced* cliff — which is
    what the bunching decision runs on — averages near zero.

WHAT THE REGENT MAY NOT SEE. ``observe()`` publishes the reported indicator, the enacted levers,
the share flagged, the intervention rate, and a small noisy audit of true harm. It does NOT publish
``line_estimate`` or ``cliff_estimate`` (the population's beliefs — inferring that the population is
learning is the task), nor the true per-unit behaviour or the evasion. The audit exists so the world
is fair rather than blind — but the signature to watch for is not the obvious one. Under the gamed
sharp rule the reported indicator does **not** fall; it is pinned at the enforcement line and drifts
slightly *up* (mean 0.504 → 0.558 between the pre- and post-break windows). What moves is the truth
underneath it: true mean behaviour 0.504 → 0.836, and the audit tracks it, 0.053 → 0.349. So the
diagnosable pattern is "my indicator has gone quiet and stayed quiet while the audit climbs", not
"my reports are falling". The audit's between-seed spread over that late window is ≈0.04 against a
≈0.30 move, so the signal is roughly 7× the noise: legible, but a subsample with noise on it will
not hand over the answer.

DISCRIMINATING METRIC: ``post_evasion_gap`` — the mean post-break distance between what units
actually do and what they report. It separates the failure from total loss by construction: two
policies can land on similar loss for entirely different reasons (one spends on interventions, the
other is being lied to), and only the gap says which. ``post_bunched_share`` is the same failure
counted rather than measured, kept as a corroborating view.

MEASURED SEPARATION (10 seeds, horizon 240, break at 120), which is the only thing that decides
whether this file is an instrument or a story:

    policy                     loss   post_loss   post_harm   post_evasion_gap (sd)
    NAIVE sharp   (0.6, 0.0)  100.1        83.1        32.0   0.1929  (0.0199)
    CORRECT smooth(0.6, 1.2)   36.8        18.2         2.2   0.0008  (0.0005)
    CORRECT moving             38.2        19.0         3.9   0.0000  (0.0000)
    DO NOTHING                107.2        54.1        54.0   0.0000  (0.0000)

The margin is not marginal: worst naive seed minus best smooth seed is +0.162, i.e. the ranges are
separated by roughly 8× the naive arm's own seed-to-seed spread (Cohen's d ≈ 14). The gap is also
the only column that tells the two *failures* apart — DO NOTHING and NAIVE post to comparable harm
(54.0 vs 32.0) for opposite reasons, and loss cannot say which is which while the gap reads 0.000
against 0.193. Note the corollary: a near-zero gap is necessary, not sufficient. DO NOTHING scores a
perfect gap and the worst harm, so the gap is a diagnosis of one specific failure and must always be
read next to the loss, never instead of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem


class StrategicPopulation(LeverSystem):
    """A population of heterogeneous units that reports an indicator and learns the enforcement line.

    Per step, for each unit ``i`` with true (hidden) behaviour ``b_i``:

    1. **Report.** The unit bunches at ``line_estimate - bunch_margin`` iff the intervention it
       expects to avoid, ``penalty × cliff_estimate``, exceeds the concealment cost
       ``evasion_unit_cost × (b_i - target)``. Otherwise it reports honestly. The decision runs on
       the population's *beliefs*, not on the current rule — the population is backward-looking, so
       a rule that keeps moving is a rule it cannot exploit.
    2. **Enforce.** Intervention intensity ``A(y) = clip(0.5 + (y - threshold)/w, 0, 1)`` on the
       REPORT, where ``w = max(response_width, min_width)``. ``response_width → 0`` is a step at
       ``threshold``; a large width is a proportional response.
    3. **React.** ``b_i`` mean-reverts to its natural level and is pushed down by ``suppression × A``.
       A unit that evaded is not intervened on, so its behaviour drifts straight back up — the
       indicator holds still at the line while the thing it proxies climbs away underneath it.
    4. **Learn.** ``line_estimate`` and ``cliff_estimate`` move by ``adaptation_rate`` toward what
       enforcement revealed this step, but ONLY when there was enforcement to learn from
       (``mean intensity > evidence_floor``). Silence teaches nothing, which is why intermittent
       enforcement is a third defence this world happens to reward.

    Boundedness is structural, not incidental: ``b_i`` is clipped to ``[0, behaviour_max]``, ``A`` to
    ``[0, 1]``, the two beliefs to their own ranges, and evasion is non-negative and never exceeds
    ``b_i``. No arm can diverge over any horizon.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        self.n_units: int = int(p.get("n_units", 60))
        # -- population heterogeneity (drawn per seed in reset; see the note there) --
        self.natural_level: float = float(p.get("natural_level", 0.85))
        self.natural_sigma: float = float(p.get("natural_sigma", 0.50))
        self.behaviour_max: float = float(p.get("behaviour_max", 2.5))
        self.behaviour_sigma: float = float(p.get("behaviour_sigma", 0.02))
        # -- what the mandate is about --
        self.safe_level: float = float(p.get("safe_level", 0.50))  # harm accrues only above this
        # -- plant response to enforcement --
        self.reversion: float = float(p.get("reversion", 0.15))     # pull back toward the natural level
        self.suppression: float = float(p.get("suppression", 0.35))  # behaviour removed per unit intensity
        # -- the strategic layer. penalty/evasion_unit_cost is the SATURATED TEMPTATION DEPTH: under a
        # step rule a unit will buy concealment as long as it sits within this distance of the
        # believed line. At the defaults that is 1.5/2.0 = 0.75 — most of the behaviour range. It is
        # NOT a break-even width; widening the ramp scales the depth by 2*bunch_margin/w rather than
        # switching the incentive off, so the realized gap decays like 1/w^2 and never reaches zero.
        self.penalty0: float = float(p.get("penalty", 1.5))
        self.evasion_unit_cost0: float = float(p.get("evasion_unit_cost", 2.0))
        # bunch_margin is doing double duty and is the second-most load-bearing constant here: it is
        # both how far below the believed line an evader aims AND the half-window over which the
        # population perceives the cliff. Shrinking it makes wide bands look even flatter to the
        # population (less temptation everywhere); growing it past w/2 saturates the perceived cliff
        # and a ramp starts to read as a step. test_perceived_cliff_sets_the_temptation_depth pins it.
        self.bunch_margin: float = float(p.get("bunch_margin", 0.05))  # how far below the believed line
        self.min_width: float = float(p.get("min_width", 0.02))        # width 0 is still a finite step
        # -- how fast the population learns. THIS is the shocked parameter: the break makes the
        # population quicker, it does not make the trap exist. --
        self.adaptation_rate0: float = float(p.get("adaptation_rate", 0.006))
        self.evidence_floor: float = float(p.get("evidence_floor", 0.01))
        # The initial belief sits near the top of the behaviour range: at t=0 the population thinks
        # enforcement is far away, so nobody evades and a sharp rule looks excellent. The decay has
        # to be *earned* by the rule's own persistence, otherwise the world would just be a world
        # where evasion is on from the first step and there is no trap to fall into.
        self.line_init: float = float(p.get("line_init", 1.90))
        self.line_init_sigma: float = float(p.get("line_init_sigma", 0.12))
        # -- what the authority can measure about the truth: a noisy subsample audit --
        self.audit_sample: int = int(p.get("audit_sample", 12))
        self.audit_sigma: float = float(p.get("audit_sigma", 0.05))
        self.bunch_band: float = float(p.get("bunch_band", 0.15))  # width of the "just below θ" bin
        # -- levers' resting values before any policy is installed (do-nothing = no enforcement) --
        self.threshold_init: float = float(p.get("threshold_init", 2.5))
        self.response_width_init: float = float(p.get("response_width_init", 0.0))
        # -- the exogenous accelerant (named-parameter overwrite, as every other world does it) --
        ss = p.get("shock_step", 120)
        self.shock_step: int | None = None if ss is None else int(ss)
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {"adaptation_rate": 0.10}))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        # ``threshold`` spans the whole behaviour range so "never intervene" and "always intervene"
        # are both reachable; ``response_width`` spans from a step (0) to a ramp wider than the
        # break-even width, so the correct policy is inside the action space rather than at its edge.
        return {"threshold": (0.0, self.behaviour_max), "response_width": (0.0, 1.5)}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        # GENUINE per-seed heterogeneity. Without it every seed is the same run and the paired
        # bootstrap the platform reports would be an interval of width zero: the natural levels, the
        # starting beliefs, and the per-step behaviour noise all come from this Generator, never from
        # the module-global RNG (a CI AST test bans that).
        self.natural = np.clip(
            self.natural_level * np.exp(self.rng.normal(0.0, self.natural_sigma, self.n_units)),
            0.05, self.behaviour_max,
        )
        self.behaviour = self.natural.copy()
        self.evasion = np.zeros(self.n_units)
        self.reported = self.behaviour.copy()
        # Restore every shockable parameter from its ``*0`` snapshot, so a second run of the same
        # object starts from a pristine pre-shock world.
        self.adaptation_rate: float = self.adaptation_rate0
        self.penalty: float = self.penalty0
        self.evasion_unit_cost: float = self.evasion_unit_cost0
        # Hidden population beliefs — NOT in observe(). Inferring that these exist and are moving is
        # the task; publishing them would delete the problem.
        self.line_estimate: float = float(
            np.clip(self.line_init * np.exp(self.rng.normal(0.0, self.line_init_sigma)), 0.0, self.behaviour_max)
        )
        self.cliff_estimate: float = 0.0
        self.threshold: float = self.threshold_init
        self.response_width: float = self.response_width_init
        # Observable cache, initialized from the true starting state so the first ``_reeval_levers``
        # sees a coherent view rather than zeros.
        self.reported_mean: float = float(self.reported.mean())
        self.reported_max: float = float(self.reported.max())
        self.flagged_share: float = 0.0
        self.intervention_rate: float = 0.0
        self.audit_harm: float = float(np.maximum(self.behaviour - self.safe_level, 0.0).mean())
        # Accounting. Cumulative quantities are running totals from t=0 and MUST be differenced
        # across any window that is scored — see StrategicComplianceLoss.
        self.harm: float = float(np.maximum(self.behaviour - self.safe_level, 0.0).mean())
        self.evasion_gap: float = 0.0
        self.bunched_share: float = 0.0
        self.cum_intervention: float = 0.0
        self.cum_evasion_cost: float = 0.0
        self._t: int = 0
        self._levers.clear()

    def _response(self, y: Any, threshold: float, width: float) -> Any:
        """Intervention intensity as a function of the REPORTED indicator.

        ``width → 0`` gives a step at ``threshold`` (the notch the population can bunch against);
        a wide band gives a proportional response whose marginal relief per unit of concealment is
        ``1/width``, which is what the bunching calculus compares against the concealment price.
        """
        w = max(width, self.min_width)
        return np.clip(0.5 + (y - threshold) / w, 0.0, 1.0)

    def step(self) -> StepInfo:
        self._reeval_levers()  # re-eval the installed lever expressions EACH step (the contract)
        if self.shock_step is not None and self._t == self.shock_step:
            for name, value in self.shock_params.items():  # the accelerant, not the cause
                setattr(self, name, float(value))

        # 1) Reporting. Concealment cost is LINEAR, so the choice is bang-bang: sit at the believed
        # line or report honestly. That is not a modelling shortcut — it is why notches produce a
        # spike of mass at the notch in every empirical bunching study, rather than a smear.
        target = self.line_estimate - self.bunch_margin
        need = self.behaviour - target                      # concealment required to reach the line
        expected_relief = self.penalty * self.cliff_estimate
        evades = (need > 0.0) & (expected_relief > self.evasion_unit_cost * need)
        self.evasion = np.where(evades, np.maximum(need, 0.0), 0.0)
        self.reported = np.maximum(self.behaviour - self.evasion, 0.0)

        # 2) Enforcement lands on the REPORT — the authority cannot intervene on what it cannot see.
        intensity = self._response(self.reported, self.threshold, self.response_width)

        # 3) The plant reacts to enforcement, not to reports. A unit that evaded is untouched and
        # drifts back to its natural level: the indicator improves, the harm does not.
        noise = self.rng.normal(0.0, self.behaviour_sigma, self.n_units)
        self.behaviour = np.clip(
            self.behaviour + self.reversion * (self.natural - self.behaviour) - self.suppression * intensity + noise,
            0.0, self.behaviour_max,
        )

        # 4) Learning, gated on there being enforcement to learn FROM. The cliff estimate is what
        # peers either side of the believed line visibly experience; the line estimate walks toward
        # where enforcement actually bites. Both are EWMAs, so a rule that keeps moving is tracked
        # with a lag the population never closes.
        mean_intensity = float(intensity.mean())
        if mean_intensity > self.evidence_floor:
            observed_cliff = float(
                self._response(self.line_estimate + self.bunch_margin, self.threshold, self.response_width)
                - self._response(self.line_estimate - self.bunch_margin, self.threshold, self.response_width)
            )
            a = self.adaptation_rate
            self.cliff_estimate = float(np.clip(self.cliff_estimate + a * (observed_cliff - self.cliff_estimate), 0.0, 1.0))
            self.line_estimate = float(np.clip(self.line_estimate + a * (self.threshold - self.line_estimate), 0.0, self.behaviour_max))

        # 5) Accounting.
        excess = np.maximum(self.behaviour - self.safe_level, 0.0)
        self.harm = float(excess.mean())
        self.evasion_gap = float(self.evasion.mean())          # the discriminating flow
        self.intervention_rate = mean_intensity
        self.cum_intervention += mean_intensity
        self.cum_evasion_cost += self.evasion_unit_cost * self.evasion_gap
        self.bunched_share = float(
            ((self.reported <= self.threshold) & (self.reported > self.threshold - self.bunch_band)).mean()
        )
        # 6) Observable cache. The audit is a small noisy subsample — enough to notice that reports
        # and reality have parted company, not enough to read off the population's beliefs.
        self.reported_mean = float(self.reported.mean())
        self.reported_max = float(self.reported.max())
        self.flagged_share = float((self.reported > self.threshold).mean())
        idx = self.rng.integers(0, self.n_units, self.audit_sample)
        self.audit_harm = max(0.0, float(excess[idx].mean() + self.rng.normal(0.0, self.audit_sigma)))

        self._t += 1
        return StepInfo(terminated=False, truncated=False, info={})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # Deliberately absent: line_estimate / cliff_estimate (the population's beliefs), the
            # true per-unit behaviour, and the evasion. The regent sees the indicator it governs by,
            # what it enacted, and a noisy audit of the truth — the same information a real
            # regulator has, and exactly enough to catch the divergence without being told about it.
            vars={
                "t": float(self._t),
                "reported_mean": self.reported_mean,
                "reported_max": self.reported_max,
                "flagged_share": self.flagged_share,
                "intervention_rate": self.intervention_rate,
                "audit_harm": self.audit_harm,
                "threshold": self.threshold,
                "response_width": self.response_width,
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # The enacted levers are part of the record, not just the state they produced: the
        # governance question is what the institution *did*. ``evasion_gap`` and ``bunched_share``
        # are truths the regent cannot see but the scorer must, which is what makes them
        # diagnostic rather than another thing to optimize against.
        return {
            "t": float(self._t),
            "harm": self.harm,
            "evasion_gap": self.evasion_gap,
            "bunched_share": self.bunched_share,
            "reported_mean": self.reported_mean,
            "true_mean": float(self.behaviour.mean()),
            "intervention_rate": self.intervention_rate,
            "cum_intervention": self.cum_intervention,
            "cum_evasion_cost": self.cum_evasion_cost,
            "threshold": self.threshold,
            "response_width": self.response_width,
        }


class StrategicComplianceLoss(Objective):
    """Minimize harm + λ_i·enforcement + λ_e·concealment deadweight (negated → higher is better).

    Both weights are substantive. ``lam_intervention`` prices enforcement, so it decides whether the
    optimal institution is a narrow line (cheap, touches few) or a broad band (dearer, touches many)
    — set it too low and the smooth policy wins for the wrong reason, namely that intervention is
    free, and the world stops measuring anticipation. ``lam_evasion`` prices the resources the
    population burns on concealment; it is a real welfare loss borne by the governed, not a
    bookkeeping penalty on the regent.

    ``post_shock_step`` mirrors ``EpidemicLoss``: the ``post_*`` components cover the rows at-or-after
    the break, so the early window — where the sharp rule is cheaper than the correct answer because
    the population has not learned yet — is not averaged into the headline comparison. The
    discriminating metric ``post_evasion_gap`` lives there too. (Precisely: cheaper than the correct
    answer, 17.05 vs 18.61 over the pre-break window, 10/10 seeds. Not pre-break *optimal* — a narrow
    band at ``w = 0.4`` scores 14.64 — but a trap only has to out-score the right answer to be
    baited, and it does.)

    BOUNDARY, inherited deliberately: ``metrics()["t"]`` is read after ``step`` increments, so the row
    produced by the shock step carries ``t == shock_step + 1`` and a ``>= shock_step`` filter admits
    exactly one pre-break row. That is the same off-by-one ``SIRSystem``/``EpidemicLoss`` carry, and
    consistency with the rest of the platform is worth more here than one row in 120 — the
    contamination is 0.8% of the window and does not move any figure in this docstring.
    """

    def __init__(self, lam_intervention: float = 0.5, lam_evasion: float = 1.0,
                 harm_key: str = "harm", intervention_key: str = "cum_intervention",
                 evasion_cost_key: str = "cum_evasion_cost", gap_key: str = "evasion_gap",
                 bunched_key: str = "bunched_share", post_shock_step: int | None = None,
                 step_key: str = "t") -> None:
        self.lam_intervention = lam_intervention
        self.lam_evasion = lam_evasion
        self.harm_key = harm_key
        self.intervention_key = intervention_key
        self.evasion_cost_key = evasion_cost_key
        self.gap_key = gap_key
        self.bunched_key = bunched_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Minimize total social harm PLUS "
            f"{self.lam_intervention} times total enforcement effort PLUS "
            f"{self.lam_evasion} times the resources the population burns on concealment, summed "
            "over the whole horizon.\n"
            "  - harm accrues every step as the population's average activity ABOVE the safe level; "
            "it is driven by what units actually DO, not by what they report;\n"
            "  - enforcement effort accrues as the average intervention intensity you impose, and "
            "intensity is computed from the REPORTED indicator — you cannot intervene on what is "
            "not reported to you;\n"
            "  - concealment is a real cost borne by the governed: a unit that hides activity to "
            "stay under your rule destroys resources doing it, and you are charged for that too.\n"
            "You set two levers: the threshold of the indicator at which you respond, and the WIDTH "
            "of the response band around it. A width near zero means an all-or-nothing response at "
            "the threshold; a wide band means intensity rises gradually with the indicator.\n"
            "Lower total is better. The population is not a fixed object: it observes how you "
            "enforce and it responds."
        )

    def _parts(self, rows: Trajectory) -> tuple[float, float, float, float, float]:
        """(harm, enforcement, concealment, mean evasion gap, mean bunched share) over ``rows``.

        The two cumulative series are DIFFERENCED across the window rather than read off the last
        row. They are running totals from t=0, so an undifferenced read would bill a post-break
        window for enforcement bought before the break — which turns the score into a clock that
        only ever gets worse. That exact bug shipped here once, in the epidemic objective, and the
        realized-performance signal it fed the regent reported "worse than last time" every step of
        every seed. The flows (harm, gap, bunched share) are averaged or summed, never differenced.
        """
        if not rows:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        harm = sum(row.get(self.harm_key, 0.0) for row in rows)
        enforcement = rows[-1].get(self.intervention_key, 0.0) - rows[0].get(self.intervention_key, 0.0)
        concealment = rows[-1].get(self.evasion_cost_key, 0.0) - rows[0].get(self.evasion_cost_key, 0.0)
        gap = sum(row.get(self.gap_key, 0.0) for row in rows) / len(rows)
        bunched = sum(row.get(self.bunched_key, 0.0) for row in rows) / len(rows)
        return harm, enforcement, concealment, gap, bunched

    def _post_shock_rows(self, trajectory: Trajectory) -> Trajectory:
        s = self.post_shock_step
        if s is None:
            return list(trajectory)
        out = [row for row in trajectory if row.get(self.step_key, -1) >= s]
        if out:
            return out
        return trajectory[s:] if s < len(trajectory) else []

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        """Negated loss over ``trajectory``, which may be a WINDOW rather than a whole run.

        Differenced exactly as ``components`` does, and for the same reason: the Runner calls this
        on the interval since the regent's previous decision to build the realized-performance
        signal, and a running total read undifferenced makes that signal a monotone function of
        elapsed time instead of of the policy.
        """
        harm, enforcement, concealment, _gap, _bunched = self._parts(trajectory)
        return -(harm + self.lam_intervention * enforcement + self.lam_evasion * concealment)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        harm, enforcement, concealment, gap, bunched = self._parts(trajectory)
        loss = harm + self.lam_intervention * enforcement + self.lam_evasion * concealment
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and trajectory and not post_rows:
            # The run ended BEFORE the break: there is no post-break data, so every post_* metric is
            # worst-case, NOT a flattering 0.0. A zero here would make "end the run early" the
            # optimal way to score a perfect evasion gap, and a reference calibrated that way would
            # be measuring truncation rather than governance.
            inf = float("inf")
            return {
                "loss": loss, "harm": harm, "enforcement": enforcement, "concealment": concealment,
                "evasion_gap": gap, "bunched_share": bunched,
                "post_loss": inf, "post_harm": inf, "post_enforcement": inf,
                "post_concealment": inf, "post_evasion_gap": inf, "post_bunched_share": inf,
            }
        p_harm, p_enf, p_conc, p_gap, p_bunched = self._parts(post_rows)
        return {
            "loss": loss,
            "harm": harm,
            "enforcement": enforcement,
            "concealment": concealment,
            # Full-horizon view of the discriminating metric; the headline is the post_ variant.
            "evasion_gap": gap,
            "bunched_share": bunched,
            "post_loss": p_harm + self.lam_intervention * p_enf + self.lam_evasion * p_conc,
            "post_harm": p_harm,
            "post_enforcement": p_enf,
            "post_concealment": p_conc,
            # THE DISCRIMINATING METRIC: mean post-break distance between what the population does
            # and what it reports. Loss alone cannot say whether a policy is expensive because it
            # enforces a lot or because it is being lied to; this can.
            "post_evasion_gap": p_gap,
            "post_bunched_share": p_bunched,
        }
