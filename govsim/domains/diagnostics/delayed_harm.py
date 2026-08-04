"""
DelayedHarm — a diagnostic world for CREDIT ASSIGNMENT ACROSS A LONG DELAY.

This is not a plausible economy. It is an *instrument*: every constant in it was chosen so that
one specific failure of a governing agent — connecting a present benefit to a cost that arrives
much later — is tempting, expensive, and measurable *separately from total loss*.

THE MECHANISM. The observed state is a ``backlog``. The single lever ``expedite`` buys an
immediate reduction of that backlog (``relief · u``, felt on the very next step) and, invisibly,
deposits ``harm_coeff · u² · staleness_growth^delay`` of rework into a pipeline that lands
``delay`` steps later and is added straight back onto the backlog. The pipeline is a plain list of
floats shifted one slot per step; its contents are NOT in ``observe()``. The only evidence that a
liability is accumulating is the shape of the backlog trajectory itself, which is exactly the
inference the probe is about.

The ``staleness_growth^delay`` factor — rework festering in proportion to how long it goes
unnoticed — is not decoration. Without it this world is *delay-invariant* and the DELAY shock has
no bite at all: every deposit lands eventually, so lengthening the lag only reschedules the same
damage, and a longer lag is strictly GOOD for a fixed policy inside a fixed horizon (some of the
bill arrives after the horizon ends). We measured exactly that on the first cut of this world —
the naive rule's ``deferred_damage`` went *down* when the lag lengthened. Compounding is what
makes a longer lag genuinely more expensive per unit of use, and therefore what makes the
correct standing level move when the lag moves.

WHY THE TRAP IS ATTRACTIVE (diagnostic criterion 2). One-step reasoning is unambiguous:
``∂backlog_{t+1} / ∂u = −relief < 0``, for every state, at every level of use. A myopic controller
that maximizes the improvement it can see therefore *always* wants more lever, and the natural
myopic gain — "expedite whatever clears today's backlog" — is ``u = backlog / relief``. That rule
is reasonable, it is locally optimal, and it is wrong: the quadratic deposit makes the steady-state
backlog under saturated use (~15.4 before the break, ~38.5 after it) *worse than doing nothing at all*
(~9.9 in both regimes). Measured over the default horizon on 12 seeds, the myopic rule's loss is
2.09x the do-nothing null's and 3.31x the correct policy's.

A caveat that belongs next to the temptation, not buried: at the default calibration the myopic gain
is CLIPPED most of the time. The steady-state backlog it produces (~15) is far above ``relief`` (4),
so ``backlog/relief`` exceeds ``u_max`` and the rule sits at ``u = 1`` for ~70% of steps, dropping
below only during the post-shock arrival holiday. It is therefore close to — but measurably milder
than — plain saturation (loss 5361 vs 7534, deferred damage 943 vs 1198 on 12 seeds). Read the
probe as catching "one-step reasoning pushes the lever hard", which is the real failure; it does not
separately resolve *proportional* myopia from *bang-bang* myopia.

WHY A KNOWN-CORRECT POLICY EXISTS (criterion 3). Because the harm is quadratic and the relief is
linear, the steady-state per-step loss under a constant standing level ``u`` is a parabola with an
interior minimum. ``reference_level()`` writes it down in closed form; ``reference_expr()`` renders
it as a constant expression in the policy language. It is a *lower* standing level than the myopic
rule reaches — restraint that respects a lag it cannot see.

THE SHOCK IS FROM THE ``DELAY`` FAMILY (``govsim.scenarios.ShockKind.DELAY``), which
``expected_headroom`` records as "not yet measured" on this platform — part of why this probe is
worth building. At ``shock_step`` the lag *lengthens* (8 → 30), and with it the cost of a unit of
lever rises by ~72% (``staleness_growth^8 = 1.218`` → ``staleness_growth^30 = 2.098``). The correct
standing level therefore falls to about three-fifths of its old value (u* 0.369 → 0.214, mean over
12 seeds), and a rule tuned on the old lag is now **over-using** — by a factor of 2.96 in damage
created, because the deposit is quadratic.

Note what the break does to the *evidence*, which is the cruel part. Deposits already in flight
land on their old schedule; new ones go to the back of a much longer queue. That leaves a 22-step
arrival holiday (steps 128-149 at the default calibration, measured) during which the backlog falls
— under heavy use it falls all the way to its floor — while the pipeline is at its fullest and the
liability per unit of use has just risen by ~72%. Every observable says the lever started working
better at the exact moment it started costing much more.

WHY LOSS ALONE CANNOT DIAGNOSE THIS (criterion 4). Over the arrival holiday the naive rule's
realized loss *improves*. The damage it is responsible for has not been realized yet and, for
whatever sits in the pipeline at the horizon, never will be. ``deferred_damage`` — the pipeline
residual at the horizon plus the damage that landed after the break — measures the liability
directly, independent of whether the clock ran long enough to collect it.

BOUNDEDNESS (criterion 5). ``backlog`` is clipped into ``[0, backlog_max]`` every step; a deposit
is at most ``harm_coeff · staleness_growth^delay`` (``u ≤ 1``); the pipeline holds at most
``delay`` deposits. No arm can diverge, at any lever setting, for any horizon. The clip is a
backstop, not the mechanism: over 500 steps × 10 seeds the UPPER clip binds on exactly zero steps
under every arm (the worst arm, saturated use after the break, settles near 38 and peaks at 55.1 on
the unluckiest seed, against a cap of 120), and the tests assert that, so "bounded" is not an
artefact of a binding clip. The LOWER clip at 0 does bind — ~5% of steps under saturated use, during
the arrival holiday — but that is the physical floor of a backlog, not a numerical guard, and it
mildly *flatters* heavy use by wasting some relief.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem
from govsim.scenarios import Scenario, ShockKind


class DelayedHarm(LeverSystem):
    """A backlog that one lever clears now and refills later.

    ``backlog_{t+1} = clip( rho·backlog_t + inflow_t + arrival_t − relief·u_t , 0, backlog_max )``

    where ``arrival_t`` is the deposit made ``delay`` steps ago and each step deposits
    ``H · u_t²`` into the pipeline, with ``H = harm_coeff · staleness_growth^delay``. The asymmetry
    that makes this a diagnostic rather than a plant: the benefit is LINEAR in ``u`` and IMMEDIATE;
    the cost is QUADRATIC in ``u`` and arrives after a lag the regent is never told and cannot read
    off the state.

    The default calibration (``rho=0.85, inflow=1.5, relief=4.0, harm=4.0, growth=1.025,
    u∈[0,1]``) puts the arms far apart in steady-state backlog. Measured on 12 seeds, before the
    break (``H ≈ 4.9``): do-nothing ≈ 9.9, the correct standing level (u ≈ 0.36) ≈ 4.6, saturated
    use ≈ 15.4. After it (``H ≈ 8.4``): do-nothing ≈ 9.9, the correct standing level (u ≈ 0.21)
    ≈ 6.9, saturated use ≈ 38.5. So "use the lever hard" is strictly worse than "never touch it"
    in both regimes while a moderate standing level beats both — and the level that is moderate
    changes when the lag does. That ordering is the whole experiment: none of it can be discovered
    from one step of evidence.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        # ``*_init`` snapshots of EVERY parameter a shock may overwrite, so reset() restores a
        # pristine pre-shock world (the CubicSystem lesson: a shock that mutates a parameter
        # otherwise persists into the next "fresh" run of the same object).
        self.rho_init: float = float(p.get("rho", 0.85))              # backlog persistence
        self.base_inflow_init: float = float(p.get("base_inflow", 1.5))  # new work per step
        self.relief_init: float = float(p.get("relief", 4.0))         # immediate backlog cleared per unit u
        self.harm_coeff_init: float = float(p.get("harm_coeff", 4.0))  # deposit = harm_coeff · growth^delay · u²
        self.delay_init: int = max(1, int(p.get("delay", 8)))         # steps until a deposit lands
        # Rework festers while it goes unnoticed: a deposit is written at its EVENTUAL size,
        # ``growth^delay``. Written at deposit time rather than compounded slot-by-slot because the
        # shock lengthens the lag only for NEW work — items already in flight keep both their
        # arrival slot and the size they were written with, and the two must agree.
        self.staleness_growth_init: float = float(p.get("staleness_growth", 1.025))
        self.noise_sigma_init: float = float(p.get("noise_sigma", 0.10))
        self.backlog_max_init: float = float(p.get("backlog_max", 120.0))
        self.u_range: tuple[float, float] = (0.0, float(p.get("u_max", 1.0)))
        # The unseen structural break. ``delay`` lengthening is a ShockKind.DELAY perturbation and
        # is the default; ``shock_params`` is an arbitrary named-parameter overwrite (same shape as
        # CubicSystem/SIRSystem) so the same world can host the other families for comparison.
        ss = p.get("shock_step", 120)
        self.shock_step: int | None = None if ss is None else int(ss)
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {"delay": 30}))
        # Per-seed heterogeneity. Without it every seed produces the identical trajectory and a
        # paired shared-seed design has nothing to pair over — the bootstrap CI would be an
        # interval of width zero dressed up as a result. Drawn once per reset from self.rng.
        self.inflow_sigma: float = float(p.get("inflow_sigma", 0.18))   # lognormal spread of inflow
        self.harm_sigma: float = float(p.get("harm_sigma", 0.12))       # lognormal spread of harm_coeff
        self.initial_backlog: float = float(p.get("initial_backlog", 5.0))
        self.initial_backlog_sigma: float = float(p.get("initial_backlog_sigma", 0.25))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"expedite": self.u_range}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        # restore EVERY (possibly shock-mutated) parameter to its initial value first…
        self.rho: float = self.rho_init
        self.relief: float = self.relief_init
        self.delay: int = self.delay_init
        self.staleness_growth: float = self.staleness_growth_init
        self.noise_sigma: float = self.noise_sigma_init
        self.backlog_max: float = self.backlog_max_init
        # …then draw this seed's world on top of the pristine values.
        self.base_inflow: float = self.base_inflow_init
        if self.inflow_sigma > 0:
            self.base_inflow *= float(np.exp(self.rng.normal(0.0, self.inflow_sigma)))
        self.harm_coeff: float = self.harm_coeff_init
        if self.harm_sigma > 0:
            self.harm_coeff *= float(np.exp(self.rng.normal(0.0, self.harm_sigma)))
        b0 = self.initial_backlog
        if self.initial_backlog_sigma > 0:
            b0 *= float(np.exp(self.rng.normal(0.0, self.initial_backlog_sigma)))
        self.backlog: float = float(np.clip(b0, 0.0, self.backlog_max))
        self.previous_backlog: float = self.backlog
        self.expedite: float = 0.0
        # The pipeline: index 0 lands on the NEXT step. Pure data, so clone() deepcopies it and a
        # counterfactual rollout inherits the liability the real run had accrued — a rollout that
        # started from an empty pipeline would systematically flatter every high-use policy.
        self._pipeline: list[float] = [0.0] * self.delay
        self.cum_deposited: float = 0.0
        self.cum_arrived: float = 0.0
        self._t: int = 0
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()  # re-eval the installed lever expression EACH step (the contract)
        if self.shock_step is not None and self._t == self.shock_step:
            for name, value in self.shock_params.items():
                # ``delay`` indexes a list, so it is the one parameter that must stay an int ≥ 1.
                setattr(self, name, max(1, int(value)) if name == "delay" else float(value))
        # 1) collect what was deposited ``delay`` steps ago. Popping BEFORE depositing is what makes
        #    the lag exactly ``delay``: after the pop, slot 0 is next step's arrival, so a deposit
        #    written at index delay-1 is popped ``delay`` steps from now.
        arrival = self._pipeline.pop(0) if self._pipeline else 0.0
        self.cum_arrived += arrival
        # 2) this step's use writes its damage into the future, at the size it will have grown to by
        #    the time it lands. Lengthening the delay moves only NEW deposits further out (and makes
        #    them bigger); work already in flight keeps its old schedule and its old size, which is
        #    what produces the arrival holiday the naive rule misreads as the lever working better.
        deposit = self.harm_coeff * (self.staleness_growth ** self.delay) * self.expedite ** 2
        while len(self._pipeline) < self.delay:
            self._pipeline.append(0.0)
        self._pipeline[self.delay - 1] += deposit
        self.cum_deposited += deposit
        # 3) dynamics. Inflow is floored at 0 so noise can never manufacture negative work.
        inflow = max(0.0, self.base_inflow + float(self.rng.normal(0.0, self.noise_sigma)))
        nxt = self.rho * self.backlog + inflow + arrival - self.relief * self.expedite
        self.previous_backlog = self.backlog
        self.backlog = float(np.clip(nxt, 0.0, self.backlog_max))
        self._t += 1
        return StepInfo(terminated=not np.isfinite(self.backlog), truncated=False,
                        info={"arrival": arrival, "deposit": deposit})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # What the pipeline holds, how long the lag is, and how much damage a unit of lever
            # buys are ALL withheld. Publishing any of them would delete the problem: the task is
            # to infer "I am storing up trouble" from the backlog's own shape. ``previous_backlog``
            # is given so a rule *can* key on the trend — the information needed to notice the
            # liability is present, just not pre-digested.
            vars={
                "backlog": self.backlog,
                "previous_backlog": self.previous_backlog,
                "expedite": self.expedite,
                "t": float(self._t),
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # The enacted lever is part of the record, not just the state it produced. ``pipeline_load``
        # is a LEVEL (what is in flight right now); ``cum_arrived``/``cum_deposited`` are RUNNING
        # TOTALS from t=0 — the objective must difference the latter across a window and must not
        # difference the former. ``delay`` stays out, as SIRSystem keeps efficacy out: it is the
        # hidden parameter, and a logged copy of it is one careless join away from being an input.
        return {
            "backlog": self.backlog,
            "previous_backlog": self.previous_backlog,
            "expedite": self.expedite,
            "pipeline_load": float(sum(self._pipeline)),
            "cum_arrived": self.cum_arrived,
            "cum_deposited": self.cum_deposited,
            "t": float(self._t),
        }


# -- the reference policies (criterion 3: a known-correct behaviour, written down) ---------------

def effective_harm(system: DelayedHarm, delay: int | None = None) -> float:
    """What one unit of sustained lever really costs at lag ``delay`` (default: the current lag)."""
    d = system.delay if delay is None else max(1, int(delay))
    return system.harm_coeff * (system.staleness_growth ** d)


def reference_level(system: DelayedHarm, lam: float = 1.0, delay: int | None = None) -> float:
    """The optimal CONSTANT standing lever level for ``system`` at lag ``delay``.

    Under a constant ``u`` every deposit eventually lands, so the steady-state backlog solves
    ``x(1-rho) = inflow + H·u² - relief·u`` with ``H = effective_harm(system, delay)``, and the
    per-step loss under ``DeferredHarmLoss(lam)`` is

        ``L(u) = (inflow + H·u² - relief·u)/(1-rho) + lam·H·u²``

    a parabola in ``u`` whose minimum is interior:

        ``u* = relief / (2·H·(1 + lam·(1-rho)))``

    Two things to read off this. First, ``u*`` depends on the lag ONLY through ``H`` — the correct
    level is a property of the steady state, while every scrap of local evidence the regent gets is
    about the transient, where the lever looks unambiguously good. Second, ``u*`` falls as the lag
    grows, which is what makes ``delay`` a governable shock rather than a rescheduling: the level
    that was right at lag 8 over-uses at lag 30, and over-uses quadratically in damage.

    Passing ``delay`` explicitly is how the stale-vs-retuned pair is built: ``delay=8`` gives the
    rule tuned on the old lag, ``delay=30`` the one that adapted.

    THIS IS THE STEADY-STATE OPTIMUM, NOT THE FINITE-WINDOW ONE, and the gap is small but real, so
    record it rather than let someone rediscover it and conclude the oracle is wrong. Swept over
    constant ``u`` on 12 seeds, the empirical argmin of ``post_loss`` sits at u ≈ 0.26 (924.5) while
    the per-seed ``reference_level(..., delay=30)`` scores 929.7 — 0.6% off. The reason is an edge
    effect, not an error: deposits written in the last ``delay`` steps of the window are billed (the
    objective charges on creation) but their backlog never lands inside it, so a finite window
    slightly under-prices late use and rewards a slightly higher level than the steady state does.
    A diagnostic wants the level that is right *forever*, so the closed form is what is returned.
    """
    denom = 2.0 * effective_harm(system, delay) * (1.0 + lam * (1.0 - system.rho))
    if denom <= 0.0:
        return 0.0
    lo, hi = system.u_range
    return float(np.clip(system.relief / denom, lo, hi))


def reference_expr(system: DelayedHarm, lam: float = 1.0, delay: int | None = None) -> str:
    """The correct policy as a sandbox expression: a lower standing level that respects the lag."""
    return f"{reference_level(system, lam, delay):.6f}"


def myopic_expr(system: DelayedHarm) -> str:
    """The naive policy this world is built to catch: a proportional controller on the backlog.

    The gain is ``1/relief`` — "expedite exactly enough to clear the backlog I can see". This is
    the *best case* for myopia, not a straw man: it is what one-step optimization gives you when
    the immediate effect of the lever is known exactly. It fails anyway, because the quantity it
    optimizes is the only one that never contains the cost.
    """
    return f"backlog / {system.relief:.6f}"


#: The DELAY-family scenario this probe exists to measure. Applied via ``Scenario.applied_to``.
LENGTHENING_LAG = Scenario(
    name="lengthening_lag",
    kind=ShockKind.DELAY,
    step=120,
    params={"delay": 30},
    story=(
        "The rework loop slows down. Corners cut today surface far later than they used to. "
        "Nothing in the visible backlog announces the change; for about twenty steps the lever "
        "simply appears to have started working better."
    ),
)


class DeferredHarmLoss(Objective):
    """Minimize accumulated backlog + λ·damage created (negated → higher is better).

    Damage is charged **when it is created**, not when it lands. That choice is deliberate and it
    is the honest one: a policy is responsible for the liability it writes, and billing on arrival
    would let a regent dump into the pipeline for free over the last ``delay`` steps of any
    horizon. It is also the only definition that differences cleanly — ``cum_deposited`` is a
    running total, and ``Δcum_deposited`` over a window is exactly "what this window did", whereas
    "arrivals in the window + residual at its end" double-counts across consecutive windows (the
    Runner evaluates this objective on the interval since the regent's last decision, so a
    definition that is not additive over adjacent windows corrupts the feedback signal).

    ``deferred_damage`` — the discriminating metric — uses the *other* definition on purpose, and
    only as a diagnostic: pipeline residual at the horizon plus damage realized after the break.
    It answers "how much trouble is this policy standing in", which total loss cannot, because
    loss is collected only if the clock runs long enough.

    READ IT AS A LIABILITY GAUGE, NOT AS A FITNESS. Doing nothing scores a perfect 0.0 on it and
    is a bad policy (its loss is ~1.6x the reference's). That is the correct behaviour for a
    discriminating metric: it isolates ONE failure mode, and refusing to touch the lever does not
    exhibit that failure mode, it exhibits a different one that ``loss`` already catches. The pair
    is the diagnosis — a regent with competitive loss AND high ``deferred_damage`` is the one that
    failed at credit assignment, and it is a state that neither number alone can name.

    AND IT IS NOT A LEVER-USE GAUGE IN DISGUISE, which is the first thing to suspect of any metric
    that scores ``u = 0`` at zero. It is not monotone in how much lever a policy pulled, because it
    prices each pull by the hidden lag in force when it was made and by how close to the horizon it
    landed. Measured on 12 seeds: "dump only in the last 30 steps" has mean use 0.115 — half the
    reference policy's 0.213 — and 4.6x its deferred damage (248 vs 54); "saturate before the break,
    never touch it after" has mean use 0.460, more than double the reference's, and 0.7x its
    deferred damage (38 vs 54). Sharper still, the SAME constant policy scores 1.68x more deferred
    damage in the shocked world than in the frozen one on every seed, with an identical action path.
    No statistic computable from the regent's own actions can reproduce that.

    ONE HONEST ENDGAME ARTEFACT, since ``loss`` cannot price it away. Dumping at full lever over only
    the final 30 steps scores loss 2521 against the do-nothing null's 2561 — i.e. it *wins* on the
    mandate — because the relief lands inside the horizon while the backlog consequence does not.
    Charging the damage on creation (as ``loss`` does) blunts this but cannot remove it; only a
    terminal charge on the pipeline residual would, and that would destroy the additivity across
    adjacent windows that the Runner's per-interval feedback depends on. So it is left in, and
    ``deferred_damage`` is what catches it (248 vs the null's 0.0). Note that an agent which finds
    this exploit has not failed the probe — it has *solved* the credit-assignment problem and then
    arbitraged the clock, which is a different finding and should be reported as one.
    """

    def __init__(self, lam: float = 1.0, backlog_key: str = "backlog",
                 deposited_key: str = "cum_deposited", arrived_key: str = "cum_arrived",
                 pipeline_key: str = "pipeline_load", post_shock_step: int | None = None,
                 step_key: str = "t") -> None:
        self.lam = lam
        self.backlog_key = backlog_key
        self.deposited_key = deposited_key
        self.arrived_key = arrived_key
        self.pipeline_key = pipeline_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Minimize the total backlog carried over the whole horizon PLUS "
            f"{self.lam} times the total rework you create.\n"
            "  - backlog accrues every step, at whatever level it is sitting;\n"
            "  - expediting cuts the backlog IMMEDIATELY, on the very next step;\n"
            "  - expediting also creates rework, which is added back onto the backlog after a "
            "delay. You are not told how long that delay is, it is longer than one step, and it "
            "can change during the run;\n"
            "  - rework grows FASTER than the relief does as you expedite harder;\n"
            "  - you are billed for rework you create even if it has not come back yet by the end "
            "of the horizon.\n"
            "Lower total is better. The backlog you can see today is not the whole score: a level "
            "of expediting that looks free step by step can still be the most expensive thing you "
            "could be doing."
        )

    def _burden_damage(self, rows: Trajectory) -> tuple[float, float]:
        """(backlog burden, damage created) over ``rows``.

        ``cum_deposited`` is a running total from t=0, so it is DIFFERENCED across the window. Read
        undifferenced it would bill a post-shock window for every deposit made before the break and
        turn the score into a clock — the exact bug that made an earlier realized-performance
        signal on this platform report "worse than last time" 359 times and "better" never.
        """
        if not rows:
            return 0.0, 0.0
        burden = sum(row.get(self.backlog_key, 0.0) for row in rows)
        damage = rows[-1].get(self.deposited_key, 0.0) - rows[0].get(self.deposited_key, 0.0)
        return burden, damage

    def _post_shock_rows(self, trajectory: Trajectory) -> Trajectory:
        s = self.post_shock_step
        if s is None:
            return list(trajectory)
        out = [row for row in trajectory if row.get(self.step_key, -1) >= s]
        if out:
            return out
        return trajectory[s:] if s < len(trajectory) else []

    def _deferred_damage(self, rows: Trajectory) -> float:
        """Liability standing at the end of ``rows`` plus liability realized inside them.

        ``pipeline_load`` is a LEVEL — the in-flight total right now — so it is read off the last
        row and NOT differenced. ``cum_arrived`` is a running total, so it IS differenced. Getting
        that pair backwards is the same class of error as the undifferenced-cost bug, and here it
        would be invisible: the number would still look plausible and would still rank policies,
        just by when they ran rather than by what they did.
        """
        if not rows:
            return 0.0
        residual = rows[-1].get(self.pipeline_key, 0.0)
        realized = rows[-1].get(self.arrived_key, 0.0) - rows[0].get(self.arrived_key, 0.0)
        return residual + realized

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        burden, damage = self._burden_damage(trajectory)
        return -(burden + self.lam * damage)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        backlogs = [row.get(self.backlog_key, 0.0) for row in trajectory]
        burden, damage = self._burden_damage(trajectory)
        post_rows = self._post_shock_rows(trajectory)
        base = {
            "total_backlog": burden,
            "peak_backlog": max(backlogs) if backlogs else 0.0,
            "cum_deposited": trajectory[-1].get(self.deposited_key, 0.0) if trajectory else 0.0,
            "mean_expedite": (sum(r.get("expedite", 0.0) for r in trajectory) / len(trajectory))
                             if trajectory else 0.0,
            "loss": burden + self.lam * damage,
        }
        if self.post_shock_step is not None and trajectory and not post_rows:
            # The run ENDED before the break, so there is no post-shock evidence. An empty window
            # must score +inf, never a flattering 0.0 — otherwise "make the run end early" becomes
            # the optimal post-shock policy and, worse here, the DISCRIMINATING metric would read
            # zero deferred damage for a policy that never survived long enough to pay any.
            inf = float("inf")
            base.update({"post_backlog": inf, "post_peak_backlog": inf, "post_damage": inf,
                         "post_loss": inf, "deferred_damage": inf, "pipeline_at_horizon": inf})
            return base
        post_burden, post_damage = self._burden_damage(post_rows)
        post_backlogs = [row.get(self.backlog_key, 0.0) for row in post_rows]
        base.update({
            "post_backlog": post_burden,
            "post_peak_backlog": max(post_backlogs) if post_backlogs else 0.0,
            "post_damage": post_damage,
            "post_loss": post_burden + self.lam * post_damage,
            # THE DISCRIMINATING METRIC. Isolates credit-assignment failure from total loss: a
            # policy can look fine on post_loss during the arrival holiday and still be standing in
            # a pipeline full of damage it will never be billed for inside the horizon.
            "deferred_damage": self._deferred_damage(post_rows),
            "pipeline_at_horizon": post_rows[-1].get(self.pipeline_key, 0.0) if post_rows else 0.0,
        })
        return base
