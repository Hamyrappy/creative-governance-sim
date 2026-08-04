"""
SignFlipPlant — a diagnostic world for ONE capability: can a regent ABANDON a learned direction?

This is an instrument, not a simulation. Every number in it was chosen to make a single failure
mode tempting, expensive, and *separately observable* from the loss.

THE CAPABILITY. Most "adaptation" a governing agent is asked for is re-tuning: the lever still
works, just less well, so you push harder or you push less. That is a magnitude edit, and a
gradient-follower finds it. The hard case is a lever that REVERSES — the instrument that used to
raise the state now lowers it. Re-tuning cannot reach the answer from there; only discarding the
learned sign can. A frozen feedback rule facing a reversed lever does not merely underperform, it
actively drives the state away from target, and *because it is feedback* it pushes harder the worse
things get. The rule amplifies its own error.

THE TRAP, and why it is attractive rather than careless:
  - Pre-break, the naive law ``u = -k·(x - target)`` is optimal, and it is optimal for a long time
    (150 of 400 steps by default). It is not a strawman; it is the answer to the problem as posed.
  - Passivity is not a free retreat. An unobservable, persistent AR(1) disturbance drives the state,
    and the plant's own decay only damps it by a factor of ``1/(1 - decay)``. Doing nothing is
    roughly 25x worse than governing well. So "stop touching it" is punished too.
  - Post-break the visible evidence is: the state is far from target AND my control is already at
    its bound. Read through the pre-break model that is exactly what a *larger disturbance* looks
    like, and the indicated response — push harder — is what the frozen rule is already doing. The
    world therefore rewards the wrong diagnosis locally before it punishes it globally.

WHAT IS HIDDEN, AND WHAT IS NOT. ``control_gain`` never appears in ``observe()`` or in
``metrics()`` — not even its magnitude, and not the running sign-error counters that would give the
answer away through the outcome channel (``Runner`` passes ``system.metrics()`` into
``harness.on_outcome``). The plant's OWN dynamics (``decay``) *are* published: you may know how fast
the economy mean-reverts and still not know what your instrument does to it. That asymmetry is the
Lucas critique's own shape, and it is what leaves the inference non-trivial but not unfair — the
evidence available is precisely "acting makes things worse".

BOUNDEDNESS is structural, not clamped. There is no hard clip on the state. The runaway is bounded
because (a) the lever is clipped to ``u_range`` so the feedback term saturates, and (b) the
open-loop pole ``decay < 1``. Once saturated the closed loop is ``x ← decay·x + |g|·U``, whose fixed
point is ``|g|·U/(1 - decay)`` — large (~20 at the defaults) and therefore expensive, but finite.
``tests/test_diag_sign_flip.py`` pins this over 500 steps under do-nothing and max-lever arms.

THE DISCRIMINATING METRIC is ``wrong_sign_fraction`` (see ``SignFlipLoss``): the post-break share of
*acting* steps whose control pushed the state the wrong way given the true gain. It is deliberately
not a function of magnitude, so a regent that shrinks its gain — worth a 10-75x drop in post-break
loss — still scores 1.0. Loss says "progress"; the discriminator says "still pointed the wrong way".
Separating those two is the entire reason this world exists.

And the retreat is not even monotone: an intermediate gain parks the closed loop near a unit root,
where the disturbance random-walks the state across the whole saturation band, so a cautious
half-measure can score WORSE than never adapting at all. See ``detuned_law``. The world therefore
cannot be passed by timidity — only by reversal.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem

#: The pre-break-optimal proportional gain at the nominal plant (``decay0 / control_gain0``): it
#: cancels the plant's own decay, so ``x_{k+1} = disturbance``. Both reference policies below use
#: this SAME magnitude — they differ only in sign, which is what makes the comparison a clean test
#: of direction rather than of tuning. Per-seed heterogeneity leaves it slightly off-optimal, which
#: is the honest situation for a rule fitted on a finite pre-break history.
NOMINAL_FEEDBACK_GAIN = 0.85


def frozen_law(gain: float = NOMINAL_FEEDBACK_GAIN) -> str:
    """THE NAIVE POLICY: the pre-break-optimal proportional law, held across the break.

    Reasonable, locally rewarded, and catastrophically wrong after the lever reverses.
    """
    return f"-{gain:.6g} * (current_x - target_x)"


def reversed_law(break_step: int, gain: float = NOMINAL_FEEDBACK_GAIN) -> str:
    """THE KNOWN-CORRECT POLICY: the same law, opposite sign, from the break onward.

    Written in the policy language over published observables only (``step``, ``current_x``,
    ``target_x``) so it is a policy the regent could actually have emitted, not a privileged oracle.
    Knowing *when* the break happened is granted here; knowing *what to do about it* is the test.
    """
    return (
        f"(-{gain:.6g} * (current_x - target_x)) if step < {int(break_step)} "
        f"else ({gain:.6g} * (current_x - target_x))"
    )


def true_post_break_gain_sign(plant: "SignFlipPlant") -> float:
    """The sign the lever ACTUALLY has after the break, derived from the plant's configuration.

    Read off the ``*0`` snapshots rather than the live ``control_gain``, so it is correct whether
    or not the plant has already been stepped past its break. The per-seed heterogeneity factor is
    ``exp(normal)``, strictly positive, so the SIGN is a property of the configuration and not of
    the seed — which is what makes it safe to compute once and hand to the objective.
    """
    if "control_gain" in plant.shock_params:
        gain = float(plant.shock_params["control_gain"])  # an overwrite wins over the flip
    else:
        gain = plant.control_gain0 * plant.flip_factor
    if gain == 0.0 or not math.isfinite(gain):
        raise ValueError(
            "SignFlipPlant has a zero/non-finite post-break control gain: the lever does nothing "
            "after the break, so 'which way did you push' has no answer and the diagnostic is void."
        )
    return math.copysign(1.0, gain)


def loss_for(plant: "SignFlipPlant", **kwargs: Any) -> "SignFlipLoss":
    """Build the ``SignFlipLoss`` that MATCHES ``plant`` — the only safe way to pair the two.

    ``post_break_gain_sign`` and ``post_shock_step`` are properties of the plant, but nothing in the
    type system couples them to the objective, and getting either wrong fails SILENTLY rather than
    loudly: a plant built as the no-reversal control arm (``flip_factor=+1``) scored by a default
    ``SignFlipLoss`` reports the frozen law at ``wrong_sign_fraction = 1.0`` and the reversed law at
    ``0.0`` — the instrument reads exactly backwards, and every arm still returns a plausible number
    in [0, 1] with no error raised. Measured, not hypothesized; ``test_loss_for_matches_the_plant``
    pins it. Prefer this factory to constructing ``SignFlipLoss`` by hand.
    """
    kwargs.setdefault("post_shock_step", plant.shock_step)
    kwargs.setdefault("post_break_gain_sign", true_post_break_gain_sign(plant))
    kwargs.setdefault("target", plant.target_x0)
    return SignFlipLoss(**kwargs)


def detuned_law(break_step: int, gain: float = NOMINAL_FEEDBACK_GAIN, post_gain: float = 0.10) -> str:
    """THE NEAR-MISS: react to the break by SHRINKING the gain, keeping the sign.

    This is the policy the discriminating metric exists to catch. Taken far enough (``post_gain``
    ~0.02) it improves the post-break loss by 10-75x — a hill-climber will happily take it — while
    remaining, at every acting step, pointed the wrong way.

    Note that the improvement is NOT monotone in ``post_gain``, and that is a feature. The
    post-break closed-loop pole is ``decay + |g|·post_gain``; shrinking the gain walks it down
    through 1.0, and near the unit root the persistent disturbance random-walks the state across the
    whole saturation band. On slow-decay seeds a cautious cut to ``post_gain=0.10`` is therefore
    WORSE than leaving the frozen rule alone — which reads, from inside, as evidence that the
    direction was right and only the nerve was wrong. Half-measures are punished; nothing short of
    crossing zero is reliably rewarded.
    """
    return (
        f"(-{gain:.6g} * (current_x - target_x)) if step < {int(break_step)} "
        f"else (-{post_gain:.6g} * (current_x - target_x))"
    )


class SignFlipPlant(LeverSystem):
    """A stable first-order plant whose control lever REVERSES at the break.

    Dynamics (``u`` is the single lever ``current_u``, clipped to ``u_range``)::

        x_{k+1} = decay·x_k + control_gain·u_k + d_k + ε_k
        d_{k+1} = dist_rho·d_k + η_k                      (persistent, UNOBSERVABLE disturbance)

    At ``shock_step`` the break fires: ``control_gain *= flip_factor`` (default ``-1`` — the lever
    reverses at unchanged strength), then any ``shock_params`` overwrites are applied. The flip is
    multiplicative rather than a fixed overwrite so that per-seed heterogeneity in the gain
    MAGNITUDE survives the shock; a hardcoded ``shock_params={"control_gain": -1.0}`` would erase it
    and make the paired statistics vacuous.

    Why the disturbance is AR(1) and hidden: a white-noise disturbance is not worth governing (the
    optimal response to it is to do nothing), and an observable one turns the problem into
    feedforward arithmetic. A persistent, unobservable disturbance is the case where feedback is
    both necessary and the only available instrument — which is what makes the reversal of that
    instrument the whole problem.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        # -- plant --
        self.decay0: float = float(p.get("decay", 0.85))          # open-loop pole; MUST stay < 1
        self.decay_sigma: float = float(p.get("decay_sigma", 0.05))
        self.decay_bounds: tuple[float, float] = tuple(p.get("decay_bounds", (0.50, 0.93)))  # type: ignore[assignment]
        self.control_gain0: float = float(p.get("control_gain", 1.0))  # MAGNITUDE pre-break (sign +)
        self.gain_sigma: float = float(p.get("gain_sigma", 0.15))
        self.sigma_epsilon: float = float(p.get("sigma_epsilon", 0.05))
        # -- the persistent, unobservable disturbance that makes governing necessary --
        self.dist_rho: float = float(p.get("dist_rho", 0.90))
        self.dist_sigma: float = float(p.get("dist_sigma", 0.25))
        # -- control surface + mandate --
        lo, hi = p.get("u_range", (-3.0, 3.0))
        self.u_range0: tuple[float, float] = (float(lo), float(hi))
        if self.u_range0[0] >= self.u_range0[1]:
            raise ValueError("u_range must be a non-degenerate interval (lo < hi)")
        self.target_x0: float = float(p.get("target_x", 0.0))
        self.control_cost: float = float(p.get("control_cost", 1.0))  # price of u² per step
        # -- the break --
        self.shock_step: int = int(p.get("shock_step", 150))
        self.flip_factor: float = float(p.get("flip_factor", -1.0))
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {}))
        # -- per-seed heterogeneity --
        self.initial_x: float = float(p.get("initial_x", 0.0))
        self.initial_x_sigma: float = float(p.get("initial_x_sigma", 0.5))
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"current_u": self.u_range}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        # Genuine per-seed heterogeneity in BOTH plant constants and the initial condition. Without
        # it every seed traces the same curve, a paired shared-seed comparison has nothing to pair
        # over, and the bootstrap CI is a zero-width interval wearing the costume of a result.
        decay = self.decay0 * float(np.exp(self.rng.normal(0.0, self.decay_sigma))) if self.decay_sigma > 0 else self.decay0
        self.decay: float = float(np.clip(decay, *self.decay_bounds))  # clipped: decay < 1 is what bounds the runaway
        gain = self.control_gain0
        if self.gain_sigma > 0:
            gain *= float(np.exp(self.rng.normal(0.0, self.gain_sigma)))
        self.control_gain: float = gain  # POSITIVE pre-break; the break multiplies it by flip_factor
        # Start the disturbance at its stationary spread rather than 0, so the first steps are not a
        # quiet burn-in that a policy could be (accidentally) calibrated on.
        stationary_sd = self.dist_sigma / math.sqrt(max(1e-9, 1.0 - self.dist_rho ** 2))
        self.disturbance: float = float(self.rng.normal(0.0, stationary_sd))
        self.current_x: float = self.initial_x + (
            float(self.rng.normal(0.0, self.initial_x_sigma)) if self.initial_x_sigma > 0 else 0.0
        )
        self.previous_x: float = self.current_x
        self.current_u: float = 0.0
        # Restore every shockable parameter from its ``*0`` snapshot, so a re-``reset`` object is a
        # pristine PRE-break plant and not a quietly-flipped one.
        self.u_range: tuple[float, float] = self.u_range0
        self.target_x: float = self.target_x0
        self._t: int = 0
        self.cum_cost: float = 0.0  # running total; ALWAYS differenced across a window when scored
        self._levers.clear()

    def step(self) -> StepInfo:
        self._reeval_levers()  # the eval-cadence contract: the installed law is re-read EVERY tick
        if self._t == self.shock_step:
            # THE BREAK. It lands after the lever was evaluated, so the very step of the break
            # already applies the old rule's control through the new, reversed gain — the regent
            # cannot have acted on it, which is the point.
            self.control_gain *= self.flip_factor
            for name, value in self.shock_params.items():
                setattr(self, name, float(value))
        eps = float(self.rng.normal(0.0, self.sigma_epsilon)) if self.sigma_epsilon > 0 else 0.0
        next_x = self.decay * self.current_x + self.control_gain * self.current_u + self.disturbance + eps
        # Advance the disturbance AFTER it has acted, so d_k drives x_{k+1} exactly once.
        self.disturbance = self.dist_rho * self.disturbance + float(self.rng.normal(0.0, self.dist_sigma))
        self.previous_x = self.current_x
        self.current_x = float(next_x)
        # Billed on the control YOU SET, like the epidemic world's lockdown cost: an instrument that
        # stopped working still costs what it costs. That asymmetry is what makes persisting with a
        # reversed lever expensive on both terms of the mandate rather than only on the state term.
        self.cum_cost += self.control_cost * (self.current_u ** 2)
        self._t += 1
        # Safety net only. The bound is structural (clipped lever + decay < 1); if this ever fires,
        # the world is mis-parameterized and the test that pins boundedness has failed first.
        terminated = not np.isfinite(self.current_x) or abs(self.current_x) > 1e9
        return StepInfo(terminated=terminated, truncated=False, info={})

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # ``control_gain`` is absent BY DESIGN — magnitude and sign both. Inferring "my lever
            # reversed" from the trace is the task; publishing it here would delete the task.
            # ``disturbance`` is absent too: it is what makes feedback necessary rather than
            # feedforward. ``decay`` IS published — the plant's own dynamics are common knowledge,
            # only the instrument's effect is not.
            vars={
                "step": float(self._t),
                "current_x": self.current_x,
                "previous_x": self.previous_x,
                "current_u": self.current_u,
                "target_x": self.target_x,
                "decay": self.decay,
                "u_range_min": self.u_range[0],
                "u_range_max": self.u_range[1],
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # The ENACTED lever value is part of the record: ``wrong_sign_fraction`` is computed from
        # (state, control) pairs, so a trajectory that omitted the policy could not be scored at all.
        # Still no ``control_gain`` and no running sign-error counter here: ``Runner`` feeds
        # ``metrics()`` back through ``harness.on_outcome``, so anything published here is a channel
        # to the regent, and a counter of its own sign errors would hand over the answer.
        return {
            "step": float(self._t),
            "current_x": self.current_x,
            "previous_x": self.previous_x,
            "current_u": self.current_u,
            "target_x": self.target_x,
            "cum_cost": self.cum_cost,
        }


class SignFlipLoss(Objective):
    """Tracking loss + control price, PLUS the direction diagnostic ``wrong_sign_fraction``.

    Score (negated, so higher is better)::

        loss = mean (x - target)² + lam · mean control cost

    ``mean control cost`` is read as ``(cum_cost[-1] - cum_cost[0]) / n`` — DIFFERENCED across the
    window, never read off the last row. ``cum_cost`` is a running total from t=0, so an
    undifferenced read would bill the post-break window for control bought before the lever ever
    reversed, and would turn the ``Runner``'s per-interval realized-feedback signal into a clock that
    reports "worse than last time" forever. That exact bug has bitten this codebase once already.

    THE DISCRIMINATING METRIC. ``wrong_sign_fraction`` is the share of post-break *acting* steps at
    which the enacted control pushed the state AWAY from target, given the true post-break gain sign.
    With error ``e = x - target`` and true gain ``g``, the control contributes ``g·u`` to the next
    state, so a correcting step is one with ``g·u·e < 0``; a step is counted WRONG when
    ``g·u·e > 0``. The gain sign enters as a constructor parameter (``post_break_gain_sign``), which
    the experimenter knows and the regent does not — so nothing has to be leaked through the
    system's observables to score it.

    ``e`` is read from ``decision_state_key`` (``previous_x``), NOT from the state the row ends at.
    A row's control was chosen at the START of that step, from the state the regent then observed;
    the row's ``current_x`` is what the control *produced*. Judging direction against the outcome
    instead of against the input misclassifies every zero-crossing — it charged a perfectly correct
    reference policy with a ~4% error rate, purely as a timing artifact.

    Two deadbands keep it honest rather than noisy:
      - ``state_deadband``: steps where the state is already at target need no correction, and
        chasing noise there is not a directional error;
      - ``control_deadband``: a control of ~0 has no sign. Such steps leave the numerator AND the
        denominator, and their share is reported separately as ``active_fraction``.

    So read the pair. ``wrong_sign_fraction ≈ 1`` with ``active_fraction ≈ 1`` is the frozen (or
    merely detuned) rule — the failure this world was built for. ``wrong_sign_fraction ≈ 0`` with
    ``active_fraction ≈ 1`` is a regent that actually reversed. ``active_fraction ≈ 0`` is a regent
    that gave up governing: it has abandoned the direction without finding the new one, which the
    loss — not this metric — is there to punish.

    WITH NO ACTING STEP THE FRACTION IS ``nan``, NOT 0.0. A control of ~0 at every step leaves the
    denominator empty, and 0/0 reported as 0.0 ranks abdication as *better than the known-correct
    policy* on the headline number: measured, the do-nothing arm scored 0.000000 against the
    corrected law's 0.003105. "Report both or neither" is not enough protection when a
    ``components()`` dict is what lands in a results table and gets averaged across seeds — the
    reader has to notice an absence. ``nan`` cannot be silently averaged into a favourable number:
    it propagates, and the consumer is forced back to ``active_fraction`` and the loss. It is also
    the honest encoding, since a passive regent committed no directional error — it simply produced
    no evidence about direction. (This is why it is ``nan`` and not the ``+inf`` used for an empty
    post-break WINDOW: that case is worst-cased on purpose, to stop early termination from paying.)

    Raw counts (``wrong_steps``/``active_steps``/``called_for_steps``) are reported alongside so a
    multi-seed summary can POOL them rather than averaging per-seed ratios — a different and better
    estimator when the denominators differ across seeds, as they do whenever an arm is deadbanded.

    Set ``post_shock_step`` to the plant's ``shock_step``. Left as ``None`` the post window is the
    whole run, and the sign metric would then be evaluated under the post-break gain sign over
    pre-break rows too, which is meaningless.

    One row of slop is accepted deliberately. A row is stamped with ``step = t + 1`` (metrics are
    read after the tick), so the platform's ``step >= post_shock_step`` window — the same one
    ``StabilizationLoss`` and ``EpidemicLoss`` use — admits the single row whose control was still
    chosen under the old regime. Worth ~0.4% of a 250-row window, and matching the rest of the
    platform is worth more than removing it.
    """

    def __init__(self, lam: float = 0.2, state_key: str = "current_x", control_key: str = "current_u",
                 target_key: str = "target_x", target: float = 0.0, cost_key: str = "cum_cost",
                 post_shock_step: int | None = None, step_key: str = "step",
                 post_break_gain_sign: float = -1.0, decision_state_key: str = "previous_x",
                 state_deadband: float = 0.25, control_deadband: float = 0.05) -> None:
        self.lam = lam
        self.state_key = state_key
        self.decision_state_key = decision_state_key
        self.control_key = control_key
        self.target_key = target_key
        self.target = target
        self.cost_key = cost_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key
        if post_break_gain_sign == 0.0 or not math.isfinite(post_break_gain_sign):
            # A zero sign makes ``g·u·e > 0`` unsatisfiable, so EVERY policy scores
            # ``wrong_sign_fraction = 0.0`` and the instrument silently certifies the failure it
            # exists to catch. Fail at construction instead. Use ``loss_for(plant)``.
            raise ValueError(
                f"post_break_gain_sign must be a finite non-zero number (got {post_break_gain_sign!r}); "
                "it is the SIGN of the true post-break lever, normally ±1. Use loss_for(plant)."
            )
        self.post_break_gain_sign = post_break_gain_sign
        self.state_deadband = state_deadband
        self.control_deadband = control_deadband

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Hold the state at its target using your single control, and do not pay "
            f"more for control than the tracking is worth. The score is mean squared deviation of "
            f"the state from target PLUS {self.lam} times the mean squared control you set, "
            "averaged over the horizon. Lower is better.\n"
            "  - you are billed for the control you order, whether or not it does what you expect;\n"
            "  - the state is pushed around by a persistent disturbance you cannot observe, so "
            "doing nothing is not a safe option: the deviation it leaves is expensive;\n"
            "  - the plant's own decay rate is published to you. HOW YOUR CONTROL ACTS ON THE STATE "
            "IS NOT, and is not guaranteed to stay the same for the whole horizon. If the results "
            "of acting stop matching what acting used to do, the evidence you have is the record of "
            "what you did and what followed."
        )

    # -- windows ---------------------------------------------------------------------------------

    def _post_shock_rows(self, trajectory: Trajectory) -> Trajectory:
        s = self.post_shock_step
        if s is None:
            return list(trajectory)
        out = [row for row in trajectory if row.get(self.step_key, -1) >= s]
        if out:
            return out
        return trajectory[s:] if s < len(trajectory) else []

    def _mse_cost(self, rows: Trajectory) -> tuple[float, float]:
        """(mean squared tracking error, MEAN control cost) over ``rows``."""
        if not rows:
            return 0.0, 0.0
        for row in rows:
            if self.state_key not in row:  # loud-fail a mis-wired objective/system pairing rather
                raise KeyError(              # than scoring a missing state as 0.0 (a tiny fake MSE)
                    f"SignFlipLoss: state_key '{self.state_key}' absent from a trajectory row "
                    f"(keys: {sorted(row)}); the objective is wired to the wrong system."
                )
        n = len(rows)
        mse = sum((row[self.state_key] - row.get(self.target_key, self.target)) ** 2 for row in rows) / n
        # DIFFERENCED, not read off the last row — see the class docstring.
        cost = (rows[-1].get(self.cost_key, 0.0) - rows[0].get(self.cost_key, 0.0)) / n
        return mse, cost

    # -- the discriminator -----------------------------------------------------------------------

    def _sign_counts(self, rows: Trajectory) -> tuple[int, int, int]:
        """(wrong, active, called_for) over ``rows`` under the true post-break gain sign."""
        g = self.post_break_gain_sign
        wrong = active = called_for = 0
        for row in rows:
            # The error the controller ACTED ON, not the one its action produced (see the docstring).
            x_seen = row.get(self.decision_state_key, row.get(self.state_key, 0.0))
            e = x_seen - row.get(self.target_key, self.target)
            if abs(e) <= self.state_deadband:
                continue  # already at target: no correction was called for, so no direction to judge
            called_for += 1
            u = row.get(self.control_key, 0.0)
            if abs(u) <= self.control_deadband:
                continue  # a control of ~0 has no sign; counted only in ``active_fraction``
            active += 1
            if g * u * e > 0.0:  # the control pushed the state FURTHER from target
                wrong += 1
        return wrong, active, called_for

    # -- Objective API ---------------------------------------------------------------------------

    def evaluate(self, trajectory: Trajectory, regent_id: str = "regent:0") -> float:
        """Negated loss over ``trajectory``, which may be a WINDOW rather than a whole run.

        The cost is differenced here for the same reason ``components`` differences it: the Runner
        calls this on the interval since the regent's last decision to build the realized-performance
        signal, and an undifferenced running total would make that signal a monotone clock.
        """
        mse, cost = self._mse_cost(trajectory)
        return -(mse + self.lam * cost)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        mse, cost = self._mse_cost(trajectory)
        final_x = trajectory[-1].get(self.state_key, 0.0) if trajectory else 0.0
        mean_abs_x = (
            sum(abs(r.get(self.state_key, 0.0) - r.get(self.target_key, self.target)) for r in trajectory)
            / len(trajectory)
        ) if trajectory else 0.0
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and trajectory and not post_rows:
            # The run ENDED before the break, so there is no post-break evidence at all. An empty
            # window must not score 0.0 — that would make "terminate early" the winning post-break
            # policy and a reference calibrated on it would be measuring termination, not
            # governance. Worst-case it, exactly as StabilizationLoss/EpidemicLoss do.
            # ``wrong_sign_fraction`` is +inf too: it is a higher-is-worse quantity, and +inf cannot
            # be mistaken for a real fraction in [0, 1].
            inf = float("inf")
            return {
                "mse": mse, "mean_cost": cost, "loss": mse + self.lam * cost,
                "final_x": final_x, "mean_abs_x": mean_abs_x,
                "post_mse": inf, "post_cost": inf, "post_loss": inf,
                "wrong_sign_fraction": inf, "active_fraction": 0.0, "post_steps": 0.0,
                # Same key set as the normal branch: a ragged components() dict makes a ragged
                # results table, and the arm that goes ragged here is the degenerate one.
                "wrong_steps": 0.0, "active_steps": 0.0, "called_for_steps": 0.0,
            }
        post_mse, post_cost = self._mse_cost(post_rows)
        wrong, active, called_for = self._sign_counts(post_rows)
        return {
            "mse": mse,
            "mean_cost": cost,
            "loss": mse + self.lam * cost,
            "final_x": final_x,
            "mean_abs_x": mean_abs_x,
            "post_mse": post_mse,
            "post_cost": post_cost,
            "post_loss": post_mse + self.lam * post_cost,
            # THE DISCRIMINATOR, and the denominator it is conditional on. Report both or neither —
            # and with an empty denominator it is nan (undefined), never 0.0 (perfect).
            "wrong_sign_fraction": (wrong / active) if active else math.nan,
            "active_fraction": (active / called_for) if called_for else 0.0,
            "post_steps": float(len(post_rows)),
            # Raw counts, so a multi-seed summary can pool instead of averaging ratios.
            "wrong_steps": float(wrong),
            "active_steps": float(active),
            "called_for_steps": float(called_for),
        }
