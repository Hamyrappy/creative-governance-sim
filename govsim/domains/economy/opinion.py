"""
``OpinionPolity`` — a bounded-confidence opinion distribution governed through a moderation lever.

The world is a public sphere summarized by where its people stand on one issue. Opinions live on a
fixed grid over ``[-1, 1]``; the state is the population *mass* in each bin. Three forces move that
mass every step: bounded-confidence attraction (each opinion drifts toward the mean of what it still
listens to), a polarizing pull toward whichever of two ideological camps is nearer, and a small churn
that keeps individuals wandering. Left alone the polity hollows out at the centre and settles into
two blocs at roughly ``±0.85``, where mean squared distance from the centre sits near 0.5.

The authority's one instrument is ``moderation``: an effort to widen who hears whom and to fund a
neutral common ground at the centre of the scale. It is priced, and the price is charged on the
effort *ordered*, not on the mixing achieved.

**Why a distribution and not N agents.** A population of agents would carry its own sampling noise,
would put a Monte-Carlo error bar on every measurement, and would make the ``clone()`` a rollout
needs expensive. A 21-bin histogram is exact given the seed, cheap to deepcopy, and conserves mass
by construction — which is what turns the boundedness claim below into a proof rather than a hope.

**BOUNDED DYNAMICS (no arm can diverge).** The state is only ever advanced by ``_transfer``, in which
each bin sends a fraction ``|f| <= 0.45`` of its own mass to one neighbour and the flux off either end
of the grid is pinned to zero. Three consequences hold for every parameter setting, every policy, and
every shock. Total mass stays 1, because each transfer subtracts from one bin exactly what it adds to
another. Every bin stays non-negative, because within a pass a bin never sends more than half of what
it holds, and each pass starts from the non-negative result of the last.
And the support never leaves ``[-1, 1]``, because there is nowhere off-grid to send mass to. So every
reported statistic is a bounded functional of a probability vector on a fixed grid: polarization lies
in ``[0, 1]``, the mean opinion in ``[-1, 1]``. No configuration can blow up — divergence here is not
merely unlikely, it is unrepresentable. Both the conservation and the non-negativity are pinned by
tests over long runs at extreme parameters.

**Two design choices that look cosmetic and are not.** The platform needs an interior optimum before
any shock can move anything, and two earlier versions of this world failed to have one. Both failures
are worth recording, because the fixes are the only reason the calibration below works:

  - *A Gaussian confidence kernel makes the world bistable.* Two camps at ``±0.85`` would hear each
    other with weight ``~1e-5`` — deaf, for practical purposes — so the polarized equilibrium stopped
    depending on the instrument at all. Moderation then bought nothing until it bought everything,
    the burden was convex in the lever, and the loss was minimized at a corner for every price. The
    Cauchy kernel used here leaves a residual cross-camp audience, so the equilibrium separation
    moves continuously with the instrument.
  - *Without the civic centre, nothing distinguishes the middle of the scale.* Confidence attraction
    pulls each opinion toward the mean of its own audience, so a polity that has already agreed is
    at rest wherever it happens to stand — and the camp pull then walks that consensus out to a pole
    anyway. Moderation only postponed the walk, so the score measured a delay, and on the seeds whose
    initial tilt was strong enough to unify the polity on its own it measured nothing at all. The
    civic centre gives the instrument something to actually build.

Calibration at the defaults, 16 seeds, horizon 240: mean polarization under a constant lever runs
0.50 at ``moderation = 0`` down to 0.02 at ``moderation = 1``. Against ``PolarizationLoss(lam=0.2)``
the best constant lever is ``0.4``, with never-moderating 4.0x worse and always-maximum 1.8x worse —
an interior optimum with room on both sides. Re-run that sweep after touching any parameter here: a
regime whose optimum has slid to a corner has zero adaptation headroom by construction, and will
report a null no matter which agent governs it.

**THE INSTRUMENT SHOCK.** ``shock_params`` overwrites named parameters at ``shock_step``, exactly as
``SIRSystem`` does. The entry that matters is ``moderation_efficacy``: the share of ordered moderation
that actually reaches the public sphere. When it collapses — audiences migrate to platforms the
authority does not reach, or simply learn to route around it — the same order buys a fraction of the
mixing at the full political price. Efficacy is deliberately absent from ``observe``: the regent sees
the distribution it governs and the effort it ordered, never the coefficient joining them. Noticing
from the trace that the instrument stopped working is the task, and publishing the parameter would
delete the problem.

That break is not absorbable by re-tuning a threshold, which is the point of building it here. A rule
of the form "moderate harder when polarization exceeds θ" answers an efficacy collapse by spending
*more*: polarization rises, so the rule fires, and the spending lands on nothing. Holding the
pre-shock policy at the calibrated ``0.4`` and sweeping the post-shock level instead (16 seeds,
``lam=0.2``) shows where that goes. A mild collapse to ``0.5`` leaves the best post-shock level at
``0.4`` — the instrument still works, and there is nothing to adapt to, which is the null this design
predicts. A severe collapse to ``0.05`` moves it to ``0.0``: abandon the drive, bank the political
capital, and accept the polarization you can no longer prevent. A frozen ``0.4`` loses 1.16x there,
and 1.44x at ``lam=0.4``. Spending *nothing* is the one response a threshold on a rising observable
cannot produce, which is why the headroom survives feedback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from govsim.core.objective import Objective, Trajectory
from govsim.core.system import Observation, StepInfo
from govsim.domains.scalar.systems import LeverSystem

#: Per-step cap on the fraction of a bin's mass that may cross to a neighbour. Comfortably below the
#: 0.5 at which a bin could send away more than it holds, which is what keeps every bin non-negative
#: no matter how large a velocity the parameters and the noise conspire to produce.
_MAX_TRANSFER = 0.45

#: Bins at least this far from the centre count as "extreme" in the reported extreme-mass share.
_EXTREME_THRESHOLD = 0.8


class OpinionPolity(LeverSystem):
    """Bounded-confidence opinion dynamics on a mass histogram, governed by a priced moderation lever.

    State: ``mass``, a length-``n_bins`` probability vector over opinions on a fixed grid spanning
    ``[-1, 1]``. See the module docstring for the conservation and boundedness argument.

    Lever: ``moderation`` in ``[0, 1]``. Its realized strength is ``moderation * moderation_efficacy``;
    that product both widens the confidence kernel and funds the civic centre, while the bill is
    computed from ``moderation`` alone.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        p = params or {}
        self.n_bins: int = int(p.get("n_bins", 21))
        if self.n_bins < 5 or self.n_bins % 2 == 0:
            raise ValueError("n_bins must be an odd integer >= 5 (an odd grid has an exact centre bin)")
        self.grid: np.ndarray = np.linspace(-1.0, 1.0, self.n_bins)
        self.h: float = float(self.grid[1] - self.grid[0])
        self._sq_gap: np.ndarray = (self.grid[:, None] - self.grid[None, :]) ** 2
        self.dt: float = float(p.get("dt", 0.12))
        # -- the three forces -----------------------------------------------------------------
        self.attraction: float = float(p.get("attraction", 0.80))
        self.confidence_radius: float = float(p.get("confidence_radius", 0.35))
        self.polar_strength_init: float = float(p.get("polar_strength", 0.35))
        self.camp_pole: float = float(p.get("camp_pole", 0.85))    # where a camp's faithful settle
        self.camp_width: float = float(p.get("camp_width", 0.35))  # how fast one is claimed by a camp
        self.churn: float = float(p.get("churn", 0.03))            # undirected individual wandering
        # -- what the instrument buys: a wider audience, and a bigger neutral common ground -----
        self.radius_gain: float = float(p.get("radius_gain", 1.5))
        self.civic_base: float = float(p.get("civic_base", 0.05))
        self.civic_gain: float = float(p.get("civic_gain", 3.0))
        self.moderation_efficacy: float = float(p.get("moderation_efficacy", 1.0))
        self.moderation_cost: float = float(p.get("moderation_cost", 1.0))
        # -- the unseen structural break (the SIRSystem mechanism, unchanged) -------------------
        ss = p.get("shock_step", 120)
        self.shock_step: int | None = None if ss is None else int(ss)
        self.shock_params: dict[str, float] = dict(p.get("shock_params", {}))
        # -- per-seed heterogeneity -------------------------------------------------------------
        # Without this every seed replays one trajectory, a paired shared-seed design has nothing to
        # pair over, and the bootstrap CI is a zero-width interval dressed up as a result. Four
        # independent draws: where the public starts, how spread out it is, which wing is heavier,
        # and how strongly it polarizes — so seeds differ in the *problem*, not only in the noise.
        self.init_center_sigma: float = float(p.get("init_center_sigma", 0.18))
        self.init_width: float = float(p.get("init_width", 0.45))
        self.init_width_sigma: float = float(p.get("init_width_sigma", 0.25))
        self.init_tilt_scale: float = float(p.get("init_tilt_scale", 0.6))
        self.polar_sigma: float = float(p.get("polar_sigma", 0.22))
        # Per-step agitation: the polarizing pull fluctuates with the news cycle. Multiplicative and
        # lognormal, so no realization of the noise can flip the pull's sign.
        self.agitation_sigma: float = float(p.get("agitation_sigma", 0.10))
        # Snapshot EVERY scalar parameter, by reflection, before the first reset. ``shock_params``
        # setattrs arbitrary names, so ``reset`` has to be able to put any of them back; a hand-kept
        # list of ``*0`` shadows rots the moment a parameter is added, and this world shipped with
        # exactly that rot — `attraction`, `camp_pole`, `camp_width`, `churn`, `dt`,
        # `moderation_cost` and `agitation_sigma` had no shadow, so a shock on any of them leaked
        # into the next run of the same object. `moderation_cost` is the dangerous one: a leaked
        # re-pricing silently bills a later arm at the shocked rate and the comparison between arms
        # is then measuring the leak. Reflection cannot drift out of sync with the parameter list.
        self._pristine: dict[str, float] = {
            name: value for name, value in vars(self).items()
            if not name.startswith(("_", "shock_"))
            and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self.reset(int(p.get("seed", 0)))

    @property
    def lever_attrs(self) -> dict[str, tuple[float, float]]:
        return {"moderation": (0.0, 1.0)}

    # -- lifecycle ---------------------------------------------------------------------------

    def _initial_mass(self) -> np.ndarray:
        centre = float(self.rng.normal(0.0, self.init_center_sigma))
        width = self.init_width * float(np.exp(self.rng.normal(0.0, self.init_width_sigma)))
        tilt = float(self.rng.uniform(-self.init_tilt_scale, self.init_tilt_scale))
        shape = np.exp(-0.5 * ((self.grid - centre) / max(width, 1e-3)) ** 2)
        # The tilt makes the two wings unequal, so seeds differ in which camp ends up dominant. A
        # perfectly symmetric start would sit on a knife edge that the dynamics resolve the same way
        # every time, which is heterogeneity in the noise but not in the governance problem.
        shape = shape * np.clip(1.0 + tilt * self.grid, 0.05, None)
        return shape / shape.sum()

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        # Restore every parameter a shock may overwrite, so a reused object starts pristine. This
        # runs BEFORE the draws below, so the heterogeneity knobs a shock touched are pristine too.
        for name, value in self._pristine.items():
            setattr(self, name, value)
        self.polar_strength: float = self.polar_strength_init
        if self.polar_sigma > 0:
            self.polar_strength *= float(np.exp(self.rng.normal(0.0, self.polar_sigma)))
        self.mass: np.ndarray = self._initial_mass()
        self.moderation: float = 0.0
        self.cum_cost: float = 0.0
        self._t: int = 0
        self.polarization: float = self._polarization()
        self.prev_polarization: float = self.polarization
        self._levers.clear()

    # -- summary statistics of the distribution -----------------------------------------------

    def _polarization(self) -> float:
        """Second moment about the centre: 0 when the polity agrees at 0, 1 when it has split."""
        return float(self.mass @ (self.grid ** 2))

    def _mean_opinion(self) -> float:
        return float(self.mass @ self.grid)

    def _extreme_mass(self) -> float:
        return float(self.mass[np.abs(self.grid) >= _EXTREME_THRESHOLD].sum())

    def _dispersion(self) -> float:
        mu = self._mean_opinion()
        return float(np.sqrt(max(0.0, self.mass @ ((self.grid - mu) ** 2))))

    # -- dynamics -------------------------------------------------------------------------------

    def _transfer(self, mass: np.ndarray, fraction: np.ndarray) -> np.ndarray:
        """Move ``|fraction|`` of each bin's mass one step along the grid; zero flux off the ends.

        The one primitive every update is built from, so mass conservation and non-negativity are
        properties of *this* function rather than of each force in turn.
        """
        f = np.clip(fraction, -_MAX_TRANSFER, _MAX_TRANSFER)
        right = np.clip(f, 0.0, None) * mass
        left = np.clip(-f, 0.0, None) * mass
        right[-1] = 0.0
        left[0] = 0.0
        out = mass - right - left
        out[1:] += right[:-1]
        out[:-1] += left[1:]
        return out

    def step(self) -> StepInfo:
        self._reeval_levers()
        if self.shock_step is not None and self._t == self.shock_step:
            for name, value in self.shock_params.items():
                setattr(self, name, float(value))
        realized = self.moderation * self.moderation_efficacy  # the lever is the order; this lands
        radius = self.confidence_radius * (1.0 + self.radius_gain * realized)
        civic = self.civic_base + self.civic_gain * realized
        agitation = float(np.exp(self.rng.normal(0.0, self.agitation_sigma))) if self.agitation_sigma > 0 else 1.0
        pull = self.polar_strength * agitation

        # Bounded confidence with a heavy-tailed audience: each bin drifts toward the mass-weighted
        # mean of the opinions it hears, and hears distant ones faintly rather than not at all.
        kernel = 1.0 / (1.0 + self._sq_gap / (radius * radius))
        heard = kernel * self.mass[None, :]
        # The civic centre: a neutral common ground of weight `civic` sitting at opinion 0, heard
        # through the same kernel. It is what the instrument actually builds, and without it nothing
        # in this world distinguishes the centre — a unified polity would sit wherever its camp left
        # it, moderation would only postpone the drift to a pole, and the score would be measuring a
        # delay rather than a governed outcome. It enters the denominator only, because its own
        # opinion is 0 and contributes nothing to the numerator.
        civic_share = civic / (1.0 + (self.grid ** 2) / (radius * radius))
        local_mean = (heard @ self.grid) / np.maximum(heard.sum(axis=1) + civic_share, 1e-300)
        # Each opinion is also claimed by the nearer camp. tanh both hollows out the centre (near 0
        # the claim points outward) and stops at a pole strictly inside the grid, so the polarizing
        # force never asks for mass to leave [-1, 1] — boundedness holds before the clip, not because
        # of it.
        camp = self.camp_pole * np.tanh(self.grid / self.camp_width)
        velocity = self.attraction * (local_mean - self.grid) + pull * (camp - self.grid)

        self.mass = self._transfer(self.mass, velocity * self.dt / self.h)
        if self.churn > 0.0:
            # Undirected wandering, as a symmetric pair of transfers. It keeps the equilibrium a
            # distribution with width rather than a point mass, which is what makes the summary
            # statistics respond smoothly to the lever instead of in steps of one bin.
            spread = np.full(self.n_bins, self.churn)
            self.mass = self._transfer(self._transfer(self.mass, spread), -spread)

        # Billed on the order, not on the mixing: a moderation drive nobody sees still spends the
        # authority's standing. That asymmetry is what makes an efficacy collapse expensive to ignore.
        self.cum_cost += self.moderation * self.moderation_cost
        self.prev_polarization = self.polarization
        self.polarization = self._polarization()
        self._t += 1
        return StepInfo(terminated=False, truncated=False, info={})

    # -- views ------------------------------------------------------------------------------------

    def observe(self, viewer_id: str = "regent:0") -> Observation:
        return Observation(
            # `moderation_efficacy` is absent on purpose. The regent sees the distribution, the
            # previous distribution, and the effort it ordered; whether that effort landed is an
            # inference, and that inference is the governance problem this world poses.
            vars={
                "polarization": self.polarization,
                "prev_polarization": self.prev_polarization,
                "extreme_mass": self._extreme_mass(),
                "mean_opinion": self._mean_opinion(),
                "dispersion": self._dispersion(),
                "moderation": self.moderation,
                "cum_cost": self.cum_cost,
                "t": float(self._t),
            },
            scope=viewer_id,
            t=self._t,
        )

    @property
    def time(self) -> int:
        return self._t

    def metrics(self) -> dict[str, float]:
        # The enacted lever is in the record next to the state it produced: the governance question
        # is what the authority DID, and a trajectory that logs only outcomes cannot answer it.
        return {
            "polarization": self.polarization,
            "extreme_mass": self._extreme_mass(),
            "mean_opinion": self._mean_opinion(),
            "dispersion": self._dispersion(),
            "moderation": self.moderation,
            "cum_cost": self.cum_cost,
            "t": float(self._t),
        }


class PolarizationLoss(Objective):
    """Minimize accumulated polarization + λ·moderation cost (negated → higher is better).

    **λ is the substantive knob, not a nuisance parameter.** It prices a step of moderation against a
    step of polarization and therefore decides whether the best institution is interior or a corner.
    At a corner — "always moderate at the maximum", or "never moderate" — no shock can move the
    optimum, so a regime calibrated there has zero adaptation headroom by construction and reports a
    null whatever the agent does. The default 0.2 comes from sweeping constant moderation levels
    against ``OpinionPolity``'s defaults: it puts the best constant lever at 0.4, with never-moderating
    4.0x worse and always-maximum 1.8x worse. Re-run that sweep if either side's parameters move.

    ``post_shock_step`` mirrors ``EpidemicLoss``: ``components`` also reports the metric over the rows
    at-or-after the break, because a pre-shock-optimal rule is by definition near-optimal on the
    pre-shock half, and averaging the halves dilutes — and can invert — the verdict.
    """

    def __init__(self, lam: float = 0.2, polarization_key: str = "polarization",
                 cost_key: str = "cum_cost", post_shock_step: int | None = None,
                 step_key: str = "t") -> None:
        self.lam = lam
        self.polarization_key = polarization_key
        self.cost_key = cost_key
        self.post_shock_step = post_shock_step
        self.step_key = step_key

    def describe(self) -> str:
        return (
            "YOUR MANDATE. Keep the public sphere from splitting apart, without spending more "
            "political capital on the effort than the split is worth.\n"
            "  - polarization accrues every step as the population's mean squared distance from the "
            "centre of the opinion scale: 0 if everyone agrees at the centre, 1 if the polity has "
            "split into two blocs at the extremes;\n"
            "  - moderation cost accrues on the EFFORT YOU ORDER, not on the mixing it achieves: you "
            "are billed for the moderation drive whether or not it reaches anyone;\n"
            f"  - the score is total polarization PLUS {self.lam} times total moderation cost, so a "
            f"unit of moderation is worth ordering only while it removes more than {self.lam} units "
            "of polarization.\n"
            "Lower total is better. Both terms matter: a polity held together by permanent maximum "
            "moderation and a polity left to tear itself apart are both failures."
        )

    def _burden_cost(self, rows: Trajectory) -> tuple[float, float]:
        """(polarization burden, moderation cost) over ``rows``.

        The cost is DIFFERENCED across the window rather than read off the last row. ``cum_cost`` is
        a running total from t=0, so on a post-shock window the undifferenced value bills the
        post-shock policy for every drive ordered before the break — which turns the score into a
        clock reporting how late the window starts. That bug has been shipped here before.
        """
        if not rows:
            return 0.0, 0.0
        burden = sum(row.get(self.polarization_key, 0.0) for row in rows)
        cost = rows[-1].get(self.cost_key, 0.0) - rows[0].get(self.cost_key, 0.0)
        return burden, cost

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

        Differenced for the same reason ``components`` is: the Runner calls this on the interval
        since the regent's last decision to build the realized-performance signal the harness shows
        the model, and an undifferenced running total makes that signal fall monotonically no matter
        what the policy does.
        """
        burden, cost = self._burden_cost(trajectory)
        return -(burden + self.lam * cost)

    def components(self, trajectory: Trajectory, regent_id: str = "regent:0") -> dict[str, float]:
        series = [row.get(self.polarization_key, 0.0) for row in trajectory]
        burden, cost = self._burden_cost(trajectory)
        post_rows = self._post_shock_rows(trajectory)
        if self.post_shock_step is not None and trajectory and not post_rows:
            # The run ended BEFORE the break, so there is no post-shock evidence. Scoring that 0.0
            # would crown whichever policy ends the run soonest; worst-case it instead, exactly as
            # EpidemicLoss does. Early termination is the collapse detector's business, not the score's.
            inf = float("inf")
            return {
                "total_polarization": sum(series),
                "peak_polarization": max(series) if series else 0.0,
                "final_polarization": series[-1] if series else 0.0,
                "cum_cost": trajectory[-1].get(self.cost_key, 0.0),
                "loss": burden + self.lam * cost,
                "post_polarization": inf, "post_peak_polarization": inf,
                "post_cost": inf, "post_loss": inf,
            }
        post_burden, post_cost = self._burden_cost(post_rows)
        post_series = [row.get(self.polarization_key, 0.0) for row in post_rows]
        return {
            "total_polarization": sum(series),
            "peak_polarization": max(series) if series else 0.0,
            "final_polarization": series[-1] if series else 0.0,
            "cum_cost": trajectory[-1].get(self.cost_key, 0.0) if trajectory else 0.0,
            "loss": burden + self.lam * cost,
            "post_polarization": post_burden,
            "post_peak_polarization": max(post_series) if post_series else 0.0,
            "post_cost": post_cost,
            "post_loss": post_burden + self.lam * post_cost,
        }
