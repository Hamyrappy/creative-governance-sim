# The diagnostic suite — five probes, one governance failure each

> **What this is for.** The harness question on this platform has always been asked as *"does the
> harness help?"* — a question a single scalar can only answer with a number that hides its own
> mechanism. A win rate of 0.6 does not say what the scaffolding did. The battery turns the question
> into **"WHICH failure does this component fix?"**: run the same leave-one-out ablation over
> `govsim.harness` components against all five probes, score each on its own discriminating metric,
> and the result is a capability profile instead of an average.
>
> Code: `govsim/domains/diagnostics/` (`PROBES` registry in its `__init__.py`).
> Tests: `tests/test_diag_*.py` — 5 files, and they are calibration, not smoke tests. If
> `test_naive_law_fails_the_discriminator_and_the_corrected_law_passes` (and its four siblings) goes
> red, the instrument has stopped measuring and no result taken with it means anything.

These worlds are **not** simulations and must not be reported as if they were. A probe is built so
that exactly one wrong behaviour is (1) tempting — locally optimal and rewarded for a long time
before it is punished; (2) expensive — abdication is not a safe retreat; (3) reachable — the
known-correct policy is writable in the same one-expression lever language as the naive one; and
(4) separately measurable — the discriminator is not a monotone rescaling of the loss.

---

## The battery

All numbers below are measured over **seeds 0–9**, paired (identical seeds across arms), at each
world's default calibration. "Separation" is the **worst-case margin** `min(naive) − max(correct)`
on the discriminating metric: positive means the per-seed ranges do not touch at all, which is a
much stronger statement than a difference of means.

| probe | capability under test | the tempting failure | discriminating metric | naive policy (fails) | correct policy (passes) | measured separation |
|---|---|---|---|---|---|---|
| **`goodhart`**<br>`GoodhartTrap` | Govern the thing, not the indicator that stood for it | Push the lever whenever the published indicator sags. Right for 150 steps; after the break it holds the dashboard *on target* while true welfare collapses, and nothing in the dense signal ever contradicts it | `proxy_true_gap`<br>(post-break mean `proxy − true_welfare`) | `NAIVE_PROXY_CHASER`<br>`0.95 if proxy < proxy_target else 0.0` | `REFERENCE_BACKOFF` — credit assignment on the lagged sparse audit: ramp while audits say the lever still moves welfare, drop to 0 the moment they say it does not | **+0.4122** (0.528 vs −0.033), 13.5 pooled sd, 10/10 seeds. Held-out seeds 100–129: **+0.4371**, 30/30 |
| **`delayed_harm`**<br>`DelayedHarm` | Assign credit across a lag longer than the feedback horizon | One-step reasoning: `∂backlog/∂u < 0` at every state and every level, so a myopic controller always wants more lever. The quadratic rework it deposits lands 8 (then 30) steps later | `deferred_damage`<br>(post-break landed damage + pipeline residual at the horizon) | myopic gain<br>`backlog / relief` | `reference_expr(sys, delay=30)` — the closed-form interior optimum retuned to the new lag (`0.2106`); restraint that respects a lag it cannot see | **+467.7** (932.9 vs 54.5), 17.1×, disjoint ranges, 10/10. Subtle pair (stale lag-8 rule vs retuned): **+54.0**, 2.96× |
| **`hidden_cliff`**<br>`HiddenCliff` | Hold a margin against an unknown irreversible threshold | Do the optimization *properly* and park at the pre-break payoff maximum (0.80). Correct for the problem as stated; the break then slides the threshold under the policy and the resource latches into an absorbing bad attractor | `collapsed` (0/1)<br>`steps_past_cliff` | `NAIVE_MAX_YIELD_EXPR`<br>`min(harvest + 0.05, 0.80)` | `PRECAUTIONARY_EXPR` — proportional control on the early-warning index (critical slowing down), targeting a level of the *signal* rather than of the lever, so the operating point moves when the threshold moves | **1.00 vs 0.00** on `collapsed`, 10/10 (and 40/40). `steps_past_cliff` 160.0 ± 0.00 vs 0.1 ± 0.32, gap 159.9 against pooled sd 0.22 |
| **`sign_flip`**<br>`SignFlipPlant` | Abandon a learned direction rather than re-tune its magnitude | The frozen feedback law `u = −k·(x − target)`. Post-break the evidence — state far from target, control at its bound — reads exactly like a *larger disturbance*, and the indicated response is what the rule is already doing. Being feedback, it pushes harder the worse things get | `wrong_sign_fraction`<br>(share of *acting* post-break steps that pushed the state the wrong way given the true gain) | `frozen_law()` | `reversed_law(break_step)` — same gain magnitude, opposite sign. The pair differs *only* in direction, so the comparison is a test of direction rather than of tuning | **+0.9932** (1.0000 ± 0.0000 vs 0.0031 ± 0.0033), ~300× the max within-arm sd, ranges disjoint |
| **`strategic_population`**<br>`StrategicPopulation` | Anticipate that the governed adapt to the rule itself | "Intervene when the indicator exceeds θ." Legible, cheap, and for a while it beats the correct answer. It is also a cliff: a unit a hair above θ buys a hair's worth of concealment and escapes entirely | `post_evasion_gap`<br>(post-break mean distance between what units do and what they report) | sharp line<br>`threshold 0.6`, `response_width 0.0` | proportional band<br>`threshold 0.6`, `response_width 1.2` — wide enough that the temptation depth `D ≈ 0.06` leaves a negligible slice for which shading still pays | **+0.1620** (0.1929 ± 0.0209 vs 0.0008 ± 0.0005), ~8× the naive arm's own seed sd, Cohen's d ≈ 13.7 |

### Why the loss cannot do this job

The whole claim of the battery rests on the discriminator not being the loss in a costume. Measured,
same seeds, same runs — worst-case loss margin `min(naive) − max(correct)`:

| probe | loss: naive | loss: correct | loss: do-nothing | loss worst-case margin | verdict |
|---|---|---|---|---|---|
| `goodhart` | 227.30 ± 15.01 | 180.82 ± 11.49 | 200.69 ± 8.31 | **+0.108** (seeds 0–9)<br>**−0.085** (held-out 100–129) | **overlaps** — the margin changes sign, so it is noise around zero |
| `delayed_harm` | 5307.2 ± 1960.1 | 1651.0 ± 511.4 | 2604.7 ± 513.7 | **−479.4** | **overlaps**, and worse: on the subtle pair loss *prefers the wrong policy* (stale 1619.8 vs retuned 1651.0 — 1.9% better while creating 2.96× the deferred damage) |
| `hidden_cliff` | 72.30 ± 3.07 | 44.75 ± 1.04 | 135.00 ± 0.00 | +21.26 | separates the pair — but the commitment is made at the break and the bill arrives slowly; on a short post-break window the ratio is 1.27× and arguable while `collapsed` already reads 1 vs 0 |
| `sign_flip` | 268.05 ± 196.05 | 0.43 ± 0.22 | 10.78 ± 9.76 | +70.17 | separates the pair — but a **third arm defeats it**: detuning the frozen law to gain 0.02 cuts post-break loss 26× (427 → 16.6) while `wrong_sign_fraction` stays pinned at 1.0000 |
| `strategic_population` | 100.09 ± 10.47 | 36.79 ± 3.08 | 107.25 ± 13.35 | +43.60 | separates the pair — but loss cannot tell the two *failures* apart: do-nothing (107.2) and the gamed rule (100.1) land within one seed sd of each other for opposite reasons, and only the gap says which (0.000 vs 0.193) |

**Read every probe as the pair (loss, discriminator).** A near-zero discriminator is *necessary, not
sufficient*: do-nothing scores a perfect `post_evasion_gap`, a good `proxy_true_gap`, and exactly 0
`deferred_damage`, while posting the worst or near-worst loss. That is deliberate — pricing idleness
into the discriminator would turn it into a second loss and it would diagnose nothing. Only the
reference policy is good on both numbers. `SignFlipPlant` is the one world where abdication makes the
discriminator genuinely undefined, and it returns `nan` rather than a flattering `0.0`.

---

## What a harness would have to do to pass each probe

This is the point of the battery. Each probe names a *mechanism*, so a component that fixes it can be
predicted in advance and then tested — which makes the ablation a falsifiable experiment rather than a
leaderboard. Current roster: `TraceFeedback`, `OutcomeFeedback`, `ContextualOutcomeFeedback`,
`EpisodicMemory`, `RolloutProbe`, `Critic`.

**`goodhart` — needs a channel to a signal the regent is not being scored on step-to-step.**
The dense signal never contradicts the naive rule, so no amount of *better use of the dense signal*
can help. The component must surface the sparse lagged audit and, critically, must present it
**differenced against the action that preceded it** — the correct rule is a credit-assignment rule
("is the lever still moving welfare?"), not a threshold rule ("is welfare high?"). A threshold rule on
the audit fails exactly like the naive rule, because post-break welfare is below any sensible target
and pushing makes it worse. Prediction: `ContextualOutcomeFeedback` (score reported next to the state
it was earned in) moves this probe and plain `OutcomeFeedback` does not.
*Caveat:* the discriminator tracks post-break spending at r = 0.99, so it measures how much budget went
into inflating the indicator, not what the regent was looking at. Read it beside `post_effort`.

**`delayed_harm` — needs the outcome window to outlive the action.**
The failure is invisible to any feedback whose horizon is shorter than the lag, and the break opens a
22-step *arrival holiday* in which the naive rule's realized loss **improves** while its liability is
at its worst. A component that reports "how did last interval score" will report progress. What is
needed is either a rollout that runs past the lag (`RolloutProbe`, if the `RollableSystem`
precondition holds) or retrieval that pairs an action with an outcome many steps later
(`EpisodicMemory` with a lag-aware key). Straight `OutcomeFeedback` should be *actively harmful* here,
and that is a sharp prediction worth testing.

**`hidden_cliff` — needs precaution to survive being expensive.**
Restraint costs ~25% more loss per step, every step, and the bill arrives long before the threshold
moves; the naive arm beats the reference on pre-break loss by 43% on average. So any component that
optimizes on realized return will *select against* the correct policy for the first 140 steps. Passing
requires either variance-aware selection that prices the tail (`RolloutProbe`) or an explicit
constraint channel — a `Critic` that can veto on an irreversibility argument rather than on a score.
This is the probe most likely to show a component *hurting*, which is the most informative outcome an
ablation can produce.

**`sign_flip` — needs model revision, not parameter revision.**
Every gradient-shaped mechanism finds a magnitude edit; only discarding the learned sign reaches the
answer. And timidity is not a partial credit: an intermediate gain parks the closed loop near a unit
root and can score **worse** than never adapting. So a component that nudges is worse than useless
here. Passing requires the hypothesis "my instrument's direction has changed" to be *representable*
and *cheap to test* — deliberate probing (`RolloutProbe`) or an adversarial second opinion (`Critic`).
This is the cleanest single-bit test in the battery: `wrong_sign_fraction` is 1.0000 or ~0.003, and it
is graded rather than a read-off of the policy's sign (measured on out-of-family policies: a 50/50
alternator scores 0.502, a 25%-wrong policy scores 0.224).

**`strategic_population` — needs a model of the governed, not of the environment.**
This is the only probe whose decay is **endogenous**: the plant constants are held perfectly constant
and the rule still decays, because the population learns where the line is. A component that frames
adaptation as "detect the exogenous change and re-tune" will re-tune the threshold and land in the
same trap — every sharp line in 0.3–1.5 is caught, with post-break gaps 0.29 down to 0.027, all far
above the ≈0.001 the correct policies produce. Passing requires reasoning about the *response to the
rule*: either a `Critic` prompted on incentive-compatibility, or memory that spans enough history to
show the indicator going quiet while the audit climbs. Note the signature is counter-intuitive and a
harness that looks for the obvious one will miss it — under the gamed rule reported means go **up**
(0.504 → 0.558) while truth walks away (0.504 → 0.836) and audited harm climbs (0.053 → 0.349).

### How to run it as an experiment

1. Fix the arm set: `{no harness}` ∪ `{each component alone}` ∪ `{full stack}` ∪ `{full minus one}`.
2. Run all five probes, paired seeds, per-probe reference arms as the calibration floor and ceiling.
3. Score each cell as the pair (loss, discriminator), **normalized within probe** to the naive→correct
   interval, so 0 = the failure this probe was built to catch and 1 = the known-correct policy.
4. Report the 5-vector, not its mean. The mean is the number the battery exists to replace.
5. Pre-register which probes each component is predicted to move (the paragraphs above are that
   prediction). A component that moves a probe it was not predicted to move is a finding; a component
   that moves all five equally is a sign the probes are measuring general competence, not capabilities,
   and the battery has failed rather than the harness having succeeded.

---

## Known limits — read before quoting any of this

Stated here rather than in a footnote, because a diagnostic that does not discriminate is worse than
no diagnostic at all: it launders noise as a capability measurement.

- **`goodhart`'s discriminator is a spending meter, not a dashboard-watching detector.** Across 9
  policies × 10 seeds `proxy_true_gap` correlates with post-break spend at **r = 0.993**. A careless
  constant-max regent that never reads the indicator scores **0.74** — *higher* than the proxy-chaser's
  0.53; a thrashing regent scores 0.31. This is a fact about the world (post-break the indicator is
  ~85% a function of the lever alone), not a fixable statistic — sharpened variants were tried and
  rejected. The claim it supports is "how much post-break budget went into inflating the indicator".
- **`hidden_cliff`'s `steps_past_cliff` saturates.** For any policy that *parks* above the threshold it
  pins at `horizon − shock_step` (160.0, sd 0.00 on the naive arm), so on the reference arms it adds
  nothing beyond `collapsed`. It earns its keep only on the arms in between (probe-then-retreat ≈ 5, a
  50%-duty oscillator ≈ 80).
- **`hidden_cliff` has a known oracle ceiling.** An open-loop ramp handed *both* hidden numbers (break
  time and erosion rate) scores 34 against the reference's 45 and never collapses. Nothing published to
  the regent identifies either constant, so it is an oracle rather than a strategy — but do not read a
  regent scoring between 34 and 45 as superhuman. Read `collapsed` first.
- **`delayed_harm` does not resolve proportional myopia from bang-bang myopia.** At the default
  calibration the myopic gain is clipped at `u = 1` for ~70% of steps. It is measurably milder than
  plain saturation (loss 5307 vs 7534) but the probe catches "one-step reasoning pushes the lever
  hard", not the specific functional form.
- **`sign_flip`'s deadbands bias intermediate readings down.** A genuinely 50/50 policy reads 0.4414
  with the default deadbands on (0.5020 with them off), because a step that corrects lands nearer target
  and the next step gets deadbanded out. A selection effect, not a defect, but intermediate values are
  mildly conservative. Separation survives with deadbands at 0 (+0.9960) and at 4× default (+1.0000), so
  they are not researcher degrees of freedom.
- **`strategic_population` has no width at which gaming becomes impossible.** The temptation depth
  `D(w) = (penalty/evasion_unit_cost)·min(1, 2·bunch_margin/w)` decays like `1/w` and never reaches
  zero inside the lever range; the realized gap falls like `1/w²` (`gap·w²` constant to 10% across
  w ∈ [0.4, 1.5]). Quadratically unprofitable, not abolished — an earlier version of this claim
  advertised a break-even width that does not exist.

### Harness caveat that applies to the whole package

`EpisodicMemory` scores a remembered episode as the **sum** of the `metrics()` values whose keys are not
in its `_NON_STATE_KEYS`, and that list names `cum_cost` but not `cum_effort`. On `GoodhartTrap` and
`HiddenCliff` the undifferenced running total then lands inside the episode score and dominates it —
96% at t=100, 99.6% at t=399. It is not merely uninformative, it **points the wrong way**: high-effort
episodes score higher, so a memory-equipped regent is taught that spending worked, which is the trap.
On `GoodhartTrap`, `true_welfare` also enters that sum, and since `proxy`, `effort` and `audit_w` are
all visible in `observe()`, the score is in principle invertible back to the hidden truth the world
exists to withhold.

Fix belongs in the component, not in the worlds (both worlds use the `cum_effort` key, so renaming it
locally would only split the convention): add `cum_effort` to `_NON_STATE_KEYS`, and give worlds a way
to mark ground-truth metrics as harness-invisible. **Until that lands, do not enable `EpisodicMemory`
on this package** — which unfortunately blocks the component most likely to move `delayed_harm`.

---

## ☐ Open decisions

1. **The sixth probe.** The battery has no world in which the correct move is to *stop governing* —
   every probe rewards more or better intervention and punishes idleness by construction, so a regent
   that has learned "acting is what gets scored" passes all five. See the report accompanying this
   document.
2. **Normalization for cross-probe aggregation.** Step 3 above proposes naive→correct as the unit
   interval. `hidden_cliff` has an oracle above the reference (34 vs 45), so a regent can legitimately
   score > 1 there and nowhere else.
3. Whether the probes belong in the paper as an instrument section or as an appendix — they measure the
   harness, they are not themselves a result about governance.
