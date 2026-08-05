# STATUS — WHAT-first gates & phase progress

> These are the **executable preconditions** of doc-09 §1.3: the `Runner` refuses to run an
> `Experiment` without a `Hypothesis` (claim + named baseline). The docs in this folder are the
> human side of that gate — they record the *scientific* decisions (doc-08 §8) that the code
> cannot invent.

## The headline change (2026-08): measure the environment before crediting the agent

The project's H1 was, until this point, "the harnessed LLM adapts to an unseen shock better than a
frozen controller", tested on the scalar cubic plant. That experiment was **unanswerable as
specified**, and we found out by measuring rather than by running it.

`govsim/analysis/calibration.py` computes two references inside one shared policy family — `frozen`
(optimal on the pre-shock world, then held fixed) and `oracle` (optimal on the post-shock window) —
and their ratio, the **adaptation headroom**. It bounds what any controller could recover.

| Regime | Headroom | Verdict |
|---|---|---|
| cubic scalar, state-dynamics shock (`param_A`↑, `cubic_coeff`↑, kick) | **1.00–1.26×** over 45 cells | near-null by construction |
| cubic scalar, instrument shock with sign flip | 270–2865×, or divergence | ungraded; a stability cliff, not a control problem |
| coupled multi-state, periodic aux shocks | **1.01×** | near-null |
| SIR, transmissibility shock (`beta0`×1.8) | **1.00×** | absorbed by feedback |
| **SIR, instrument-efficacy collapse (lockdown 1.0→0.25, λ=0.08)** | **1.70×** | ✅ **the flagship** |

**Structural finding 1:** a feedback rule ("intervene when the observable exceeds θ") *absorbs*
shocks to the state — prevalence rises, the rule fires more often, and it stays near-optimal without
anyone changing it. It cannot absorb a shock to **instrument efficacy**, because the mapping from
observation to correct action has changed rather than the observation. That is the Lucas critique in
miniature, and as far as we can measure it is the only regime where the adaptation question is
answerable at all.

**Structural finding 2 — the decomposition.** What a stale rule costs factors exactly:

```
staleness            =  adaptation headroom      ×  robustness headroom
L(frozen)/L(switch)  =  L(best_fixed)/L(switch)  ×  L(frozen)/L(best_fixed)
```

Only the **adaptation** term is recoverable by changing behaviour mid-run; the **robustness** term is
recoverable by having legislated a different standing rule in the first place.

| regime | staleness | **adaptation** | robustness |
|---|---|---|---|
| epidemic (efficacy 1.0→0.25, λ=0.08) | 1.315 | **1.186** | 1.109 |
| epidemic_severe (efficacy→0.0, λ=0.15) | 1.511 | **1.000** | 1.511 |
| scalar cubic (state shock) | 1.125 | **1.039** | 1.083 |

The severe row is the instructive one, and it **overturned the expectation that added it**: a harsher
break was assumed to leave more for adaptation to recover, and in fact leaves none — the optimal
policy is identical either side of the break, so the entire 51% is a rule-design failure. An agent
evaluated there would look rigid while facing no adaptive problem at all. Never report staleness
alone; it cannot tell the two diagnoses apart, and they call for different remedies.

**Reference-construction traps** (both found by the reference contradicting itself, both now guarded):
- scoring only the post-break window **rewards passivity** ⇒ score the full horizon against
  `best_fixed`;
- composing the switching reference from two separately-optimal legs is wrong when state carries
  across the break — it produced an "upper bound" that a fixed law beat (0.92×). Search the pair
  jointly; `recalibrate.py` now hard-flags `adaptation < 1` as an under-powered search.

Two calibration artifacts were found and fixed on the way; both had produced clean, plausible,
wrong numbers:
- calibrating a reference on a *component* (`mse`, `total_infected`) instead of the objective puts
  every optimum at a grid corner, and corner optima are regime-invariant ⇒ every regime read 1.0×;
- an empty post-shock window scored `0.0`, so "let the epidemic burn out before the shock" won the
  post-shock comparison by emptying it. Now `+inf`, and the disease is endemic so it cannot be
  outlasted.

## Five validity gates — run these before believing any LLM-arm table

Every one of them caught a defect in this project that had already produced a clean, plausible,
publishable-looking number. They are automatic now (`scripts/analyze_matrix.py`,
`scripts/recalibrate.py`), and each refuses to let the tables be read rather than merely warning.

| Gate | What it catches | What it caught here |
|---|---|---|
| **no-action rate** (>2% disqualifying) | a model that reasons past its `max_tokens` emits nothing, the previous law stays installed, and the arm silently becomes "sticky policy" — *correlated with the treatment*, since harness channels lengthen the prompt | outcome arm **32.5%**, no-harness arm **0%**, at `max_tokens=1500` |
| **channel liveness** | a channel that fires but carries no information; rank-correlates what it reported against what it should track | all three channels dead: trace never fired; outcome and memory both reported a **clock** (`cum_cost` read undifferenced) |
| **anchor consistency** | stored reference runs enacting laws from a superseded calibration | a store held a single-lever law two calibrations old; R moved 0.81 → 1.32 and the headline contrast flipped sign |
| **headroom ≥ 1** for the switching reference | a "clairvoyant" adaptor a fixed policy can beat is impossible, so it is a free self-check on the reference search | composed (pre,post) legs gave 0.92× before the pair was searched jointly |
| **model responsiveness** | the four gates above certify the APPARATUS, none certifies the SUBJECT. A model can pass all four, emit one constant on every decision, and produce a flat ablation that reads as "these components do not help" — a statement about the harness derived from a fact about the model | `qwen3.5:0.8b`: memory live on **100%** of prompts, **400/400 identical decisions**, six arms agreeing to four decimals. `gemma-3-27b` on the same worlds/seeds: TV **0.38**, top-policy share **50%** |

**The fifth gate is two-stage on purpose.** `TraceFeedback` reports rejected actions; when the agent
emits only valid ones it has nothing to say, and that arm *should* score identically to bare —
calling that a null about the component is backwards, because the component was never on trial. So
stage 1 asks whether the channel altered the prompt at all (measured by diffing prompts per
`(seed, step)`, never by grepping for a header string, which goes stale when wording changes); only
a *live* channel is eligible to be called deaf. Below a 20% liveness floor the arm-level verdict is
not identifiable and is reported `THIN` rather than `DEAF`.

**Two distinct no-action failures, with OPPOSITE remedies.** The no-action gate says an arm is
contaminated; it does not say why, and guessing wrong makes it worse.

| | truncation | reasoning collapse |
|---|---|---|
| seen at | `max_tokens=1500` | `max_tokens=12000` |
| signature | answer cut off mid-emission | `completion_tokens=0`, `total_tokens≈12862` |
| text | partial action | pure `<thought>`, ending in a verbatim repetition loop |
| fix | **raise** the cap | **raising the cap only buys more loop** — must re-ask |

In the recorded sweep **100% of no-action decisions were collapses**, and they track exactly one
factor: outcome present **7.8 / 5.8 / 4.0 / 3.8%**, outcome absent **≤ 0.2%**. `completion_tokens==0`
predicts "produced no action" on **1003 of 1003** cached calls (against 6389/6433 the other way), so
it is the signature to gate on. The mitigation re-asks with a nudge that says only *stop
deliberating and answer* — never what to answer, or it becomes a treatment applied preferentially to
the context-heavy arms — and its residual is reported (`deliberation_unrecovered`), because a live
test recovered only 2 of 3.

**Statistics, corrected the same way.** The percentile bootstrap under-covers at n=20 (measured
0.921 vs nominal 0.95 ⇒ the "CI excludes 0" rule rejects at ~8%). BCa does not help — a symmetric
paired difference has little bias or skew; what is missing is variance uncertainty. We use
**bootstrap-t** (0.935) and require a **Wilcoxon** agreement, and report a **minimum detectable
effect** with every null.

**And one design rule:** the regent must be *told its objective*. It was not, while every reference
was exhaustively optimized for `burden + λ·cost` — so the study was partly measuring
mandate-guessing. `Objective.describe()` now reaches every prompt, and adding it moved the headline
by more than any harness channel.


## Measured results (2026-08-05)

**Adaptation headroom is scarce across worlds, not just across configurations of one world.**
Decomposing all ten library worlds (`scripts/decompose_library.py`), only **3 of 10** clear 1.1x and
**1 clears 1.2x**. `monetary` leads at **1.359x** adaptation (against the epidemic flagship's 1.199x)
with 26.3% relative adaptation budget vs the epidemic's 15.7% — ~1.7x more resolving power from the
*world* rather than from more seeds. Two declared nulls are reported rather than dropped: `fiscal` is
*arithmetically* inert (optimum is tau*=0 and the shock multiplies the rate), and `supply_chain`
settles `DELAY`, the one family the taxonomy had never measured, in the negative.

**Episodic memory causes policy lock-in — the headline harness result.** On the completed 2^3
factorial it is the only term surviving Holm correction on loss (**+0.630**, p_Holm=0.002), and the
mechanism is not misinformation:

| harness | churn | distinct policies/run |
|---|---|---|
| none | 0.850 | 14.4 |
| trace | 0.855 | 14.5 |
| outcome | 0.774 | 8.2 |
| **memory** | **0.082** | **2.1** |
| outcome+memory | 0.641 | 7.2 |

Factorial on churn, all at p_Holm=0.0007: memory **-0.430**, outcome **+0.190**, outcome x memory
**+0.338**. Precedent tells the authority what it did; monitoring tells it that what it did stopped
working. Precedent alone is a machine for continuity.

*The obvious explanation was tested and refuted*: only **39.6%** of episodes retrieved post-break
predate it, against a chance baseline near **69%**. Memory does not serve stale precedent — it
suppresses revision as such.

*And churn is a diagnostic, not a target*: between arms corr(churn, loss) = **-0.82**, but under
two-way fixed effects it is **-0.11** (n=160), indistinguishable from zero. The arm-demeaned-only
figure is **+0.203**, the opposite sign, and is a seed-difficulty artifact.

**A domain prior can override an explicit mandate.** On `monetary` the regent is worse than every
scripted reference including do-nothing (R ~ 3.0), while attaining the **lowest mandate burden of any
arm** — it stabilizes better than the clairvoyant and pays 3x for it, holding the post-break rate at
9.51 where the clairvoyant sits at 1.53. It writes 123 distinct, competent Taylor rules; the
calibrated optimum sets that feedback gain to **zero**. The mandate warns in every prompt that the
rate is charged "whether or not the rate reaches the economy". Two controls are running to separate
prior from difficulty: `disguised_llm_bare` (identical dynamics, de-economized names) and
`epidemic_llm_contrastive` (identical retrieved episodes, reframed as options rather than
precedent). Both predictions are pre-registered in `preregistration.md`.

## Gate docs

| Doc | Purpose | Status |
|---|---|---|
| [`preregistration.md`](preregistration.md) | H1/H1b/H3/H4, primary metric, correction, falsification, abandon condition | **COMMITTED before the treatment arms ran** |
| [`calibration.json`](calibration.json) | generated frozen/oracle anchors + headroom, with provenance | GENERATED — `scripts/recalibrate.py` |
| [`hypotheses.md`](hypotheses.md) | the falsifiable claims + named baselines (H1–H8) | superseded for H1/H3 by `preregistration.md` |
| [`objectives.md`](objectives.md) | the named objective set | DRAFT |
| [`creativity-metric.md`](creativity-metric.md) | domain-scoped creativity construct (or drop the word) | DRAFT |
| [`stats-protocol.md`](stats-protocol.md) | paired shared-seed design, bootstrap CI, pre-registration | ADOPTED |
| [`decisions.md`](decisions.md) | ADR log (supersession chain + deviations) | LIVING |

## The flagship experiment kit

World: `govsim/domains/scalar/regimes.py::EPIDEMIC_SHOCKED` — endemic SIRS (waning immunity +
imported cases), two costed levers, lockdown efficacy 1.0→0.25 at t=100, transmissibility unchanged,
efficacy never observable. `S+I+R` conserved ⇒ no arm can diverge.

| Arm | Role | Key-free? |
|---|---|---|
| `epidemic_frozen` | calibrated pre-shock optimum, held fixed — the R=1 anchor | ✅ |
| `epidemic_oracle` | calibrated post-shock optimum — the R=0 anchor | ✅ |
| `epidemic_llm_{bare,trace,outcome,memory,…}` | the full 2³ factorial, registered programmatically | needs LLM |
| `epidemic_opro` | budget-matched trace-less OPRO — the named rival | needs LLM |
| `epidemic_llm_critic` | + Critic; **not** budget-matched, reported separately | needs LLM |
| `scalar_{frozen,oracle,llm_full}` | the declared negative control at 1.13× headroom | mixed |

```bash
uv run python scripts/recalibrate.py --seeds 20          # regenerate the anchors
uv run python scripts/headroom_audit.py --seeds 8        # which regimes can host the question
uv run python scripts/run_matrix.py --arms epidemic --seeds 20 --models <model>
uv run python scripts/analyze_matrix.py --store logs/runs_* --cross-model --json logs/analysis.json
uv run python scripts/make_tables.py && uv run python scripts/make_figures.py
```

> **Re-run `recalibrate.py` after ANY change to a regime.** A stale calibration silently re-anchors
> every normalized-regret number in the paper.

## Phase progress (doc-09 §7)

| Phase | What | Status |
|---|---|---|
| 0 | thin core + data spine (6 seams, LLM cache/replay, scalar plugin, Runner, ResultStore, CLI, golden-master, guard tests) | **DONE** |
| 1 | `LLMRegent` + prompt assembler + PID/LQR/OPRO baselines; `StabilizationLoss`; coupled system | **DONE** |
| 2 | rollout-free harness (`TraceFeedback`, `OutcomeFeedback`, `EpisodicMemory`); `RolloutProbe`; `Critic`; paired-bootstrap + factorial analysis | **DONE** |
| 2.5 | **regime calibration + headroom**; the flagship epidemic regime; pre-registration; paper scaffold | **DONE** |
| 3 | rung-1.5 non-economy (SIR, company) on the SAME core | systems + experiments landed; SIR promoted to flagship |
| 4 | rung-2 SFC economy + `EconomyActionInterface` (doc-07 kernel) | not started |
| 5 | multi-regent (N=2, jurisdictions, comms, turn protocol) | hooks threaded; game logic not started |
| 6 | adopt EconAgent / Mandel (gated, optional) | not started |

## LLM access (verified 2026-08-04)

`AIRI` 403 (token rotated); `SambaNova` 402 (zero balance). **Google Gemini free tier works** with
tool-calling. Measured throughput: `gemma-4-31b-it` sustains ~58 calls/min with no errors (the
primary model, and open-weights); `gemini-3.5-flash-lite` / `gemini-3.1-flash-lite` ~47–59/min with
occasional 429s; `gemini-2.5-flash-lite` daily quota exhausted.

```
OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_API_KEY_ENV=GOOGLE_API_KEY OPENAI_MODEL="gemma-4-31b-it"
GOVSIM_LLM_MODE=cache GOVSIM_LLM_MAX_TOKENS=1500
GOVSIM_LLM_DROP_PARAMS=seed GOVSIM_LLM_MIN_INTERVAL=1.0
```

`GOVSIM_LLM_DROP_PARAMS=seed` is required: Gemini's OpenAI-compat layer 400s on `seed`. It is
stripped at the wire only — `seed` still enters the cache key, so recorded tapes replay exactly.

## ☐ AUTHOR — open decisions

- **☐ Rename the project.** `GovSim` is already Piatti et al., NeurIPS 2024 (arXiv:2404.16698) for
  LLM commons governance. A reviewer will assume a relationship that does not exist. The paper
  currently avoids the name entirely; the package rename is the author's call.
- **☐ Venue.** The literature scan ranks AAMAS 2027 main track > NeurIPS 2027 Evaluations & Datasets
  (mandatory code release + Croissant metadata) > JASSS (wants an ODD-style model description and
  will read a scalar plant as a toy).
- **☐ Whether `creativity` stays a construct** or is dropped in favour of the measured
  "policy escaped the reference family" result, which is sharper and already instrumented.

## What "Phase 0 DONE" means here (the doc-09 §7.1 gate)

- ✅ green deterministic CI **without an API key** (`uv run pytest`, 122 tests)
- ✅ `python -m govsim run <experiment>` reproduces a fixed trajectory from `(spec, seed)`
- ✅ adding a world/regent/experiment is one file + one registration
- ✅ per-system `np.random.Generator`; bare-RNG banned by a test; clone/rollout faithful
- ✅ no domain noun in `govsim/core` (leakage test); LLM cache/replay tape; reject-with-feedback
