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
