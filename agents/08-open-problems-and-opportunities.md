# Open Problems & Opportunities — the course-correction review

*Lead-reviewer synthesis of five adversarial critiques (research-scientist, architecture-pragmatist,
economics/ABM, AI-agents/LLM, strategy/vision) of the whole project **including its own plan docs
`01`–`07`**. Claims are grounded in the real files; line/section references are given so each can be
checked. This doc's job is to correct course, not to reassure. Where the critiques disagreed, the
disagreement is resolved explicitly. Read this **before** committing to the `05` roadmap.*

> Reviewed and endorsed by the lead agent. Two findings here **reverse earlier advice in `01`/`02`**, and
> *this* doc is authoritative on them: (1) **keep** the `current_x**3` cubic + obfuscated-prompt experiment
> — it is a coordinated, deliberate partial-information nonlinear-control arm (the repo's most novel), so
> `01` P0.3 / `02` §3.3 ("revert it") are overruled; (2) **`01`'s citations are unverified leads with
> several suspect 2026 arXiv IDs — verify every one before it enters the article** (§3.6).

---

## 1. TL;DR — the 7 things that matter most

1. **There is no falsifiable research question for the next phase — the plans optimize engineering,
   not science.** `05` ships deliverables ("approaches LQR, keeps the ABM out of collapse"), never a
   hypothesis with a named rival. → **Write one falsifiable claim with a baseline before any Phase**
   (recommended spine: *adaptation to an unknown/shifted plant where LQR/PID cannot be precomputed*).
2. **The project's literal selling point — "creative/complex behavior" — is never operationalized.**
   The only concrete prior evidence is the agent rediscovering a PD controller (thesis l.541). →
   **Adopt a measurable construct now** (functional novelty / generalization gap / the author's own
   *Policy Innovation Score*, `Модель креативного правительства.txt:30`), or drop "creativity" as a claim.
3. **"Good governance" has no defined objective.** `03` §3.2 defers it to config; `single_market_model.py:223`
   silently bakes in efficiency-only welfare (CS+PS+gov revenue, no distribution). → **Commit to a small
   set of named objectives and make "which objective" an experimental variable** (it is a result, not a config key).
4. **The two largest design docs (`06`, `07`) specify the least-buildable machinery first.** `07`'s
   Chancery (Effects + Ledger + Mediator + Code-of-Laws + DSL) is, by its own §10.5, "dead weight on the
   toy worlds and premature on an unstable Mandel." → **Freeze `07` as a design spike. Build only the
   conservation assertion (`04` §A.3) + structured reject-with-feedback now.**
5. **Reproducibility is claimed but structurally false: the regent is a live `temp=0.5` Gemini call
   with no logging/replay/cache.** `03` §7 promises "reproducible from (spec, seed)" while the decisive
   actor drifts (thesis used `gemini-2.5-flash-preview-04-17-thinking`; `config.py:87` now uses
   `gemini-2.5-flash`). → **Add an LLM response cache + replay tape (prompt-hash + pinned model) to core
   from day one** — it is the cheapest fix for *both* reproducibility and cost.
6. **The plan architecturally forecloses the actual long-term vision.** The author's own notes
   (`Диплом.txt:64-69, 82-87`) define the program-maximum as an **open multi-government league /
   natural-selection-of-economies** platform; `05` reduces this to one bullet in the last phase and
   `06`/`07` hard-bake a single-regent / single-treasury / single-ledger world. → **Introduce a
   `polity_id` dimension across ledger/observation/objective now, even while running N=1.**
7. **Mandel is the project's biggest risk and is on the critical path.** It "never stabilized"
   (`AGENTS.md:24`, thesis l.546); `04` is an 8-stage from-scratch AB-SFC rebuild of a 1411-line monolith.
   → **Decouple the paper from Mandel. Build a small SFC economy conserved-by-construction; make full
   Mandel an optional track with a kill-criterion.**

---

## 2. The meta-problem (all five critiques converge here)

**The project is over-designed and under-asked: it is building HOW before deciding WHAT.**

Four of the five critiques independently land on the same shape. The planning corpus (`01`–`07`, ~2,300
lines) is overwhelmingly about *machinery* — package layout, a declarative WorldSpec, a conserved-Effect
governance kernel, an 8-stage ABM rebuild — while the three load-bearing **scientific** decisions are
unmade:

- **No falsifiable hypothesis** for the next phase (research-scientist, AI-agents).
- **No chosen objective** for "good governance" (research-scientist, economics).
- **No operationalization of "creativity"** — the thesis title and the entire pitch (research-scientist,
  strategy, AI-agents).

The consequence is concrete and damaging. As written, Phases 1–3 (the stated publishable result, `05:127`)
deliver *"an LLM approaches the LQR optimum on a linear toy and stops an ABM from collapsing"* — and **both
are problems classical control + a bounded automatic stabilizer already solve** (`04` §E, `01` Cluster D).
A fair reviewer's verdict is the one `01` itself raises and never answers: *"expensive LLM-generated PID
controllers."* Meanwhile the genuinely novel experiment — controlling an **unknown nonlinear plant from
partial information** — is sitting uncommitted in the working tree (`linear_stochastic_system.py:242` +
`linear_system_prompt_obfuscated.md:1`), and the plans (`01` P0.3, `02` §3.3) tell you to **delete it**.

The second face of the meta-problem is **build-vs-research risk for a solo author**: summed, `03`+`04`+`06`+`07`
propose a package rename, registry, typed config, RNG, results schema, CLI, objective abstraction,
clone/rollout, provider abstraction, an 8-stage Mandel SFC rewrite, an evolutionary engine, an RL arm,
and a full governance kernel — a multi-person, multi-year program with **no honest "if you only do N
things" triage**. The most likely outcome of executing the plan as written is a half-finished refactor on
top of a still-unstable Mandel: the 2025 state with more scaffolding.

**The corrective stance of this document:** decide the WHAT first (§3 research-design, §4 decide-from-start,
§8 open questions), cut the premature HOW (§5), and follow a brutally minimal critical path to one
defensible result (§7).

---

## 3. Critical weaknesses (grouped by theme)

### 3.1 Research design

**[CRITICAL] No falsifiable research question or hypothesis for the next phase.**
*What's wrong:* every doc states deliverables; none states a hypothesis that could be falsified or the
rival it must beat. `01` §5's "can an evolved regent keep an ABM out of collapse?" is a capability demo;
the honest null ("a 2-line PID or a bounded automatic stabilizer does it just as well") is exactly what
`04` §E and `01`'s "Simple Baselines" citation predict will win.
*Evidence:* `05:120-131` (critical path = deliverables); `01` §5 ll.122-136; `AGENTS.md:20`.
*Why it matters:* it determines which of Phases 1–5 are even worth building; without it the article has
no measurable dependent variable.
*Fix:* before any Phase, write one falsifiable claim with a named baseline, e.g. *"H1: a code-as-policy
regent with the trace side-channel recovers from an UNSEEN structural shock (regime change in A/B, or a
new nonlinearity) in fewer interventions / lower post-shock regret than (a) the frozen LQR for the
pre-shock plant and (b) OPRO without the trace channel."* The science is **adaptation to the unknown**,
where LQR/PID cannot be precomputed.

**[CRITICAL] "Creativity / complex behavior" is never operationalized.**
*What's wrong:* the project's identity word has no metric anywhere in `01`–`07`; the one empirical
"creative" result is P→PD (thesis l.541), i.e. rediscovering textbook control.
*Evidence:* `AGENTS.md:20`; thesis ll.328-335 (creativity = "theoretical discussion" only), l.541; the
author's own unused metrics in `Модель креативного правительства.txt:29-30` (*Institutional Efficiency
Index*, *Policy Innovation Score*).
*Why it matters:* if "creativity" stays a vibe, the paper is a systems/demo paper at best; with
LLM-mutation evolution the emitted laws will become `np.clip`/`where`/`tanh` spaghetti, making the claim
*worse*, not better.
*Fix:* commit to one defensible construct and measure it pre-Phase-3: (a) functional novelty — distance
of the evolved law from the best-fit PID (does it use conditionals / state-history / regime detection a
PID structurally cannot?); (b) generalization gap — does it beat a tuned PID *specifically* on
shifted/nonlinear regimes while tying on the linear one?; (c) MAP-Elites behavioral diversity. If none
separate the LLM from a tuned controller, **reframe as adaptive in-context control and drop "creativity."**
Consider a parsimony term / restricted grammar to keep the laws human-readable (the interpretability
pitch fails the moment evolution produces spaghetti).

**[HIGH] "Good governance" has no defined objective; the chosen scalar becomes the fitness, so it is not
cosmetic.**
*What's wrong:* on the linear toy MSE+MSU = an LQR cost (fine). But the pitch is governing an *economy*,
where "good" is contested (welfare? growth? Gini? employment? stability?), and `03` §3.2 defers it to a
pluggable Objective that is never chosen. `single_market_model.py:223` already hard-codes
`social_welfare = CS + PS + gov_net_revenue` — efficiency-only, distribution-blind — which an economics
reviewer rejects on sight. MSE/MSU is a *control loss*, not welfare.
*Evidence:* `03` §3.2 ("the experiment spec picks one"); `07` (conservation enforced, objective not);
`single_market_model.py:223`; thesis ll.290-291.
*Why it matters:* whatever scalar is chosen *is* what the regent evolves toward; choosing it badly
Goodharts the whole program.
*Fix:* commit in the article to a small set of named objectives (utilitarian SWF / inequality-weighted /
output-gap+inflation loss / growth) **with distributional metrics (Gini, deciles) as first-class
observables**, and study the regent's behavior *as the objective changes* — the comparison is the result.
Run the LQR/analytic check against the economically-correct objective, not against MSE.

**[HIGH] Goodhart / specification-gaming is acknowledged for other systems but not instrumented in this
design's own fitness loop.**
*What's wrong:* Phase 3 makes a scalar economic score the evolutionary fitness, and `07` §3 itself
enumerates the leaks a score-maximizer will exploit (SetRate off-ledger value, valuation arbitrage,
negative-balance laundering via the world's `max(0,…)` clamp). These are treated as "conservation leaks
to patch," but scientifically they are Goodhart attack surfaces *the evolved regent will find because they
raise the score*.
*Evidence:* `07` §3 ll.237-251; `01` P2.12 (collusion deferred to Phase 5); `05` Phase 3.
*Fix:* make gaming a *measured outcome of Phase 3*, not a Phase-5 afterthought: (a) hold out a "true"
welfare measure distinct from the fitness proxy and report proxy-vs-true divergence; (b) red-team the
evolved policies against the three `07` §3 leaks; (c) report worst-case across seeds. **A characterized
gaming episode (proxy up, true welfare down) is a stronger, more honest paper than a clean stabilization
— and it needs no multi-agent machinery.**

**[HIGH] The Lucas critique is the load-bearing economic question and no doc treats it as first-class.**
*What's wrong:* across every world the regent can govern, the micro-agents do **not** adapt to the policy
*regime*. The linear world has no agents at all; `SingleMarketModel`'s agents only smooth on *realized*
outcomes (`single_market_model.py:33-44, 87-108`) — they never anticipate the announced rate; Mandel's
adaptive agents exist but the model is unstable. The original proposal explicitly demanded regime-feedback
("ensuring agents' adaptation to new conditions").
*Evidence:* `single_market_model.py:33-44, 87-108`; `01` Cluster B; `05` Phase 5 (multi-agent deferred,
agent-adaptivity never scheduled); `Project Proposal.txt`.
*Why it matters:* if agents are static w.r.t. the rule, the control problem collapses to system-ID of a
fixed plant (LQR territory, LLM decorative) and any "the regent learned to govern" result is
non-generalizable — agents never game the policy, so nothing transfers.
*Fix:* make **policy-aware adaptive expectations a testable requirement now** — at minimum agents that
condition on the announced lever, plus a falsification test that governance results *change* under
naive→adaptive agents. Adopt AI-Economist's Stackelberg leader-follower as the reference. If unaffordable,
scope the claim down to "optimal control of a partially-observed plant" and stop calling it governance.

**[HIGH] External validity / calibration is the thesis's already-flagged weakness and the plan defers it
behind an uncalibrated synthetic ABM.**
*What's wrong:* the thesis was dinged for the toy-model limitation (l.551); the roadmap's answer is a
*stabilized-but-still-synthetic* Mandel whose only validity target is qualitative stylized facts
(`04` §3.5). That upgrades the toy to a bigger toy. "Stylized facts reproduced" is internal plausibility,
not calibration to data — and `04` lets the headline learning work (Phase 3) start *before* the economy is
validated.
*Evidence:* thesis ll.519, 551; `04` §3.5; `05` Phase 2/3.
*Fix:* either (a) scope the article honestly as a controlled-environment methods contribution and stop
implying real-economy relevance, or (b) add one concrete calibration touchpoint (a handful of real moments
— inflation/unemployment volatility ranges — via the surrogate/SMC tooling `04` §3 already cites). And
**promote stylized-fact reproduction to a hard GO/NO-GO gate before any governance claim on the ABM.**

**[MEDIUM] Multi-seed discipline is named but not wired into selection; the loop will still select under
variance.** All docs cite the high-variance-evaluator danger, but only prescribe *reporting* mean/variance,
not a variance-aware *selection* rule. `05` Phase 3's exit is "beats OPRO with CIs" (CIs on the mean), not a
worst-case guarantee. → Specify fitness = `mean − λ·std` or accept-only-if-worst-of-N-bounded; land seeding
in Phase 0 (it is half a day, `05:135`). Move the **collapse detector + variance into the Phase-1
evaluator** (not Phase 4) so it protects every downstream claim.

**[MEDIUM] No statistical methodology is actually specified.** "Multi-seed CIs" is repeated as a slogan with
no comparison procedure: number of seeds, paired-vs-unpaired, Wilcoxon/bootstrap, how LLM nondeterminism is
separated from world-seed nondeterminism, multiple-comparison control. → Write a half-page stats protocol
(shared seeds across regents = paired design; bootstrap CI on the per-seed difference; cache LLM responses
so the only varying source per comparison is the world seed; pre-register the primary metric). This is a
doc, not code, and it protects every result.

### 3.2 Engineering / over-build

**[CRITICAL — infrastructure] No LLM reproducibility/replay layer or response cache anywhere in ~2,300
lines of planning.** Seeding numpy/random while the regent is a live `temp=0.5` Gemini call
(`config.py:87-90`) is reproducibility theatre: `03` §7's "reproducible from (spec, seed)" is false for the
component under study, whose snapshot already drifted between the thesis
(`gemini-2.5-flash-preview-04-17-thinking`) and current config. At ~6.1s/call (`config.py:88`) every re-run
costs money and time. → **Add to core *now*: an on-disk LLM cache keyed by `sha256(prompt)+model+temp+seed`
with a `--no-cache` override; persist every raw prompt+response+model-snapshot+token-usage in the RunRecord;
a `replay` mode that re-runs from logged outputs without calling the API.** This is the single cheapest
load-bearing fix and is expensive to retrofit because it shapes the data model.

**[CRITICAL — infrastructure] The research harness that gates publishability is largely absent:** no
queryable result store (the `RunRecord` is "JSON by default", `03` §4 — not a runs table for cross-run
aggregation), no versioning schema tying results to `(world, regent, prompt-file, model-id, seed, git-commit,
objective)`, no compute/cost budget model. `01` cites ~150 evals (ShinkaEvolve) but no `$/eval` budget;
Phase-3 evolution on Mandel (deepcopy of hundreds of agents × pop × 20–50 seeds × horizon) may be
infeasible. → Add a Phase-0.5 harness: a `runs` table (sqlite/parquet) with those columns + per-step series
as artifacts + one generic plotter keyed off the schema; a budget ledger + evaluation cascade (cheap worlds
first, short rollout pre-screen).

**[HIGH] The two heaviest docs specify the furthest-out, least-validated machinery — inverted effort
allocation.** `07` (750 lines) + `06` (595) vastly outweigh `02` (203) + `03` (224) + `05` (143). None of
`07`'s kernel can run on any world that exists today (toy worlds have no balances; Mandel is unstable and
unmigrated), and `07` §10.5 says so. Meanwhile the actual critical-path items (seeding, decoupling the
regent, LLM caching, an OPRO baseline) get a few lines each. → Demote `06`/`07` to "future design" status
with a banner; the near-term keystone is a 1-page critical-path doc (§7 here).

**[MEDIUM] The "golden-master test of a thesis experiment" (`05` Phase 0 / `03` §5 step 1) cannot reproduce
the thesis, and no doc flags it.** The thesis used `total_steps=1000`, `performance_window=20`, the
`-preview-04-17-thinking` model; `config.py` now uses `total_steps=200`, `performance_window=30`,
`gemini-2.5-flash`, plus the uncommitted cubic term — a golden-master taken now freezes the *wrong*
behavior, and the preview model is likely deprecated. → Make the golden-master a **deterministic regression
baseline** (`TestPoliciesAgent` with a fixed expression + fixed seed, NO live LLM); treat thesis-number
reproduction as best-effort and pin thesis values in a dedicated spec, not live `config.py`.

**[MEDIUM] `clone()`/`rollout()` via `copy.deepcopy` is the proposed keystone but is unsound for the only
worlds that matter.** `deepcopy` cannot capture module-global `random`/`np.random` state, which is how both
Mandel and the *current linear world* (`linear_stochastic_system.py:239`) draw randomness; `06` §4 caveat 3
"fixes" Mandel by re-seeding inside `__init__`, which restarts the stream and makes a rollout *not* a faithful
continuation — silently biasing fitness comparisons. → Make **each world own a `np.random.Generator`** a HARD
prerequisite of `clone()`/`rollout()`; forbid bare `random.*`/`np.random.*` with a test. Until then, do not
claim rollout is a sound fitness oracle.

**[MEDIUM] The planning corpus itself is becoming a tangle — and is untracked in git.** There is a
supersession chain `03 → 06 → 07` resolved only in buried prose, no decision log, and (verified)
**`AGENTS.md` and the entire `agents/` directory are `??` untracked** (`git status`), so "living documents,
keep updated" has no version history and the chain cannot be diffed. → `git add` the corpus now; replace
scattered supersession prose with one ADR-style `decisions.md` + a STATUS table at the top of the README
(doc | status | last-touched | superseded-by); collapse `03`'s dead registry/context sections to a one-line
pointer. **Stop expanding the corpus** — the next artifact should be a 1-page critical path + `decisions.md`,
not a `09-*.md`.

**[MEDIUM] Smaller real bugs the audit under-weights or missed:**
- The LLM regent is the *lone* interface non-conformer: base + Random/Static/Test all declare
  `llm_extra_context`; only `IntelligentLLMAgent` omits it (`gov_agent_linear.py:221-225`), and the loop
  passes positionally (`simulation.py:108`) — so the "tools/archive" extra-context channel the proposal wants
  is **dead on arrival for the main regent**. Re-rank Medium; fix when generalizing the regent.
- `simulation.py:94` hardcodes a `% 50` progress print unrelated to `agent_decision_frequency`, and `:101`'s
  `< total_steps` clause silently suppresses the decision on the final interval boundary — an off-by-one in
  *when* the regent acts, which matters for reproducing the thesis's timed interventions. Move scheduling to
  `core/schedule.py` and add a test asserting the exact decision-step set.
- Policy-execution semantics already disagree: `SingleMarketModel.apply_policy_change` evaluates the
  expression **once at apply time** (`:261`) while every other world re-evals **per step**. Fix the contract
  (re-eval everywhere) *before* stacking `06`/`07` on the abstraction.

### 3.3 Economics

**[HIGH] "Resurrect Mandel" is likely the wrong economic goal.** `04` itself concedes the fixes amount to
rebuilding Mandel as a stock-flow-consistent economy from scratch (SFC ledger + Caiani steady-state init +
atomic settlement + 8 feature-flag stages + SMC validation) — i.e. paying the full cost of a benchmark model
while inheriting a non-validated, non-citable bespoke variant whose only virtue is sunk cost. Governing a
*recognized* benchmark (Caiani 2016 AB-SFC, Dosi K+S, JAMEL, or **EconAgent**, which has public code and
already reproduces Phillips+Okun and whose agents are LLM-adaptive) is far more defensible. → Add an explicit
**build-vs-adopt decision** to `05` before Phase 2; strong default = adopt EconAgent or a Caiani/K+S
benchmark (this also closes the Lucas-critique gap for free), demote Mandel to an optional "native model"
showcase with a kill-criterion.

**[MEDIUM] Money/credit/GE feedback are economically central but the stable worlds have none.** Fiscal
governance bites *through* balance sheets and credit; `04` frames the credit loop only as a divergence bug to
clamp, and `economic_models.py:151` uses an un-microfounded `growth_modifier = 1 − 2·tax`. → Ensure the chosen
arena has balance sheets + an automatic-stabilizer channel the regent can move; retire the ad-hoc tax→growth
link or microfound it.

**[MEDIUM] The LQR "ground-truth ceiling" (`01` P0.3) is both undermined by the live cubic term and a weak
bar even when restored.** LQR is optimal only for a linear-quadratic problem; with `current_x**3` and
`param_A=0.95` the system is nonlinear and explosive for |x|≳1.03, so LQR is neither optimum nor globally
stabilizing. Even restored, "LLM ≈ LQR on a scalar linear plant" reads as trivial to an economist. → Keep LQR
as an **internal sanity check**, not a headline; for the nonlinear arm use a numerically-computed
optimal-control baseline (DP on the discretized state).

### 3.4 LLM / cost

**[HIGH] No compute/cost budget model** (covered in §3.2 infra) — population × seeds × horizon × ~6.1s/paid
call makes Phase-3-on-Mandel a budget question the plan never asks.

**[MEDIUM] Coherence-collapse and token-cost metrics are imported wholesale from long-horizon autonomy
benchmarks without checking they fit.** The regent decides every 50 steps from a *fresh* prompt
(`config.py:9`), so it is not a long-horizon coherent agent in the Vending-Bench sense — "steps-to-coherence
failure" may be the wrong failure mode, and folding *LLM API token cost* into an *economic welfare* objective
(`01` P2.11 / `05` Phase 4) is a category error: `07`'s enact/upkeep already models the in-model cost of
governance correctly. → Keep token cost as an engineering/sample-efficiency metric only; use `07` upkeep as the
economic cost; only adopt coherence-collapse if a persistent long-horizon regent is actually built.

**[MEDIUM] LLM micro-agent validity is flagged but unused (scope inconsistency).** `01` Cluster B calls
LLM-population variance-collapse the central danger, but the build uses heuristic agents so it never bites.
→ Either commit to LLM micro-agents *with* a distributional-fidelity check, or label Cluster B as future-risk
only.

**[LOW] Security framing is contradictory and slightly overstated.** `02`/`06` call the sandbox the best
safety property; `07` §8 calls host-security obsolete; "airtight" rests partly on a regex placeholder
extractor (`06` §2.3) and RestrictedPython, not a proven OS boundary. → One framing: *economic-write denial +
basic hygiene*, not a hard host boundary; OS isolation only if untrusted code is ever run at scale.

### 3.5 Strategy / positioning

**[CRITICAL] The plan silently amputates the project's actual vision and rebrands a single-regent control
benchmark as the whole project.** The author's notes are explicit: the program-maximum is an **open
international research project** where outsiders submit competing policy-agents *or* political systems that
survive best in given worlds — a public **league for institutional-design theory** — plus elections, forms of
government, federations, knowledge diffusion, and "natural selection of economies"
(`Диплom.txt:64-69, 82-87`; `Модель креативного правительства.txt:20-24`). `05` reduces all of this to one
bullet in the last phase (`05:109`) and never carries it into the load-bearing interfaces; `06`/`07` hard-bake
single world / single regent / single treasury / single Ledger / single Code-of-Laws. The plan thus
over-invests in the *least* novel half (single-agent control, already occupied by TaxAgent/SimCity/AI-Economist
and at risk of losing to a trivial baseline per `01:16`) while architecturally foreclosing the **white-space,
fundable half** (`01:69`: "LLM-as-mechanism-designer for an economy remains white space"). → Re-anchor the
article on the **benchmark/league**, with single-regent-on-linear as the calibration baseline *inside* it;
introduce a `polity_id`/`regent_id` dimension now (§4).

**[HIGH] The defensible novelty is narrow and partly perishable, and the plan leads with the contested half.**
`01` honestly concedes the loop is commoditized (AlphaEvolve + ≥4 reimpls) and "LLM governs an economy" is
prior art. What remains is "code-as-policy (open functional form) + evolution against an economic evaluator"
— and `01:16`'s "Simple Baselines are Competitive" warns the fancy loop may not beat OPRO/random. Staking the
headline on "evolved regent **beats** OPRO on the linear worlds" (`05` Phase 1/3 exit) is likely a **null
result**: on a scalar linear plant with a fixed-gain optimum there is no headroom, and `01` says today's
regent *is* ~OPRO. → Scope the linear arm to "**matches** OPRO+LQR" (sanity check) and move the
evolutionary-advantage claim to non-stationary / nonlinear / partial-info regimes; aim the article at the
unoccupied claim (institutions + cross-world comparison).

**[MEDIUM] `07`'s conserved vocabulary forbids the most creative, most vision-aligned moves.** The vision
wants the regent to change "the rules of the game for agents" (`Интерфейс экономических политик.txt`);
`07` §10.3/§10 explicitly excludes rule-rewriting institutions as "semantic corruption," and `06` §4 declines
to govern Mandel's `tax_rate` for the same reason. On the linear world LQR ties the regent, so creativity
*must* come from richer/rule-level interventions — yet the design rules them out to preserve a conservation
property `07` itself admits is only locally guaranteed. → Promote **world-declared rule levers** (bounded,
`SetRate`-able coefficients of agents' own reaction functions) to a first-class capability rather than a
forbidden case.

**[MEDIUM] Cross-government knowledge diffusion and "natural selection of economies" — two of the most novel,
testable hypotheses in the vision — are entirely absent from `01`–`07`.** Including the crisp, citable claim
"longer planning-horizon → richer economies" (`Диплom.txt:101`) and the diffusion update rule
(`Модель креативного правительства.txt:20-24`). → List these as named research questions the architecture must
not preclude, and keep the `polity_id` hooks alive so a second paper is buildable cheaply.

### 3.6 Science-integrity hazard

**[MEDIUM] `01` is presented as a literature foundation but disclaims its own citability and contains
likely-fabricated arXiv IDs.** Its masthead says "not a citable academic literature review — verify before
quoting," marks items `[uncertain]`, and several load-bearing IDs carry **2026 year-month prefixes**
(`26xx`): "Simple Baselines" `2602.16805` (the whole P0 argument leans on it), `2603.17694`, `2605.10447`,
`2604.04543`, `2603.08956`. → Treat `01` as **leads to verify, never a citation list**; independently confirm
every ID and venue (especially Simple Baselines, ShinkaEvolve, TaxAgent, SimCity, and all `26xx` IDs) before
anything enters the article. Citing a non-existent paper is review-ending.

---

## 4. Decide-from-the-start list (expensive to retrofit)

| Decision | Recommended default | Why now |
|---|---|---|
| **The falsifiable hypothesis + baseline** | "Adaptation to an unknown/shifted plant, fewer interventions / lower post-shock regret than frozen-LQR and trace-less OPRO." | Determines which phases are worth building; everything downstream serves it. |
| **The objective(s)** | A *small named set* (utilitarian SWF / inequality-weighted / output-gap+inflation), with Gini/deciles first-class; objective is an **experimental variable**. | The scalar *is* the fitness; changing it later re-runs everything (§3.1). |
| **Creativity metric** | Functional novelty (distance from best-fit PID) + the author's *Policy Innovation Score*. | No dependent variable for the headline claim without it (§3.1). |
| **LLM caching + replay tape** | On-disk cache `sha256(prompt)+model+temp+seed`; raw I/O + model snapshot in RunRecord; replay mode. | Shapes the data model; only path to reproducibility with a drifting remote model (§3.2). |
| **Result/versioning schema** | A `runs` table keyed by `(world, regent, prompt-file, model-id, seed, git-commit, objective, KPIs, cost)`; series as artifacts. | Cross-run aggregation and stats depend on it; painful to backfill (§3.2). |
| **Local per-world RNG (Generator)** | Each world owns a `np.random.Generator`; ban bare `random.*`/`np.random.*` via test. | Hard precondition for a *sound* `clone()`/`rollout()` fitness oracle (§3.2). |
| **`polity_id` / `regent_id` dimension** | Ledger accounts = `"treasury@polity:0"`; `EconomyView`/context take a viewer id; Objective per-regent. Run N=1 for the first paper. | Nearly free now; a rewrite after `06`/`07` ship as singletons. Keeps the entire multi-government vision buildable (§3.5). |
| **Policy-aware adaptive agents** | A row in the eval harness; falsification test naive→adaptive. | The Lucas critique decides whether results transfer at all (§3.1). |
| **Stats protocol** | Paired shared-seed design; bootstrap CI on per-seed difference; pre-registered primary metric. | Cheap (a doc); protects every claim (§3.1). |
| **Lazy LLM key + thin `LLMClient` Protocol seam** | Don't `raise` at import; fail on first call. | Unblocks CI/tests without a key; needed to swap a non-deprecated model. Pull forward from `03` step 9. |

---

## 5. Dead-ends / over-engineering to cut or defer

- **DROP/DEFER the full `07` Chancery stack** (7-primitive Effect algebra + private double-entry Ledger +
  Mediator with cost/rate-limit/noise/lag/leakage + versioned git-like Code-of-Laws + declarative DSL +
  tool-calling transport, and the proposed 7th `Credit` primitive + `ExternalSink` enum + valuation-arbitrage
  mitigations). `07` §10.5 admits it is "dead weight on the toy worlds and premature on an unstable Mandel,"
  and §10.1 concedes the guarantee is "LOCAL until doc-04 lands." **Keep only:** the cheap per-step
  money-conservation assertion (`04` §A.3) and structured **reject-with-feedback** (today a bad expression
  silently returns `None`: `single_market_model.py:262-264`, `linear_stochastic_system.py:227-229`). The
  conserved-Effects idea is good; the timing is wrong.
- **Do NOT "revert the cubic to linear" globally** (as `01` P0.3 / `02` §3.3 imply). The cubic
  (`linear_stochastic_system.py:242`) + obfuscated prompt (`linear_system_prompt_obfuscated.md:1`, "главным
  членом третьей степени") + config flip are a **coordinated working-tree experiment** (all three files `M` in
  `git status`) — a partial-information nonlinear-control task, the most genuinely novel arm in the repo. Keep
  **both** a linear arm (for the LQR sanity check) and a nonlinear/partial-info arm; parameterize
  `state_exponent` and the prompt's information level.
- **Do NOT stake the publication on "evolutionary regent beats OPRO on the linear worlds."** Likely null
  (§3.5). Scope to "matches OPRO+LQR"; move the advantage claim to non-stationary/nonlinear/partial-info.
- **Do NOT make full Mandel stabilization a hard prerequisite** (`05` Phase 2). Build a small purpose-made SFC
  economy (1 good; hh+firm+gov+bank; conserved by construction from line 1); keep Mandel as an optional track
  with a kill-criterion (e.g. "if Stage 0–2 gates not met in N weeks, fall back").
- **Do NOT pursue the full `01` P1 stack** (MAP-Elites + islands + migration + artifact side-channel + bandit
  routing + novelty rejection) before an OPRO baseline demonstrably loses. Try **OPRO + a cheap episodic-RAG
  regent** first; add machinery only where it beats them.
- **Drop the RL-regent arm** (`03` step 8 / `05` Phase 3 optional) for the first article — `01` Cluster D's own
  cited survey calls RL-in-economics brittle; it doubles the surface for a comparison the paper can make
  qualitatively.
- **Drop Mesa from `pyproject.toml` now** (declared, zero imports — `06` §7); **do not add Gymnasium** (`06` §7
  — hold the line; no RL-adapter before an RL baseline exists).
- **Don't do the big package rename/reshuffle now** (`03` §2's `core/`/`worlds/`/`regents/`/`llm/` tree with
  shims). Churn with no research output and risk to the only working entry point. Do only the minimal seams
  (decouple regent, seed RNGs, one Objective).
- **Remove vestigial tooling:** `combined_code.txt` + `scripts/combine_scripts.py` (obsolete given `AGENTS.md`),
  experiment-specific viz (`special_visualize_exp4.py`, `visualize_universal_legacy.py`). Low value, goes stale.
- **Stop calling RNG seeding "reproducibility"** while the regent is a live `temp>0` model — it is reproducible
  *world*, irreproducible *experiment*, until the replay tape lands.
- **Treat `01` as unverified leads, not a citation list** (§3.6).

**The brutally-minimal critical path to the next result needs NONE of:** Effect algebra, Ledger, Mediator,
Code-of-Laws, DSL, Institutions, tool-loop, provider package, RL regent, Mesa, Gymnasium, package rename, or a
stable Mandel.

---

## 6. Promising directions to consider from the start

- **Partial-information nonlinear control as the scientific spine** (cubic dynamics + obfuscated prompt
  revealing only "third-degree main term"). *Why:* this is where code-as-policy has a real edge LQR/PID cannot
  match — the regent must *infer* structure from partial hints + history and synthesize a nonlinear law. It
  answers the falsifiable-question and the creativity gaps at once, and is **already half-built in the working
  tree**. *From-start hook:* parameterize `state_exponent` and the prompt info-level as experimental variables;
  do not delete the cubic.
- **Same world, different social-welfare objectives → different discovered policies.** *Why:* turns the
  hand-waved objective choice into the result; "code-as-policy makes the value→institution mapping legible and
  comparable" is a genuine economics contribution AI-Economist/TaxAgent (single fixed welfare) did not deliver.
  *Hook:* make Objective a first-class experiment axis with distributional metrics from day one.
- **Instrument the Phase-3 loop for specification-gaming; report a characterized episode.** *Why:* `07` §3
  already enumerates the leaks; a diagnosed Goodhart episode (proxy up, true welfare down) is a stronger, more
  honest paper than a clean stabilization and needs no multi-agent machinery. *Hook:* hold out a true-welfare
  measure distinct from fitness now.
- **The public benchmark / league as the project's identity** (the author's own program-maximum). *Why:* a
  benchmark outlives a one-off result, is the standard way an AI subfield gains traction, occupies the
  white-space `01:69` names, and reuses the Phase-0/1 eval harness you must build anyway. *Hook:* a fixed
  battery of worlds + fixed objective/eval harness + a submission interface; `polity_id` ready.
- **A tiny SFC economy (hh+firm+gov+bank, conserved by construction) as the "real economy worth governing."**
  *Why:* delivers a non-trivial governable arena in weeks with conservation guaranteed from line 1, sidesteps
  the Mandel tar pit, and is a far better fit for `07`'s tax/cap-and-trade examples than the unedited Mandel
  `06` says can't be taxed.
- **Adaptive micro-agents (policy-aware) as a falsifiable axis.** *Why:* turns the biggest economic
  vulnerability (Lucas) into a headline result — "does the regent's policy survive when agents anticipate it?"
  is open and unanswered by any LLM-governance paper. *Hook:* naive vs adaptive as a harness toggle now.
- **The artifact/trace side-channel as the core mechanism whose value is the ablation** (`01` P1.6). *Why:* the
  cheapest upgrade; gives a clean contrast (trace vs no-trace vs OPRO) on recovery from shocks — a result that
  can exist even when "beat OPRO on the linear toy" cannot.
- **A cheap episodic-RAG regent as the learning baseline** (in the proposal's Info-Archive and `TODO`). *Why:*
  answers the advisor's "no learning" flag in days, is what the fancy evolution must *beat*, and feeds a corpus
  for the novelty metric.
- **Full LLM I/O persistence + replay** (§4). *Why:* makes the irreproducible component reproducible-by-replay,
  supports the platform pitch, and yields a reasoning corpus for the creativity analysis.

---

## 7. Revised "if you only do N things" prioritization (supersedes `05`)

For a **solo researcher**, an ordered path that sharpens `05`. Each step is shippable; the first article needs
only steps 1–6.

1. **Foundations + replay (Phase 0, sharpened).** Local per-world RNG (Generator); seed everything; **LLM
   response cache + replay tape + raw-I/O RunRecord**; lazy LLM key + thin `LLMClient` seam; deterministic
   golden-master via `TestPoliciesAgent` (no live LLM); `git add` the corpus + `decisions.md` + STATUS table;
   drop Mesa. *(Replaces `05` Phase 0; adds the missing infra meta-problem.)*
2. **Decide the WHAT (a half-page each, before more code).** The falsifiable hypothesis + baseline; the
   objective set (with distributional metrics); the creativity metric; the stats protocol. *(This is the gate
   `05` is missing.)*
3. **Generalize the regent + one Objective abstraction.** Drop the `LinearSystemAgentContext` coupling
   (`gov_agent_linear.py:232-238`); move KPIs to an Objective; unify policy execution (re-eval per step
   everywhere). Acceptance: the LLM regent governs SingleMarket + SimpleGrowth.
4. **Sound evaluator + baselines (the scientific spine).** `clone()`/`rollout()` on the local-RNG worlds;
   multi-seed eval with a **variance-aware selection rule** and a collapse detector; **OPRO baseline**; LQR
   ceiling on the *linear* arm. Result target: "matches OPRO + LQR with paired CIs."
5. **The novelty arm.** Keep the cubic + obfuscated prompt as a first-class **partial-information nonlinear
   control** experiment (parameterized); add the **trace side-channel** and a **cheap episodic-RAG regent** as
   the learning baselines to beat. Result target: "code-as-policy with trace recovers from unseen shocks better
   than frozen-LQR and trace-less OPRO" (the falsifiable H1) — *this is the publishable contribution.*
6. **Richer policy on one purpose-built SFC toy (conserved by construction)** + policy-aware adaptive agents +
   objective-as-variable + a gaming probe. Demonstrates "institutions beyond a scalar" and turns Lucas + Goodhart
   into results — without Mandel, without the Chancery.
7. **(Optional, parallel, kill-criteria) Mandel or an adopted benchmark.** Pursue full Mandel stabilization
   (`04`) *only* as a stretch track; prefer adopting EconAgent/Caiani/K+S. Reintroduce the regent on whichever
   stabilizes first.
8. **(v2) The league + multi-polity.** With `polity_id` already threaded, build the submission interface and the
   first cross-world leaderboard — the strongest, most durable, most fundable framing.

What this changes vs `05`: **Mandel and the Chancery move off the critical path; the WHAT-decision and the LLM
replay/cache move on; the headline contrast moves from "beat OPRO on a linear toy" (likely null) to "adapt to
an unknown plant" (where the LLM can actually win).**

---

## 8. Open questions the author must answer before serious building

1. **What is the one falsifiable claim of the next paper, and what baseline must it beat?** (Until answered,
   building is premature.)
2. **Which objective(s)?** Is "good governance" efficiency, welfare, equality-weighted, growth, or stability —
   and is the *choice itself* an experiment or a fixed config value?
3. **Is "creativity" a measured construct or a marketing word?** If measured, which: functional novelty,
   generalization gap, behavioral diversity, or the *Policy Innovation Score*? If not, is the claim dropped?
4. **Is the project single-regent control or a multi-polity institutional-design platform?** This decides
   whether `polity_id` is threaded now (cheap) or never (foreclosed). The vision docs say the latter; the plan
   builds the former.
5. **Build Mandel or adopt a benchmark?** Given Mandel "never stabilized" once already, what is the
   kill-criterion, and is EconAgent/Caiani/K+S the lower-risk arena?
6. **Do the micro-agents adapt to the regent's rule?** If not, is the contribution honestly "control of a
   partially-observed plant" rather than "governance"?
7. **What is the per-experiment compute/$ budget**, and does Phase-3-on-Mandel fit inside it?
8. **Is the article a controlled-environment methods contribution or a claim about real economies?** (Pick one;
   the second needs at least one calibration touchpoint.)
9. **Where is the rule-override line** — which agent behavioral coefficients become world-declared
   `SetRate`-able levers vs forbidden? (`07` open question (a); decides whether the vision-aligned "creative"
   moves are even expressible.)
10. **Have `01`'s load-bearing citations been independently verified?** (No claim from `01` enters the article
    until yes.)

---

*Bottom line:* the engineering instinct in `02`–`07` is mostly sound *as engineering*, but the project is
about to spend its scarcest resource — a solo researcher's months — on plumbing (`07`'s Chancery, `04`'s
8-stage Mandel rebuild) while the three decisions that determine whether any of it is *science* (the
hypothesis, the objective, the creativity metric) remain unmade, and while the genuinely novel experiment is
sitting uncommitted in the working tree marked for deletion. Decide the WHAT, build the minimal evaluator,
keep the cubic, thread `polity_id`, and defer the cathedral.*
