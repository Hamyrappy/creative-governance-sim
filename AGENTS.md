# AGENTS.md — working guide for developer agents on `creative-governance-sim`

This file orients human and AI contributors: **what the project is, where each piece of context
lives, how to run it, and how to work on it.** Read this first. Deeper analysis and plans live in
[`agents/`](agents/) (see the index in [`agents/README.md`](agents/README.md)).

> Vocabulary: the author calls the LLM-based governing agent a **"regent"** (a "regent system" =
> an intelligent control system that governs a complex economy). Use this term consistently in
> new code and docs. In the current code it is still named `government agent` / `IntelligentLLMAgent`.

---

## 1. What this project is

An **experiment machine for LLM "regents" (controllers) that try to control complex systems** —
national economies *and* companies *and* other complex systems (economies are **domain #1**, not the
framework) — and a testbed for **making such controllers more effective via harnesses**. Re-scoped
2026-06; the authoritative architecture + implementation plan is [`agents/09-grand-plan.md`](agents/09-grand-plan.md)
(read [`agents/08-open-problems-and-opportunities.md`](agents/08-open-problems-and-opportunities.md) first
for *why*). Originally (and below, for the economy domain) framed as a platform for intelligent control of
simulated economies.
A *world* (an economic model) runs step by step; a *regent* (an LLM, today; RL/evolutionary later)
periodically reads the world's state and emits an **economic policy as a one-line Python
expression** (e.g. `np.clip(-1.9*(current_x - target_x), -2, 2)`). The expression is validated and
executed in a sandbox each step. The research question: can such a regent find effective,
adaptive, "creative" ways to govern dynamic systems that rule-based agents cannot?

**Status (2026-06):** thesis prototype, defended (grade 10/10) in 2025. It works on toy worlds
(scalar linear stochastic systems). The target Mandel/Lagom agent-based macro model was built but
**never stabilized** and is not yet integrated. The current effort is to reorganize the code into
a clean, extensible platform and finally stabilize a real ABM. A follow-up article on
"regent systems" is planned.

## 2. Repo map — where to find what

| You want… | Look in |
|---|---|
| The main loop / how a run is wired | `govsim/simulation.py` (`run_simulation`) |
| The core contracts (`Policy`, `PolicyDescriptor`, base classes) | `govsim/utils/interfaces.py` |
| Economic models ("worlds") | `govsim/economic_models/` — `economic_models.py` (SimpleGrowth), `single_market_model.py`, `linear_stochastic_system.py`, `coupled_linear_stochastic_system.py` |
| Governing agents ("regents") | `govsim/governing_agents/` — `government_agents.py` (Static/Random/Test), `gov_agent_linear.py` (`IntelligentLLMAgent`, the LLM regent) |
| The policy code sandbox (RestrictedPython) | `govsim/utils/policy_utils.py` |
| The LLM client layer (OpenAI-*compatible*) | `govsim/core/llm/` (`client.py`, `cache.py`) — replaced the removed Gemini/LangChain layer |
| Prompt templates + formatting | `govsim/prompts/*.md`, `govsim/utils/prompts_utils.py` |
| All tunable parameters | `govsim/config.py` |
| Plots | `govsim/chart_generators/` |
| The Mandel ABM (orphan, not on the interface, not stable) | `mandel testing/mandel_test.py` and `.ipynb` |
| Raw run outputs | `logs/*.json` |
| Helper scripts | `scripts/` |

A precise, line-referenced critique of all of the above is in [`agents/02-code-audit.md`](agents/02-code-audit.md).

## 3. Where the *research* context lives (outside the repo)

The author keeps source materials in the **parent folder** of this repo
(`…/Диплом/`, i.e. `../` from here). These are **not** in git. Most useful:

| Context | Path (relative to repo root) |
|---|---|
| **Final defended thesis (source)** — full method, the 4 experiments, limitations, future work | `../Текст ВКР/thesis.tex` (+ `thesis.pdf`, figures, `document.bib`) |
| Defended thesis PDF + defense slides | `../Защита ВКР/Каримов_ВКР.pdf`, `…/Каримов_Всеволод_защита_ВКР.pptx` |
| **Advisor's review** (named limitations & next steps) | `../Защита ВКР/review1_karimov_v_t.pdf` |
| The feedback-loop architecture diagram | `../Защита ВКР/Feedback Loop of Economic System and LLM-Agent.svg` |
| Original project proposal (the full intended design, incl. "prediction emulation") | `../txt выжимки/Project Proposal.txt`, `../Project Proposal/` |
| Concept notes (vision, policy interface, creative-government idea) | `../txt выжимки/` (`Управляемая агентная модель.txt`, `Интерфейс экономических политик.txt`, `Модель креативного правительства.txt`, `Диплом.txt`, `TODO для Диплома.txt`, `Глоссарий.txt`) |
| **Mandel papers** (the target ABM family) | `../Литература по агентным моделям/AgentBasedDynamics.AntoineMandel2012.pdf`, `Mandel 2010.pdf`, `mandel-fuerst-lass-meissner-jaeger…2009-01.pdf`; text extracts in `../txt выжимки/Mandel*.txt` |
| ABM background / LLM-for-control | `../Литература по агентным моделям/` (Tesfatsion ACE; "Do complex financial models really lead to complex dynamics?"; "Управление системой с помощью LLM.pdf"; "Simulation of the Role of Government in Spatial Agent…") |
| Older flat version of the codebase | `../code_legacy/` |
| Early article drafts | `../document/` (`document.tex`, `report.tex` + PDFs) |

When a task needs "why is it designed this way", start with the proposal and the thesis; when it
needs "what's the target economy", start with the Mandel papers and
[`agents/04-mandel-stabilization.md`](agents/04-mandel-stabilization.md).

## 4. Setup & run

```bash
# 1. Install deps + create the venv (uv; Python >=3.12,<3.14)
uv sync                      # add `--group notebook` for jupyter/ipykernel

# 2. Provide an LLM key — create .env in the repo root. The engine is OpenAI-*compatible*
#    (any provider / local model / proxy), selected by base_url + model in config, NOT hardcoded:
#    OPENAI_API_KEY="..."           # or whichever api_key_env the client is configured with
#    OPENAI_BASE_URL="..."          # optional: OpenRouter / vLLM / Ollama / a proxy

# 3. Run an experiment through the real CLI (Phase 0 — the primary entry point; key-free baselines)
uv run govsim list                       # registered experiments
uv run govsim run cubic_stabilization    # deterministic baseline, no API key
uv run python -m govsim run cubic_nonlinear --seeds 0 1 2 --store logs/runs --plot

# 4. Tests (all key-free)
uv run pytest
```

> **Status update (Phase 0–1 + Phase-2 machinery done):** the real CLI exists (`govsim/__main__.py` +
> `govsim/experiments/` registry); `python -m govsim run <experiment>` reproduces a fixed trajectory
> from `(spec, seed)` (golden-master), and `python -m govsim compare A B` runs a paired bootstrap
> comparison. The new domain-agnostic stack supersedes the config-driven `simulation.py` path:
> - `govsim/core/` — six seams + `Experiment`/`Runner`/`ResultStore` + `llm/` cache-replay +
>   `rollout.py` (the `RollableSystem`-gated fitness oracle).
> - `govsim/domains/scalar/` — `CubicSystem` (linear/cubic), `CoupledSystem`, `SIRSystem`,
>   `CompanySystem`; `ScalarLeverInterface`; objectives.
> - `govsim/regents/` — `LLMRegent` (+ 4-source/obfuscated prompt assemblers + `check_prompt`),
>   `PIDRegent`, `LQRRegent`, `OPRORegent`.
> - `govsim/harness/` — `TraceFeedback`, `EpisodicMemory` (rollout-free), `RolloutProbe` (gated),
>   `Critic`.
> - `govsim/analysis/` — paired bootstrap CI, variance-aware selection, collapse detector.
>
> Registered experiments (see `govsim list`): cubic (linear sanity + nonlinear + LLM + OPRO +
> obfuscated + critic), coupled (stabilization + regime-shift), SIR, company. The LLM stack is
> **OpenAI-compatible** and was verified live against a vLLM cluster (`Openai/Gpt-oss-120b`,
> tool-calling) with byte-exact replay. The legacy `uv run simulation` /
> `economic_models`/`governing_agents` path still imports but is deprecated (removal deferred to
> Phase 4). See `agents/09-grand-plan.md` and `govsim/docs_gates/STATUS.md`.

To choose what runs, edit `govsim/config.py`:
- `SIMULATION_CONFIG.economic_model_type` ∈ {`LinearStochasticSystem`, `CoupledLinearStochasticSystem`, `SimpleGrowthModel`, `SingleMarketModel`}
- `SIMULATION_CONFIG.government_agent_type` ∈ {`IntelligentLLMAgent`, `StaticPolicyAgent`, `RandomAgent`, `TestPoliciesAgent`}
- `SIMULATION_CONFIG.agent_decision_frequency`, `total_steps`, and the per-model / per-agent param blocks.

## 5. How a regent decides (the policy DSL contract)

1. The world exposes `get_policy_descriptors()` → `PolicyDescriptor`s: each says
   `policy_type_id`, `value_type`, `value_range`, and **`available_context_vars`** (the only
   names allowed in the expression).
2. The regent returns a `Policy` whose `value_expression` is a **single Python expression** using
   only those vars + whitelisted `math`/`numpy` helpers (no statements, imports, or side effects).
3. `govsim/utils/policy_utils.py` validates identifiers, compiles under RestrictedPython, and the
   world re-evaluates it each step. The world clips the result to `value_range`.

If you add a new control surface, add a `PolicyDescriptor` in the world — do **not** widen the
sandbox unless you mean to.

## 6. Gotchas that will bite you (see audit for line refs)

> **These are the DEPRECATED legacy path's gotchas (`economic_models`/`governing_agents`/
> `simulation.py`), all RESOLVED in the new `govsim/core` + `domains` + `regents` stack:** the regent is
> domain-blind (no context-class coupling); every `System` owns a `np.random.Generator` (a CI test bans
> bare RNG); the cubic is parameterized (`cubic_coeff`×`state_exponent`), not an uncommitted edit; the
> eval-cadence contract (re-eval per step) is enforced in one place; reproducibility is real via the
> LLM cache/replay tape. Treat the list below as historical context for the legacy code only.

- **The LLM regent only works on the two linear worlds.** `IntelligentLLMAgent` hard-checks
  `LinearSystemAgentContext`; `SimpleGrowthModel`/`SingleMarketModel` return a different context
  shape. Fixing this is refactor step 3.
- **Nothing is seeded.** Runs are not reproducible yet. Add seeding before trusting any results.
- **`linear_stochastic_system.py` has an uncommitted cubic term** (`x**3`) marked "temporary".
  The "linear" model is currently nonlinear. Confirm intent before relying on it.
- **`RandomAgent` is broken** (bad import of a removed symbol).
- **Logging is `print()`**; `config.log_level` is ignored.
- **Policy eval semantics differ:** most worlds re-evaluate each step; `SingleMarketModel`
  evaluates once at apply time.
- **`emulate_policy` is unimplemented** everywhere (needed for "prediction emulation" and for
  future learning regents).

## 7. How to work in this repo

- **Branches:** work happens on `dev`; `main` is the stable branch (use it for PRs). Branch off
  `dev` for a change; keep it green.
- **Don't big-bang rewrite.** Follow the incremental migration in
  [`agents/03-refactor-plan.md`](agents/03-refactor-plan.md): one shippable, tested step per PR,
  with re-export shims so old import paths keep working mid-migration.
- **Tests:** there are none yet. The first refactor step adds a golden-master test for a thesis
  experiment plus sandbox/determinism tests — add to these; run `uv run pytest`.
- **Add-a-thing should be one file + one registration**, never an edit to the core loop. If you
  find yourself editing `simulation.py` to add a world/agent, that's the signal to do the
  registry refactor first.
- **Commits:** history is in Russian; match the surrounding style or write English — be
  consistent within a PR. Keep secrets (`.env`) out of git.
- **Before changing a model's economics**, read the relevant thesis/Mandel section in §3 so you
  preserve the intended semantics.

## 8. Index of the planning docs (`agents/`)

| Doc | What it gives you |
|---|---|
| [`agents/01-landscape-and-positioning.md`](agents/01-landscape-and-positioning.md) | Engineering brief: what the field built in 2025–2026 (AlphaEvolve, generative agents, Vending-Bench, …), how this project relates, what to borrow |
| [`agents/02-code-audit.md`](agents/02-code-audit.md) | Line-referenced inventory of bugs, smells, and what to keep |
| [`agents/03-refactor-plan.md`](agents/03-refactor-plan.md) | Target architecture, package layout, interface redesign, migration order |
| [`agents/04-mandel-stabilization.md`](agents/04-mandel-stabilization.md) | Why the Mandel ABM diverges and a concrete plan to stabilize + integrate it |
| [`agents/05-roadmap.md`](agents/05-roadmap.md) | Phased roadmap from prototype to the "regent-systems" research platform |
| [`agents/06-model-integration.md`](agents/06-model-integration.md) | **The recommended design** for near-automatic model registration/integration + wrapping legacy models (e.g. Mandel) without rewriting them |
| [`agents/07-governance-interface.md`](agents/07-governance-interface.md) | **How the regent governs** — conserved Effects (no thin-air resources), Ledger + Mediator (action cost, rate limits, control noise), Institutions as declarative specs or sandboxed effect-API code, tool-calling transport |
| [`agents/08-open-problems-and-opportunities.md`](agents/08-open-problems-and-opportunities.md) | **Course-correction review — READ FIRST.** Red team of 01–07: HOW-before-WHAT meta-problem, critical weaknesses, decide-from-the-start choices, dead-ends to cut, revised prioritization (supersedes 05) |
| [`agents/09-grand-plan.md`](agents/09-grand-plan.md) | **AUTHORITATIVE architecture + implementation plan.** Experiment machine for LLM regents controlling complex systems; `ActionInterface` domain seam; multi-regent + harness-as-research; phased rewrite → test-driven growth. Supersedes 05; repositions 06/07 |
