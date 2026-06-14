# Code Audit — `creative-governance-sim` (as of 2026-06)

Status: written by an AI dev agent after a full read of the repository, the supporting
materials folder, the defended thesis (`Текст ВКР/thesis.tex`), and the advisor's review.
Scope: describe the code **as it is today**, name concrete problems, and separate
"bug / will-bite-you" from "smell / will-slow-you-down" from "keep / this is good".

The companion documents are [`03-refactor-plan.md`](03-refactor-plan.md) (the target
architecture and migration), [`04-mandel-stabilization.md`](04-mandel-stabilization.md),
and [`05-roadmap.md`](05-roadmap.md). The repo map for orientation lives in the root
[`AGENTS.md`](../AGENTS.md).

---

## 1. What the system does (one paragraph)

A *world* (an `BaseEconomicSystem` subclass) simulates an economy step by step. A *regent*
(a `BaseGovernmentAgent` subclass) is invoked periodically; it reads world state + history and
returns a `Policy` — a one-line Python **expression as a string** (e.g.
`np.clip(-1.9*(current_x - target_x), -2, 2)`). The expression is validated and compiled in a
RestrictedPython sandbox, then the world re-evaluates it each step to drive a control variable.
The `IntelligentLLMAgent` produces these expressions with an LLM (Gemini). The two-level
`PolicyDescriptor` (what the world allows) / `Policy` (what the agent decided) interface is the
conceptual *idea* worth keeping — but its current *realization* is weak and leaks badly (see §5; this corrects an earlier over-charitable "the core is strong" framing). Everything else is scaffolding of uneven quality.

## 2. Current module map

| Area | File(s) | Role |
|---|---|---|
| Entry / orchestration | `govsim/simulation.py` | `run_simulation()` — the main loop, factories, JSON dump |
| Entry (broken) | `govsim/__main__.py` | **empty** (1 line) despite README documenting `python -m govsim <exp>` |
| Core interfaces | `govsim/utils/interfaces.py` | `Policy`, `PolicyDescriptor`, `BaseEconomicSystem`, `BaseGovernmentAgent` |
| Worlds | `govsim/economic_models/economic_models.py` | `SimpleGrowthModel` |
| | `.../single_market_model.py` | `SingleMarketModel` (+ `SMM_Consumer`, `SMM_Firm`) |
| | `.../linear_stochastic_system.py` | `LinearStochasticSystem` (+ `LinearSystemAgentContext`) |
| | `.../coupled_linear_stochastic_system.py` | `CoupledLinearStochasticSystem` |
| Regents | `govsim/governing_agents/government_agents.py` | `StaticPolicyAgent`, `RandomAgent`, `TestPoliciesAgent` |
| | `.../gov_agent_linear.py` | `IntelligentLLMAgent` (LLM regent) |
| Policy sandbox | `govsim/utils/policy_utils.py` | RestrictedPython compile/eval, `SandboxConfig` |
| LLM layer | `govsim/utils/gemini_utils.py` | LangChain + google-generativeai; `BaseAgent`/`LaconicAgent`/`CheckerAgent`; `create_agent` |
| Prompt utils | `govsim/utils/prompts_utils.py`, `govsim/prompts/*.md` | template loading & descriptor formatting |
| Config | `govsim/config.py` | plain dicts: `SIMULATION_CONFIG`, `ECONOMIC_MODEL_PARAMS`, `GOVERNMENT_AGENT_PARAMS` |
| Viz | `govsim/chart_generators/*` | per-experiment matplotlib scripts |
| Mandel (orphan) | `mandel testing/mandel_test.py` (+`.ipynb`) | 1411-line standalone ABM, **not** on the interface, never stabilized |
| Misc | `scripts/`, `logs/`, `combined_code.txt` | helper scripts, raw run logs, an LLM-feed concat dump |

## 3. Bugs / correctness issues (fix or decide explicitly)

1. **Empty `__main__.py` — the documented CLI does not exist.** `README.md:54-73` tells the
   user to run `poetry run python -m govsim linear` / `adaptive` and "see `--help`". None of
   that is wired; `__main__.py` is empty. The only working entry point is `poetry run simulation`
   (which only obeys `config.py`). Either implement the CLI or fix the docs — right now a new
   contributor's first command fails.

2. **`RandomAgent` is broken.** `government_agents.py:199-204` defines an `allowed_math_funcs`
   property that does `from policy_utils import ALLOWED_MATH_NAMES`. That import path is wrong
   (it would be `govsim.utils.policy_utils`) **and** `ALLOWED_MATH_NAMES` no longer exists after
   `policy_utils.py` was rewritten onto RestrictedPython. Any `set_tax_rate` path that hits
   `choice == 2` (`government_agents.py:78`) raises. `RandomAgent` is registered as a usable
   agent in `simulation.py:39-44` but is effectively dead for its main world.

3. **The "linear" model is secretly cubic.** `linear_stochastic_system.py:242` computes
   `next_x = self.param_A * self.current_x ** 3 + ...` with the comment *"временный эксперимент,
   не забыть убрать"*. This is an **uncommitted local change** (`git status` shows the file
   modified). It silently contradicts the class name, the docstring (`x_{k+1}=A·x_k+B·u_k+C+ε`),
   the thesis (which describes a linear model), and the obfuscated prompt (which hints at a
   "third-degree main term"). Decide: is this a deliberate nonlinearity experiment (then make it
   a parameter, e.g. `state_exponent`, and rename/relabel) or leftover scaffolding (then revert)? **Resolved by [`08-open-problems-and-opportunities.md`](08-open-problems-and-opportunities.md) §5: KEEP it** (parameterize `state_exponent` + the prompt info-level) — `git status` shows the cubic, the obfuscated prompt, and `config.py` were changed together: a deliberate partial-information nonlinear-control experiment (the repo's most novel arm), not stray scaffolding.

4. **Duplicate `select_model` in `gemini_utils.py`.** Two defs (`:53-89` and `:92-141`); the
   second silently shadows the first, so the first is dead code with different fallback behavior.
   Confusing and a latent footgun.

5. **Interface/implementation signature drift.** `BaseGovernmentAgent.decide_policy`
   (`interfaces.py:193-198`) declares a fourth arg `llm_extra_context`, but
   `IntelligentLLMAgent.decide_policy` (`gov_agent_linear.py:221-225`) omits it. It only works
   because `simulation.py:108-112` calls positionally with three args. Add one keyword and it breaks.

6. **The LLM regent only works with the two linear worlds.** `IntelligentLLMAgent`
   hard-asserts `isinstance(current_state_for_agent, LinearSystemAgentContext)`
   (`gov_agent_linear.py:232-238`) and imports that class directly. But `get_state_for_agent`
   has **two incompatible contracts**: `SimpleGrowthModel`/`SingleMarketModel` return a plain
   `dict {metrics, active_policies}`, while `LinearStochasticSystem`/`CoupledLinearStochasticSystem`
   return the Pydantic `LinearSystemAgentContext`. So the "intelligent" regent silently refuses
   to govern `SimpleGrowthModel` and `SingleMarketModel` even though `simulation.py` lets you pair
   them. This is the single biggest architectural coupling problem.

7. **KPI computation is world-specific but lives in the agent.**
   `_calculate_performance_kpis` (`gov_agent_linear.py:59-85`) assumes `current_x`/`target_x`
   metrics. On worlds without them it yields `None`/garbage. Objectives/KPIs belong to the world
   (or a pluggable objective), not baked into one agent.

8. **Tangled decision-frequency condition.** `simulation.py:101`:
   `(current_step + 1) % freq == 0 and current_step+1 >= 1 and current_step+1 < total_steps`.
   The `>= 1` is always true; the `< total_steps` silently means the agent never decides on the
   final step; the whole thing conflates scheduling with edge-trimming. Hard to reason about and
   to reproduce the thesis experiments (which used irregular schedules like "steps 200, 600, 800").

9. **No reproducibility.** Both `random` and `numpy.random` are used (worlds use `random`,
   agents/`numpy`), and **nothing is ever seeded**. Runs are not reproducible — a hard
   requirement for a scientific platform, and the thesis explicitly offers the repo for
   reproduction. (See `linear_stochastic_system.py:239`, `mandel_test.py` throughout.)

10. **Two different notions of `previous_x`.** `LinearStochasticSystem.get_current_metrics`
    derives `previous_x` from `history[-2]` (`:142-147`), while `get_state_for_agent` derives a
    separate `prev_x` from `history[-1]` (`:153-171`). They disagree by one step; the agent and
    the policy-eval context can see different "previous" values.

11. **Policy execution semantics differ across worlds.** `LinearStochasticSystem`,
    `CoupledLinearStochasticSystem`, and `SimpleGrowthModel` re-evaluate the policy expression
    **every step** inside `step()`. `SingleMarketModel.apply_policy_change` (`:251-314`) instead
    evaluates the expression **once at apply time** and stores the resulting scalar. Same
    `Policy` object, two different runtime meanings. Pick one contract (re-eval each step) and
    enforce it.

12. **`emulate_policy` is unimplemented everywhere.** It is a headline feature of the original
    Project Proposal ("simplified prediction emulation" — test a policy on a model subset/clone).
    Every world stubs or raises (`interfaces.py:158-178`, and each model's bottom method). This
    same primitive (clone + rollout + score) is also what any evolutionary or RL regent will need.

## 4. Smells / maintainability (will slow extension)

- **Config is untyped plain dicts** (`config.py`). `pydantic` is a dependency but unused for
  config; param typos fall through `.get(..., default)` silently. No validation, no schema, no
  per-experiment files.
- **DRY violation:** `_format_policy_descriptors_for_prompt` exists identically in both
  `gov_agent_linear.py:87-99` and `prompts_utils.py:42-54`.
- **LLM layer is over-built and import-fragile.** `gemini_utils.py` mixes LangChain and the
  google-generativeai SDK (its own top TODO admits this is probably redundant), calls
  `genai.configure` at import time, and **raises if `GOOGLE_API_KEY` is missing on import** — so
  importing the agent module hard-fails without a key (the agent file even comments on this at
  `gov_agent_linear.py:16`). It is also **single-provider** (Google only).
- **`print()` everywhere instead of `logging`.** `config.py` has a `log_level` that is never
  used. No way to silence per-module output or set levels; stdout is noisy.
- **No typed result schema.** History is `list[dict]` with metrics recomputed and embedded;
  output is one big hand-serialized JSON via a custom `default_serializer`. Hard to query,
  diff, or aggregate across runs.
- **`mandel_test.py` is an unmodularized 1411-line monolith** with dozens of magic constants and
  comment trails like `MODIFIED`, `ЗНАЧИТЕЛЬНО УМЕНЬШЕНО ДЛЯ СТАБИЛЬНОСТИ` — i.e. evidence of
  blind hand-tuning. It has **no stock-flow / accounting conservation checks**, couples
  `matplotlib.show()` into the run, and is not on the `BaseEconomicSystem` interface. (Its own
  problems are the subject of [`04-mandel-stabilization.md`](04-mandel-stabilization.md).)
- **No tests, no CI.** `tests/` holds a Gemini demo and a manual notebook. There is no way to
  catch the regressions above automatically.
- **Experiment-specific viz.** `chart_generators/special_visualize_exp4.py`,
  `visualize_universal_legacy.py` — ad hoc, named after one-off experiments.
- **Vestigial tooling.** `combined_code.txt` + `scripts/combine_scripts.py` concatenate all
  source into one text file for pasting into an LLM. With `AGENTS.md` + proper tooling this is
  obsolete and just goes stale.
- **Mixed RU/EN identifiers, strings, and comments.** Fine internally, but for the intended
  international/open research platform pick one language for the public API surface
  (recommendation in `03-refactor-plan.md`).
- **Secret hygiene:** confirm `.env` is gitignored in the repo (it is referenced by
  `gemini_utils.py`). The parent materials folder contains `.env` / `.env.txt` — keep those out
  of any repo.

## 5. What to keep vs. what is rewrite material

An earlier draft of this audit called the core "strong." That was too charitable: the *concept* is
sound, but the *machinery implementing it* is weak — manual three-place sync, hand-maintained
registries, per-world dispatch boilerplate, and a context class that couples the agent to one world.
Separate the two honestly. The chosen replacement is [`06-model-integration.md`](06-model-integration.md).

**Keep (genuinely good):**
- **The RestrictedPython sandbox.** `policy_utils.py` with a configurable `SandboxConfig` (allowed
  builtins / math / numpy) is the right safety primitive, reasonably clean, and the load-bearing
  guarantee (out-of-whitelist identifiers fail at *compile* time). Keep as-is.
- **The *idea* behind `Policy` / `PolicyDescriptor`** — "the world declares *what can be done*; the
  regent decides *how*" (the thesis and review single it out). Keep the `Policy` *shape*
  (`policy_type`, `value_expression`, `_compiled_safe_code`) as the downstream contract — but
  `PolicyDescriptor` should be **generated from a declaration**, not hand-written per world.

**Rewrite material (the "core" that isn't strong):**
- **The registration machinery.** The hand `create_*` dicts in `simulation.py` + the parallel param
  dicts in `config.py` are pure friction (3+ files to add a world). Replace with auto-registration
  via `__init_subclass__` — `06` §5.
- **The stringly-typed three-place sync** (`available_context_vars` vs `get_current_metrics()` keys
  vs prompt placeholders). Replace with a single declarative source of truth from which the
  whitelist, eval context, and descriptors are *all derived* — `06` §2.
- **The per-world boilerplate** (`apply_policy_change` + the `if policy_type == ...` dispatch in
  `step()`). Generate it from the declaration — `06` §2.2.
- **The Pydantic `LinearSystemAgentContext`.** This is the *coupling*, not a strength — it is why the
  LLM regent governs only the linear worlds. Replace with a uniform `dict` context every world
  returns — `06` §2.2 / §6.
- **The `BaseEconomicSystem` ABC** survives, but only as the base the new declarative `WorldSpec`
  extends — `06` keeps it precisely so `simulation.py`'s loop runs migrated worlds unchanged.

## 6. Severity-ordered fix list (quick reference)

| # | Item | Severity | Effort |
|---|---|---|---|
| 1 | LLM regent locked to `LinearSystemAgentContext` (§3.6) | High (blocks generality) | Medium |
| 2 | No RNG seeding / reproducibility (§3.9) | High (science) | Low |
| 3 | Empty `__main__.py` vs documented CLI (§3.1) | High (onboarding) | Low |
| 4 | `RandomAgent` broken import (§3.2) | Medium | Low |
| 5 | Cubic term in "linear" model — decide intent (§3.3) | Medium (correctness/clarity) | Low |
| 6 | Policy eval semantics differ across worlds (§3.11) | Medium | Medium |
| 7 | Untyped config (§4) | Medium (extensibility) | Medium |
| 8 | `print` → `logging`; typed run records (§4) | Medium | Medium |
| 9 | Duplicate `select_model`, DRY formatter, signature drift (§3.4, §3.5, §4) | Low | Low |
| 10 | `emulate_policy` / clone+rollout missing (§3.12) | Strategic (enables learning) | Medium-High |

These map directly onto the phased migration in [`03-refactor-plan.md`](03-refactor-plan.md).
