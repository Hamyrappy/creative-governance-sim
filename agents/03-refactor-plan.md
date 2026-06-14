# Refactoring Plan — toward a "regent-systems" research platform

This is the concrete plan for reorganizing `creative-governance-sim` from a thesis prototype
into an extensible platform for studying intelligent control systems ("regents") that govern
simulated economies. It builds on the problems catalogued in [`02-code-audit.md`](02-code-audit.md).

Guiding constraint: **incremental and always-green**. Every step below is a small change that
keeps `poetry run simulation` working. Do not do a big-bang rewrite — the prototype already
produces thesis-grade results, and we want to preserve that while widening the foundation.

---

## 1. Design principles (what "good" means here)

1. **The world declares capabilities; the regent decides behavior.** Keep the
   `PolicyDescriptor`/`Policy` split — it is the project's best idea. Generalize it so *any*
   regent can drive *any* world.
2. **The agent must be world-agnostic.** No regent should `import` or `isinstance`-check a
   specific world's context class. Worlds and regents communicate through stable contracts only.
3. **Everything is reproducible.** One seed per run, threaded through every RNG, recorded in the
   run output. No global `random`/`np.random` state.
4. **Configuration is data, validated.** Experiments are declarative specs, type-checked at load.
5. **Observability over print().** Structured logging + typed, queryable run records.
6. **Make the keystone primitive real:** `world.clone()` + `world.rollout(policy, horizon) -> score`.
   It powers the proposal's "prediction emulation" *and* every future learning/evolutionary regent.
7. **Pluggable seams:** worlds, regents, objectives, LLM providers, decision schedules. Each is an
   interface with a registry, so adding one is a new file + one registration line — never an edit
   to the core loop.

## 2. Target package layout

> **Registration, integration, and the world base class are now specified concretely in
> [`06-model-integration.md`](06-model-integration.md)** (a declarative `WorldSpec` on the existing
> `BaseEconomicSystem`; auto-registration via `__init_subclass__`; `clone()`/`rollout()`; a uniform
> `dict` agent context). Where `06` and the sketch below differ, **`06` wins** — e.g. registration
> lives in the world base (not a separate `core/registry.py`), and there is no `context/` package
> (the context is a plain dict). The layout below is the broader target; `06` is the load-bearing detail.

Rename for clarity ("economic_models" → `worlds`, "governing_agents" → `regents`) and split the
overloaded `utils/` into honest packages. Proposed:

```
govsim/
  core/
    interfaces.py        # Policy, PolicyDescriptor, BaseWorld, BaseRegent, AgentContext (base), Objective
    runner.py            # the simulation loop, decoupled from config.py
    schedule.py          # DecisionSchedule: EveryN | AtSteps | Adaptive
    rng.py               # SeededRng: one np.random.Generator, child streams per agent/world
    results.py           # pydantic RunRecord (config + seed + time series + decisions)
    registry.py          # name -> class registries for worlds/regents/objectives/providers
    logging.py           # logging setup honoring config log level
  policy/
    sandbox.py           # RestrictedPython compile/eval (from utils/policy_utils.py)
    config.py            # SandboxConfig
  worlds/
    base.py              # BaseWorld (was BaseEconomicSystem) + default clone()/rollout()
    linear_stochastic.py
    coupled_linear_stochastic.py
    simple_growth.py
    single_market.py
    mandel/              # the Mandel model, modularized (see 04-mandel-stabilization.md)
      world.py  firm.py  household.py  sector.py  government.py  finance.py  accounting.py
  regents/
    base.py              # BaseRegent (was BaseGovernmentAgent)
    static.py random.py test.py
    llm_regent.py        # model-agnostic LLM regent (was gov_agent_linear.IntelligentLLMAgent)
    # future: evolutionary_regent.py  rl_regent.py  hybrid_regent.py  ensemble_regent.py
  context/
    base.py              # AgentContext base (step, metrics, history, descriptors_text, objective, KPIs)
    # per-world subclasses if a world needs extra fields
  llm/
    client.py            # LLMClient Protocol: __call__(prompt)->str ; structured(prompt, schema)->obj
    providers/
      google.py          # current Gemini code, cleaned, lazy key handling
      # future: anthropic.py  openai.py
    prompting.py         # template load + descriptor/context/history formatting (de-duplicated)
  objectives/
    base.py              # Objective: score(history, world) -> float ; reward(...) for RL
    stabilization.py     # MSE/MSU
    welfare.py gdp_growth.py
  experiments/
    registry.py          # name -> ExperimentSpec
    linear_single_intervention.py  ...  (the 4 thesis experiments as specs)
  viz/                   # was chart_generators; generic plotters keyed off RunRecord
  cli.py                 # argparse/typer: run / list / viz / sweep
  __main__.py            # delegates to cli.py  (fixes the README contract)
  config/                # pydantic Settings + default experiment configs (py or yaml)
tests/
  test_sandbox.py test_runner.py test_worlds_determinism.py test_mandel_invariants.py ...
```

Keep thin re-export shims at the old import paths during migration (e.g.
`govsim/utils/interfaces.py` re-exports from `govsim/core/interfaces.py`) so nothing breaks
mid-flight; delete the shims at the end.

## 3. Interface redesign (the load-bearing changes)

### 3.1 One agent-context contract (fixes audit §3.6)

> **Refined by [`06-model-integration.md`](06-model-integration.md):** the design panel concluded a
> Pydantic `AgentContext` is over-engineered, and that a **uniform plain `dict`** context returned by
> every world is simpler and equally decoupling. The prompt is then assembled by the agent from four
> sources (metrics + model params + objective KPIs + formatted text) and guarded by a build-time
> validator. Prefer that; the sketch below is kept for intent.

Define `AgentContext` in `core/interfaces.py` as the *only* thing a regent consumes:

```python
class AgentContext(BaseModel):
    step: int
    metrics: dict[str, float]          # world-defined, flat, JSON-safe
    target: dict[str, float] | None    # objective targets, if any
    kpis: dict[str, float | None]      # computed by the Objective, not the agent
    history_text: str                  # pre-formatted recent history
    policy_descriptors_text: str       # pre-formatted descriptors
    available_context_vars: list[str]  # union allowed in expressions
    extra: dict[str, Any] = {}         # world-specific escape hatch (optional)
```

`world.get_agent_context(objective, history) -> AgentContext`. Worlds that need extra structured
fields subclass it, but **regents only depend on the base**. Delete the `isinstance` check and
the direct `LinearSystemAgentContext` import from the LLM regent.

### 3.2 Objectives become first-class (fixes audit §3.7)
Move MSE/MSU out of the agent into `objectives/stabilization.py`. An `Objective` exposes
`kpis(history, world) -> dict`, `target(world) -> dict`, and (for learning) `reward(...)`. The
world stays agnostic about *which* objective is in play; the experiment spec picks one. This is
the abstraction that lets the same world be scored by stabilization, welfare, or growth — and is
required for evolutionary/RL regents.

### 3.3 One policy-execution contract (fixes audit §3.11)
Mandate: **policies are re-evaluated every `world.step()`** against a world-provided eval context.
Fix `SingleMarketModel` to follow this (it currently evaluates once at apply time). Document the
contract in `policy/sandbox.py` and assert it in a test.

### 3.4 `clone()` + `rollout()` (fixes audit §3.12; keystone)
Add to `BaseWorld`:

```python
def clone(self) -> "BaseWorld": ...                 # deep, RNG-independent copy of state
def rollout(self, policy, horizon, seed) -> RolloutResult: ...  # pure, no mutation of self
```

A correct default can be `copy.deepcopy(self)` with a fresh child RNG; worlds with non-copyable
handles override. This single method implements `emulate_policy` *and* serves as the fitness
evaluator for evolutionary and RL regents.

### 3.5 Consistent regent signature & decision scheduling (fixes audit §3.5, §3.8)
`BaseRegent.decide(context: AgentContext, world: BaseWorld) -> Policy | None` — one signature,
no positional-only hacks. Move the tangled modulo logic out of the loop into
`core/schedule.py` (`EveryN(n)`, `AtSteps([...])`, `Adaptive(predicate)`); the runner just asks
`schedule.should_decide(step)`.

## 4. Cross-cutting infrastructure

- **Reproducibility (`core/rng.py`):** a `SeededRng` wrapping one `np.random.Generator`; worlds
  and regents draw from named child streams. Seed is a required experiment field and is written
  into the `RunRecord`. Replace every bare `random.*` and `np.random.*`.
- **Typed config (`config/`):** pydantic models for `SimulationConfig`, `WorldParams`,
  `RegentParams`. An `ExperimentSpec` bundles world+params, regent+params, objective, schedule,
  steps, seed. Load from py or YAML; validate on load; fail loud on unknown keys.
- **Registries (`core/registry.py`):** `@register_world("LinearStochastic")` decorators replace
  the hand dicts in `simulation.py:23-49`. The CLI and experiment registry both read from here.
- **Logging (`core/logging.py`):** replace `print` with module loggers honoring `log_level`.
  Keep a `--verbose` for the LLM I/O dumps.
- **Results (`core/results.py`):** one `RunRecord` per run (config, seed, per-step metrics,
  decisions with reasoning, timing). JSON by default; this is what `viz/` consumes, so plots stop
  being experiment-specific.
- **LLM provider abstraction (`llm/`):** an `LLMClient` Protocol; `providers/google.py` wraps the
  existing Gemini code with **lazy** key handling (don't `raise` at import — fail only on first
  call), and the duplicate `select_model` collapsed to one. Add `providers/anthropic.py` later.
  Per the environment guidance, default new LLM work to the latest Claude models while keeping
  Gemini supported. Decouple from LangChain unless structured-output is actually needed.

## 5. Migration order (each step is shippable and tested)

> Rule of thumb: land one numbered step per PR, with tests, keeping `main` green.

1. **Foundations, no behavior change.** Add `core/rng.py` (seed everything), `core/logging.py`
   (swap prints), pydantic config in `config/`. Snapshot a thesis experiment as a golden-master
   test so later refactors can't silently change results.
2. **Carve out `core/`.** Move `Policy`/`PolicyDescriptor`/base classes into `core/interfaces.py`;
   add the uniform `dict` context + `Objective` (per [`06-model-integration.md`](06-model-integration.md)). Leave re-export shims at old paths.
3. **Generalize the LLM regent.** Make `llm_regent` consume `AgentContext` only; move KPIs to an
   `Objective`. Acceptance test: the LLM regent now governs `SimpleGrowthModel` and
   `SingleMarketModel`, not just the linear worlds.
4. **Unify policy execution.** Make `SingleMarketModel` re-evaluate per step; add the contract
   test. Fix the cubic-term ambiguity (audit §3.3) — parameterize or revert, explicitly.
5. **`clone()` + `rollout()`** on `BaseWorld`; wire `emulate_policy` to it; add a regent flag to
   use rollout-based evaluation before committing a policy ("prediction emulation").
6. **Real CLI + experiment registry.** Implement `cli.py`/`__main__.py` (`run`, `list`, `viz`,
   `sweep`); register the 4 thesis experiments as specs; update `README.md` to match reality.
7. **Modularize Mandel** into `worlds/mandel/` behind `BaseWorld`, executed in lockstep with
   [`04-mandel-stabilization.md`](04-mandel-stabilization.md) (SFC invariants, seeding, staged
   feature flags). This is where audit + stabilization meet.
8. **Learning regents.** With objectives + rollout in place, add `evolutionary_regent.py`
   (policy population, LLM as mutation/crossover operator, rollout as fitness — AlphaEvolve-style;
   see [`01-landscape-and-positioning.md`](01-landscape-and-positioning.md)) and later
   `rl_regent.py`.
9. **Provider abstraction + cleanup.** Land `llm/providers/`, delete shims, remove
   `combined_code.txt`/`combine_scripts.py`, retire experiment-specific viz.

Steps 1-6 turn the prototype into a clean, general, reproducible framework. Steps 7-9 turn it
into the research platform.

## 6. Decisions to confirm with the author before coding

- **Public-API language:** keep RU comments but make identifiers/docstrings/CLI English for an
  international OSS? (Recommended; the planned "regent-systems" article is English-facing.)
- **Config format:** Python specs vs YAML. (Recommend pydantic + YAML for sweeps.)
- **Keep LangChain** or go provider-SDK-only behind the `LLMClient` Protocol. (Recommend the
  latter; LangChain is only pulling weight for structured output, which the SDKs now do natively.)
- **Rename packages now or keep `economic_models`/`governing_agents`** with shims indefinitely.
  (Recommend rename with shims, delete shims at step 9.)

## 7. Definition of done for the refactor

- The LLM regent governs every registered world; no world-specific imports in any regent.
- Any run is reproducible from `(ExperimentSpec, seed)`; the seed is in the `RunRecord`.
- `pytest` covers the sandbox, world determinism, the policy-execution contract, and Mandel
  conservation invariants; CI runs it.
- `python -m govsim run <experiment>` works and matches the README.
- Adding a new world / regent / objective / provider is a new file + one `@register`, with no
  edit to `core/runner.py`.
