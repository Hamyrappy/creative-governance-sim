# 09 — The Grand Plan: an Experiment Machine for LLM Regents

> **Status: AUTHORITATIVE.** This document is the single source of truth for architecture and
> implementation. It **supersedes** the roadmap in [`05-roadmap.md`](05-roadmap.md), **repositions**
> [`06-model-integration.md`](06-model-integration.md) and [`07-governance-interface.md`](07-governance-interface.md)
> as the *economy domain plugin* (not the top level), and **honors** [`08-open-problems-and-opportunities.md`](08-open-problems-and-opportunities.md)
> as the governing discipline (WHAT-first gates, keep the cubic, LLM cache/replay, per-world Generator,
> thread `regent_id`/`polity_id`, the dead-ends-to-cut list).
>
> It is the lead-architect synthesis of four facet-proposals (core architecture, multi-regent,
> harness-as-research, sequencing) with their adversarial critiques. Where a proposal was overruled,
> §10 records why. It is grounded in the real files: `govsim/utils/interfaces.py`,
> `govsim/utils/policy_utils.py`, `govsim/simulation.py`, `govsim/config.py`,
> `govsim/governing_agents/gov_agent_linear.py`, `govsim/governing_agents/government_agents.py`,
> `govsim/economic_models/linear_stochastic_system.py`,
> `govsim/economic_models/coupled_linear_stochastic_system.py`, `govsim/utils/gemini_utils.py`.

---

## 1. Vision & scope

### 1.1 The two north stars

**Goal A — study.** We are building an **experiment machine for LLM "regents" (controllers) that try to
control different complex systems** — national economies *and* companies *and* other complex systems.
We must not hardcode transactional/economic governance at the highest level of abstraction. Economic
governance is **domain #1**, not the framework.

**Goal B — improve.** We do not only *research* regents; we develop **harnesses** — the scaffolding
(memory, rollout/probing, trace feedback, critic, reflection, population, inter-regent comms) that makes
an LLM controller *more effective*. The harness is a first-class, ablatable research object, not a fixed
wrapper. "How do we make LLM controllers of complex systems better?" is a question the machine answers
empirically.

### 1.2 The load-bearing inversion (vs docs 03/06/07)

Today the **economy is the top of the tree**: `BaseEconomicSystem` (`interfaces.py:104`) fuses *being a
system* with *being acted upon*, the only LLM regent is hard-bound to `LinearSystemAgentContext`
(`gov_agent_linear.py:12, 232`), and docs 06/07 specify a declarative `WorldSpec` and a conserved
`Effect`/`Ledger`/`Mediator` Chancery *as the framework*.

The re-scope inverts this. The top level is **six domain-agnostic seams** — `System`, `ActionInterface`,
`Regent`, `Harness`, `Objective`, and the experiment spine (`Experiment` / `Runner` / `ResultStore`) —
plus an OpenAI-compatible `LLMClient` seam with a cache/replay tape. The keystone is **`ActionInterface`**:
the per-domain vocabulary of *what a controller may do* and *the domain invariants*. It is the only
domain-coupled seam. Doc-07's entire Chancery collapses into **one** `ActionInterface` implementation
(`EconomyActionInterface`); doc-06's `WorldSpec`/`Observable`/`Control` becomes how an *economy* `System`
declaratively exposes state and levers to that interface. Neither is visible at the top level. An SIR
epidemic, a company, and the cubic all implement the same `System` ABC and bind to a trivial sibling
`ScalarLeverInterface` with **zero ledger code**.

### 1.3 The WHAT-first gates (doc-08, non-negotiable)

The machine *enforces* doc-08's discipline at runtime. An `Experiment` carries `hypothesis`,
`objective(s)`, and `creativity_metric` as **required fields**; the `Runner` refuses to run if any gate
is unset. Deciding WHAT (hypothesis + named baseline, which objective, is creativity measured) is a
**gate, not an afterthought**. This is stronger than doc-08 itself prescribed — there it was a discipline;
here it is an executable precondition that cannot rot.

### 1.4 Honest scope boundary (what this machine is, and is NOT)

The critiques correctly flagged that "control **any** complex system" is near-vacuous. We scope it
precisely:

> **In scope:** any system that can be expressed as **`observe → (bounded levers OR a domain action
> algebra) → step`, with a `clone()` for rollout where rollout-based harnesses are used.** Three families
> ship: (1) scalar/low-dim dynamical controllers (cubic, SIR, company-as-dynamics) via `ScalarLeverInterface`;
> (2) the economy domain (`EconomyActionInterface`) with a richer `Effect` action algebra; (3) — proven by
> construction in Phase 4 — a *structurally different* action algebra so generality is exercised, not
> asserted.

**Explicitly OUT of scope (state it, don't pretend otherwise):**

- **Free-form mechanism/organization redesign as the *only* action** (rewriting agents' behavioral rules
  arbitrarily, open-ended org-chart/hiring/R&D-portfolio design). The vision names "companies and other
  complex systems"; we serve the *controllable-lever / bounded-rule-coefficient* subset of those, not
  arbitrary institutional design. Rule-level levers are admitted **only when world-declared and bounded**
  (doc-07 open question (a); §3.4).
- **Systems with no clonable model** (a live external API world, a real datacenter) as targets of
  rollout-based harnesses. They may still be controlled by non-rollout harnesses; `clone()` lives on a
  `RollableSystem` capability, not the base `System` (critique-driven; §2.3).
- **A single creativity number identical across *all* domains.** The creativity construct is
  **domain-scoped** (§6.3): control domains use generalization-gap + functional-novelty-vs-best-fit-PID;
  the governance domain adds a Policy-Innovation-Score. We do **not** claim "best-fit-PID residual" is
  computable on a company P&L. Where no defensible construct separates the LLM from a tuned controller,
  we **drop the word "creativity"** and call it adaptive in-context control (doc-08 §3.1).
- **The full doc-07 Chancery now** (Mediator noise/lag/leakage, declarative DSL, versioned Code-of-Laws
  store, `Credit` primitive). Deferred behind a result, per doc-08 §5.
- **Multi-government league / cross-world leaderboard** (the v2 program-maximum). Hooks threaded now;
  built last.
- **Resurrecting Mandel as a prerequisite.** A tiny conserved-by-construction SFC toy is the economy
  rung; full Mandel / EconAgent adoption is an optional, gated track (doc-08 §3.3).
- **An RL regent arm and a Gymnasium adapter** for the first paper (doc-08 dead-end).

---

## 2. Top-level architecture

### 2.1 The layer stack (dependency direction is top → bottom; lower layers never import higher)

```
Experiment  = system_factory × {regent_id: Regent} × Harness × {regent_id: Objective}
              × jurisdictions × Schedule × seeds × Hypothesis × CreativityMetric
   │  Runner orchestrates the paired-seed loop → RunRecord → ResultStore
   ▼
Harness(es)   wrap each Regent; own rollout / memory / trace / critic / reflection / comms
   │  call LLMClient (→ CachingReplayClient); read Observation, write ActionRequest
   ▼
Regent(s)    decide(view, action_space, scratch) → [ActionRequest];   domain-BLIND
   ▼
ActionInterface (THE domain plugin — the only domain-coupled seam)
   │  action_space(system, regent_id) | validate(req,…) → Result | apply([req], system) → ApplyReport
   │   • economy plugin  = doc-07 Effect/Ledger/Mediator/Institutions, hidden ENTIRELY inside here
   │   • scalar plugin   = trivial lever validator over a sandboxed expr; NO ledger
   ▼
System (domain-agnostic)   rng · reset(seed) · step()→StepInfo · observe(viewer_id) · metrics()
   │   • economy System = doc-06 WorldSpec (Observable/Control declarations)
   │   • cubic / SIR / company System = plain dynamics, ~25 lines each
   ▼
(capability) RollableSystem.clone()    — only systems that can be cloned; rollout harnesses require it
```

`Objective`, `LLMClient`, `Schedule`, `ResultStore` are orthogonal services injected at the `Experiment`/
`Runner` level. **Acting is not a `System` concern** — that is the whole inversion. `System` advances
dynamics; `ActionInterface` installs/validates/applies what a regent did.

### 2.2 The seam interfaces (concrete sketches)

These live under `govsim/core/`. No name above the `ActionInterface` layer encodes a domain noun
(enforced by a CI test, §6.4 / §7.2).

```python
# govsim/core/system.py — a System is ANY controllable process. NOT "economy".
class System(ABC):
    rng: np.random.Generator                       # doc-08 HARD: per-system Generator, no bare random.*/np.random.*
    @abstractmethod
    def reset(self, seed: int) -> None: ...         # deterministic; seeds self.rng
    @abstractmethod
    def step(self) -> "StepInfo": ...               # advance ONE tick; {terminated, truncated, info}
    @abstractmethod
    def observe(self, viewer_id: str = "regent:0") -> "Observation": ...   # jurisdiction-scoped, READ-ONLY
    @property
    @abstractmethod
    def time(self) -> int: ...
    @abstractmethod
    def metrics(self) -> dict[str, float]: ...      # flat name->float, for Objective + logging
    # NOTE: NO apply_policy_change here (today fused into LinearStochasticSystem.step()). Acting = ActionInterface.

class RollableSystem(System):                       # capability split (critique fix): clone is NOT universal
    @abstractmethod
    def clone(self) -> "RollableSystem": ...        # deepcopy + Generator carried → sound rollout (doc-08)
```

```python
# govsim/core/action.py — THE keystone seam. Generalizes doc-06 Control + doc-07 Effect.
@dataclass(frozen=True)
class ActionRequest:
    regent_id: str
    verb: str                                       # lever name OR economy verb ("set_lockdown" | "Transfer")
    payload: dict                                   # {"expr": "..."} | {"value": x} | {"effect": {...}}  (§3.3 on the union)
    meta: dict = field(default_factory=dict)

class ActionInterface(ABC):
    """Per-domain vocabulary of what a controller MAY do + the domain invariants.
       The ONLY domain-coupled seam. economy-vs-epidemic-vs-company differ HERE and nowhere above."""
    @abstractmethod
    def action_space(self, system: System, regent_id: str) -> "ActionSpace": ...
        # machine- & prompt-renderable: verbs, ranges, types, capabilities, jurisdiction-scoped. .as_tools() for tool-calling.
    @abstractmethod
    def validate(self, req: ActionRequest, system: System, regent_id: str) -> "Result": ...
        # structured reject-WITH-FEEDBACK (doc-08 dead-end fix: today a bad expr silently returns None).
    @abstractmethod
    def apply(self, reqs: list[ActionRequest], system: System) -> "ApplyReport": ...
        # mutate system through domain invariants; returns {applied, rejected:[(req,reason)], costs}.
```

```python
# govsim/core/regent.py — N regents; N=1 degenerate. Domain-blind.
class Regent(ABC):
    id: str
    @abstractmethod
    def decide(self, view: "Observation", space: "ActionSpace", scratch: "Scratch") -> list[ActionRequest]: ...

class LLMRegent(Regent):
    def __init__(self, id, llm: "LLMClient", prompt_assembler): ...
    def decide(self, view, space, scratch):
        msgs = self.prompt_assembler(view, space, scratch)      # 4-source assembly (doc-06 §2.3)
        out  = self.llm.complete(msgs, tools=space.as_tools())  # native tool-calling OR structured JSON
        return parse_action_requests(out, space)                # validated against the space
# Baselines (same decide signature): StaticRegent, RandomRegent, TestRegent, PIDRegent, LQRRegent, OPRORegent.
# NO regent imports LinearSystemAgentContext (kills the gov_agent_linear.py:232 dead-on-arrival coupling).
```

```python
# govsim/core/harness.py — the scaffolding being STUDIED (Goal B). Decorator around a Regent.
class HarnessComponent(ABC):                         # all hooks default to no-ops; override only what you need
    name: str; enabled: bool = True
    def on_observe(self, obs, space, scratch): ...    # inject memory/trace/critic notes/inbox
    def propose_hook(self, regent, obs, space, scratch, base): return base(obs, space, scratch)  # wrap (rollout/critic/evolve)
    def on_outcome(self, obs, reqs, outcome, scratch): ...   # reflect, write memory, post to bus

class Harness:
    components: list[HarnessComponent]
    def act(self, regent, system, action_iface, objective, scratch) -> list[ActionRequest]: ...
    def _on(self): return [c for c in self.components if c.enabled]   # ← THE ablation switch (H3 = config sweep)
```

```python
# govsim/core/objective.py — pluggable, per-system, PER-REGENT. "Which objective" is an experimental variable.
class Objective(ABC):
    @abstractmethod
    def evaluate(self, trajectory, regent_id: str) -> float: ...
    @abstractmethod
    def components(self, trajectory, regent_id: str) -> dict[str, float]: ...   # Gini/deciles/MSE always logged
# StabilizationLoss(MSE+λ·MSU, generalizes the cubic loss & LQR cost), UtilitarianSWF, AtkinsonSWF/iso-elastic,
# GrowthObjective, welfare-λ·Gini, FiscalSustainabilityConstraint, EpidemicLoss(infections+λ·cost),
# CompanyProfit/ServiceLevel, RobustWrapper(mean-λ·std over seeds/shocks), MultiObjective/Pareto.
# Objectives are DOMAIN-BOUND and live in the same plugin layer as the ActionInterface, NOT in core/ (critique fix, §10).
```

```python
# govsim/core/experiment.py — the WHAT-first gated experiment + the data spine.
@dataclass
class Experiment:
    system_factory: Callable[[int], System]
    action_interface: ActionInterface
    regents: dict[str, Regent]                       # {regent_id: Regent}; N=1 → one entry
    jurisdictions: dict[str, "JurisdictionSpec"]     # which verbs/accounts/observables each regent governs
    harness: Harness
    objectives: dict[str, Objective]                 # per regent_id
    schedule: "Schedule"                             # decision steps (fixes simulation.py off-by-one + %50)
    seeds: list[int]
    # ---- WHAT-FIRST GATES (Runner refuses to run if any is None) ----
    hypothesis: "Hypothesis"                         # claim + NAMED baseline + primary metric (H1..H8)
    creativity_metric: "CreativityMetric | None"     # generalization-gap / functional-novelty / PIS

class Runner:
    def run(self, exp: Experiment) -> list["RunRecord"]:
        assert exp.hypothesis and exp.hypothesis.baseline, "WHAT-first gate (doc-08 §3.1)"
        # paired shared-seed loop; CachingReplayClient makes world-seed the only varying source.
```

### 2.3 The `LLMClient` seam + cache/replay (doc-08, day one)

`gemini_utils.py` (LangChain + `langchain_google_genai` + `google.generativeai`, raises at import on
missing key `gemini_utils.py:39`) is **deleted entirely**. Replacement is a thin OpenAI-compatible
Protocol with **lazy** key (fail on first call, not import — unblocks CI without a key, doc-08 §4):

```python
# govsim/core/llm/client.py
class LLMClient(Protocol):                                   # any OpenAI-compatible endpoint
    def complete(self, messages, *, tools=None, response_format=None,
                 temperature, seed, model) -> "LLMResponse": ...

class OpenAICompatClient:                                    # openai.OpenAI(base_url=...); native tools + structured outputs
    def __init__(self, base_url, api_key_env): ...           # lazy: no raise at import

class CachingReplayClient(LLMClient):                        # WRAPS any LLMClient
    key = sha256(json(messages) + model + f"{temperature}" + f"{seed}" + json(tools) + SCHEMA_VERSION)
    # modes: live (call+persist) | cache (call iff miss) | replay (cache-only, never call → reproducible & free)
```

The cache key includes `seed` + `tools` + a `SCHEMA_VERSION` (so a prompt/tool-schema edit invalidates
stale replays — critique "replay drift" fix). Every call's raw request/response/model-snapshot/token-
usage/cost lands in the `RunRecord`. This is the single cheapest load-bearing fix and shapes the data
model, so it is built first.

### 2.4 Where `RunRecord` / `ResultStore` and `polity_id`/`regent_id` live

`RunRecord` is the data model that shapes everything (doc-08 §3.2):
`(system_id, action_iface_id, {regent_id → (prompt_file, model, temperature)}, objective_ids, seed,
git_commit, hypothesis_id, schedule_id)` + per-step metric series as parquet artifacts + raw LLM I/O +
token/cost + model snapshot.

`ResultStore` is a `runs` table (sqlite + parquet side-files) keyed by exactly those columns, with **one
generic schema-driven plotter**. Rows carry `regent_id`/`polity_id` from day one (N=1 fills `"regent:0"`/
`"polity:0"`). Threaded now (cheap; a rewrite later):

- `System.observe(viewer_id)` → jurisdiction-scoped view.
- `ActionRequest.regent_id` → attribution; `ActionInterface.action_space(system, regent_id)` → per-regent
  verbs.
- `Objective` per `regent_id`; `Schedule` can interleave regents.
- Economy plugin (Phase 4+): ledger accounts namespaced **inside the plugin** as `"treasury@polity:0"`
  (doc-08 hook). **`polity_id`/ledger naming never leaks above the plugin** — the general seam carries only
  `viewer_id`/`regent_id`/`JurisdictionSpec` (critique LEAK-3 fix, §10).

### 2.5 Module / package layout (Phase 0 target)

```
govsim/
  core/
    system.py            # System, RollableSystem, Observation, StepInfo
    action.py            # ActionInterface, ActionRequest, ActionSpace, Result, ApplyReport
    regent.py            # Regent + StaticRegent/RandomRegent/TestRegent (PID/LQR/OPRO arrive Phase 1)
    harness.py           # Harness, HarnessComponent (no components shipped in Phase 0)
    objective.py         # Objective ABC (concrete objectives live in domains/*)
    experiment.py        # Experiment, RunRecord, Hypothesis, CreativityMetric
    runner.py            # Runner (paired-seed loop; extracted from simulation.py)
    schedule.py          # Schedule (fixes simulation.py:94 %50 + :101 off-by-one)
    result_store.py      # runs table (sqlite/parquet) + schema-driven plotter
    llm/
      client.py          # LLMClient, OpenAICompatClient (replaces gemini_utils.py)
      cache.py           # CachingReplayClient
    sandbox.py           # re-exports policy_utils (KEEP RestrictedPython); the lever/effect expr validator
  domains/
    scalar/
      interface.py       # ScalarLeverInterface (NO ledger)
      systems.py         # CubicSystem (migrated linear), CoupledSystem (migrated), SIRSystem, CompanySystem
      objectives.py      # StabilizationLoss, EpidemicLoss, CompanyProfit
    economy/             # Phase 4: doc-07 lives ENTIRELY here
      interface.py       # EconomyActionInterface (owns Effect/Ledger/Mediator/CodeOfLaws)
      worldspec.py       # doc-06 WorldSpec/Observable/Control (economy System base)
      sfc_toy.py         # tiny conserved-by-construction SFC economy (hh+firm+gov+bank, 1 good)
      objectives.py      # UtilitarianSWF, AtkinsonSWF, welfare-λ·Gini, FiscalSustainability
  harness/
    components.py        # TraceFeedback, EpisodicMemory, RolloutProbe, Critic, Reflection, Evolution, InterRegentComms
  regents/
    llm_regent.py        # LLMRegent + prompt_assembler
    baselines.py         # PIDRegent, LQRRegent, OPRORegent
  docs_gates/            # the WHAT-first spec docs (versioned, reviewed, BLOCK phases)
    hypotheses.md objectives.md creativity-metric.md stats-protocol.md decisions.md
```

`prompts/` and `policy_utils.py` are KEPT and re-homed; `chart_generators/special_visualize_exp4.py`,
`visualize_universal_legacy.py`, `combined_code.txt`, `scripts/combine_scripts.py` are REMOVED (doc-08 §5).

---

## 3. Domain plugins

### 3.1 The scalar plugin — `ScalarLeverInterface` (cubic, SIR, company; NO ledger)

This is literally today's linear-world apply path, generalized and pulled **out** of the `System`
(`linear_stochastic_system.py:199-235` currently fuses eval + advance). ~40 lines:

```python
# govsim/domains/scalar/interface.py
@dataclass(frozen=True)
class Lever: name: str; range: tuple[float, float]; attr: str

class ScalarLeverInterface(ActionInterface):
    def __init__(self, levers: list[Lever]): self.levers = {l.name: l for l in levers}
    def action_space(self, system, regent_id):
        return ActionSpace(verbs=list(self.levers), ranges={n: l.range for n, l in self.levers.items()},
                           context_vars=list(system.observe(regent_id).vars))
    def validate(self, req, system, regent_id):
        ctx = system.observe(regent_id).vars
        return compile_expr(req.payload["expr"], allowed=ctx)   # policy_utils → Result(ok|reject+feedback)
    def apply(self, reqs, system):
        for r in reqs:
            v = eval_safe(self._compiled[r], system.observe(r.regent_id).vars)   # re-eval per step via System hook
            lv = self.levers[r.verb]; system.set_lever(lv.attr, clip(v, lv.range))   # NO ledger
        return ApplyReport(...)
```

**The apply/step contract (critique-flagged, must be right).** Today `SingleMarketModel.apply_policy_change`
eval's once at apply time while every other world re-evals per step (doc-08 §3.2). Resolution:
`ActionInterface.apply` **installs/updates** the active action; `System.step()` calls a per-step
`re_eval()` hook into the installed action. Get this wrong and rollout fitness is biased. A test asserts
the eval-cadence contract (re-eval per step everywhere).

### 3.2 The economy plugin — `EconomyActionInterface` (doc-07 lives ENTIRELY here; Phase 4)

Doc-07 is **repositioned**: it is no longer "the governance interface of the framework"; it is *one
sibling of `ScalarLeverInterface`*. The Chancery (Effect algebra, private double-entry `Ledger` with the
conserved `_post`, `Mediator` with cost/rate-limit/noise, `CodeOfLaws`/`Institutions`) and doc-04's
`M_total` assertion all live **inside** `EconomyActionInterface`. doc-06's `WorldSpec` is repositioned one
layer lower still — it is how an *economy* `System` exposes `Observable`s (read side of `EconomyView`) and
`Control`s so `EconomyActionInterface` can wire them. The chain is:

```
economy System (doc-06 WorldSpec)  →  EconomyActionInterface (doc-07)  →  generic Regent (sees only ActionSpace + Observation)
```

doc-07's **Wall 1** ("the regent holds no balance write handle") becomes a *consequence of the top-level
layering*: the seam already denies the Regent any reference to `System` internals — it holds an
`Observation` (read-only) and an `ActionSpace`. Conservation thus moves from a special property of the
governance kernel to a structural property of the architecture (system mutation happens **only** inside
`ActionInterface.apply`).

```python
# govsim/domains/economy/interface.py  (Phase 4; sibling of ScalarLeverInterface)
class EconomyActionInterface(ActionInterface):
    def __init__(self, gov_spec):                       # doc-07 GovernanceSpec (capabilities, cost, noise)
        self.ledger = Ledger(...); self.mediator = Mediator(...); self.col = CodeOfLaws()
    def action_space(self, system, rid):
        # exposes the enabled Effect verbs {Transfer, Allocate, Mint, Burn, SetRate, Define, Repeal},
        # accounts/goods/levers scoped to rid's jurisdiction
    def validate(self, req, system, rid):
        # compile institution body via policy_utils; check capability subset; Walls 1/2/3 of doc-07; reject+feedback
    def apply(self, reqs, system):
        effects = self.col.fire(system.time, EconomyView(system, self.ledger))
        self.mediator.commit(effects)                  # cost/rate-limit/noise → Ledger._post (conserved)
        # doc-04 M_total assertion runs HERE; the regent never holds a balance write handle.
```

**The boundary-leak that MUST be resolved on paper before code (critique DEFECT-1).**
`ScalarLeverInterface`'s `ActionRequest.payload` is `{"expr": ...}`/`{"value": ...}`. An economy
`Transfer(src, dst, amount, good)` does **not** fit that scalar shape; it needs `payload = {"effect":
{...}}`. We accept the `payload` **union arm `{"expr" | "value" | "effect"}`** as the *one* place the core
action type is domain-aware — and we keep it honest: the union is a tagged variant whose `"effect"` arm is
opaque to the core (the core never interprets it; only `EconomyActionInterface` does). The shape is fixed
**now, on paper** (this section), so the "drop-in sibling" claim is proven before Phase 4, not after. If
in Phase 4 the union proves leaky in practice, the fallback is an `ActionEnvelope` per interface (the core
carries an opaque `bytes`/`dict` the interface (de)serializes) — but the union is the default.

**Staging inside the plugin (doc-08 §5).** `EconomyActionInterface` ships at Phase 4 start as
**`SetRate`-only + the conservation assertion + reject-with-feedback** (recovers today's single-expression
regent as the degenerate one-lever case). `Transfer`/`Allocate`/`Mint`/`Burn` + the full `Ledger` arrive
only when the conserved SFC `System` lands. `Mediator` noise/lag/leakage, the declarative DSL, the
versioned Code-of-Laws store, and the `Credit` primitive are deferred until an economy experiment **names**
them.

### 3.3 The non-economy proof — `SIRSystem` (zero ledger, same core)

SIR epidemic with intervention levers, slotting into the SAME core via `ScalarLeverInterface`:

```python
# govsim/domains/scalar/systems.py
class SIRSystem(RollableSystem):
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.S, self.I, self.R = 0.99, 0.01, 0.0
        self.beta0, self.gamma = 0.35, 0.10
        self.lockdown = 0.0; self.vacc = 0.0; self._t = 0; self.cum_cost = 0.0
    def step(self):
        self._reeval_actions()                                  # ActionInterface re-eval hook
        beta = self.beta0 * (1 - self.lockdown)
        newI = beta * self.S * self.I
        self.S += -newI - self.vacc * self.S * 0.02
        self.I += newI - self.gamma * self.I
        self.R += self.gamma * self.I + self.vacc * self.S * 0.02
        if self._t == 120: self.beta0 *= 1.8                    # H1 unseen structural shock: a variant
        self.cum_cost += self.lockdown * 1.0 + self.vacc * 0.5
        self._t += 1
        return StepInfo(terminated=self.I < 1e-4, truncated=False, info={})
    def observe(self, viewer_id="regent:0"):
        return Observation(vars={"S":self.S,"I":self.I,"R":self.R,
                                 "lockdown":self.lockdown,"vacc":self.vacc,"t":self._t}, scope=viewer_id)
    def set_lever(self, attr, v): setattr(self, attr, v)
    def metrics(self): return {"infected": self.I, "cum_cost": self.cum_cost}
    def clone(self): return copy.deepcopy(self)                 # Generator carried → sound rollout
    @property
    def time(self): return self._t

sir_iface = ScalarLeverInterface([Lever("set_lockdown",(0.0,0.9),"lockdown"),
                                  Lever("set_vaccination",(0.0,0.5),"vacc")])
```

The **same** `LLMRegent`, `RolloutProbe` harness, `Runner`, `ResultStore`, `CachingReplayClient` run
unchanged. The regent emits `{verb:"set_lockdown", payload:{"expr":"0.7 if I > 0.1 else 0.2"}}` — a
sandboxed expression, no `Mint`/`Transfer`/`Ledger` anywhere. The `t==120` β-shock is exactly H1's "unseen
structural shock," the same scientific spine as the cubic, in a non-economic domain. A `CompanySystem`
(levers `set_price`/`set_production`/`set_reorder_point`; `Objective = CompanyProfit`) is analogous and
also ledger-free — a *dynamics* model, not the economy plugin. This is the proof the abstraction is not
secretly economy-only: the ledger is confined to **one** of several sibling interfaces, and two whole
domains run on the trivial sibling.

### 3.4 Rule-level levers (vision-aligned, bounded)

Where "creativity must come from richer interventions" (doc-08 §3.5), worlds may declare **bounded,
`SetRate`-able coefficients of their own agents' reaction functions** as first-class levers (not a
forbidden case). These are still `ActionInterface` verbs with ranges; they do **not** grant arbitrary
rule-rewriting (out of scope, §1.4). The rule-override line (which coefficients are levers vs frozen) is an
explicit per-world declaration (doc-07 open question (a)).

---

## 4. Multi-regent (first-class; N=1 trivial)

Multi-regent is a property of the **loop and three small records**, threaded now even while N=1, because
widening the single `decide_policy(...)`/`apply_policy_change(policy)` path later (simulation.py:108/135)
touches every world.

### 4.1 The records (domain-neutral; live in `core/`)

```python
@dataclass(frozen=True)
class JurisdictionSpec:
    verbs: frozenset[str]                       # subset of action_space verbs this regent may emit ("*" = all)
    observable_scope: frozenset[str] | None = None   # None = full view; else a mask (information asymmetry)
# Jurisdiction = a PARTITION of the world's ALREADY-ADVERTISED action_space (grafted best idea, §10).
# Worlds need ZERO changes to host N regents; contention exists ONLY on a declared-shared verb subset.
```

### 4.2 The interaction protocol (the `Runner`'s decision step; replaces simulation.py:99-138)

```
order = schedule.order(regents, step)               # SIMULTANEOUS | SEQUENTIAL | STACKELBERG(leader, followers)
for regent in protocol.activations(order):
    view   = system.observe(regent.id)              # jurisdiction-scoped (info asymmetry for competition)
    space  = action_iface.action_space(system, regent.id)
    reqs   = harness.act(regent, system, action_iface, objectives[regent.id], scratch[regent.id])
    bus.publish(regent.id, reqs)                     # InterRegentComms (empty/no-op at N=1)
report = action_iface.apply(all_reqs, system)       # SINGLE atomic mutation through domain invariants
# per-regent feedback (applied/rejected + reason) → that regent's next decide (uses the dead llm_extra_context channel)
system.step()
```

The **single `apply([all reqs], system)`** is the one invariant checkpoint: conservation/solvency (economy)
or lever-clip (scalar) is enforced over the *merged* batch, so guarantees hold regardless of N. Contested
verbs (two regents touch a shared lever/treasury) are resolved inside `apply` by a domain policy
(identity at N=1; LAST_WRITER/PRIORITY/VETO/AVERAGE for scalar; ledger-merge for economy).

### 4.3 N=1 as the trivial case + the experiment shape for H7

- N=1: `regents={"regent:0": …}`, one full-jurisdiction mandate, `SIMULTANEOUS`, empty bus — reduces to
  today's path. A **golden-master test** asserts N=1 is behaviorally identical to the pre-rewrite run.
- Coordination/competition (H7): the `TurnProtocol` is the experiment surface — `mode`
  (simultaneous/sequential/Stackelberg), `observe_others`, `communication` (NONE/BROADCAST/PAIRWISE),
  `agreements` (NONE/NONBINDING/BINDING), `rounds`. **Collusion is measured as the multi-regent Goodhart
  probe** (grafted, §10): joint per-regent objectives rise while a held-out social objective falls. Built
  in Phase 5 — hooks now, game logic later.

What is **threaded now** (cheap, N=1 bodies ignore it): `regent_id` everywhere, `observe(viewer_id)`,
per-regent `Objective`, `JurisdictionSpec`, `Schedule` as a first-class object, `ResultStore` `regent_id`
column, an empty `InterRegentComms` bus. What is **NOT built now**: any N>1 game logic, collusion
detection, leaderboard, BINDING agreements — Phase 5, gated on a registered H7 with a baseline. A single
N=2 smoke test in Phase 5 exercises two real jurisdictions so the threaded parameters are not dead vapor.

---

## 5. Harness-as-research (Goal B)

### 5.1 The Regent/Harness split

`Regent` is a thin `decide(view, space, scratch) → [ActionRequest]`. `Harness` is an ordered stack of
independently-toggleable `HarnessComponent`s decorating a fixed `on_observe → propose → on_outcome`
lifecycle. Components communicate **only** through the additive `scratch` dict and the `Outcome` — never by
importing each other. That decoupling is what makes leave-one-out attribution *valid* (removing X cannot
silently disable Y). The single `_on() = [c for c in components if c.enabled]` list-comprehension is the
**entire ablation apparatus** (grafted keystone, §10).

### 5.2 The pluggable + ablatable components

| Component | Hooks | What it does | Rollout-dependent? |
|---|---|---|---|
| `TraceFeedback` | on_observe | injects last `Outcome.error` (sandbox compile/validate msg, conservation-reject, runtime) as a corrective message — the cheapest upgrade (doc-08 §6) | no |
| `EpisodicMemory` (RAG) | on_observe / on_outcome | retrieves k past (state-summary, action, objective) episodes by current metrics; the cheap learning baseline (doc-08 §6) | no |
| `RolloutProbe` | propose_hook | regent emits N candidates; score each via `system.clone().rollout(cand, H, seeds)` → `Objective`; keep `argmax(mean − λ·std)` (H2, variance-aware) | **YES** |
| `Critic` | propose_hook | a 2nd LLM call audits proposal vs objective/constraints; veto/annotate → revise once | no |
| `Reflection` | on_outcome | summarizes realized-vs-predicted gap into a NL lesson for next on_observe | (rollout if scoring) |
| `Evolution` | propose_hook + run-level | population across decisions; LLM-mutate, select on `Objective` via rollout; MAP-Elites cells = creativity descriptors | **YES** |
| `InterRegentComms` | on_observe / on_outcome | message bus keyed by `regent_id` (H7); empty no-op at N=1 | no |

**Rollout-soundness gate (critique fix).** All rollout-dependent components are gated behind **one
precondition test: the `System` is a `RollableSystem` that owns a `np.random.Generator`** (doc-08 HARD;
today every world draws module-global `random.gauss`, e.g. `linear_stochastic_system.py:239`). Until that
holds, `RolloutProbe`/`Evolution`/rollout-`Reflection` produce biased fitness, so the **rollout-free
components ship first** (`TraceFeedback`, `EpisodicMemory`), inverting the naive "rollout is highest value
so build it first" order.

### 5.3 The effectiveness metric (the WHAT-gate, defined before components)

"Regent effectiveness" is a tuple, primary first:

1. **PRIMARY — adaptation regret (H1):** post-shock cumulative `Objective` gap vs (a) the frozen
   pre-shock-optimal controller (LQR/numeric-DP on the cubic) and (b) a trace-less baseline. This is where
   the LLM can actually win; on a stationary linear toy it is ≈0 by design.
2. **SECONDARY — generalization gap (creativity primary):** effectiveness on un-tuned regimes minus a
   tuned-PID's effectiveness on the same. Positive ⇒ transferable skill, not overfit.
3. **GUARDS (separate axes, never folded into Objective — doc-08 §3.4 category-error):** sample/compute
   cost; worst-of-N robustness (`mean − λ·std`); held-out true-objective vs fitness-proxy divergence
   (Goodhart probe).

### 5.4 The ablation design (H3)

Pre-registered, paired, replay-cached. The `enabled` flag + `CachingReplayClient` (replay mode) is the
whole apparatus — paired shared world-seeds, LLM responses served from cache so the **only** varying source
is the world seed.

| Phase | Design | Output |
|---|---|---|
| P0 | base regent, all components off, cubic shock-regime | reference regret `R0` |
| P1 single-add | enable exactly ONE component (one run each) | additive gain `g_i = R0 − R_i` |
| P2 leave-one-out | full stack minus ONE | LOO gain `g_i' = R_full − R_{−i}` (catches interactions) |
| P3 ordered | greedily add by P1 gain; report the curve | the "separable measurable gain" claim of H3 |
| P4 Shapley (opt) | sample subset masks; Shapley value per component | interaction-robust attribution |

A component "yields a separable gain" iff its paired bootstrap CI excludes 0 in **both** P1 and P2. A
component that can't show that gets **cut** — the ablation harness is also the over-engineering filter.

---

## 6. Where the science plugs in

### 6.1 Hypotheses (the swappable spine; each is `claim + named baseline + falsification test`)

| ID | Claim | Named baseline | First rung |
|---|---|---|---|
| **H1** | adaptation-to-the-unknown: recover from an unseen structural shock better than the pre-shock-optimal controller AND a trace-less baseline | frozen LQR/numeric-DP + trace-less OPRO | rung 1 (cubic) |
| **H2** | experimentation > reasoning: probing via rollouts beats pure reasoning | reasoning-only regent (no `RolloutProbe`) | rung 1–2 |
| **H3** | harness-ablation: each scaffolding component yields a separable measurable gain | full stack minus component | rung 1–2 |
| **H4** | objective → institution mapping: same world, different objective ⇒ different discovered policy | fixed-objective regent | rung 2 (SFC) |
| **H5** | specification-gaming / Goodhart: a characterized proxy-up/true-welfare-down episode | held-out true welfare | rung 2 |
| **H6** | Lucas: governance results change under naive → policy-aware adaptive agents | naive-agent run | rung 2–3 |
| **H7** | multi-regent collusion/competition (collusion = joint-obj↑ while held-out social-obj↓) | independent (no-comms) regents | rung 5 |
| **H8** | planning-horizon → richer economy | short-horizon regent | rung 3 |

### 6.2 Objectives (pluggable; "which objective" is itself an experimental variable)

growth/GDP · utilitarian SWF · inequality-weighted (Atkinson/iso-elastic, or welfare−λ·Gini) ·
stabilization loss (dual-mandate, generalizes MSE/MSU and the LQR cost) · distributional (Gini) as
secondary/constraint · fiscal-sustainability constraint · multi-objective/Pareto · `RobustWrapper`
(mean−λ·std over seeds/shocks). Gini/deciles are first-class `Objective.components` everywhere. Objectives
are **domain-bound plugins** (in `domains/*/objectives.py`), not `core/` residents (critique fix, §10).

### 6.3 Creativity metrics (domain-scoped; engineer interpretability or drop the word)

- **Control domains (primary):** generalization gap (beats a tuned PID *specifically* on un-tuned regimes)
  + functional novelty (residual of the best-fit simple controller; does it use conditionals / state-
  history / regime detection a PID structurally cannot?).
- **Governance domain (adds):** Policy-Innovation-Score (semantic distance from a known-policy library) +
  institution-type novelty.
- **Cross-cutting (optional):** MAP-Elites QD diversity.
- **Discipline:** there is **no single creativity number across all domains** (§1.4). Where none separates
  the LLM from a tuned controller, drop "creativity" and report adaptive in-context control. A parsimony /
  restricted-grammar term keeps emitted laws human-readable (else evolution yields `np.clip`/`where`/`tanh`
  spaghetti and the interpretability pitch fails).

### 6.4 The system ladder + the first publishable result

- **Rung 1 — scalar/low-dim, incl. the cubic partial-information system. BEGIN HERE.** LQR/numeric-DP
  ground truth; cheapest; where the novel adaptation result lives. Keep the cubic
  (`linear_stochastic_system.py:242`) + obfuscated prompt (both currently `M` in git) as the H1 arm.
- **Rung 1.5 — a non-economic system on the SAME core** (SIR; then company) — proves generality is
  *exercised*, not asserted. A CI **leakage test** (zero domain nouns in `govsim/core/`) enforces it.
- **Rung 2 — a tiny conserved-by-construction SFC economy** (hh+firm+gov+bank, 1 good). Where
  `EconomyActionInterface` (doc-07 kernel) lands. H4/H5/H6.
- **Rung 3 — an adopted validated adaptive-agent ABM** (default: EconAgent — public code, LLM-adaptive
  agents, reproduces Phillips+Okun, closes the Lucas gap). Optional/gated; Mandel is a kill-criterion
  stretch track.
- **Generality proof beyond economics:** SIR (rung 1.5) + company.

> **★ FIRST PUBLISHABLE RESULT (mark): rung 1, H1.** *A code-as-policy LLM regent with the trace
> side-channel recovers from an UNSEEN structural shock (the cubic / coupled regime change, partial
> information) in fewer interventions / lower post-shock regret than (a) the frozen pre-shock-optimal
> controller and (b) a trace-less OPRO baseline* — reachable **before any economy code exists** (doc-08
> recommended spine). The linear arm is scoped to **"matches OPRO + LQR"** (sanity check); the advantage
> claim lives only in the non-stationary / nonlinear / partial-info regime (doc-08 §3.5: "beat OPRO on the
> linear toy" is a likely null).

---

## 7. The grand implementation plan

Coding-agent-assisted solo timeline. The discipline (critique-driven): **Phase 0 is a THIN vertical slice
+ the data spine, not a six-ABC cathedral.** The ABCs crystallize from cubic → SIR; the Harness taxonomy
and `EconomyActionInterface` are deferred behind a result.

### 7.1 Phase 0 — the big rewrite (thin core + data spine). ~3–5 weeks.

**DELETE:** all of `gemini_utils.py` + `google-generativeai` + `langchain` + `langchain-google-genai`
(pyproject.toml:16-18); `mesa` (pyproject.toml:21, zero imports); `chart_generators/special_visualize_exp4.py`,
`visualize_universal_legacy.py`; `combined_code.txt`, `scripts/combine_scripts.py`; the
`LinearSystemAgentContext` Pydantic coupling (`gov_agent_linear.py:12, 232`).

**BUILD (the vertical slice):** `core/system.py` (`System` + `RollableSystem`), `core/action.py`
(`ActionInterface` + `ActionRequest` + reject-with-feedback `Result`), `core/regent.py`
(`Regent` + `TestRegent`), `core/llm/{client,cache}.py` (`OpenAICompatClient` + `CachingReplayClient`,
sha256 key incl. seed/tools/`SCHEMA_VERSION`, lazy key), `core/schedule.py` (fixes simulation.py:94 %50 +
:101 off-by-one), `core/result_store.py` (`runs` table + schema-driven plotter), `core/runner.py`
(paired-seed loop + WHAT-first asserts). Migrate the **cubic** to `CubicSystem(RollableSystem)` +
`ScalarLeverInterface`. Per-system `np.random.Generator` (ban bare `random.*`/`np.random.*` via AST test).

**WHAT-decision gate docs (BLOCK the phase if absent):** `docs_gates/{hypotheses,objectives,creativity-metric,
stats-protocol}.md` + `decisions.md` (ADR) + a STATUS table; `git add` the whole `agents/` corpus.

**TESTS + CI:** sandbox (RestrictedPython) tests; system determinism (same seed → same trajectory);
clone/rollout faithfulness; the **no-bare-RNG** AST test; the **no-domain-noun-in-core** leakage test; cache
hit == live; replay reproduces; **N=1 golden-master via `TestRegent`** (fixed expr `-0.9*current_x`, NO live
LLM) reproducing the pre-rewrite cubic run; the apply/step **re-eval-cadence** contract test. CI runs all of
it without an API key.

**GATE:** green deterministic CI; `python -m govsim run <exp>` works; golden-master matches.

### 7.2 Test-driven feature-growth phases

| Phase | Build (components added) | Hypotheses | Acceptance gate / tests | Publishable result |
|---|---|---|---|---|
| **1 — rung 1 scalar/cubic** | `LLMRegent` (decoupled, OpenAI tool-calling/structured) + 4-source prompt assembler + boot `check_prompt`; `PIDRegent`/`LQRRegent`/`OPRORegent`; `StabilizationLoss`; migrate `CoupledSystem` | H1, H2, H3 | paired bootstrap CI on per-seed diff; collapse detector; variance-aware select (`mean−λ·std`); leakage test green | core runs end-to-end on rung 1; **"matches OPRO+LQR"** sanity |
| **2 — harness + H1 result** | `TraceFeedback` + `EpisodicMemory` (rollout-FREE, ship first); then `RolloutProbe` (gated on `RollableSystem`); replay-mode ablation | H1 (falsifiable), H2, H3 | rollout-soundness precondition test passes before `RolloutProbe`; H3 separable-gain (P1∧P2 CI excludes 0) | **★ FIRST PAPER: code-as-policy + trace adapts to an unseen shock better than frozen-optimal + trace-less OPRO** |
| **3 — rung 1.5 non-economy** | `SIRSystem` + `CompanySystem` (reuse `ScalarLeverInterface`); `EpisodicRAG` reused byte-for-byte | H1 (transfer) | the EXACT rung-1 harness/regent/objective run unchanged except System+levers; H1 reproduces on SIR; **leakage test = 0 domain nouns in core/** | generality demonstrated, not asserted ("the machine is domain-general") |
| **4 — rung 2 SFC economy** | tiny conserved SFC `System` (doc-06 `WorldSpec`); `EconomyActionInterface` (doc-07 kernel: `SetRate`-only → +`Transfer`/`Allocate`/`Mint`/`Burn` + `Ledger` + conservation assertion + reject-feedback); a **structurally-different action algebra** exercised here | H4, H5, H6 | per-step `M_total` conservation assertion green; objective-as-variable sweep; a characterized gaming episode (proxy↑/true-welfare↓); naive→adaptive-agent falsification | 2nd paper: legible objective→institution mapping + a diagnosed Goodhart episode (no multi-agent needed) |
| **5 — multi-regent** | flip N=2; `JurisdictionSpec` partitions; per-regent `Objective`; `InterRegentComms` bus; `TurnProtocol` (simultaneous/sequential/Stackelberg) | H7, H8 | N=2 smoke test exercises two real jurisdictions; collusion-as-Goodhart metric; reproducible | 3rd paper / league seed: coordinate vs compete |
| **6 — (opt, gated, parallel)** | adopt EconAgent (closes Lucas; reproduces Phillips+Okun) as a `System`; Mandel only as a kill-criterion stretch | H6, H8 | adopted-model stylized-facts GO/NO-GO **before** any governance claim | richer-economy results on a validated ABM |
| **v2 — league** | submission interface + cross-world leaderboard (`polity_id` already threaded) | — | additive, no core rewrite | the durable benchmark framing |

### 7.3 Realistic milestones

Phase 0 ≈ 3–5 weeks; Phase 1 ≈ 2–3 weeks; Phase 2 (first paper result) ≈ **~3–4 months in total**; Phase 3
≈ 1–2 weeks (cheap — only a new System file); Phase 4 ≈ 4–6 weeks; Phase 5 ≈ 3–4 weeks. The first
publishable contribution arrives **before** any economy/Chancery code, honoring doc-08's critical path.

---

## 8. Risks & honest limits

| Risk | Concrete failure | Mitigation / what would make us simplify |
|---|---|---|
| **Over-abstraction (the doc-08 sin)** | shipping six ABCs + Harness taxonomy + `EconomyActionInterface` before one result = "the 2025 state with more scaffolding" | Phase 0 is a THIN vertical slice (cubic + `ScalarLeverInterface` + `TestRegent` + replay + one `runs` row); ABCs crystallize from cubic→SIR; Harness taxonomy & economy plugin gated on a result existing. **Simplify trigger:** if rung-1 H1 fails, stop and re-decide WHAT — do not build rung 2. |
| **Boundary leakage (economy climbs up)** | `payload` `\|effect-spec`, `polity_id`/ledger naming, conservation-as-core-invariant leak into core/ | the `payload` union is the ONE accepted domain-aware spot, fixed on paper now (§3.2); `polity_id`/ledger naming confined to the economy plugin; conservation is an `EconomyActionInterface` concern, not a core property; a **CI leakage test** fails if any domain noun appears in `govsim/core/`. |
| **Generality theater** | rung 1.5 reuses the SAME scalar algebra ⇒ proves only "scalar-control-general" | Phase 4 exercises a **structurally-different action algebra** (economy `Effect` verbs); the generality claim is earned by a different algebra leaving Regent/Harness/Objective/Runner untouched, not by a second scalar toy. |
| **Rollout unsoundness** | components lean on `clone()`/`rollout()` but worlds use module-global RNG ⇒ biased fitness | per-system `Generator` + `clone()` carrying it is a HARD Phase-0 item; `clone()` on `RollableSystem` only; rollout-dependent components gated behind a precondition test; rollout-FREE components ship first. |
| **Multi-regent complexity** | threaded `regent_id`/`viewer_id` rot into untested dead parameters; Arena becomes a doc-07-style cathedral | thread the parameters now but build NO N>1 game logic until Phase 5; one N=2 smoke test exercises two jurisdictions so the params are not vapor; Jurisdiction = partition of the already-advertised action_space (no new world API). |
| **Cost / compute** | population × seeds × horizon × paid calls makes ablation/rollout a budget question | `CachingReplayClient` (replay mode) makes ablation reruns free and paired; a budget axis is a separate guard (never folded into Objective); evaluation cascade (cheap worlds/short horizons first). |
| **Creativity stays a vibe** | the identity word never separates LLM from a tuned controller | creativity is a defined, domain-scoped dependent variable (§6.3) decided in a Phase-0 gate doc; if no construct separates, **drop the word** and report adaptive in-context control. |
| **Solo bandwidth** | Phase 0 sprawls into a cathedral | timebox Phase 0; accept a deferred economy plugin; the phase gate (green CI + golden-master + first H1) is the forcing function. |

**The brutally-minimal path to the first result needs NONE of:** `EconomyActionInterface`/`Ledger`/
`Mediator`/`CodeOfLaws`/DSL, the full Harness taxonomy, N>1 game logic, a stable Mandel, an RL regent, a
Gymnasium adapter, or the league. Build the seam, keep the cubic, thread `regent_id`, defer the cathedral.

---

## 9. Repositioning summary (06/07 → plugins)

- **doc-06 (`WorldSpec`/`Observable`/`Control`/`clone`/`rollout`)** → the declarative way an **economy
  `System`** exposes state+levers to `EconomyActionInterface`. Its `clone()`/`rollout()` keystone is
  generalized to the core `RollableSystem` capability and reused by rollout harnesses for ALL domains. Its
  honest "prompt is a 4-source contract + boot `check_prompt`" survives as the `LLMRegent` prompt assembler.
- **doc-07 (Chancery: Effect/Ledger/Mediator/CodeOfLaws/Institutions)** → lives **entirely inside**
  `EconomyActionInterface`, a Phase-4 sibling of `ScalarLeverInterface`, NOT the top-level action
  abstraction. Wall-1 conservation becomes a structural consequence of the layering. Ships as the kernel
  (`SetRate` + conservation assertion + reject-feedback) first; the rest deferred until an experiment names
  it (doc-08 §5).
- **doc-05 roadmap** → superseded by §7.

---

## 10. Where facet-proposals were overruled

- **Core-architecture proposal — `Objective` placement.** It placed `objective.py` in the domain-neutral
  core. **Overruled** (sequencing critique): welfare/Gini logic cannot be typed without importing domain
  observables, so Objectives are **domain-bound plugins** (`domains/*/objectives.py`); only the `Objective`
  *ABC* lives in core.
- **Core-architecture proposal — `clone()` on the base `System`.** **Overruled** (sequencing critique):
  "any complex system" is false at the type level if `clone()` is mandatory (a live API world can't clone).
  `clone()` moved to a `RollableSystem` capability; rollout harnesses require it, the base does not.
- **Core-architecture + sequencing proposals — Phase 0 scope.** Both front-loaded six ABCs + Harness
  taxonomy + full result infra before a result. **Overruled/right-sized** (both critiques): Phase 0 is a
  thin vertical slice; the ABCs and the three-class Harness taxonomy crystallize from cubic→SIR and are
  gated on a result, not the spine.
- **Core-architecture proposal — `EconomyActionInterface` "drop-in sibling" left unproven.** **Required
  fix adopted:** the `payload` union (`expr|value|effect`) and the economy `ActionRequest`/`ActionSpace`
  shape are pinned **on paper now** (§3.2), with `ActionEnvelope` as the documented fallback — the seam is
  proven before Phase 4, not after.
- **Multi-regent proposal — M0 builds twelve types + Arena/Resolver/Channel/Protocol quartet up front.**
  **Overruled** (critique): that is the doc-07-cathedral pattern. We thread `regent_id` + `JurisdictionSpec`
  + collect/apply + identity contention now; `InterRegentComms`/`TurnProtocol`/agreements are deferred
  behind a registered H7 with a baseline (Phase 5). **Grafted verbatim:** Jurisdiction = partition of the
  already-advertised action_space; collusion = the multi-regent Goodhart probe.
- **Multi-regent proposal — `Effect`/`budget`/conservation vocabulary in the neutral core.** **Overruled**
  (critique): the core carries `actions`/`ActionRequest` (not `Effect`s) and a generic atomic `apply`;
  `budget` and conservation are `EconomyActionInterface` concerns. `commit` is a batched atomic write with
  an optional domain validator, not a conservation checkpoint in core.
- **Harness proposal — rollout-dependent `RolloutProbe` shipped first as "cheapest, highest-value."**
  **Overruled** (critique): unsound on module-global RNG. Rollout-FREE `TraceFeedback`/`EpisodicMemory`
  ship first; all rollout components gated behind the `RollableSystem`/Generator precondition test.
- **Harness proposal — `economic_system`-named parameter + single-`Policy` return through the core.**
  **Overruled** (critique LEAK-1/LEAK-2): the core seam uses `system`/`world` naming and returns a
  `list[ActionRequest]` applied by the domain-routed `ActionInterface.apply`, never a single `Policy`
  through simulation.py:135. **Grafted keystone:** the `enabled`-flag + replay-tape as the entire H3
  ablation apparatus; components communicate only via `scratch`/`Outcome`.
- **All proposals — "control any complex system."** **Overruled/scoped** (every critique): scoped to
  `observe → (bounded levers | domain action algebra) → step` with `clone()` for rollout (§1.4), with an
  explicit OUT-of-scope list and a CI leakage test, rather than the vacuous universal claim.
