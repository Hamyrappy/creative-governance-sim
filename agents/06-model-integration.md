# Model Integration — the recommended design

> **Status (per [`08`](08-open-problems-and-opportunities.md) §5): SOUND, but NOT on the near-term critical
> path.** Build only the minimal seams now — decouple the regent, give each world a `np.random.Generator`,
> add one Objective, make policy re-eval per step; defer the full declarative `WorldSpec` / registry /
> package-rename until a research result justifies it. `08` adds two from-start hooks this doc lacks: a
> **local per-world RNG** (precondition for a *sound* `clone()`/`rollout()`) and a **`polity_id` dimension**.
> **Repositioned by [`09-grand-plan.md`](09-grand-plan.md):** this is now the *economy domain plugin* —
> `WorldSpec`/`Observable`/`Control` are how an **economy `System`** exposes state+levers to
> `EconomyActionInterface`, and `clone()`/`rollout()` is generalized to the core `RollableSystem`
> capability for all domains. Not the top-level framework.

Lead-architect synthesis of four integration-API proposals (Gymnasium-style env, declarative
base class, zero-boilerplate wrapper, Mesa-native). This is the design we build. It is grounded in
the real files: `govsim/utils/interfaces.py`, `govsim/economic_models/*.py`,
`govsim/governing_agents/gov_agent_linear.py`, `govsim/utils/policy_utils.py`,
`govsim/simulation.py`, `govsim/config.py`, and the unmodified `mandel testing/mandel_test.py`.

> Provenance: synthesized from a 4-architect design panel with adversarial critique (scores:
> declarative-base **7/10**, zero-boilerplate-wrapper **7/10**, Gymnasium-env **6.5/10**, Mesa-native
> **6/10**). Honest headline: **there is no silver bullet.** The policy/eval/whitelist contract *can*
> be made drift-proof from one declaration, but the **prompt cannot** be fully auto-derived (it pulls
> from four sources) — so "automatic" has a real, explicit boundary, drawn in §2.3. This doc
> supersedes the registration/integration/context specifics in [`03-refactor-plan.md`](03-refactor-plan.md).
> The **governance/action model** — how the regent actually acts (conserved *Effects*, not a scalar
> `Control` write) — is specified in [`07-governance-interface.md`](07-governance-interface.md), which
> generalizes the `Control` here into Effects emitted by Institutions on a double-entry ledger.

---

## 1. Verdict (3 sentences)

Build a thin **declarative descriptor layer on the existing `BaseEconomicSystem`**: a world declares
two lists — `observables` and `controls` (each an accessor-closure spec) — as the **single source of
truth**, and a `WorldSpec` base + a `@register` decorator auto-derive the sandbox whitelist, the
eval context, `get_policy_descriptors()`, `apply_policy_change()`, the per-step re-evaluation,
and a uniform agent context, killing the three-way drift and the per-world dispatch boilerplate.
Graft the one genuinely great idea from the Gymnasium proposal — **a numeric world plus an
`obs -> action` `Policy`, scored by `clone()` + `rollout()`** — because it implements the
long-stubbed `emulate_policy` for free and is the one primitive shared by the LLM, RL, and
evolutionary regents. **Drop Mesa entirely; keep RestrictedPython, `Policy`, and `PolicyDescriptor`;
and explicitly treat the prompt as a four-source contract** (observables + model params + agent KPIs
+ formatted text) assembled by the agent — *not* something that "falls out" of the observables, with
a build-time validator that fails loud if the prompt references a name the spec cannot supply.

The backbone is **Proposal 2/3/4's shared core** (declarative Control/Observable -> projections).
We reject the four proposals' common oversell — that the prompt auto-derives from observables — and
we reject Proposal 1's hand-rolled `Box`/`Dict` space system and `to_gymnasium()` adapter as YAGNI.

---

## 2. The single source of truth

A world declares controls and observables **once**, as two lists of accessor-closure specs. Every
downstream artifact is a *computed projection* of those two lists, so the surfaces that drift today
(`PolicyDescriptor.available_context_vars` whitelist vs `get_current_metrics()` keys vs the eval
context) become structurally identical — they are the same list read three times.

### 2.1 The spec types

```python
# govsim/world/spec.py
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type

def attr(path: str) -> Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]:
    """get/set accessors for a (possibly nested) attribute path, e.g. 'government.tax_rate'."""
    parts = path.split(".")
    def _get(root):
        o = root
        for p in parts: o = getattr(o, p)
        return o
    def _set(root, val):
        o = root
        for p in parts[:-1]: o = getattr(o, p)
        setattr(o, parts[-1], val)
    return _get, _set

def hist(list_name: str, index: int = -1, default: float = 0.0) -> Callable[[Any], float]:
    """Read a scalar from the tail of a history list, e.g. Mandel's history_gdp[-1]."""
    def _get(root):
        lst = getattr(root, list_name, None)
        return float(lst[index]) if lst else default
    return _get

@dataclass(frozen=True)
class Observable:
    name: str                              # whitelist id == metrics key == context key (ONE name)
    get: Callable[[Any], float]            # model -> value
    desc: str = ""

@dataclass(frozen=True)
class Control:
    name: str                              # == policy_type_id (e.g. "set_control_input")
    get: Callable[[Any], Any]              # read current lever value (for context/log)
    set: Callable[[Any, Any], None]        # write computed+clipped value into the model
    value_range: Optional[Tuple[float, float]] = None
    value_type: Type = float
    nullable: bool = False                 # SingleMarketModel price_ceiling/floor: "none" -> None
    desc: str = ""
    uses: Optional[list] = None            # observable names legal in THIS control's expr (default: all)
    constraints: dict = field(default_factory=dict)
```

Two things the four proposals got wrong that are fixed here:

- **`nullable`** is a first-class field. Without it, the generic base *cannot reproduce*
  `SingleMarketModel.price_ceiling`/`price_floor`, which are `Optional[float]` and accept the literal
  string `"none"`/`""` -> `None` (see `single_market_model.py:265-274, 303-308`). A float-only
  `Control(low, high)` is a regression on a world that already exists.
- **`uses`** is explicit, defaulting to `None` (= all observables). We do **not** default the
  in-expression name to the write-attribute. Proposal 2's `context_as = writes` default would
  silently rename `tax_rate` (the prompt/whitelist name in `SimpleGrowthModel`,
  `economic_models.py:40`) into `current_tax_rate` (the attribute, `:37`) — relocating the very drift
  the design exists to kill into an easy-to-forget default.

### 2.2 The base class: every projection derived from the two lists

```python
# govsim/world/base.py
import copy, random
from typing import Any, Dict, List, Optional
import numpy as np
from govsim.utils.interfaces import BaseEconomicSystem, Policy, PolicyDescriptor
from govsim.utils.policy_utils import evaluate_safe_policy_code

REGISTRY: Dict[str, type] = {}

class WorldSpec(BaseEconomicSystem):
    # Authors declare these (usually in __init__ after params, like the linear world builds u_range):
    controls: List[Control] = []
    observables: List[Observable] = []
    default_params: dict = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        REGISTRY[cls.__name__] = cls          # auto-registration (kills pain point 1)

    def __init__(self, params: Dict[str, Any]):
        self.current_step = 0
        self.state: Dict[str, Any] = {}
        self.active_policies: List[Policy] = []
        self.history: List[Dict[str, Any]] = []
        self._init_params = dict(params)
        self.rng = random.Random(params.get("seed"))    # local RNG (pain point 5)

    # -- the ONE place where (model, observable) -> value; feeds whitelist, context, prompt --
    def get_current_metrics(self) -> Dict[str, float]:
        m = {"step": float(self.current_step)}
        for o in self.observables:
            m[o.name] = float(o.get(self))
        for c in self.controls:                          # current lever values are observable too
            v = c.get(self)
            m[c.name] = float(v) if isinstance(v, (int, float)) else -1.0
        return m

    def _context_vars(self) -> List[str]:                # the sandbox whitelist (one projection)
        return sorted({"step", *(o.name for o in self.observables), *(c.name for c in self.controls)})

    def get_policy_descriptors(self) -> List[PolicyDescriptor]:
        all_vars = self._context_vars()
        return [PolicyDescriptor(
                    policy_type_id=c.name, description=c.desc,
                    value_type=c.value_type, value_range=c.value_range,
                    target_variable_name=c.name,
                    available_context_vars=(c.uses or all_vars),
                    constraints=c.constraints)
                for c in self.controls]

    def _control(self, name) -> Optional[Control]:
        return next((c for c in self.controls if c.name == name), None)

    # -- generic apply: replace-or-append (identical to every hand-written version today) --
    def apply_policy_change(self, policy: Optional[Policy]) -> None:
        if policy is None:
            return
        if self._control(policy.policy_type) is None:
            print(f"[WorldSpec] unknown policy_type '{policy.policy_type}' ignored")
            return
        for i, p in enumerate(self.active_policies):
            if p.policy_type == policy.policy_type:
                self.active_policies[i] = policy
                return
        self.active_policies.append(policy)

    # -- generic per-step eval + clip + write, then advance the dynamics --
    def _apply_active_policies(self) -> None:
        ctx = self.get_current_metrics()
        for p in self.active_policies:
            c = self._control(p.policy_type)
            if c is None:
                continue
            v = self._eval_control(c, p, ctx)
            if v is _UNCHANGED:                           # eval failed -> keep last value (today's behavior)
                continue
            c.set(self, v)

    def _eval_control(self, c: Control, p: Policy, ctx: dict):
        # Nullable direct-value path (price_ceiling/floor): expression may be "none"/"" -> None
        if c.nullable and p._compiled_safe_code is None:
            s = (p.value_expression or "").strip().lower()
            return None if s in ("none", "") else _clip(c, _to_float(s))
        if p._compiled_safe_code is None:
            return _UNCHANGED
        raw = evaluate_safe_policy_code(p._compiled_safe_code, ctx)
        if raw is None:                                  # runtime error -> keep last value, NOT garbage
            return _UNCHANGED
        if c.nullable and raw <= 0:
            return None                                  # SMM convention: value<=0 means "no cap"
        if not isinstance(raw, (int, float)):
            return _UNCHANGED
        return _clip(c, float(raw))

    def step(self) -> None:
        self._apply_active_policies()
        self._advance()                                  # the ONLY thing an author writes
        self.current_step += 1
        self.state = {"step": self.current_step,
                      "metrics": self.get_current_metrics(),
                      "active_policies_log": [p.to_dict() for p in self.active_policies]}
        self.history.append(self.state)

    def _advance(self) -> None:
        raise NotImplementedError

    # -- uniform agent context: a plain dict for EVERY world (kills pain point 4) --
    def get_state_for_agent(self) -> Dict[str, Any]:
        return {"metrics": self.get_current_metrics(),
                "params": self._init_params,             # param_A, sigma_epsilon, ... for the prompt
                "context_vars": self._context_vars(),
                "descriptors": self.get_policy_descriptors(),
                "active_policies": self.active_policies,
                "world_name": type(self).__name__}

    # -- the keystone primitive (section 6) --
    def clone(self) -> "WorldSpec":
        return copy.deepcopy(self)

    def rollout(self, policy: Optional[Policy], horizon: int, seed=None) -> List[dict]:
        w = self.clone()
        if seed is not None:
            w.rng = random.Random(seed); random.seed(seed); np.random.seed(seed)
        w.apply_policy_change(policy)
        traj = [w.get_current_metrics()]
        for _ in range(horizon):
            w.step(); traj.append(w.get_current_metrics())
        return traj

    def emulate_policy(self, policy, duration, agents_subset=None) -> Dict[str, Any]:
        return {"trajectory": self.rollout(policy, duration)}   # finally non-stub

_UNCHANGED = object()
def _to_float(s): 
    try: return float(s)
    except ValueError: return None
def _clip(c, v):
    if v is None or not c.value_range: return v
    lo, hi = c.value_range
    return max(lo, min(hi, v))
```

**Why drift is now structurally impossible for the policy/eval contract.** `get_current_metrics()`,
`_context_vars()` (whitelist source for `get_policy_descriptors()`), and the eval context passed to
`evaluate_safe_policy_code` all iterate the same `self.observables` + `self.controls`. The validator
in `policy_utils.validate_and_compile_policy_expression` is fed `descriptor.available_context_vars`,
which is `_context_vars()`. So *an expression that validates at compile time cannot reference a name
absent at eval time.* This is the airtight 80% the four proposals agree on, and it is the part we
keep without compromise.

### 2.3 The prompt is a FOUR-source contract — assembled, not auto-derived

This is where every proposal oversold, and where this design is honest. The real
`linear_system_prompt_obfuscated.md` placeholders are:

| placeholder | source | in observables? |
|---|---|---|
| `{current_x} {previous_x} {current_u} {target_x} {current_step}` | observables / metrics | yes |
| `{available_context_vars}` | `_context_vars()` | derived |
| `{param_A} {param_B} {param_C} {sigma_epsilon}` | **model params** | **no** |
| `{u_range}` | **a control's `value_range`** | **no** |
| `{perf_window} {current_mse} {current_msu}` | **agent KPIs** (`_calculate_performance_kpis`) | **no** |
| `{history_text}` | **agent formatter** (`_format_history_for_prompt`) | **no** |

So the prompt is `metrics ∪ params ∪ control-ranges ∪ agent-KPIs ∪ agent-text`. The agent assembles
it explicitly (section 6). To stop silent drift — `prompts_utils.DefaultMapping.__missing__` returns
the literal `"{param_A}"` string with a `# TODO Добавить Warning` and no warning today — we add a
**build-time validator**, run once per registered world at import:

```python
# govsim/world/validate.py  — fails LOUD at boot, not silently at the LLM
import re
def check_prompt(world: WorldSpec, template: str, agent_supplied: set) -> None:
    placeholders = set(re.findall(r"(?<!\{)\{([a-zA-Z_]\w*)\}", template))
    suppliable = (set(world.get_current_metrics())
                  | set(world._init_params)
                  | {f"{c.name}_range" for c in world.controls} | {"u_range"}
                  | {"available_context_vars"} | agent_supplied)   # mse/msu/perf_window/history_text/...
    missing = placeholders - suppliable
    if missing:
        raise ValueError(f"Prompt for {type(world).__name__} references unsuppliable names: {sorted(missing)}")
```

This single test is what *actually* closes pain point 2's prompt leg. The four proposals promised it
"for free"; it is not free, it costs ~15 lines and one assertion — but then it cannot rot silently.

---

## 3. Add a new toy world in N lines

A new toy world is a pure-dynamics class plus a `@register`'d `WorldSpec` subclass. Re-expressing
the existing `LinearStochasticSystem`:

```python
# govsim/world/worlds/linear.py
from govsim.world.spec import Observable, Control
from govsim.world.base import WorldSpec

class LinearWorld(WorldSpec):                                              # 1
    default_params = {"initial_x": 0.0, "param_A": 0.95, "param_B": 0.5,   # 2
                      "param_C": 0.0, "sigma_epsilon": 0.1,               # 3
                      "target_x": 0.0, "u_range": (-2.0, 2.0)}            # 4

    def __init__(self, params):                                            # 5
        super().__init__({**self.default_params, **params})               # 6
        p = self._init_params                                              # 7
        self.A, self.B, self.C = p["param_A"], p["param_B"], p["param_C"]  # 8
        self.sigma, self.target_x = p["sigma_epsilon"], p["target_x"]     # 9
        self.x, self.x_prev, self.u = p["initial_x"], p["initial_x"], 0.0  # 10
        self.observables = [                                               # 11
            Observable("current_x",  lambda m: m.x, "state x_k"),         # 12
            Observable("previous_x", lambda m: m.x_prev, "x_{k-1}"),      # 13
            Observable("target_x",   lambda m: m.target_x, "setpoint"),   # 14
        ]                                                                 # 15
        self.controls = [                                                  # 16
            Control("set_control_input", *attr_uc(self),                  # 17
                    value_range=p["u_range"], desc="control u_k")]        # 18
        self.history.append({"step": 0, "metrics": self.get_current_metrics(),
                             "active_policies_log": []})                   # 19

    def _advance(self):                                                    # 20
        self.x_prev = self.x                                               # 21
        self.x = self.A*self.x + self.B*self.u + self.C + self.rng.gauss(0, self.sigma)  # 22
```

(`current_u` is exposed automatically because every control's value is in `get_current_metrics()` and
`_context_vars()`; `attr_uc(self)` is just `(lambda m: m.u, lambda m, v: setattr(m, "u", v))`.)

**Count: ~22 lines of substance, one file, zero edits elsewhere** — no `create_economic_model`
branch in `simulation.py`, no `ECONOMIC_MODEL_PARAMS` copy in `config.py` (params live in
`default_params`; `config.py` keeps only *overrides*), no hand-written `apply_policy_change`, no
`if policy_type == ...` dispatch in `step()`, no `LinearSystemAgentContext` Pydantic class, no
whitelist list, no prompt-placeholder sync. Compare to the 265-line
`linear_stochastic_system.py` it replaces.

> Note: line 22 drops the `self.x**3` cubic that is flagged in the current code as a
> `# ! Временный эксперимент, не забыть убрать`. Keep or restore it per the experiment; it is one
> token either way.

---

## 4. Integrate the existing Mandel model

**`mandel testing/mandel_test.py` is NOT edited.** It is loaded by path and driven via its native
`run_period()`. Verified against the source: `Simulation.__init__(num_goods, num_households,
num_firms_per_sector_list, num_sectors)` (line 714); `import copy` and `import matplotlib.pyplot`
are at module top (lines 5, 4); `history_gdp/history_unemployment_rate/history_inflation` exist and
are appended inside `run_period` (lines 770-771, 1085/1096/1178); the `if __name__ == '__main__'`
guard (line 1403) does **not** fire under `importlib` because `__name__` becomes `"mandel_test"`.

```python
# govsim/world/worlds/mandel.py   (mandel_test.py is untouched)
import importlib.util
from pathlib import Path
from govsim.world.spec import Observable, Control, attr, hist
from govsim.world.base import WorldSpec

_p = Path(__file__).resolve().parents[3] / "mandel testing" / "mandel_test.py"
_s = importlib.util.spec_from_file_location("mandel_test", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

class MandelWorld(WorldSpec):
    default_params = {"num_goods": 3, "num_households": 100,
                     "num_firms_per_sector_list": [8, 8, 8], "num_sectors": 3, "seed": 0}

    def __init__(self, params):
        super().__init__({**self.default_params, **params})
        p = self._init_params
        if p.get("seed") is not None:                       # the ONLY reproducibility hook we can add
            import random, numpy as np; random.seed(p["seed"]); np.random.seed(p["seed"])
        self.sim = _m.Simulation(p["num_goods"], p["num_households"],
                                 p["num_firms_per_sector_list"], p["num_sectors"])
        self.observables = [                                                          # ── 5 lines ──
            Observable("gdp",          hist("history_gdp"), "real output"),
            Observable("unemployment", hist("history_unemployment_rate"), "U rate 0..1"),
            Observable("inflation",    hist("history_inflation"), "CPI inflation")]
        self.controls = [
            Control("set_unemployment_benefit_rate",
                    *attr("government.unemployment_benefit_rate"),
                    value_range=(0.0, 1.0), desc="ЦБ benefit rate (free lever)"),
            Control("set_target_inflation",
                    *attr("financial_system.target_inflation_rate"),
                    value_range=(0.0, 0.1), desc="Taylor-rule inflation target (free lever)")]
        self.history.append({"step": 0, "metrics": self.get_current_metrics(),
                             "active_policies_log": []})

    @property
    def current_step(self): return getattr(self.sim, "current_period", len(self.history))
    @current_step.setter
    def current_step(self, _): pass                          # Mandel owns its own counter

    def _advance(self):
        self.sim.run_period()                                # untouched 14-phase native step
```

**The wrap is ~5 lines of spec** (the two `observables`/`controls` blocks) plus boilerplate import
and `__init__`. The body is ~25 lines total.

**Honest lever choice — and the caveats the proposals hid.** We govern
`government.unemployment_benefit_rate` and `financial_system.target_inflation_rate` because they are
**read but never reassigned** inside `run_period` (lines 629, 671-673). We deliberately **avoid**
`government.tax_rate`: `set_tax_rate_and_collect_taxes` *recomputes* it every period to balance the
budget (`mandel_test.py:636`), so a `tax_rate` control would silently no-op. (Pinning
`min_tax_rate == max_tax_rate == u` *would* force it via the `np.clip` at line 637, but that
overrides the government's entire fiscal rule rather than "governing the tax rate" — a semantic
corruption we choose not to ship by default. If the experiment wants it, add a `Control` whose `set`
pins both bands; document it as a rule-override.)

Two real timing caveats, documented here rather than hidden:

1. **Observation/control phase mismatch.** `_apply_active_policies()` writes the levers *before*
   `run_period()`, but `unemployment`/`gdp` are appended at lines 1085/1096 *before* the tax/Taylor
   phases (1140/1180). So the regent's policy is computed from period `k-1`'s gauges and bites in
   period `k`'s second half. Acceptable, but it is *not* "untouched native step" with zero
   semantics.
2. **Asymmetric latency.** `target_inflation_rate` feeds the Taylor rule at the *end* of the period
   (line 1180), so it affects rates used *next* period; benefit rate bites *this* period (line 629).
   The two controls have different effective lags. The regent must not assume symmetry.

3. **`clone()`/`rollout()` cost & RNG.** `deepcopy` of a `Simulation` with hundreds of
   Firm/Household/Sector objects is correct but slow, so evolutionary rollout over Mandel will want
   fewer seeds / checkpoints. And because Mandel routes all randomness through *module-global*
   `random`/`np.random` (it never threads a local RNG), reproducible Mandel rollouts require the
   explicit `random.seed`/`np.random.seed` in `__init__` shown above — `deepcopy` alone cannot
   capture the global stream. This is the one place the no-rewrite rule limits us.

4. **No terminal signal.** Mandel can degenerate to "no active firms"; `run_period()` keeps
   appending, so observables freeze rather than raising. Add a soft check in `MandelWorld._advance`
   (`if not self.sim.firms: ...`) if a `done` flag is needed; the base `step()` has no `terminated`
   concept (we are not Gymnasium — section 7).

---

## 5. Auto-registration mechanism

`__init_subclass__` on `WorldSpec` writes `REGISTRY[cls.__name__] = cls` (shown in 2.2). This
replaces both hand-maintained dicts:

```python
# govsim/world/__init__.py  — importing the package populates REGISTRY
from importlib import import_module
import pkgutil
from govsim.world import worlds                       # the worlds subpackage
for _m in pkgutil.iter_modules(worlds.__path__):
    import_module(f"govsim.world.worlds.{_m.name}")   # triggers each __init_subclass__

# govsim/simulation.py  — the factory collapses to a lookup
from govsim.world.base import REGISTRY
import govsim.world                                   # noqa: F401  (populates REGISTRY)
def create_economic_model(model_type, params):
    try:
        return REGISTRY[model_type]({**REGISTRY[model_type].default_params, **params})
    except KeyError:
        raise ValueError(f"Неизвестный тип экономической модели: {model_type}")
```

The `create_economic_model` dict in `simulation.py:25-30`, its four per-model imports
(`simulation.py:14-17`), and the `ECONOMIC_MODEL_PARAMS` full copies in `config.py:24-78` all
disappear (config keeps only deltas). Adding a world is **one new file in `govsim/world/worlds/`**;
the loop discovers it. We use `__init_subclass__` over a decorator because subclassing `WorldSpec`
is already mandatory, so registration cannot be forgotten; we use `pkgutil.iter_modules` over
setuptools entry-points because this is a single in-repo package, not a plugin ecosystem — entry
points are the right tool only if third-party worlds ship as separate distributions, which they do
not.

---

## 6. One world, three regents — via `clone()` / `rollout()`

The keystone (grafted from Proposal 1's best idea, already a stated design principle in
`agents/03-refactor-plan.md` §1.6): the world is **numeric and regent-blind** — it applies an
`action` and advances — while a `Policy` is a **function `obs -> value`**, scored by
`rollout(clone(), policy, horizon)`. This is the one primitive every regent shares.

```python
# govsim/regents/policy.py  — the LLM expression as a first-class Policy
import numpy as np
class CodePolicy:                          # wraps an LLM/evolved expression for the sandbox
    def __init__(self, policy_type, expr, whitelist):
        from govsim.utils.policy_utils import validate_and_compile_policy_expression
        self.policy_type = policy_type
        self.value_expression = expr
        self._compiled_safe_code = validate_and_compile_policy_expression(expr, whitelist)
```

`Policy` and `CodePolicy` are the *same shape* the existing `step()`/`apply_policy_change` consume
(`policy_type`, `value_expression`, `_compiled_safe_code`), so nothing downstream changes.

- **LLM-code regent (today, `IntelligentLLMAgent`).** Decoupled in ~15 lines: drop
  `from ...linear_stochastic_system import LinearSystemAgentContext` and the
  `isinstance(current_state_for_agent, LinearSystemAgentContext)` guard
  (`gov_agent_linear.py:12, 232-235` — the exact reason it governs only linear worlds), and build
  the prompt dict explicitly from the four sources:

  ```python
  ctx = current_state_for_agent                              # the uniform dict from 2.2
  prompt_data = dict(ctx["metrics"])                         # observables + control values
  prompt_data.update(ctx["params"])                          # param_A, sigma_epsilon, ...
  prompt_data.update(self._calculate_performance_kpis(history))   # current_mse, current_msu
  prompt_data["perf_window"] = self.performance_window
  prompt_data["available_context_vars"] = ctx["context_vars"]
  prompt_data["u_range"] = next(d.value_range for d in ctx["descriptors"])   # control range
  prompt_data["history_text"] = self._format_history_for_prompt(history)
  prompt_data["policy_descriptors_text"] = format_policy_descriptors_for_prompt(ctx["descriptors"])
  prompt = self.prompt_template_content.format_map(DefaultMapping(prompt_data))
  ```

  Validation still uses `validate_and_compile_policy_expression(expr, descriptor.available_context_vars)`
  unchanged — and because `available_context_vars` is now the same projection as the eval context,
  a validating expression cannot fail at eval. The same regent then governs Linear, SingleMarket,
  and Mandel. (`_calculate_performance_kpis` is generalized to take the target/observable names from
  params instead of hard-coding `current_x`.) Optionally, before committing the policy, the regent
  scores it: `economic_system.clone().rollout(candidate, H)` -> pick the better — AlphaEvolve-style
  selection that falls straight out of the primitive.

- **RL regent (future).** A `Policy` whose value is a numeric action from a net. The world already
  exposes bounded controls (`value_range`) and a flat `get_current_metrics()` observation, so an RL
  loop is `reset = create_economic_model(...)`, `obs = get_current_metrics()`, `step()`. No
  per-world glue. RL emits numbers, bypassing the sandbox via a thin `apply_action({name: v})`; the
  LLM path stays sandboxed. We do **not** ship a Gymnasium adapter now (section 7).

- **Evolutionary regent (future).** Generate N candidate `CodePolicy` expressions, score each with
  `rollout(clone(), cand, H)`, keep the best. Fitness oracle = the *same* `rollout`. No per-world
  code; the previously-unimplemented `emulate_policy` is now the shared evaluation path.

---

## 7. Keep-or-drop calls

| Dependency / class | Call | Reasoning |
|---|---|---|
| **Mesa** (`mesa = "^3.2.0"`, `pyproject.toml:21`) | **DROP** | Declared but **zero `import mesa`** in the repo and not even installed. Linear/coupled worlds have no micro-agents (scalar difference equations); `SingleMarketModel` and Mandel *do*, but with bespoke same-step clearing (SMM's 10-iteration tâtonnement, Mandel's 14 ordered phases) that Mesa's per-agent scheduler cannot express without abandoning the scheduler. DataCollector's only win — time-series logging — is already `self.history` + `get_current_metrics()`. Remove the line; keep the repo dependency-light. |
| **Gymnasium** | **DROP (do not add)** | The brief never asks for it. A hand-rolled `Box`/`Dict` you must keep "bug-compatible with real Gymnasium" plus a `to_gymnasium()` adapter for SB3/CleanRL — which **no part of the repo uses** — is pure speculative surface. Plain `(low, high)` tuples on `Control.value_range` already give clipping and bounds. Add the ~120-line space layer *only when* an RL baseline actually lands. |
| **RestrictedPython** (`restrictedpython = "^8.0"`) | **KEEP** | It is the project's best safety property and works. `validate_and_compile_policy_expression` raising `PolicyValidationError` on out-of-whitelist identifiers at *compile* time is the loud-failure guarantee the whole design leans on. The `SandboxConfig` (math/np whitelist) is exactly right; unchanged. |
| **`Policy` / `PolicyDescriptor`** (`interfaces.py`) | **KEEP** | The world-declares-capability / regent-decides-behavior split is the project's keystone (`agents/03-refactor-plan.md` §1.1). `PolicyDescriptor` is now *generated* from the spec rather than hand-written; `Policy` (and `CodePolicy`) is unchanged. Do not invent a parallel shape. |
| **`LinearSystemAgentContext`** (Pydantic) | **DROP last** | Replaced by the uniform dict context. Delete only after the linear worlds are migrated, so old and new can coexist. |
| **`BaseEconomicSystem` ABC** | **KEEP** | `WorldSpec` *is a* `BaseEconomicSystem`, so `simulation.py`'s loop (`get_state_for_agent` / `decide_policy` / `apply_policy_change` / `step`) runs migrated worlds unchanged. No bidirectional shim needed (section 8). |

---

## 8. Migration path, and where the magic bites

**Incremental, always-green** (matches `03-refactor-plan.md`'s guiding constraint). New code is
~3 small files (`spec.py`, `base.py`, `validate.py`, ~200 LOC) plus one wrapper per world.

1. **Land the framework.** Add `govsim/world/{spec,base,validate}.py`. No behavior change;
   `simulation.py` still imports the old classes. Green.
2. **Migrate one world (Linear) behind the registry.** Add `govsim/world/worlds/linear.py`; switch
   `create_economic_model` to the `REGISTRY` lookup (section 5). The old
   `linear_stochastic_system.py` stays importable so nothing else breaks. Run
   `poetry run simulation` with the linear config — green.
3. **Decouple the regent.** Apply the ~15-line `gov_agent_linear.py` change (section 6) + the
   prompt-dict assembly + the `check_prompt` boot validator. Now the regent is world-agnostic.
4. **Migrate SimpleGrowth, SingleMarket (needs `nullable`), Coupled.** Each is a new file in
   `worlds/`; delete the old class once its wrapper passes. SingleMarket validates the `nullable`
   control path.
5. **Add `MandelWorld`** (section 4). New file only; `mandel_test.py` untouched.
6. **Delete `LinearSystemAgentContext` and the `config.py` full param copies** last.

**Honest downsides / where magic bites:**

- **The prompt is the leaky seam, not the policy contract.** The single source of truth is airtight
  for whitelist/eval/descriptor, but the prompt also needs params + KPIs + formatted text from
  outside the two lists. We mitigate with the boot-time `check_prompt` validator and explicit agent
  assembly — but if someone adds a `{new_param}` to a `.md` without it being a param/observable/KPI,
  `DefaultMapping` would still silently blank it *at runtime*; the validator catches it *at import*
  only if it runs. **Make `check_prompt` part of agent `__init__`, not optional.**
- **Lambda accessors aren't introspectable or serializable.** A typo inside a closure
  (`lambda m: m.gpd`) is caught at first `get`, not at construction. And a raising getter inside
  `get_current_metrics()` propagates into `step()`, which `simulation.py:147` catches and **breaks
  the whole loop**. Mitigation (mandatory): a boot smoke test that calls `get_current_metrics()`
  once per registered world, converting silent/late getter failures into loud boot failures.
- **`deepcopy` rollout is correct but not free for Mandel**, and Mandel reproducibility leans on the
  module-global re-seed (the no-rewrite rule's price). Defer heavy evolutionary use until needed.
- **`__init_subclass__` + accessor closures are implicit.** A reader (or the author in six months)
  must know the convention to see where `apply_policy_change` "comes from." This is the lowest-magic
  variant available (plain dataclasses + one base class + mro-walk), and it is documented here — but
  it is still indirection. The payoff (40-60 lines deleted per world, drift made impossible) earns
  it by the third world.
- **No `terminated`/`truncated`.** We are deliberately not Gymnasium; worlds that can collapse
  (Mandel) need an explicit soft check if a done-signal matters.

**Net.** Registration and toy-world integration become near-automatic (one file, ~22 lines). Mandel
wraps in ~5 lines of spec without editing it, with two documented timing caveats. The prompt and the
agent KPI math remain *explicit agent concerns* — assembled from four declared sources and guarded by
a build-time check — which is the honest version of "automatic" for this repo.
