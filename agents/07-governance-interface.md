# Governance Interface — the recommended design

> **Status (per [`08`](08-open-problems-and-opportunities.md) §5): DESIGN SPIKE — DEFERRED off the critical
> path.** This whole Chancery stack (Effect algebra + Ledger + Mediator + Code-of-Laws + DSL + tool-loop +
> `Credit`) is, by §10.5 below, dead weight on the toy worlds and premature on an unstable Mandel. **Build
> now only:** the per-step money-conservation assertion (`04` §A.3) and structured **reject-with-feedback**
> (today a bad expression silently returns `None`). Keep the rest as the validated target for when a
> conserved-by-construction economy actually exists.
> **Repositioned by [`09-grand-plan.md`](09-grand-plan.md):** this entire Chancery lives **inside**
> `EconomyActionInterface` — one Phase-4 sibling of `ScalarLeverInterface`, **not** the top-level action
> abstraction. "Wall 1" (the regent holds no balance write handle) becomes a structural consequence of the
> core layering: the regent only ever holds a read-only `Observation` + an `ActionSpace`.

Lead-architect synthesis of four governance-interface proposals (effect-API/capability engine,
pure declarative DSL, code-as-legislation, tool-calling-native loop) with adversarial critique
(scores: declarative-DSL **7/10**, effect-API **7/10**, code-as-legislation **6.5/10**,
tool-calling-native **6.5/10**). This is the design we build. It is grounded in the real files:
`govsim/utils/interfaces.py` (`Policy`, `PolicyDescriptor`, `BaseEconomicSystem`,
`BaseGovernmentAgent`), `govsim/utils/policy_utils.py` (the RestrictedPython sandbox),
`govsim/economic_models/single_market_model.py` (5 hand-coded policy types),
`govsim/economic_models/linear_stochastic_system.py` (the toy world), and
`mandel testing/mandel_test.py` (the real ledger). It **extends** [`06-model-integration.md`](06-model-integration.md)
(WorldSpec / Control / Observable / `clone()` / `rollout()`) and **sits on**
[`04-mandel-stabilization.md`](04-mandel-stabilization.md) (the SFC ledger / money-conservation discipline).

> Provenance: synthesized from a 4-architect design panel with adversarial critique. The single
> sharpest correct insight, recovered independently by three of the four critiques, is the spine of
> this doc: **in this repo conservation cannot come from "the ledger is double-entry"; it must come
> from the regent never holding a write handle to any balance.** Today every world mutates balances
> in place — `SingleMarketModel` does `self.tax_per_unit = float(policy_value)`
> (`single_market_model.py:298`) and Mandel does `hh.monetary_holdings -= tax_amount`
> (`mandel_test.py:650`), destroys money with `self.monetary_holdings = max(0, …)` (`:387`), and
> creates it with the `gov_debt_limit` clamp (`:658-659`). The fix is to deny the regent any name
> that mutates a balance and route every effect through one private ledger writer.
>
> The second honest headline, also recovered by every critique: **token conservation is necessary
> but NOT sufficient for the owner's ask.** The owner wants "cannot break the economy"; a structural
> ledger only guarantees "cannot create *tokens* from thin air, untraceably." Value (mark-to-market),
> behavior-rule changes, and off-ledger flows driven by a published rate are leaks a counts-only
> invariant misses. So the doc-04 global `M_total` assertion is **not** demoted to "belt-and-braces"
> — it is load-bearing, and the design carries it.

---

## 1. Verdict

Build **the Chancery**: an agentic governance loop in which the regent **never mutates state** and
instead (a) **studies** the world through a read-only `EconomyView` plus `clone()`/`rollout()`
counterfactuals, and (b) **legislates** by committing parameterized **Institutions** to a versioned
**Code of Laws**; each Institution fires once per period and returns only **Effects** from a closed,
conserved primitive vocabulary, which a **Mediator** costs/rate-limits/perturbs before a **private
double-entry Ledger** commits them — rejecting any batch that does not balance. The conceptual model
is **Ledger + Capabilities + Institutions + Mediation + Observation**; conservation is *structural*
(the regent has no name in scope that writes a balance — Wall 1), reinforced by the doc-04 global
`M_total` assertion that catches the value/off-ledger leaks a counts-only guarantee misses.
**Transport is a hybrid, split by loop level:** tool-calling for the outer deliberation loop
(`study` / `rollout` / `propose` / `amend` / `repeal`, with structured reject-with-feedback);
**declarative-JSON-first** for the per-period Institution body, with **effect-API-constrained
sandboxed code** as the escape hatch for novel institutions; and a **versioned Code-of-Laws file
store** as the persistence layer behind the tools (never as the live input channel). The
security-sandbox is kept **not** for host safety (obsolete per the brief) but because it is the
mechanism that denies the institution a write handle and keeps bodies pure and `clone()`-able — the
real, still-present concern is **economic** mediation, not host security.

This primary model is the **tool-calling-native loop** ("Chancery"), with the **conservation kernel
+ capability-injected EffectAPI** grafted from the effect-API and code-as-legislation proposals, the
**declarative-first institution surface + structured rejection** from the declarative-DSL proposal,
and the **versioned Code-of-Laws genome** from code-as-legislation. We **reject** the oversold claim
common to all four ("thin-air resources are structurally impossible / the economy can't be broken")
and replace it with the honest, defensible claim the kernel actually delivers: **thin-air resources
are never *untraceable*, and every regent effect is conserved-by-construction — but the world's own
non-conserved dynamics remain doc-04's job, and a `SetRate` lever can still drive off-ledger value.**

---

## 2. The conceptual model

Five nouns, one loop. The regent's entire universe of action is `{edit the Code of Laws} ∪ {ask the
Observation channel}`. It holds no reference to any account, agent, or the Ledger.

- **LEDGER** — doc-04's stock-flow-consistent book. Sole owner of every balance (Mandel's
  `monetary_holdings`, `savings`, `debt`, `government.monetary_holdings`; SMM's would-be money
  accounts; goods quantities). Its only mutating method `_post(batch)` is **private** and commits
  iff the batch sums to zero per resource dimension (except authorized mint/burn, §3). Reads are
  free for everyone via `balance(acct)`; writes go only through `_post`, which only the engine calls.
- **CAPABILITIES** — what *this* experiment lets the regent do: which Effect verbs are enabled,
  which accounts/goods/levers exist, which external sinks are mintable, the cost schedule, the
  legislative budget, and the control-error model. All **data** in a `GovernanceSpec` (an extension
  of doc-06's WorldSpec), never code. A future experiment is authored by choosing capabilities, not
  by writing a framework branch.
- **INSTITUTIONS** — named, parameterized, versioned standing rules. Each fires once per period,
  reads the `EconomyView`, and **returns** a `List[Effect]`. A tax is an institution; a subsidy is
  an institution; cap-and-trade is a set of institutions. Body is **declarative** (the common case)
  or **effect-API-constrained sandboxed code** (the escape hatch). Persisted in the Code of Laws.
- **MEDIATION** — the clerk between an emitted Effect and the Ledger. Charges treasury (action
  accounting), enforces the legislative budget and cooldowns (rate limits), and injects
  noise/lag/leakage (control errors). Leakage is a *real Transfer to a loss account*, so even
  corrupted effects stay conserved.
- **OBSERVATION** — the read-only channel: `EconomyView` (balances/goods/rates/observables, no
  setters) + `read_code_of_laws()` + `rollout(proposal, horizon, seed)` (doc-06 `clone()`/`rollout()`,
  which finally implements the stubbed `emulate_policy`, `interfaces.py:158-178`).

### 2.1 The control loop (diagram-in-text)

```
                          ┌────────────────────────── OUTER LOOP (per decision interval) ─────────────────────────┐
                          │                                                                                        │
   ┌──────────┐  study   │  ┌───────────────┐   draft    ┌──────────────┐  rollout(clone)   ┌──────────────────┐  │
   │  REGENT  │─────────────▶│ EconomyView   │───────────▶│  candidate   │──────────────────▶│  Counterfactual  │  │
   │  (LLM)   │◀─────────────│ (READ ONLY)   │            │  Institution │◀──────────────────│  trajectory      │  │
   └────┬─────┘  view/      │  └───────────────┘  Define/  └──────┬───────┘   keep if better  └──────────────────┘  │
        │        rollout    │                     Amend/Repeal     │                                                 │
        │  propose(effects) │                                      ▼                                                 │
        └───────────────────┼──────────────────────────▶ ┌──────────────────┐  reject {effect, reason, budget_left} │
                            │   tool call                 │  MEDIATOR        │──────────────────────────────────────┘
                            └─────────────────────────────│  cost+rate+noise │   (structured feedback to regent)
                                                          └────────┬─────────┘
                                                                   │ well-formed Effects only
   ┌─────────────── INNER LOOP (every period, inside step() AND inside rollout) ───────────────┐
   │                                                                                            │
   │   ┌──────────────┐  fire(view)  ┌──────────────┐  Effects   ┌──────────┐  _post(balanced) ┌────────┐
   │   │ CODE OF LAWS │─────────────▶│ each active  │───────────▶│ MEDIATOR │─────────────────▶│ LEDGER │
   │   │ (versioned)  │              │ Institution  │            │ (same as │   else raise      │(private│
   │   └──────────────┘              │ (view→[Eff]) │            │  above)  │  Conservation     │ writer)│
   │          ▲                      └──────────────┘            └──────────┘   Error           └───┬────┘
   │          │ Define/Repeal commit                                                                │
   │          │ (charged, rate-limited)                                                  world._advance()
   │          └────────────────────────────────────────────────────────────────────────────────────┘
   │   then doc-04 GLOBAL ASSERTION: M_total(inside∪external) unchanged → else localize the leak     │
   └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The decisive point made by the diagram: the **same** Mediator + Ledger run in the inner loop and
inside `rollout()`, so counterfactuals include costs, noise, and conservation — the regent can A/B a
draft institution before paying to enact it. And the regent's only arrows are *study* (read) and
*propose* (data); there is no arrow from the regent to a balance.

---

## 3. The conserved primitive vocabulary, and why thin-air is impossible

Six frozen-dataclass Effect types. The regent/institution **never instantiates an account object**;
it names accounts by string id, validated by the Ledger against the world's declared registry. The
Ledger knows how to **expand** each Effect into balanced double-entry rows; the institution body
never sees `_post`.

```python
# govsim/gov/effects.py
from dataclasses import dataclass
from typing import Optional, Literal, Union

Account = str        # "hh:7", "firm:3", "gov", "treasury", "regulator", "permit_pool"
Good    = Optional[str]   # None => money; else a goods dimension ("permit", "good:0", ...)

# External sinks: a CLOSED enum of "outside-the-economy" accounts that may go negative.
ExternalSink = Literal["CENTRAL_BANK_RESERVE", "FOREIGN_SECTOR", "NATURE_SINK", "EXOGENOUS_ENDOWMENT"]

@dataclass(frozen=True)
class Transfer:                       # relabel who-holds-what. Conserves money (or `good`).
    src: Account; dst: Account; amount: float; good: Good = None; memo: str = ""
    #  -> [(src,-amount),(dst,+amount)]  (+ goods rows if good is not None). amount>=0; sign is src/dst.

@dataclass(frozen=True)
class Allocate:                       # spend FROM TREASURY. A Transfer pinned to src="treasury".
    dst: Account; amount: float; good: Good = None; purpose: str = ""
    #  -> Transfer("treasury", dst, amount). Mediator REJECTS on insufficient treasury (no overdraft).

@dataclass(frozen=True)
class Mint:                           # the ONLY net-change channel. amount>0 mint, <0 burn.
    account: Account; amount: float; sink: ExternalSink; reason: str; good: Good = None
    #  -> [(account,+amount),(sink,-amount)]  with mint_burn_authorized=amount. sink REQUIRED.

@dataclass(frozen=True)
class Burn:                           # sugar for Mint(account, -amount, sink, ...). Same expansion.
    account: Account; amount: float; sink: ExternalSink; reason: str; good: Good = None

@dataclass(frozen=True)
class SetRate:                        # publish/overwrite a world-declared POLICY PARAMETER.
    lever: str; value: Union[float, list, dict, None]
    #  -> NO balance rows. Writes world.policy_params[lever] (clamped to the Control.value_range).

@dataclass(frozen=True)
class Define:                         # register/replace an Institution in the Code of Laws.
    institution_id: str; spec: dict   # declarative spec OR {"code": "<sandboxed source>", "params": {...}}
@dataclass(frozen=True)
class Repeal:
    institution_id: str

Effect = Union[Transfer, Allocate, Mint, Burn, SetRate, Define, Repeal]
```

The Ledger's private writer is the entire conservation surface:

```python
# govsim/gov/ledger.py
EXTERNAL = {"CENTRAL_BANK_RESERVE", "FOREIGN_SECTOR", "NATURE_SINK", "EXOGENOUS_ENDOWMENT"}
TOL = 1e-9

class Ledger:
    def __init__(self, accounts, goods=("money",)):
        self._bal = {(g, a): 0.0 for g in goods for a in accounts}   # per (good, account)
        self.journal = []
    def balance(self, acct, good="money"):           # READ-ONLY handle for everyone
        return self._bal[(good, acct)]
    def _post(self, rows, *, authorized_mint_burn=0.0, good="money"):   # PRIVATE; only the engine calls it
        s  = sum(d for _, d in rows)
        mb = sum(d for a, d in rows if a in EXTERNAL)
        if abs(s) > TOL and (abs(mb) < TOL or abs(s) > abs(authorized_mint_burn) + TOL):
            raise ConservationError(good, rows)      # belt-and-braces; constructors make this unreachable
        for a, d in rows:
            self._bal[(good, a)] += d
        self.journal.append((good, tuple(rows), authorized_mint_burn))
```

**Why thin-air resources are impossible — three nested walls; Walls 1 and 2 are type-level.**

- **WALL 1 — the regent holds no write handle.** The institution sandbox namespace contains exactly
  `view: EconomyView` (read-only accessors, no setters, no reference to any account/agent/Ledger)
  plus the Effect constructors plus the math/np whitelist. This reuses
  `policy_utils.validate_and_compile_policy_expression`'s existing identifier whitelist
  (`policy_utils.py:142-149`): an institution literally cannot name `monetary_holdings`, `firms`,
  `government`, `setattr`, or `_post` — they are not in the whitelist and the call raises
  `PolicyValidationError` at compile. There is no attribute to write and no object to write it on.
  (Contrast: the *world* today does `firm.monetary_holdings -= cost`; the regent's code will never
  have `firm` or `monetary_holdings` in scope.)
- **WALL 2 — the write surface is a closed, conserved algebra.** Institution code does not write; it
  **returns** a `List[Effect]`. Of the money-touching verbs, `Transfer`/`Allocate` take *both* legs
  and the Ledger derives `[-amount, +amount]` from the single object — you **cannot author the two
  legs separately**, so there is no way to express "credit dst without debiting src." `SetRate`
  touches no balance. Hence `Σ Δbalance` over any multiset of `Transfer`/`Allocate`/`SetRate` is
  zero by construction: `Σ (transfers: -a+a) + Σ (allocate: treasury -a, dst +a) + 0 = 0`.
- **WALL 3 — mint/burn is the only net-change primitive, and it is forced double-entry against a
  named, audited external sink.** `Mint(account, amount, sink, reason)` posts `+amount@account` and
  `-amount@sink`, so the **full** (inside ∪ external) total is conserved; the inside-economy total
  changes by exactly `amount`, traceable to one named sink line and one `reason`. Minting is not
  "thin air" — it is an explicit, logged issuance from a named external balance that goes negative by
  the minted amount. `sink` is a required field validated against the closed `ExternalSink` enum; a
  regent that lacks the mint capability for this experiment simply cannot construct a `Mint` the
  Mediator will accept.

So thin-air money requires either (W1) getting a balance object into scope — impossible, not in the
whitelist; or (W2) constructing an unbalanced non-mint Effect — impossible, constructors take both
legs; or (W3) minting without a sink — impossible, `sink` is a required typed field. The guarantee is
at construction/type time, before any value the LLM chose is even looked at.

**What the kernel does NOT guarantee (and where doc-04's assertion is load-bearing).** Token
conservation is not value conservation and not economic safety. Three residual leaks the critiques
correctly identified, all of which keep `_post` perfectly balanced:

1. **`SetRate` drives off-ledger flows.** A published rate is read by the world's *own* non-conserved
   code. In SMM, `subsidy_cost = actual_traded_quantity * subsidy_per_unit` (`single_market_model.py:440`)
   credits firm profit with no posting; in Mandel, `benefit_per_unemployed = rate * average_wage`
   then `hh.monetary_holdings += benefit` (`mandel_test.py:629, 651`) moves money off-ledger. A
   `SetRate` is conjectural, not conserved.
2. **Valuation arbitrage.** `Transfer` conserves the *count* of a good and, separately, of money —
   never their product. A regent that controls a price/valuation lever and can move goods can inflate
   mark-to-market net worth (Mandel values `fixed_capital` at `market_prices_snapshot`) with zero
   offsetting posting.
3. **Negative-balance laundering.** `_post` checks net-zero but not solvency; a regent can drive an
   account negative with a legal balanced Transfer, then the world's own `max(0, …)` clamp (`:387`)
   or `gov_debt_limit` clamp (`:658-659`) launders the negative into created money.

The mitigations (carried by this design, not optional): (a) `_post` also enforces **non-negativity /
solvency** for inside accounts unless an explicit `Mint` precedes (kills #3); (b) **every world
mint/burn channel** — production, interest, firm entry/exit (`:862`) — is itself a typed, logged
`Mint`/`Burn` with its own per-period sub-test, so a `SetRate` that resizes a channel is bounded and
audited (bounds #1); (c) the **doc-04 global `M_total` assertion runs every step over inside ∪
external** and *localizes* any value/off-ledger leak the structural wall misses. The honest claim is
**"thin-air is never untraceable, and every regent effect is conserved-by-construction,"** with #1/#2
explicitly contingent on doc-04's SFC refactor of `mandel_test.py` landing first (§9, §10).

---

## 4. The institution model: declarative + effect-API-constrained code

An **Institution** is a named, parameterized, versioned entry in the Code of Laws. It is the
generalization of the 5 hand-coded `SingleMarketModel` policy types and the linear-world `u_k`
one-liner.

```python
# govsim/gov/institutions.py
from dataclasses import dataclass, field

@dataclass
class Institution:
    id: str
    body: "InstitutionBody"        # declarative spec OR sandboxed code; both -> (view, params) -> List[Effect]
    trigger: dict = field(default_factory=lambda: {"every": 1})   # {"every":k} | {"when":"<sandboxed bool over view>"}
    params: dict = field(default_factory=dict)   # the knobs the LLM/evolution mostly tunes
    capabilities: frozenset = frozenset()        # subset of {transfer, allocate, mint, burn, set_rate}
    enact_cost: float = 0.0
    upkeep: float = 0.0            # treasury charged each period it fires (req 2)
    enacted_at: int = 0
    version: int = 1
```

**The read-only-view + emit-handle boundary that stops raw-state writes.** Both body kinds are typed
to return `List[Effect]`. The body's *entire* namespace is `{view, params, Transfer, Allocate, Mint,
Burn, SetRate, math, np, clip}` — **no** `_post`, **no** `_bal`, **no** world/account object, **no**
`setattr`. The only outward channel is the return value, which the engine routes through Mediator →
Ledger. "Arbitrary code over a safe alphabet is still safe": the body can compute any number, but the
only way a number reaches the world is inside an Effect the engine audits. A body that emits a verb
outside its `capabilities` (e.g. a stray `Mint` when `capabilities={transfer}`) is rejected at emit.

**Body option A — DECLARATIVE (the common 80%; introspectable, diffable, rollout-safe, no code runs).**
A small list of effect templates whose scalar fields are RestrictedPython expressions over the view,
compiled by the existing `validate_and_compile_policy_expression` (whitelist = observable names +
`params`). Example shape:

```json
{ "kind": "per_account", "over": "households",
  "emit": [ {"effect": "Transfer", "src": "$account", "dst": "treasury",
             "amount": "progressive(view.income($account), params.brackets)"} ] }
```

The fixed DSL resolvers (`$account`, `view.*`, `params.*`, helpers like `progressive`,
`clear_uniform_price`) are a whitelist; output is always a `List[Effect]`.

**Body option B — CODE (the escape hatch for novel institutions).** Sandboxed source defining
`def fire(view, params):` returning a `List[Effect]`, compiled by the **same** `policy_utils`
machinery (extended from eval-of-expression to exec-of-one-function) with the safe alphabet above.

**Persistence — the Code of Laws.**

```python
# govsim/gov/code_of_laws.py
class CodeOfLaws:
    """Append-only, versioned registry; the ONLY thing a regent proposal mutates. deepcopy-safe so it
       travels with world.clone() (doc-06) — counterfactuals include already-passed laws."""
    def __init__(self):
        self.active: dict[str, Institution] = {}
        self.log: list = []           # full enact/amend/repeal history with step + cost
    def apply(self, effect):          # Mediator calls this for Define/Repeal only
        if isinstance(effect, Define):
            prev = self.active.get(effect.institution_id)
            inst = compile_institution(effect.institution_id, effect.spec,
                                       version=(prev.version + 1 if prev else 1))
            self.active[effect.institution_id] = inst
            self.log.append(("define", inst.id, inst.version))
        elif isinstance(effect, Repeal):
            self.active.pop(effect.institution_id, None)
            self.log.append(("repeal", effect.institution_id))
    def fire(self, step, view) -> list:
        out = []
        for inst in sorted(self.active.values(), key=lambda i: i.enacted_at):  # deterministic order
            if inst_should_fire(inst, step, view):
                out += run_body(inst, view)        # declarative-eval or sandboxed-eval -> List[Effect]
        return out
```

**Unification with doc-06.** A doc-06 `Control("set_control_input", range (-1,1))` is the degenerate
institution `Define(id="control_law", spec={trigger:{every:1}, emit:[SetRate("u", "$proposed clipped")]})`.
The 5 SMM policies become four `SetRate` institutions + one `Allocate` institution
(`government_purchase` = `Allocate("treasury" → market, amount)`, which is *correctly conserved* —
fixing that today `government_net_revenue` is a free-floating scalar at `single_market_model.py:443`).

---

## 5. Two worked examples end to end

### 5.1 Progressive income tax — conservation holds because tax is a Transfer

DECLARATIVE (preferred). The regent emits one `Define`:

```python
Define(
  institution_id="progressive_income_tax",
  spec={
    "trigger": {"every": 1},
    "capabilities": ["transfer"],
    "upkeep": 2.0,                                   # collection bureaucracy costs 2/treasury/period (req 2)
    "params": {"brackets": [[0,0.00],[20,0.10],[50,0.25],[100,0.40]]},   # (upper_bound, marginal_rate)
    "body": {"kind": "per_account", "over": "households",
             "emit": [{"effect": "Transfer", "src": "$account", "dst": "treasury",
                       "amount": "progressive(view.income($account), params.brackets)"}]}})
```

`progressive(income, brackets)` is a fixed safe DSL function (marginal-bracket integral, returns
`>= 0`). The engine expands the template each period into one `Transfer($hh, "treasury", owed)` per
household.

CODE form (for a curve the bracket DSL can't express, e.g. a refundable credit):

```python
# namespace = {view, params, Transfer, Allocate, math, clip}
def fire(view, params):
    effects, b = [], params["brackets"]
    for acc in view.accounts_in("households"):
        inc, owed, lo = view.income(acc), 0.0, 0.0
        for upper, rate in b:
            if inc <= lo: break
            owed += (min(inc, upper) - lo) * rate; lo = upper
        if inc < 8.0:                                # refundable credit = Allocate (NOT thin-air)
            effects.append(Allocate(dst=acc, amount=(8.0 - inc) * 0.5, purpose="EITC"))
        elif owed > 1e-9:
            effects.append(Transfer(src=acc, dst="treasury", amount=owed))
    return effects
```

**Conservation, exactly.** Each household's tax is `Transfer(hh → treasury, owed)`, which the Ledger
expands to `[(hh,-owed),(treasury,+owed)]`, summing to zero. `Σ` over households: money lost by
households == money gained by treasury, every step. The refundable credit is `Allocate(treasury →
hh)` = a Transfer with `src="treasury"`, also net-zero; if treasury can't fund it the Mediator
**rejects that single Allocate** with feedback and keeps the rest of the batch. Contrast the status
quo: Mandel does `hh.monetary_holdings -= tax_amount`, separately `gov += collected - benefits`, then
**clamps** to `gov_debt_limit` (`mandel_test.py:650-659`), creating money whenever the clamp binds.
Here the clamp is unrepresentable (no setattr); a deficit must be an explicit
`Mint(account="treasury", amount=deficit, sink="CENTRAL_BANK_RESERVE", reason="gov bond issuance")` —
visible, logged, costed.

### 5.2 Cap-and-trade emission permits — a novel institution, NO new framework code

`permit` is declared by the `GovernanceSpec` as a **conserved good dimension** with external sink
`NATURE_SINK`; firms emit per unit produced and production is gated on held permits. (One-time world
declaration, not a per-institution code path — and the genuine prerequisite the critiques flagged:
permits only become a *conserved* good once the goods-ledger exists; until then this rides on the
declared dimension, see §10.) The regent enacts the scheme as institutions:

```python
# (1) Annual cap: MINT the year's permits into the regulator from NATURE_SINK (the environment's
#     carrying capacity going negative by the licensed amount). good="permit".
Define("carbon_cap", spec={"trigger": {"every": 12}, "capabilities": ["mint"], "upkeep": 1.0,
  "params": {"annual_cap": 500.0},
  "body": {"code": "return [Mint('regulator', params['annual_cap'], sink='NATURE_SINK',"
                   "             good='permit', reason='annual emission cap issuance')]"}})

# (2) Auction: clear at a uniform price; revenue firm->treasury (money), permits regulator->firm (good).
#     BOTH legs are Transfers => both conserved. clear_uniform_price is a safe DSL helper.
Define("permit_auction", spec={"trigger": {"every": 1}, "capabilities": ["transfer"], "upkeep": 3.0,
  "params": {"reserve_price": 2.0},
  "body": {"code":
    "effects=[]\n"
    "supply = view.holdings('regulator', good='permit')\n"
    "alloc  = clear_uniform_price(view.observable('permit_bids'), supply, params['reserve_price'])\n"
    "for firm, qty, price in alloc:\n"
    "    effects.append(Transfer(firm, 'treasury', qty*price))                       # money leg\n"
    "    effects.append(Transfer('regulator', firm, qty, good='permit'))             # good leg\n"
    "return effects"}})

# (3) Production cap = a SetRate the world's production phase reads (no balance touched).
Define("emission_gate", spec={"trigger": {"every": 1}, "capabilities": ["set_rate"],
  "body": {"code": "return [SetRate('production_requires_permit', True)]"}})

# (4) Retirement: when a firm emits, its permits return to NATURE_SINK via Burn.
Define("permit_retirement", spec={"trigger": {"every": 1}, "capabilities": ["burn"], "upkeep": 0.5,
  "body": {"code":
    "return [Burn(f, view.observable('emitted_by:'+f), sink='NATURE_SINK', good='permit',"
    "             reason='permit retired on emission') for f in view.accounts_in('firms')]"}})
```

**Conservation, leg by leg.** MONEY: the only mover is the auction `Transfer(firm → treasury)` →
`-qty*price@firm + qty*price@treasury = 0`; revenue-to-treasury is automatic and exact. PERMITS (new
good): issuance `Mint(+cap@regulator, -cap@NATURE_SINK)` (inside ∪ sink conserved); auction
`Transfer(regulator → firm)` (`Σ Δpermit inside = 0`); retirement `Burn` returns permits to the sink.
Over a full cycle `NATURE_SINK`'s net change = `-(issued) + (retired)`; if every issued permit is
retired the sink returns to zero — the environmental budget is *accounted*, not hand-waved. PRODUCTION
CAP: enforced via `SetRate` read by the world's production gate; a firm with zero permits produces
zero (and per §3 mitigation (b), the production channel that `SetRate` gates is itself a logged
`Mint`/`Burn` of the real good, so the lever-resizes-a-channel leak is bounded and audited).

A **Pigouvian-tax** variant is smaller and uses the same machinery:
`Define("carbon_tax", spec={"capabilities":["transfer"], "params":{"rate":4.0}, "body":{"code":
"return [Transfer(f, 'treasury', params['rate']*view.observable('emitted_by:'+f)) for f in
view.accounts_in('firms')]"}})` — emissions taxed, revenue to treasury, conserved by Transfer
construction; make it state-contingent with `trigger={"when":"view.observable('total_emissions') >
params['threshold']"}`.

Both examples are entirely **data** (Define effects carrying specs/bodies) over the same six
primitives + one declared good. No new `step()` branch, no new world method, no framework edit (req 6).

---

## 6. The mediation layer

The Mediator sits between every emitted Effect and the Ledger, in both the inner and outer loops.
Configured per-experiment as `GovernanceSpec` data (req 6) — pure capability config, no code.

```python
# govsim/gov/mediation.py
class Mediator:
    def __init__(self, spec, ledger, code_of_laws, rng):
        self.spec, self.led, self.col, self.rng = spec, ledger, code_of_laws, rng
        self.budget_left = spec.legislative_budget      # reset each period
        self.delay_queue = []                           # for lag
    def process_legislation(self, effects):             # OUTER loop: Define/Repeal/SetRate changes
        applied, rejected = [], []
        for e in effects:
            if isinstance(e, (Define, Repeal)):
                if self.budget_left <= 0:               # req 3: legislative budget
                    rejected.append((e, "legislative budget exhausted")); continue
                if self._cooldown_active(e):
                    rejected.append((e, "cooldown active")); continue
                fee = self.spec.cost.enact if isinstance(e, Define) else self.spec.cost.repeal
                if not self._charge(fee, "enactment"):  # req 2: irreversible treasury debit
                    rejected.append((e, f"treasury < {fee}")); continue
                self.budget_left -= 1
                self.col.apply(e); applied.append(e)
        return {"applied": applied, "rejected": rejected,
                "budget_left": self.budget_left, "treasury": self.led.balance("treasury")}
    def commit(self, effects):                          # INNER loop: institution-emitted primitives
        for e in self._release_due() + self._distort(effects):
            self._expand_and_post(e)                    # -> Ledger._post(balanced rows)
    # --- req 2: action accounting ---
    def _charge(self, amount, purpose):
        if self.led.balance("treasury") < amount: return False
        self.led._post([("treasury", -amount), ("BUREAUCRACY", +amount)]); return True   # conserved
    def upkeep_pass(self):                              # each period, per active institution
        for inst in list(self.col.active.values()):
            if not self._charge(inst.upkeep, f"upkeep:{inst.id}"):
                self.col.apply(Repeal(inst.id))         # unaffordable institutions lapse (logged)
    # --- req 4: control errors (all via the world-local rng so they travel with clone()) ---
    def _distort(self, effects):
        out = []
        for e in effects:
            if isinstance(e, (Transfer, Allocate)) and self.spec.noise.sigma:
                e = replace(e, amount=e.amount * (1 + self.rng.gauss(0, self.spec.noise.sigma)))   # NOISE
            if self.rng.random() < self.spec.noise.drop_prob:                                       # FAILURE
                continue
            if self.spec.noise.leak and isinstance(e, (Transfer, Allocate)):                        # LEAKAGE
                leak = e.amount * self.spec.noise.leak
                out.append(Transfer(src=_dst_of(e), dst="LOSS", amount=leak))   # conserved diversion
                e = replace(e, amount=e.amount - leak)
            (self.delay_queue.append((self.spec.noise.lag, e)) if self.spec.noise.lag else out.append(e))  # LAG
        return out
```

- **Action accounting (req 2).** `enact`/`repeal`/`amend` charge treasury at commit; each active
  institution charges `upkeep` every period; charges are themselves conserved
  `Transfer(treasury → BUREAUCRACY)`. A regent that founds many institutions drains treasury and
  eventually *cannot* enact more (Allocate/charge rejected) — institutions are not free to create or
  to keep. This generalizes `PolicyDescriptor.constraints {'change_cost': ...}`
  (`interfaces.py:75-78`, used at `single_market_model.py:175`) from a logged number to an enforced,
  conserved debit.
- **Rate limits (req 3).** A per-period `legislative_budget` caps Define/Repeal/SetRate *changes*;
  per-institution cooldowns (`trigger {"every":k}` + min-interval from `enacted_at`) cap
  twiddling. Generalizes `constraints {'min_change_interval': 5}` (`interfaces.py:76`). Effects from
  an *already-active* institution are free (the law was already passed) — only **changes to the Code
  of Laws** consume the budget, which is the right economic semantics.
- **Control errors (req 4).** Noise (multiplicative on amounts/rates), lag (a delay queue releasing
  after `d` steps), leakage (a fraction diverted to a real `LOSS` account — *conserved*), and
  implementation failure (drop with feedback). All seeded by the world-local `self.rng` (doc-06
  §2.2), so they are reproducible and travel with `clone()`.

---

## 7. The observation channel

Read-only, side-effect-free, and the single home for "study the model" (req 5). Three capabilities,
all tying to doc-06's `clone()`/`rollout()`:

```python
class EconomyView:                       # the ONLY object in an institution's namespace besides effects
    def balance(self, acct, good="money"): ...      # read-only
    def holdings(self, acct, good): ...
    def income(self, acct): ...
    def rate(self, name): ...                        # world.policy_params[name]
    def observable(self, name): ...                  # == doc-06 get_current_metrics()[name]
    def accounts_in(self, sector): ...               # names only; cannot conjure an account
    def treasury(self): ...
    # NO setters, NO reference to any account/agent/Ledger.
```

- **query / inspect** — `EconomyView` exposes balances, goods, published rates, doc-06 observables,
  and `read_code_of_laws()` returns the active institutions + journal. Within-period, the institution
  body reads the same view.
- **counterfactual rollout** — `rollout(proposal, horizon, seed)` is doc-06's `clone()` + `rollout()`
  (`06-model-integration.md` §2.2), extended to deep-copy the `CodeOfLaws`, `Mediator` queues, and
  `Ledger` alongside the world (all plain Python, deepcopy-safe). It applies a proposed Define/Repeal
  to the clone and runs the **same** Mediator + Ledger, returning the trajectory of
  `get_current_metrics()` **plus** ledger balances. This finally implements the stubbed
  `emulate_policy` (`interfaces.py:158-178`; today `raise NotImplementedError` /
  `linear_stochastic_system.py:258`, `single_market_model.py:462`).
- **MANDATE rollout-before-enact, and average over seeds.** Because control-errors are stochastic
  (req 4) and Mandel routes randomness through *module-global* `random`/`np.random` (doc-06 §4 caveat
  3 — `deepcopy` can't capture the global stream, so re-seed in `clone()`), a single rollout is one
  sample path. `rollout()` accepts `n_seeds` and returns mean/variance; the harness should require a
  `dry_run`+`rollout` before every `propose()` (current models use a voluntary study channel
  inconsistently), and the prompt must warn against single-seed conclusions (doc-04 §3.3 Monte-Carlo
  discipline; Vending-Bench tail-variance caveat).

The regent's loop becomes: `study(view) → draft → rollout(clone, draft, H, n_seeds) → keep if better
→ propose`. This is doc-06's "simplified prediction emulation" keystone, now spanning institutions,
not just scalar controls.

---

## 8. Transport verdict

A **hybrid, split by loop level** — and an explicit rejection of "per-period tool calls," "policy
files as the live input channel," and "free JSON-with-code as the only representation."

**OUTER LOOP — tool calling (primary).** Once per decision interval (today every 50 steps,
`config.py` `agent_decision_frequency`), expose a small fixed tool set:
- `study_economy()` → the `EconomyView` snapshot (read-only, req 5)
- `rollout(proposal, horizon, seed, n_seeds)` → trajectory + mean/variance (req 5)
- `propose(effects)` → Mediator result `{applied, rejected:[{effect, reason}], cost_charged, budget_left}`
- `amend(id, param_patch)` / `repeal(id)` → cheap re-tune / teardown (charged, rate-limited)

This matches the existing `decide_policy` cadence (`interfaces.py:193`, `BaseGovernmentAgent`), gives
the LLM clean read/simulate/act verbs, and — crucially — gives every owner requirement a *home*: the
tool implementation **is** the Mediation layer, where cost / rate-limit / noise / conservation run
*before* anything touches the Code of Laws. A one-shot returned-Policy (today's design) has nowhere to
charge budget, inject noise, or loop study→act. Structured **reject-with-feedback** is the single most
important usability property — a strict upgrade over the status quo where a bad expression silently
returns `None` and the world keeps the old value (`single_market_model.py:262-264`,
`linear_stochastic_system.py:227-229`) with no signal to the agent.

**INNER REPRESENTATION — declarative-JSON first, sandboxed effect-API code as escape hatch.** The
per-period institution body must be inert data the engine can execute *and clone*: a tool round-trip
per period is un-`rollout()`-able (you cannot `clone()` a network call) and would pay an LLM call per
institution per step — requirement 5 is **incompatible** with per-period tool calls. So the body is a
declarative spec (common case: introspectable, diffable, safe by construction) or sandboxed source
compiled by the existing `policy_utils` machinery
(`validate_and_compile_policy_expression` / `evaluate_safe_policy_code`, `policy_utils.py:126,200`)
with the effect-constructors added to `SandboxConfig.extra_globals` (`policy_utils.py:77`) and `view`
as the sole context var.

**PERSISTENCE — versioned Code-of-Laws file store (behind the tools).** The active institutions +
append-only journal serialize to a versioned store (one JSON/`.py` per law version + a HEAD manifest;
git-like commit objects: parent, author="regent", period, cost). This gives reproducibility, audit,
diff-between-regents, replayable rollouts, and a human seed/override path — and it is the "code of
laws" genome an evolutionary outer loop reads via `amend`/`set_params`. **Reject a watched directory
as the live input channel:** files lose atomic commit, attribution, pre-commit mediation, and clean
rollback. Files are the *output* of `propose`, never the input.

**On the owner's point that security-sandboxing is now obsolete but economic mediation is not.** We
keep RestrictedPython, but **not** for host security (host trust is granted per the brief). We keep it
because it is precisely the mechanism of **Wall 1**: it denies the institution body any name that
mutates a balance, and keeps bodies pure and `clone()`-able for `rollout()`. A raw policy-file with
arbitrary code would re-grant a balance handle and break conservation *and* rollout — so code is
allowed, but only **effect-API-constrained sandboxed** code, never a free file. The sandbox's job
moved from "block `import os`" to "block `firm.monetary_holdings`."

**Fallback for non-tool models** (the Gemini-flash path, `config.py:87`): a text-protocol shim — the
model emits the same structured JSON list of `{"tool":..., "args":...}` inside the strict-JSON
envelope the current prompt already demands; a thin parser turns each into a Chancery call. Identical
mediation, identical conservation. Today's single-expression regent is recovered as the degenerate
`Define` of a one-`SetRate` institution — backward compatible with `-0.9*current_x`.

---

## 9. How it extends doc-06 and sits on doc-04

**On doc-06 (WorldSpec / Control / Observable / clone / rollout) — strict superset, one supersession.**
- `Observable` is the read side of `EconomyView`: `view.observable(name)` reads the same
  `get_current_metrics()` projection (doc-06 §2.2). The single-source-of-truth (whitelist == metrics
  keys == context vars) extends to: whitelist == view accessors == effect target names. An effect
  naming an undeclared account/param is rejected, exactly as a doc-06 expression referencing an
  undeclared observable can't validate.
- `Control` is *generalized*: a `Control(name, get, set, value_range)` is the degenerate institution
  emitting one `SetRate`. For `SetRate`, doc-06's `c.set(self, v)` writes `policy_params` (a
  non-account dict) — fine. For anything touching a balance, doc-06's direct `setattr` is **replaced**
  by routing through the Ledger: doc-06's `_apply_active_policies` becomes
  `effects = CodeOfLaws.fire(step, view); Mediator.commit(effects)`. The world author still writes
  only `_advance()` (doc-06's one author method).
- **WorldSpec gains a capability/effect declaration**: alongside `observables`/`controls`, a world
  declares `accounts`, `goods`, settable `policy_params`, the `ExternalSink` set, and a
  `GovernanceSpec` (cost schedule, legislative budget, noise model, enabled Effect verbs, starter
  institution templates). All data.
- `clone()`/`rollout()`/`emulate_policy` are reused; we additionally deepcopy the `CodeOfLaws`,
  `Mediator` queues, and `Ledger`. Supersession: doc-06 wraps Mandel *without editing it* ("untouched
  native step"); the full conservation guarantee requires routing Mandel's settlements through the
  Ledger (doc-04 §A.2), so this design **requires** editing `mandel_test.py` for the *full* guarantee
  — until then you get the guarantee for the regent's *own* effects only (§10 downside 1).

**On doc-04 (SFC / money conservation) — operationalizes it.**
- The Ledger **is** doc-04 §A.2's `settle(payer, payee, amount, good, qty)` made executable: cash +
  goods move together atomically (Transfer with `good`), replacing the in-place mutations at
  `mandel_test.py:948-955, 985-992, 1044-1049`. `Mint`/`Burn` vs a named sink **is** doc-04 §A.3's
  "money is created only by the bank as offsetting asset+liability."
- doc-04 §A.3's per-step `M_total` assertion **is** the engine's per-step global check over inside ∪
  external — kept, not demoted, because it catches the value/off-ledger leaks of §3. The four
  documented Mandel leaks map 1:1 to `Mint`/`Burn` channels: `max(0,…)` destruction (`:387`) →
  `Burn(... , reason)`; `gov_debt_limit` creation (`:658-659`) →
  `Mint("treasury", x, "CENTRAL_BANK_RESERVE", "gov deficit")`; firm-removal deletion (`:800, :836`)
  → a `Mint`/`Burn` of the residual on orderly exit; entry endowment (`:862`) →
  `Mint(new_firm, seed, "EXOGENOUS_ENDOWMENT")`. doc-04 §H ("no exogenous money on entry") is
  enforced because the only injection path is a logged, sinked `Mint`.
- This is **additive** to doc-04's stabilization program (the governance layer does not fix Mandel's
  behavioral divergence — that's doc-04 §B-H), but it guarantees the **regent** cannot be a *new*
  source of leaks and gives doc-04's required global assertion a structural home. Per doc-04 Stage 8,
  the regent is reintroduced on top of the stabilized SFC core; this interface is the contract for
  that reintroduction.

---

## 10. Staged adoption path, downsides, and open questions

**Staged adoption — from today's single expression, always-green.** Mirrors doc-04's staging (the
regent re-enters at Stage 8) and doc-06's incremental migration.

0. **Today.** Regent returns one `Policy` whose expression is eval'd to set one variable
   (`linear_stochastic_system.py:217`, `single_market_model.py:261`).
1. **Land the kernel, independently.** Add `govsim/gov/{effects,ledger}.py` (the conserved Effect
   types + private `_post` with the non-negativity invariant) + the doc-04 global `M_total`
   assertion. On its own this delivers req-1 for the money subset and yields the journal req-2/req-5
   ride on. **Do this on ONE ledger-bearing world**, not the toy worlds (which have no balances).
2. **Effect-API sandbox change.** Generalize `policy_utils` from "return a scalar the world assigns"
   to "return a `List[Effect]` the engine posts," with `view` as the sole context var (Wall 1). The
   linear `u_k` and the 5 SMM levers become one-`SetRate`/one-`Allocate` institutions.
3. **Add the outer-loop tools** (`study` / `rollout` / `propose`) + structured rejection; wire
   `rollout()` to doc-06 `clone()`. Re-tune via `amend`. Now the regent studies before it acts.
4. **Add the Mediator** (cost → rate-limit → control-error). Configure per-experiment via
   `GovernanceSpec`. The 5 SMM policies migrate as the proof on the existing stress test.
5. **Declarative DSL + Code of Laws store** for multi-effect/novel institutions (progressive tax,
   cap-and-trade). Defer until an experiment names them (YAGNI before then).
6. **SFC-refactor Mandel** (doc-04 §A) so the *whole* economy — not just regent effects — is
   conserved; reintroduce the regent on the stabilized core (doc-04 Stage 8).

**Honest downsides and where conservation can still leak.**

1. **The guarantee is LOCAL until doc-04 lands.** Steps 1-5 conserve the regent's *own* effects and
   quarantine the world's native leaks; the world (Mandel) still destroys/creates money at
   `:387/:658-659/:800/:862` and mints/burns goods every period (`Firm.produce`,
   `Household.consume_goods`). Whole-economy conservation needs the §A refactor (step 6). Do not
   oversell "the model is conserved"; claim "the regent's effects are conserved, and the world's
   flows become conserved as doc-04 lands."
2. **`SetRate` / valuation / negative-balance leaks (§3).** A published rate drives off-ledger
   value; `Transfer` conserves counts not value; an account driven negative can be laundered by a
   world clamp. Mitigations carried here: `_post` non-negativity invariant, every world mint/burn
   channel typed-and-logged, and the doc-04 global assertion as the catch-all. These reduce, not
   eliminate, the value leak — which is why the assertion is non-negotiable.
3. **Expressiveness ceiling — endogenous bank credit is the native case it strains.** Mandel's
   `provide_loan_to_firm` does `firm.monetary_holdings += amount; firm.debt += amount`
   (`mandel_test.py:694-695`): money created as a matched asset+liability, the canonical SFC move.
   The six primitives express this only awkwardly — `Transfer` needs an existing source (the bank has
   none to lend) and `Mint` is against an *external* sink, not an *inside* interest-bearing liability.
   **Resolution:** add a **seventh primitive `Credit(lender, borrower, amount, rate)`** that posts a
   matched inside asset+liability pair (and its repayment/write-down), with its own conservation
   sub-invariant — rather than abusing `Mint` (loans are routine and internal, not exogenous and
   conspicuous). This is the one place "five total verbs" was too few; it is additive, not a redesign.
   Equally out of reach by design: institutions that **rewrite an agent's behavioral rule** (a new
   Taylor reaction function, credit-rationing predicate, markup ceiling) rather than move a stock —
   the closed vocabulary deliberately excludes rule-override (doc-06 §4 "semantic corruption"); such
   needs are a separately-budgeted capability or a world-declared lever, not a regent institution.
4. **Performance.** `clone()`/`rollout()` deepcopies world + ledger + code-of-laws + queues per
   counterfactual; for Mandel (hundreds of agents) this is slow (doc-06 §4). Use few seeds / shallow
   horizons / checkpoints; free on the toy worlds.
5. **Over-engineering risk if mis-sequenced.** The full kernel + Mediator + DSL + versioned store is
   dead weight on the toy worlds (no balances) and premature on an unstable Mandel (doc-04 is Stage 0
   of 8). The staged path above builds only the kernel + effect-API change + tools first, on one
   ledger-bearing world, and defers the DSL/store/noise until an experiment demands them.
6. **LLM-reliability limits.** The declarative + tool + rejection + rollout path is *more* reliable
   than today's single-expression interface for the common 80% (template-filling, strong fiscal
   priors, loud recoverable failures). The **code escape hatch** — where the novel institutions the
   owner actually wants live — is the high-failure surface: models emit non-`List[Effect]` returns,
   reference disallowed names, or call view methods/observables that don't exist verbatim. Mitigations:
   bias hard to declarative; **dry-run every code body on a `clone()` at registration** and reject
   with the precise compile message before activation; ship 6-8 vetted templates; `rollout()` surfaces
   bad-but-conserved policies before enactment. Net: a wrong institution is *suboptimal policy the
   model can see and revise*, never a broken ledger — every conservation property holds regardless of
   what the model emits.

**Open questions.** (a) Where exactly to draw the rule-override line — which behavioral rules become
world-declared levers (`SetRate`-able reaction-function coefficients) vs. forbidden? (b) Whether the
`Credit` primitive's repayment/default/write-down cascade needs a stateful sub-ledger (it likely
does). (c) Float-tolerance drift in `_post` over thousands of tiny Allocate fan-outs — needs an
integral re-check, not just per-batch `EPS`. (d) How aggressively to mandate rollout-before-enact in
the harness without making the loop too slow on Mandel. (e) Calibrating the cost schedule /
legislative budget / noise model so the task is neither trivial nor impossible — itself an
experiment variable (doc-04 §3 sweep discipline).
