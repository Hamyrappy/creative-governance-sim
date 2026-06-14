# ABM Stabilization Playbook — Mandel(2012)/Lagom-style Agent-Based Economy

A field-grounded, code-specific guide to stop divergence (prices/GDP blow-up, universal
bankruptcy) in the model in `mandel testing/mandel_test.py`. Every recommendation is tied to a
verified source and, where possible, to the exact line in the code that violates it.

> Provenance: written from a verified research pass (the ABM-stabilization cluster of the landscape
> study, see [`01-landscape-and-positioning.md`](01-landscape-and-positioning.md)) **plus a direct
> read of `mandel_test.py`**. This is the concrete plan for the thesis's explicit unfinished goal —
> getting a real agent-based economy stable enough to hand to a regent. It pairs with the
> modularization in [`03-refactor-plan.md`](03-refactor-plan.md) (step 7) and the line-level
> findings in [`02-code-audit.md`](02-code-audit.md).

**Code facts confirmed by reading the model (the leaks the global invariant in §A will catch):**
- Firms buy capital/intermediary goods with **no monetary constraint** (lines 943, 980: "No
  monetary constraint for firms here") — a buyer's `monetary_holdings` can go negative, then
  `accounting()` covers any deficit with **unlimited new debt** (lines 373–376). This is the
  runaway-debt loop.
- `accounting()` does `self.monetary_holdings = max(0, …)` (line 387) — this **destroys money**
  silently (a negative balance is zeroed with no offsetting entry). A literal accounting leak.
- Household tax/benefit flows pass through `government.monetary_holdings`, but the `gov_debt_limit`
  clamp (lines 658–659) **creates money** when the government hits its floor.
- Firm exit just does `self.firms.remove(firm)` (lines 800, 836) — deleting that firm's debt and
  cash from the system; firm **entry injects exogenous money** (`new_firm_start_money_base=7500`,
  line 862). Both break conservation.
- There is **no global money-conservation assertion anywhere** in `run_period` (confirmed by
  reading the full loop).

The central reframe from the literature: **collapse is almost never "a bug to clamp away." It is
either (a) money silently leaking because there is no accounting closure, or (b) the model sitting
in a "bad" phase of its parameter space.** The fixes are structural — accounting discipline,
bounded feedback, steady-state initialization, an automatic stabilizer, calibration — not cosmetic
`np.clip` calls. (Stock-flow consistency: Godley & Lavoie via Nikiforos & Zezza,
https://www.levyinstitute.org/pubs/wp_891.pdf ; phase-transition view: Gualdi/Tarzia/Zamponi/Bouchaud,
https://arxiv.org/abs/1307.5319 .)

---

## 1. Top root causes of divergence in this model family (with symptoms)

| # | Root cause | Symptom it produces | Where it lives in the code |
|---|-----------|---------------------|------------------------------|
| 1 | **No accounting closure / money leaks (no SFC).** No global money-conservation or balance-sheet check exists anywhere. | Aggregate money silently drifts; balances reach extremes; "collapse" with no localizable cause. | Absent across the whole `run_period`. `accounting()` does `monetary_holdings = max(0, …)` (line 387) → **destroys** negative balances; `Government.set_tax_rate_and_collect_taxes` clamps to `gov_debt_limit` (lines 658–659) → **creates** money. (Nikiforos & Zezza, https://www.levyinstitute.org/pubs/wp_891.pdf ; Eurace dynamic balance sheets, https://yildizoglu.fr/macroabm1/docs/vanderHoog-Eurace.pdf ) |
| 2 | **Unbounded credit/debt loop.** Firms buy capital and intermediates with *no monetary constraint*, then any cash deficit becomes *unlimited new debt*. | Runaway debt → interest snowball → bankruptcy cascade → all firms exit → collapse. SMC of K+S independently finds credit/financial params dominate macro dynamics. | Lines 943 & 980 ("No monetary constraint for firms here"); `accounting()` lines 373–376 ("Cover deficit with debt") borrows `-monetary_holdings` with no per-step cap (the `max_debt_to_capital_ratio=3.0` gate only guards *planned* loans in `provide_loan_to_firm`, not the deficit-borrowing path). (Blando et al., SMC of K+S, https://arxiv.org/abs/2605.10447 ; JAMEL deleveraging, https://hal.science/hal-01110642v2/preview/seppecher-salle-2014-deleveraging-crises-and-deep-recessions.pdf ) |
| 3 | **Adjustment-speed asymmetries → wrong phase.** Collapse is generically induced by an asymmetry between hiring and firing (and any asymmetric adjustment speed) crossing a critical threshold. | Unemployment low until a tipping point, then "the economy suddenly collapses"; regime oscillation near the boundary. | Hand-set asymmetric speeds: `price_update_smoothing=0.25`, `expected_sales_update_rate=0.1`, `dividend_rate=0.3`, debt-repayment fraction `0.5` (line 379); hiring is greedy/sorted but firing is implicit via full re-hire each period (lines 1053–1081). (Gualdi et al., https://arxiv.org/abs/1307.5319 ; JEDC 50 (2015) 29–61, https://www.sciencedirect.com/science/article/abs/pii/S0165188914001924 ) |
| 4 | **Over-parameterization.** ~40 hand-tuned magic constants in `Firm.__init__` alone. Fragility comes from *structural switching mechanisms*, not parameter count; more constants buy fragility and non-identifiability, not realism. | Brittleness; tiny constant changes flip stable↔divergent; impossible to attribute collapse. | `Firm.__init__` lines 187–239; `Simulation.__init__` lines 765–786 (`new_firm_start_money_base=7500.0`, `bankruptcy_cash_threshold=-1000.0`). (Kukacka & Kristoufek, https://www.sciencedirect.com/science/article/abs/pii/S0165188920300257 ) |
| 5 | **Arbitrary, non-steady-state initialization.** Initial stocks/flows are magic constants (`monetary_holdings=700`, `debt=20`, `price≈1.0`, `expected_income=25`) not derived from a consistent stationary state. | First steps blow up because the economy starts far from any fixed point; transients never die. | `Household.__init__` lines 129–135; `Firm.__init__` lines 195–219. (Caiani et al. 2016 six-step SFC initialization, https://business.columbia.edu/sites/default/files-efs/imce-uploads/Joseph_Stiglitz/Agent%20based-stock%20flow.pdf ) |
| 6 | **Unbounded price/markup/productivity feedback.** Markup evolves by imitation+mutation with only a floor (`max(0.01,…)`, line 508); price = `(1+markup)*unit_cost` feeds others' unit costs; productivity indexes wages. | Price/markup random-walk to extremes; cost-push spiral; nominal GDP blows up while real activity collapses. | `update_price` lines 303–314; `evolve_markup` lines 490–508 (no markup ceiling); productivity→wage indexing lines 606–607. (Mandel 2012, https://shs.hal.science/halshs-00732823 ) |
| 7 | **Excessive innovation/mutation vs selection.** `mutation_rate=0.05`, `mutation_strength=0.15`, `imitation_rate=0.3` applied multiplicatively every 5 periods. Mandel(2012) and SMC of the Island model find **moderate exploration is optimal**. | Firm population never settles; coefficients/markups random-walk; no convergence to the GE fixed point. | `evolve_technology` 461–488, `evolve_markup` 490–508, `evolve_wage_index` 532–546. (Mandel 2012, https://shs.hal.science/halshs-00732823 ; Blando et al. Island model, https://arxiv.org/abs/2604.04543 ) |
| 8 | **Order-of-update / simultaneity + disruptive entry/exit churn.** Trades mutate buyer and seller balances in place in shuffled sequential loops; combined with no-constraint purchases this lets a firm "spend" money it doesn't have. New firms injected with huge resources shock the system; bankruptcy thresholds very loose. | Phantom money / double-spend within a step; jumps on entry/exit; `num_firms` oscillation. | Trading loops 919–1049 (in-place mutation, no atomic settlement, no solvency precheck); `manage_firm_entry_exit_bankruptcy` 791–887. (Eurace explicit update order, https://link.springer.com/chapter/10.1007/978-3-642-14746-3_74 ; JAMEL orderly exit, http://p.seppecher.free.fr/jamel/ ) |

---

## 2. Prioritized fix checklist

Apply **top to bottom**. Earlier fixes turn silent divergence into loud, locatable failures,
which makes the later fixes diagnosable.

### A. Stock-flow consistency + conservation invariants with runtime assertions — *do this first*
The single highest-value fix. (Godley & Lavoie via Nikiforos & Zezza,
https://www.levyinstitute.org/pubs/wp_891.pdf ; AB-SFC integration, Caiani et al.,
https://www.sciencedirect.com/science/article/abs/pii/S0165188915301020 )

1. **Write the two matrices before touching code:** a balance-sheet matrix (stocks; every
   financial asset is someone's liability so each row nets to zero) and a transactions-flow matrix
   (flows: `+` inflow, `−` outflow; each sector's column = its budget constraint). Sectors:
   households, firms, government, bank/financial system.
2. **Make every transaction quadruple-entry:** each payment debits one agent and credits another
   (and moves the real good the opposite way). Wrap settlement in one helper
   `settle(payer, payee, amount, good=None, qty=None)` so cash + goods always move together
   atomically — replace the in-place balance mutations at lines 948–955, 985–992, 1044–1049.
3. **Add a per-step conservation assertion.** Compute
   `M_total = Σ hh.monetary_holdings + Σ hh.savings + Σ firm.monetary_holdings − Σ firm.debt + gov.monetary_holdings + bank_position`
   and assert it equals the prior step's value up to float tolerance (money is created only by the
   bank as offsetting asset+liability). The step where it breaks localizes the leak. **This will
   immediately flag** the `max(0, monetary_holdings)` money-destruction (line 387) and the
   `gov_debt_limit` money-creation (lines 658–659).
4. **Drop exactly one redundant identity** ("the redundant equality") to avoid overdetermination.
   (Nikiforos & Zezza WP891 p.15.)
5. **Per-agent identity unit tests:** assert each firm's
   `Δmonetary_holdings == sales_revenue − costs − dividends + new_debt − repayment`; each
   household's `Δ == wages + interest + dividends − consumption − taxes + benefits`.

> SFC is necessary but **not sufficient** — an SFC model can still diverge from explosive
> behavioral rules. That is what B–H address.

### B. Normalization / dimensionless variables, and radical parameter reduction
- Pick a numéraire (fix the labor unit or set an aggregate price index = 1); express prices, wages,
  debt as ratios. Removes the nominal blow-up channel and makes clamps interpretable.
- **Cut/normalize the ~40 free constants.** Keep only the few *structural* levers (imitation/mutation
  strength, adjustment speeds, credit limit, dividend rate); fix or normalize the rest. Fewer
  parameters = a far smaller space to search for the stable phase and much better identifiability.
  (Kukacka & Kristoufek, https://library.utia.cas.cz/separaty/2020/E/kukacka-0522039.pdf )

### C. Economically-justified clamps (replace arbitrary `max/clip` with real constraints)
- **Credit constraint, not unlimited deficit-borrowing.** Remove the "no monetary constraint" firm
  purchases (lines 943, 980): a firm can only buy what cash + an *approved, capped* loan covers.
  Route **all** borrowing — including deficit cover in `accounting()` (lines 373–376) — through
  `provide_loan_to_firm` with the `max_debt_to_capital_ratio` gate, and add **credit rationing**
  (the bank can refuse). A firm that can't cover costs should cut output/employment, not silently
  borrow.
- **Markup ceiling** (currently only a floor, line 508) and a **price band** justified by competition.
- **Buffer-stock / inventory target** already exists (`inventory_to_sales_ratio_target=0.2`) — keep
  it, but make dividend and investment rules *unable to drain the firm below operating cash*
  (Deaton-style buffer for firms, mirroring the household Deaton rule at lines 168–182). (K+S
  buffer/automatic-stabilizer logic, https://www.iris.sssup.it/bitstream/11382/302310/1/JEDC_2010.pdf ;
  Project Vend's discount death-spiral shows what unconstrained "give it away" rules do,
  https://www.anthropic.com/research/project-vend-1 )

### D. Slow + symmetric adjustment speeds
- Lower the dangerous speeds and make hiring/firing **symmetric** in responsiveness — the
  tipping-points result says hiring/firing asymmetry is *the* generic collapse driver. Treat
  `price_update_smoothing`, `expected_sales_update_rate`, the `0.5` debt-repayment fraction, and the
  implicit hire/fire speed as adjustment-speed parameters to sweep (§3). (Gualdi et al.,
  https://arxiv.org/abs/1307.5319 )
- Lower `mutation_strength`/`mutation_rate` and `imitation_rate` toward "moderate exploration."
  (Mandel 2012, https://shs.hal.science/halshs-00732823 )

### E. Buffer stocks / credit constraints / automatic stabilizer
- You already have an unemployment-benefit + tax government (lines 616–660). **Make it a working
  automatic stabilizer with sane bounds** — K+S shows demand-side policy (benefits/public spending)
  is often exactly what keeps the economy out of the high-unemployment/collapse trap. This is *also*
  the natural lever for the LLM regent. (Dosi/Fagiolo/Roventini,
  https://www.iris.sssup.it/bitstream/11382/302310/1/JEDC_2010.pdf )

### F. Start at/near a steady state, then perturb
- Use the **Caiani six-step procedure**: derive an aggregate version of the model, constrain it to a
  real stationary state, solve numerically by fixing empirically-meaningful parameters
  (unemployment, mark-ups, interest, tax rates), then distribute the resulting consistent
  stocks/flows across agents. Replace the magic constants at `Household.__init__` (129–135) and
  `Firm.__init__` (195–219). (Caiani et al. 2016, §4,
  https://business.columbia.edu/sites/default/files-efs/imce-uploads/Joseph_Stiglitz/Agent%20based-stock%20flow.pdf )
- **Discard a burn-in transient** before measuring anything.

### G. Deterministic seeding
- Seed both `random` and `numpy.random` from a single run seed; thread it through `Simulation`. The
  code currently calls `random.*`/`np.random.*` with no global seed, so runs aren't reproducible —
  mandatory before any multi-seed validation or debugging. (Variance-control discipline: Gideoni et
  al., https://arxiv.org/abs/2602.16805 ; Vending-Bench tail events, https://arxiv.org/abs/2502.15840 )

### H. Order-of-update / simultaneity correctness
- Adopt an **explicit, documented update order** with **staged/synchronous settlement**: decisions
  read the *start-of-step* state; balances change only through the atomic `settle()` helper; no agent
  can spend money it doesn't hold within a step. (Eurace, https://link.springer.com/chapter/10.1007/978-3-642-14746-3_74 )
- **Tame entry/exit churn:** lower `new_firm_start_money_base` (7500 is a large exogenous money
  injection — an SFC leak unless funded), and make bankruptcy a proper orderly exit (settle debts to
  the bank, distribute residual) rather than just `self.firms.remove(firm)` (lines 800, 836), which
  deletes the firm's debt and cash from the system — another leak the §A assertion will catch.

---

## 3. Minimal validation protocol

Run these as code/tests; report **ensembles, not single runs**.

1. **Identity unit tests (cheap, every step in debug mode).** The per-agent and global conservation
   assertions from §A. A model that "blows up" is non-stationary by construction, so these fire
   immediately at the leak site. (Fagiolo et al. validation review,
   https://link.springer.com/chapter/10.1007/978-3-319-70766-2_31 )
2. **Steady-state hold test.** Initialize from a hand-computed stationary point and run with noise =
   0. If the model can't even *hold* a fixed point, the dynamics are explosive (not the noise) — fix
   the dynamics before adding stochasticity. The known GE fixed point of the no-capital Mandel
   economy is your ground truth. (Mandel 2012, https://shs.hal.science/halshs-00732823 ; "recover the
   analytic optimum in the tractable case" discipline, AI Economist,
   https://www.science.org/doi/10.1126/sciadv.abk2607 )
3. **Monte-Carlo ensemble over seeds.** N≥20–50 seeds; discard burn-in; report **mean, variance,
   and worst-case/min** with CIs. Failures here are **high-variance tail events** — the median run
   can be fine while rare runs derail, so never trust a single seed. (Vending-Bench,
   https://arxiv.org/abs/2502.15840 ; in 2026 automated by Statistical Model Checking / MultiVeStA,
   which auto-sizes runs to a target precision/confidence, https://arxiv.org/abs/2605.10447 )
4. **Sensitivity sweep / phase map.** Sweep the key structural levers — **credit limit
   `max_debt_to_capital_ratio`, debt-repayment fraction, `price_update_smoothing`, hiring/firing
   speed, mutation/imitation strength** — and plot **survival-time (and stylized-fact stability) vs
   parameter** to locate the stable region; pin the economy *inside* the good phase with margin from
   the tipping boundary. Probe the credit/finance loop first. Accelerate with an ML surrogate
   (XGBoost/GP) if full runs are expensive. (Gualdi et al., https://arxiv.org/abs/1307.5319 ; Blando
   et al., https://arxiv.org/abs/2605.10447 ; surrogate calibration: Lamperti/Roventini/Sani,
   https://arxiv.org/pdf/1703.10639 ; Bayesian > frequentist: Platt 2020, https://arxiv.org/abs/1902.05938 )
5. **Target stylized facts.** Once it doesn't diverge, require it to reproduce, with CIs across
   seeds: a **negative Phillips curve** and **Okun's law** (EconAgent: Phillips r=−0.619, p<0.01;
   Okun r=−0.918, p<0.001, https://aclanthology.org/2024.acl-long.829/ ), plausible firm-size/growth
   distributions, and bounded inflation (roughly −5%…+5%). Tolerate **endogenous business cycles**
   ("never collapses to zero" is the target, not "perfectly smooth"). (K+S,
   https://www.iris.sssup.it/bitstream/11382/302310/1/JEDC_2010.pdf ; JAMEL, http://p.seppecher.free.fr/jamel/ )

---

## 4. Staged plan: smallest stable core first, then re-enable behind flags

Put every advanced feature behind a boolean flag (`enable_entry_exit`, `enable_genetics`,
`enable_capital`, `enable_multi_sector`, `enable_credit`, …), all **off** in the core. Re-enable
**one at a time**, re-running the §3 protocol after each; if a feature reintroduces divergence,
you've isolated the culprit. (Smallest-stable-core + ablation discipline, Gideoni et al.,
https://arxiv.org/abs/2602.16805 ; build no-capital → capital incrementally, Mandel 2012,
https://shs.hal.science/halshs-00732823 .)

- **Stage 0 — Smallest stable core.** 1 good, a handful of households + a few firms, **no
  entry/exit, no genetics, no capital accumulation, single sector.** Full SFC + conservation
  assertions on. Deterministic seed. Initialize at the analytic steady state. **Gate:** steady-state
  hold test (noise off) and a 20-seed Monte-Carlo with zero conservation-assertion failures and no
  divergence over the full horizon.
- **Stage 1 — Add stochastic demand/noise** around the steady state. Gate: stationary, ergodic,
  bounded fluctuations across seeds.
- **Stage 2 — Enable credit** (real credit constraint + rationing from §C, *not* unlimited
  deficit-borrowing). Highest-risk loop per SMC of K+S — validate the phase map for
  `max_debt_to_capital_ratio` here. Gate: no runaway-debt cascades.
- **Stage 3 — Enable the automatic stabilizer** (government benefits/taxes with bounds). Gate:
  stabilizer reduces variance and prevents the high-unemployment trap.
- **Stage 4 — Enable genetics** (imitation/mutation) at **moderate** strength. Gate: firm population
  *settles* near the fixed point; sweep mutation/imitation strength for the stable band.
- **Stage 5 — Enable capital accumulation + productivity indexing** (Mandel's flagged harder case).
  Gate: no nominal price/markup spiral.
- **Stage 6 — Enable orderly entry/exit/bankruptcy** (SFC-consistent: settle debts, no exogenous
  money on entry). Gate: `num_firms` stable, conservation holds across entry/exit.
- **Stage 7 — Scale up** (more goods, sectors, agents), re-validating stylized facts each step.
- **Stage 8 — Re-introduce the LLM regent** on top of the now-stable AB-SFC core. A properly-built
  AB-SFC model with an automatic stabilizer is the natural arena for studying whether an LLM
  controller can keep the economy out of the collapse phase, with 2026 SMC tooling giving the
  statistical machinery to claim "the policy helped" defensibly.

---

### Bottom line for the diagnosis
The fastest path to a stable economy: (1) add the global money-conservation assertion **today** — it
pinpoints the `max(0, monetary_holdings)` money-destruction (line 387), the `gov_debt_limit`
money-creation (lines 658–659), the firm-deletion debt/cash leaks (lines 800, 836), and the
exogenous money injected at firm entry (line 862); (2) close the **unlimited deficit-borrowing**
loop (lines 373–376, 943, 980), which the K+S SMC result and JAMEL both identify as the prime
collapse driver in this model family; (3) **start at a steady state** (Caiani) instead of magic
constants; (4) sweep the credit/adjustment-speed parameters to land in the stable phase (Gualdi
tipping-points). SFC + bounded credit + steady-state start will likely stop the collapse before you
ever need to retune the ~40 magic constants — and the literature says retuning them was never the
right lever anyway.

Relevant file: `mandel testing/mandel_test.py` (1411 lines). When migrating, modularize it into
`govsim/worlds/mandel/` behind `BaseWorld` per [`03-refactor-plan.md`](03-refactor-plan.md) §2.
