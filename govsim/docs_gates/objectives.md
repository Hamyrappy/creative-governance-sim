# Objectives — "which objective" is an experimental variable

> The chosen scalar **becomes the fitness** a selection/evolution harness optimizes, so it is a
> first-class scientific decision, not a config detail (doc-08 §3.1, doc-09 §6.2). Objectives are
> domain-bound plugins (`govsim/domains/*/objectives.py`); only the `Objective` ABC is in core.

Convention: `evaluate` returns a score where **higher is better** (a loss is returned negated), so
selection can `argmax` uniformly. `components(...)` always logs the raw sub-metrics — and, in the
economy domain, **distributional metrics (Gini, deciles) are first-class `components` even when not
in `evaluate`**, so a "proxy up / true-welfare down" Goodhart episode (H5) is detectable after the fact.

## Shipped (scalar domain)
- `StabilizationLoss` — dual-mandate control loss `MSE(x→target) + λ·MSU(u)`; generalizes the
  thesis loss and the LQR cost. (rungs 1 / 1.5)
- `EpidemicLoss` — cumulative infection-burden + λ·intervention-cost. (SIR)
- `CompanyProfit` — mean per-step profit. (company)

## Planned (economy domain, rung 2+)
growth/GDP · utilitarian SWF · inequality-weighted (Atkinson / iso-elastic, or `welfare − λ·Gini`) ·
fiscal-sustainability **constraint** · multi-objective / Pareto · `RobustWrapper(mean − λ·std)` over
seeds/shocks. Gini/deciles first-class everywhere.

## Guards that are NEVER folded into the objective (doc-08 §3.4 — category error)
- sample/compute **cost** (an engineering axis);
- **worst-of-N robustness** (`mean − λ·std`) — a *selection* rule, reported separately;
- the **held-out true-objective** vs the fitness proxy (the Goodhart divergence).

## ☐ AUTHOR decisions still open
1. The committed **named set** for paper 1 (recommended: utilitarian SWF + `welfare − λ·Gini` +
   output-gap/inflation loss, with Gini/deciles always logged).
2. Whether the *objective sweep itself* (H4) is paper 1 or paper 2.
3. `λ` for the dual mandate / robustness wrapper (pre-register, don't tune on the test seeds).
