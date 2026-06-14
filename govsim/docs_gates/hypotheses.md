# Hypotheses — the falsifiable spine (the WHAT-first gate)

> Each `Experiment` carries a `Hypothesis(claim, baseline, primary_metric)`; the `Runner` will not
> run without one (doc-09 §1.3). This file is the catalogue (doc-09 §6.1). **Defaults are proposed
> from the plan; ☐ AUTHOR must confirm the *primary* claim of the first paper before a live run.**

A hypothesis is `claim + NAMED baseline + falsification test` — never a bare deliverable
(doc-08 §3.1). The headline contrast is **adaptation to the unknown**, where LQR/PID cannot be
precomputed; "beat OPRO on the stationary linear toy" is a likely null and is scoped to *matches*.

| ID | Claim | Named baseline | First rung | Status |
|---|---|---|---|---|
| **H0** | a proportional code-as-policy regent **matches** LQR/OPRO on the *linear* plant | LQR ground-truth + OPRO; no-control floor | cubic (linear) | wired (`cubic_stabilization`) |
| **H1** ★ | a code-as-policy regent + trace recovers from an **unseen structural shock** in lower post-shock regret than (a) the frozen pre-shock-optimal controller and (b) trace-less OPRO | frozen LQR/numeric-DP + trace-less OPRO | cubic (nonlinear, partial info) | wired (`cubic_nonlinear`); **needs live model run** |
| **H2** | experimentation > reasoning: rollout-probing beats pure reasoning | reasoning-only regent (no `RolloutProbe`) | cubic | pending (needs `RolloutProbe`) |
| **H3** | each harness component yields a **separable measurable gain** | full stack minus the component | cubic/SIR | pending (ablation harness ready) |
| **H4** | objective → institution mapping: same world, different objective ⇒ different policy | fixed-objective regent | SFC economy | pending (rung 2) |
| **H5** | specification-gaming: a characterized proxy-up / true-welfare-down episode | held-out true welfare | SFC economy | pending (rung 2) |
| **H6** | Lucas: governance results change under naive → policy-aware adaptive agents | naive-agent run | SFC / adopted ABM | pending |
| **H7** | multi-regent collusion (joint-obj↑ while held-out social-obj↓) | independent (no-comms) regents | multi-polity | hooks threaded (Phase 5) |
| **H8** | longer planning-horizon → richer economy | short-horizon regent | adopted ABM | pending |

★ = **the first publishable result** (doc-09 §6.4): reachable on the cubic *before any economy code
exists*. The advantage claim lives **only** in the non-stationary / nonlinear / partial-info regime.

## ☐ AUTHOR decisions still open (doc-08 §8)
1. Is **H1** the primary claim of paper 1? (recommended: yes.)
2. The exact "unseen structural shock" for the cubic arm (regime change in A/B, or a new
   nonlinearity at a known step) — pick one and pin it in `cubic_nonlinear`.
3. Are H4/H5 paper 2, or folded into paper 1?
