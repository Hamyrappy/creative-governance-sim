# Roadmap — from thesis prototype to a "regent-systems" research platform

> ⚠️ **Superseded in priority by [`08-open-problems-and-opportunities.md`](08-open-problems-and-opportunities.md) §7.**
> A red team found this roadmap optimizes engineering before deciding the science (no falsifiable
> hypothesis / objective / creativity metric), keeps Mandel and the `07` Chancery on the critical path, and
> stakes the result on a likely-null "beat OPRO on a linear toy." Use `08` §7 as the operative plan (decide
> the WHAT first; minimal evaluator; keep the cubic; thread `polity_id`; defer Mandel + the Chancery). The
> phases below remain a useful component inventory, not the execution order.

This sequences everything in `agents/` into one plan: the code reorganization
([`03-refactor-plan.md`](03-refactor-plan.md)), the ABM stabilization
([`04-mandel-stabilization.md`](04-mandel-stabilization.md)), and the techniques to borrow
([`01-landscape-and-positioning.md`](01-landscape-and-positioning.md)) — aligned with what the
defended thesis itself named as future work and what the advisor's review asked for.

## What the thesis and review already told us to do

The defended ВКР explicitly lists future directions; the advisor's review echoes them. Mapped to
this roadmap:

| Stated next step (thesis / review) | Where it lands here |
|---|---|
| Integrate the regent into a **calibrated complex ABM** (stable Mandel) | Phase 2 |
| Add **real learning** (RL / fine-tuning / RAG), not just in-context | Phase 3 |
| **Expand the policy API** beyond parameter-setting (new institutions/rules) | Phase 4 |
| **Robustness / sensitivity / many-seed** experiments | Phases 1 & 3 (built into the eval harness) |
| **Multi-agent** (several governments / branches of power) | Phase 5 |

The landscape brief adds two things the 2025 thesis could not know: (a) the learning upgrade now
has a named, commoditized template (AlphaEvolve/OpenEvolve/ShinkaEvolve), and (b) "LLM governs an
economy" is now partly prior art (TaxAgent, SimCity), so the defensible novelty narrows to
**code-as-policy + evolutionary self-improvement against an economic evaluator** — which the
roadmap is built to deliver and demonstrate.

---

## Phase 0 — Clean foundation *(refactor steps 1–6; no new science)*
**Goal:** a reproducible, general, testable framework that does exactly what the prototype does
today, but cleanly.

- Seed all RNGs; typed pydantic config; `logging` over `print`; golden-master test of a thesis
  experiment. *(refactor §5 step 1)*
- Carve out `core/`; add the **single `AgentContext` contract** and the **`Objective`** abstraction;
  generalize the LLM regent so it governs *every* world, not just the linear ones. *(audit §3.6/§3.7;
  refactor §3.1–3.2, steps 2–3)*
- Unify policy-execution semantics (re-evaluate each step everywhere); resolve the cubic-term
  ambiguity explicitly. *(audit §3.11/§3.3; refactor step 4)*
- Real CLI + experiment registry; fix the README. *(audit §3.1; refactor step 6)*

**Exit criteria:** `python -m govsim run <experiment>` reproduces a thesis figure from
`(spec, seed)`; `pytest` green; adding a world/regent is one file + one `@register`.

## Phase 1 — Evaluator & baselines (the scientific spine) *(landscape P0)*
**Goal:** make every future "the regent is good" claim defensible before building anything fancy.

- `world.clone()` + `world.rollout(policy, horizon, seed) -> score` (the keystone; also implements
  the proposal's "prediction emulation"). *(audit §3.12; refactor §3.4, step 5)*
- **Multi-seed evaluation** returning mean / variance / worst-case (Vending-Bench's tail-event
  lesson). *(landscape P0.1)*
- **OPRO baseline** — sorted `(policy_code, score)` history, no archive: the mandatory thing to beat.
  *(landscape P0.2)*
- **LQR ground truth** on the (re-linearized) scalar system: prove near-optimality, not just
  stability. *(landscape P0.3)*

**Exit criteria:** a one-command benchmark reports, with CIs over seeds, the LLM regent vs OPRO vs
LQR on the linear worlds.

## Phase 2 — Stabilize a real economy *(the thesis's unfinished goal)*
**Goal:** a Mandel/Lagom-class agent-based economy that does not collapse and reproduces basic
stylized facts — the environment worth governing.

- Modularize `mandel_test.py` into `worlds/mandel/` behind `BaseWorld`. *(refactor §2, step 7)*
- Execute the **stabilization playbook**: SFC + per-step money-conservation assertion → bound the
  credit loop → steady-state initialization → phase-map sweep → automatic stabilizer, built up via
  the **staged feature flags** (Stage 0 → 8). *(all of [`04-mandel-stabilization.md`](04-mandel-stabilization.md))*

**Exit criteria:** Stage 0–3 pass their gates (conservation holds; 20+ seeds survive the full
horizon; Phillips + Okun reproduced with CIs).

## Phase 3 — Close the learning gap *(landscape P1; thesis "real learning")*
**Goal:** move from a single in-context policy to an evolving population scored by the economy —
the headline upgrade and the article's core method.

- Wrap the simulator as an **OpenEvolve/ShinkaEvolve-style evaluator** returning
  `{score, stability_flag, behavioral_features}`. *(landscape P1.4)*
- **MAP-Elites + islands** over control-meaningful behavioral axes (so a *diversity* of stable
  controllers survives instead of one fragile family). *(landscape P1.5)*
- **Artifact side-channel**: feed divergence traces / which-firm-went-bankrupt back into the next
  prompt — informed repair, not blind mutation. *(landscape P1.6)*
- Sample-efficiency (novelty rejection, adaptive parent sampling, bandit two-tier LLM routing) +
  **evaluation cascade**. *(landscape P1.7–P1.8)*
- Optional contrast arm: a small **RL regent** for the "code-as-policy vs RL-net" comparison
  (framed as a trade-off per the 2026 RL-in-economics survey). *(landscape Cluster D)*

**Exit criteria:** the evolutionary regent beats the OPRO baseline (Phase 1) on the linear worlds
*and* stabilizes the Phase-2 economy, with multi-seed CIs.

## Phase 4 — Richer policies & long-horizon safety *(landscape P2; thesis "expand policy API")*
**Goal:** policies beyond setting one scalar, plus coherence safeguards.

- Expand the policy API to structurally richer interventions (new `PolicyDescriptor`s: transfers,
  brackets, conditional rules — eventually institution-level rules). *(thesis future work; refactor
  keeps this a per-world concern)*
- **"Regent + auditor"** two-agent design: a critic validates emitted policy against objectives
  before compile/apply. *(landscape P2.9 — Project Vend's main fix)*
- Coherence instrumentation: Eureka-style reflection on rollout stats; checkpoint/partial-credit
  trajectory scoring; "steps-to-coherence-failure" as a headline metric; token cost in the objective.
  *(landscape P2.10–P2.11)*

**Exit criteria:** the regent operates richer policy types without coherence collapse over long
horizons; the auditor measurably reduces catastrophic policies.

## Phase 5 — Multi-actor & emergent behavior *(thesis "multi-agent")*
**Goal:** several governing actors — and study what emerges.

- Multiple regents (competing governments / branches of power / federations).
- **Collusion / deception / denial as first-class metrics** — a documented emergent behavior under
  optimization and a thesis-grade angle on "creative behavior of intelligent control systems."
  *(landscape P3.12 / Cluster C)*
- Validate comparative claims with **Statistical Model Checking** (MultiVeStA). *(landscape /
  stabilization §3.3)*

**Exit criteria:** reproducible multi-regent experiments with statistically defensible findings.

---

## Critical path & sequencing notes

- **Phases 0 → 1 are prerequisites for everything.** Without a general agent, reproducibility, and a
  trustworthy evaluator, later results aren't defensible (the "Simple Baselines" critique).
- **Phase 2 (stabilization) and Phase 3 (learning) can proceed partly in parallel** once Phase 1
  lands: Phase 3 can be developed against the *linear* worlds while Phase 2 stabilizes the ABM. They
  converge at Mandel Stage 8 (regent on a stable economy).
- **The publishable result** ("regent systems" article) is essentially **Phases 1–3**: an evolved,
  code-emitting regent that provably approaches the LQR optimum on tractable systems and keeps an
  agent-based economy out of its documented collapse phase — positioned per
  [`01-landscape-and-positioning.md`](01-landscape-and-positioning.md) §5. Phases 4–5 are the
  follow-on research program.

## Smallest useful next step (if starting tomorrow)

1. Seed the RNGs and add the golden-master test (Phase 0 step 1) — half a day, unblocks reproducibility.
2. Add the **global money-conservation assertion** to `mandel_test.py` (Phase 2 / stabilization §A.3)
   — it will immediately localize the known leaks (lines 387, 658–659, 800/836, 862) and is the
   single highest-leverage hour for the model that "never stabilized."
3. Generalize the LLM regent off `LinearSystemAgentContext` (Phase 0 step 3) — unlocks governing the
   other worlds and is the precondition for the evaluator harness.

Keep this file and the status notes in [`../AGENTS.md`](../AGENTS.md) updated as phases complete, so
the next contributor (human or agent) starts from truth.
