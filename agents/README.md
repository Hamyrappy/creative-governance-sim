# `agents/` — analysis, plans, and shared context for developer agents

This folder is the working memory for evolving `creative-governance-sim` from a defended thesis
prototype into a research platform for **regent systems** (intelligent control systems that govern
simulated economies). It was produced after a full read of the repository, the author's materials
folder, the defended thesis, and the advisor's review.

Start with the root [`AGENTS.md`](../AGENTS.md) for repo orientation and how to run things, then
read these in order:

| # | Doc | Purpose | Depends on |
|---|---|---|---|
| 01 | [`01-landscape-and-positioning.md`](01-landscape-and-positioning.md) | Engineering brief on the 2025–2026 field (self-improving LLM optimizers, LLMs simulating humans, LLMs running businesses, AI for economic policy); how this project relates and what to borrow | verified web research |
| 02 | [`02-code-audit.md`](02-code-audit.md) | Line-referenced inventory of bugs, smells, and what to keep | code read |
| 03 | [`03-refactor-plan.md`](03-refactor-plan.md) | Target architecture, package layout, interface redesign, incremental migration | code read + audit |
| 04 | [`04-mandel-stabilization.md`](04-mandel-stabilization.md) | Diagnosis of why the Mandel/Lagom ABM diverges and a concrete plan to stabilize and integrate it | research + Mandel papers |
| 05 | [`05-roadmap.md`](05-roadmap.md) | Phased roadmap from prototype to platform, tying together audit, refactor, stabilization, and landscape | all of the above |
| 06 | [`06-model-integration.md`](06-model-integration.md) | The recommended near-automatic model registration/integration/wrapping design (declarative `WorldSpec` + `__init_subclass__` registration + `clone()`/`rollout()`); **supersedes the registration/context specifics in 03** | design panel + code |
| 07 | [`07-governance-interface.md`](07-governance-interface.md) | **The governance/action model**: how the regent governs safely — conserved Effects + double-entry Ledger + Mediator (cost/rate-limit/control-noise) + Institutions (declarative or effect-API-constrained code) via tool-calling; generalizes the scalar `Control` of 06 | design panel + code |
| 08 | [`08-open-problems-and-opportunities.md`](08-open-problems-and-opportunities.md) | **Course-correction review (read this FIRST)** — red team of 01–07: the HOW-before-WHAT meta-problem, critical weaknesses, a decide-from-the-start table, dead-ends to cut, and a revised prioritization that **supersedes 05** | red team |
| 09 | [`09-grand-plan.md`](09-grand-plan.md) | **AUTHORITATIVE architecture + implementation plan.** The experiment machine for LLM regents controlling complex systems (economy = domain #1); `ActionInterface` as the only domain seam; multi-regent + harness-as-research; phased rewrite → test-driven growth. **Supersedes 05; repositions 06/07 as the economy plugin.** | design panel + code |

## How to use this folder

- **Treat these as living documents.** When you complete a migration step or stabilize a model,
  update the relevant doc (and the status notes in `AGENTS.md`) so the next agent starts from truth.
- **02 and 03 are the immediate work plan.** They translate directly into PRs.
- **01 and 05 are the strategy.** Use them to decide *what* to build next and how to frame the
  planned "regent systems" article.
- **04 is the first big research milestone** — getting a real agent-based economy stable enough to
  hand to a regent (the thesis's explicit unfinished goal).

## Conventions for docs added here

- English, markdown, with relative links between docs and into the codebase (`govsim/...:line`).
- Keep claims grounded: cite source URLs for external facts; cite `file:line` for code claims.
- Prefer short, scannable structure (tables, ordered fix-lists) over prose.
