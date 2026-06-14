# STATUS — WHAT-first gates & phase progress

> These are the **executable preconditions** of doc-09 §1.3: the `Runner` refuses to run an
> `Experiment` without a `Hypothesis` (claim + named baseline). The docs in this folder are the
> human side of that gate — they record the *scientific* decisions (doc-08 §8) that the code
> cannot invent. **Items marked ☐ AUTHOR need the author's sign-off before they enter the article.**

## Gate docs

| Doc | Purpose | Status |
|---|---|---|
| [`hypotheses.md`](hypotheses.md) | the falsifiable claims + named baselines (H1–H8) | DRAFT — defaults proposed, ☐ AUTHOR to confirm the primary claim |
| [`objectives.md`](objectives.md) | the named objective set; "which objective" is an experimental variable | DRAFT — ☐ AUTHOR to pick the committed set |
| [`creativity-metric.md`](creativity-metric.md) | domain-scoped creativity construct (or drop the word) | DRAFT — ☐ AUTHOR to confirm or drop |
| [`stats-protocol.md`](stats-protocol.md) | paired shared-seed design, bootstrap CI, pre-registration | DRAFT — ready to adopt |
| [`decisions.md`](decisions.md) | ADR log (supersession chain + deviations) | LIVING |

## Phase progress (doc-09 §7)

| Phase | What | Status |
|---|---|---|
| 0 | thin core + data spine (6 seams, LLM cache/replay, scalar plugin, cubic migration, Runner, ResultStore, CLI, golden-master, guard tests) | **DONE** |
| 1 | `LLMRegent` + prompt assembler + PID/LQR/OPRO baselines; `StabilizationLoss`; coupled system | **DONE** — OPRO + `CoupledSystem` landed; obfuscated partial-info prompt + `check_prompt` boot validator landed |
| 2 | rollout-free harness (`TraceFeedback`, `EpisodicMemory`); then `RolloutProbe`; H1 result | **machinery DONE** — `RolloutProbe` + `core/rollout.py` (RollableSystem-gated) + `Critic` + `govsim/analysis` (paired bootstrap CI, variance-aware select, collapse detector). The H1 *result* (a full multi-seed run + the author's WHAT sign-off) is the remaining science step |
| 3 | rung-1.5 non-economy (SIR, company) on the SAME core | systems + experiments landed |
| 4 | rung-2 SFC economy + `EconomyActionInterface` (doc-07 kernel) | not started (gated on rung-1 H1) |
| 5 | multi-regent (N=2, jurisdictions, comms, turn protocol) | hooks threaded; game logic not started |
| 6 | adopt EconAgent / Mandel (gated, optional) | not started |

> **LLM endpoint verified (2026-06-15):** the OpenAI-compatible stack was exercised end-to-end against
> a live vLLM cluster (`Openai/Gpt-oss-120b`, tool-calling) for `cubic_nonlinear_llm`,
> `cubic_nonlinear_llm_obfuscated`, and `cubic_nonlinear_llm_critic`; **replay mode reproduced a run
> byte-for-byte with a deliberately-wrong key** — the cache/replay reproducibility guarantee holds live.
> Running the *headline H1 comparison* (harnessed LLM vs `cubic_nonlinear_opro` + frozen LQR, ≥20 seeds,
> `govsim compare`) is the author's remaining call (seed count + objective sign-off — see the ☐ AUTHOR
> items below).

## What "Phase 0 DONE" means here (the doc-09 §7.1 gate)

- ✅ green deterministic CI **without an API key** (`uv run pytest`)
- ✅ `python -m govsim run <experiment>` works; reproduces a fixed trajectory from `(spec, seed)`
- ✅ adding a world/regent/experiment is one file + one registration
- ✅ per-system `np.random.Generator`; bare-RNG banned by a test; clone/rollout faithful
- ✅ no domain noun in `govsim/core` (leakage test); LLM cache/replay tape; reject-with-feedback
