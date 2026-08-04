# Decisions log (ADR) — supersession chain + implementation deviations

> doc-08 §3.2 asked for one ADR file instead of supersession prose scattered across docs. This is it.
> Newest first. Each entry: **decision**, **why**, **status**.

## Doc supersession chain (authoritative)
- **05 (roadmap)** → superseded by **08 §7** (priority) and **09 §7** (execution).
- **06 (model integration)** / **07 (governance interface)** → repositioned by **09** as the
  *economy domain plugin* (Phase 4), not the top-level framework.
- **08** is authoritative on two reversals of 01/02: (1) **keep** the cubic + obfuscated prompt;
  (2) treat 01's citations as unverified leads.
- **09** is AUTHORITATIVE for architecture + implementation. This branch implements **Phase 0, Phase 1,
  and the Phase-2 harness machinery** of 09 (the H1 *result* — a multi-seed live comparison + author
  sign-off — is the remaining science step; see STATUS.md).

## ADR-0016 — Review-fix batch: post-shock metric, sandbox hardening, fair OPRO, stats floor (2026-07-07)
**Decision:** a batch of correctness/methodology fixes from a full scientific + engineering review:
- **H1 headline metric is now `post_mse`** (post-shock window), not whole-horizon `mse`. Every H1 arm
  binds `StabilizationLoss(post_shock_step=shock_step)`; `compare` defaults to `post_mse`. The score
  (full-horizon) is unchanged, so golden-master is stable. *Why:* the claim is "lower **post-shock**
  regret", but averaging in the 100 pre-shock steps (where a frozen LQR is near-optimal) can invert the
  verdict.
- **Sandbox:** reject non-ASCII expressions (closes a Unicode/NFKC whitelist bypass that reached the
  un-seeded global `random`, breaking reproducibility); reject chained/huge-exponent power bombs; drop
  `arange`/`linspace` (unbounded allocation) and `random`/`string` modules from the sandbox.
- **OPRO `realized` mode** now explore/exploits (deterministic schedule) and **validates before deploy**,
  instead of deploying a fresh temp-0.8 proposal every decision — otherwise the FAIR H1 rival was
  silently crippled by permanent exploration cost.
- **Stats:** `compare` will not call a result significant below `min_n=2` paired seeds (a single shared
  seed gives a zero-width CI that falsely "excludes 0") and flags `underpowered`; `metric_by_seed` now
  raises on an unknown metric instead of fabricating NaN → "no significant difference"; `collapse_summary`
  keeps a non-finite (diverged) score as `worst_score=-inf` instead of dropping it.
- **SIR** now conserves S+I+R (recoveries snapshot `I` before mutation); `CubicSystem.reset` restores
  ALL shockable params; `rollout` records post-step rows only (matches the Runner's slice); a zero-action
  regent decision surfaces an `Outcome.error` so `TraceFeedback` fires; `RunRecord` carries
  `harness_components` (H3 ablation provenance); the obfuscated H1 prompt no longer coaches the solution
  structure (a prompt-hint confound vs OPRO).
**Why:** the compared quantity must equal the claimed quantity; the primary sandbox control must not be
bypassable; a named baseline must not be handicapped; and a degenerate CI must not read as a win.
**Status:** done; 16 regression tests in `tests/test_review_fixes.py`. Seed count for the headline claim
(≥20) and the regime-severity knob remain ☐ AUTHOR calls (STATUS.md).

## ADR-0015 — `CoupledSystem` migrated to `LeverSystem`; lever drives `u_commanded` (2026-06-15)
**Decision:** the legacy `CoupledLinearStochasticSystem` is migrated onto `LeverSystem`/`RollableSystem`;
the lever writes `u_commanded` and `step()` smooths it into `current_u` (`u_eff`), so the existing
control-inertia semantics survive while the eval-cadence contract (per-step re-eval) holds. All
randomness moves from bare `random.gauss` to `self.rng`. **Why:** completes Phase-1's "coupled system";
bare RNG would have made `clone()`/rollout unsound and tripped the invariant test. **Status:** done
(`coupled_stabilization`, `coupled_regime_shift`). Legacy `economic_models/*` removal still deferred to
Phase 4 (ADR-0006).

## ADR-0014 — Rollout primitive in `core`; `RollableSystem` is the precondition gate (2026-06-15)
**Decision:** `core/rollout.py` provides a domain-neutral clone→install→step→score oracle; the Runner
injects a `RolloutContext` into a regent's scratch **only when the system is a `RollableSystem`** — its
presence/absence IS the rollout-soundness gate (doc-09 §5.2). `seed=None` continues the carried
Generator faithfully; an explicit seed resamples an independent future (for `mean−λ·std` selection).
**Why:** rollout-dependent components (`RolloutProbe`, OPRO) must be sound and must no-op where clone is
impossible. **Status:** done; `RolloutProbe` ships, gated.

## ADR-0013 — OPRO scores candidates by rollout; archive lives in scratch (2026-06-15)
**Decision:** `OPRORegent` (the trace-less H1 rival) scores each proposed law by rollout on a clone
(rather than waiting for realized multi-step feedback), and keeps its `(law, score)` archive in
`scratch` (which the Runner resets per seed). **Why:** rollout scoring is self-contained and
replay-deterministic; a scratch-held archive resets per seed so the paired comparison is unbiased (a
`self`-held archive would leak across seeds). It degrades to emitting the LLM's raw proposal on a
non-rollable system. **Status:** done (`cubic_nonlinear_opro`).

## ADR-0012 — `govsim/analysis`: the stats protocol as code (2026-06-15)
**Decision:** ship paired bootstrap CI on per-seed differences, variance-aware selection (`mean−λ·std`),
and a collapse/tail-event detector as `govsim/analysis/` + a `govsim compare A B` CLI subcommand.
**Why:** `stats-protocol.md` was a slogan with no code; a "X beats Y" claim needs the paired CI to
exclude 0, and selection must never be on the mean alone. **Status:** done; the H1/H3 comparisons run
through it.

## ADR-0011 — Partial-information obfuscated prompt + `check_prompt` boot validator (2026-06-15)
**Decision:** the H1 nonlinear arm gets a `make_obfuscated_assembler` that tells the regent only
`x_(k+1)=f(x_k,u_k,noise)` with `f` UNKNOWN (infer the cubic from history); `check_prompt`
(doc-06 §2.3) validates any `{placeholder}` template against the world's suppliable names at
construction, failing loud at boot instead of blanking at the LLM. **Why:** "partial information" is the
regime where code-as-policy can beat a fixed-form PID, and was previously unwired. **Status:** done
(`cubic_nonlinear_llm_obfuscated`).

## ADR-0010 — `Critic` harness component (rollout-free audit→revise) (2026-06-15)
**Decision:** add a `Critic` component: a 2nd LLM audits the proposal and, on a concrete veto, the
regent revises once with the critique on `scratch["critic"]` (surfaced by both assemblers). Conservative
by default (approves unless it can name a problem). **Why:** an H3 leave-one-out ablation arm and a
cheap, rollout-free upgrade. **Status:** done (`cubic_nonlinear_llm_critic`).

## ADR-0009 — LLM seam carries `max_tokens` + provider `extra`; verified live (2026-06-15)
**Decision:** extend `LLMClient.complete` (and the cache key, `SCHEMA_VERSION`→`2`) with `max_tokens` +
a provider `extra` dict (e.g. gpt-oss `reasoning_effort="low"`); the CLI loads `.env`. **Why:** reasoning
models return empty content unless given a token budget + low effort; the cache key must cover them.
Verified end-to-end against a live OpenAI-compatible vLLM cluster (`Openai/Gpt-oss-120b`, tool-calling),
including byte-exact replay with a wrong key. **Status:** done; tests stay key-free (fake clients +
committed replay tapes only).

## ADR-0008 — Persist via git bundle when remote writes are 403 (2026-06-14)
**Decision:** in this session both `git push` and the GitHub API (MCP `push_files`) return 403
(write/contents not provisioned), so completed work is handed off as a `git bundle` over file
transfer for the author to apply + push. **Why:** the container is ephemeral; this is the only
write path available from inside. **Status:** active workaround; revert to normal `git push` once
write permission is restored.

## ADR-0007 — ResultStore uses sqlite + CSV/JSON artifacts, not parquet (2026-06-14)
**Decision:** the `runs` table is sqlite (stdlib, queryable); per-step series are CSV, raw LLM I/O
is JSON. **Why:** doc-09 §2.4 says "sqlite + parquet", but parquet needs `pyarrow`/`fastparquet` —
heavy deps that slow CI sync for no Phase-0 benefit. pandas (already a dep) reads/writes CSV. **Status:**
accepted; swap the artifact writer to parquet later if cross-run series volume demands it.

## ADR-0006 — Additive migration; legacy path kept green until superseded (2026-06-14)
**Decision:** the new `core`/`domains`/`experiments` stack is built additively; the legacy
`simulation.py` / `economic_models` / `governing_agents` path is left importable until the new path
fully covers it, then removed. **Why:** doc-09 Phase 0 is a "big rewrite", but staying green every
commit (the existing strategy) is safer for a solo author. **Status:** vestigial files
(`combine_scripts.py`, `special_visualize_exp4.py`, `visualize_universal_legacy.py`) removed in the
cleanup increment; the economy-model legacy removal is deferred to Phase 4 (its replacement).

## ADR-0005 — Cubic kept + parameterized, not reverted to linear (2026-06-14)
**Decision:** `CubicSystem` exposes `cubic_coeff` × `state_exponent`; `cubic_coeff=0` is the linear
LQR-sanity arm, `cubic_coeff!=0` is the H1 nonlinear partial-information arm. **Why:** doc-08 §5
overrules 01/02's "revert to linear" — the cubic + obfuscated prompt is the repo's most novel arm.
**Status:** done (`cubic_stabilization`, `cubic_nonlinear`).

## ADR-0004 — Per-system `np.random.Generator`; bare RNG banned by a test (2026-06-14)
**Decision:** every `System` owns `self.rng`; `tests/test_invariants.py` fails on any bare
`random.*`/`np.random.*`. **Why:** the HARD doc-08 precondition for a sound `clone()`/`rollout()`
fitness oracle (module-global RNG makes a rollout a re-seeded restart, biasing fitness). **Status:** done.

## ADR-0003 — `ActionInterface.apply` installs; `System.step` re-evaluates (2026-06-14)
**Decision:** levers are installed as **pure data** on the system; `step()` re-evals each tick.
**Why:** fixes the doc-08 §3.2 eval-cadence bug (legacy `SingleMarketModel` eval'd once at apply) and
is clone-safe (a deepcopied clone re-evals against itself, not a closure over the original). **Status:** done.

## ADR-0002 — WHAT-first gate is executable (2026-06-14)
**Decision:** `Experiment.hypothesis` (claim + named baseline) is required; `Runner.run` raises
without it. **Why:** doc-09 §1.3 — stronger than doc-08's discipline; a precondition that cannot rot.
**Status:** done.

## ADR-0001 — `clone()` on `RollableSystem`, not the base `System` (2026-06-14)
**Decision:** clone is a capability subtype; rollout harnesses require it, the base does not. **Why:**
"control any complex system" is false at the type level if clone is mandatory (a live API world can't
clone) — doc-09 §10. **Status:** done.
