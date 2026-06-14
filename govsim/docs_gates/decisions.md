# Decisions log (ADR) — supersession chain + implementation deviations

> doc-08 §3.2 asked for one ADR file instead of supersession prose scattered across docs. This is it.
> Newest first. Each entry: **decision**, **why**, **status**.

## Doc supersession chain (authoritative)
- **05 (roadmap)** → superseded by **08 §7** (priority) and **09 §7** (execution).
- **06 (model integration)** / **07 (governance interface)** → repositioned by **09** as the
  *economy domain plugin* (Phase 4), not the top-level framework.
- **08** is authoritative on two reversals of 01/02: (1) **keep** the cubic + obfuscated prompt;
  (2) treat 01's citations as unverified leads.
- **09** is AUTHORITATIVE for architecture + implementation. This branch implements **Phase 0** of 09.

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
