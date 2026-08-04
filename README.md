# creative-governance-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An experiment machine for **LLM "regents" — controllers that govern a dynamical system by emitting
policy as sandboxed executable code** — and for measuring whether the *harness* around such a
controller does anything.

> **Name collision, unresolved.** `GovSim` is already an established artifact (Piatti et al.,
> NeurIPS 2024, arXiv:2404.16698) for LLM commons governance. This project is unrelated to it and
> should be renamed before any submission. See `govsim/docs_gates/STATUS.md`.

---

## The question, and the measurement that comes before it

The obvious experiment is: break the world in a way the controller was not told about, and see
whether it recovers better than a rule that cannot adapt. The problem is that the answer is bounded
above by a property of the environment that nobody reports — **how much better anything could have
done by adapting**. If a shock leaves a fixed rule near-optimal, a null result says nothing about
the controller.

So this repo measures that first. For any regime it calibrates, inside one shared policy vocabulary:

| reference | what it knew |
|---|---|
| `frozen` | optimal before the break, then held unchanged through it |
| `best_fixed` | the best single **fixed** law over the whole broken horizon, chosen in hindsight |
| `switching` | the pre-break optimum, then the post-break optimum, switched at the exact break |

`headroom = L(best_fixed) / L(switching)` is what changing behaviour is worth. Headroom ≈ 1 means the
question is unanswerable in that regime, whatever controller you put in it.

**What that measurement found.** Feedback rules ("intervene when the indicator crosses θ") *absorb*
shocks to the governed system — the indicator rises, the rule fires more often, and it stays
near-optimal without anyone touching it. They cannot absorb shocks to **instrument efficacy**, where
the lever keeps costing what it cost and stops working. Most of the shock families this kind of
benchmark reaches for first are in the first category, and are near-null by construction.

```bash
uv run python scripts/headroom_audit.py --seeds 8    # which regimes can host the question at all
```

## The flagship regime

An endemic SIRS epidemic. The authority holds two costed instruments, lockdown and vaccination, and
writes its policy as a Python expression the world re-evaluates every step. At `t=100`, without
announcement, lockdown efficacy collapses to a quarter while lockdown still costs what it did.
Efficacy is **never observable**. `S+I+R` is conserved, so no arm can diverge.

Calibrated jointly over both instruments, the pre-break optimum uses *no* vaccination and the
post-break optimum uses the *maximum* — the correct response to a broken instrument is substitution
toward the one that still works.

## The harness ablation

A full $2^3$ factorial over three channels that carry **different kinds** of information:

- `TraceFeedback` — rejections and runtime errors. A channel about *malformedness*; it cannot fire
  on well-formed policy, which is exactly what makes it the right null control.
- `OutcomeFeedback` — what the law you actually deployed achieved over the last interval. The only
  channel in the stack from which an unobservable efficacy collapse is inferable.
- `EpisodicMemory` — retrieved precedent, which across a structural break may be actively misleading.

Interactions are estimated rather than assumed away, and the family is Holm-corrected. Every LLM arm
decides on the same schedule, so no arm wins by thinking more often.

## Quick start

```bash
uv sync                  # Python >=3.12,<3.14
uv run pytest            # 122 tests, all key-free
uv run govsim list       # registered experiments
uv run govsim run epidemic_frozen --seeds 0 1 2      # key-free reference arm
```

For LLM arms, put a key in `.env` and point the client at any OpenAI-compatible endpoint:

```bash
OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
OPENAI_API_KEY_ENV=GOOGLE_API_KEY OPENAI_MODEL="gemma-4-31b-it" \
GOVSIM_LLM_MODE=cache GOVSIM_LLM_DROP_PARAMS=seed \
uv run python scripts/run_matrix.py --arms epidemic --seeds 20 --workers 6 --store logs/runs
```

`GOVSIM_LLM_DROP_PARAMS` strips wire parameters a given endpoint rejects (Gemini 400s on `seed`) —
at the wire only, so the cache key is unchanged and recorded tapes still replay exactly.

### Reproducing without a key

Every model call is written to a content-addressed tape. `GOVSIM_LLM_MODE=replay` serves every
reported run from disk with no network and no key; a deliberately wrong key changes nothing. That is
what makes a sampled decision-maker compatible with a reproducibility claim at all.

## Full pipeline

```bash
uv run python scripts/recalibrate.py --seeds 20        # -> govsim/docs_gates/calibration.json
uv run python scripts/run_matrix.py --arms epidemic --seeds 20 --models <model> --workers 6
uv run python scripts/analyze_matrix.py --store logs/runs_* --cross-model --json logs/analysis.json
uv run python scripts/policy_audit.py --store logs/runs_*   # what the regents actually wrote
uv run python scripts/make_tables.py && uv run python scripts/make_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex   # and social.tex
```

> Re-run `recalibrate.py` after **any** change to a regime. A stale calibration silently re-anchors
> every normalized-regret number downstream.

## Layout

| Where | What |
|---|---|
| `govsim/core/` | the six domain-neutral seams + `Experiment`/`Runner`/`ResultStore` + the LLM cache/replay tape |
| `govsim/domains/scalar/` | the worlds (cubic, coupled, SIR/SIRS, company), their objectives, and the **pinned regimes** |
| `govsim/regents/` | `LLMRegent`, `OPRORegent`, and the calibrated baselines (`LQR`, `Oracle`, `Switching`) |
| `govsim/harness/` | the ablatable channels: trace, outcome, memory, critic, rollout probe |
| `govsim/analysis/` | paired bootstrap, factorial effects with interactions, Holm, **regime calibration** |
| `govsim/docs_gates/` | the pre-registration, the generated calibration artifact, the ADR log |
| `scripts/` | calibration, sweeps, analysis, policy audit, table and figure generation |
| `paper/` | the technical manuscript (`main.tex`) and the social-science one (`social.tex`) |

The `Runner` refuses to run an `Experiment` without a `Hypothesis` carrying a claim and a *named*
baseline. Deciding what is being tested is an executable precondition, not a discipline.

Deeper design docs are in [`agents/`](agents/); [`AGENTS.md`](AGENTS.md) is the contributor guide;
current state and open author decisions are in
[`govsim/docs_gates/STATUS.md`](govsim/docs_gates/STATUS.md).

## License

MIT. See [LICENSE](LICENSE).
