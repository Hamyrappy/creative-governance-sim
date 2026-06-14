# Statistics protocol (pre-registration template)

> "Multi-seed CIs" is repeated as a slogan with no procedure (doc-08 §3.1). This half-page is the
> procedure; it is a doc, not code, and it protects every result. Adopt it before any comparison.

## Design
- **Paired, shared-seed.** Every regent under comparison is run on the **same** set of world seeds
  (`Experiment.seeds`); the comparison statistic is computed **per-seed**, then aggregated. The
  `Runner` already runs each regent over the identical seed list, so this holds by construction.
- **Isolate the LLM as a nuisance source.** Run the LLM through `CachingReplayClient` in **replay**
  mode during a comparison so the LLM response is fixed and the **only** varying source is the world
  seed. (For "LLM nondeterminism" as its *own* question, vary the LLM seed with the world seed fixed.)
- **Primary metric pre-registered.** Pick ONE primary metric per hypothesis (`Hypothesis.primary_metric`)
  *before* looking at results. Secondary metrics are exploratory.

## Inference
- **Bootstrap CI on the per-seed difference** (regent − baseline) for the primary metric; report the
  CI and the point estimate. A claim "X beats Y" requires the paired bootstrap CI to **exclude 0**.
- For the ablation (H3): a component "yields a separable gain" iff its paired bootstrap CI excludes 0
  in **both** the single-add and the leave-one-out design (doc-09 §5.4).
- **Variance-aware selection** (not just reporting): when a harness *selects* among candidates, use
  `mean − λ·std` over seeds (or accept-only-if-worst-of-N-bounded) — never select on the mean alone.
- **Collapse detector.** A run that diverges/terminates early is recorded (`terminated_at_step`) and
  counted as worst-case, not dropped — the Vending-Bench tail-event lesson.

## Reporting
- N seeds (recommend ≥ 20 for a headline claim; ≥ 5 for development), the seed list, the git commit,
  the model snapshot, and the replay-tape hash — all already in the `RunRecord` / `ResultStore` row.
- Multiple comparisons: if reporting many secondary metrics, say so; control or label as exploratory.

## ☐ AUTHOR decision still open
Number of seeds for the headline H1 result, and the per-experiment compute/$ budget (doc-08 §8 Q7).
