# Creativity metric — measure it, or honestly drop the word

> "Creativity / complex behavior" is the project's identity word and (doc-08 §3.1) has no metric
> anywhere in 01–07. Either it becomes a **defined, domain-scoped dependent variable**, or we drop
> it and call the result *adaptive in-context control*. There is deliberately **no single creativity
> number across all domains** (doc-09 §1.4, §6.3). `Experiment.creativity_metric` is `None` for the
> arms where no construct separates the LLM from a tuned controller (e.g. the linear sanity arm).

## Domain-scoped constructs
- **Control domains (primary):**
  - *generalization gap* — the regent beats a tuned PID **specifically on un-tuned regimes**
    (nonlinear / shifted), while tying on the regime the PID was tuned for. Positive ⇒ transferable.
  - *functional novelty* — the residual of the best-fit simple controller: does the emitted law use
    conditionals / state-history / regime detection a PID structurally **cannot**?
- **Governance domain (adds):** *Policy-Innovation-Score* (semantic distance from a known-policy
  library) + institution-type novelty (the author's own metric, `Модель креативного правительства.txt`).
- **Cross-cutting (optional):** MAP-Elites QD behavioral diversity.

## Discipline (non-negotiable)
- A **parsimony / restricted-grammar** term keeps emitted laws human-readable — otherwise evolution
  yields `np.clip`/`where`/`tanh` spaghetti and the interpretability pitch fails (doc-08 §3.1).
- If **none** of the constructs separate the LLM from a tuned controller on a given arm ⇒ set
  `creativity_metric=None` and reframe that arm as adaptive in-context control. (This is honest, not
  a failure: it is exactly what `cubic_stabilization` already declares.)

## ☐ AUTHOR decision still open
Is creativity a measured construct for paper 1 (recommended: generalization-gap + functional-novelty
on the nonlinear cubic arm), or dropped for paper 1 and deferred to the evolution work?
