"""
Statistics for paired, shared-seed comparisons (the executable stats-protocol).

Everything here is pure and deterministic given its inputs (the bootstrap uses an explicit
``np.random.Generator`` seed), so a reported CI is itself reproducible.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

# Metrics where SMALLER is better (losses/costs); used to orient "A beats B" when not told.
_LOWER_IS_BETTER_HINTS = {
    "mse", "msu", "loss", "cost", "cum_cost", "total_infected", "peak_infected", "regret",
    "mean_abs_x", "final_abs_x", "post_mse", "post_msu", "post_loss",
}


def infer_lower_is_better(metric: str) -> bool:
    """Heuristic orientation for a metric name (losses/costs are minimized). Override explicitly
    when a name is ambiguous; ``score`` (the negated-loss Objective output) is higher-is-better."""
    return metric.lower() in _LOWER_IS_BETTER_HINTS


def robust_score(scores: Sequence[float], lam: float = 0.5) -> float:
    """``mean − λ·std`` over a candidate's per-future scores (worst-case-aware selection value)."""
    vals = list(scores)
    if not vals:
        return float("-inf")
    m = statistics.fmean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m - lam * s


def variance_aware_select(candidate_scores: Sequence[Sequence[float]], lam: float = 0.5) -> int:
    """Index of ``argmax(mean − λ·std)`` — the stats-protocol selection rule (never the mean alone)."""
    best_i, best = 0, float("-inf")
    for i, sc in enumerate(candidate_scores):
        v = robust_score(sc, lam)
        if v > best:
            best, best_i = v, i
    return best_i


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the MEAN of ``values`` (typically paired per-seed differences).

    Returns ``(point_estimate, ci_low, ci_high)``. Degenerate inputs (0 or 1 value) return the
    point as a zero-width interval — honest about there being no spread to estimate.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    point = float(arr.mean())
    if arr.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (point, float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_diff(a_by_seed: Mapping[int, float], b_by_seed: Mapping[int, float]) -> tuple[list[int], list[float]]:
    """Per-seed ``a − b`` over the seeds present in BOTH (the paired design)."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    return seeds, [float(a_by_seed[s]) - float(b_by_seed[s]) for s in seeds]


def compare(a_by_seed: Mapping[int, float], b_by_seed: Mapping[int, float], *,
            lower_is_better: bool = True, n_boot: int = 10_000, alpha: float = 0.05,
            seed: int = 0, min_n: int = 2, dev_floor: int = 5) -> dict[str, Any]:
    """Paired bootstrap comparison of A vs B over shared seeds.

    ``a_better_than_b`` is True iff the whole CI lies on the winning side of 0 (the stats-protocol
    "CI excludes 0" rule, oriented by ``lower_is_better``) AND there are at least ``min_n`` paired
    seeds. The floor matters: a single shared seed gives a zero-WIDTH bootstrap CI that trivially
    "excludes 0", which would falsely read as a significant win — so significance is suppressed below
    ``min_n``. ``underpowered`` flags ``n < dev_floor`` (the stats-protocol development floor); a
    headline claim wants ≥20. ``no_shared_seeds`` flags the pathological empty pairing (n == 0).
    """
    seeds, diffs = paired_diff(a_by_seed, b_by_seed)
    n = len(seeds)
    point, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    reliable = n >= min_n  # below the floor a degenerate CI must not be called "significant"
    excludes_zero = reliable and (lo > 0 or hi < 0)
    a_better = excludes_zero and ((hi < 0) if lower_is_better else (lo > 0))
    return {
        "seeds": seeds,
        "n": n,
        "diffs": diffs,
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": excludes_zero,
        "a_better_than_b": a_better,
        "lower_is_better": lower_is_better,
        "underpowered": n < dev_floor,
        "no_shared_seeds": n == 0,
    }


def metric_by_seed(records: Sequence[Any], metric: str, regent_id: str = "regent:0") -> dict[int, float]:
    """Extract ``{seed: value}`` for ``metric`` from a list of ``RunRecord``s.

    Looks in ``components[regent_id]`` first (where MSE/MSU/cost live), then the special name
    ``"score"`` (the Objective scalar for ``regent_id``). Raises ``KeyError`` for a metric the
    objective does not emit — a mistyped/mis-registered ``primary_metric`` must fail LOUD, not
    silently become NaN that ``compare`` then reports as "no significant difference".
    """
    out: dict[int, float] = {}
    for rec in records:
        comps = rec.components.get(regent_id, {})
        if metric in comps:
            out[rec.seed] = float(comps[metric])
        elif metric == "score":
            out[rec.seed] = float(rec.score.get(regent_id, float("nan")))
        else:
            raise KeyError(
                f"metric {metric!r} not emitted for {regent_id!r} on run seed={getattr(rec, 'seed', '?')}"
                f" (available: {sorted(comps) + ['score']}). Check the experiment's primary_metric and"
                " the Objective.components it maps to."
            )
    return out


def collapse_summary(records: Sequence[Any], regent_id: str = "regent:0") -> dict[str, Any]:
    """Collapse/tail-event detector: count early-terminated runs and report the WORST score (never
    drop a diverged run — it is the tail event the headline claim must survive)."""
    terminated = [rec for rec in records if rec.terminated_at_step is not None]
    scores = [rec.score.get(regent_id, float("nan")) for rec in records]
    finite = [s for s in scores if math.isfinite(s)]
    nonfinite = [s for s in scores if not math.isfinite(s)]
    # A diverged run (NaN/inf score) is the tail event the headline claim must survive — it must NOT
    # be silently dropped from the worst-case. Any non-finite score ⇒ worst-case is unbounded-bad
    # (score is higher-is-better), so worst_score = -inf; the count is surfaced explicitly.
    worst = float("-inf") if nonfinite else (min(finite) if finite else float("nan"))
    return {
        "n_runs": len(records),
        "n_terminated": len(terminated),
        "terminated_seeds": [r.seed for r in terminated],
        "n_nonfinite_score": len(nonfinite),
        "worst_score": worst,
        "mean_score": statistics.fmean(finite) if finite else float("nan"),
    }
