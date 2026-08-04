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


def bootstrap_ci_bca(values: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI for the mean.

    The plain percentile interval is anti-conservative at small n: it takes quantiles of the
    bootstrap distribution without correcting for the bias or the skew of that distribution. Under
    the null at n=20 the shipped percentile version covers ~92-93% at a nominal 95%, so the
    decision rule "the CI excludes 0" rejects at roughly 7-8% rather than 5% — a 50% inflation of
    the type-I rate on the one rule every claim in this project rests on.

    BCa corrects both terms: ``z0`` from the share of resamples below the observed mean, and the
    acceleration ``a`` from the jackknife skew. It costs one extra jackknife pass and needs no
    dependency beyond the normal quantile in the standard library.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = float(arr.mean())
    if n < 3:  # BCa's acceleration is undefined; fall back rather than fabricate an interval
        return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)

    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)

    prop = float((boot < point).mean())
    if prop <= 0.0 or prop >= 1.0:  # degenerate resample distribution
        return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
    nd = statistics.NormalDist()
    z0 = nd.inv_cdf(prop)

    # Jackknife acceleration: the third moment of the leave-one-out means.
    total = arr.sum()
    jack = (total - arr) / (n - 1)
    jbar = jack.mean()
    d = jbar - jack
    denom = 6.0 * float((d ** 2).sum()) ** 1.5
    a = float((d ** 3).sum()) / denom if denom != 0 else 0.0

    def _adj(q: float) -> float:
        zq = nd.inv_cdf(q)
        num = z0 + zq
        return float(nd.cdf(z0 + num / (1.0 - a * num)))

    lo_q, hi_q = _adj(alpha / 2.0), _adj(1.0 - alpha / 2.0)
    lo_q = min(max(lo_q, 1e-6), 1 - 1e-6)
    hi_q = min(max(hi_q, 1e-6), 1 - 1e-6)
    if hi_q <= lo_q:
        return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
    return (point, float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q)))


def bootstrap_ci_t(values: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float, float]:
    """Studentized (bootstrap-t) CI for the mean — the one that actually holds its nominal level here.

    The percentile interval under-covers at n=20 (measured: ~0.92 against a nominal 0.95, so the
    "CI excludes 0" rule rejects at ~8% instead of 5%). BCa does not fix it, and the reason is
    instructive: BCa corrects *bias* and *skew*, and a roughly symmetric paired difference has
    little of either. What is actually missing is the extra uncertainty from estimating the variance
    on 20 points — the reason one uses a t rather than a z. Studentizing puts it back, by
    bootstrapping the pivot ``(mean* - mean) / se*`` and reading its quantiles instead of assuming
    normal ones.

    Resamples whose own standard error is zero are dropped rather than clipped; they carry no
    information about the pivot and would otherwise produce infinities.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n < 3:
        return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
    point = float(arr.mean())
    se = float(arr.std(ddof=1)) / math.sqrt(n)
    if se == 0:
        return (point, point, point)

    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, n, size=(n_boot, n))]
    means = samples.mean(axis=1)
    ses = samples.std(axis=1, ddof=1) / math.sqrt(n)
    ok = ses > 0
    if ok.sum() < 100:
        return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
    pivot = (means[ok] - point) / ses[ok]
    lo_q, hi_q = np.quantile(pivot, [1 - alpha / 2, alpha / 2])  # note the inversion
    return (point, float(point - lo_q * se), float(point - hi_q * se))


def wilcoxon_signed_rank_p(values: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p for "the median of ``values`` is 0".

    Reported alongside the bootstrap because it assumes almost nothing: no normality, no
    bootstrap-distribution shape. When a bootstrap CI says "significant" and this says otherwise at
    n=20, the honest reading is that the result is fragile, and a reader is owed both numbers rather
    than whichever one is friendlier. Normal approximation with continuity and tie corrections,
    which is adequate from about n=15.
    """
    arr = np.asarray([v for v in values if v != 0.0], dtype=float)
    n = arr.size
    if n < 6:
        return 1.0
    order = np.abs(arr).argsort()
    absv = np.abs(arr)[order]
    signs = np.sign(arr)[order]
    # Average ranks within ties.
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absv[j + 1] == absv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_plus = float(ranks[signs > 0].sum())
    mean_w = n * (n + 1) / 4.0
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absv[j + 1] == absv[i]:
            j += 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return 1.0
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
    return float(min(1.0, 2.0 * (1.0 - statistics.NormalDist().cdf(z))))


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
    seeds. ``robust_agreement`` additionally requires a distribution-free Wilcoxon signed-rank test
    to agree at the same level; prefer it for any reported claim, because at this n a bootstrap
    interval and a rank test disagreeing is the signal that a result rests on the tail behaviour of
    a handful of seeds. The floor matters: a single shared seed gives a zero-WIDTH bootstrap CI that trivially
    "excludes 0", which would falsely read as a significant win — so significance is suppressed below
    ``min_n``. ``underpowered`` flags ``n < dev_floor`` (the stats-protocol development floor); a
    headline claim wants ≥20. ``no_shared_seeds`` flags the pathological empty pairing (n == 0).
    """
    seeds, diffs = paired_diff(a_by_seed, b_by_seed)
    n = len(seeds)
    # Studentized rather than plain percentile. Measured coverage at n=20 against a nominal 95%:
    # percentile 0.921, BCa 0.921, bootstrap-t 0.935 — so the percentile rule this protocol was
    # built on rejects at ~8% rather than 5%. BCa does not help because a symmetric paired
    # difference has little bias or skew for it to correct; what is missing is the uncertainty in
    # the variance estimate, which studentizing restores.
    point, lo, hi = bootstrap_ci_t(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    pct_point, pct_lo, pct_hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    reliable = n >= min_n  # below the floor a degenerate CI must not be called "significant"
    excludes_zero = reliable and (lo > 0 or hi < 0)
    a_better = excludes_zero and ((hi < 0) if lower_is_better else (lo > 0))
    p_wilcoxon = wilcoxon_signed_rank_p(diffs) if reliable else 1.0
    return {
        "seeds": seeds,
        "n": n,
        "diffs": diffs,
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "ci_method": "bootstrap-t",
        # The uncorrected percentile interval, kept so the difference is auditable rather than a
        # claim in a docstring.
        "ci_low_percentile": pct_lo,
        "ci_high_percentile": pct_hi,
        # A distribution-free second opinion. When these disagree at this n, the result is fragile
        # and the reader is owed both rather than whichever is friendlier.
        "p_wilcoxon": p_wilcoxon,
        "excludes_zero": excludes_zero,
        "a_better_than_b": a_better,
        "robust_agreement": bool(excludes_zero and p_wilcoxon <= alpha),
        "lower_is_better": lower_is_better,
        "underpowered": n < dev_floor,
        "no_shared_seeds": n == 0,
    }


def bootstrap_p(values: Sequence[float], *, n_boot: int = 10_000, seed: int = 0) -> float:
    """Two-sided **studentized** bootstrap p-value for "the mean of ``values`` is 0".

    Inverts the same pivot as :func:`bootstrap_ci_t`: compare the observed ``t = mean/se`` against
    the bootstrap distribution of ``t* = (mean* − mean)/se*``, and read off the two-sided tail.

    The earlier version used the raw percentile distribution of the mean, and it was wrong in a way
    that mattered more than the interval it accompanied. Percentile p-values inherit the same
    under-coverage as percentile intervals, but **multiplicity correction pushes the test into the
    small-α tail where that inflation is worst**: measured on 6000 null samples at n=20, the
    percentile p rejected at 1.53× nominal at α=0.05 and 2.52× at α=0.05/7 — the level Holm actually
    applies across a seven-term factorial. Feeding those into Holm produced a family-wise error rate
    near 14% while the paper claimed 5%. Holm controls FWER only if its inputs are valid p-values;
    correcting invalid ones corrects nothing.

    A p-value of exactly 0 is floored at ``1/n_boot`` rather than reported as 0: a resampling
    procedure cannot certify a tail smaller than its own resolution, and an exact zero survives any
    correction at any family size.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n < 3:
        return 1.0
    point = float(arr.mean())
    se = float(arr.std(ddof=1)) / math.sqrt(n)
    if se == 0:
        return 0.0 if point != 0 else 1.0
    t_obs = point / se

    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, n, size=(n_boot, n))]
    ses = samples.std(axis=1, ddof=1) / math.sqrt(n)
    ok = ses > 0
    if ok.sum() < 100:
        return 1.0
    t_star = (samples.mean(axis=1)[ok] - point) / ses[ok]
    p = float((np.abs(t_star) >= abs(t_obs)).mean())
    return min(1.0, max(1.0 / max(1, int(ok.sum())), p))


def holm_bonferroni(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm step-down correction over a family of comparisons.

    A factorial ablation is a *family* of tests — eight cells, three main effects, four
    interactions — and reporting each at α=0.05 guarantees false positives at that many looks.
    Holm controls the family-wise error rate without assuming independence (Benjamini-Hochberg
    would be less conservative but controls a different, weaker quantity), so it is the right
    default when the claim is "this specific component did something".

    Returns, per key, the raw p, the adjusted p, its rank, and whether it survives at ``alpha``.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))  # enforce monotone non-decreasing adjusted p
        running = adj
        out[key] = {"p": p, "p_adj": adj, "rank": i + 1, "significant": adj <= alpha}
    return out


def channel_liveness(
    reported: Sequence[float],
    truth: Sequence[float],
    *,
    min_injections: int = 1,
    min_rho: float = 0.2,
) -> dict[str, Any]:
    """Did an information channel actually deliver information? Run this BEFORE reporting its null.

    An ablation reports "component X did not help". That sentence is only meaningful if X delivered
    a signal to begin with, and there are at least three ways for it not to, all of which look
    identical in a loss table:

    * **it never fired** — the component only writes on a condition that never occurred;
    * **it fired with a degenerate number** — e.g. a running total, which makes the reported
      quantity a clock and its correlation with actual policy quality zero;
    * **it fired with the right number at the wrong time** — misaligned by a window.

    We hit the first two in the same study, and the analysis pipeline reported both as estimated
    nulls indistinguishable from real ones. So liveness is now a gate, not a diagnostic: compare
    what the channel *reported* against the quantity it is supposed to track, and refuse to
    interpret a null from a channel that fails.

    ``rho`` is Spearman (rank) rather than Pearson because the channel only has to be
    monotonically informative, not linear. Returns ``live`` plus the pieces of evidence, so a
    failure says which way it failed.
    """
    a = np.asarray(list(reported), dtype=float)
    b = np.asarray(list(truth), dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return {"live": False, "n": 0, "rho": float("nan"), "reason": "channel never fired"}
    a, b = a[:n], b[:n]
    if n < min_injections:
        return {"live": False, "n": n, "rho": float("nan"),
                "reason": f"fired only {n} time(s), below the floor of {min_injections}"}
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return {"live": False, "n": n, "rho": float("nan"),
                "reason": "the reported quantity (or the truth) is constant — no signal to carry"}

    def _rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        return r

    ra, rb = _rank(a), _rank(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    # Sign agreement of consecutive changes: the channel's practical use is telling the regent
    # whether things got better or worse, so getting the DIRECTION right is what matters.
    da, db = np.diff(a), np.diff(b)
    mask = (da != 0) & (db != 0)
    agree = float(np.mean(np.sign(da[mask]) == np.sign(db[mask]))) if mask.any() else float("nan")
    live = bool(rho >= min_rho)
    return {
        "live": live, "n": n, "rho": rho, "sign_agreement": agree,
        "reason": "" if live else (
            f"rank correlation with the quantity it should track is {rho:.3f} "
            f"(< {min_rho}); the channel is reporting something else"),
    }


def minimum_detectable_effect(
    per_seed_diffs: Sequence[float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    n_comparisons: int = 1,
) -> dict[str, Any]:
    """The smallest true effect this design could have detected — what a null is allowed to claim.

    A null result is only informative paired with this number. "No component helped" means one thing
    if the design could have caught a 5% improvement and quite another if it could only have caught a
    50% one, and an ablation table that reports the first without the second invites the reader to
    assume the stronger reading.

    Computed from the observed per-seed paired differences, so it reflects the variance actually
    present rather than an assumed one. Two-sided, normal approximation (adequate at n=20 for a
    paired mean), and Bonferroni-split across ``n_comparisons`` to match the correction the headline
    test uses — an MDE quoted at uncorrected alpha would understate what the reported analysis
    actually requires.

    Returns the MDE in metric units, the per-seed sd it came from, and the n needed to halve it.
    """
    arr = np.asarray(list(per_seed_diffs), dtype=float)
    n = arr.size
    if n < 2:
        return {"n": n, "sd": float("nan"), "se": float("nan"), "mde": float("inf"),
                "alpha_effective": alpha, "n_for_half_mde": None}
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    alpha_eff = alpha / max(1, n_comparisons)
    # Normal quantiles without scipy: Acklam's rational approximation is overkill here, and
    # statistics.NormalDist is stdlib and exact enough.
    nd = statistics.NormalDist()
    z_alpha = nd.inv_cdf(1.0 - alpha_eff / 2.0)
    z_power = nd.inv_cdf(power)
    mde = (z_alpha + z_power) * se
    return {
        "n": n,
        "sd": sd,
        "se": se,
        "mde": mde,
        "alpha_effective": alpha_eff,
        "power": power,
        # Halving the MDE needs 4x the seeds — worth stating, because it is usually the honest
        # answer to "why not just add seeds until it is significant".
        "n_for_half_mde": 4 * n,
    }


def factorial_effects(
    cells: Mapping[tuple[bool, ...], Mapping[int, float]],
    factor_names: Sequence[str],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Main effects AND interactions of a full 2^k factorial, with paired per-seed bootstrap CIs.

    ``cells`` maps a factor-presence tuple (aligned with ``factor_names``) to that arm's
    ``{seed: metric}``. Effects use ±1 contrast coding, so the effect of a subset S of factors is

        E_S = (1 / 2^(k-1)) · Σ_cells [ Π_(f∈S) sign_f(cell) ] · y(cell)

    evaluated **within each seed** before aggregating. Pairing inside the seed is the point: every
    cell ran the same worlds, so per-seed contrasts cancel the world variance that would otherwise
    swamp a modest component effect.

    Interactions are computed rather than assumed away because scaffold components are known to
    substitute for and interfere with each other; a one-at-a-time sweep reports the main effects of
    a model in which those terms are constrained to zero, and is wrong exactly when they are not.
    Orientation follows the metric: for a loss, a NEGATIVE effect means the factor helped.
    """
    k = len(factor_names)
    if len(cells) != 2 ** k:
        raise ValueError(f"a full 2^{k} factorial needs {2 ** k} cells, got {len(cells)}")
    shared = sorted(set.intersection(*(set(v) for v in cells.values())))
    if not shared:
        raise ValueError("the factorial cells share no seeds — the paired design has nothing to pair")

    out: dict[str, dict[str, Any]] = {}
    for mask in range(1, 2 ** k):
        subset = [i for i in range(k) if mask >> i & 1]
        label = ":".join(factor_names[i] for i in subset)
        per_seed: list[float] = []
        for s in shared:
            total = 0.0
            for cell, by_seed in cells.items():
                coef = 1.0
                for i in subset:
                    coef *= 1.0 if cell[i] else -1.0
                total += coef * float(by_seed[s])
            per_seed.append(total / (2 ** (k - 1)))
        point, lo, hi = bootstrap_ci(per_seed, n_boot=n_boot, alpha=alpha, seed=seed)
        out[label] = {
            "order": len(subset),
            "effect": point,
            "ci_low": lo,
            "ci_high": hi,
            "p": bootstrap_p(per_seed, n_boot=n_boot, seed=seed),
            "n": len(shared),
            "per_seed": per_seed,
        }
    return out


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
