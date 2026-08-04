"""
Turn a ResultStore into the paper's tables: normalized regret, factorial attribution, contrasts.

    uv run python scripts/analyze_matrix.py --store logs/runs --model gemini-3.5-flash-lite

Everything here is paired on the world seed. Each arm ran the identical seeds, so per-seed
differencing removes the world variance — which in this regime is roughly as large as the effects
being measured, so an unpaired comparison would be mostly noise about the epidemic rather than
about the regent.

Three outputs, in the order a reader needs them:

1. **Arm table** — post-shock loss and normalized regret R against the calibrated anchors, where
   R=0 is the clairvoyant oracle and R=1 is the frozen pre-shock rule.
2. **Factorial attribution** — main effects *and* interactions of the harness components, Holm
   corrected across the whole family, because eight cells and seven effect terms is a family and
   reporting each at α=0.05 buys false positives at that many looks.
3. **Named contrasts** — each arm against frozen (did adaptation happen?) and against
   budget-matched OPRO (is the harness doing anything a plain score-optimizer would not?).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.analysis import compare, normalized_regret
from govsim.analysis.stats import (
    bootstrap_p, factorial_effects, holm_bonferroni, minimum_detectable_effect,
)
from govsim.core.result_store import ResultStore

FACTORS = ("trace", "outcome", "memory")
METRIC = "loss"  # FULL horizon: a post-break-only metric rewards passivity (preregistration §8)
REGENT = "regent:0"


def arm_name(cell: tuple[bool, ...]) -> str:
    on = [f for f, b in zip(FACTORS, cell) if b]
    return "epidemic_llm_" + ("_".join(on) if on else "bare")


def load_by_seed(stores: list[ResultStore], experiment: str, model: str | None) -> dict[int, float]:
    """``{seed: metric}`` for an arm, keeping the LATEST run per seed (re-runs supersede).

    Takes several stores because models run as separate processes against separate sqlite files —
    one shared file would have them contending for the write lock for hours.
    """
    out: dict[int, float] = {}
    for store in stores:
        for row in store.query(experiment=experiment):
            if model:
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                row_model = spec.get("model")
                if row_model is not None and row_model != model:
                    continue
            comps = (row.get("components") or {}).get(REGENT, {})
            if METRIC in comps:
                out[int(row["seed"])] = float(comps[METRIC])  # ORDER BY run_id ⇒ last write wins
    return out


def available_models(stores: list[ResultStore]) -> list[str]:
    """Every LLM model id that appears in any store (for the cross-model replication table)."""
    seen: set[str] = set()
    for store in stores:
        for row in store.query():
            m = ((row.get("regent_specs") or {}).get(REGENT, {}) or {}).get("model")
            if m:
                seen.add(str(m))
    return sorted(seen)


def no_action_rate(stores: list[ResultStore], experiment: str, model: str | None) -> tuple[int, int]:
    """``(calls that produced no parseable action, total calls)`` for an arm.

    This is a validity check, not a curiosity, and it is run before any result is believed. A model
    that reasons before acting can spend its whole output budget on the reasoning and be truncated
    before it emits the tool call. When that happens the decision is a silent no-op: the previously
    installed law simply stays in force, and the arm quietly becomes "sticky policy" rather than the
    treatment it is labelled as.

    The failure is *correlated with the treatment*, which is what makes it lethal here. A harness
    channel lengthens the prompt and invites longer deliberation, so exactly the arms carrying more
    information are the ones most likely to run out of budget. An ablation run this way measures
    truncation and reports it as information.

    We hit precisely this: at ``max_tokens=1500`` the outcome-feedback arm failed to act on 32.5% of
    its decisions while the no-harness arm failed on 0%.
    """
    empty = total = 0
    for store in stores:
        for row in store.query(experiment=experiment):
            if model:
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                if spec.get("model") not in (None, model):
                    continue
            path = row.get("llm_io_path")
            if not path or not Path(path).exists():
                continue
            for call in json.loads(Path(path).read_text(encoding="utf-8")):
                if call.get("regent") == "critic":
                    continue
                total += 1
                if not (call.get("tool_calls") or []):
                    empty += 1
    return empty, total


def fmt(v: float | None, w: int = 9, p: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return f"{'—':>{w}}"
    return f"{v:>{w}.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", nargs="+", default=["logs/runs"],
                    help="one or more ResultStore roots (models run into separate stores)")
    ap.add_argument("--model", default=None, help="restrict LLM arms to this model id")
    ap.add_argument("--cross-model", action="store_true",
                    help="also print the per-model replication table")
    ap.add_argument("--json", default=None, help="write the full analysis here")
    args = ap.parse_args()

    stores = [ResultStore(p) for p in args.store]
    store = stores  # every loader takes the list
    # R is anchored on the NON-ADAPTIVE ceiling and the clairvoyant ADAPTOR, so R<1 means
    # "did better than any fixed law could have, in hindsight" rather than the much weaker
    # "did better than the stale rule", which passivity alone can achieve.
    frozen = load_by_seed(store, "epidemic_best_fixed", None)
    oracle = load_by_seed(store, "epidemic_switching", None)
    stale = load_by_seed(store, "epidemic_frozen", None)
    if not frozen or not oracle:
        print("missing calibrated anchors (epidemic_best_fixed / epidemic_switching) in the "
              "store; run `scripts/run_matrix.py --arms epidemic-refs` first", file=sys.stderr)
        return 1

    cells = {}
    for bits in range(2 ** len(FACTORS)):
        cell = tuple(bool(bits >> i & 1) for i in range(len(FACTORS)))
        by_seed = load_by_seed(store, arm_name(cell), args.model)
        if by_seed:
            cells[cell] = by_seed

    extra_arms = {name: load_by_seed(store, name, args.model)
                  for name in ("epidemic_opro", "epidemic_llm_critic")}
    extra_arms = {k: v for k, v in extra_arms.items() if v}

    # ---- 1. arm table ------------------------------------------------------------------------
    fm, om = statistics.fmean(frozen.values()), statistics.fmean(oracle.values())
    print(f"\n=== arms (metric={METRIC}, model={args.model or 'any'}) ===")
    print(f"{'arm':<38} {'n':>3} {'mean':>9} {'sd':>8} {'R':>7}   R: 0=oracle, 1=frozen")
    print(f"{'best_fixed (R=1: non-adaptive ceiling)':<38} {len(frozen):>3} {fmt(fm)} "
          f"{fmt(statistics.pstdev(frozen.values()) if len(frozen) > 1 else 0.0, 8)} {1.0:>7.3f}")
    print(f"{'switching (R=0: clairvoyant adaptor)':<38} {len(oracle):>3} {fmt(om)} "
          f"{fmt(statistics.pstdev(oracle.values()) if len(oracle) > 1 else 0.0, 8)} {0.0:>7.3f}")

    table = []
    all_arms = {arm_name(c): v for c, v in cells.items()} | extra_arms
    for name, by_seed in all_arms.items():
        shared = sorted(set(by_seed) & set(frozen) & set(oracle))
        if not shared:
            continue
        # R is computed on the SHARED seeds only, so an arm that happened to run a friendlier
        # subset of worlds cannot look better for that reason.
        #
        # The HEADLINE R is a ratio of means, not a mean of per-seed ratios. Per-seed ratios divide
        # by that seed's own (fixed - switching) gap, and in this regime that gap ranges over an
        # order of magnitude across seeds (0.43 to 7.16). A single seed with a small denominator
        # then dominates the average and moves R far more than it moves any loss. The per-seed
        # distribution is still reported — its median and spread say something the ratio of means
        # does not — but it is not what the tables lead with.
        rs = [normalized_regret(by_seed[s], frozen[s], oracle[s]) for s in shared]
        rs = [r for r in rs if math.isfinite(r)]
        mean = statistics.fmean(by_seed[s] for s in shared)
        sd = statistics.pstdev([by_seed[s] for s in shared]) if len(shared) > 1 else 0.0
        fm_shared = statistics.fmean(frozen[s] for s in shared)
        om_shared = statistics.fmean(oracle[s] for s in shared)
        row = {"arm": name, "n": len(shared), "mean": mean, "sd": sd,
               "R": normalized_regret(mean, fm_shared, om_shared),
               "R_perseed_mean": statistics.fmean(rs) if rs else None,
               "R_median": statistics.median(rs) if rs else None}
        table.append(row)
        print(f"{name:<38} {row['n']:>3} {fmt(mean)} {fmt(sd, 8)} "
              f"{(f'{row['R']:>7.3f}' if row['R'] is not None else '      —')}")

    # ---- 1b. VALIDITY GATE: did every arm actually act? ---------------------------------------
    # Run before the factorial, because a factorial over truncated arms is a table of nonsense.
    print("\n=== validity: decisions that produced NO parseable action ===")
    worst = 0.0
    action_rates = {}
    for name in list(all_arms):
        empty, tot = no_action_rate(store, name, args.model)
        if not tot:
            continue
        rate = empty / tot
        action_rates[name] = {"empty": empty, "total": tot, "rate": rate}
        worst = max(worst, rate)
        flag = "  <-- CONTAMINATED" if rate > 0.02 else ""
        print(f"  {name:<40} {empty:>4}/{tot:<5} {100 * rate:>5.1f}%{flag}")
    if worst > 0.02:
        print("\n  [!!] At least one arm silently failed to act on >2% of its decisions. A decision")
        print("       that emits nothing leaves the PREVIOUS law in force, so that arm is not the")
        print("       treatment it is labelled as. This failure correlates with the treatment —")
        print("       harness channels lengthen the prompt and invite longer reasoning — so the")
        print("       ablation would be measuring truncation. Raise GOVSIM_LLM_MAX_TOKENS and re-run")
        print("       before believing anything below.")

    # ---- 2. factorial attribution -------------------------------------------------------------
    factorial = None
    if len(cells) == 2 ** len(FACTORS):
        eff = factorial_effects(cells, FACTORS)
        holm = holm_bonferroni({k: v["p"] for k, v in eff.items()})
        for k in eff:
            eff[k].update(holm[k])
            eff[k].pop("per_seed", None)
        factorial = eff
        print(f"\n=== factorial attribution on {METRIC} (negative effect = the factor HELPED) ===")
        print(f"{'term':<26} {'ord':>3} {'effect':>9} {'95% CI':>22} {'p':>7} {'p_holm':>8}  sig")
        for k, v in sorted(eff.items(), key=lambda kv: (kv[1]["order"], kv[1]["p"])):
            ci = f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
            print(f"{k:<26} {v['order']:>3} {v['effect']:>+9.4f} {ci:>22} "
                  f"{v['p']:>7.4f} {v['p_adj']:>8.4f}  {'YES' if v['significant'] else '·'}")
    else:
        print(f"\n(factorial skipped: {len(cells)}/{2 ** len(FACTORS)} cells present in the store)")

    # ---- 2b. what this design could have detected ---------------------------------------------
    # A null is only informative next to the smallest effect that would have shown up. Reported at
    # the CORRECTED alpha, because that is the bar the headline analysis actually applies.
    mde = None
    if len(cells) >= 2:
        base = cells.get(tuple(False for _ in FACTORS))
        if base:
            ref = {s: frozen[s] for s in base if s in frozen}
            diffs = [base[s] - ref[s] for s in sorted(set(base) & set(ref))]
            n_terms = len(factorial) if factorial else 7
            mde = minimum_detectable_effect(diffs, n_comparisons=n_terms)
            budget = fm - om
            print(f"\n=== power: what this design could have detected ({METRIC}) ===")
            print(f"  per-seed sd of (arm - best_fixed) = {mde['sd']:.4f}   se = {mde['se']:.4f}   n = {mde['n']}")
            print(f"  minimum detectable effect at alpha={mde['alpha_effective']:.4f} "
                  f"(Bonferroni over {n_terms} terms), power {mde['power']:.0%}: "
                  f"{mde['mde']:.4f} {METRIC} units")
            print(f"  total adaptation budget (best_fixed - switching) = {budget:.4f}")
            if budget > 0:
                print(f"  => the design can only resolve a component worth "
                      f">= {100 * mde['mde'] / budget:.0f}% of the whole adaptation budget.")
                print(f"  => a null here rules out LARGE component effects, not small ones. "
                      f"Halving the MDE needs n={mde['n_for_half_mde']} seeds.")

    # ---- 3. named contrasts -------------------------------------------------------------------
    print(f"\n=== contrasts (paired bootstrap on {METRIC}; lower is better) ===")
    contrasts = {}
    refs = {"vs_best_fixed": frozen, "vs_switching": oracle}
    if stale:
        refs["vs_stale_rule"] = stale
    if "epidemic_opro" in all_arms:
        refs["vs_opro"] = all_arms["epidemic_opro"]
    for name, by_seed in all_arms.items():
        for ref_label, ref in refs.items():
            if by_seed is ref:
                continue
            res = compare(by_seed, ref, lower_is_better=True)
            if res["n"] < 2:
                continue
            res["p"] = bootstrap_p(res["diffs"])
            contrasts[f"{name} {ref_label}"] = {
                k: res[k] for k in ("n", "point_estimate", "ci_low", "ci_high",
                                    "a_better_than_b", "excludes_zero", "p")}
    holm_c = holm_bonferroni({k: v["p"] for k, v in contrasts.items()}) if contrasts else {}
    for k, v in contrasts.items():
        v.update(holm_c.get(k, {}))
    for k, v in sorted(contrasts.items(), key=lambda kv: kv[1]["point_estimate"]):
        verdict = "BETTER" if v["a_better_than_b"] else ("worse" if v["excludes_zero"] else "ns")
        print(f"{k:<52} n={v['n']:>3} Δ={v['point_estimate']:>+9.4f} "
              f"[{v['ci_low']:+.4f},{v['ci_high']:+.4f}] p_holm={v.get('p_adj', float('nan')):.4f}  {verdict}")

    # ---- 4. cross-model replication -----------------------------------------------------------
    cross = None
    if args.cross_model:
        cross = {}
        models = available_models(store)
        rungs = ["epidemic_llm_bare", "epidemic_llm_outcome", "epidemic_llm_trace_outcome_memory"]
        print(f"\n=== cross-model replication (mean normalized regret R; lower = closer to oracle) ===")
        print(f"{'model':<26} " + " ".join(f"{r.replace('epidemic_llm_', ''):>22}" for r in rungs))
        for m in models:
            row = {}
            cellstrs = []
            for arm in rungs:
                by_seed = load_by_seed(store, arm, m)
                shared = sorted(set(by_seed) & set(frozen) & set(oracle))
                if not shared:
                    cellstrs.append(f"{'—':>22}")
                    row[arm] = None
                    continue
                rs = [normalized_regret(by_seed[s], frozen[s], oracle[s]) for s in shared]
                rs = [r for r in rs if math.isfinite(r)]
                val = statistics.fmean(rs) if rs else None
                row[arm] = {"R": val, "n": len(shared)}
                cellstrs.append(f"{val:>16.3f} (n={len(shared):>2})" if val is not None else f"{'—':>22}")
            cross[m] = row
            print(f"{m:<26} " + " ".join(cellstrs))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "model": args.model, "metric": METRIC,
            "anchors": {"frozen_mean": fm, "oracle_mean": om,
                        "headroom": (fm / om) if om else None},
            "arms": table, "factorial": factorial, "contrasts": contrasts, "power": mde,
            "action_rates": action_rates,
            "cross_model": cross,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
