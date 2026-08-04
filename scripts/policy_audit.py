"""
Audit the policies the regents actually enacted — the mechanism behind the loss numbers.

    uv run python scripts/policy_audit.py --store logs/runs_gemma --json logs/policy_audit.json

Losses say an arm did better. They do not say *why*, and in a governance setting the "why" is the
part a reader can argue with. Because a policy here is a short Python expression rather than a
number, we can read it. This script reports four things per arm:

**Revision rate.** How often the regent changed its standing rule, split pre- and post-break. The
mechanism this paper claims is that outcome feedback is the only channel carrying evidence of an
unobservable instrument failure. If that is right, arms with outcome feedback should revise *more*
after the break, and arms without it should not notice.

**Direction of revision.** Whether the enacted lockdown intensity went up or down across the break.
The oracle's answer is to back off — the instrument stopped working and never stopped costing — so
an arm that responds to worsening prevalence by clamping down harder is failing in the specific way
a frozen feedback rule fails, even though it is nominally adapting.

**Expressiveness class.** Each enacted policy is sorted into one of three classes relative to the
calibrated threshold family, which is the space the oracle was optimal within:

    constant   no observable appears — a fixed lockdown level. This is *less* expressive than the
               reference family, not more: a constant is the degenerate threshold whose trigger sits
               outside the observed range. Counting it as "novel" would be flattering nonsense.
    in-family  exactly `<a> if I > <thr> else 0.0`.
    outside    uses the observables in a way the family cannot express — proportional in I, a
               non-zero floor, multiple branches, or any dependence on S, R, or t.

This is the operational version of the "creative policy" construct this project has circled for a
while: not a judgement about novelty, but a decidable syntactic question. It matters because an arm
that beats the oracle from *outside* the family is telling us about our choice of family, while one
that beats it from inside is telling us about adaptation — and an arm sitting on *constants* is
telling us it never engaged with the state at all.

**Verbatim policies.** The most common enacted rules per arm, for the reader to inspect.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.core.result_store import ResultStore
from govsim.domains.scalar import regimes as R

REGENT = "regent:0"
#: The instrument that SURVIVES the break. The clairvoyant response is to substitute toward it.
SECOND_INSTRUMENT = "set_vaccination"

# The calibrated reference family is exactly `<a> if I > <thr> else 0.0` with numeric a, thr.
# Anything else is outside the space the oracle was optimal within.
_THRESHOLD_FORM = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*if\s+I\s*>\s*(-?\d+(?:\.\d+)?)\s*else\s*0(?:\.0+)?\s*$"
)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
#: Observable names the epidemic world publishes. A policy referencing none of them is a constant.
_OBSERVABLES = ("I", "S", "R", "lockdown", "vacc", "t")
_OBS_RE = re.compile(r"\b(" + "|".join(_OBSERVABLES) + r")\b")


def policy_class(expr: str) -> str:
    """``constant`` | ``in-family`` | ``outside`` — see the module docstring."""
    e = (expr or "").strip()
    if not _OBS_RE.search(e):
        return "constant"
    if _THRESHOLD_FORM.match(e):
        return "in-family"
    return "outside"


def in_reference_family(expr: str) -> bool:
    return policy_class(expr) in ("in-family", "constant")


def intensity_proxy(expr: str) -> float | None:
    """A crude scalar for 'how hard is this policy clamping down', for direction-of-revision only.

    Deliberately crude: the largest numeric literal in the expression. For the threshold family it
    recovers the intensity when the intensity exceeds the threshold, which holds for every
    calibrated reference; for a proportional law it recovers the gain. It is used ONLY to sign a
    change, never as a measurement, because a general expression has no well-defined intensity.
    """
    nums = [abs(float(m)) for m in _NUM.findall(expr or "")]
    return max(nums) if nums else None


def extract_calls(store: ResultStore, experiment: str) -> dict[int, list[dict]]:
    """``{seed: [{step, verb, expr}, …]}`` from the persisted LLM I/O artifacts."""
    out: dict[int, list[dict]] = {}
    for row in store.query(experiment=experiment):
        path = row.get("llm_io_path")
        if not path or not Path(path).exists():
            continue
        calls = json.loads(Path(path).read_text(encoding="utf-8"))
        events = []
        for c in calls:
            if c.get("regent") == "critic":
                continue  # audit calls are not enacted policy
            for tc in c.get("tool_calls") or []:
                try:
                    args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                except (json.JSONDecodeError, TypeError):
                    continue
                expr = args.get("expr")
                if isinstance(expr, str):
                    events.append({"step": c.get("step", 0), "verb": tc.get("name"), "expr": expr.strip()})
        if events:
            out[int(row["seed"])] = sorted(events, key=lambda e: e["step"])
    return out


def audit_arm(store: ResultStore, experiment: str) -> dict | None:
    by_seed = extract_calls(store, experiment)
    if not by_seed:
        return None
    shock = R.EPIDEMIC_SHOCK_STEP
    pre_rev, post_rev, direction, total = [], [], [], 0
    touched_second, touched_second_post = [], []
    classes: Counter[str] = Counter()
    verbs: Counter[str] = Counter()
    exprs: Counter[str] = Counter()

    for events in by_seed.values():
        # One standing rule per decision; a "revision" is the rule differing from the previous one.
        pre = [e for e in events if e["step"] < shock]
        post = [e for e in events if e["step"] >= shock]
        for window, acc in ((pre, pre_rev), (post, post_rev)):
            if len(window) < 2:
                continue
            changes = sum(1 for a, b in zip(window, window[1:]) if a["expr"] != b["expr"])
            acc.append(changes / (len(window) - 1))
        if pre and post:
            a, b = intensity_proxy(pre[-1]["expr"]), intensity_proxy(post[-1]["expr"])
            if a is not None and b is not None and a != b:
                direction.append(1 if b > a else -1)
        for e in events:
            exprs[e["expr"]] += 1
            classes[policy_class(e["expr"])] += 1
            verbs[e["verb"]] += 1
            total += 1
        # Did this run EVER legislate on the instrument that still works? The clairvoyant response
        # to the break is to substitute toward vaccination, so an arm that never touches that lever
        # has not merely adapted badly — it has not considered the correct move at all.
        touched_second.append(any(e["verb"] == SECOND_INSTRUMENT for e in events))
        post = [e for e in events if e["step"] >= shock]
        touched_second_post.append(any(e["verb"] == SECOND_INSTRUMENT for e in post))

    return {
        "arm": experiment,
        "n_seeds": len(by_seed),
        "revision_rate_pre": statistics.fmean(pre_rev) if pre_rev else None,
        "revision_rate_post": statistics.fmean(post_rev) if post_rev else None,
        "n_tightened_after_break": sum(1 for d in direction if d > 0),
        "n_relaxed_after_break": sum(1 for d in direction if d < 0),
        "share_constant": classes["constant"] / total if total else None,
        "share_in_family": classes["in-family"] / total if total else None,
        "share_outside": classes["outside"] / total if total else None,
        "share_used_second_instrument": (sum(touched_second) / len(touched_second)
                                         if touched_second else None),
        "share_used_second_after_break": (sum(touched_second_post) / len(touched_second_post)
                                          if touched_second_post else None),
        "verb_counts": dict(verbs),
        "n_policies": total,
        "top_policies": exprs.most_common(5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", nargs="+", default=["logs/runs"])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    stores = [ResultStore(p) for p in args.store]
    arms = args.arms or [
        "epidemic_llm_bare", "epidemic_llm_trace", "epidemic_llm_outcome", "epidemic_llm_memory",
        "epidemic_llm_trace_outcome", "epidemic_llm_trace_memory", "epidemic_llm_outcome_memory",
        "epidemic_llm_trace_outcome_memory", "epidemic_opro", "epidemic_llm_critic",
    ]

    rows = []
    for arm in arms:
        merged = None
        for s in stores:
            r = audit_arm(s, arm)
            if r and (merged is None or r["n_seeds"] > merged["n_seeds"]):
                merged = r
        if merged:
            rows.append(merged)

    if not rows:
        print("no enacted-policy records found in the given store(s)", file=sys.stderr)
        return 1

    pct = lambda v: f"{100 * v:.0f}%" if v is not None else "—"  # noqa: E731
    print(f"\n{'arm':<38} {'seeds':>5} {'rev.pre':>8} {'rev.post':>9} "
          f"{'tighten/relax':>14} {'const':>7} {'in-fam':>7} {'outside':>8} {'used 2nd lever':>15}")
    for r in rows:
        rp = f"{r['revision_rate_pre']:.2f}" if r["revision_rate_pre"] is not None else "—"
        rq = f"{r['revision_rate_post']:.2f}" if r["revision_rate_post"] is not None else "—"
        print(f"{r['arm']:<38} {r['n_seeds']:>5} {rp:>8} {rq:>9} "
              f"{r['n_tightened_after_break']:>6}/{r['n_relaxed_after_break']:<7} "
              f"{pct(r['share_constant']):>7} {pct(r['share_in_family']):>7} "
              f"{pct(r['share_outside']):>8} {pct(r['share_used_second_after_break']):>15}")

    print("\n=== most frequently enacted policies ===")
    marks = {"constant": "c", "in-family": " ", "outside": "*"}
    for r in rows:
        print(f"\n{r['arm']}  (n={r['n_policies']} enacted rules)")
        for expr, count in r["top_policies"]:
            print(f"  {count:>4}x {marks[policy_class(expr)]} {expr}")
    print("\n(* outside the calibrated threshold family; c constant, i.e. no observable referenced —")
    print(" a constant is a DEGENERATE member of that family, not an escape from it)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
