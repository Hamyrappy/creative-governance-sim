"""Policy churn: how often does the regent actually REVISE the law it has in force?

The loss factorial says episodic memory hurts. It does not say why. Churn is the mechanism variable:
the share of decisions at which the enacted policy differs from the one before it.

    churn = 0  the authority legislated once and never revisited
    churn = 1  it rewrote the law at every review

This is worth measuring separately from loss because the two answer different questions. Loss says a
component was costly; churn says what it did to the institution's behaviour, and across a structural
break "did it revise" is the behaviour the whole study is about.

A COLLAPSED DECISION COUNTS AS "NO CHANGE", not as a gap. That is the causally correct reading — a
decision that emits nothing leaves the previous law in force, which is precisely a non-revision — and
it also removes a bias that would otherwise manufacture the result. Skipping collapsed decisions
splices the sequence, so the two surviving neighbours are further apart in time and more likely to
differ; that inflates churn exactly in the arms that collapse most, which are the outcome arms. Both
conventions are computed and reported so the difference is visible rather than argued.

Run: ``uv run python scripts/policy_churn.py --store logs/runs_v3``
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from govsim.analysis.stats import factorial_effects, holm_bonferroni
from govsim.core.action import ActionSpace, VerbSpec
from govsim.core.llm.client import LLMResponse, response_is_collapsed
from govsim.core.result_store import ResultStore
from govsim.regents.llm_regent import parse_action_requests

FACTORS = ("trace", "outcome", "memory")
SPACE = ActionSpace(
    verbs=[VerbSpec(name=v) for v in ("set_lockdown", "set_vaccination", "set_policy_rate")],
    context_vars=[],
)


def arm_name(prefix: str, cell: tuple[bool, ...]) -> str:
    on = [f for f, b in zip(FACTORS, cell) if b]
    return f"{prefix}_llm_" + ("_".join(on) if on else "bare")


def _sequence(calls: list[dict], carry_forward: bool) -> list[tuple[str, ...]]:
    """The law in force at each decision.

    ``carry_forward`` keeps a collapsed/unparseable decision as a repeat of the standing law, which
    is what the world actually experienced. Without it the decision is dropped entirely.
    """
    seq: list[tuple[str, ...]] = []
    standing: tuple[str, ...] | None = None
    for c in calls:
        resp = LLMResponse(text=c.get("response_text") or "",
                           tool_calls=list(c.get("tool_calls") or []))
        reqs = parse_action_requests(resp, SPACE, "regent:0")
        if reqs:
            standing = tuple(sorted(f"{q.verb}={q.payload.get('expr')}" for q in reqs))
            seq.append(standing)
        elif carry_forward and standing is not None:
            seq.append(standing)
    return seq


def churn_by_seed(store: ResultStore, experiment: str, carry_forward: bool) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in store.query(experiment=experiment):
        path = row.get("llm_io_path")
        if not path or not Path(path).exists():
            continue
        calls = sorted(
            [c for c in json.loads(Path(path).read_text(encoding="utf-8"))
             if c.get("regent_id") != "critic"],
            key=lambda c: int(c.get("step", 0)),
        )
        seq = _sequence(calls, carry_forward)
        if len(seq) >= 2:
            out[int(row["seed"])] = sum(1 for a, b in zip(seq, seq[1:]) if a != b) / (len(seq) - 1)
    return out


def distinct_by_seed(store: ResultStore, experiment: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in store.query(experiment=experiment):
        path = row.get("llm_io_path")
        if not path or not Path(path).exists():
            continue
        calls = sorted(
            [c for c in json.loads(Path(path).read_text(encoding="utf-8"))
             if c.get("regent_id") != "critic"],
            key=lambda c: int(c.get("step", 0)),
        )
        seq = _sequence(calls, carry_forward=False)
        if seq:
            out[int(row["seed"])] = float(len(set(seq)))
    return out


def _report(title: str, cells: dict, factors: tuple[str, ...]) -> None:
    eff = factorial_effects(cells, factors)
    holm = holm_bonferroni({k: v["p"] for k, v in eff.items()})
    print(f"\n=== {title} ===")
    print(f"{'term':<24}{'effect':>10}{'95% CI':>26}{'p_holm':>9}  sig")
    for k, v in sorted(eff.items(), key=lambda kv: (kv[1]["order"], kv[1]["p"])):
        ci = f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
        mark = "YES" if holm[k]["significant"] else "·"
        print(f"{k:<24}{v['effect']:>+10.4f}{ci:>26}{holm[k]['p_adj']:>9.4f}  {mark}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="logs/runs_v3")
    ap.add_argument("--prefix", default="epidemic")
    ap.add_argument("--json", default="logs/churn.json")
    args = ap.parse_args()

    store = ResultStore(args.store)
    grid = list(itertools.product((False, True), repeat=3))

    print(f"{'arm':<38}{'churn':>8}{'churn_cf':>10}{'distinct':>10}")
    rows = {}
    for cell in grid:
        name = arm_name(args.prefix, cell)
        skip = churn_by_seed(store, name, carry_forward=False)
        carry = churn_by_seed(store, name, carry_forward=True)
        dis = distinct_by_seed(store, name)
        if not skip:
            continue
        rows[name] = {"churn_skip": st.fmean(skip.values()),
                      "churn_carry": st.fmean(carry.values()) if carry else None,
                      "distinct": st.fmean(dis.values()) if dis else None,
                      "n": len(skip)}
        print(f"{name:<38}{rows[name]['churn_skip']:>8.3f}"
              f"{rows[name]['churn_carry']:>10.3f}{rows[name]['distinct']:>10.2f}")

    for label, cf in (("churn, collapsed decisions DROPPED", False),
                      ("churn, collapsed decisions COUNTED AS NO-CHANGE", True)):
        cells = {c: churn_by_seed(store, arm_name(args.prefix, c), carry_forward=cf) for c in grid}
        cells = {k: v for k, v in cells.items() if v}
        if len(cells) == 8:
            _report(f"factorial on {label} (negative = SUPPRESSED revision)", cells, FACTORS)
        else:
            print(f"\n(factorial skipped for '{label}': {len(cells)}/8 cells)")

    Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
